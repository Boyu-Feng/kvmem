import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest
import torch

from scripts import latency_datasets as data
from scripts import latency_runtime as runtime
from scripts import run_latency_experiment as original
from scripts import run_latency_experiment_queue as original_queue
from scripts import run_multidataset_latency as runner
from scripts import run_multidataset_latency_queue as queue
from scripts.run_qa32_rebuttal import atomic_json, file_hash


@pytest.mark.parametrize("dataset", data.ADDITIONAL_DATASETS)
def test_default_plan_never_loads_assets_or_queries_cuda(dataset, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(original, "execute", lambda *args: pytest.fail("Must not execute"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("Must not query CUDA"))
    output = tmp_path / "output"
    assert runner.main(["--dataset", dataset, "--method", "stepkv", "--model-ready",
                        str(tmp_path / "missing.json"), "--data-ready", str(tmp_path / "missing-data.json"),
                        "--output", str(output)]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["dataset"] == dataset and plan["sample_positions"] == list(range(20))
    assert plan["warmup_positions"] == [500, 501] and plan["clean_repeats_per_question"] == 2
    assert not output.exists()


@pytest.fixture(params=data.ADDITIONAL_DATASETS)
def normalized(tmp_path, monkeypatch, request):
    dataset = request.param
    directory = tmp_path / "assets/datasets" / dataset
    directory.mkdir(parents=True)
    rows = [{"id": f"{dataset}-{i}", "question": f"Question {i}?", "answer": f"Answer {i}"}
            for i in range(600)]
    path = directory / "normalized_dev.json"
    atomic_json(path, rows)
    source = tmp_path / "normalizer.py"
    source.write_text("def normalize():\n    return []\n")
    metadata = {
        "dataset": dataset, "repository": "test/repo", "revision": "pinned", "split": "dev",
        "rows": 600, "sha256": file_hash(path), "raw_sha256": "raw",
        "normalizer": "normalizer.normalize", "normalizer_sha256": file_hash(source), "path": str(path),
    }
    spec = tmp_path / "spec.json"
    atomic_json(spec, {"additional_datasets": {dataset: metadata}})
    atomic_json(directory / "ready.json", metadata)
    monkeypatch.setattr(data, "ASSET_SPEC", spec)
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    monkeypatch.setitem(data.DATASETS, dataset, 600)
    return SimpleNamespace(dataset=dataset, path=path, rows=rows, metadata=metadata,
                           ready=tmp_path / "assets/data_ready.json", directory=directory)


def base_selector():
    base = SimpleNamespace(RANDOM_SEED=0, NUM_SAMPLES=0)

    def select(rows):
        indices = list(range(len(rows)))
        random.Random(base.RANDOM_SEED).shuffle(indices)
        return [(i, rows[i]) for i in indices[:base.NUM_SAMPLES]]

    base.select_samples = select
    return base


def test_normalized_pin_and_identical_main_warmup_selection(normalized):
    item = normalized
    path, rows, provenance = data.normalized_assets(item.ready, item.dataset)
    assert path == item.path and rows == item.rows
    assert provenance["normalized_payload_sha256"] == item.metadata["sha256"]
    first, warmups = data.select_samples(base_selector(), rows, 20, item.dataset)
    repeated, repeated_warmups = data.select_samples(base_selector(), rows, 20, item.dataset)
    assert first == repeated and warmups == repeated_warmups
    assert len(first) == 20 and len(warmups) == 2
    assert not {row["id"] for _, row in first} & {row["id"] for _, row in warmups}


def test_payload_corruption_is_explicitly_rejected(normalized):
    normalized.path.write_text("[]")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        data.normalized_assets(normalized.ready, normalized.dataset)


def test_dataset_scope_reuses_worker_and_restores_all_globals(normalized, monkeypatch):
    import datasets

    item = normalized
    assets = lambda *args: ("model7", "fullwiki", ["hotpot.parquet"], {"parquet_sha256": {"old": "hash"}})
    monkeypatch.setattr(original, "load_assets", assets)
    previous = original.load_assets, original.select_samples, original.experiment_hashes, datasets.load_dataset
    with data.dataset_scope(item.dataset):
        model, index, files, provenance = original.load_assets("model.json", item.ready)
        assert (model, index) == ("model7", "fullwiki") and files == [item.path]
        assert provenance["evaluation_dataset"]["dataset"] == item.dataset
        assert provenance["parquet_sha256"] == {}
        rows = datasets.load_dataset("parquet", data_files={"validation": [str(item.path)]},
                                     split="validation", cache_dir="unused")
        selected, warmups = original.select_samples(base_selector(), rows, 20)
        assert len(selected) == 20 and len(warmups) == 2
        with pytest.raises(RuntimeError, match="Unexpected normalized"):
            datasets.load_dataset("parquet", data_files={"validation": ["wrong.json"]}, split="validation")
    assert previous == (original.load_assets, original.select_samples,
                        original.experiment_hashes, datasets.load_dataset)
    with pytest.raises(RuntimeError, match="test failure"):
        with data.dataset_scope(item.dataset):
            raise RuntimeError("test failure")
    assert previous == (original.load_assets, original.select_samples,
                        original.experiment_hashes, datasets.load_dataset)


def test_extended_bindings_preserve_every_live_hotpot_binding():
    current = runtime.experiment_hashes()
    extended = data.experiment_hashes()
    assert all(extended[name] == value for name, value in current.items())
    assert len(extended) == len(current) + len(data.SOURCE_FILES)


def test_30_jobs_do_not_recursively_call_the_patched_queue(monkeypatch):
    expected = queue.jobs()
    assert len(expected) == 30 and len({name for _, _, name in expected}) == 30
    assert [name.split("/")[0] for _, _, name in expected].count("2wiki") == 15
    assert [name.split("/")[0] for _, _, name in expected].count("musique") == 15
    monkeypatch.setattr(original_queue, "jobs", queue.jobs)
    assert queue.jobs() == expected


def test_bootstrap_dispatch_changes_only_dataset_entrypoint(tmp_path):
    worker = str(queue.ROOT / "scripts/run_latency_experiment.py")
    command = ["/python", "bootstrap.py", "--", "/python", worker, "--execute",
               "--method", "stepkv", "--output", str(tmp_path / "musique/stepkv/r20")]
    actual = queue.worker_command(command, tmp_path)
    assert actual[4] == str(queue.ROOT / "scripts/run_multidataset_latency.py")
    assert actual[-2:] == ["--dataset", "musique"]
    assert actual[:4] == command[:4] and command[4] == worker
    nvml = ["nvidia-smi", "--query-gpu=uuid"]
    assert queue.worker_command(nvml, tmp_path) == nvml
    assert queue.worker_command("unrelated-command", tmp_path) == "unrelated-command"
    with pytest.raises(RuntimeError, match="Unexpected dataset"):
        queue.worker_command([worker, "--output", str(tmp_path / "hotpotqa/fullkv/r100")], tmp_path)


def test_reservation_holder_is_cpu_only_and_exits_with_owner(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", queue.GPU7_UUID)
    monkeypatch.setattr(queue, "identity_alive", lambda *args: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("Holder must not query CUDA"))
    status = tmp_path / "hold.json"
    queue.hold_main(["--owner-pid", "1", "--owner-start-ticks", "2", "--status", str(status)])
    record = json.loads(status.read_text())
    assert record["gpu"] == 7 and "no CUDA" in record["purpose"]


def test_reports_have_separate_fullkv_references_for_each_dataset(tmp_path):
    runs, calls = [], []
    for _, _, name in queue.jobs():
        path = tmp_path / name
        atomic_json(path / "manifest.json", {"protocol": {"dataset": name.split("/")[0]}})
        runs.append(path)

    def report(directories, output):
        calls.append((directories, output))
        assert len(directories) == 15
        assert len({str(path.relative_to(tmp_path)).split("/")[0] for path in directories}) == 1
        return {"dataset": output.name, "conditions": 15}

    summary = queue.summarize_datasets(runs, tmp_path / "report", report)
    assert len(calls) == 2 and set(summary) == {"2wiki", "musique"}
    assert json.loads((tmp_path / "report/DATASET_SUMMARIES.json").read_text()) == summary
    with pytest.raises(RuntimeError, match="Incomplete condition"):
        queue.summarize_datasets(runs[:-1], tmp_path / "bad-report", report)


def test_execute_entry_delegates_to_the_original_measurement_pipeline(normalized, monkeypatch):
    item, observed = normalized, []
    monkeypatch.setattr(original, "load_assets", lambda *args: ("model7", "wiki", [], {}))

    def execute(args, plan):
        import datasets
        _, _, files, assets = original.load_assets(args.model_ready, args.data_ready)
        rows = datasets.load_dataset("parquet", data_files={"validation": list(map(str, files))},
                                     split="validation")
        selected, warmups = original.select_samples(base_selector(), rows, 20)
        observed.append((plan["dataset"], assets["evaluation_dataset"]["dataset"],
                         len(selected), len(warmups)))

    monkeypatch.setattr(original, "execute", execute)
    assert runner.main(["--execute", "--dataset", item.dataset, "--method", "stepkv",
                        "--model-ready", "unused.json", "--data-ready", str(item.ready),
                        "--output", str(item.directory / "results")]) == 0
    assert observed == [(item.dataset, item.dataset, 20, 2)]


def test_complete_continuation_reuses_30_queue_jobs_and_releases_holder_before_report(tmp_path, monkeypatch):
    from scripts import summarize_latency_experiment as reporter
    from scripts.run_idle_gpu_load_guard import GPU7_UUID

    output, predecessor, node = tmp_path / "additional", tmp_path / "hotpot", tmp_path / "node"
    guard = {"pid": 123, "start_ticks": 123}
    atomic_json(predecessor / "queue_status.json", {
        "owner": {"pid": 999, "start_ticks": 999}, "worker": None, "status": "complete",
        "completed_conditions": 15,
    })
    atomic_json(predecessor / "queue_plan.json", {"source_hashes": {"core": "same"}})
    atomic_json(node / "idle-load-guard.status.json", {"owner": guard, "status": "running"})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(runtime, "experiment_hashes", lambda: {"core": "same"})
    monkeypatch.setattr(data, "experiment_hashes", lambda: {"core": "same", "overlay": "new"})
    monkeypatch.setattr(data, "normalized_assets", lambda *args: ("unused", [], {}))
    monkeypatch.setattr(data, "select_samples", lambda *args: (
        [(i, {"id": str(i)}) for i in range(20)], [(500, {"id": "500"}), (501, {"id": "501"})]))
    monkeypatch.setattr(original_queue, "load_assets", lambda *args: None)
    monkeypatch.setattr(original_queue, "inspect_process", lambda pid: {
        "identity": guard, "argv": ["/python", "-u", str(queue.ROOT / "scripts/run_idle_gpu_load_guard.py")]})
    monkeypatch.setattr(original_queue, "gpu_samples", lambda: {
        7: {"uuid": GPU7_UUID, "pids": [], "utilization": 0}})
    monkeypatch.setattr(original_queue, "claim_maintenance", lambda *args: None)
    monkeypatch.setattr(original_queue, "release_maintenance", lambda *args: True)
    monkeypatch.setattr(original_queue, "validate_run", lambda *args: None)
    for module in (queue, original_queue):
        monkeypatch.setattr(module, "identity_alive", lambda identity: identity == guard)
        monkeypatch.setattr(module, "process_identity", lambda pid: {"pid": pid, "start_ticks": pid})
    processes, commands, reports = [], [], []

    class Process:
        def __init__(self, command, **kwargs):
            self.pid = 200 + len(processes)
            self.returncode = None if "--hold-reservation" in command else 0
            processes.append(self)
            if self.returncode == 0:
                assert str(queue.ROOT / "scripts/run_multidataset_latency.py") in command
                dataset = command[command.index("--dataset") + 1]
                target = Path(command[command.index("--output") + 1])
                atomic_json(target / "manifest.json", {"protocol": {"dataset": dataset}})
                commands.append(command)

        def poll(self):
            return self.returncode

        def terminate(self):
            self.returncode = 0

        def wait(self, timeout):
            return self.returncode

    def report(runs, destination):
        assert processes[0].returncode == 0
        reports.append((runs, destination))
        return {"conditions": len(runs)}

    monkeypatch.setattr(queue.subprocess, "Popen", Process)
    monkeypatch.setattr(reporter, "summarize_runs", report)
    queue.main(["--after-output", str(predecessor), "--output", str(output),
                "--model-ready", "unused.json", "--data-ready", "unused-data.json",
                "--load-script", "unused-load.py", "--load-node", str(node)])
    assert len(commands) == 30 and len(reports) == 2
    assert [command[command.index("--dataset") + 1] for command in commands].count("2wiki") == 15
    assert original_queue.jobs() == data.legacy_jobs()
    assert json.loads((output / "continuation_status.json").read_text())["status"] == "complete"
    assert json.loads((output / "queue_status.json").read_text())["completed_conditions"] == 30
    assert not (node / "latency-reservation-gpu7.json").exists()
