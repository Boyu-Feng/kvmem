import os
from pathlib import Path

import pytest

from scripts import run_component_ablation_queue as queue


def test_two_models_have_only_requested_seed_and_variant_jobs():
    jobs = queue.jobs(["qwen25_7b", "llama31_8b"])
    assert len(jobs) == 168
    assert sum(job["purpose"] == "main" for job in jobs) == 84
    assert {job["seed"] for job in jobs} == {233, 3407}
    assert {job["dataset"] for job in jobs} == set(queue.DATASETS)
    assert {job["variant"] for job in jobs} == set(queue.VARIANTS)
    assert all(job["variant"] != "fullkv" and job["ratio"] == .5 for job in jobs)
    assert len({job["name"] for job in jobs}) == len(jobs)


def test_group_smokes_gate_all_main_conditions():
    jobs = queue.jobs(["qwen25_7b"])
    ready = queue.next_jobs(jobs, set(), set(), {"qwen25_7b"})
    assert len(ready) == 6
    assert all(job["variant"] == "full" and job["purpose"] == "smoke" for job in ready)
    completed = {job["name"] for job in jobs if job["group"] == ready[0]["group"]
                 and job["purpose"] == "smoke"}
    ready = queue.next_jobs(jobs, completed, set(), {"qwen25_7b"})
    assert ready[0]["variant"] == "full" and ready[0]["purpose"] == "main"


def test_active_group_is_not_launched_twice():
    jobs = queue.jobs(["qwen25_7b"])
    group = jobs[0]["group"]
    ready = queue.next_jobs(jobs, set(), {group}, {"qwen25_7b"})
    assert all(job["group"] != group for job in ready)


def test_missing_llama_does_not_block_qwen():
    jobs = queue.jobs(["qwen25_7b", "llama31_8b"])
    ready = queue.next_jobs(jobs, set(), set(), {"qwen25_7b"})
    assert len(ready) == 6
    assert {job["model"] for job in ready} == {"qwen25_7b"}


def test_shared_load_is_allowed_but_other_compute_is_not(monkeypatch):
    monkeypatch.setattr(queue, "own_load", lambda pid, gpu: pid == 10 and gpu == 0)
    device = {"index": 0, "pids": [10], "free_mib": 48 * 1024}
    assert queue.available(device, 48)
    assert not queue.available({**device, "pids": [10, 11]}, 48)
    assert not queue.available({**device, "free_mib": 48 * 1024 - 1}, 48)


def test_own_load_checks_exact_command_and_gpu(tmp_path):
    directory = tmp_path / "10"
    directory.mkdir()
    args = ["/usr/bin/python3", "-B", str(queue.LOAD_SCRIPT), "worker", "--gpu", "0"]
    directory.joinpath("cmdline").write_bytes(b"\0".join(os.fsencode(arg) for arg in args) + b"\0")
    assert queue.own_load(10, 0, tmp_path)
    assert not queue.own_load(10, 1, tmp_path)
    directory.joinpath("cmdline").write_bytes(b"/usr/bin/python3\0unrelated.py\0")
    assert not queue.own_load(10, 0, tmp_path)
    assert not queue.own_load(11, 0, tmp_path)


def test_worker_command_never_contains_fullkv():
    from types import SimpleNamespace

    args = SimpleNamespace(model_readys={"qwen25_7b": Path("/model/ready.json")},
                           data_ready=Path("/data/ready.json"), output=Path("/results"))
    cmd = queue.command(args, queue.jobs(["qwen25_7b"])[0])
    assert "--execute" in cmd and "--variant" in cmd
    assert cmd[cmd.index("--variant") + 1] == "full"
    assert not any("fullkv" in argument for argument in cmd)


def test_validation_rejects_wrong_seed(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    job = queue.jobs(["qwen25_7b"])[0]
    module = SimpleNamespace(validate_completed_run=lambda directory: {
        "completed_samples": 2, "has_actual_eviction": True,
    })
    monkeypatch.setitem(sys.modules, "scripts.run_component_ablation_rebuttal", module)
    monkeypatch.setattr(queue, "read_json", lambda path: {"fingerprint": {"protocol": {**job, "seed": 42}}})
    with pytest.raises(RuntimeError, match="differs in seed"):
        queue.validate_job(tmp_path, job)


def test_cpu_only_experiment_blocks_gpu_before_cuda_context(tmp_path):
    directory = tmp_path / "10"
    directory.mkdir()
    directory.joinpath("cmdline").write_bytes(b"/usr/bin/python3\0experiment.py\0")
    directory.joinpath("environ").write_bytes(b"CUDA_VISIBLE_DEVICES=GPU-test\0")
    assert queue.assigned_experiment(0, "GPU-test", tmp_path)
    assert not queue.assigned_experiment(1, "GPU-other", tmp_path)


def test_complete_controller_dispatches_all_jobs_without_gpu_or_load_signals(monkeypatch, tmp_path):
    from types import SimpleNamespace

    ready = tmp_path / "ready.json"
    ready.write_text('{"repo":"Qwen/Qwen2.5-7B-Instruct"}')
    args = SimpleNamespace(
        output=tmp_path / "results", tmp_root=tmp_path / "tmp",
        model_readys={"qwen25_7b": ready}, data_ready=ready,
        gpus=[0, 1, 2], max_workers=3, min_free_gib=48, poll_seconds=.01,
    )
    calls = []
    class Child:
        pid = 99999999

        def __init__(self, cmd, **kwargs):
            calls.append((cmd, kwargs))

        def poll(self):
            return 0

        def wait(self, **kwargs):
            return 0

        def terminate(self):
            pytest.fail("Completed simulation should not terminate anything")

    monkeypatch.setattr(queue, "source_gate", lambda: {"source": "unchanged"})
    monkeypatch.setattr(queue, "gpu_snapshot", lambda: {
        gpu: {"index": gpu, "uuid": f"GPU-{gpu}", "free_mib": 60 * 1024, "pids": []}
        for gpu in args.gpus
    })
    monkeypatch.setattr(queue, "assigned_experiment", lambda *args: False)
    monkeypatch.setattr(queue, "maintenance_blockers", lambda path: {})
    monkeypatch.setattr(queue, "maintenance_launch_lock", lambda gpu: __import__("contextlib").nullcontext())
    monkeypatch.setattr(queue.subprocess, "Popen", Child)
    monkeypatch.setattr(queue.time, "sleep", lambda delay: None)
    monkeypatch.setattr(queue.signal, "signal", lambda *args: None)
    monkeypatch.setattr(queue, "validate_job", lambda directory, job: {
        "completed_samples": job["samples"], "sample_ids": [f"{job['group']}-id"],
        "has_actual_eviction": True, "summary": {"exact_match": 0, "f1_score": 0},
    })
    queue.execute(args)
    assert len(calls) == 84
    assert queue.read_json(args.output / "queue_status.json")["status"] == "complete"
    assert queue.read_json(args.output / "SUMMARY.json")["completed_main"] == 42
    assert all(len(kwargs["pass_fds"]) == 2 for _, kwargs in calls)
    assert all(kwargs["env"]["CUDA_VISIBLE_DEVICES"].startswith("GPU-") for _, kwargs in calls)
