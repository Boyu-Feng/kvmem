import contextlib
import fcntl
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_parallel_qwen_ablation as qwen
from scripts.run_qa32_rebuttal import atomic_json, file_hash, read_json


@pytest.fixture
def args(tmp_path):
    output = tmp_path / "results/component"
    output.mkdir(parents=True)
    model = tmp_path / "7b_ready.json"
    atomic_json(model, {"repo": qwen.queue.MODEL_REPOS["qwen25_7b"]})
    return SimpleNamespace(
        output=output, tmp_root=tmp_path / "ablation_caches/tmp",
        data_ready=tmp_path / "data_ready.json", model_readys={"qwen25_7b": model},
        gpus=[0, 1, 2, 7], max_workers=4, min_free_gib=32, poll_seconds=0.001,
        lease_dir=tmp_path / "results/shared_accuracy/qwen_component_parallel_leases",
    )


def original_plan(args):
    # The transferred Llama placeholder must never be opened or marked complete.
    models = {**{key: str(path) for key, path in args.model_readys.items()},
              "llama31_8b": "/does/not/exist/wrong_llama_ready.json"}
    plan = {
        "protocol": "component_ablation_original_shared_gpu_v1",
        "model_readys": models, "jobs": qwen.queue.jobs(models),
        "data_ready": str(args.data_ready), "tmp_root": str(args.tmp_root),
        "gpus": list(range(8)), "max_workers": 3, "min_free_gib": 48,
        "source_gate": qwen.queue.source_gate(), "timing_policy": "diagnostic only",
        "full_control": "full StepKV",
    }
    # Include whitespace: equality of parsed JSON is insufficient for preservation.
    import json
    (args.output / "queue_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    atomic_json(args.output / "queue_status.json", {"owner": None, "active": []})
    return plan


def proc_process(proc, pid, command, visible="GPU-0"):
    directory = proc / str(pid)
    directory.mkdir(parents=True)
    directory.joinpath("cmdline").write_bytes(
        b"\0".join(os.fsencode(value) for value in command) + b"\0")
    directory.joinpath("environ").write_bytes(
        f"CUDA_VISIBLE_DEVICES={visible}\0".encode())
    return directory


def peer_command(model="qwen25_7b"):
    root, ready = (qwen.QWEN_ROOT, qwen.QWEN_MODEL) if model == "qwen25_7b" else (
        qwen.LLAMA_ROOT, qwen.LLAMA_MODEL)
    job = qwen.queue.jobs({model: ready})[0]
    peer_args = SimpleNamespace(output=root, model_readys={model: ready},
                                data_ready=qwen.DATA_READY)
    return qwen.queue.command(peer_args, job)


def test_plan_bytes_and_original_source_gate_unchanged_qwen_filter(args):
    original_plan(args)
    path = args.output / "queue_plan.json"
    before = path.read_bytes()
    plan, _ = qwen.prepare_plan(args)
    assert path.read_bytes() == before
    assert plan["original_plan_sha256"] == file_hash(path)
    assert plan["source_gate"] == qwen.queue.source_gate()
    assert "scripts/run_shared_llama_ablation.py" not in plan["source_gate"]
    assert len(plan["jobs"]) == 84
    assert sum(job["purpose"] == "main" for job in plan["jobs"]) == 42
    assert {job["model"] for job in plan["jobs"]} == {"qwen25_7b"}
    assert plan["jobs"] == qwen.queue.jobs(args.model_readys)
    assert qwen.prepare_plan(args)[0] == plan


@pytest.mark.parametrize("key,value", [
    ("jobs", []), ("source_gate", {}), ("data_ready", "different"),
    ("tmp_root", "different"), ("protocol", "different"),
    ("model_readys", {"qwen25_7b": "/other", "llama31_8b": "/placeholder"}),
])
def test_prepare_rejects_frozen_job_source_assets_changes(args, key, value):
    plan = original_plan(args)
    atomic_json(args.output / "queue_plan.json", {**plan, key: value})
    before = (args.output / "queue_plan.json").read_bytes()
    with pytest.raises(RuntimeError, match="Frozen"):
        qwen.prepare_plan(args)
    assert (args.output / "queue_plan.json").read_bytes() == before


def test_prepare_rejects_alive_workers_owner_and_provenance_change(args, monkeypatch):
    original_plan(args)
    monkeypatch.setattr(qwen, "identity_alive", lambda value: value == {"pid": 10})
    for record in ({"owner": {"pid": 10}}, {"active": [{"worker": {"pid": 10}}]}):
        atomic_json(args.output / "queue_status.json", record)
        with pytest.raises(RuntimeError, match="still alive"):
            qwen.prepare_plan(args)
    atomic_json(args.output / "queue_status.json", {})
    qwen.prepare_plan(args)
    path = args.output / "parallel_qwen_plan.json"
    atomic_json(path, {**read_json(path), "launcher_sha256": "other"})
    with pytest.raises(RuntimeError, match="provenance"):
        qwen.prepare_plan(args)


def test_individual_smokes_full_baseline_and_same_group_concurrency(args):
    jobs = qwen.queue.jobs(args.model_readys)
    group = jobs[0]["group"]
    jobs = [job for job in jobs if job["group"] == group]
    assert all(job["purpose"] == "smoke" for job in qwen.next_jobs(jobs, set(), {}))
    completed = {f"{group}/smoke/full", f"{group}/smoke/no_repeat"}
    available = qwen.next_jobs(jobs, completed, {})
    assert f"{group}/main/full" in {job["name"] for job in available}
    assert f"{group}/main/no_repeat" not in {job["name"] for job in available}
    completed.add(f"{group}/main/full")
    available = qwen.next_jobs(jobs, completed, {})
    assert f"{group}/main/no_repeat" in {job["name"] for job in available}
    assert f"{group}/main/no_novelty" not in {job["name"] for job in available}
    completed.add(f"{group}/smoke/no_novelty")
    running = next(job for job in jobs if job["variant"] == "no_repeat"
                   and job["purpose"] == "main")
    available = qwen.next_jobs(jobs, completed, {1: {"job": running}})
    assert f"{group}/main/no_novelty" in {job["name"] for job in available}
    assert running["name"] not in {job["name"] for job in available}
    assert all(job["group"] == group for job in available)


@pytest.mark.parametrize("model", ["qwen25_7b", "llama31_8b"])
def test_exact_component_peer_recognition_uid_and_loading(args, tmp_path, monkeypatch, model):
    directory = proc_process(tmp_path, 10, peer_command(model))
    device = {"index": 0, "uuid": "GPU-0", "free_mib": 100000, "pids": [10]}
    assert qwen.known_peer(10, tmp_path)
    monkeypatch.setattr(qwen, "loaded_status", lambda *a: True)
    assert qwen.available(device, args, {}, tmp_path)
    monkeypatch.setattr(qwen, "loaded_status", lambda *a: False)
    assert qwen.assigned_blocker(0, "GPU-0", args, {}, device, tmp_path)
    monkeypatch.setattr(qwen, "loaded_status", lambda *a: True)
    assert not qwen.assigned_blocker(0, "GPU-0", args, {}, device, tmp_path)
    assert qwen.assigned_blocker(0, "GPU-0", args, {}, {**device, "pids": []}, tmp_path)
    for option, value in (("--model-ready", "/fake/assets/7b_ready.json"),
                          ("--output", str(qwen.BASE / "results/qa32/job")),
                          ("--data-ready", "/fake/data_ready.json")):
        command = peer_command(model)
        command[command.index(option) + 1] = value
        directory.joinpath("cmdline").write_bytes(b"\0".join(os.fsencode(x) for x in command))
        assert not qwen.known_peer(10, tmp_path)
        assert not qwen.available(device, args, {}, tmp_path)
    directory.joinpath("cmdline").write_bytes(
        b"\0".join(os.fsencode(x) for x in peer_command(model)))
    monkeypatch.setattr(qwen.os, "getuid", lambda: directory.stat().st_uid + 1)
    assert not qwen.known_peer(10, tmp_path)
    assert not qwen.available(device, args, {}, tmp_path)


def test_sensitivity_exact_runner_root_and_command(tmp_path):
    output = qwen.SENS_ROOT / "qwen25_7b/seed233/musique/r20/main/repeat_penalty_0.6"
    command = [
        "/usr/bin/python3", "-u", str(qwen.SENS_WORKER), "--execute",
        "--model-ready", str(qwen.QWEN_MODEL), "--data-ready", str(qwen.DATA_READY),
        "--output", str(output), "--dataset", "musique", "--seed", "233",
        "--variant", "full", "--purpose", "main", "--case", "repeat_penalty_0.6",
        "--ratio", "0.2",
    ]
    directory = proc_process(tmp_path, 10, command)
    assert qwen.known_peer(10, tmp_path)
    for index, value in ((2, "latency.py"), (command.index("--ratio") + 1, "0.5"),
                         (command.index("--output") + 1, str(output / "../escape"))):
        altered = command.copy()
        altered[index] = value
        directory.joinpath("cmdline").write_bytes(b"\0".join(os.fsencode(x) for x in altered))
        assert not qwen.known_peer(10, tmp_path)


def test_nonpython_cpu_environ_permission_is_skipped_before_read(args, tmp_path, monkeypatch):
    proc_process(tmp_path, 10, ["/usr/sbin/sshd"])
    proc_process(tmp_path, 11, ["/usr/lib/systemd/systemd"])
    original = Path.read_bytes

    def read(path):
        if path.name == "environ":
            raise PermissionError("CPU service environ cannot be read")
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", read)
    device = {"index": 0, "uuid": "GPU-0", "pids": []}
    assert not qwen.assigned_blocker(0, "GPU-0", args, {}, device, tmp_path)
    # Inaccessible owned Python assignments are still fail-closed.
    proc_process(tmp_path, 12, ["/usr/bin/python3", "unknown.py"])
    assert qwen.assigned_blocker(0, "GPU-0", args, {}, device, tmp_path)


def test_unknown_qa32_cuda_and_minimum_memory(args, tmp_path):
    proc_process(tmp_path, 10, ["/usr/bin/python3", "-u", "run_qa32_rebuttal.py"])
    device = {"index": 0, "uuid": "GPU-0", "pids": [10], "free_mib": 100000}
    assert not qwen.available(device, args, {}, tmp_path)
    assert qwen.assigned_blocker(0, "GPU-0", args, {}, device, tmp_path)
    assert qwen.available({**device, "pids": [], "free_mib": 32 * 1024}, args, {}, tmp_path)
    assert not qwen.available({**device, "pids": [], "free_mib": 32 * 1024 - 1},
                              args, {}, tmp_path)


def test_startup_status_context_and_caps(args):
    job = qwen.queue.jobs(args.model_readys)[0]
    directory = args.output / job["name"]
    directory.mkdir(parents=True)
    record = dict(process=SimpleNamespace(pid=10), job=job, gpu=0,
                  started=0, startup_ready=False)
    atomic_json(directory / "status.json", {"status": "running", "pid": 10})
    assert not qwen.slots_available({"index": 0, "pids": []}, args, {1: record}, 4, {})
    record["started"] = (directory / "status.json").stat().st_mtime + 1
    assert not qwen.slots_available({"index": 0, "pids": [10]}, args, {1: record}, 4, {})
    record["started"] = 0
    atomic_json(directory / "status.json", {"status": "loading_model", "pid": 10})
    assert not qwen.slots_available({"index": 0, "pids": [10]}, args, {1: record}, 4, {})
    atomic_json(directory / "status.json", {"status": "running", "pid": 10})
    assert qwen.slots_available({"index": 0, "pids": [10]}, args, {1: record}, 4, {})
    assert not qwen.slots_available({"index": 0, "pids": [10]}, args,
                                    {1: record, 2: record.copy()}, 4, {})
    assert not qwen.slots_available({"index": 0, "pids": [10]}, args, {1: record}, 4, {0: 1})
    assert not qwen.slots_available({"index": 7, "pids": []}, args,
                                    {i: {**record, "gpu": i % 3} for i in range(4)}, 4, {})


def test_frozen_summary_expected_42_and_identical_ids(args, monkeypatch):
    original_plan(args)
    plan, _ = qwen.prepare_plan(args)
    jobs = plan["jobs"]
    completed = {job["name"] for job in jobs if job["purpose"] == "main"}
    monkeypatch.setattr(qwen.queue, "validate_job",
                        lambda directory, job: {"sample_ids": ["a", "b"], "completed_samples": 500})
    qwen.queue.summarize(args.output, plan, completed)
    summary = read_json(args.output / "SUMMARY.json")
    assert summary["expected_main"] == summary["completed_main"] == 42
    assert {row["model"] for row in summary["results"]} == {"qwen25_7b"}
    monkeypatch.setattr(qwen.queue, "validate_job", lambda directory, job: {
        "sample_ids": ["a", "b"] if job["variant"] == "full" else ["b", "a"],
    })
    with pytest.raises(RuntimeError, match="identical ordered IDs"):
        qwen.queue.summarize(args.output, plan, completed)


@pytest.mark.parametrize("message,expected", [
    ("CUDA out of memory", True), ("torch.cuda.OutOfMemoryError", True),
    ("CUDA error: out of memory", True), ("MemoryError: host allocation", False),
    ("Killed", False), ("CUDA illegal memory access", False),
])
def test_oom_current_attempt_only(tmp_path, message, expected):
    path = tmp_path / "run.log"
    path.write_text("historical CUDA out of memory\n")
    offset = path.stat().st_size
    with path.open("a") as log:
        log.write(message)
    assert qwen.is_cuda_oom(tmp_path, offset, 0) == expected


def simulation(args, monkeypatch, failure=None):
    """Exercise the real loop using fake child objects; never spawn or signal."""
    frozen_jobs = qwen.queue.jobs
    base = frozen_jobs(args.model_readys)[0]
    group = base["group"]
    jobs = [{**base, "name": f"{group}/{purpose}/{variant}", "purpose": purpose,
             "variant": variant, "samples": 2 if purpose == "smoke" else 500}
            for purpose in ("smoke", "main")
            for variant in ("full", "no_repeat", "no_novelty", "no_success", "no_cite", "no_token")]
    placeholder = [{**jobs[0], "name": "llama_placeholder", "model": "llama31_8b"}]
    monkeypatch.setattr(qwen.queue, "jobs",
                        lambda models: jobs + placeholder if "llama31_8b" in models else jobs)
    original_plan(args)
    processes, attempts, completed, admissions = [], {}, set(), []
    monkeypatch.setattr(qwen, "process_identity", lambda pid: {"pid": pid, "start_ticks": 1})
    monkeypatch.setattr(qwen, "identity_alive", lambda *a: False)
    monkeypatch.setattr(qwen, "own_worker", lambda pid, args, active, proc=Path("/proc"):
                        any(r["process"].pid == pid for r in active.values()))
    monkeypatch.setattr(qwen, "assigned_blocker", lambda *a, **kw: False)
    monkeypatch.setattr(qwen, "maintenance_blockers", lambda *a: {})
    monkeypatch.setattr(qwen, "maintenance_launch_lock", lambda gpu: contextlib.nullcontext())
    monkeypatch.setattr(qwen.queue, "validate_job", lambda directory, job: {})
    monkeypatch.setattr(qwen.queue, "summarize", lambda *a: None)
    monkeypatch.setattr(qwen.time, "sleep", lambda seconds: None)

    def snapshot():
        return {gpu: {"index": gpu, "uuid": f"GPU-{gpu}",
                      "free_mib": 100000 - 14000 * sum(p.gpu == gpu and not p.done for p in processes),
                      "pids": [p.pid for p in processes if p.gpu == gpu and not p.done]}
                for gpu in args.gpus}
    monkeypatch.setattr(qwen.queue, "gpu_snapshot", snapshot)

    class Child:
        def __init__(self, command, **kwargs):
            self.pid = 900000 + len(processes)
            self.gpu = int(kwargs["env"]["CUDA_VISIBLE_DEVICES"].split("-")[-1])
            self.output = Path(qwen.flag(command, "--output"))
            self.name = str(self.output.relative_to(args.output))
            self.done, self.polls, self.code = False, 0, 0
            self.kwargs = kwargs
            attempts[self.name] = attempts.get(self.name, 0) + 1
            self.attempt = attempts[self.name]
            job = next(job for job in jobs if job["name"] == self.name)
            assert command == qwen.queue.command(args, job)
            if job["purpose"] == "main":
                assert f"{group}/smoke/{job['variant']}" in completed
                if job["variant"] != "full":
                    assert f"{group}/main/full" in completed
            live = [p for p in processes if not p.done]
            assert len(live) < 4
            assert sum(p.gpu == self.gpu for p in live) < 2
            assert all(p.polls >= 2 for p in live if p.gpu == self.gpu)
            assert self.gpu in (0, 1, 2, 7)
            for path in (args.output / "queue.lock",
                         args.lease_dir / f"alfworld_gpu{self.gpu}.lock"):
                with path.open("a") as handle:
                    with pytest.raises(BlockingIOError):
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            # No original exclusive GPU lease is claimed.
            assert not (args.output.parent / f"alfworld_gpu{self.gpu}.lock").exists()
            assert len(kwargs["pass_fds"]) == 2
            assert all(os.fstat(fd) for fd in kwargs["pass_fds"])
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                assert kwargs["env"][key] == "4"
            assert kwargs["env"]["TMPDIR"] == str(args.tmp_root)
            assert kwargs["env"]["HF_HOME"] == str(args.tmp_root.parent / "hf")
            admissions.append((self.name, self.gpu, len(live) + 1))
            processes.append(self)
            atomic_json(self.output / "status.json", {"status": "loading_model", "pid": self.pid})

        def poll(self):
            if self.done:
                return self.code
            self.polls += 1
            if self.polls == 2:
                atomic_json(self.output / "status.json", {"status": "running", "pid": self.pid})
            if self.polls < 5:
                return None
            self.done = True
            if failure and (failure == "always_oom" or
                            (self.name == jobs[0]["name"] and self.attempt == 1)):
                self.code = 1
                self.kwargs["stdout"].write(
                    b"CUDA out of memory\n" if failure in ("oom", "always_oom") else b"invalid data\n")
                self.kwargs["stdout"].flush()
                (self.output / "checkpoint.json").write_text("committed prefix")
                return 1
            completed.add(self.name)
            atomic_json(self.output / "status.json", {"status": "complete", "pid": self.pid})
            return 0

        def terminate(self):
            self.done, self.code = True, -15

        def wait(self, timeout):
            assert self.done
            return self.code
    monkeypatch.setattr(qwen.subprocess, "Popen", Child)
    return jobs, processes, attempts, admissions, completed


def test_loop_four_slots_same_group_gates_and_original_bytes(args, monkeypatch):
    jobs, processes, attempts, admissions, _ = simulation(args, monkeypatch)
    before = (args.output / "queue_plan.json").read_bytes()
    qwen.execute(args)
    status = read_json(args.output / "queue_status.json")
    assert status["status"] == "complete"
    assert status["peak_active_workers"] == 4
    assert max(item[2] for item in admissions) == 4
    assert status["completed_main"] == len(jobs) // 2
    assert status["active"] == []
    assert status["oom_events"] == []
    assert all(p.done for p in processes)
    assert all(count == 1 for count in attempts.values())
    assert (args.output / "queue_plan.json").read_bytes() == before


def test_loop_oom_retries_preserves_checkpoint_and_reduces_limit(args, monkeypatch):
    jobs, _, attempts, _, _ = simulation(args, monkeypatch, failure="oom")
    qwen.execute(args)
    status = read_json(args.output / "queue_status.json")
    assert status["status"] == "complete"
    assert status["effective_max_workers"] == 3
    assert attempts[jobs[0]["name"]] == 2
    assert status["gpu_worker_limits"]
    assert len(status["oom_events"]) == 1
    assert (args.output / jobs[0]["name"] / "checkpoint.json").read_text() == "committed prefix"


@pytest.mark.parametrize("failure,error", [
    ("ordinary", "Ablation failed"), ("always_oom", "retry limit exhausted"),
])
def test_loop_failures_loud_bounded_and_children_stopped(args, monkeypatch, failure, error):
    _, processes, attempts, _, _ = simulation(args, monkeypatch, failure=failure)
    with pytest.raises(RuntimeError, match=error):
        qwen.execute(args)
    status = read_json(args.output / "queue_status.json")
    assert status["status"] == "failed"
    assert status["active"] == []
    assert all(p.done for p in processes)
    assert max(attempts.values()) == (3 if failure == "always_oom" else 1)


def test_resume_complete_skips_llama_and_keeps_partial_checkpoint(args, monkeypatch):
    jobs, processes, _, _, completed = simulation(args, monkeypatch)
    for job in jobs[:-1]:
        directory = args.output / job["name"]
        directory.mkdir(parents=True)
        atomic_json(directory / "status.json", {"status": "complete"})
        completed.add(job["name"])
    last = args.output / jobs[-1]["name"]
    last.mkdir(parents=True)
    atomic_json(last / "status.json", {"status": "blocked", "completed_samples": 174})
    (last / "checkpoint.json").write_text("174 committed")
    qwen.execute(args)
    assert len(processes) == 1
    assert processes[0].name == jobs[-1]["name"]
    assert (last / "checkpoint.json").read_text() == "174 committed"
    assert read_json(args.output / "queue_status.json")["status"] == "complete"


@pytest.mark.parametrize("change", ["source", "original_plan_bytes", "scheduler_plan", "model_pin"])
def test_periodic_protection_stops_children(args, monkeypatch, change):
    _, processes, _, _, _ = simulation(args, monkeypatch)

    def modify(seconds):
        if change == "source":
            monkeypatch.setattr(qwen.queue, "source_gate", lambda: {"changed": "source"})
        elif change == "original_plan_bytes":
            path = args.output / "queue_plan.json"
            path.write_bytes(path.read_bytes() + b" ")
        elif change == "scheduler_plan":
            atomic_json(args.output / "parallel_qwen_plan.json", {"changed": True})
        else:
            atomic_json(args.model_readys["qwen25_7b"], {
                "repo": qwen.queue.MODEL_REPOS["qwen25_7b"], "changed": True,
            })
    monkeypatch.setattr(qwen.time, "sleep", modify)
    with pytest.raises(RuntimeError, match="changed"):
        qwen.execute(args)
    assert all(p.done for p in processes)
    assert read_json(args.output / "queue_status.json")["status"] == "failed"


def test_interrupt_during_creation_tracks_child_for_cleanup(args, monkeypatch):
    _, processes, _, _, _ = simulation(args, monkeypatch)
    original = qwen.subprocess.Popen

    def launch(*command, **kwargs):
        process = original(*command, **kwargs)
        qwen.signal.getsignal(qwen.signal.SIGTERM)(qwen.signal.SIGTERM, None)
        return process
    monkeypatch.setattr(qwen.subprocess, "Popen", launch)
    with pytest.raises(KeyboardInterrupt, match="Received signal"):
        qwen.execute(args)
    assert len(processes) == 1 and processes[0].done
    assert read_json(args.output / "queue_status.json")["status"] == "failed"


def test_exhausted_retry_not_reset_on_restart(args, monkeypatch):
    jobs, processes, _, _, _ = simulation(args, monkeypatch)
    atomic_json(args.output / "queue_status.json", {
        "oom_retries": {jobs[0]["name"]: 3}, "effective_max_workers": 1,
        "owner": None, "active": [],
    })
    with pytest.raises(RuntimeError, match="retry limit exhausted"):
        qwen.execute(args)
    assert processes == []


def test_resume_completed_main_requires_validated_smoke(args, monkeypatch):
    jobs, processes, _, _, _ = simulation(args, monkeypatch)
    main = next(job for job in jobs if job["purpose"] == "main" and job["variant"] == "full")
    path = args.output / main["name"]
    path.mkdir(parents=True)
    atomic_json(path / "status.json", {"status": "complete"})
    with pytest.raises(RuntimeError, match="lacks validated smoke"):
        qwen.execute(args)
    assert processes == []


def test_own_worker_exact_identity_command(args, tmp_path, monkeypatch):
    job = qwen.queue.jobs(args.model_readys)[0]
    record = {"process": SimpleNamespace(pid=10), "worker": {"pid": 10}, "job": job}
    path = proc_process(tmp_path, 10, qwen.queue.command(args, job))
    monkeypatch.setattr(qwen, "identity_alive", lambda *a: True)
    assert qwen.own_worker(10, args, {1: record}, tmp_path)
    assert not qwen.own_worker(10, args, {}, tmp_path)
    monkeypatch.setattr(qwen, "identity_alive", lambda *a: False)
    assert not qwen.own_worker(10, args, {1: record}, tmp_path)
    monkeypatch.setattr(qwen, "identity_alive", lambda *a: True)
    path.joinpath("cmdline").write_bytes(b"/usr/bin/python3\0unknown.py")
    assert not qwen.own_worker(10, args, {1: record}, tmp_path)


def test_maintenance_prevents_every_launch_and_leases_release(args, monkeypatch):
    _, processes, _, _, _ = simulation(args, monkeypatch)
    monkeypatch.setattr(qwen, "maintenance_blockers", lambda *a: {"owner": {"pid": 10}})

    def stop_wait(seconds):
        assert processes == []
        raise KeyboardInterrupt("end CPU-only maintenance test")
    monkeypatch.setattr(qwen.time, "sleep", stop_wait)
    with pytest.raises(KeyboardInterrupt):
        qwen.execute(args)
    for gpu in args.gpus:
        with (args.lease_dir / f"alfworld_gpu{gpu}.lock").open("a") as lease:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_cpu_controller_cli_rejects_before_any_execution(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(qwen.sys, "argv", [
        "run_parallel_qwen_ablation.py", "--model-ready", str(qwen.QWEN_MODEL),
        "--data-ready", str(qwen.DATA_READY), "--output", str(qwen.QWEN_ROOT),
        "--tmp-root", str(qwen.BASE / "ablation_caches/tmp"),
    ])
    called = []
    monkeypatch.setattr(qwen, "execute", lambda args: called.append(args))
    with pytest.raises(SystemExit):
        qwen.main()
    assert called == []


def qa32_command(dataset="hotpotqa", cap="44"):
    output = qwen.BASE / f"results/qa32_{dataset}_v1/{dataset}/main/stepkv/r20"
    return [
        "/usr/bin/python3", "-u", str(qwen.QA32_WORKER), "--assets",
        str(qwen.BASE / "qa32_assets"), "--output", str(output), "--memory-limit-gib", cap,
        "--dataset", dataset, "--method", "stepkv", "--ratio", "0.2", "--purpose", "main",
        "--samples", "500", "--sample-start", "0", "--seed", "233", "--max-steps", "7",
    ]


@pytest.mark.parametrize("dataset", ["hotpotqa", "2wiki", "musique"])
@pytest.mark.parametrize("cap", ["44", "44.0"])
def test_qa32_exact_peer_only_after_own_running_status_and_context(
        args, tmp_path, monkeypatch, dataset, cap):
    monkeypatch.setattr(qwen, "BASE", tmp_path / "base")
    proc = tmp_path / "proc"
    command = qa32_command(dataset, cap)
    proc_process(proc, 10, command, visible="0,7")
    output = Path(qwen.flag(command, "--output"))
    output.mkdir(parents=True)
    assert qwen.known_peer(10, proc)
    for gpu in (0, 7):
        device = {"index": gpu, "uuid": f"GPU-{gpu}", "free_mib": 100000, "pids": [10]}
        devices = {card: {"index": card, "uuid": f"GPU-{card}",
                          "free_mib": 100000, "pids": [10]} for card in (0, 7)}
        for stage in ("loading_retriever", "loading_model", "complete", "blocked"):
            atomic_json(output / "status.json", {"status": stage, "pid": 10})
            assert not qwen.available(device, args, {}, proc, devices)
            assert qwen.assigned_blocker(gpu, device["uuid"], args, {}, device, proc, devices)
        atomic_json(output / "status.json", {"status": "running", "pid": 10})
        assert qwen.available(device, args, {}, proc, devices)
        assert not qwen.assigned_blocker(gpu, device["uuid"], args, {}, device, proc, devices)
        assert qwen.assigned_blocker(gpu, device["uuid"], args, {},
                                     {**device, "pids": []}, proc, devices)
        assert not qwen.available({**device, "free_mib": 32 * 1024 - 1},
                                  args, {}, proc, devices)
        other_gpu = 7 if gpu == 0 else 0
        one_context = {**devices, other_gpu: {**devices[other_gpu], "pids": []}}
        assert not qwen.available(device, args, {}, proc, one_context)
        assert qwen.assigned_blocker(gpu, device["uuid"], args, {}, device, proc, one_context)
        atomic_json(output / "status.json", {"status": "running", "pid": 999})
        assert not qwen.available(device, args, {}, proc, devices)


@pytest.mark.parametrize("option,value", [
    ("--assets", "/other/qa32_assets"), ("--memory-limit-gib", "43"),
    ("--memory-limit-gib", "88"), ("--output", "/unofficial/qa32_hotpotqa_v1/job"),
    ("--samples", "499"), ("--sample-start", "1"), ("--seed", "42"),
    ("--ratio", "0.5"),
])
def test_qa32_no_broad_runner_cap_asset_output_or_protocol_exemption(
        args, tmp_path, monkeypatch, option, value):
    monkeypatch.setattr(qwen, "BASE", tmp_path / "base")
    command = qa32_command()
    command[command.index(option) + 1] = value
    proc_process(tmp_path / "proc", 10, command)
    assert not qwen.known_peer(10, tmp_path / "proc")
    device = {"index": 0, "uuid": "GPU-0", "free_mib": 100000, "pids": [10]}
    assert not qwen.available(device, args, {}, tmp_path / "proc")
    assert qwen.assigned_blocker(0, "GPU-0", args, {}, device, tmp_path / "proc")


def test_qa32_foreign_uid_blocked_and_worker_already_frozen(tmp_path, monkeypatch):
    assert "scripts/run_qa32_rebuttal.py" in qwen.queue.source_gate()
    assert qwen.queue.source_gate()["scripts/run_qa32_rebuttal.py"] == file_hash(qwen.QA32_WORKER)
    monkeypatch.setattr(qwen, "BASE", tmp_path / "base")
    directory = proc_process(tmp_path / "proc", 10, qa32_command())
    assert qwen.known_peer(10, tmp_path / "proc")
    monkeypatch.setattr(qwen.os, "getuid", lambda: directory.stat().st_uid + 1)
    assert not qwen.known_peer(10, tmp_path / "proc")


@pytest.mark.parametrize("visible,allowed", [
    ("3,4", True), ("5,6", True), ("0,7", True), ("GPU-0,GPU-7", True),
    ("0,1", False), ("0,0", False), ("0", False), ("", False), ("7,0", False),
])
def test_qa32_both_actual_contexts_configured_pair_only(tmp_path, visible, allowed):
    proc_process(tmp_path, 10, qa32_command(), visible=visible)
    devices = {gpu: {"index": gpu, "uuid": f"GPU-{gpu}", "pids": [10]} for gpu in range(8)}
    assert qwen.qa32_contexts_ready(10, devices, tmp_path) is allowed
    assert not qwen.qa32_contexts_ready(10, None, tmp_path)
    if allowed:
        for selector in visible.split(","):
            gpu = int(selector.removeprefix("GPU-"))
            missing_context = {**devices, gpu: {**devices[gpu], "pids": []}}
            assert not qwen.qa32_contexts_ready(10, missing_context, tmp_path)
