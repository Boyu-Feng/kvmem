import fcntl
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_parallel_llama_ablation as parallel
from scripts.run_qa32_rebuttal import atomic_json, file_hash, read_json


@pytest.fixture
def args(tmp_path):
    output = tmp_path / "shared_accuracy/llama"
    output.mkdir(parents=True)
    model = tmp_path / "llama_ready.json"
    atomic_json(model, {"repo": parallel.queue.MODEL_REPOS["llama31_8b"]})
    return SimpleNamespace(
        output=output, tmp_root=tmp_path / "cache/tmp", data_ready=tmp_path / "data.json",
        model_readys={"llama31_8b": model}, gpus=[0, 1, 2],
        max_workers=5, min_free_gib=32, poll_seconds=0.001,
    )


def old_plan(args):
    plan = {**parallel.expected_plan(args), "max_workers": 1}
    atomic_json(args.output / "queue_plan.json", plan)
    atomic_json(args.output / "queue_status.json", {"owner": None, "active": []})
    return plan


def process(proc, pid, command, gpu="GPU-0"):
    directory = proc / str(pid)
    directory.mkdir(parents=True)
    directory.joinpath("cmdline").write_bytes(
        b"\0".join(os.fsencode(value) for value in command) + b"\0")
    directory.joinpath("environ").write_bytes(f"CUDA_VISIBLE_DEVICES={gpu}\0".encode())
    return directory


def peer_command(script=None, model=None, output=None):
    script = script or str(parallel.queue.WORKER)
    output = output or str(parallel.PEER_SCOPES[script] / "qwen25_7b/seed233/hotpotqa/main/full")
    return ["/usr/bin/python3", "-u", script, "--execute", "--model-ready",
            str(model or parallel.PEER_MODEL), "--output", output]


def test_source_gate_frozen_and_no_helper_monkeypatch():
    before = parallel.shared.existing_accuracy
    expected = {**parallel.queue.source_gate(),
                "scripts/run_shared_llama_ablation.py": file_hash(Path(parallel.shared.__file__))}
    assert parallel.source_gate() == expected
    assert parallel.shared.existing_accuracy is before
    assert parallel.ORIGINAL_EXISTING_ACCURACY is before
    assert "scripts/run_parallel_llama_ablation.py" not in expected


def test_plan_migration_only_changes_concurrency_and_retains_provenance(args):
    plan = old_plan(args)
    migrated, _ = parallel.migrate_plan(args)
    assert migrated == {**plan, "max_workers": 5}
    provenance = read_json(args.output / "parallel_scheduler.json")
    assert provenance["previous_plan"] == plan
    assert provenance["launcher_sha256"] == file_hash(Path(parallel.__file__))
    assert parallel.migrate_plan(args)[0] == migrated


@pytest.mark.parametrize("key,value", [
    ("min_free_gib", 31), ("gpus", [3, 4, 5]), ("max_workers", 3),
    ("jobs", []), ("data_ready", "other"), ("tmp_root", "other"),
    ("source_gate", {}), ("model_readys", {}), ("protocol", "other"),
    ("timing_policy", "other"), ("extra", "not allowed"),
])
def test_migration_rejects_experiment_and_resource_changes(args, key, value):
    plan = old_plan(args)
    atomic_json(args.output / "queue_plan.json", {**plan, key: value})
    with pytest.raises(RuntimeError, match="Only max_workers"):
        parallel.migrate_plan(args)
    assert not (args.output / "parallel_scheduler.json").exists()


def test_migration_rejects_live_controller_worker_and_changed_provenance(args, monkeypatch):
    old_plan(args)
    monkeypatch.setattr(parallel, "identity_alive", lambda value: value == {"pid": 10})
    for previous in ({"owner": {"pid": 10}}, {"active": [{"worker": {"pid": 10}}]}):
        atomic_json(args.output / "queue_status.json", previous)
        with pytest.raises(RuntimeError, match="still alive"):
            parallel.migrate_plan(args)
    atomic_json(args.output / "queue_status.json", {})
    parallel.migrate_plan(args)
    record = read_json(args.output / "parallel_scheduler.json")
    atomic_json(args.output / "parallel_scheduler.json", {**record, "launcher_sha256": "changed"})
    with pytest.raises(RuntimeError, match="provenance changed"):
        parallel.migrate_plan(args)


def test_unknown_qa32_and_non_owned_peers_block(args, tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    proc.mkdir()
    device = {"index": 0, "uuid": "GPU-0", "free_mib": 100000, "pids": [10]}
    directory = process(proc, 10, peer_command())
    monkeypatch.setattr(parallel, "loaded_status", lambda *a: True)
    assert parallel.known_accuracy(10, proc)
    assert parallel.available(device, args, {}, proc)
    # Known peers still block admission until their loading allocations are visible.
    monkeypatch.setattr(parallel, "loaded_status", lambda *a: False)
    assert parallel.assigned_blocker(0, "GPU-0", args, {}, device, proc)
    monkeypatch.setattr(parallel, "loaded_status", lambda *a: True)
    assert not parallel.assigned_blocker(0, "GPU-0", args, {}, device, proc)
    assert parallel.assigned_blocker(0, "GPU-0", args, {}, {**device, "pids": []}, proc)
    for command in (
        peer_command(model="/some/assets/7b_ready.json"),
        peer_command(output=str(parallel.BASE / "results/qa32/qwen25_7b/seed233/x/main/full")),
        ["/usr/bin/python3", "-u", str(parallel.ROOT / "scripts/run_qa32_rebuttal.py")],
        ["/usr/bin/python3", "-u", "latency.py"],
        ["/bin/unknown"],
    ):
        directory.joinpath("cmdline").write_bytes(b"\0".join(os.fsencode(x) for x in command))
        assert not parallel.known_accuracy(10, proc)
        assert not parallel.available(device, args, {}, proc)
        assert parallel.assigned_blocker(0, "GPU-0", args, {}, device, proc) == (
            command[0] != "/bin/unknown")
    directory.joinpath("cmdline").write_bytes(b"\0".join(os.fsencode(x) for x in peer_command()))
    monkeypatch.setattr(parallel.os, "getuid", lambda: directory.stat().st_uid + 1)
    assert not parallel.known_accuracy(10, proc)
    assert not parallel.available(device, args, {}, proc)
    assert not parallel.assigned_blocker(0, "GPU-0", args, {}, device, proc)


def test_non_python_cpu_process_does_not_require_environment_access(args, tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    proc.mkdir()
    directory = process(proc, 10, ["/usr/lib/systemd/systemd", "--user"])
    original = Path.read_bytes

    def read(path):
        if path == directory / "environ":
            raise PermissionError("Non-dumpable CPU-only process")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    device = {"index": 0, "uuid": "GPU-0", "free_mib": 100000, "pids": []}
    assert not parallel.assigned_blocker(0, "GPU-0", args, {}, device, proc)
    directory.joinpath("cmdline").write_bytes(b"/usr/bin/python3\0unknown.py\0")
    assert parallel.assigned_blocker(0, "GPU-0", args, {}, device, proc)


def test_sensitivity_exact_scope_and_malformed_command(tmp_path):
    script = str(parallel.ROOT / "scripts/run_stepkv_sensitivity.py")
    process(tmp_path, 10, peer_command(
        script=script, output=str(parallel.PEER_SCOPES[script] /
                                "qwen25_7b/seed233/hotpotqa/r20/main/repeat_penalty_0.6")))
    assert parallel.known_accuracy(10, tmp_path)
    path = tmp_path / "10/cmdline"
    path.write_bytes(b"/usr/bin/python3\0-u\0" + script.encode() + b"\0--model-ready\0")
    assert not parallel.known_accuracy(10, tmp_path)
    assert not parallel.known_accuracy(11, tmp_path)
    path.write_bytes(b"\0".join(os.fsencode(x) for x in [
        "/usr/bin/python3", "-u", str(parallel.queue.WORKER),
        "--execute", "--model-ready",
    ]))
    assert not parallel.known_accuracy(10, tmp_path)


def test_own_workers_require_exact_command_and_live_identity(args, tmp_path, monkeypatch):
    job = parallel.queue.jobs(args.model_readys)[0]
    record = {"process": SimpleNamespace(pid=10), "worker": {"pid": 10}, "job": job}
    directory = process(tmp_path, 10, parallel.queue.command(args, job))
    monkeypatch.setattr(parallel, "identity_alive", lambda *a: True)
    assert parallel.own_worker(10, args, {1: record}, tmp_path)
    assert not parallel.own_worker(10, args, {}, tmp_path)
    directory.joinpath("cmdline").write_bytes(b"/usr/bin/python3\0latency.py")
    assert not parallel.own_worker(10, args, {1: record}, tmp_path)


def test_minmemory_unchanged_even_known_peer(args, tmp_path, monkeypatch):
    process(tmp_path, 10, peer_command())
    monkeypatch.setattr(parallel, "loaded_status", lambda *a: True)
    device = {"index": 0, "pids": [10], "free_mib": 32 * 1024}
    assert parallel.available(device, args, {}, tmp_path)
    assert not parallel.available({**device, "free_mib": 32 * 1024 - 1}, args, {}, tmp_path)


def test_qa32_peer_requires_exact_protocol_running_and_both_devices(args, tmp_path, monkeypatch):
    monkeypatch.setattr(parallel.peers, "BASE", tmp_path / "base")
    output = parallel.peers.BASE / "results/qa32_hotpotqa_v1/hotpotqa/main/tova/r20"
    output.mkdir(parents=True)
    command = [
        "/usr/bin/python3", "-u", str(parallel.peers.QA32_WORKER),
        "--assets", str(parallel.peers.BASE / "qa32_assets"), "--output", str(output),
        "--memory-limit-gib", "44",
    ]
    for name, value in {
        "dataset": "hotpotqa", "method": "tova", "ratio": "0.2", "purpose": "main",
        "samples": "500", "sample_start": "0", "seed": "233", "max_steps": "7",
    }.items():
        command.extend(["--" + name.replace("_", "-"), value])
    proc = tmp_path / "proc"
    process(proc, 10, command, gpu="0,7")
    devices = {
        0: {"index": 0, "uuid": "GPU-0", "pids": [10], "free_mib": 100000},
        7: {"index": 7, "uuid": "GPU-7", "pids": [10], "free_mib": 100000},
    }
    assert parallel.known_accuracy(10, proc)
    atomic_json(output / "status.json", {"status": "running", "pid": 10})
    assert parallel.available(devices[0], args, {}, proc, devices)
    assert not parallel.assigned_blocker(0, "GPU-0", args, {}, devices[0], proc, devices)
    devices[7]["pids"] = []
    assert not parallel.available(devices[0], args, {}, proc, devices)
    assert parallel.assigned_blocker(0, "GPU-0", args, {}, devices[0], proc, devices)
    devices[7]["pids"] = [10]
    atomic_json(output / "status.json", {"status": "loading_model", "pid": 10})
    assert not parallel.available(devices[0], args, {}, proc, devices)


def test_pending_startup_requires_current_running_status_and_cuda_context(args):
    job = parallel.queue.jobs(args.model_readys)[0]
    directory = args.output / job["name"]
    directory.mkdir(parents=True)
    record = dict(process=SimpleNamespace(pid=10), job=job, gpu=0,
                  started=0, startup_ready=False)
    active = {1: record}
    device = {"index": 0, "pids": []}
    atomic_json(directory / "status.json", {"status": "running", "pid": 10})
    assert not parallel.slots_available(device, args, active, 5, {})
    device["pids"] = [10]
    record["started"] = (directory / "status.json").stat().st_mtime + 1
    assert not parallel.slots_available(device, args, active, 5, {})
    record["started"] = 0
    atomic_json(directory / "status.json", {"status": "loading_model", "pid": 10})
    assert not parallel.slots_available(device, args, active, 5, {})
    atomic_json(directory / "status.json", {"status": "running", "pid": 10})
    assert parallel.slots_available(device, args, active, 5, {})
    active[2] = {**record, "process": SimpleNamespace(pid=11)}
    assert not parallel.slots_available(device, args, active, 5, {})
    assert not parallel.slots_available({"index": 1, "pids": []}, args, active, 2, {})
    assert not parallel.slots_available(device, args, {1: record}, 5, {0: 1})
    assert not parallel.slots_available(
        {"index": 2, "pids": []}, args, {i: {**record, "gpu": i % 2} for i in range(5)}, 5, {})


@pytest.mark.parametrize("text,expected", [
    ("torch.cuda.OutOfMemoryError: CUDA out of memory.", True),
    ("RuntimeError: CUDA error: out of memory", True),
    ("MemoryError: host allocation failed", False),
    ("Killed", False), ("RuntimeError: CUDA illegal memory access", False),
])
def test_oom_classification_only_current_attempt(tmp_path, text, expected):
    log = tmp_path / "run.log"
    log.write_text("old CUDA out of memory\n")
    offset = log.stat().st_size
    with log.open("a") as handle:
        handle.write(text)
    assert parallel.is_cuda_oom(tmp_path, offset, 0) is expected


def test_oom_retry_bounded_and_explicit():
    retries, events, limits = {}, [], {}
    limit = 5
    for attempt in range(1, 4):
        limit, allowed = parallel.record_oom({"name": "job"}, 0, retries, events, limit, limits)
        assert allowed == (attempt <= 2)
        assert limit == 5 - attempt
    assert retries == {"job": 3}
    assert limits == {0: 1}
    assert len(events) == 3
    assert events[-1]["retry_allowed"] is False


def simulated_queue(args, monkeypatch, fail=None, interrupt=False):
    """Run the actual scheduler with no subprocesses, signals, CUDA, or model loads."""
    jobs = [
        {"name": f"group{group}/{stage}", "group": f"group{group}", "model": "llama31_8b",
         "purpose": "main", "samples": 1, "dataset": "hotpotqa", "seed": 233,
         "variant": "full"}
        for group in range(6) for stage in range(2)
    ]
    monkeypatch.setattr(parallel.queue, "jobs", lambda models: jobs)
    old_plan(args)
    processes, attempts, admissions = [], {}, []
    monkeypatch.setattr(parallel, "process_identity", lambda pid: {"pid": pid, "start_ticks": 1})
    monkeypatch.setattr(parallel, "identity_alive", lambda *a: False)
    monkeypatch.setattr(parallel, "own_worker",
                        lambda pid, args, active, proc=Path("/proc"):
                        any(r["process"].pid == pid for r in active.values()))
    monkeypatch.setattr(parallel, "assigned_blocker", lambda *a, **kw: False)
    monkeypatch.setattr(parallel, "maintenance_blockers", lambda *a: {})
    monkeypatch.setattr(parallel.queue, "validate_job", lambda *a: {})
    monkeypatch.setattr(parallel.queue, "summarize", lambda *a: None)
    monkeypatch.setattr(parallel, "maintenance_launch_lock",
                        lambda gpu: __import__("contextlib").nullcontext())

    def snapshot():
        return {gpu: {"index": gpu, "uuid": f"GPU-{gpu}", "free_mib": 100000 -
                      14000 * sum(p.gpu == gpu and not p.done for p in processes),
                      "pids": [p.pid for p in processes if p.gpu == gpu and not p.done]}
                for gpu in args.gpus}
    monkeypatch.setattr(parallel.queue, "gpu_snapshot", snapshot)

    class Child:
        def __init__(self, cmd, **kwargs):
            self.pid = 900000 + len(processes)
            self.gpu = int(kwargs["env"]["CUDA_VISIBLE_DEVICES"].split("-")[-1])
            self.output = Path(parallel.flag(cmd, "--output"))
            self.name = str(self.output.relative_to(args.output))
            self.done, self.polls, self.code = False, 0, 0
            self.kwargs = kwargs
            attempts[self.name] = attempts.get(self.name, 0) + 1
            self.attempt = attempts[self.name]
            live = [p for p in processes if not p.done]
            assert len(live) < 5
            assert sum(p.gpu == self.gpu for p in live) < 2
            assert all(p.name.split("/")[0] != self.name.split("/")[0] for p in live)
            assert all(p.polls >= 2 for p in live if p.gpu == self.gpu)
            for lock_path in (args.output / "queue.lock",
                              args.output.parent / f"alfworld_gpu{self.gpu}.lock"):
                with lock_path.open("a") as lock:
                    with pytest.raises(BlockingIOError):
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert len(kwargs["pass_fds"]) == 2
            assert all(os.fstat(fd) for fd in kwargs["pass_fds"])
            assert kwargs["env"]["OMP_NUM_THREADS"] == "4"
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
            if fail and (fail == "always_oom" or (self.name == jobs[0]["name"]
                                                and self.attempt == 1)):
                self.code = 1
                self.kwargs["stdout"].write(
                    b"CUDA out of memory\n" if fail in ("oom", "always_oom") else b"invalid data\n")
                self.kwargs["stdout"].flush()
                # Represents a committed checkpoint: scheduler must not remove it.
                (self.output / "checkpoint.json").write_text("committed")
                return self.code
            atomic_json(self.output / "status.json", {"status": "complete", "pid": self.pid})
            return 0

        def terminate(self):
            self.done, self.code = True, -15

        def wait(self, timeout):
            assert self.done
            return self.code

    monkeypatch.setattr(parallel.subprocess, "Popen", Child)

    def sleep(seconds):
        if interrupt:
            raise KeyboardInterrupt("simulated SIGTERM")
    monkeypatch.setattr(parallel.time, "sleep", sleep)
    return processes, attempts, admissions


def test_scheduler_five_unique_slots_serial_groups_and_physical_limits(args, monkeypatch):
    processes, attempts, admissions = simulated_queue(args, monkeypatch)
    parallel.execute(args)
    status = read_json(args.output / "queue_status.json")
    assert status["status"] == "complete"
    assert status["completed_main"] == 12
    assert status["peak_active_workers"] == 5
    assert max(item[2] for item in admissions) == 5
    assert all(p.done for p in processes)
    assert len(attempts) == 12
    assert status["active"] == []
    assert status["oom_events"] == []


def test_scheduler_oom_retries_checkpoint_and_effective_limit(args, monkeypatch):
    _, attempts, _ = simulated_queue(args, monkeypatch, fail="oom")
    parallel.execute(args)
    status = read_json(args.output / "queue_status.json")
    assert status["status"] == "complete"
    assert status["effective_max_workers"] == 4
    assert status["oom_retries"] == {"group0/0": 1}
    assert attempts["group0/0"] == 2
    assert (args.output / "group0/0/checkpoint.json").read_text() == "committed"
    assert len(status["oom_events"]) == 1


def test_scheduler_ordinary_failure_stops_owned_children_no_retry(args, monkeypatch):
    processes, attempts, _ = simulated_queue(args, monkeypatch, fail="ordinary")
    with pytest.raises(RuntimeError, match="Ablation failed"):
        parallel.execute(args)
    status = read_json(args.output / "queue_status.json")
    assert status["status"] == "failed"
    assert status["active"] == []
    assert all(p.done for p in processes)
    assert attempts["group0/0"] == 1
    assert status["oom_events"] == []


def test_scheduler_oom_exhaustion_is_loud_and_persistent(args, monkeypatch):
    processes, attempts, _ = simulated_queue(args, monkeypatch, fail="always_oom")
    with pytest.raises(RuntimeError, match="retry limit exhausted"):
        parallel.execute(args)
    status = read_json(args.output / "queue_status.json")
    assert status["status"] == "failed"
    assert any(count == 3 for count in attempts.values())
    assert status["oom_events"][-1]["retry_allowed"] is False
    assert all(p.done for p in processes)


def test_scheduler_interrupt_explicit_failure_and_cleanup(args, monkeypatch):
    processes, _, _ = simulated_queue(args, monkeypatch, interrupt=True)
    with pytest.raises(KeyboardInterrupt):
        parallel.execute(args)
    status = read_json(args.output / "queue_status.json")
    assert status["status"] == "failed"
    assert status["active"] == []
    assert all(p.done for p in processes)


@pytest.mark.parametrize("change", ["source", "plan", "model_pin"])
def test_periodic_frozen_gate_checks_stop_own_children(args, monkeypatch, change):
    processes, _, _ = simulated_queue(args, monkeypatch)

    def change_during_run(seconds):
        if change == "source":
            monkeypatch.setattr(parallel, "source_gate", lambda: {"changed": "source"})
        elif change == "plan":
            plan = read_json(args.output / "queue_plan.json")
            atomic_json(args.output / "queue_plan.json", {**plan, "min_free_gib": 31})
        else:
            atomic_json(args.model_readys["llama31_8b"], {
                "repo": parallel.queue.MODEL_REPOS["llama31_8b"], "changed": True,
            })
    monkeypatch.setattr(parallel.time, "sleep", change_during_run)
    with pytest.raises(RuntimeError, match="changed"):
        parallel.execute(args)
    assert read_json(args.output / "queue_status.json")["status"] == "failed"
    assert all(p.done for p in processes)


def test_resume_complete_jobs_without_any_subprocess(args, monkeypatch):
    processes, attempts, _ = simulated_queue(args, monkeypatch)
    for job in parallel.queue.jobs(args.model_readys):
        directory = args.output / job["name"]
        directory.mkdir(parents=True)
        atomic_json(directory / "status.json", {"status": "complete"})
    validated = []
    monkeypatch.setattr(parallel.queue, "validate_job",
                        lambda directory, job: validated.append(job["name"]))
    parallel.execute(args)
    assert len(validated) == 12
    assert processes == []
    assert attempts == {}
    assert read_json(args.output / "queue_status.json")["status"] == "complete"


def test_exhausted_retry_cannot_restart_silently(args, monkeypatch):
    processes, _, _ = simulated_queue(args, monkeypatch)
    atomic_json(args.output / "queue_status.json", {
        "owner": None, "active": [], "oom_retries": {"group0/0": 3},
        "effective_max_workers": 1, "gpu_worker_limits": {"0": 1},
    })
    with pytest.raises(RuntimeError, match="retry limit exhausted"):
        parallel.execute(args)
    assert processes == []
    status = read_json(args.output / "queue_status.json")
    assert status["status"] == "failed"
    assert status["oom_retries"] == {"group0/0": 3}


def test_worker_lease_is_reused_and_passed_for_orphan_safety(args, monkeypatch):
    processes, _, _ = simulated_queue(args, monkeypatch)
    inherited = []
    original = parallel.subprocess.Popen

    def launch(*positional, **kwargs):
        gpu = int(kwargs["env"]["CUDA_VISIBLE_DEVICES"].split("-")[-1])
        lease_fd, queue_fd = kwargs["pass_fds"]
        inherited.append((gpu, os.fstat(lease_fd).st_ino, os.fstat(queue_fd).st_ino))
        return original(*positional, **kwargs)
    monkeypatch.setattr(parallel.subprocess, "Popen", launch)
    parallel.execute(args)
    assert len(inherited) == len(processes)
    assert len({record[2] for record in inherited}) == 1
    for gpu in args.gpus:
        assert len({record[1] for record in inherited if record[0] == gpu}) == 1


def test_sigterm_during_launch_registers_child_before_cleanup(args, monkeypatch):
    processes, _, _ = simulated_queue(args, monkeypatch)
    original = parallel.subprocess.Popen

    def launch(*positional, **kwargs):
        child = original(*positional, **kwargs)
        # Invoke the handler only; never signal any real process.
        parallel.signal.getsignal(parallel.signal.SIGTERM)(parallel.signal.SIGTERM, None)
        return child
    monkeypatch.setattr(parallel.subprocess, "Popen", launch)
    with pytest.raises(KeyboardInterrupt, match="Received signal"):
        parallel.execute(args)
    assert len(processes) == 1
    assert processes[0].done
    assert read_json(args.output / "queue_status.json")["status"] == "failed"
