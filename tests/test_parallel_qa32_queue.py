import json
import os
import contextlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_parallel_qa32_queue as parallel


def devices():
    return {g: {"index": g, "uuid": f"GPU-{g}", "free_mib": 48 * 1024,
                "name": "NVIDIA H200", "allocations": {}} for g in range(8)}


def proc_entry(proc, pid, argv, visible=b"", environ=True):
    directory = proc / str(pid)
    directory.mkdir(parents=True)
    (directory / "cmdline").write_bytes(b"\0".join(os.fsencode(a) for a in argv) + b"\0")
    (directory / "stat").write_text(f"{pid} (python) " + " ".join(["S"] + ["0"] * 18 + ["123"]))
    if environ:
        (directory / "environ").write_bytes(b"CUDA_VISIBLE_DEVICES=" + visible + b"\0")
    return directory


def record(root, job, pid, pair=(3, 4)):
    return {
        "root": root, "directory": root / job["name"], "job": job, "gpus": list(pair),
        "worker": {"pid": pid, "start_ticks": 123}, "started": 0,
        "command": parallel.command(Path("/venv/bin/python"), Path("/assets"), root, job),
        "log_path": root / "worker.log",
    }


def plan_roots(root):
    return {root / d: {"jobs": parallel.queue.jobs([d]), "datasets": [d]} for d in parallel.DATASETS}


def test_exact_original_worker_command_and_environment(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    root, assets = Path("/results/qa32_hotpotqa_v1"), Path("/base/qa32_assets")
    job = parallel.queue.jobs(["hotpotqa"])[11]
    cmd = parallel.command(Path("/base/venv/bin/python"), assets, root, job)
    assert cmd == [
        "/base/venv/bin/python", "-u", str(parallel.queue.RUNNER),
        "--assets", str(assets), "--output", str(root / job["name"]),
        "--memory-limit-gib", "44", "--dataset", "hotpotqa", "--method", "fullkv",
        "--ratio", "0.5", "--purpose", "main", "--samples", "500", "--sample-start", "0",
        "--seed", "233", "--max-steps", "7",
    ]
    env = parallel.environment(assets, Path("/base/latency_caches/tmp"), (5, 6))
    assert env["CUDA_VISIBLE_DEVICES"] == "5,6"
    assert all(env[k] == "4" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"))
    assert env["HF_HUB_OFFLINE"] == env["HF_DATASETS_OFFLINE"] == "1"
    assert env["TOKENIZERS_PARALLELISM"] == "false"
    assert env["PYTHONUNBUFFERED"] == env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["TMPDIR"] == "/base/latency_caches/tmp"
    assert env["HF_HUB_CACHE"] == "/base/assets/hf_cache/hub"
    assert env["TORCHINDUCTOR_CACHE_DIR"] == "/base/caches/inductor"


def test_six_total_three_per_pair_and_two_device_startup(monkeypatch):
    snapshot, active = devices(), {}
    limits = {"3,4": 3, "5,6": 3}
    job = parallel.queue.jobs(["hotpotqa"])[11]
    monkeypatch.setattr(parallel, "loaded_status", lambda *args: True)
    for index in range(6):
        pair = (3, 4) if index < 3 else (5, 6)
        rec = record(Path("/results"), job, index + 100, pair)
        assert parallel.slots_available(pair, snapshot, active, limits)
        active[index] = rec
        for gpu in pair:
            snapshot[gpu]["allocations"][index + 100] = 32000
    assert not parallel.slots_available((3, 4), snapshot, active, limits)
    assert not parallel.slots_available((5, 6), snapshot, active, limits)
    active = {0: active[0]}
    assert parallel.slots_available((3, 4), snapshot, active, limits)
    del snapshot[4]["allocations"][100]
    assert not parallel.slots_available((3, 4), snapshot, active, limits)
    snapshot[4]["allocations"][100] = 0
    assert not parallel.slots_available((3, 4), snapshot, active, limits)
    snapshot[4]["allocations"][100] = 32000
    monkeypatch.setattr(parallel, "loaded_status", lambda *args: False)
    assert not parallel.slots_available((3, 4), snapshot, active, limits)
    monkeypatch.setattr(parallel, "loaded_status", lambda *args: True)
    snapshot[4]["free_mib"] -= 1
    assert not parallel.slots_available((3, 4), snapshot, active, limits)
    snapshot[4]["free_mib"] += 1
    limits["3,4"] = 1
    assert not parallel.slots_available((3, 4), snapshot, active, limits)


def test_loading_or_stale_status_does_not_release_startup_gate(tmp_path):
    path = tmp_path / "status.json"
    path.write_text('{"pid":100,"status":"loading_model"}')
    assert not parallel.loaded_status(tmp_path, 100)
    path.write_text('{"pid":100,"status":"running"}')
    assert parallel.loaded_status(tmp_path, 100)
    assert not parallel.loaded_status(tmp_path, 101)
    assert not parallel.loaded_status(tmp_path, 100, path.stat().st_mtime + 1)


def test_eight_global_three_per_pair_with_opportunistic_pair(monkeypatch):
    snapshot, active = devices(), {}
    limits = {parallel.pair_key(p): 3 for p in parallel.PAIRS}
    job = parallel.queue.jobs(["hotpotqa"])[11]
    monkeypatch.setattr(parallel, "loaded_status", lambda *args: True)
    pairs = [(3, 4)] * 3 + [(5, 6)] * 3 + [(0, 7)] * 2
    for index, pair in enumerate(pairs):
        assert parallel.slots_available(pair, snapshot, active, limits, 8)
        active[index] = record(Path("/results"), job, 100 + index, pair)
        for gpu in pair:
            snapshot[gpu]["allocations"][100 + index] = 32000
    assert not parallel.slots_available((0, 7), snapshot, active, limits, 8)
    del active[0]
    assert parallel.slots_available((0, 7), snapshot, active, limits, 8)
    snapshot[7]["free_mib"] -= 1
    assert not parallel.slots_available((0, 7), snapshot, active, limits, 8)
    snapshot[7]["free_mib"] += 1
    snapshot[7]["name"] = "different GPU"
    assert not parallel.slots_available((0, 7), snapshot, active, limits, 8)
    assert not parallel.slots_available((0, 7), devices(), active, limits, 6)


def test_specific_smoke_dependency_fair_musique_and_resume_pair(tmp_path):
    roots = plan_roots(tmp_path)
    pins = {(str(r), j["name"]): {"committed": 0, "pair": None}
            for r, p in roots.items() for j in p["jobs"]}
    done = {(str(r), j["name"]) for r, p in roots.items() for j in p["jobs"]
            if j["purpose"] == "smoke" and j["method"] == "tova" and j["ratio"] == .2}
    ready = parallel.ready_jobs(roots, done, {}, pins, {}, (3, 4))
    main = [j for _, j in ready if j["purpose"] == "main"]
    assert len(main) == 3
    assert all(j["method"] == "tova" and j["ratio"] == .2 for j in main)
    assert not any(j["purpose"] == "main" and j["method"] == "fullkv" for _, j in ready)
    active = {}
    for ds, pid in (("hotpotqa", 100), ("2wiki", 101)):
        r = tmp_path / ds
        job = next(j for j in roots[r]["jobs"] if j["purpose"] == "main" and j["method"] == "tova")
        active[(str(r), job["name"])] = record(r, job, pid)
    ready = parallel.ready_jobs(roots, done, active, pins, {}, (3, 4))
    assert ready[0][1]["dataset"] == "musique"
    pinned_key = (str(ready[0][0]), ready[0][1]["name"])
    pins[pinned_key]["pair"] = [5, 6]
    assert pinned_key not in {(str(r), j["name"]) for r, j in
                              parallel.ready_jobs(roots, done, active, pins, {}, (3, 4))}


def test_nonpython_unreadable_environment_skipped_and_pending_python_blocks(tmp_path, monkeypatch):
    proc = tmp_path / "proc"
    proc_entry(proc, 100, ["/usr/sbin/sshd"], environ=False)
    proc_entry(proc, 101, ["/usr/bin/python3", "/unknown.py"], visible=b"3,4")
    original = parallel.process_env
    def read_env(pid, path):
        if pid == 100:
            raise AssertionError("Non-Python environ must never be read")
        return original(pid, path)
    monkeypatch.setattr(parallel, "process_env", read_env)
    blockers = parallel.admission_blockers((3, 4), devices(), {}, Path("/python"), Path("/assets"), {}, proc)
    assert {b["pid"] for b in blockers} == {101}
    assert all(b["reason"] == "unknown/pending assignment" for b in blockers)
    monkeypatch.setattr(parallel, "process_env", lambda *args: (_ for _ in ()).throw(PermissionError()))
    assert parallel.admission_blockers((3, 4), devices(), {}, Path("/python"), Path("/assets"), {}, proc)


def test_unknown_registered_foreign_or_empty_cuda_python_still_blocks(tmp_path):
    proc = tmp_path / "proc"
    proc_entry(proc, 100, ["/usr/bin/python3", "/unknown.py"], visible=b"")
    snapshot = devices()
    snapshot[3]["allocations"][100] = 1
    snapshot[4]["allocations"][999] = 1
    blockers = parallel.admission_blockers((3, 4), snapshot, {}, Path("/python"), Path("/assets"), {}, proc)
    assert {b["pid"] for b in blockers} == {100, 999}
    assert all(b["reason"] == "unknown CUDA process" for b in blockers)


@pytest.mark.parametrize("visible", [b"", b"0,1,2,3,4,5,6,7"])
def test_assigned_guard_blocks_but_unassigned_cpu_guard_does_not(tmp_path, visible):
    proc = tmp_path / "proc"
    argv = ["/usr/bin/python3", "-u", str(parallel.ROOT / "scripts/run_deadline_gpu_load_guard.py"),
            "--execute", "--record", str(tmp_path / "guard_record.json")]
    proc_entry(proc, 100, argv, visible=visible)
    args = ((3, 4), devices(), {}, Path("/python"), Path("/assets"), {}, proc)
    if visible:
        assert parallel.admission_blockers(*args)
    else:
        assert parallel.admission_blockers(*args) == []
    args[1][3]["allocations"][100] = 1
    assert parallel.admission_blockers(*args)


def test_own_and_expected_workers_only_exact_live_commands(tmp_path):
    proc = tmp_path / "proc"
    root = tmp_path / "qa32"
    assets, python = Path("/assets"), Path("/venv/bin/python")
    job = parallel.queue.jobs(["hotpotqa"])[11]
    rec = record(root, job, 100)
    directory = proc_entry(proc, 100, rec["command"], visible=b"3,4")
    assert parallel.own_worker(100, {1: rec}, proc)
    assert not parallel.own_worker(100, {1: rec}, proc, gpu=5)
    rec["worker"]["start_ticks"] = 999
    assert not parallel.own_worker(100, {1: rec}, proc)
    rec["worker"]["start_ticks"] = 123
    snapshot = devices()
    for gpu in (3, 4):
        snapshot[gpu]["allocations"][100] = 32000
    output = root / job["name"]
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps({
        "hardware_attempts": [{"cuda_visible_devices": "3,4"}]}))
    (output / "status.json").write_text(json.dumps({"status": "running", "pid": 100}))
    roots = {root: {"jobs": [job]}}
    assert parallel.expected_worker(100, (3, 4), roots, python, assets, snapshot, proc)
    del snapshot[4]["allocations"][100]
    assert not parallel.expected_worker(100, (3, 4), roots, python, assets, snapshot, proc)
    directory.joinpath("cmdline").write_bytes(b"/venv/bin/python\0-u\0/other.py\0")
    assert not parallel.own_worker(100, {1: rec}, proc)


def test_owned_load_recognition_exact_gpu_and_start_identity(tmp_path):
    proc = tmp_path / "proc"
    argv = ["/usr/bin/python3", "-B", str(parallel.LOAD), "worker", "--gpu", "3"]
    proc_entry(proc, 100, argv, visible=b"3")
    assert parallel.own_load(100, 3, proc)
    assert not parallel.own_load(100, 4, proc)


@pytest.mark.parametrize("kind", ["qwen", "llama", "sensitivity"])
def test_exact_loaded_small_accuracy_peers_coexist_but_pending_or_unknown_block(tmp_path, monkeypatch, kind):
    proc = tmp_path / "proc"
    root, python = tmp_path / "peer", Path("/venv/bin/python")
    root.mkdir()
    script = parallel.ROOT / ("scripts/run_stepkv_sensitivity.py" if kind == "sensitivity"
                              else "scripts/run_component_ablation_rebuttal.py")
    model = "llama31_8b" if kind == "llama" else "qwen25_7b"
    model_ready, data_ready = tmp_path / "ready.json", tmp_path / "data_ready.json"
    monkeypatch.setattr(parallel, "PEER_SCOPES", {root: (script, model_ready, model)})
    monkeypatch.setattr(parallel, "PEER_DATA", data_ready)
    job = {"name": f"{model}/seed233/hotpotqa/main/full", "model": model,
           "dataset": "hotpotqa", "seed": 233, "variant": "full", "purpose": "main"}
    cmd = [
        str(python), "-u", str(script), "--execute", "--model-ready", str(model_ready),
        "--data-ready", str(data_ready), "--output", str(root / job["name"]),
        "--dataset", "hotpotqa", "--seed", "233", "--variant", "full", "--purpose", "main",
    ]
    if kind == "sensitivity":
        job.update(case="default", ratio=.2)
        cmd.extend(["--case", "default", "--ratio", "0.2"])
    (root / "queue_plan.json").write_text(json.dumps({
        "model_readys": {model: str(model_ready)}, "data_ready": str(data_ready), "jobs": [job]}))
    output = root / job["name"]
    output.mkdir(parents=True)
    status = output / "status.json"
    status.write_text('{"status":"running","pid":100}')
    directory = proc_entry(proc, 100, cmd, visible=b"0")
    snapshot = devices()
    snapshot[0]["allocations"][100] = 16000
    args = ((0, 7), snapshot, {}, python, Path("/assets"), {}, proc)
    assert parallel.known_small_peer(100, 0, snapshot, python, proc)
    assert not parallel.admission_blockers(*args)
    status.write_text('{"status":"loading_model","pid":100}')
    assert parallel.admission_blockers(*args)
    status.write_text('{"status":"running"}')
    assert not parallel.admission_blockers(*args)
    snapshot[0]["allocations"].clear()
    assert parallel.admission_blockers(*args)
    snapshot[0]["allocations"][100] = 16000
    directory.joinpath("cmdline").write_bytes(b"\0".join(os.fsencode(a) for a in cmd + ["--unknown"]) + b"\0")
    assert parallel.admission_blockers(*args)
    bad = [str(tmp_path / "wrong_ready.json") if a == str(model_ready) else a for a in cmd]
    directory.joinpath("cmdline").write_bytes(b"\0".join(os.fsencode(a) for a in bad) + b"\0")
    assert parallel.admission_blockers(*args)
    wrong_data = tmp_path / "wrong_data_ready.json"
    plan = parallel.runner.read_json(root / "queue_plan.json")
    plan["data_ready"] = str(wrong_data)
    (root / "queue_plan.json").write_text(json.dumps(plan))
    bad = [str(wrong_data) if a == str(data_ready) else a for a in cmd]
    directory.joinpath("cmdline").write_bytes(b"\0".join(os.fsencode(a) for a in bad) + b"\0")
    assert parallel.admission_blockers(*args)


def test_plans_and_source_gate_remain_byte_identical(tmp_path, monkeypatch):
    assets = tmp_path / "assets"
    gate = {"original_core_hashes": {}, "source_hashes": {}, "test": "frozen"}
    monkeypatch.setattr(parallel.queue, "source_gate", lambda: gate)
    for dataset in parallel.DATASETS:
        root = tmp_path / f"qa32_{dataset}_v1"
        root.mkdir()
        (root / "queue_plan.json").write_text(json.dumps({
            "protocol": "qa32_frozen_legacy_shared_h200_v1", "datasets": [dataset],
            "jobs": parallel.queue.jobs([dataset]), "source_gate": gate, "assets": str(assets),
            "memory_limit_gib_per_gpu": 44, "min_free_gib_per_gpu": 48, "gpus": [3, 4],
        }, indent=3) + "\n\n")
    roots, raw, actual_gate = parallel.load_plans(tmp_path, assets)
    assert len(roots) == 3 and actual_gate == gate
    parallel.assert_pins(raw, gate, {})
    first = next(iter(raw))
    assert (first / "queue_plan.json").read_bytes() == raw[first]
    (first / "queue_plan.json").write_bytes(raw[first] + b"\n")
    with pytest.raises(RuntimeError, match="bytes changed"):
        parallel.assert_pins(raw, gate, {})
    (first / "queue_plan.json").write_bytes(raw[first])
    monkeypatch.setattr(parallel.queue, "source_gate", lambda: {**gate, "test": "changed"})
    with pytest.raises(RuntimeError, match="source gate"):
        parallel.assert_pins(raw, gate, {})
    monkeypatch.setattr(parallel.queue, "source_gate", lambda: gate)
    ready = tmp_path / "model_ready.json"
    ready.write_text("{}")
    digest = parallel.runner.file_hash(ready)
    ready.write_text('{"changed":true}')
    with pytest.raises(RuntimeError, match="ready asset"):
        parallel.assert_pins(raw, gate, {ready: digest})


def test_checkpoint_prefix_is_read_only_and_corruption_fails_closed(tmp_path, monkeypatch):
    job = parallel.queue.jobs(["hotpotqa"])[11]
    fp = {**job, "memory_limit_gib": 44, "original_core_hashes": {},
          "source_hashes": {}, "versions": {}, "sample_ids": [f"id-{i}" for i in range(500)]}
    placement = {"example": "original-two-device"}
    manifest = {"schema_version": 1, "fingerprint": fp,
                "fingerprint_sha256": parallel.runner.digest(fp),
                "actual_placement": placement, "actual_placement_sha256": parallel.runner.digest(placement),
                "hardware_attempts": [{"cuda_visible_devices": "3,4"}]}
    rows = [{"id": "id-0"}, {"id": "id-1"}]
    state = {"schema_version": 1, "fingerprint_sha256": parallel.runner.digest(fp),
             "results": rows, "rows_sha256": parallel.runner.digest(rows)}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    path = tmp_path / "checkpoint.json"
    path.write_text(json.dumps(state))
    before = path.read_bytes()
    monkeypatch.setattr(parallel.runner, "_read_question_audit", lambda *args: {})
    pin = parallel.resume_pin(tmp_path, job, {"original_core_hashes": {}})
    assert pin["committed_ids"] == ["id-0", "id-1"] and pin["pair"] == [3, 4]
    assert pin["fingerprint"] == fp and path.read_bytes() == before
    rows[1]["id"] = "id-5"
    state["rows_sha256"] = parallel.runner.digest(rows)
    path.write_text(json.dumps(state))
    with pytest.raises(RuntimeError, match="nonprefix checkpoint"):
        parallel.resume_pin(tmp_path, job, {"original_core_hashes": {}})


def test_original_logical_manifest_allows_h200_pair_portability(tmp_path):
    manifest = {
        "fingerprint": {"placement_policy": {
            "device_map": "balanced", "dtype": "bfloat16",
            "max_memory": {"0": "40GiB", "1": "40GiB"}}},
        "actual_placement": {"hf_device_map": {"first": 0, "last": 1},
                             "layer_devices": ["cuda:0", "cuda:1"]},
        "hardware_attempts": [{"cuda_visible_devices": "3,4", "devices": {
            "0": {"name": "NVIDIA H200"}, "1": {"name": "NVIDIA H200"}}}],
    }
    assert parallel.portable_manifest(manifest)
    roots = plan_roots(tmp_path)
    pins = {(str(r), j["name"]): {"committed": 0, "pair": [3, 4], "portable_pair": True}
            for r, p in roots.items() for j in p["jobs"]}
    assert parallel.ready_jobs(roots, set(), {}, pins, {}, (0, 7))
    manifest["fingerprint"]["gpus"] = [3, 4]
    assert not parallel.portable_manifest(manifest)
    for pin in pins.values():
        pin["portable_pair"] = False
    assert not parallel.ready_jobs(roots, set(), {}, pins, {}, (0, 7))


def test_oom_retry_is_logged_bounded_and_never_changes_protocol(tmp_path):
    limits, retries = {"3,4": 3, "5,6": 3}, {}
    job = parallel.queue.jobs(["hotpotqa"])[11]
    key = str(tmp_path), job["name"]
    rec = record(tmp_path, job, 100)
    original = dict(job)
    error = "torch.OutOfMemoryError: CUDA out of memory."
    assert parallel.oom_failure(error)
    assert not parallel.oom_failure("permission denied: model shard")
    parallel.record_oom(tmp_path, key, rec, limits, retries, error)
    assert limits["3,4"] == 2
    parallel.record_oom(tmp_path, key, rec, limits, retries, error)
    assert limits["3,4"] == 1 and limits["5,6"] == 3
    with pytest.raises(RuntimeError, match="exhausted"):
        parallel.record_oom(tmp_path, key, rec, limits, retries, error)
    events = [json.loads(s) for s in (tmp_path / "oom_events.jsonl").read_text().splitlines()]
    assert [e["retry"] for e in events] == [1, 2, 3] and rec["job"] == original


def test_global_and_additive_root_status_shapes(tmp_path):
    root = tmp_path / "qa32"
    root.mkdir()
    (root / "queue_status.json").write_text(json.dumps({"devices_before_loading": "preserved"}))
    job = parallel.queue.jobs(["hotpotqa"])[11]
    active = {1: record(root, job, 100), 2: record(root, {**job, "name": "second"}, 101)}
    owner = {"pid": 999, "start_ticks": 123}
    parallel.publish(tmp_path, {root: {"jobs": [job]}}, active, set(), owner,
                     "running", {"3,4": 3, "5,6": 3})
    status = parallel.runner.read_json(tmp_path / "queue_status.json")
    original = parallel.runner.read_json(root / "queue_status.json")
    assert len(status["active"]) == len(original["active"]) == 2
    assert {"root", "job", "gpus", "worker"} <= status["active"][0].keys()
    assert original["job"] == job and original["worker_identity"] == active[1]["worker"]
    assert original["devices_before_loading"] == "preserved" and status["owner"] == owner


def test_graceful_stop_signals_only_exact_own_child(monkeypatch):
    job = parallel.queue.jobs(["hotpotqa"])[11]
    active = {i: {**record(Path("/root"), job, i), "process": SimpleNamespace(poll=lambda: 0)}
              for i in (100, 101, 102)}
    monkeypatch.setattr(parallel, "identity_alive", lambda worker: worker["pid"] != 102)
    monkeypatch.setattr(parallel, "process_args",
                        lambda pid: active[pid]["command"] if pid != 101 else ["foreign"])
    calls = []
    monkeypatch.setattr(parallel.os, "kill", lambda pid, sig: calls.append((pid, sig)))
    assert parallel.graceful_stop(active, 0) == []
    assert calls == [(100, parallel.signal.SIGTERM)]


@pytest.mark.parametrize("fatal", [None, "CUDA fatal: illegal memory access"])
@pytest.mark.parametrize("pair_count,max_workers", [(2, 6), (3, 8)])
def test_full_simulated_queue_runs_all_33_mains_with_caps_and_inherited_leases(tmp_path, monkeypatch, fatal,
                                                                            pair_count, max_workers):
    assets, results, output = tmp_path / "qa32_assets", tmp_path / "results", tmp_path / "scheduler"
    assets.mkdir()
    results.mkdir()
    for kind in ("model", "data"):
        (assets / f"{kind}_ready.json").write_text("{}")
    roots = {results / f"qa32_{ds}_v1": {"jobs": parallel.queue.jobs([ds]), "datasets": [ds]}
             for ds in parallel.DATASETS}
    raw = {}
    for root, plan in roots.items():
        root.mkdir()
        path = root / "queue_plan.json"
        path.write_text(json.dumps(plan, indent=2))
        raw[root] = path.read_bytes()
        (root / "queue_status.json").write_text("{}")
    python = tmp_path / "venv/bin/python"
    gate = {}
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(parallel, "load_plans", lambda *args: (roots, raw, gate))
    monkeypatch.setattr(parallel.queue, "source_gate", lambda: gate)
    monkeypatch.setattr(parallel.runner, "load_assets", lambda *args: (None, None, None, {}, None))
    monkeypatch.setattr(parallel.queue, "summarize", lambda *args: None)
    validated = []
    monkeypatch.setattr(parallel.queue, "validate_job", lambda root, job: validated.append(job))
    hot_root = results / "qa32_hotpotqa_v1"
    for job in roots[hot_root]["jobs"]:
        if job["method"] == "fullkv":
            directory = hot_root / job["name"]
            directory.mkdir(parents=True)
            (directory / "status.json").write_text('{"status":"complete"}')
    monkeypatch.setattr(parallel, "admission_blockers", lambda *args: [])
    monkeypatch.setattr(parallel, "loaded_status", lambda *args: True)
    monkeypatch.setattr(parallel, "process_identity",
                        lambda pid, *args: {"pid": pid, "start_ticks": 123} if isinstance(pid, int) else None)
    monkeypatch.setattr(parallel, "identity_alive", lambda *args: False)
    monkeypatch.setattr(parallel.signal, "signal", lambda *args: None)
    monkeypatch.setattr(parallel, "maintenance_launch_lock", lambda *args: contextlib.nullcontext())
    monkeypatch.setattr(parallel, "maintenance_blockers", lambda *args: {})
    monkeypatch.setattr(parallel.os, "kill",
                        lambda *args: (_ for _ in ()).throw(AssertionError("No real signaling")))
    monkeypatch.setattr(parallel.time, "sleep", lambda *args: None)
    workers, launched, observed = [], [], []
    class Worker:
        def __init__(self, cmd, **kwargs):
            self.pid, self.age, self.done = 10000 + len(workers), 0, False
            self.pair = tuple(map(int, kwargs["env"]["CUDA_VISIBLE_DEVICES"].split(",")))
            self.cmd = cmd
            assert kwargs["cwd"] == parallel.ROOT and len(kwargs["pass_fds"]) == 8 + 2 * (pair_count - 2)
            for fd in kwargs["pass_fds"]:
                os.fstat(fd)
            assert kwargs["env"]["OMP_NUM_THREADS"] == "4"
            assert cmd[cmd.index("--memory-limit-gib") + 1] == "44"
            if fatal:
                kwargs["stdout"].write(fatal + "\n")
                kwargs["stdout"].flush()
            workers.append(self)
            launched.append((self.pair, cmd[cmd.index("--dataset") + 1],
                             cmd[cmd.index("--purpose") + 1]))
        def poll(self):
            if self.done:
                return 0
            self.age += 1
            if self.age >= 10:
                self.done = True
                return 1 if fatal else 0
            return None
    monkeypatch.setattr(parallel.subprocess, "Popen", Worker)
    def snapshot():
        result = devices()
        live = [w for w in workers if not w.done]
        observed.append(live)
        assert len(live) <= max_workers
        for pair in parallel.PAIRS:
            assert sum(w.pair == pair for w in live) <= 3
        for worker in live:
            for gpu in worker.pair:
                result[gpu]["allocations"][worker.pid] = 32000
        return result
    monkeypatch.setattr(parallel, "gpu_snapshot", snapshot)
    args = SimpleNamespace(results=results, assets=assets, output=output, python=python,
                           tmp_root=tmp_path / "scratch", poll_seconds=.01,
                           pairs=parallel.PAIRS[:pair_count], max_workers=max_workers)
    if fatal:
        with pytest.raises(RuntimeError, match="illegal memory access"):
            parallel.execute(args)
        status = parallel.runner.read_json(output / "queue_status.json")
        assert status["status"] == "failed"
        assert not (output / "oom_events.jsonl").exists()
        assert "illegal memory access" in status["traceback"]
        return
    parallel.execute(args)
    assert len(workers) == 64 and len(validated) == 66
    assert sum(j["purpose"] == "main" for j in validated) == 33
    assert sum(j["samples"] for j in validated if j["purpose"] == "main") == 16500
    assert max(len(live) for live in observed) == max_workers
    assert {row[1] for row in launched[:3]} == set(parallel.DATASETS)
    for root, original in raw.items():
        assert (root / "queue_plan.json").read_bytes() == original
    assert parallel.runner.read_json(output / "queue_status.json")["completed_main"] == 33
