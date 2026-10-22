import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

SPEC = importlib.util.spec_from_file_location(
    "deadline_guard", Path(__file__).resolve().parents[1] / "scripts/run_deadline_gpu_load_guard.py")
guard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guard)


def summary(scope):
    return {"results": [{"name": name, "model": guard.SCOPES[scope][1],
                         "completed_samples": guard.SCOPES[scope][2]}
                        for name in sorted(guard.expected_names(scope))]}


def fixtures(root):
    for scope, (relative, _, _) in guard.SCOPES.items():
        directory = root / relative
        directory.mkdir(parents=True)
        guard.write(directory / "SUMMARY.json", summary(scope))


def work(complete=True, active=False):
    return {scope: {"complete": complete, "active": active, "error": None,
                    "workers": [], "controllers": []} for scope in guard.SCOPES}


def load_info(gpu=0):
    return {"uid": os.getuid(), "identity": {"pid": 99999990 + gpu, "start_ticks": 1},
            "argv": ["/usr/bin/python3", "-B", str(guard.LOAD), "worker", "--gpu", str(gpu)],
            "cwd": str(guard.NODE)}


def samples():
    return {str(g): {"uuid": f"uuid{g}", "name": "NVIDIA H200", "utilization": 0,
                     "pids": [], "free_mib": 140000} for g in range(8)}


def record(tmp_path):
    token = guard.identity(os.getpid())
    return {"version": 2, "status": "released", "owner": token, "rescue": token,
            "record_path": str(tmp_path / "status.json"),
            "failsafe": token, "idle_guard": {"identity": token},
            "deadline": time.time() - 1, "results": str(tmp_path),
            "guard_sha256": guard.digest(guard.SELF), "load_sha256": "load_hash",
            "idle_guard_sha256": "idle_hash", "poll_seconds": .05,
            "quiet_seconds": .1, "restore_timeout": .1, "release_authorized": True,
            "suspend_planned": time.time(), "idle_guard_suspended": True,
            "rescue_heartbeat": time.time(), "failsafe_heartbeat": time.time(),
            "gpu_uuids": {str(g): f"uuid{g}" for g in range(8)},
            "loads": {str(g): {"worker": load_info(g), "environment": {"PATH": "/usr/bin"}}
                      for g in range(8)}}


def isolate(monkeypatch, statuses):
    monkeypatch.setattr(guard, "work_status", lambda _: statuses)
    monkeypatch.setattr(guard, "validate_idle", lambda *args, **kwargs: {"identity": guard.identity(os.getpid())})
    monkeypatch.setattr(guard, "monitor_valid", lambda *args: True)
    monkeypatch.setattr(guard, "stopped", lambda _: True)
    monkeypatch.setattr(guard, "signal_exact", lambda *args: None)
    monkeypatch.setattr(guard, "restore_idle", lambda _: None)
    monkeypatch.setattr(guard, "verify_restored", lambda _: (_ for _ in ()).throw(RuntimeError("Not returned")))
    original = guard.digest
    monkeypatch.setattr(guard, "digest", lambda p: "load_hash" if p == guard.LOAD else original(p))


@pytest.mark.parametrize("scope,count", [("qwen", 42), ("llama", 42), ("sensitivity", 120),
                                       ("qa32_hotpotqa", 11), ("qa32_2wiki", 11), ("qa32_musique", 11)])
def test_expected_formal_conditions(scope, count):
    assert len(guard.expected_names(scope)) == count
    assert guard.scope_complete(scope, summary(scope)) == (True, count)
    partial = summary(scope)
    partial["results"].pop()
    assert guard.scope_complete(scope, partial) == (False, count - 1)


@pytest.mark.parametrize("change", ["count", "duplicate", "unexpected"])
def test_invalid_formal_summary(change):
    rows = summary("qwen")["results"]
    if change == "count":
        rows[0]["completed_samples"] = 100
    elif change == "duplicate":
        rows.append(dict(rows[0]))
    else:
        rows[0]["name"] = "unregistered/main"
    with pytest.raises(RuntimeError):
        guard.scope_complete("qwen", {"results": rows})


def test_stale_qwen_placeholder_and_delayed_worker(tmp_path, monkeypatch):
    fixtures(tmp_path)
    path = tmp_path / guard.SCOPES["qwen"][0] / "SUMMARY.json"
    rows = guard.read(path)
    rows["results"].append({"name": "stale", "model": "llama31_8b", "completed_samples": 0})
    guard.write(path, rows)
    live = {s: {"controllers": [{"pid": 1}], "workers": []} for s in guard.SCOPES}
    monkeypatch.setattr(guard, "processes", lambda _: live)
    assert guard.restore_decision(guard.work_status(tmp_path)) == "success"
    live["qwen"]["workers"].append({"pid": 2})
    assert guard.restore_decision(guard.work_status(tmp_path)) is None


def test_incomplete_live_controller_is_not_failure(tmp_path, monkeypatch):
    fixtures(tmp_path)
    path = tmp_path / guard.SCOPES["sensitivity"][0] / "SUMMARY.json"
    guard.write(path, {"results": []})
    live = {s: {"controllers": [], "workers": []} for s in guard.SCOPES}
    live["sensitivity"]["controllers"].append({"pid": 1})
    monkeypatch.setattr(guard, "processes", lambda _: live)
    assert guard.restore_decision(guard.work_status(tmp_path)) is None
    live["sensitivity"]["controllers"] = []
    assert guard.restore_decision(guard.work_status(tmp_path)) == "experiments_failed"


def test_new_parallel_qwen_controller_matches_only_original_scope(tmp_path, monkeypatch):
    results = tmp_path / "results"
    directory = tmp_path / "proc/100001"
    directory.mkdir(parents=True)
    root = results / guard.SCOPES["qwen"][0]
    argv = [sys.executable, "-u", str(guard.ROOT / "scripts/run_parallel_qwen_ablation.py"),
            "--output", str(root)]
    directory.joinpath("cmdline").write_bytes(b"\0".join(os.fsencode(arg) for arg in argv) + b"\0")
    token = {"pid": 100001, "start_ticks": 1}
    info = {"identity": token, "uid": os.getuid(), "argv": argv, "cwd": str(guard.ROOT)}
    monkeypatch.setattr(guard, "identity", lambda _: token)
    monkeypatch.setattr(guard, "inspect", lambda _: info)
    live = guard.processes(results, proc=tmp_path / "proc")
    assert live["qwen"]["controllers"] == [token]
    assert all(not live[s]["controllers"] for s in guard.SCOPES if s != "qwen")
    info["argv"][-1] = str(tmp_path / "wrong_result_root")
    assert not guard.processes(results, proc=tmp_path / "proc")["qwen"]["controllers"]


@pytest.mark.parametrize("field,value", [("uid", -1), ("argv", ["python", "other.py"]), ("cwd", "/foreign")])
def test_wrong_uid_argv_cwd(field, value):
    info = load_info()
    info[field] = value
    with pytest.raises((PermissionError, RuntimeError)):
        guard.validate_load(info, 0)


def test_exact_identity_required():
    info = load_info()
    guard.validate_load(info, 0)
    with pytest.raises(RuntimeError):
        guard.validate_load(info, 0, {**info, "identity": {"pid": 1, "start_ticks": 3}})


def test_deadline_only_warns_and_idle_guard_stays_suspended(tmp_path, monkeypatch):
    isolate(monkeypatch, work(active=True))
    path = tmp_path / "status.json"
    guard.write(path, record(tmp_path))
    assert not guard.tick(path, "owner")
    current = guard.read(path)
    assert current["deadline_expired_warning"] and current["idle_guard_suspended"]
    assert not current.get("restore_started")
    assert not any("keeper" in key for key in current)


def test_suspend_is_durable_before_signal_and_no_gpu_holder(tmp_path, monkeypatch):
    isolate(monkeypatch, work(active=True))
    path = tmp_path / "status.json"
    item = record(tmp_path)
    item.pop("suspend_planned")
    workers = {entry["worker"]["identity"]["pid"]: entry["worker"] for entry in item["loads"].values()}
    monkeypatch.setattr(guard, "alive", lambda token: bool(token))
    monkeypatch.setattr(guard, "inspect", lambda pid: workers[pid])
    guard.write(path, item)
    state = {"stopped": False}
    monkeypatch.setattr(guard, "stopped", lambda _: state["stopped"])
    def send(info, sig):
        assert sig == signal.SIGSTOP and guard.read(path)["suspend_planned"]
        state["stopped"] = True
    monkeypatch.setattr(guard, "signal_exact", send)
    guard.tick(path, "owner")
    current = guard.read(path)
    assert current["status"] == "awaiting_release" and current["release_ready"]
    assert current["idle_guard_suspended"] and current["release_planned"]


def test_partial_release_existing_load_race(tmp_path, monkeypatch):
    isolate(monkeypatch, work(active=True))
    path = tmp_path / "status.json"
    item = record(tmp_path)
    info = {**load_info(), "identity": guard.identity(os.getpid())}
    item["loads"]["0"]["worker"] = info
    monkeypatch.setattr(guard, "inspect", lambda _: info)
    guard.write(path, item)
    guard.tick(path, "owner")
    current = guard.read(path)
    assert current["status"] == "partial_release" and len(current["released_gpus"]) == 7
    assert current["original_loads_alive"]["0"]


def test_worker_resets_quiet_window(tmp_path, monkeypatch):
    statuses = work(active=True)
    isolate(monkeypatch, statuses)
    item = record(tmp_path)
    item["quiet_since"] = time.time() - 2
    path = tmp_path / "status.json"
    guard.write(path, item)
    guard.tick(path, "owner")
    assert "quiet_since" not in guard.read(path)
    for scope in statuses:
        statuses[scope]["active"] = False
    guard.tick(path, "owner")
    assert not guard.read(path).get("restore_started")


@pytest.mark.parametrize("role", ["owner", "rescue", "failsafe"])
def test_restore_conditional_and_monitor_takeover(tmp_path, monkeypatch, role):
    isolate(monkeypatch, work(complete=False))
    item = record(tmp_path)
    dead = {"pid": 99999998, "start_ticks": 1}
    if role != "owner":
        item["owner"] = dead
    if role == "failsafe":
        item["rescue"] = dead
    monkeypatch.setattr(guard, "spawn", lambda *args: guard.identity(os.getpid()))
    item["quiet_since"] = time.time() - 2
    restored = {}
    monkeypatch.setattr(guard, "restore_idle", lambda r: restored.update(same_guard=r["idle_guard"]))
    monkeypatch.setattr(guard, "verify_restored", lambda _: {"all8": "verified"})
    path = tmp_path / "status.json"
    guard.write(path, item)
    assert guard.tick(path, role)
    current = guard.read(path)
    assert current["status"] == "restored_after_experiment_failure"
    assert restored["same_guard"] == item["idle_guard"] and current["monitor_role"] == role


def test_source_tamper_is_loud(tmp_path, monkeypatch):
    isolate(monkeypatch, work(active=True))
    item = record(tmp_path)
    item["guard_sha256"] = "tampered"
    path = tmp_path / "status.json"
    guard.write(path, item)
    guard.tick(path, "owner")
    current = guard.read(path)
    assert "source changed" in " ".join(current["errors"])
    assert not current["release_ready"] and not current.get("restored_at")


def test_restore_failure_is_never_claimed_complete(tmp_path, monkeypatch):
    isolate(monkeypatch, work())
    item = record(tmp_path)
    item.update(restore_started=time.time() - 10, quiet_since=time.time() - 10)
    path = tmp_path / "status.json"
    guard.write(path, item)
    assert not guard.tick(path, "owner")
    current = guard.read(path)
    assert current["status"] == "restore_failed_retrying" and not current.get("restored_at")


def test_unregistered_monitor_refused(tmp_path):
    item = record(tmp_path)
    item["owner"] = {"pid": os.getpid(), "start_ticks": 0}
    path = tmp_path / "status.json"
    guard.write(path, item)
    with pytest.raises(RuntimeError, match="Unregistered"):
        guard.tick(path, "owner")


def test_fallback_uses_only_safe_env_skips_existing_workers(tmp_path, monkeypatch):
    load = tmp_path / "gpu_load.py"
    load.write_text("# fake fixture\n")
    monkeypatch.setattr(guard, "LOAD", load)
    monkeypatch.setattr(guard, "NODE", tmp_path)
    item = record(tmp_path)
    item["idle_guard"]["identity"] = {"pid": 99999998, "start_ticks": 1}
    item["load_sha256"] = guard.digest(load)
    monkeypatch.setattr(guard, "work_status", lambda _: work())
    monkeypatch.setattr(guard, "gpu_samples", samples)
    monkeypatch.setattr(guard, "assigned_gpus", lambda _: set())
    monkeypatch.setattr(guard, "find_loads", lambda _: {"0": load_info(0)})
    launched = []
    def launch(command, **kwargs):
        assert set(kwargs["env"]).issubset(guard.ENV_ALLOWLIST)
        assert "HF_TOKEN" not in kwargs["env"] and kwargs["cwd"] == tmp_path
        launched.append(command)
        return type("Child", (), {"pid": os.getpid()})()
    monkeypatch.setattr(guard.subprocess, "Popen", launch)
    guard.restore_idle(item)
    assert len(launched) == 7 and all(command[-1] != "0" for command in launched)
    assert item["idle_guard_dead_warning"]


@pytest.mark.parametrize("reason", ["active", "uuid", "busy", "assigned", "source", "foreign_environment"])
def test_fallback_rejects_unsafe_conditions(tmp_path, monkeypatch, reason):
    load = tmp_path / "gpu_load.py"
    load.write_text("# fake\n")
    monkeypatch.setattr(guard, "LOAD", load)
    monkeypatch.setattr(guard, "NODE", tmp_path)
    item = record(tmp_path)
    item["idle_guard"]["identity"] = {"pid": 99999998, "start_ticks": 1}
    item["load_sha256"] = "changed" if reason == "source" else guard.digest(load)
    monkeypatch.setattr(guard, "work_status", lambda _: work(active=reason == "active"))
    hardware = samples()
    if reason == "uuid":
        hardware["0"]["uuid"] = "wrong"
    if reason == "busy":
        hardware["0"]["pids"] = [123]
    if reason == "foreign_environment":
        item["loads"]["0"]["environment"]["SECRET_TOKEN"] = "not-real"
    monkeypatch.setattr(guard, "gpu_samples", lambda: hardware)
    monkeypatch.setattr(guard, "assigned_gpus", lambda _: {"0"} if reason == "assigned" else set())
    monkeypatch.setattr(guard, "find_loads", lambda _: {})
    monkeypatch.setattr(guard.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Must not launch"))
    with pytest.raises(RuntimeError):
        guard.restore_idle(item)


@pytest.mark.parametrize("failure", [None, "compute", "source", "stopped", "missing"])
def test_actual_restoration_verifier(tmp_path, monkeypatch, failure):
    item = record(tmp_path)
    monkeypatch.setattr(guard, "validate_idle", lambda *args, **kw: {})
    monkeypatch.setattr(guard, "digest", lambda p: item["guard_sha256"] if p == guard.SELF else
                        ("changed" if failure == "source" else item["load_sha256"]))
    found = {str(g): load_info(g) for g in range(8)}
    if failure == "missing":
        found.pop("7")
    hardware = samples()
    for gpu, info in found.items():
        hardware[gpu].update(pids=[] if failure == "compute" else [info["identity"]["pid"]], utilization=100)
    monkeypatch.setattr(guard, "gpu_samples", lambda: hardware)
    monkeypatch.setattr(guard, "find_loads", lambda _: found)
    monkeypatch.setattr(guard, "stopped", lambda _: failure == "stopped")
    if failure:
        with pytest.raises(RuntimeError):
            guard.verify_restored(item)
    else:
        assert len(guard.verify_restored(item)) == 8


def wait_for(predicate, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.02)
    raise AssertionError("Timed out waiting for own CPU test subprocess")


def test_real_owned_cpu_suspend_resume_and_safe_environment():
    child = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(30)"],
                             env=dict(os.environ, CUDA_VISIBLE_DEVICES="", TEST_SECRET_TOKEN="never-record"),
                             start_new_session=True)
    try:
        wait_for(lambda: "import time; time.sleep(30)" in guard.inspect(child.pid)["argv"])
        info = guard.inspect(child.pid)
        assert set(guard.safe_environment(child.pid)).issubset(guard.ENV_ALLOWLIST)
        assert "TEST_SECRET_TOKEN" not in guard.safe_environment(child.pid)
        guard.signal_exact(info, signal.SIGSTOP)
        wait_for(lambda: guard.stopped(child.pid))
        assert guard.alive(info["identity"])
        guard.signal_exact(info, signal.SIGCONT)
        wait_for(lambda: not guard.stopped(child.pid))
        maps = (Path("/proc") / str(child.pid) / "maps").read_text().lower()
        assert "libcuda" not in maps and "libtorch" not in maps
    finally:
        if child.poll() is None:
            guard.signal_exact(guard.inspect(child.pid), signal.SIGCONT)
            child.terminate()
            child.wait(timeout=8)


def test_real_owner_sigkill_rescue_resumes_own_suspended_cpu_guard(tmp_path):
    fixtures(tmp_path)
    path = tmp_path / "status.json"
    fake_source = tmp_path / "fake_load.py"
    fake_source.write_text("# no CUDA\n")
    idle = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(30)"],
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), start_new_session=True)
    owner = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(30)"],
                             start_new_session=True)
    wait_for(lambda: "import time; time.sleep(30)" in guard.inspect(idle.pid)["argv"])
    item = record(tmp_path)
    item.update(owner=guard.identity(owner.pid), idle_guard=guard.inspect(idle.pid),
                load_sha256=guard.digest(fake_source), quiet_since=time.time() - 2)
    guard.signal_exact(item["idle_guard"], signal.SIGSTOP)
    wait_for(lambda: guard.stopped(idle.pid))
    guard.write(path, item)
    code = (
        "import importlib.util,pathlib;"
        f"s=importlib.util.spec_from_file_location('g',{str(guard.SELF)!r});"
        "g=importlib.util.module_from_spec(s);s.loader.exec_module(g);"
        f"g.LOAD=pathlib.Path({str(fake_source)!r});"
        f"g.NODE=pathlib.Path({str(tmp_path)!r});"
        "g.validate_idle=lambda r,**kw:r['idle_guard'];"
        "g.gpu_samples=lambda:{};g.validate_hardware=lambda *a:None;"
        "g.verify_restored=lambda r:{'fixture':'verified'};"
        "g.monitor_valid=lambda *a:True;"
        f"g.monitor(pathlib.Path({str(path)!r}),'rescue')"
    )
    with guard.locked(path):
        rescue = subprocess.Popen([sys.executable, "-B", "-c", code],
                                  env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        item["rescue"] = guard.identity(rescue.pid)
        guard.write(path, item)
    try:
        wait_for(lambda: guard.read(path).get("rescue_heartbeat") is not None)
        assert guard.stopped(idle.pid)
        owner.kill()  # Only this test's own CPU sleeper, never production.
        owner.wait(timeout=8)
        assert rescue.wait(timeout=8) == 0
        assert not guard.stopped(idle.pid)
        current = guard.read(path)
        assert current["status"] == "restored" and current["monitor_role"] == "rescue"
        assert current["restoration_mode"] == "same_idle_guard_resumed"
    finally:
        if idle.poll() is None:
            guard.signal_exact(guard.inspect(idle.pid), signal.SIGCONT)
        for child in (owner, idle, rescue):
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=8)


def test_execute_rejects_cuda_visible_controller_before_any_action(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(RuntimeError, match="must not advertise"):
        guard.execute(tmp_path / "missing.json")


def test_real_cpu_rescue_inherits_resource_lock_without_gpu_reservation(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    resource = tmp_path / "resource.json"
    item = record(tmp_path)
    item["rescue"] = None
    item.pop("rescue_heartbeat", None)
    guard.write(path, item)
    token = None
    try:
        with guard.locked(resource) as lock:
            monkeypatch.setattr(guard, "RESOURCE_LOCK_FD", lock.fileno())
            token = guard.spawn("rescue", path)
            wait_for(lambda: guard.read(path).get("rescue_heartbeat") is not None)
            assert guard.monitor_valid(guard.read(path), "rescue")
            assert guard.gpu_selector(token["pid"]) == ""
            maps = (Path("/proc") / str(token["pid"]) / "maps").read_text().lower()
            assert "libcuda" not in maps and "libtorch" not in maps
        # The parent closed its descriptor, but the detached rescue still owns
        # the inherited open-file description. A different output cannot adopt.
        with pytest.raises(BlockingIOError):
            with guard.locked(resource, nonblocking=True):
                pass
    finally:
        if token and guard.alive(token):
            guard.signal_exact(guard.inspect(token["pid"]), signal.SIGTERM)
            os.waitpid(token["pid"], 0)
    with guard.locked(resource, nonblocking=True):
        pass
