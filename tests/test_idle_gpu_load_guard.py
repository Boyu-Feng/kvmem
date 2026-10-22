import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_idle_gpu_load_guard as guard
from scripts.run_latency_experiment_queue import bootstrap, jobs, terminate_child
from scripts.run_rebuttal_efficiency_queue import identity_alive, process_identity
from run_rebuttal_legacy import atomic_json


@pytest.mark.parametrize("pids,util,reserved,assigned,expected", [
    ([], 0, False, False, True), ([], 1, False, False, False),
    ([123], 0, False, False, False), ([], 0, True, False, False),
    ([], 0, False, True, False),
])
def test_load_start_requires_genuinely_unassigned_idle_gpu(pids, util, reserved, assigned, expected):
    assert guard.can_start_load({"pids": pids, "utilization": util}, reserved, assigned) is expected


def test_live_child_retains_reservation_after_controller_exit(tmp_path, monkeypatch):
    marker = tmp_path / "reservation.json"
    atomic_json(marker, {"gpu": 7, "owner": {"pid": 1}, "child": {"pid": 2}})
    monkeypatch.setattr(guard, "identity_alive", lambda identity: identity == {"pid": 2})
    assert guard.reservation_active(marker)
    monkeypatch.setattr(guard, "identity_alive", lambda identity: False)
    assert not guard.reservation_active(marker)
    marker.unlink()
    assert not guard.reservation_active(marker)


def test_reservation_never_authorizes_stopping_other_cards(tmp_path):
    marker = tmp_path / "reservation.json"
    atomic_json(marker, {"gpu": 6, "owner": {"pid": 1}})
    with pytest.raises(RuntimeError, match="GPU 7"):
        guard.reservation_active(marker)


def test_queue_covers_all_fifteen_conditions_with_one_fullkv():
    plan = jobs()
    assert len(plan) == 15
    assert plan[0] == ("fullkv", .5, "fullkv/r100")
    assert {method for method, _, _ in plan} == {
        "fullkv", "stepkv", "h2o", "tova", "h2o_step", "tova_step", "flowkv_style", "lazyeviction",
    }
    assert len({name for _, _, name in plan}) == 15


def test_only_specific_own_child_is_terminated():
    events = []
    child = SimpleNamespace(poll=lambda: None, terminate=lambda: events.append("terminate"),
                            wait=lambda timeout: events.append(("wait", timeout)))
    terminate_child(child)
    assert events == ["terminate", ("wait", 20)]
    terminate_child(None)


def test_signal_rejects_pid_reuse_before_any_kill(monkeypatch):
    monkeypatch.setattr(guard, "identity_alive", lambda identity: False)
    monkeypatch.setattr(guard.os, "kill", lambda *args: pytest.fail("Must not signal a reused PID"))
    with pytest.raises(RuntimeError, match="identity changed"):
        guard.signal_owned({"pid": 100, "start_ticks": 1}, signal.SIGTERM)


@pytest.fixture
def load_processes(tmp_path, monkeypatch):
    proc, node, script = tmp_path / "proc", tmp_path / "node", tmp_path / "gpu_load.py"
    proc.mkdir()
    node.mkdir()

    def create(pid, ppid, argv):
        path = proc / str(pid)
        path.mkdir(exist_ok=True)
        fields = ["S", str(ppid)] + ["0"] * 18
        fields[19] = str(pid + 5)
        (path / "stat").write_text(f"{pid} (python3) " + " ".join(fields))
        (path / "cmdline").write_bytes(b"\0".join(value.encode() for value in argv) + b"\0")
        (path / "environ").write_bytes(b"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\0HOME=/private/test\0")
        (path / "cgroup").write_text("0::/cpu_test")
        (path / "cwd").symlink_to(node, target_is_directory=True)
        return {"pid": pid, "start_ticks": pid + 5}

    parent = create(900, 1, ["/usr/bin/python3", "-B", str(script), "supervisor"])
    for gpu in range(8):
        create(100 + gpu, 900, ["/usr/bin/python3", "-B", str(script), "worker", "--gpu", str(gpu)])
    samples = {gpu: {"uuid": guard.GPU7_UUID if gpu == 7 else f"GPU-{gpu}",
                     "pids": [100 + gpu], "utilization": 90, "free_mib": 60000}
               for gpu in range(8)}
    signals = []

    def send(identity, signum):
        signals.append((identity["pid"], signum))
        if signum == signal.SIGKILL:
            (proc / str(identity["pid"]) / "stat").unlink()

    monkeypatch.setattr(guard, "Path", lambda value: proc if str(value) == "/proc" else Path(value))
    monkeypatch.setattr(guard, "process_identity", lambda pid: process_identity(pid, proc))
    monkeypatch.setattr(guard, "identity_alive", lambda identity: identity_alive(identity, proc))
    monkeypatch.setattr(guard, "signal_owned", send)
    monkeypatch.setattr(guard, "gpu_samples", lambda: samples)
    return SimpleNamespace(proc=proc, node=node, script=script, parent=parent, samples=samples,
                           signals=signals, create=create, checkpoint=node / "idle-load-handover.json")


def test_handover_preserves_all_worker_identities_and_never_stores_environment(load_processes):
    data = load_processes
    workers, environment = guard.adopt_supervisor(data.parent, data.script, data.checkpoint)
    assert data.signals == [(900, signal.SIGSTOP), (900, signal.SIGKILL)]
    assert [workers[gpu]["identity"]["pid"] for gpu in range(8)] == list(range(100, 108))
    assert environment["CUDA_VISIBLE_DEVICES"] == "0,1,2,3,4,5,6,7"
    assert "/private/test" not in data.checkpoint.read_text()
    recovered, recovered_environment = guard.resume_workers(data.checkpoint, data.script)
    assert recovered == workers
    assert recovered_environment == environment


def test_handover_rolls_back_if_gpu7_has_an_experiment(load_processes):
    data = load_processes
    data.samples[7]["pids"].append(888)
    with pytest.raises(RuntimeError, match="no longer load-only"):
        guard.adopt_supervisor(data.parent, data.script, data.checkpoint)
    assert data.signals == [(900, signal.SIGSTOP), (900, signal.SIGCONT)]
    assert not data.checkpoint.exists()
    assert all(guard.process_identity(100 + gpu) for gpu in range(8))


def test_systemd_control_group_is_rejected_before_any_signal(load_processes):
    data = load_processes
    (data.proc / "900/cgroup").write_text("0::/system.slice/bruli-h200-load.service")
    with pytest.raises(RuntimeError, match="Refusing direct systemd"):
        guard.adopt_supervisor(data.parent, data.script, data.checkpoint)
    assert not data.signals


def test_tick_stops_only_gpu7_then_restores_only_when_truly_idle(load_processes, monkeypatch):
    data = load_processes
    workers, environment = guard.adopt_supervisor(data.parent, data.script, data.checkpoint)
    data.create(700, 1, ["cpu_controller"])
    marker = data.node / "reservation.json"
    atomic_json(marker, {"gpu": 7, "owner": guard.process_identity(700), "child": None})
    stopped, launched, children, states = [], [], {}, []
    assigned = {7}

    def stop(identity):
        stopped.append(identity["pid"])
        (data.proc / str(identity["pid"]) / "stat").unlink()
        data.samples[7].update(pids=[], utilization=0)

    def launch(command, **kwargs):
        launched.append((command, kwargs))
        data.create(808, 1, command)
        return SimpleNamespace(pid=808, poll=lambda: None)

    monkeypatch.setattr(guard, "stop_worker", stop)
    monkeypatch.setattr(guard, "experiment_bindings", lambda *args: assigned)
    monkeypatch.setattr(guard.subprocess, "Popen", launch)
    state = lambda status, **kwargs: states.append(status)
    guard.tick(workers, children, marker, data.script, data.node, environment, state)
    assert stopped == [107] and not launched
    marker.unlink()
    guard.tick(workers, children, marker, data.script, data.node, environment, state)
    assert not launched
    assigned.clear()
    data.samples[7]["utilization"] = 1
    guard.tick(workers, children, marker, data.script, data.node, environment, state)
    assert not launched
    data.samples[7]["utilization"] = 0
    guard.tick(workers, children, marker, data.script, data.node, environment, state)
    assert launched[0][0] == ["/usr/bin/python3", "-B", str(data.script), "worker", "--gpu", "7"]
    assert launched[0][1]["env"] == environment
    assert len(launched) == 1
    assert [workers[gpu]["identity"]["pid"] for gpu in range(7)] == list(range(100, 107))


def test_bootstrap_registers_child_before_exec(tmp_path, monkeypatch):
    from scripts import run_latency_experiment_queue as queue

    marker = tmp_path / "reservation.json"
    owner, child = {"pid": 1, "start_ticks": 5}, {"pid": 2, "start_ticks": 6}
    atomic_json(marker, {"gpu": 7, "owner": owner, "child": None})
    events = []
    monkeypatch.setattr(queue, "register_worker", lambda *args: events.append("registered"))
    monkeypatch.setattr(queue, "identity_alive", lambda identity: identity == owner)
    monkeypatch.setattr(queue, "process_identity", lambda pid: child)
    monkeypatch.setattr(queue.os, "execvpe", lambda *args: events.append("exec"))
    bootstrap(["--reservation", str(marker), "--owner-pid", "1", "--owner-start-ticks", "5",
               "--", "/python", "worker.py"])
    assert events == ["registered", "exec"]
    assert queue.read_json(marker)["child"] == child
