import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import intentkv_rule_qa_queue as queue


def owner():
    return {"pid": 100, "start_ticks": 123, "uid": 1011, "cwd": str(queue.ROOT),
            "argv": ["/venv/bin/python", str(queue.QUEUE), "controller",
                     "--assets", "/assets", "--output", str(queue.RESULTS / "isolated")],
            "selector": ",".join(queue.UUIDS.values())}


def devices():
    return {gpu: {"uuid": uuid, "used_mib": 0, "free_mib": 64 * 1024,
                  "utilization": 0, "pids": []} for gpu, uuid in queue.UUIDS.items()}


def proc_entry(proc, pid, argv, visible="", ticks=123, state="S"):
    directory = proc / str(pid)
    directory.mkdir(parents=True)
    (directory / "stat").write_text(
        f"{pid} (python with spaces) " + " ".join([state] + ["0"] * 18 + [str(ticks)]))
    (directory / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
    (directory / "environ").write_bytes(f"CUDA_VISIBLE_DEVICES={visible}\0".encode())
    (directory / "cwd").symlink_to(queue.ROOT, target_is_directory=True)
    return directory


def fake_controller(proc):
    info = owner()
    proc_entry(proc, info["pid"], info["argv"], info["selector"])
    return queue.full_identity(info["pid"], proc)


def test_phase_barrier_uses_all_three_main_not_smoke():
    done = {f"smoke/{d}/r50" for d in queue.DATASETS}
    assert all(j["ratio"] == .5 and j["stage"] == "main"
               for j in queue.ready_jobs(done, []))
    done |= {"main/hotpotqa/r50", "main/2wiki/r50"}
    assert [j["key"] for j in queue.ready_jobs(done, [])] == ["main/musique/r50"]
    done.add("main/musique/r50")
    assert [j["key"] for j in queue.ready_jobs(done, [])] == [
        f"smoke/{d}/r20" for d in queue.DATASETS]


def test_smoke_dependencies_and_two_parallel_first_then_third():
    ready = queue.ready_jobs(set(), [])
    assert [j["dataset"] for j in ready[:2]] == ["hotpotqa", "2wiki"]
    assert all(j["stage"] == "smoke" for j in ready)
    active = ready[:2]
    assert [j["dataset"] for j in queue.ready_jobs(set(), active)] == ["musique"]
    done = {"smoke/hotpotqa/r50"}
    assert queue.ready_jobs(done, [active[1]])[0]["key"] == "main/hotpotqa/r50"
    assert len(queue.jobs()) == 12
    assert {j["samples"] for j in queue.jobs()} == {2, 500}


def test_continuous_reservation_requires_initial_exact_uuid_selector(tmp_path):
    proc = tmp_path / "proc"
    info = fake_controller(proc)
    assert queue.validate_controller(info, proc) == info
    for bad in ("", "0,1", queue.UUIDS[0], ",".join(reversed(queue.UUIDS.values()))):
        (proc / "100" / "environ").write_bytes(f"CUDA_VISIBLE_DEVICES={bad}\0".encode())
        actual = queue.full_identity(100, proc)
        with pytest.raises(RuntimeError):
            queue.validate_controller(actual, proc)


def test_reused_pid_and_zombie_are_not_live_identity(tmp_path):
    proc = tmp_path / "proc"
    info = fake_controller(proc)
    (proc / "100" / "stat").write_text(
        "100 (python) " + " ".join(["S"] + ["0"] * 18 + ["124"]))
    assert not queue.exact_alive(info, proc)
    (proc / "100" / "stat").write_text(
        "100 (python) " + " ".join(["Z"] + ["0"] * 18 + ["123"]))
    assert queue.full_identity(100, proc) is None


@pytest.mark.parametrize("field,value", [
    ("uid", 1000), ("cwd", "/foreign"), ("argv", ["/python", "/foreign.py"]),
    ("selector", "0,1"),
])
def test_exact_identity_never_exempts_changed_owner(monkeypatch, field, value):
    info = owner()
    changed = {**info, field: value}
    monkeypatch.setattr(queue, "identity_alive", lambda *a: True)
    monkeypatch.setattr(queue, "full_identity", lambda *a: changed)
    with pytest.raises(RuntimeError, match="changed"):
        queue.validate_controller(info)


def test_classification_precedes_environment_read(tmp_path):
    proc = tmp_path / "proc"
    info = fake_controller(proc)
    harmless = proc_entry(proc, 101, ["/usr/sbin/sshd"])
    (harmless / "environ").unlink()
    cuda = proc_entry(proc, 102, ["/python3", "/unknown.py", "--gpu", "0"],
                      queue.UUIDS[0])
    assert queue.pending_blockers(0, info, proc=proc)[0]["pid"] == 102
    (cuda / "environ").unlink()
    with pytest.raises(queue.Waiting, match="Unreadable"):
        queue.pending_blockers(0, info, proc=proc)


def test_arbitrary_own_cuda_pid_is_not_exempt_and_other_gpu_is_allowed(tmp_path):
    proc = tmp_path / "proc"
    info = fake_controller(proc)
    proc_entry(proc, 102, ["/python", "/gpu_load.py", "--gpu", "0"], queue.UUIDS[0])
    proc_entry(proc, 103, ["/python", "/gpu_load.py", "--gpu", "1"], queue.UUIDS[1])
    assert [b["pid"] for b in queue.pending_blockers(0, info, proc=proc)] == [102]
    assert [b["pid"] for b in queue.pending_blockers(1, info, proc=proc)] == [103]


@pytest.mark.parametrize("field,value", [
    ("pids", [999]), ("used_mib", 33), ("utilization", 1), ("free_mib", 24575),
])
def test_empty_gpu_checks_fail_closed(monkeypatch, field, value):
    snapshot = devices()
    snapshot[0][field] = value
    monkeypatch.setattr(queue.socket, "gethostname", lambda: "dev-NDH200-023")
    monkeypatch.setattr(queue.os, "getuid", lambda: 1011)
    monkeypatch.setattr(queue, "validate_controller", lambda *a: owner())
    monkeypatch.setattr(queue, "gpu_snapshot", lambda: snapshot)
    monkeypatch.setattr(queue, "pending_blockers", lambda *a: [])
    with pytest.raises(queue.Waiting):
        queue.admit(0, owner())


def test_cpu_owner_context_uuid_and_node_checks(monkeypatch):
    snapshot = devices()
    monkeypatch.setattr(queue.socket, "gethostname", lambda: "dev-NDH200-023")
    monkeypatch.setattr(queue.os, "getuid", lambda: 1011)
    monkeypatch.setattr(queue, "validate_controller", lambda *a: owner())
    monkeypatch.setattr(queue, "gpu_snapshot", lambda: snapshot)
    monkeypatch.setattr(queue, "pending_blockers", lambda *a: [])
    assert queue.admit(0, owner()) == snapshot[0]
    snapshot[1]["pids"] = [100]
    with pytest.raises(RuntimeError, match="CPU"):
        queue.admit(0, owner())
    snapshot[1]["pids"] = []
    snapshot[0]["uuid"] = "GPU-wrong"
    with pytest.raises(RuntimeError, match="UUID"):
        queue.admit(0, owner())
    monkeypatch.setattr(queue.socket, "gethostname", lambda: "wrong")
    with pytest.raises(RuntimeError, match="restricted"):
        queue.admit(0, owner())


def test_worker_bootstrap_command_and_private_output_root():
    args = SimpleNamespace(model_ready=Path("/model.json"), data_ready=Path("/data.json"),
                           output=queue.RESULTS / "new_intentkv")
    job = queue.jobs()[0]
    command = queue.worker_command(args, job, 0)
    assert command[command.index("--seed") + 1] == "233"
    assert command[command.index("--purpose") + 1] == "smoke"
    assert command[command.index("--gpu") + 1] == "0"
    assert "--execute" in command
    boot = queue.bootstrap_command(args, owner(), job, 0)
    assert boot[boot.index("--owner-start-ticks") + 1] == "123"
    assert boot[boot.index("--") + 1:] == command
    assert queue.check_output_root(args.output) == args.output
    for path in (queue.RESULTS, queue.ROOT / "results", queue.RESULTS / "../escape"):
        with pytest.raises(RuntimeError):
            queue.check_output_root(path)


def test_worker_identity_requires_pinned_command_selector_uid_and_cwd(monkeypatch):
    args = SimpleNamespace(model_ready=Path("/model.json"), data_ready=Path("/data.json"),
                           output=queue.RESULTS / "new_intentkv")
    command = queue.worker_command(args, queue.jobs()[0], 0)
    plan = {"output": str(args.output), "jobs": [{"commands": {"0": command}}]}
    monkeypatch.setattr(queue, "validate_controller", lambda *a: owner())
    info = {**owner(), "pid": 101, "argv": command, "selector": queue.UUIDS[0]}
    assert queue.validate_child(info, 0, owner(), plan) == info
    for change in ({"uid": 999}, {"cwd": "/foreign"}, {"selector": "0"},
                   {"argv": command + ["--unknown"]}):
        with pytest.raises(RuntimeError):
            queue.validate_child({**info, **change}, 0, owner(), plan)


def test_durable_json_replace(tmp_path):
    path = tmp_path / "checkpoint.json"
    queue.write(path, {"status": "running"})
    queue.write(path, {"status": "complete"})
    assert queue.read(path) == {"status": "complete"}
    assert not path.with_suffix(".json.writing").exists()


def completed_artifact(directory, stage="smoke", ratio=.5):
    count = 2 if stage == "smoke" else 500
    job = {"dataset": "hotpotqa", "stage": stage, "ratio": ratio, "samples": count}
    ids, indices = [f"q{i}" for i in range(count)], list(range(count))
    content = [(i, {"id": ids[i], "question": f"question{i}", "answer": f"answer{i}"})
               for i in indices]
    protocol = {
        "runtime": "intentkv_rule_frozen_qa_v1", "method": "intentkv_rule",
        "dataset": "hotpotqa", "ratio": ratio, "purpose": stage, "samples": count,
        "sample_start": 500 if stage == "smoke" else 0, "seed": 233, "max_steps": 7,
        "model": "Qwen/Qwen2.5-7B-Instruct", "model_revision": queue.MODEL_REVISION,
        "sample_ids": ids, "sample_indices": indices,
        "selected_content_sha256": queue.value_hash(content),
        "source_hashes": {str(queue.QUEUE.relative_to(queue.ROOT)): queue.digest(queue.QUEUE)},
        **queue.READY_HASHES, "model_ready": {"model": "pinned"}, "data_ready": {"data": "pinned"},
        "model_config_hashes": {"config.json": "pinned"},
        "weight_shard_inventory": {"model.safetensors": {"bytes": 1, "mtime_ns": 1}},
        "retrieval_index_inventory": {"meta.json": {"bytes": 1, "mtime_ns": 1}},
        "dataset_metadata": {"dataset": "pinned"}, "dataset_payload_sha256": {"data": "pinned"},
        "generation_config": {},
    }
    fingerprint = {**protocol, "protocol_inputs": protocol,
                   "protocol_inputs_sha256": queue.value_hash(protocol),
                   "assets": {**queue.READY_HASHES,
                              **{k: protocol[k] for k in ("model_config_hashes",
                                                        "weight_shard_inventory",
                                                        "retrieval_index_inventory")}},
                   "versions": {"torch": "2.10.0+cu128", "transformers": "5.0.0"},
                   "kv_config": {"cache_ratio": ratio}, "method_scope": "rule-only"}
    checksum = queue.value_hash(fingerprint)
    (directory / "questions").mkdir(parents=True)
    rows = []
    for index, sample in content:
        audit_path = directory / "questions" / f"{index}.json"
        queue.write(audit_path, {"id": sample["id"], "fingerprint_sha256": checksum,
                                 "intentkv_rule": {"method": "intentkv_rule"}})
        rows.append({"index": index, **sample, "gold_answer": sample["answer"],
                     "kv_audit": {"path": f"questions/{index}.json",
                                  "sha256": queue.digest(audit_path)}})
    queue.write(directory / "manifest.json",
                {"fingerprint": fingerprint, "fingerprint_sha256": checksum})
    queue.write(directory / "checkpoint.json",
                {"results": rows, "rows_sha256": queue.value_hash(rows),
                 "fingerprint_sha256": checksum})
    queue.write(directory / "results.json", {"results": rows, "fingerprint_sha256": checksum})
    queue.write(directory / "status.json", {"status": "complete", "completed_samples": count})
    return {**job, "protocol": protocol}, rows


def rewrite_rows(directory, rows):
    checkpoint = queue.read(directory / "checkpoint.json")
    checkpoint.update(results=rows, rows_sha256=queue.value_hash(rows))
    queue.write(directory / "checkpoint.json", checkpoint)
    final = queue.read(directory / "results.json")
    final["results"] = rows
    queue.write(directory / "results.json", final)


def test_independent_exact_formal_count_and_smoke_is_not_formal(monkeypatch, tmp_path):
    job, _ = completed_artifact(tmp_path / "main", stage="main")
    monkeypatch.setitem(queue.CANONICAL_ID_HASHES, "hotpotqa",
                        queue.ids_hash(job["protocol"]["sample_ids"]))
    assert queue.independent_validate(tmp_path / "main", job)["completed_questions"] == 500
    smoke, _ = completed_artifact(tmp_path / "smoke")
    assert queue.independent_validate(tmp_path / "smoke", smoke)["completed_questions"] == 2
    with pytest.raises(RuntimeError):
        queue.independent_validate(tmp_path / "smoke", {**smoke, "stage": "main", "samples": 500})


@pytest.mark.parametrize("mutation", ["duplicate", "wrong_count", "wrong_id", "wrong_index"])
def test_independent_duplicate_or_wrong_completed_rows_fail(tmp_path, mutation):
    job, rows = completed_artifact(tmp_path)
    if mutation == "duplicate":
        rows[1] = rows[0]
    elif mutation == "wrong_count":
        rows.pop()
    elif mutation == "wrong_id":
        rows[1]["id"] = "foreign"
    else:
        rows[1]["index"] = 99
    rewrite_rows(tmp_path, rows)
    with pytest.raises(RuntimeError):
        queue.independent_validate(tmp_path, job)


@pytest.mark.parametrize("key,value", [
    ("method", "tova"), ("seed", 3407), ("max_steps", 8), ("ratio", .2),
    ("model", "Qwen/Qwen2.5-32B-Instruct"), ("source_hashes", {"missing.py": "abc"}),
    ("model_revision", "wrong"), ("kv_config", {"cache_ratio": .2}),
])
def test_protocol_wrong_method_seed_budget_model_sources_fail(tmp_path, key, value):
    job, _ = completed_artifact(tmp_path)
    changed = {**job["protocol"], key: value}
    with pytest.raises((RuntimeError, OSError)):
        queue.check_protocol(changed, job)


def test_independent_per_question_method_and_content_binding(tmp_path):
    job, rows = completed_artifact(tmp_path)
    audit = tmp_path / rows[0]["kv_audit"]["path"]
    payload = queue.read(audit)
    payload["intentkv_rule"]["method"] = "tova"
    queue.write(audit, payload)
    rows[0]["kv_audit"]["sha256"] = queue.digest(audit)
    rewrite_rows(tmp_path, rows)
    with pytest.raises(RuntimeError, match="method"):
        queue.independent_validate(tmp_path, job)
    rows[0]["question"] = "wrong question"
    rewrite_rows(tmp_path, rows)
    with pytest.raises(RuntimeError, match="content"):
        queue.independent_validate(tmp_path, job)


def registration_setup(monkeypatch, tmp_path):
    args = SimpleNamespace(output_root=tmp_path, gpu=0)
    command = ["/venv/bin/python", str(queue.WORKER), "--gpu", "0"]
    plan = {"output": str(tmp_path), "jobs": [{"commands": {"0": command}, "protocol": {"pin": 1}}]}
    queue.write(tmp_path / "plan.json", plan)
    info = {**owner(), "pid": 101, "argv": command, "selector": queue.UUIDS[0]}
    record = {"owner": {"pid": 100, "start_ticks": 123}, "owner_full_identity": owner(),
              "child": {"pid": 101, "start_ticks": 123}, "child_command": command,
              "output": str(tmp_path), "plan_sha256": queue.digest(tmp_path / "plan.json")}
    monkeypatch.setattr(queue, "verify_pins", lambda *a: None)
    monkeypatch.setattr(queue, "maintenance_launch_lock", lambda *a: contextlib.nullcontext())
    monkeypatch.setattr(queue, "maintenance_path", lambda *a: tmp_path / "lease.json")
    monkeypatch.setattr(queue, "read_maintenance", lambda *a: record)
    monkeypatch.setattr(queue, "validate_controller", lambda *a: owner())
    monkeypatch.setattr(queue, "full_identity", lambda *a: info)
    monkeypatch.setattr(queue, "admit", lambda *a: devices()[0])
    monkeypatch.setattr(queue, "check_protocol", lambda *a: None)
    return args, info, record


def test_pre_cuda_worker_registration_writes_exact_identity(monkeypatch, tmp_path):
    args, info, _ = registration_setup(monkeypatch, tmp_path)
    assert queue.verify_registered(args, {"pin": 1}) == info
    assert queue.read(tmp_path / "lease.json")["child_full_identity"] == info


@pytest.mark.parametrize("change", [
    {"owner": {"pid": 100, "start_ticks": 124}},
    {"child": {"pid": 101, "start_ticks": 124}},
    {"child_command": ["/python", "/foreign.py"]},
    {"output": "/foreign"},
    {"plan_sha256": "tampered"},
])
def test_worker_registration_rejects_wrong_owner_child_command_root(monkeypatch, tmp_path, change):
    args, _, record = registration_setup(monkeypatch, tmp_path)
    record.update(change)
    with pytest.raises(RuntimeError):
        queue.verify_registered(args, {"pin": 1})


def test_recovery_waits_exact_orphan_and_never_releases_live_child(monkeypatch, tmp_path):
    args = SimpleNamespace(output=tmp_path, poll_seconds=1)
    command = ["/venv/bin/python", str(queue.WORKER), "--gpu", "0"]
    plan = {"output": str(tmp_path), "jobs": [{"commands": {"0": command}}]}
    info = {**owner(), "pid": 101, "argv": command, "selector": queue.UUIDS[0]}
    record = {"owner": {"pid": 100, "start_ticks": 123}, "owner_full_identity": owner(),
              "child": {"pid": 101, "start_ticks": 123}, "child_command": command,
              "output": str(tmp_path)}
    alive = [True]
    events = []
    monkeypatch.setattr(queue, "read_maintenance", lambda p: record if "gpu0" in str(p) else None)
    monkeypatch.setattr(queue, "identity_alive",
                        lambda token: token is record["child"] and alive[0])
    monkeypatch.setattr(queue, "full_identity", lambda *a: info)
    monkeypatch.setattr(queue.time, "sleep", lambda *a: alive.__setitem__(0, False))
    def release(*a):
        assert not alive[0], "A live orphan must retain the lease"
        events.append("released")
        return True
    monkeypatch.setattr(queue, "release_maintenance", release)
    queue.recover(args, plan, lambda state, **kw: events.append(state))
    assert events == ["recovering_owned_orphan", "released"]


def test_recovery_unknown_orphan_keeps_lease(monkeypatch, tmp_path):
    args = SimpleNamespace(output=tmp_path, poll_seconds=1)
    record = {"owner": {"pid": 100, "start_ticks": 123},
              "child": {"pid": 101, "start_ticks": 123}, "output": "/foreign"}
    monkeypatch.setattr(queue, "read_maintenance", lambda *a: record)
    monkeypatch.setattr(queue, "identity_alive", lambda token: token is record["child"])
    monkeypatch.setattr(queue, "release_maintenance", lambda *a: pytest.fail("Live orphan released"))
    with pytest.raises(queue.Waiting):
        queue.recover(args, {"jobs": []}, lambda *a, **kw: None)


def test_bootstrap_registers_before_exec_and_keeps_condition_lock(monkeypatch, tmp_path):
    args = SimpleNamespace(gpu=0, owner_pid=100, owner_start_ticks=123,
                           plan=tmp_path / "plan.json")
    control_args = SimpleNamespace(output=tmp_path, model_ready=Path("/model.json"),
                                  data_ready=Path("/data.json"))
    job = queue.jobs()[0]
    command = queue.worker_command(control_args, job, 0)
    plan = {"output": str(tmp_path), "jobs": [{**job, "commands": {"0": command}}]}
    queue.write(args.plan, plan)
    directory = queue.condition_dir(tmp_path, job)
    directory.mkdir()
    info = {**owner(), "pid": 101,
            "argv": queue.bootstrap_for_command(args.plan, owner(), 0, command),
            "selector": queue.UUIDS[0]}
    record = {"owner": {"pid": 100, "start_ticks": 123}, "owner_full_identity": owner(),
              "output": str(tmp_path), "child": None, "plan_sha256": queue.digest(args.plan)}
    events = []
    monkeypatch.setattr(queue, "verify_pins", lambda *a: None)
    monkeypatch.setattr(queue, "validate_controller", lambda *a: owner())
    monkeypatch.setattr(queue, "full_identity", lambda *a: info)
    monkeypatch.setattr(queue, "process_identity", lambda *a: {"pid": 101, "start_ticks": 123})
    monkeypatch.setattr(queue, "read_maintenance", lambda *a: record)
    monkeypatch.setattr(queue, "maintenance_launch_lock", lambda *a: contextlib.nullcontext())
    monkeypatch.setattr(queue, "maintenance_path", lambda *a: tmp_path / "lease.json")
    monkeypatch.setattr(queue, "admit", lambda *a: events.append("admit"))
    def register(gpu, token, cmd):
        assert token == record["owner"] and cmd == command
        record["child"] = {"pid": 101, "start_ticks": 123}
        events.append("register")
    monkeypatch.setattr(queue, "register_worker", register)
    def execute(program, cmd, env):
        assert record["child"] is not None
        assert (directory / ".worker.lock").exists()
        events.append("exec")
        raise RuntimeError("test exec sentinel")
    monkeypatch.setattr(queue.os, "execvpe", execute)
    with pytest.raises(RuntimeError, match="sentinel"):
        queue.bootstrap(args, command)
    assert events == ["admit", "register", "admit", "exec"]


def test_unreadable_genuine_cuda_candidate_identity_blocks(tmp_path):
    proc = tmp_path / "proc"
    info = fake_controller(proc)
    directory = proc_entry(proc, 102, ["/python", "/gpu_load.py", "--gpu", "0"],
                           queue.UUIDS[0])
    (directory / "stat").unlink()
    with pytest.raises(queue.Waiting, match="CUDA candidate"):
        queue.pending_blockers(0, info, proc=proc)


def test_cpu_environment_does_not_inherit_reservation(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(queue.UUIDS.values()))
    env = queue.cpu_environment()
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["HF_HUB_OFFLINE"] == env["HF_DATASETS_OFFLINE"] == "1"
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"


def test_failed_artifact_controller_has_no_gpu_launch_or_lease_claim(monkeypatch, tmp_path):
    args = SimpleNamespace(output=tmp_path, model_ready=Path("/model.json"),
                           data_ready=Path("/data.json"), poll_seconds=1,
                           worker_timeout_seconds=10)
    job = queue.jobs()[0]
    directory = queue.condition_dir(tmp_path, job)
    directory.mkdir()
    queue.write(directory / "status.json", {"status": "failed"})
    plan = {"output": str(tmp_path), "jobs": [job]}
    queue.write(tmp_path / "plan.json", plan)
    monkeypatch.setattr(queue, "validate_controller", lambda *a: owner())
    monkeypatch.setattr(queue, "full_identity", lambda *a: owner())
    monkeypatch.setattr(queue, "build_plan", lambda *a: plan)
    monkeypatch.setattr(queue, "recover", lambda *a: None)
    monkeypatch.setattr(queue, "claim_both", lambda *a: pytest.fail("Failed artifact claimed lease"))
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *a, **kw: pytest.fail("GPU launch attempted"))
    with pytest.raises(RuntimeError, match="Partial/failed"):
        queue.controller(args)
    assert queue.read(tmp_path / "queue.json")["status"] == "failed"
    assert len(list((tmp_path / "failures").glob("*.json"))) == 1


def test_cpu_controller_full_schedule_reserves_across_gaps_and_barrier(monkeypatch, tmp_path):
    args = SimpleNamespace(output=tmp_path, model_ready=Path("/model.json"),
                           data_ready=Path("/data.json"), poll_seconds=.01,
                           worker_timeout_seconds=100)
    plan = {"output": str(tmp_path), "jobs": [
        {**job, "protocol": {"key": job["key"]}} for job in queue.jobs()]}
    events, children, verified = [], [], set()
    lease = {"held": False}
    monkeypatch.setattr(queue, "validate_controller", lambda *a: owner())
    monkeypatch.setattr(queue, "full_identity", lambda *a: owner())
    monkeypatch.setattr(queue, "build_plan", lambda *a: plan)
    monkeypatch.setattr(queue, "recover", lambda *a: None)
    monkeypatch.setattr(queue, "verify_pins", lambda *a: None)
    monkeypatch.setattr(queue.time, "sleep", lambda *a: None)

    def claim(*a):
        lease["held"] = True
        events.append("claim_both")
    monkeypatch.setattr(queue, "claim_both", claim)

    def check_reservation(*a):
        assert lease["held"], "Both leases must survive every inter-worker gap"
        return devices()[0]
    monkeypatch.setattr(queue, "admit", check_reservation)

    def prepare(args, job):
        assert lease["held"]
        if job["ratio"] == .2:
            assert {f"main/{d}/r50" for d in queue.DATASETS} <= verified
        return {"key": job["key"]}
    monkeypatch.setattr(queue, "prepare_protocol", prepare)

    def validate(args, job):
        verified.add(job["key"])
        events.append(f"validated:{job['key']}")
        return {"completed_questions": job["samples"]}
    monkeypatch.setattr(queue, "validate_completed", validate)

    class Child:
        def __init__(self, command, **kwargs):
            self.pid = 200 + len(children)
            worker = command[command.index("--") + 1:]
            self.gpu = int(worker[worker.index("--gpu") + 1])
            self.dataset = worker[worker.index("--dataset") + 1]
            self.stage = worker[worker.index("--purpose") + 1]
            self.ratio = float(worker[worker.index("--ratio") + 1])
            self.key = f"{self.stage}/{self.dataset}/r{int(self.ratio*100)}"
            assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == queue.UUIDS[self.gpu]
            assert lease["held"]
            assert sum(not c.finished for c in children) < 2
            assert not any(c.gpu == self.gpu and not c.finished for c in children)
            if self.stage == "main":
                assert f"smoke/{self.dataset}/r{int(self.ratio*100)}" in verified
            if self.ratio == .2:
                assert {f"main/{d}/r50" for d in queue.DATASETS} <= verified
            self.remaining = 3 if self.key == "main/hotpotqa/r50" else 1
            self.finished = False
            events.append(f"started:{self.key}")
            children.append(self)

        def poll(self):
            if self.finished:
                return 0
            self.remaining -= 1
            if self.remaining > 0:
                return None
            self.finished = True
            return 0

    monkeypatch.setattr(queue.subprocess, "Popen", Child)
    def release(gpu, token):
        assert all(c.finished for c in children), "No lease release while a child is live"
        events.append(f"release:{gpu}")
        return True
    monkeypatch.setattr(queue, "release_maintenance", release)
    queue.controller(args)
    assert len(children) == 12
    assert len(verified) == 12
    first20 = next(i for i, event in enumerate(events) if event.startswith("started:") and event.endswith("r20"))
    assert all(events.index(f"validated:main/{d}/r50") < first20 for d in queue.DATASETS)
    assert events[:3] == ["claim_both", "started:smoke/hotpotqa/r50", "started:smoke/2wiki/r50"]
    assert events[-2:] == ["release:0", "release:1"]
    summary = queue.read(tmp_path / "SUMMARY.json")
    assert summary["status"] == "complete"
    assert summary["formal_questions"] == 3000
    assert summary["phase50_barrier"]


def test_shared_registration_owner_death_race_and_live_child_release(monkeypatch, tmp_path):
    from scripts import run_rebuttal_efficiency_queue as maintenance
    path = tmp_path / "lease.json"
    token = {"pid": 100, "start_ticks": 123}
    child = {"pid": 101, "start_ticks": 456}
    queue.write(path, {"version": 1, "owner": token, "child": None})
    monkeypatch.setattr(maintenance, "maintenance_path", lambda *a: path)
    monkeypatch.setattr(maintenance, "maintenance_launch_lock", lambda *a: contextlib.nullcontext())
    monkeypatch.setattr(maintenance, "identity_alive", lambda *a: False)
    with pytest.raises(RuntimeError, match="owner exited"):
        maintenance.register_worker(0, token, ["python", str(queue.WORKER)])
    assert queue.read(path)["child"] is None
    monkeypatch.setattr(maintenance, "identity_alive", lambda identity: identity in (token, child))
    monkeypatch.setattr(maintenance, "process_identity", lambda *a: child)
    assert maintenance.register_worker(0, token, ["python", str(queue.WORKER)]) == child
    assert not maintenance.release_maintenance(0, token)
    assert path.exists()
    with pytest.raises(RuntimeError, match="already registered"):
        maintenance.register_worker(0, token, ["python", str(queue.WORKER)])
    monkeypatch.setattr(maintenance, "identity_alive", lambda *a: False)
    assert not maintenance.release_maintenance(0, {"pid": 100, "start_ticks": 124})
    assert maintenance.release_maintenance(0, token)
    assert not path.exists()


def test_pre_cuda_worker_requires_fresh_fingerprint(monkeypatch, tmp_path):
    args, _, _ = registration_setup(monkeypatch, tmp_path)
    for protocol in (None, {"pin": 2}):
        with pytest.raises(RuntimeError, match="freshly prepared"):
            queue.verify_registered(args, protocol)


def test_prepare_and_validate_subprocesses_are_cpu_only_contracts(monkeypatch, tmp_path):
    args = SimpleNamespace(output=tmp_path, model_ready=Path("/model.json"),
                           data_ready=Path("/data.json"))
    job = queue.jobs()[0]
    calls = []
    def run(command, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert kwargs["env"]["TMPDIR"] == str(tmp_path / "runtime")
        assert "--execute" not in command
        calls.append(command)
        if "--prepare-protocol" in command:
            queue.write(Path(command[command.index("--prepare-protocol") + 1]), {"pin": 1})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(queue.subprocess, "run", run)
    monkeypatch.setattr(queue, "check_protocol", lambda *a: None)
    assert queue.prepare_protocol(args, job) == {"pin": 1}
    assert "--gpu" in calls[0] and calls[0][calls[0].index("--gpu") + 1] == "0"
    directory = queue.condition_dir(tmp_path, job)
    directory.mkdir()
    queue.write(tmp_path / "plan.json", {})
    monkeypatch.setattr(queue, "verify_pins", lambda *a: None)
    monkeypatch.setattr(queue, "independent_validate", lambda *a: {"completed_questions": 2})
    assert queue.validate_completed(args, job)["completed_questions"] == 2
    assert calls[1] == [queue.sys.executable, str(queue.WORKER), "--validate-output", str(directory)]


@pytest.mark.parametrize("change", ["asset", "formal_ids"])
def test_plan_rejects_cross_stage_asset_and_cross_ratio_sample_mixing(monkeypatch, tmp_path, change):
    args = SimpleNamespace(output=tmp_path, model_ready=Path("/model.json"),
                           data_ready=Path("/data.json"))
    def prepare(args, job):
        protocol = {key: "same" for key in (
            "model_ready", "data_ready", "model_ready_sha256", "data_ready_sha256",
            "model_config_hashes", "weight_shard_inventory", "retrieval_index_inventory",
            "dataset_metadata", "dataset_payload_sha256", "source_hashes",
            "model", "model_revision", "generation_config")}
        protocol.update({"sample_ids": ["a"], "sample_indices": [1],
                    "selected_content_sha256": "same"}
        )
        if change == "asset" and job["stage"] == "smoke" and job["ratio"] == .2:
            protocol["model_ready"] = "foreign"
        if change == "formal_ids" and job["stage"] == "main" and job["ratio"] == .2:
            protocol["sample_ids"] = ["foreign"]
        return protocol
    monkeypatch.setattr(queue, "prepare_protocol", prepare)
    monkeypatch.setattr(queue, "pins", lambda: {})
    with pytest.raises(RuntimeError, match="provenance|Unmatched"):
        queue.build_plan(args)


def test_canonical_reference_requires_unique_completed_500_and_exact_approved_ids(monkeypatch, tmp_path):
    rows = []
    for dataset in queue.DATASETS:
        ids = [f"{dataset}_q{i}" for i in range(500)]
        monkeypatch.setitem(queue.CANONICAL_ID_HASHES, dataset, queue.ids_hash(ids))
        rows.append({"dataset": dataset, "variant": "full", "purpose": "main", "seed": 233,
                     "samples": 500, "completed_samples": 500, "max_steps": 7, "sample_ids": ids})
    path = tmp_path / "canonical.json"
    queue.write(path, {"results": rows})
    canonical = queue.canonical_inputs(path)
    assert set(canonical["datasets"]) == set(queue.DATASETS)
    assert canonical["sha256"] == queue.digest(path)
    rows[0]["sample_ids"][0] = "foreign"
    queue.write(path, {"results": rows})
    with pytest.raises(RuntimeError, match="reference changed"):
        queue.canonical_inputs(path)
    queue.write(path, {"results": rows + [rows[1]]})
    with pytest.raises(RuntimeError):
        queue.canonical_inputs(path)


def test_formal_protocol_rejects_noncanonical_unique_ids_and_changed_readiness(tmp_path):
    job, _ = completed_artifact(tmp_path, stage="main")
    with pytest.raises(RuntimeError, match="canonical"):
        queue.check_protocol(job["protocol"], job)
    job["protocol"]["data_ready_sha256"] = "foreign"
    with pytest.raises(RuntimeError, match="readiness"):
        queue.check_protocol(job["protocol"], job)


@pytest.mark.parametrize("field,value,error", [
    ("method", "tova", "identity"), ("ratio", .2, "identity"), ("seed", 3407, "identity"),
    ("kv_config", {"cache_ratio": .2}, "budget"),
    ("assets", {}, "assets"), ("versions", {"torch": "2.9.0", "transformers": "5.0.0"}, "runtime"),
])
def test_independent_completed_reported_method_budget_assets_versions(monkeypatch, tmp_path, field, value, error):
    job, _ = completed_artifact(tmp_path)
    manifest = queue.read(tmp_path / "manifest.json")
    manifest["fingerprint"][field] = value
    queue.write(tmp_path / "manifest.json", manifest)
    with pytest.raises(RuntimeError, match=error):
        queue.independent_validate(tmp_path, job)


def test_full_cpu_fingerprint_transport_is_exactly_validated(tmp_path):
    job, _ = completed_artifact(tmp_path)
    fingerprint = queue.read(tmp_path / "manifest.json")["fingerprint"]
    full_job = {**job, "protocol": fingerprint}
    queue.check_protocol(fingerprint, full_job)
    assert queue.independent_validate(tmp_path, full_job)["completed_questions"] == 2
    manifest = queue.read(tmp_path / "manifest.json")
    manifest["fingerprint"]["budget"] = "foreign"
    queue.write(tmp_path / "manifest.json", manifest)
    with pytest.raises(RuntimeError, match="full fingerprint"):
        queue.independent_validate(tmp_path, full_job)


def test_fresh_hook_normalizes_leaf_and_full_cpu_protocol_transport(monkeypatch, tmp_path):
    args, info, _ = registration_setup(monkeypatch, tmp_path)
    full = {"protocol_inputs": {"pin": 1}, "method_metadata": "full"}
    assert queue.verify_registered(args, full) == info
    with pytest.raises(RuntimeError, match="freshly prepared"):
        queue.verify_registered(args, {"protocol_inputs": {"pin": 2}})


def device_bound_load(monkeypatch, tmp_path, gpu=2, pid=102):
    proc = tmp_path / "proc"
    controller = fake_controller(proc)
    command = ["/usr/bin/python3", "-B", str(queue.LOAD), "worker", "--gpu", str(gpu)]
    directory = proc_entry(proc, pid, command)
    (directory / "cwd").unlink()
    node = tmp_path / "node"
    node.mkdir()
    (directory / "cwd").symlink_to(node, target_is_directory=True)
    (directory / "environ").write_bytes(b"OMP_NUM_THREADS=4\0")
    info = queue.full_identity(pid, proc, allow_unset=True)
    snapshot = devices()
    snapshot[gpu] = {"uuid": f"GPU-other-{gpu}", "pids": [pid]}
    if gpu in queue.UUIDS:
        snapshot[gpu]["uuid"] = queue.UUIDS[gpu]
    evidence = {"loads": {str(gpu): info}, "hardware": {str(gpu): {"uuid": snapshot[gpu]["uuid"]}}}
    monkeypatch.setattr(queue.socket, "gethostname", lambda: "dev-NDH200-023")
    monkeypatch.setattr(queue.os, "getuid", lambda: 1011)
    monkeypatch.setattr(queue, "load_binding_evidence", lambda: evidence)
    return proc, controller, directory, info, snapshot, evidence


def test_proven_other_device_load_without_initial_cvd_is_device_scoped(monkeypatch, tmp_path):
    proc, controller, _, _, snapshot, evidence = device_bound_load(monkeypatch, tmp_path)
    assert queue.pending_blockers(0, controller, proc=proc, devices=snapshot,
                                  load_evidence=evidence) == []
    assert queue.pending_blockers(1, controller, proc=proc, devices=snapshot,
                                  load_evidence=evidence) == []
    assert queue.selector(102, proc, allow_unset=True) is None
    with pytest.raises(RuntimeError, match="Exactly one"):
        queue.selector(102, proc)


def test_proven_target_load_is_a_blocker_not_a_pid_exemption(monkeypatch, tmp_path):
    proc, controller, _, _, snapshot, evidence = device_bound_load(monkeypatch, tmp_path, gpu=0)
    blockers = queue.pending_blockers(0, controller, proc=proc, devices=snapshot,
                                      load_evidence=evidence)
    assert blockers[0]["pid"] == 102 and blockers[0]["resolved_device"] == 0
    assert "target device" in blockers[0]["reason"]


@pytest.mark.parametrize("mutation", [
    "new_pid", "reused_pid", "wrong_uid", "wrong_cwd", "wrong_argv", "wrong_selector",
    "no_context", "wrong_device", "multi_device", "wrong_uuid", "missing_evidence",
    "changed_evidence", "unreadable_environ", "stopped", "wrong_node",
])
def test_other_device_load_resolution_rejects_any_unproven_binding(monkeypatch, tmp_path, mutation):
    proc, controller, directory, info, snapshot, evidence = device_bound_load(monkeypatch, tmp_path)
    if mutation == "new_pid":
        evidence["loads"]["2"] = {**info, "pid": 103}
    elif mutation == "reused_pid":
        (directory / "stat").write_text(
            "102 (python) " + " ".join(["S"] + ["0"] * 18 + ["124"]))
    elif mutation == "wrong_uid":
        evidence["loads"]["2"] = {**info, "uid": 1000}
    elif mutation == "wrong_cwd":
        evidence["loads"]["2"] = {**info, "cwd": "/foreign"}
    elif mutation == "wrong_argv":
        evidence["loads"]["2"] = {**info, "argv": info["argv"] + ["--extra"]}
    elif mutation == "wrong_selector":
        (directory / "environ").write_bytes(b"CUDA_VISIBLE_DEVICES=2\0")
    elif mutation == "no_context":
        snapshot[2]["pids"] = []
    elif mutation == "wrong_device":
        snapshot[2]["pids"] = []
        snapshot[1]["pids"] = [102]
    elif mutation == "multi_device":
        snapshot[1]["pids"] = [102]
    elif mutation == "wrong_uuid":
        snapshot[2]["uuid"] = "GPU-changed"
    elif mutation == "missing_evidence":
        evidence = None
    elif mutation == "changed_evidence":
        monkeypatch.setattr(queue, "load_binding_evidence", lambda: {"changed": True})
    elif mutation == "unreadable_environ":
        (directory / "environ").unlink()
    elif mutation == "stopped":
        (directory / "stat").write_text(
            "102 (python) " + " ".join(["T"] + ["0"] * 18 + ["123"]))
    else:
        monkeypatch.setattr(queue.socket, "gethostname", lambda: "foreign-node")
    with pytest.raises(queue.Waiting):
        queue.pending_blockers(0, controller, proc=proc, devices=snapshot, load_evidence=evidence)


def test_unbound_generic_gpu_process_never_uses_original_load_device_proof(monkeypatch, tmp_path):
    proc, controller, directory, _, snapshot, evidence = device_bound_load(monkeypatch, tmp_path)
    (directory / "cmdline").write_bytes(b"\0".join((b"/python", b"/unknown.py", b"--gpu", b"2")) + b"\0")
    with pytest.raises(queue.Waiting):
        queue.pending_blockers(0, controller, proc=proc, devices=snapshot, load_evidence=evidence)


def test_original_load_identity_rechecked_after_compute_binding(monkeypatch, tmp_path):
    proc, controller, _, info, snapshot, evidence = device_bound_load(monkeypatch, tmp_path)
    original = queue.full_identity
    calls = [0]
    def inspect(pid, *args, **kwargs):
        if pid == 102:
            calls[0] += 1
            if calls[0] == 2:
                return {**info, "start_ticks": 124}
        return original(pid, *args, **kwargs)
    monkeypatch.setattr(queue, "full_identity", inspect)
    with pytest.raises(queue.Waiting):
        queue.pending_blockers(0, controller, proc=proc, devices=snapshot, load_evidence=evidence)


def original_load_record(monkeypatch, tmp_path):
    source = tmp_path / "gpu_load.py"
    source.write_text("original source\n")
    node = tmp_path / "node"
    node.mkdir()
    path = tmp_path / "handover.json"
    monkeypatch.setattr(queue, "LOAD", source)
    monkeypatch.setattr(queue, "LOAD_NODE", node)
    monkeypatch.setattr(queue, "LOAD_EVIDENCE", path)
    monkeypatch.setattr(queue, "LOAD_SHA256", queue.digest(source))
    loads = {str(gpu): {"pid": 200 + gpu, "start_ticks": 1000 + gpu, "uid": 1011,
                       "cwd": str(node), "argv": ["/usr/bin/python3", "-B", str(source),
                                                    "worker", "--gpu", str(gpu)],
                       "selector": None} for gpu in range(8)}
    hardware = {str(g): {"uuid": queue.UUIDS.get(g, f"GPU-other-{g}"), "name": "NVIDIA H200",
                         "pids": [200 + g]} for g in range(8)}
    record = {"version": 1, "status": "restored_after_queue_failure", "record_path": str(path),
              "load_sha256": queue.digest(source), "authorized_gpus": [0, 1],
              "release_authorized": True, "loads": loads, "hardware": hardware,
              "proof": {"pair": {k: loads[k] for k in ("0", "1")},
                        "untouched_six": {str(g): loads[str(g)] for g in range(2, 8)},
                        "hardware": hardware}}
    queue.write(path, record)
    return source, path, record


def test_original_load_evidence_pins_terminal_record_source_and_all_devices(monkeypatch, tmp_path):
    source, path, _ = original_load_record(monkeypatch, tmp_path)
    evidence = queue.load_binding_evidence()
    assert evidence["sha256"] == queue.digest(path)
    assert evidence["load_source_sha256"] == queue.digest(source)
    assert set(evidence["loads"]) == {str(g) for g in range(8)}


@pytest.mark.parametrize("mutation", [
    "source", "record_source", "status", "missing_device", "changed_preserved",
    "argv", "cwd", "uid", "selector", "context", "uuid", "node_authorization",
])
def test_original_load_evidence_rejects_changed_or_incomplete_provenance(monkeypatch, tmp_path, mutation):
    source, path, record = original_load_record(monkeypatch, tmp_path)
    if mutation == "source":
        source.write_text("changed source\n")
    elif mutation == "record_source":
        record["load_sha256"] = "wrong"
    elif mutation == "status":
        record["status"] = "queue_running"
    elif mutation == "missing_device":
        del record["proof"]["untouched_six"]["7"]
    elif mutation == "changed_preserved":
        record["proof"]["untouched_six"]["2"] = {**record["loads"]["2"], "start_ticks": 999}
    elif mutation in ("argv", "cwd", "uid", "selector"):
        record["proof"]["pair"]["0"] = {**record["loads"]["0"], mutation: {
            "argv": ["/python", "/impostor.py"], "cwd": "/foreign", "uid": 1000,
            "selector": "0"}[mutation]}
    elif mutation == "context":
        record["proof"]["hardware"]["2"]["pids"] = []
    elif mutation == "uuid":
        record["proof"]["hardware"]["0"]["uuid"] = "GPU-wrong"
    else:
        record["authorized_gpus"] = [0, 1, 2]
    queue.write(path, record)
    with pytest.raises(RuntimeError):
        queue.load_binding_evidence()


def test_plan_source_guard_rechecks_pinned_external_load_evidence(monkeypatch):
    evidence = {"sha256": "record", "load_source_sha256": "source"}
    plan = {"queue_source_hashes": {"queue.py": "pin"}, "load_binding_evidence": evidence}
    monkeypatch.setattr(queue, "pins", lambda: {"queue.py": "pin"})
    monkeypatch.setattr(queue, "load_binding_evidence", lambda: dict(evidence))
    queue.verify_pins(plan)
    monkeypatch.setattr(queue, "load_binding_evidence",
                        lambda: {**evidence, "load_source_sha256": "changed"})
    with pytest.raises(RuntimeError, match="load source/handover"):
        queue.verify_pins(plan)
