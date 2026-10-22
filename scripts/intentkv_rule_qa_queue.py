"""Durable CPU-only IntentKV QA queue; never signals workers or load guards.

Missing initial GPU visibility remains fail-closed, except device resolution of
the exact original loads proven by the immutable completed pair handover.
Resolution requires pinned source/identity and a sole current compute UUID.
An original load on a target device remains a blocker, never a PID exemption.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_rebuttal_efficiency_queue import (
    claim_maintenance, identity_alive, maintenance_launch_lock, maintenance_path,
    process_identity, read_maintenance, register_worker, release_maintenance,
)
from scripts.parallel_alfworld_accuracy_resources import pending_cuda_candidate

QUEUE = Path(__file__).resolve()
WORKER = ROOT / "scripts/run_intentkv_rule.py"
BASE = Path("/mnt/checkpoints/bruli/fengboyu/kvmem_rebuttal_20260929")
RESULTS = BASE / "results"
DATASETS = ("hotpotqa", "2wiki", "musique")
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
READY_HASHES = {
    "model_ready_sha256": "e15344883f41e57bb8ff6eb6f0073d6ad240dfc5fe7e1bb77d6e5efc0cf32375",
    "data_ready_sha256": "4a1eb47f8431654bc04f10ed8207b36325b9a9f4e4134529a9495e6f700c9b96",
}
CANONICAL_REFERENCE = (
    RESULTS / "component_ablation_seeds233_3407_20260930_v1" / "SUMMARY.json")
CANONICAL_ID_HASHES = {
    "hotpotqa": "8b5f36e14fc56773d2895b1bc4424a5aeefb1b53f3025c111ca17d0546f5ff9a",
    "2wiki": "41a3883e270c139e5c455192bd695442f244900fcedd6e83b35dfae009abcc77",
    "musique": "e0bb35e46c8105aadeb47f0fa5786b43287c0ad0972b5122f49c75377561b3de",
}
UUIDS = {
    0: "GPU-85b09eef-ef2a-4202-8166-c81e0e4671a8",
    1: "GPU-0941ad2b-1a4b-1c62-22b6-284f5359de15",
}
LOAD = Path("/mnt/3fs/models/bruli/gpu-utilization-20260925/gpu_load.py")
LOAD_NODE = LOAD.parent / "nodes/dev-NDH200-023"
LOAD_SHA256 = "e25521a2cced799e6eb9825677cb6c08b1a8720d34ad6adce36e0ccc43975d22"
LOAD_EVIDENCE = RESULTS / "intentkv_gpu01_handover_20261005_v2/status.json"
HELPERS = (
    "scripts/run_rebuttal_efficiency_queue.py", "run_rebuttal_legacy.py",
    "scripts/parallel_alfworld_accuracy_resources.py",
    "scripts/reuse_semantic_resources.py", "scripts/run_deadline_gpu_load_guard.py",
    "scripts/run_alfworld_rebuttal.py", "scripts/run_idle_gpu_load_guard.py",
)


class Waiting(RuntimeError):
    """An admission conflict that may clear without any intervention."""


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    """Commit queue checkpoints, including the rename, to durable storage."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_suffix(path.suffix + ".writing")
    with staging.open("w") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    staging.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selector(pid, proc=Path("/proc"), *, allow_unset=False):
    values = [os.fsdecode(v.partition(b"=")[2])
              for v in (proc / str(pid) / "environ").read_bytes().split(b"\0")
              if v.startswith(b"CUDA_VISIBLE_DEVICES=")]
    if not values and allow_unset:
        return None
    if len(values) != 1:
        raise RuntimeError("Exactly one initial CUDA_VISIBLE_DEVICES selector is required")
    return values[0]


def full_identity(pid, proc=Path("/proc"), *, allow_unset=False):
    token = process_identity(pid, proc)
    if token is None:
        return None
    directory = proc / str(pid)
    try:
        result = {**token, "uid": directory.stat().st_uid,
                  "cwd": str((directory / "cwd").resolve(strict=True)),
                  "argv": [os.fsdecode(v) for v in
                           (directory / "cmdline").read_bytes().split(b"\0") if v],
                  "selector": selector(pid, proc, allow_unset=allow_unset)}
    except (FileNotFoundError, ProcessLookupError):
        if process_identity(pid, proc) is None:
            return None
        raise
    if process_identity(pid, proc) != token:
        raise RuntimeError("PID identity changed during inspection")
    return result


def entry(argv):
    if not argv or not Path(argv[0]).name.startswith("python"):
        return None
    index = 1
    while index < len(argv) and argv[index] in ("-B", "-u"):
        index += 1
    return argv[index] if index < len(argv) else None


def exact_alive(info, proc=Path("/proc")):
    if not isinstance(info, dict) or not identity_alive(info, proc):
        return False
    actual = full_identity(info["pid"], proc)
    if actual != info:
        raise RuntimeError("Live PID's UID/cwd/argv/start ticks/selector changed")
    return True


def validate_controller(info, proc=Path("/proc")):
    if (not exact_alive(info, proc) or info["uid"] != 1011 or info["cwd"] != str(ROOT)
            or entry(info["argv"]) != str(QUEUE)
            or info["argv"][info["argv"].index(str(QUEUE)) + 1:][:1] != ["controller"]
            or info["selector"] != ",".join(UUIDS.values())):
        raise RuntimeError("Not the exact continuously reserved CPU controller")
    return info


def gpu_snapshot():
    def query(flag):
        text = subprocess.check_output(
            ["nvidia-smi", flag, "--format=csv,noheader,nounits"],
            text=True, timeout=20)
        return list(csv.reader(text.splitlines()))
    devices = {}
    for row in query("--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu"):
        index, uuid, used, free, utilization = [v.strip() for v in row]
        if int(index) in devices:
            raise RuntimeError("Duplicate GPU accounting")
        devices[int(index)] = dict(uuid=uuid, used_mib=int(used), free_mib=int(free),
                                   utilization=int(utilization), pids=[])
    by_uuid = {d["uuid"]: d for d in devices.values()}
    for row in query("--query-compute-apps=gpu_uuid,pid"):
        uuid, pid = [v.strip() for v in row]
        if uuid not in by_uuid:
            raise RuntimeError("Unknown GPU compute context")
        by_uuid[uuid]["pids"].append(int(pid))
    return devices


def load_binding_evidence():
    """Pin the immutable completed handover, not a PID/UID allowlist."""
    if (LOAD_EVIDENCE.stat().st_uid != 1011 or LOAD.stat().st_uid != 1011
            or LOAD_NODE.stat().st_uid != 1011 or digest(LOAD) != LOAD_SHA256):
        raise RuntimeError("Original load evidence/source/storage ownership differs from pinned provenance")
    raw = LOAD_EVIDENCE.read_bytes()
    record = json.loads(raw)
    if (record.get("version") not in (1, 2) or record.get("status") not in
            ("restored", "restored_after_queue_failure")
            or record.get("record_path") != str(LOAD_EVIDENCE)
            or record.get("load_sha256") != LOAD_SHA256
            or record.get("authorized_gpus") != [0, 1] or not record.get("release_authorized")):
        raise RuntimeError("Load binding requires the exact completed original-profile handover")
    proof = record["proof"]
    loads = {**proof["pair"], **proof["untouched_six"]}
    if (set(loads) != {str(g) for g in range(8)}
            or set(proof["pair"]) != {"0", "1"}
            or set(proof["untouched_six"]) != {str(g) for g in range(2, 8)}):
        raise RuntimeError("Incomplete device-bound original load proof")
    hardware = {}
    for key, info in loads.items():
        expected = ["/usr/bin/python3", "-B", str(LOAD), "worker", "--gpu", key]
        if (info.get("uid") != 1011 or info.get("cwd") != str(LOAD_NODE)
                or info.get("argv") != expected or info.get("selector") is not None
                or type(info.get("pid")) is not int or info["pid"] <= 0
                or type(info.get("start_ticks")) is not int or info["start_ticks"] <= 0):
            raise RuntimeError(f"GPU{key}: invalid original load identity/source command")
        if int(key) >= 2 and info != record["loads"][key]:
            raise RuntimeError(f"GPU{key}: preserved original identity changed")
        device = proof["hardware"][key]
        uuid = device["uuid"]
        if (uuid != record["hardware"][key]["uuid"] or not uuid.startswith("GPU-")
                or "H200" not in device["name"] or device["pids"] != [info["pid"]]
                or (int(key) in UUIDS and uuid != UUIDS[int(key)])):
            raise RuntimeError(f"GPU{key}: original physical-device proof mismatch")
        hardware[key] = {"uuid": uuid}
    if (len({v["uuid"] for v in hardware.values()}) != 8
            or len({v["pid"] for v in loads.values()}) != 8):
        raise RuntimeError("Ambiguous original load/device proof")
    return {"path": str(LOAD_EVIDENCE), "sha256": hashlib.sha256(raw).hexdigest(),
            "load_source": str(LOAD), "load_source_sha256": LOAD_SHA256,
            "node": str(LOAD_NODE), "loads": loads, "hardware": hardware}


def admission_load_evidence(owner):
    argv = owner["argv"]
    if argv.count("--output") != 1:
        raise RuntimeError("Exact controller output required for device-binding evidence")
    path = Path(argv[argv.index("--output") + 1]) / "plan.json"
    try:
        plan = read(path)
    except FileNotFoundError:
        return None
    verify_pins(plan)
    return plan.get("load_binding_evidence")


def resolve_original_load(pid, argv, visible, devices, evidence, proc=Path("/proc")):
    """Resolve a proven process to its sole actual device; never exempt a load."""
    if evidence is None or devices is None:
        raise Waiting(f"Original load {pid} has no pinned handover/device evidence")
    if socket.gethostname() != "dev-NDH200-023" or os.getuid() != 1011:
        raise Waiting("Original load device resolution is restricted to the verified node/owner")
    if load_binding_evidence() != evidence:
        raise Waiting("Original load handover/source evidence changed")
    matches = [(int(key), info) for key, info in evidence["loads"].items() if info["pid"] == pid]
    if len(matches) != 1:
        raise Waiting(f"Original load {pid} is not an exact handover identity")
    bound, expected = matches[0]
    actual = full_identity(pid, proc, allow_unset=True)
    if (actual != expected or argv != expected["argv"] or visible != expected["selector"]):
        raise Waiting(f"Original load {pid} UID/cwd/argv/start ticks/selector changed")
    fields = (proc / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()
    if fields[0] in ("T", "t"):
        raise Waiting(f"Original load {pid} is stopped; device ownership is not active")
    locations = [index for index, device in devices.items() if pid in device["pids"]]
    if (locations != [bound] or devices[bound]["uuid"] != evidence["hardware"][str(bound)]["uuid"]):
        raise Waiting(f"Original load {pid} lacks its sole expected live compute UUID")
    if full_identity(pid, proc, allow_unset=True) != expected:
        raise Waiting(f"Original load {pid} changed during device resolution")
    return bound


def pending_blockers(gpu, owner, bootstrap_info=None, proc=Path("/proc"),
                     devices=None, load_evidence=None):
    """Classify before environ access; exemptions are exact, never UID-based."""
    validate_controller(owner, proc)
    if bootstrap_info is not None and not exact_alive(bootstrap_info, proc):
        raise RuntimeError("Bootstrap identity disappeared")
    blockers = []
    exempt = {owner["pid"]: owner}
    if bootstrap_info is not None:
        exempt[bootstrap_info["pid"]] = bootstrap_info
    for directory in proc.iterdir():
        if not directory.name.isdigit():
            continue
        pid = int(directory.name)
        if pid in exempt:
            if full_identity(pid, proc) != exempt[pid]:
                raise RuntimeError("Exact CPU exemption changed")
            continue
        try:
            argv = [os.fsdecode(v) for v in (directory / "cmdline").read_bytes().split(b"\0") if v]
            if not (pending_cuda_candidate(argv) or entry(argv) == str(QUEUE)):
                continue
            if process_identity(pid, proc) is None:
                try:
                    fields = (directory / "stat").read_text().rsplit(")", 1)[1].split()
                except FileNotFoundError:
                    if not directory.exists():
                        continue
                    raise Waiting(f"Unreadable CUDA candidate identity {pid}")
                if fields[0] in ("Z", "X", "x"):
                    continue
                raise Waiting(f"Cannot establish CUDA candidate start ticks {pid}")
            visible = selector(pid, proc, allow_unset=True)
            if entry(argv) == str(LOAD):
                bound = resolve_original_load(pid, argv, visible, devices, load_evidence, proc)
                if bound == gpu:
                    blockers.append({"pid": pid, "argv": argv, "resolved_device": bound,
                                     "reason": "verified original load on target device"})
                continue
            if visible is None:
                raise Waiting(f"Unbound genuine CUDA candidate {pid}")
        except (FileNotFoundError, ProcessLookupError):
            if process_identity(pid, proc) is None:
                continue
            raise Waiting(f"Unreadable live CUDA candidate {pid}")
        except Waiting:
            raise
        except (OSError, RuntimeError) as error:
            raise Waiting(f"Unreadable or unbound genuine CUDA candidate {pid}") from error
        if visible == "":
            continue
        if str(gpu) in visible.split(",") or UUIDS[gpu] in visible.split(","):
            blockers.append({"pid": pid, "argv": argv})
        elif any(not (v.isdigit() or v.startswith("GPU-")) for v in visible.split(",")):
            blockers.append({"pid": pid, "argv": argv, "reason": "unknown CUDA selector"})
    return blockers


def admit(gpu, owner, bootstrap_info=None):
    if socket.gethostname() != "dev-NDH200-023" or os.getuid() != 1011 or gpu not in UUIDS:
        raise RuntimeError("IntentKV is restricted to UID1011 on verified GPU0/1 node")
    validate_controller(owner)
    if bootstrap_info is not None:
        record = read_maintenance(maintenance_path(gpu))
        if record is None or record.get("owner_full_identity") != owner:
            raise RuntimeError("Pre-CUDA actor exemption lacks the exact pinned CPU owner")
        plan = read(Path(record["output"]) / "plan.json")
        verify_pins(plan)
        validate_child(bootstrap_info, gpu, owner, plan)
        if record.get("child") is not None and identity_alive(record["child"]):
            if record["child"] != {k: bootstrap_info[k] for k in ("pid", "start_ticks")}:
                raise RuntimeError("Another registered child already occupies this GPU")
    devices = gpu_snapshot()
    if any(owner["pid"] in d["pids"] for d in devices.values()):
        raise RuntimeError("CPU reservation controller opened a CUDA context")
    device = devices[gpu]
    if device["uuid"] != UUIDS[gpu]:
        raise RuntimeError("GPU UUID changed")
    if (device["pids"] or device["used_mib"] > 32 or device["utilization"] != 0
            or device["free_mib"] < 24 * 1024):
        raise Waiting(f"GPU{gpu} is not empty/exclusive: {device}")
    blockers = pending_blockers(gpu, owner, bootstrap_info, Path("/proc"), devices,
                                admission_load_evidence(owner))
    if blockers:
        raise Waiting(f"GPU{gpu} has assigned pre-CUDA workers: {blockers}")
    return device


def jobs():
    return [{"key": f"{stage}/{dataset}/r{int(ratio * 100)}", "stage": stage,
             "dataset": dataset, "ratio": ratio, "samples": 2 if stage == "smoke" else 500}
            for ratio in (.5, .2) for dataset in DATASETS for stage in ("smoke", "main")]


def ready_jobs(done, active):
    """All three *validated formal* .5 results are a strict phase barrier."""
    phase50 = {f"main/{dataset}/r50" for dataset in DATASETS}
    ratio = .2 if phase50 <= set(done) else .5
    busy = {j["dataset"] for j in active}
    return [j for j in jobs() if j["ratio"] == ratio and j["key"] not in done
            and j["key"] not in {a["key"] for a in active} and j["dataset"] not in busy
            and (j["stage"] == "smoke" or f"smoke/{j['dataset']}/r{int(ratio * 100)}" in done)]


def worker_command(args, job, gpu):
    return [sys.executable, str(WORKER), "--model-ready", str(args.model_ready),
            "--data-ready", str(args.data_ready), "--output-root", str(args.output),
            "--dataset", job["dataset"], "--ratio", str(job["ratio"]),
            "--purpose", job["stage"], "--samples", str(job["samples"]),
            "--sample-start", "500" if job["stage"] == "smoke" else "0",
            "--max-steps", "7", "--seed", "233", "--gpu", str(gpu), "--execute"]


def condition_dir(output, job):
    return Path(output) / (
        f"{job['stage']}_{job['dataset']}_ratio{int(job['ratio'] * 100)}_seed233")


def bootstrap_command(args, owner, job, gpu):
    return bootstrap_for_command(args.output / "plan.json", owner, gpu,
                                 worker_command(args, job, gpu))


def bootstrap_for_command(plan_path, owner, gpu, command):
    return [sys.executable, str(QUEUE), "bootstrap", "--gpu", str(gpu),
            "--owner-pid", str(owner["pid"]), "--owner-start-ticks", str(owner["start_ticks"]),
            "--plan", str(plan_path), "--", *command]


def pins():
    return {str(path.relative_to(ROOT)): digest(path)
            for path in [QUEUE, WORKER, *(ROOT / name for name in HELPERS)]}


def verify_pins(plan):
    if pins() != plan["queue_source_hashes"]:
        raise RuntimeError("Queue/worker/admission source pins changed; new output required")
    if "load_binding_evidence" in plan and load_binding_evidence() != plan["load_binding_evidence"]:
        raise RuntimeError("Pinned original load source/handover evidence changed")


def check_output_root(output):
    output = Path(output).resolve()
    if output == RESULTS or not output.is_relative_to(RESULTS):
        raise RuntimeError("A new private directory under the approved results root is required")
    return output


def validate_child(info, gpu, owner, plan, require_owner_alive=True):
    if require_owner_alive:
        validate_controller(owner)
    elif (owner["uid"] != 1011 or owner["cwd"] != str(ROOT)
          or entry(owner["argv"]) != str(QUEUE)
          or owner["selector"] != ",".join(UUIDS.values())):
        raise RuntimeError("Orphan lease has an invalid original CPU owner")
    if (info is None or info["uid"] != 1011 or info["cwd"] != str(ROOT)
            or info["selector"] != UUIDS[gpu] or entry(info["argv"]) not in (str(QUEUE), str(WORKER))):
        raise RuntimeError("Worker/bootstrap exact identity is invalid")
    command = info["argv"]
    valid = [c for j in plan["jobs"] for c in
             (j["commands"][str(gpu)],
              bootstrap_for_command(Path(plan["output"]) / "plan.json", owner, gpu,
                                    j["commands"][str(gpu)]))]
    if command not in valid:
        raise RuntimeError("Worker command is not pinned to this queue")
    return info


def verify_registered(args, protocol=None):
    """Called by the worker before model/CUDA imports, never optional at runtime."""
    plan_path = Path(args.output_root).resolve() / "plan.json"
    plan = read(plan_path)
    verify_pins(plan)
    gpu = args.gpu
    with maintenance_launch_lock(gpu):
        record = read_maintenance(maintenance_path(gpu))
        if (record is None or record.get("output") != str(plan_path.parent)
                or record.get("plan_sha256") != digest(plan_path)):
            raise RuntimeError("Worker has no matching queue lease")
        owner = record.get("owner_full_identity")
        if record["owner"] != {k: owner[k] for k in ("pid", "start_ticks")}:
            raise RuntimeError("Lease owner token/full identity mismatch")
        validate_controller(owner)
        info = validate_child(full_identity(os.getpid()), gpu, owner, plan)
        if record.get("child") != {k: info[k] for k in ("pid", "start_ticks")}:
            raise RuntimeError("Worker was not child-side registered")
        if record.get("child_command") != info["argv"] or entry(info["argv"]) != str(WORKER):
            raise RuntimeError("Registered command differs from worker exec")
        job = next(j for j in plan["jobs"] if j["commands"][str(gpu)] == info["argv"])
        check_protocol(job["protocol"], job)
        if (protocol is None or input_protocol(protocol) != input_protocol(job["protocol"])
                or ("protocol_inputs" in protocol and "protocol_inputs" in job["protocol"]
                    and protocol != job["protocol"])):
            raise RuntimeError("Worker must supply freshly prepared exact source/asset/sample fingerprint before CUDA")
        check_protocol(protocol, job)
        admit(gpu, owner, info)
        record["child_full_identity"] = info
        write(maintenance_path(gpu), record)
    return info


def bootstrap(args, command):
    plan = read(args.plan)
    verify_pins(plan)
    token = {"pid": args.owner_pid, "start_ticks": args.owner_start_ticks}
    record = read_maintenance(maintenance_path(args.gpu))
    if (record is None or record["owner"] != token or record["output"] != str(args.plan.parent)
            or record.get("plan_sha256") != digest(args.plan)):
        raise RuntimeError("Bootstrap does not belong to a pinned owner/lease")
    owner = record["owner_full_identity"]
    info = validate_child(full_identity(os.getpid()), args.gpu, owner, plan)
    pinned = next((j for j in plan["jobs"] if j["commands"][str(args.gpu)] == command), None)
    if pinned is None or info["argv"] != bootstrap_for_command(
            args.plan, owner, args.gpu, command):
        raise RuntimeError("Bootstrap invocation is not pinned")
    admit(args.gpu, owner, info)
    register_worker(args.gpu, token, command)
    with maintenance_launch_lock(args.gpu):
        admit(args.gpu, owner, info)
        record = read_maintenance(maintenance_path(args.gpu))
        if record["owner"] != token or record["child"] != process_identity(os.getpid()):
            raise RuntimeError("Lease changed during bootstrap")
        record["child_full_identity"] = info
        write(maintenance_path(args.gpu), record)
    output = condition_dir(plan["output"], pinned)
    lock = (output / ".worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.set_inheritable(lock.fileno(), True)
    os.execvpe(command[0], command, os.environ)


def value_hash(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def ids_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def input_protocol(protocol):
    return protocol.get("protocol_inputs", protocol)


def canonical_inputs(path=CANONICAL_REFERENCE):
    raw = Path(path).read_bytes()
    summary = json.loads(raw)
    selected = {}
    for dataset in DATASETS:
        rows = [r for r in summary["results"] if r.get("dataset") == dataset
                and r.get("variant") == "full" and r.get("purpose") == "main"
                and r.get("seed") == 233]
        if len(rows) != 1:
            raise RuntimeError(f"Canonical main reference is ambiguous/missing: {dataset}")
        row, = rows
        ids = row["sample_ids"]
        if (row.get("samples") != 500 or row.get("completed_samples") != 500
                or row.get("max_steps") != 7 or len(ids) != 500 or len(set(ids)) != 500
                or ids_hash(ids) != CANONICAL_ID_HASHES[dataset]):
            raise RuntimeError(f"Canonical seed233 matched-input reference changed: {dataset}")
        selected[dataset] = {"sample_ids": ids, "sample_ids_sha256": ids_hash(ids)}
    return {"path": str(Path(path)), "sha256": hashlib.sha256(raw).hexdigest(),
            "selection": {"variant": "full", "purpose": "main", "seed": 233},
            "datasets": selected}


def check_protocol(protocol, job):
    required = {"runtime": "intentkv_rule_frozen_qa_v1", "method": "intentkv_rule",
                "dataset": job["dataset"], "ratio": job["ratio"], "purpose": job["stage"],
                "samples": job["samples"], "sample_start": 500 if job["stage"] == "smoke" else 0,
                "seed": 233, "max_steps": 7, "model": "Qwen/Qwen2.5-7B-Instruct",
                "model_revision": MODEL_REVISION}
    if any(protocol.get(k) != v for k, v in required.items()):
        raise RuntimeError("Worker prepared an unapproved rule-only QA protocol")
    inputs = input_protocol(protocol)
    if inputs is not protocol:
        if protocol.get("protocol_inputs_sha256") != value_hash(inputs):
            raise RuntimeError("Prepared full fingerprint has an invalid CPU input binding")
        for key in (*required, "sample_ids", "sample_indices", "selected_content_sha256",
                    "source_hashes", "generation_config"):
            if protocol.get(key) != inputs.get(key):
                raise RuntimeError(f"Prepared fingerprint/CPU input identity differs: {key}")
    if "kv_config" in protocol and protocol["kv_config"].get("cache_ratio") != job["ratio"]:
        raise RuntimeError("Reported logical cache budget does not match the approved ratio")
    ids, indices = protocol["sample_ids"], protocol["sample_indices"]
    if (len(ids) != job["samples"] or len(set(ids)) != job["samples"]
            or len(indices) != job["samples"] or len(set(indices)) != job["samples"]
            or not protocol.get("source_hashes")
            or not protocol.get("model_revision") or not protocol.get("selected_content_sha256")):
        raise RuntimeError("Incomplete/duplicate sample or source/asset pins")
    if any(inputs.get(key) != expected for key, expected in READY_HASHES.items()):
        raise RuntimeError("Prepared model/data readiness assets differ from approved pins")
    if any(not inputs.get(key) for key in (
            "model_ready", "data_ready", "model_config_hashes", "weight_shard_inventory",
            "retrieval_index_inventory", "dataset_metadata", "dataset_payload_sha256")):
        raise RuntimeError("Incomplete CPU model/asset/data provenance inventories")
    if job["stage"] == "main" and ids_hash(ids) != CANONICAL_ID_HASHES[job["dataset"]]:
        raise RuntimeError("Formal input IDs do not match the approved canonical seed233 questions")
    for name, expected in protocol["source_hashes"].items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or digest(path) != expected:
            raise RuntimeError(f"Prepared source binding is not current: {name}")


def cpu_environment(output=None):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "PYTHONDONTWRITEBYTECODE": "1",
           "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
           "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4",
           "MKL_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4"}
    if output is not None:
        runtime = Path(output) / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        env["TMPDIR"] = str(runtime)
        env["XDG_CACHE_HOME"] = str(runtime / "xdg")
    return env


def run_cpu(command, args, log):
    callback = getattr(args, "_heartbeat", None)
    stop, errors = threading.Event(), []

    def pulse():
        while not stop.wait(getattr(args, "poll_seconds", 15)):
            try:
                callback("validating_cpu_provenance", cpu_command=command)
            except BaseException as error:
                errors.append(error)
                return

    thread = threading.Thread(target=pulse) if callback is not None else None
    if thread is not None:
        thread.start()
    try:
        return subprocess.run(command, cwd=ROOT, env=cpu_environment(args.output),
                              stdout=log, stderr=subprocess.STDOUT, timeout=900)
    finally:
        # Only this thread writes heartbeats while the main thread is blocked.
        stop.set()
        if thread is not None:
            thread.join()
        if errors:
            raise RuntimeError("Durable CPU validation heartbeat failed") from errors[0]


def prepare_protocol(args, job):
    path = args.output / "protocols" / f"{job['stage']}_{job['dataset']}_r{int(job['ratio']*100)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    command = worker_command(args, job, 0)
    command.remove("--execute")
    command += ["--prepare-protocol", str(path)]
    with (args.output / "preflight.log").open("a") as log:
        result = run_cpu(command, args, log)
    if result.returncode != 0:
        raise RuntimeError(f"CPU protocol preparation failed: {job['key']} (see preflight.log)")
    protocol = read(path)
    check_protocol(protocol, job)
    return protocol


def build_plan(args):
    protocols = []
    for job in jobs():
        protocols.append({**job, "protocol": prepare_protocol(args, job),
                          "commands": {str(gpu): worker_command(args, job, gpu) for gpu in UUIDS}})
    for dataset in DATASETS:
        same = [j for j in protocols if j["dataset"] == dataset]
        for key in ("model_ready", "data_ready", "model_ready_sha256", "data_ready_sha256",
                    "model_config_hashes", "weight_shard_inventory", "retrieval_index_inventory",
                    "dataset_metadata", "dataset_payload_sha256", "source_hashes",
                    "model", "model_revision", "generation_config"):
            if any(input_protocol(j["protocol"])[key] != input_protocol(same[0]["protocol"])[key]
                   for j in same):
                raise RuntimeError(f"Smoke/formal provenance differs: {dataset}/{key}")
        main = [j["protocol"] for j in same if j["stage"] == "main"]
        for key in ("sample_ids", "sample_indices", "selected_content_sha256"):
            if main[0][key] != main[1][key]:
                raise RuntimeError(f"Unmatched .5/.2 formal sample pins: {dataset}/{key}")
    return {"version": 1, "output": str(args.output), "model_ready": str(args.model_ready),
            "data_ready": str(args.data_ready), "gpus": UUIDS,
            "canonical_reference": canonical_inputs(),
            "queue_source_hashes": pins(), "jobs": protocols,
            "load_binding_evidence": load_binding_evidence(),
            "policy": "All three validated 500-question .5 mains before any .2 work; "
                      "2-question same-provenance smoke before every main; seed233 only. "
                      "CPU selector reservation and both maintenance leases persist across gaps."}


def independent_validate(directory, job):
    """Do not trust worker status or its validator's return code for the barrier."""
    directory = Path(directory)
    manifest, checkpoint, final, status = [
        read(directory / name) for name in
        ("manifest.json", "checkpoint.json", "results.json", "status.json")]
    expected = job["protocol"]
    check_protocol(expected, job)
    fingerprint = manifest["fingerprint"]
    inputs = input_protocol(expected)
    if inputs is not expected and fingerprint != expected:
        raise RuntimeError("Completed full fingerprint differs from independently prepared pin")
    if (fingerprint.get("protocol_inputs") != inputs
            or fingerprint.get("protocol_inputs_sha256") != value_hash(inputs)):
        raise RuntimeError("Completed output CPU protocol binding differs from pinned inputs")
    for key in ("runtime", "method", "dataset", "ratio", "purpose", "samples", "sample_start",
                "seed", "max_steps", "model", "model_revision", "sample_ids", "sample_indices",
                "selected_content_sha256", "source_hashes", "generation_config"):
        if fingerprint.get(key) != inputs[key]:
            raise RuntimeError(f"Reported completed method/input identity differs from CPU pin: {key}")
    if fingerprint.get("kv_config", {}).get("cache_ratio") != job["ratio"]:
        raise RuntimeError("Completed logical cache budget differs from the approved ratio")
    assets = fingerprint.get("assets", {})
    if any(assets.get(key) != inputs[key] for key in
           (*READY_HASHES, "model_config_hashes", "weight_shard_inventory",
            "retrieval_index_inventory")):
        raise RuntimeError("Completed model/data assets differ from independently prepared inputs")
    versions = fingerprint.get("versions", {})
    if (versions.get("torch", "").split("+")[0] != "2.10.0"
            or versions.get("transformers") != "5.0.0" or not fingerprint.get("method_scope")):
        raise RuntimeError("Completed runtime/method scope differs from approved rule-only QA")
    checksum = value_hash(fingerprint)
    rows = checkpoint["results"]
    if (any(doc.get("fingerprint_sha256") != checksum for doc in (manifest, checkpoint, final))
            or checkpoint.get("rows_sha256") != value_hash(rows)
            or final.get("results") != rows
            or status.get("status") != "complete"
            or status.get("completed_samples") != job["samples"]
            or len(rows) != job["samples"]):
        raise RuntimeError("Actual completed count/checkpoint binding is wrong")
    ids, indices = [r["id"] for r in rows], [r["index"] for r in rows]
    if (ids != expected["sample_ids"] or indices != expected["sample_indices"]
            or len(set(ids)) != job["samples"] or len(set(indices)) != job["samples"]):
        raise RuntimeError("Duplicate/wrong completed question IDs or sample indices")
    content = [(r["index"], {"id": r["id"], "question": r["question"],
                            "answer": r["gold_answer"]}) for r in rows]
    if value_hash(content) != expected["selected_content_sha256"]:
        raise RuntimeError("Completed question content differs from pinned input questions")
    for row in rows:
        reference = row["kv_audit"]
        path = (directory / reference["path"]).resolve()
        if not path.is_relative_to(directory.resolve() / "questions") or digest(path) != reference["sha256"]:
            raise RuntimeError("Question audit is missing/corrupt/outside the private condition")
        audit = read(path)
        if (audit.get("id") != row["id"] or audit.get("fingerprint_sha256") != checksum
                or audit.get("intentkv_rule", {}).get("method") != "intentkv_rule"):
            raise RuntimeError("Reported question method/identity is not pinned IntentKV rule-only")
    return {"completed_questions": len(rows), "fingerprint_sha256": checksum,
            "manifest_sha256": digest(directory / "manifest.json"),
            "checkpoint_sha256": digest(directory / "checkpoint.json"),
            "results_sha256": digest(directory / "results.json")}


def validate_completed(args, job):
    verify_pins(read(args.output / "plan.json"))
    directory = condition_dir(args.output, job)
    with (directory / "validation.log").open("a") as log:
        result = run_cpu(
            [sys.executable, str(WORKER), "--validate-output", str(directory)], args, log)
    if result.returncode != 0:
        raise RuntimeError(f"Worker CPU artifact validation failed: {directory}")
    return independent_validate(directory, job)


def recover(args, plan, heartbeat):
    """Wait for a pinned orphan; never signal it or clear a live-child lease."""
    for gpu in UUIDS:
        record = read_maintenance(maintenance_path(gpu))
        if record is None:
            continue
        if identity_alive(record.get("owner")):
            raise Waiting(f"GPU{gpu} already has a live maintenance owner")
        if identity_alive(record.get("child")):
            if (record.get("output") != str(args.output) or not record.get("owner_full_identity")
                    or record.get("child_command") not in
                    [j["commands"][str(gpu)] for j in plan["jobs"]]):
                raise Waiting(f"GPU{gpu} has an unknown live orphan; its lease remains protected")
            token = record["child"]
            while identity_alive(token):
                info = full_identity(token["pid"])
                validate_child(info, gpu, record["owner_full_identity"], plan,
                               require_owner_alive=False)
                heartbeat("recovering_owned_orphan", gpu=gpu, child=info)
                time.sleep(args.poll_seconds)
        if not release_maintenance(gpu, record["owner"]):
            raise Waiting(f"GPU{gpu} stale lease changed during recovery")


def claim_both(args, info, heartbeat, plan):
    claimed = []
    while True:
        try:
            recover(args, plan, heartbeat)
            for gpu in UUIDS:
                admit(gpu, info)
            for gpu in UUIDS:
                claim_maintenance(gpu, args.output, {k: info[k] for k in ("pid", "start_ticks")})
                claimed.append(gpu)
                with maintenance_launch_lock(gpu):
                    admit(gpu, info)
                    record = read_maintenance(maintenance_path(gpu))
                    record["owner_full_identity"] = info
                    record["plan_sha256"] = digest(args.output / "plan.json")
                    write(maintenance_path(gpu), record)
            return
        except Waiting as error:
            for gpu in claimed:
                release_maintenance(gpu, {k: info[k] for k in ("pid", "start_ticks")})
            claimed.clear()
            heartbeat("waiting_admission", reason=str(error))
            time.sleep(args.poll_seconds)
        except BaseException:
            for gpu in claimed:
                release_maintenance(gpu, {k: info[k] for k in ("pid", "start_ticks")})
            raise


def controller(args):
    info = validate_controller(full_identity(os.getpid()))
    token = {k: info[k] for k in ("pid", "start_ticks")}
    state_path = args.output / "queue.json"
    active, done = {}, {}
    claimed = False

    def heartbeat(state, **extra):
        write(state_path, {"version": 1, "status": state, "owner": info, "updated": time.time(),
                           "completed": done,
                           "active": {str(g): {k: v for k, v in rec.items() if k != "process"}
                                      for g, rec in active.items()},
                           "reservation": UUIDS, **extra})
        write(args.output / "SUMMARY.json", {
            "status": state, "updated": time.time(), "completed_conditions": len(done),
            "formal_completed": sum(k.startswith("main/") for k in done),
            "formal_questions": sum(v["completed_questions"] for k, v in done.items()
                                    if k.startswith("main/")),
            "phase50_barrier": all(f"main/{d}/r50" in done for d in DATASETS),
            "conditions": done})

    previous = read(state_path) if state_path.exists() else None
    if previous:
        write(args.output / "history" / f"queue_{time.time_ns()}.json", previous)
    try:
        args._heartbeat = heartbeat
        heartbeat("preparing_cpu_protocols")
        plan = build_plan(args)
        plan_path = args.output / "plan.json"
        if plan_path.exists():
            if read(plan_path) != json.loads(json.dumps(plan)):
                raise RuntimeError("Persisted plan differs from current inputs/sources; new output required")
        else:
            if any(condition_dir(args.output, j).exists() for j in jobs()):
                raise RuntimeError("Existing artifacts have no durable plan; refusing provenance mixing")
            write(plan_path, plan)
        recover(args, plan, heartbeat)
        if previous and previous.get("status") == "failed":
            raise RuntimeError("Persisted failure requires a new private output; no failed-artifact resume")
        for job in plan["jobs"]:
            directory = condition_dir(args.output, job)
            if not directory.exists():
                continue
            if not (directory / "status.json").exists() or read(directory / "status.json").get("status") != "complete":
                raise RuntimeError(f"Partial/failed artifact cannot be resumed: {directory}")
            done[job["key"]] = validate_completed(args, job)
        if any(k.endswith("r20") for k in done) and not all(f"main/{d}/r50" in done for d in DATASETS):
            raise RuntimeError("Persisted .2 artifact without a validated all-dataset .5 barrier")
        if any(j["stage"] == "main" and j["key"] in done and
               f"smoke/{j['dataset']}/r{int(j['ratio']*100)}" not in done for j in jobs()):
            raise RuntimeError("Persisted main lacks its same-provenance completed smoke")
        claim_both(args, info, heartbeat, plan)
        claimed = True
        while len(done) < len(plan["jobs"]):
            validate_controller(info)
            verify_pins(plan)
            for gpu, record in list(active.items()):
                process = record["process"]
                code = process.poll()
                if code is None:
                    if time.time() - record["started"] > args.worker_timeout_seconds:
                        raise RuntimeError("Worker deadline exceeded; exact live-child lease is retained")
                    continue
                if code != 0:
                    raise RuntimeError(f"Worker failed ({code}): {record['job']['key']}")
                done[record["job"]["key"]] = validate_completed(args, record["job"])
                del active[gpu]
                heartbeat("condition_complete", just_completed=record["job"]["key"])
            candidates = ready_jobs(done, [r["job"] for r in active.values()])
            for gpu in UUIDS:
                if gpu in active or not candidates:
                    continue
                job = candidates.pop(0)
                pinned = next(j for j in plan["jobs"] if j["key"] == job["key"])
                if prepare_protocol(args, job) != pinned["protocol"]:
                    raise RuntimeError("Source/asset/data/sample pins changed before worker admission")
                # Recheck .5 artifacts immediately before unlocking the .2 phase.
                if job["ratio"] == .2:
                    for prerequisite in plan["jobs"]:
                        if prerequisite["ratio"] == .5 and prerequisite["stage"] == "main":
                            done[prerequisite["key"]] = validate_completed(args, prerequisite)
                    write(args.output / "phase50_barrier.json",
                          {"updated": time.time(), "conditions": {
                              k: v for k, v in done.items() if k.startswith("main/") and k.endswith("r50")}})
                try:
                    admit(gpu, info)
                except Waiting as error:
                    heartbeat("waiting_admission", reason=str(error))
                    continue
                directory = condition_dir(args.output, job)
                if directory.exists():
                    raise RuntimeError(f"Never mix existing/failed condition artifacts: {directory}")
                directory.mkdir()
                command = bootstrap_command(args, info, job, gpu)
                write(directory / "launch.json",
                      {"owner": info, "gpu": gpu, "uuid": UUIDS[gpu], "command": command,
                       "protocol_sha256": value_hash(pinned["protocol"]), "started": time.time()})
                env = {**cpu_environment(args.output), "CUDA_VISIBLE_DEVICES": UUIDS[gpu]}
                with (directory / "worker.log").open("a") as log:
                    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                               stderr=subprocess.STDOUT, close_fds=True)
                active[gpu] = {"process": process, "pid": process.pid, "job": pinned,
                               "identity": process_identity(process.pid),
                               "started": time.time(), "command": command, "output": str(directory)}
                heartbeat("running")
            heartbeat("running" if active else "waiting_admission")
            time.sleep(args.poll_seconds)
        heartbeat("complete")
    except BaseException:
        failure = {"updated": time.time(), "traceback": traceback.format_exc(),
                   "protected_children": {str(g): full_identity(rec["process"].pid)
                                          for g, rec in active.items()
                                          if rec["process"].poll() is None}}
        write(args.output / "failures" / f"failure_{time.time_ns()}.json", failure)
        heartbeat("failed", **failure)
        raise
    finally:
        if claimed:
            for gpu in UUIDS:
                # The helper refuses release while its registered orphan remains live.
                release_maintenance(gpu, token)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    for mode in ("controller", "preflight"):
        control = modes.add_parser(mode)
        control.add_argument("--model-ready", type=Path, required=True)
        control.add_argument("--data-ready", type=Path, required=True)
        control.add_argument("--output", type=Path, required=True)
        control.add_argument("--poll-seconds", type=float, default=15)
        control.add_argument("--worker-timeout-seconds", type=float, default=24 * 3600)
    boot = modes.add_parser("bootstrap")
    boot.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    boot.add_argument("--owner-pid", type=int, required=True)
    boot.add_argument("--owner-start-ticks", type=int, required=True)
    boot.add_argument("--plan", type=Path, required=True)
    boot.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.mode == "bootstrap":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command:
            parser.error("Pinned worker command required")
        bootstrap(args, command)
        return
    if args.poll_seconds <= 0 or args.worker_timeout_seconds <= 0:
        parser.error("Positive polling interval and worker deadline required")
    args.output = check_output_root(args.output)
    args.model_ready, args.data_ready = args.model_ready.resolve(), args.data_ready.resolve()
    if args.output.exists() and any(args.output.iterdir()):
        path = args.output / "plan.json"
        if (not path.exists() or read(path).get("output") != str(args.output)
                or str(QUEUE.relative_to(ROOT)) not in read(path).get("queue_source_hashes", {})):
            raise RuntimeError("Existing output is not this queue's private pinned root; choose a new root")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == "preflight":
            if selector(os.getpid()) != "":
                raise RuntimeError("CPU preflight requires initially empty CUDA_VISIBLE_DEVICES")
            plan = build_plan(args)
            path = args.output / "plan.json"
            if path.exists() and read(path) != json.loads(json.dumps(plan)):
                raise RuntimeError("Preflight cannot replace a different pinned plan")
            write(path, plan)
            write(args.output / "preflight.json", {
                "status": "cpu_preflight_complete", "updated": time.time(),
                "plan_sha256": digest(path), "conditions": len(plan["jobs"]),
                "formal_questions_per_dataset": 500, "seed": 233,
                "gpu_work_launched": False})
        else:
            controller(args)


if __name__ == "__main__":
    main()
