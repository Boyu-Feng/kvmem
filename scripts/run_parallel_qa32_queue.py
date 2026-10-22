"""CPU-only scheduling of unchanged QA32 workers across shared three-slot GPU pairs."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_qa32_rebuttal as runner
from scripts import run_qa32_rebuttal_queue as queue
from scripts.run_rebuttal_efficiency_queue import (
    identity_alive, maintenance_blockers, maintenance_launch_lock, maintenance_path, process_identity,
)

BASE = Path("/mnt/checkpoints/bruli/fengboyu/kvmem_rebuttal_20260929")
PAIRS = ((3, 4), (5, 6), (0, 7))
SUPPORTED_PAIRS = (*PAIRS, (2, 7))
MAX_WORKERS = 8
DATASETS = ("hotpotqa", "2wiki", "musique")
FIELDS = ("dataset", "method", "ratio", "purpose", "samples", "sample_start", "seed", "max_steps")
LOAD = Path("/mnt/3fs/models/bruli/gpu-utilization-20260925/gpu_load.py")
MAX_OOM_RETRIES = 2
PEER_DATA = BASE / "qa32_assets/data_ready.json"
PEER_SCOPES = {
    BASE / "results/component_ablation_seeds233_3407_20260930_v1": (
        ROOT / "scripts/run_component_ablation_rebuttal.py", BASE / "assets/7b_ready.json", "qwen25_7b"),
    BASE / "results/shared_accuracy/llama_component_ablation_seeds233_3407_v1": (
        ROOT / "scripts/run_component_ablation_rebuttal.py",
        BASE / "ablation_assets/llama31_8b_shared_ready.json", "llama31_8b"),
    BASE / "results/sensitivity_shared/qwen7b_seed233_v1": (
        ROOT / "scripts/run_stepkv_sensitivity.py", BASE / "assets/7b_ready.json", "qwen25_7b"),
}


def pair_key(pair):
    return ",".join(map(str, pair))


def command(python, assets, root, job):
    result = [
        str(python), "-u", str(queue.RUNNER), "--assets", str(assets),
        "--output", str(root / job["name"]), "--memory-limit-gib", "44",
    ]
    for key in FIELDS:
        result.extend(["--" + key.replace("_", "-"), str(job[key])])
    return result


def environment(assets, tmp_root, pair):
    base = assets.parent
    env = dict(
        os.environ, CUDA_VISIBLE_DEVICES=pair_key(pair),
        OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
        HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
    )
    env.update({
        "HF_HOME": str(base / "assets/hf_cache"), "HF_HUB_CACHE": str(base / "assets/hf_cache/hub"),
        "TORCH_HOME": str(base / "caches/torch"),
        "TORCH_EXTENSIONS_DIR": str(base / "caches/torch_extensions"),
        "XDG_CACHE_HOME": str(base / "caches/xdg"), "TRITON_CACHE_DIR": str(base / "caches/triton"),
        "TORCHINDUCTOR_CACHE_DIR": str(base / "caches/inductor"), "TMPDIR": str(tmp_root),
    })
    return env


def process_args(pid, proc=Path("/proc")):
    directory = proc / str(pid)
    try:
        if directory.stat().st_uid != os.getuid():
            return None
        return [os.fsdecode(part) for part in
                (directory / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")]
    except (FileNotFoundError, ProcessLookupError):
        return None


def script_entry(argv):
    if not argv or not Path(argv[0]).name.startswith("python"):
        return None
    index = 1
    while index < len(argv) and argv[index] in ("-u", "-B"):
        index += 1
    return argv[index] if index < len(argv) and not argv[index].startswith("-") else None


def process_env(pid, proc):
    return dict(part.split(b"=", 1) for part in
                (proc / str(pid) / "environ").read_bytes().split(b"\0") if b"=" in part)


def flag(argv, name):
    if not argv or argv.count(name) != 1:
        return None
    index = argv.index(name) + 1
    return argv[index] if index < len(argv) else None


def own_worker(pid, active, proc, gpu=None):
    for record in active.values():
        if record["worker"]["pid"] == pid:
            return ((gpu is None or gpu in record["gpus"])
                    and identity_alive(record["worker"], proc)
                    and process_args(pid, proc) == record["command"])
    return False


def own_load(pid, gpu, proc):
    argv = process_args(pid, proc)
    return bool(argv and script_entry(argv) == str(LOAD)
                and argv[1:] == ["-B", str(LOAD), "worker", "--gpu", str(gpu)]
                and process_identity(pid, proc))


def loaded_status(directory, pid, started=0):
    path = directory / "status.json"
    try:
        status = runner.read_json(path)
        return (path.stat().st_mtime >= started and status.get("pid") == pid
                and status.get("status") == "running")
    except FileNotFoundError:
        return False


def expected_worker(pid, pair, roots, python, assets, devices, proc):
    """Only exact, fully loaded original-root QA commands are admissible peers."""
    argv = process_args(pid, proc)
    if script_entry(argv) != str(queue.RUNNER) or not process_identity(pid, proc):
        return False
    if not all(pid in devices[g]["allocations"] for g in pair):
        return False
    for root, plan in roots.items():
        for job in plan["jobs"]:
            directory = root / job["name"]
            if argv != command(python, assets, root, job):
                continue
            manifest = directory / "manifest.json"
            if not manifest.exists() or not loaded_status(directory, pid):
                return False
            return manifest_pair(runner.read_json(manifest)) == pair
    return False


def known_small_peer(pid, gpu, devices, python, proc=Path("/proc")):
    """Exact registered shared-accuracy CLI only; loading/pending peers still block."""
    argv = process_args(pid, proc)
    if not argv or argv[0] != str(python) or not process_identity(pid, proc):
        return False
    if (devices[gpu]["allocations"].get(pid) or 0) <= 0:
        return False
    output = flag(argv, "--output")
    if not output:
        return False
    for root, (script, model_ready, model) in PEER_SCOPES.items():
        if script_entry(argv) != str(script) or flag(argv, "--model-ready") != str(model_ready):
            continue
        try:
            relative = Path(output).relative_to(root)
        except ValueError:
            continue
        if ".." in relative.parts:
            return False
        plan_path = root / "queue_plan.json"
        status_path = Path(output) / "status.json"
        if not plan_path.exists() or not status_path.exists():
            return False
        plan, status = runner.read_json(plan_path), runner.read_json(status_path)
        if status.get("status") != "running" or status.get("pid", pid) != pid:
            return False
        if plan.get("model_readys", {}).get(model) != str(model_ready):
            return False
        if plan.get("data_ready") != str(PEER_DATA):
            return False
        for job in plan["jobs"]:
            if job["name"] != str(relative) or job["model"] != model:
                continue
            expected = [
                str(python), "-u", str(script), "--execute", "--model-ready", str(model_ready),
                "--data-ready", plan["data_ready"], "--output", str(root / job["name"]),
                "--dataset", job["dataset"], "--seed", str(job["seed"]),
                "--variant", job["variant"], "--purpose", job["purpose"],
            ]
            if script.name == "run_stepkv_sensitivity.py":
                expected.extend(["--case", job["case"], "--ratio", str(job["ratio"])])
            return argv == expected
    return False


def admission_blockers(pair, devices, roots, python, assets, active, proc=Path("/proc")):
    blockers = []

    def known(pid, gpu):
        return (own_worker(pid, active, proc, gpu) or own_load(pid, gpu, proc)
                or expected_worker(pid, pair, roots, python, assets, devices, proc)
                or known_small_peer(pid, gpu, devices, python, proc))

    for gpu in pair:
        for pid in devices[gpu]["allocations"]:
            try:
                if not known(pid, gpu):
                    blockers.append({"pid": pid, "gpu": gpu, "reason": "unknown CUDA process"})
            except PermissionError:
                blockers.append({"pid": pid, "gpu": gpu, "reason": "unreadable CUDA process"})
    for directory in proc.iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        pid = int(directory.name)
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            argv = process_args(pid, proc)
            # sshd/systemd and other CPU binaries often have unreadable environ.
            if not argv or not Path(argv[0]).name.startswith("python"):
                continue
            env = process_env(pid, proc)
            visible = env.get(b"CUDA_VISIBLE_DEVICES")
            assignments = os.fsdecode(visible).split(",") if visible is not None else []
            targets = [g for g in pair if str(g) in assignments
                       or devices[g]["uuid"] in assignments]
            for gpu in targets:
                if not known(pid, gpu):
                    blockers.append({"pid": pid, "gpu": gpu, "reason": "unknown/pending assignment"})
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            blockers.append({"pid": pid, "reason": "unreadable Python assignment"})
    return blockers


def gpu_snapshot():
    devices = {d["index"]: {**d, "allocations": {}} for d in queue.gpu_snapshot(list(range(8)))}
    names = subprocess.check_output([
        "nvidia-smi", "--query-gpu=uuid,name", "--format=csv,noheader",
    ], text=True, timeout=20)
    by_uuid = {d["uuid"]: d for d in devices.values()}
    for line in names.strip().splitlines():
        uuid, name = [part.strip() for part in line.split(",", 1)]
        if uuid not in by_uuid:
            raise RuntimeError(f"Unrecognized GPU name entry: {uuid}")
        by_uuid[uuid]["name"] = name
    raw = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=20)
    for line in raw.strip().splitlines():
        uuid, pid, memory = [part.strip() for part in line.split(",")]
        if uuid not in by_uuid:
            raise RuntimeError(f"Unrecognized CUDA device: {uuid}")
        # Unknown accounting must block admission, not be mistaken for no CUDA.
        by_uuid[uuid]["allocations"][int(pid)] = int(memory) if memory.isdigit() else None
    return devices


def slots_available(pair, devices, active, limits, max_workers=MAX_WORKERS):
    records = [r for r in active.values() if tuple(r["gpus"]) == pair]
    ready = all(
        loaded_status(r["directory"], r["worker"]["pid"], r["started"])
        and all((devices[g]["allocations"].get(r["worker"]["pid"]) or 0) > 0 for g in pair)
        for r in records
    )
    return (len(active) < min(MAX_WORKERS, max_workers) and len(records) < limits[pair_key(pair)] and ready
            and all(devices[g].get("name") == "NVIDIA H200" for g in pair)
            and queue.enough_memory([devices[g] for g in pair], 48))


def manifest_pair(manifest):
    attempts = manifest.get("hardware_attempts", [])
    if not attempts:
        return None
    value = attempts[-1].get("cuda_visible_devices")
    return next((pair for pair in SUPPORTED_PAIRS if value == pair_key(pair)), None)


def portable_manifest(manifest):
    fp = manifest["fingerprint"]
    placement = manifest.get("actual_placement", {})
    attempts = manifest.get("hardware_attempts", [])
    policy = fp.get("placement_policy", {})
    return (
        policy.get("device_map") == "balanced" and policy.get("dtype") == "bfloat16"
        and policy.get("max_memory") == {"0": "40GiB", "1": "40GiB"}
        and set(placement.get("hf_device_map", {}).values()) == {0, 1}
        and set(placement.get("layer_devices", [])) == {"cuda:0", "cuda:1"}
        and attempts and set(attempts[-1].get("devices", {})) == {"0", "1"}
        and all(device.get("name") == "NVIDIA H200" for device in attempts[-1]["devices"].values())
        and not any(key in fp for key in ("gpus", "cuda_visible_devices", "physical_devices"))
    )


def resume_pin(directory, job, gate):
    """Read-only prefix checks; the unchanged worker performs full fingerprint validation."""
    manifest_path = directory / "manifest.json"
    checkpoint_path = directory / "checkpoint.json"
    if not manifest_path.exists():
        if checkpoint_path.exists():
            raise RuntimeError(f"Checkpoint lacks manifest: {directory}")
        return {"committed": 0, "pair": None}
    manifest = runner.read_json(manifest_path)
    fp = manifest["fingerprint"]
    digest = runner.digest(fp)
    if manifest.get("schema_version") != runner.SCHEMA or manifest.get("fingerprint_sha256") != digest:
        raise RuntimeError(f"Corrupt manifest: {directory}")
    if any(fp.get(k) != job[k] for k in FIELDS) or fp.get("memory_limit_gib") != 44:
        raise RuntimeError(f"Changed job protocol: {directory}")
    if fp.get("original_core_hashes") != gate["original_core_hashes"]:
        raise RuntimeError(f"Changed core pins: {directory}")
    for name, expected in fp["source_hashes"].items():
        if runner.file_hash(ROOT / name) != expected:
            raise RuntimeError(f"Changed source pin: {name}")
    if any(importlib.metadata.version(name) != version for name, version in fp["versions"].items()):
        raise RuntimeError(f"Changed runtime pins: {directory}")
    ids = fp["sample_ids"]
    if len(ids) != job["samples"] or len(set(ids)) != len(ids):
        raise RuntimeError(f"Invalid selected IDs: {directory}")
    pair = manifest_pair(manifest)
    if manifest.get("hardware_attempts") and pair is None:
        raise RuntimeError(f"Unexpected physical device pair: {directory}")
    rows = []
    if checkpoint_path.exists():
        state = runner.read_json(checkpoint_path)
        rows = state["results"]
        if (state.get("schema_version") != runner.SCHEMA
                or state.get("fingerprint_sha256") != digest
                or state.get("rows_sha256") != runner.digest(rows)
                or [r["id"] for r in rows] != ids[:len(rows)] or len(rows) > len(ids)):
            raise RuntimeError(f"Corrupt/nonprefix checkpoint: {directory}")
        if rows and (pair is None or "actual_placement" not in manifest):
            raise RuntimeError(f"Committed rows lack verified placement: {directory}")
        for row in rows:
            runner._read_question_audit(directory, digest, row)
    if "actual_placement" in manifest:
        if manifest.get("actual_placement_sha256") != runner.digest(manifest["actual_placement"]):
            raise RuntimeError(f"Changed placement: {directory}")
    return {
        "committed": len(rows), "pair": list(pair) if pair else None,
        "portable_pair": bool(portable_manifest(manifest)),
        "manifest_sha256": runner.file_hash(manifest_path), "fingerprint": fp,
        "checkpoint_sha256": runner.file_hash(checkpoint_path) if checkpoint_path.exists() else None,
        "committed_ids": [r["id"] for r in rows],
    }


def smoke_key(root, job):
    return str(root), job["dataset"], job["method"], job["ratio"]


def ready_jobs(roots, completed, active, pins, launches, pair):
    busy = set(active)
    passed = {smoke_key(root, job) for root, plan in roots.items() for job in plan["jobs"]
              if job["purpose"] == "smoke" and (str(root), job["name"]) in completed}
    candidates = []
    for root, plan in roots.items():
        for job in plan["jobs"]:
            key = str(root), job["name"]
            pinned_pair = pins[key].get("pair")
            if key in completed or key in busy or (
                    pinned_pair and not pins[key].get("portable_pair") and tuple(pinned_pair) != pair):
                continue
            if job["purpose"] == "main" and smoke_key(root, job) not in passed:
                continue
            dataset_active = sum(r["job"]["dataset"] == job["dataset"] for r in active.values())
            # Fair dataset slots first, then longest/tokenwise jobs early to shorten the tail.
            long = job["method"] in ("h2o", "tova")
            priority = (dataset_active, launches.get(job["dataset"], 0), not long,
                        job["purpose"] != "main", -pins[key]["committed"], job["name"])
            candidates.append((priority, root, job))
    return [(root, job) for _, root, job in sorted(candidates, key=lambda row: row[0])]


def oom_failure(text):
    lowered = text.lower()
    return ("cuda out of memory" in lowered or "torch.outofmemoryerror" in lowered
            or "cuda error: out of memory" in lowered)


def record_oom(output, key, record, limits, retries, text):
    pair = pair_key(record["gpus"])
    previous = limits[pair]
    retries[key] = retries.get(key, 0) + 1
    limits[pair] = max(1, previous - 1)
    event = {
        "at": time.time(), "root": key[0], "job": key[1], "gpus": record["gpus"],
        "worker": record["worker"], "retry": retries[key], "previous_pair_limit": previous,
        "pair_limit": limits[pair], "error_tail": text[-8000:],
    }
    with (output / "oom_events.jsonl").open("a") as stream:
        stream.write(json.dumps(event) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(f"CUDA OOM {key}: pair {pair} concurrency {previous}->{limits[pair]}; "
          f"retry {retries[key]}/{MAX_OOM_RETRIES}", file=sys.stderr, flush=True)
    if previous == 1 or retries[key] > MAX_OOM_RETRIES:
        raise RuntimeError(f"Bounded OOM retries exhausted: {key}; unchanged worker cannot fit")


def public_active(active):
    return [{
        "root": str(r["root"]), "job": r["job"], "gpus": list(r["gpus"]),
        "worker": r["worker"], "command": r["command"], "log_path": str(r["log_path"]),
        "started": r["started"],
    } for r in active.values()]


def publish(output, roots, active, completed, owner, status, limits, max_workers=MAX_WORKERS, **extra):
    records = public_active(active)
    now = time.time()
    atomic = runner.atomic_json
    atomic(output / "queue_status.json", {
        "status": status, "owner": owner, "supervisor_pid": owner["pid"], "updated": now,
        "active": records, "completed_jobs": len(completed), "total_jobs": 66,
        "completed_main": sum("/main/" in key[1] for key in completed), "expected_main": 33,
        "pair_limits": limits, "max_workers": max_workers,
        "timing_policy": "Shared GPU diagnostic only, not formal latency",
        **extra,
    })
    for root, plan in roots.items():
        selected = [r for r in records if r["root"] == str(root)]
        previous_path = root / "queue_status.json"
        previous = runner.read_json(previous_path) if previous_path.exists() else {}
        current = selected[0] if selected else {}
        atomic(previous_path, {
            **previous, "status": status if status != "running" else
            ("running" if selected else "complete" if
             all((str(root), j["name"]) in completed for j in plan["jobs"]) else "waiting_for_slot"),
            "owner": owner, "supervisor_pid": owner["pid"], "updated": now,
            "job": current.get("job"), "worker_identity": current.get("worker"),
            "command": current.get("command"), "log_path": current.get("log_path"),
            "active": selected, "completed_jobs": sum(k[0] == str(root) for k in completed),
            "total_jobs": len(plan["jobs"]), "parallel_scheduler": str(output),
        })


def load_plans(results, assets):
    roots, raw_plans = {}, {}
    gate = queue.source_gate()
    for dataset in DATASETS:
        root = (results / f"qa32_{dataset}_v1").resolve()
        raw = (root / "queue_plan.json").read_bytes()
        plan = json.loads(raw)
        if (plan.get("protocol") != "qa32_frozen_legacy_shared_h200_v1"
                or plan.get("datasets") != [dataset] or plan.get("jobs") != queue.jobs([dataset])
                or plan.get("source_gate") != gate or plan.get("assets") != str(assets)
                or plan.get("memory_limit_gib_per_gpu") != 44 or plan.get("min_free_gib_per_gpu") != 48
                or tuple(plan.get("gpus", [])) not in PAIRS):
            raise RuntimeError(f"Original QA32 plan/pins differ: {root}")
        roots[root], raw_plans[root] = plan, raw
    return roots, raw_plans, gate


def assert_pins(raw_plans, gate, asset_files):
    if queue.source_gate() != gate:
        raise RuntimeError("Frozen QA32 source gate changed")
    for root, raw in raw_plans.items():
        if (root / "queue_plan.json").read_bytes() != raw:
            raise RuntimeError(f"Original queue_plan bytes changed: {root}")
    for path, expected in asset_files.items():
        if runner.file_hash(path) != expected:
            raise RuntimeError(f"Pinned ready asset changed: {path}")


def reject_live_status(path):
    if not path.exists():
        return
    state = runner.read_json(path)
    if (identity_alive(state.get("owner")) or identity_alive(state.get("worker_identity"))
            or process_identity(state.get("supervisor_pid"))
            or any(identity_alive(r.get("worker")) for r in state.get("active", []))):
        raise RuntimeError(f"Controller/worker still alive; parent must stop it safely: {path}")


def graceful_stop(active, timeout=120):
    for record in active.values():
        if identity_alive(record["worker"]) and process_args(record["worker"]["pid"]) == record["command"]:
            os.kill(record["worker"]["pid"], signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while any(r["process"].poll() is None for r in active.values()) and time.monotonic() < deadline:
        time.sleep(.2)
    # Never escalate to SIGKILL; surviving children retain inherited queue/GPU leases.
    return [r["worker"] for r in active.values() if r["process"].poll() is None]


def execute(args):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Launch the scheduler CPU-only with CUDA_VISIBLE_DEVICES=''")
    roots, raw_plans, gate = load_plans(args.results, args.assets)
    args.output.mkdir(parents=True, exist_ok=True)
    owner = process_identity(os.getpid())
    if owner is None:
        raise RuntimeError("Cannot establish scheduler identity")
    with contextlib.ExitStack() as stack:
        lease_fds = []
        paths = [args.output / "queue.lock", *(r / "queue.lock" for r in roots)]
        paths += [args.results / f"alfworld_gpu{gpu}.lock" for gpu in (3, 4, 5, 6)]
        # Shared peers retain their own namespaces; opportunistic QA never steals their leases.
        paths += [args.results / f"qa32_shared_gpu{gpu}.lock"
                  for gpu in sorted({g for pair in args.pairs for g in pair} - {3, 4, 5, 6})]
        for path in sorted(paths):
            handle = stack.enter_context(path.open("a"))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lease_fds.append(handle.fileno())
        for root in roots:
            reject_live_status(root / "queue_status.json")
        reject_live_status(args.output / "queue_status.json")
        completed, pins, asset_files = set(), {}, {}
        asset_provenance = {}
        for dataset in DATASETS:
            _, _, _, provenance, _ = runner.load_assets(args.assets, dataset)
            asset_provenance[dataset] = provenance
        for path in (args.assets / "model_ready.json", args.assets / "data_ready.json"):
            asset_files[path] = runner.file_hash(path)
        for root, plan in roots.items():
            for job in plan["jobs"]:
                directory = root / job["name"]
                key = str(root), job["name"]
                pins[key] = resume_pin(directory, job, gate)
                fp = pins[key].get("fingerprint")
                if fp and fp["assets"] != asset_provenance[job["dataset"]]:
                    raise RuntimeError(f"Asset/runtime fingerprint changed: {directory}")
                status = directory / "status.json"
                if status.exists() and runner.read_json(status).get("status") == "complete":
                    queue.validate_job(directory, job)
                    completed.add(key)
            queue.summarize(root, plan)
        scheduler_pin = {
            "protocol": "qa32_parallel_deadline_scheduling_only_v1",
            "scheduler_sha256": runner.file_hash(Path(__file__)), "source_gate": gate,
            "pairs": [list(p) for p in args.pairs], "max_workers": args.max_workers, "max_workers_per_pair": 3,
            "assets": str(args.assets), "python": str(args.python), "tmp_root": str(args.tmp_root),
            "original_plan_sha256": {str(r): hashlib.sha256(b).hexdigest() for r, b in raw_plans.items()},
        }
        if getattr(args, "priority_overlay", None) is not None:
            scheduler_pin["priority_overlay"] = {
                "path": str(args.priority_overlay),
                "sha256": runner.file_hash(args.priority_overlay),
                "dataset_order": list(args.dataset_order),
            }
        pin_path = args.output / "scheduler_plan.json"
        if pin_path.exists() and runner.read_json(pin_path) != scheduler_pin:
            raise RuntimeError("Scheduler/source/configuration pins changed; use a new scheduler output")
        runner.atomic_json(pin_path, scheduler_pin)
        handover = {
            "owner": owner, "at": time.time(), "source_gate": gate,
            "originals": [{
                "root": str(r), "queue_plan_utf8": raw_plans[r].decode(),
                "queue_status": runner.read_json(r / "queue_status.json"),
            } for r in roots],
            "jobs": [{"root": k[0], "job": k[1], **pin} for k, pin in pins.items()],
            "assets": asset_provenance,
        }
        runner.atomic_json(args.output / f"handover_{owner['pid']}_{owner['start_ticks']}.json", handover)
        limits = {pair_key(p): 3 for p in args.pairs}
        retries, launches, active = {}, {}, {}
        # A restarted scheduler must not reset the OOM safety budget.
        oom_path = args.output / "oom_events.jsonl"
        if oom_path.exists():
            for line in oom_path.read_text().splitlines():
                event = json.loads(line)
                limits[pair_key(event["gpus"])] = event["pair_limit"]
                retries[(event["root"], event["job"])] = event["retry"]
                if event["previous_pair_limit"] == 1 or event["retry"] > MAX_OOM_RETRIES:
                    raise RuntimeError("Previous bounded OOM failure requires operator review")
        def stop(signum, frame):
            raise KeyboardInterrupt(f"Received signal {signum}")
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, stop)
        try:
            while len(completed) < sum(len(p["jobs"]) for p in roots.values()):
                assert_pins(raw_plans, gate, asset_files)
                if runner.file_hash(Path(__file__)) != scheduler_pin["scheduler_sha256"]:
                    raise RuntimeError("Scheduler source changed during execution")
                changed = set()
                for key, record in list(active.items()):
                    code = record["process"].poll()
                    if code is None:
                        continue
                    record["log"].close()
                    del active[key]
                    if code:
                        with record["log_path"].open("rb") as stream:
                            stream.seek(0, os.SEEK_END)
                            stream.seek(max(record["log_offset"], stream.tell() - 65536))
                            error = stream.read().decode(errors="replace")
                        if not oom_failure(error):
                            raise RuntimeError(f"QA32 worker {key} exited {code}: {error[-8000:]}")
                        record_oom(args.output, key, record, limits, retries, error)
                        pins[key] = resume_pin(record["directory"], record["job"], gate)
                        pins[key]["pair"] = record["gpus"]
                    else:
                        queue.validate_job(record["directory"], record["job"])
                        completed.add(key)
                        changed.add(record["root"])
                for root in changed:
                    queue.summarize(root, roots[root])
                devices = gpu_snapshot()
                blocked = {}
                for pair in args.pairs:
                    with contextlib.ExitStack() as admission:
                        for gpu in sorted(pair):
                            admission.enter_context(maintenance_launch_lock(gpu))
                        maintenance = {g: maintenance_blockers(maintenance_path(g)) for g in pair}
                        if any(maintenance.values()):
                            blocked[pair_key(pair)] = {"maintenance": maintenance}
                            continue
                        devices = gpu_snapshot()
                        if not slots_available(pair, devices, active, limits, args.max_workers):
                            continue
                        blockers = admission_blockers(
                            pair, devices, roots, args.python, args.assets, active)
                        if blockers:
                            blocked[pair_key(pair)] = blockers
                            continue
                        candidates = ready_jobs(roots, completed, active, pins, launches, pair)
                        if not candidates:
                            continue
                        root, job = candidates[0]
                        assert_pins(raw_plans, gate, asset_files)
                        directory = root / job["name"]
                        cmd = command(args.python, args.assets, root, job)
                        env = environment(args.assets, args.tmp_root, pair)
                        for name in ("HF_HOME", "HF_HUB_CACHE", "TORCH_HOME", "TORCH_EXTENSIONS_DIR",
                                     "XDG_CACHE_HOME", "TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TMPDIR"):
                            Path(env[name]).mkdir(parents=True, exist_ok=True)
                        log_path = root / "logs" / (job["name"].replace("/", "__") + ".log")
                        log_path.parent.mkdir(parents=True, exist_ok=True)
                        log = log_path.open("a")
                        offset, started = log.tell(), time.time()
                        process = subprocess.Popen(
                            cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                            pass_fds=tuple(lease_fds),
                        )
                        worker = process_identity(process.pid)
                        if worker is None:
                            process.wait()
                            log.close()
                            raise RuntimeError(f"Cannot establish child identity: {cmd}")
                        key = str(root), job["name"]
                        active[key] = {
                            "root": root, "directory": directory, "job": job, "gpus": list(pair),
                            "worker": worker, "process": process, "command": cmd, "started": started,
                            "log": log, "log_path": log_path, "log_offset": offset,
                        }
                        launches[job["dataset"]] = launches.get(job["dataset"], 0) + 1
                        # Remember the pair even if OOM precedes hardware recording.
                        pins[key]["pair"] = list(pair)
                publish(args.output, roots, active, completed, owner, "running", limits,
                        max_workers=args.max_workers, blockers=blocked, devices=list(devices.values()))
                if len(completed) < 66:
                    time.sleep(args.poll_seconds)
            for root, plan in roots.items():
                queue.summarize(root, plan)
            publish(args.output, roots, active, completed, owner, "complete", limits,
                    max_workers=args.max_workers)
        except BaseException:
            failure = traceback.format_exc()
            surviving = graceful_stop(active)
            publish(args.output, roots, active, completed, owner, "failed", limits,
                    max_workers=args.max_workers, traceback=failure, surviving_workers=surviving)
            raise
        finally:
            for record in active.values():
                record["log"].close()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--results", type=Path, default=BASE / "results")
    result.add_argument("--assets", type=Path, default=BASE / "qa32_assets")
    result.add_argument("--output", type=Path, default=BASE / "results/qa32_parallel_deadline_20261002_v1")
    result.add_argument("--python", type=Path, default=BASE / "venv/bin/python")
    result.add_argument("--tmp-root", type=Path, default=BASE / "latency_caches/tmp")
    result.add_argument("--pairs", nargs="+", choices=[pair_key(p) for p in SUPPORTED_PAIRS],
                        default=[pair_key(p) for p in PAIRS])
    result.add_argument("--max-workers", type=int, default=MAX_WORKERS)
    result.add_argument("--poll-seconds", type=float, default=5)
    result.add_argument("--priority-overlay", type=Path)
    result.add_argument("--dataset-order", nargs="+", choices=DATASETS)
    return result


def main():
    args = parser().parse_args()
    if not 0 < args.poll_seconds <= 30:
        raise ValueError("poll-seconds must be in (0,30]")
    if not 1 <= args.max_workers <= MAX_WORKERS or len(set(args.pairs)) != len(args.pairs):
        raise ValueError("Require max-workers in [1,8] and nonduplicate physical pairs")
    args.pairs = tuple(tuple(map(int, value.split(","))) for value in args.pairs)
    physical = [gpu for pair in args.pairs for gpu in pair]
    if len(physical) != len(set(physical)):
        raise ValueError("Selected physical GPU pairs must be disjoint")
    for name in ("results", "assets", "output", "python", "tmp_root"):
        setattr(args, name, getattr(args, name).absolute())
    if any(args.output.is_relative_to(args.results / f"qa32_{d}_v1") for d in DATASETS):
        raise ValueError("Scheduler output must be new, not an original QA32 root")
    if args.python != Path(sys.executable).absolute():
        raise ValueError("Run scheduler and workers with the same pinned Python executable")
    if args.priority_overlay is None:
        if args.dataset_order is not None:
            raise ValueError("dataset-order requires the explicit scheduling-only priority overlay")
        execute(args)
        return
    expected = ROOT / "scripts/run_qa32_thinkkv_latency_handoff.py"
    if args.priority_overlay.resolve() != expected or (
            args.dataset_order is None or len(args.dataset_order) != len(DATASETS)
            or set(args.dataset_order) != set(DATASETS)):
        raise ValueError("Require the known priority overlay and each frozen dataset exactly once")
    args.priority_overlay = expected
    from scripts.run_qa32_thinkkv_latency_handoff import install_priority
    scheduler_sha = runner.file_hash(Path(__file__))
    overlay_sha = runner.file_hash(expected)

    def verify_priority_sources():
        if (runner.file_hash(Path(__file__)) != scheduler_sha
                or runner.file_hash(expected) != overlay_sha):
            raise RuntimeError("Priority scheduling sources changed during execution")

    with contextlib.ExitStack() as scope:
        install_priority(scope, verify_priority_sources, tuple(args.dataset_order), sys.modules[__name__])
        execute(args)


if __name__ == "__main__":
    main()
