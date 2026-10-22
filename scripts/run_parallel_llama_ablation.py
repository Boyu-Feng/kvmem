"""Scheduling-only Llama acceleration; frozen workers retain checkpoint/scoring behavior.

CPU-only controller, five slots, at most two workers per physical GPU. Model
loading on each GPU is serialized; CUDA OOMs are recorded and retried at reduced
concurrency (twice per job), never by changing the experiment or memory threshold.
"""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_component_ablation_queue as queue
from scripts import run_shared_llama_ablation as shared
from scripts import run_parallel_qwen_ablation as peers
from scripts.run_qa32_rebuttal import atomic_json, file_hash, read_json
from scripts.run_rebuttal_efficiency_queue import (
    identity_alive, maintenance_blockers, maintenance_launch_lock, maintenance_path,
    process_identity,
)

BASE = Path("/mnt/checkpoints/bruli/fengboyu/kvmem_rebuttal_20260929")
PEER_MODEL = BASE / "assets/7b_ready.json"
PEER_SCOPES = {
    str(queue.WORKER): BASE / "results/component_ablation_seeds233_3407_20260930_v1",
    str(ROOT / "scripts/run_stepkv_sensitivity.py"):
        BASE / "results/sensitivity_shared/qwen7b_seed233_v1",
}
ORIGINAL_EXISTING_ACCURACY = shared.existing_accuracy
MAX_OOM_RETRIES = 2


def source_gate():
    return {**queue.source_gate(),
            "scripts/run_shared_llama_ablation.py": file_hash(Path(shared.__file__))}


def process_args(pid, proc=Path("/proc")):
    directory = proc / str(pid)
    try:
        if directory.stat().st_uid != os.getuid():
            return None
        return [os.fsdecode(part) for part in
                directory.joinpath("cmdline").read_bytes().rstrip(b"\0").split(b"\0")]
    except FileNotFoundError:
        return None


def flag(args, name):
    if args.count(name) != 1:
        return None
    index = args.index(name) + 1
    return args[index] if index < len(args) else None


def runner(args):
    if not args or not Path(args[0]).name.startswith("python"):
        return None
    index = 2 if len(args) > 1 and args[1] in ("-u", "-B") else 1
    return args[index] if len(args) > index else None


def known_accuracy(pid, proc=Path("/proc")):
    """Recognize only this UID's exact dedicated Qwen7B accuracy scopes."""
    args = process_args(pid, proc)
    script = runner(args)
    if script == str(peers.QA32_WORKER):
        return peers.known_qa32(args)
    if script not in PEER_SCOPES or "--execute" not in args:
        return False
    if flag(args, "--model-ready") != str(PEER_MODEL):
        return False
    if script == str(queue.WORKER) and not ORIGINAL_EXISTING_ACCURACY(pid, proc):
        return False
    output = flag(args, "--output")
    if not output:
        return False
    try:
        relative = Path(output).relative_to(PEER_SCOPES[script])
    except ValueError:
        return False
    return (len(relative.parts) >= 5 and relative.parts[0] == "qwen25_7b"
            and ".." not in relative.parts)


def own_worker(pid, args, active, proc=Path("/proc")):
    record = next((r for r in active.values() if r["process"].pid == pid), None)
    return (record is not None and identity_alive(record["worker"], proc)
            and process_args(pid, proc) == queue.command(args, record["job"]))


def loaded_status(directory, pid, started=0):
    path = directory / "status.json"
    try:
        status = read_json(path)
        return (path.stat().st_mtime >= started
                and status.get("status") in ("running", "complete")
                and status.get("pid", pid) == pid)
    except FileNotFoundError:
        return False


def steady_accuracy(pid, device, proc=Path("/proc"), devices=None):
    if not known_accuracy(pid, proc):
        return False
    command = process_args(pid, proc)
    if runner(command) == str(peers.QA32_WORKER):
        return peers.steady_peer(pid, device, proc, devices)
    output = flag(command, "--output")
    return pid in device["pids"] and loaded_status(Path(output), pid)


def assigned_blocker(gpu, uuid, args, active, device, proc=Path("/proc"), devices=None):
    """Inspect assignments before CUDA registration, including pending peers."""
    for directory in proc.iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            if runner(process_args(int(directory.name), proc)) is None:
                continue
            environment = directory.joinpath("environ").read_bytes().split(b"\0")
            visible = [part.split(b"=", 1)[1].decode() for part in environment
                       if part.startswith(b"CUDA_VISIBLE_DEVICES=")]
            if not any(str(gpu) in value.split(",") or uuid in value.split(",")
                       for value in visible):
                continue
            pid = int(directory.name)
            if own_worker(pid, args, active, proc) or queue.own_load(pid, gpu, proc):
                continue
            if steady_accuracy(pid, device, proc, devices):
                continue
            return True
        except FileNotFoundError:
            continue
        except PermissionError:
            return True
    return False


def available(device, args, active, proc=Path("/proc"), devices=None):
    return (device["free_mib"] >= args.min_free_gib * 1024
            and all(queue.own_load(pid, device["index"], proc)
                    or own_worker(pid, args, active, proc)
                    or steady_accuracy(pid, device, proc, devices) for pid in device["pids"]))


def slots_available(device, args, active, effective_limit, gpu_limits):
    records = [r for r in active.values() if r["gpu"] == device["index"]]
    for record in records:
        if (not record["startup_ready"] and record["process"].pid in device["pids"]
                and loaded_status(args.output / record["job"]["name"],
                                  record["process"].pid, record["started"])):
            record["startup_ready"] = True
    return (len(active) < min(5, args.max_workers, effective_limit)
            and len(records) < min(2, gpu_limits.get(device["index"], 2))
            and all(r["startup_ready"] for r in records))


def expected_plan(args):
    return {
        "protocol": "component_ablation_original_shared_gpu_v1",
        "model_readys": {key: str(value) for key, value in args.model_readys.items()},
        "data_ready": str(args.data_ready), "gpus": args.gpus, "max_workers": 5,
        "min_free_gib": args.min_free_gib, "tmp_root": str(args.tmp_root),
        "jobs": queue.jobs(args.model_readys), "source_gate": source_gate(),
        "timing_policy": "Shared GPU accuracy only; existing load workers are never stopped. "
                         "Timings are diagnostic, not latency measurements.",
        "full_control": "full means full StepKV, not FullKV; new same-seed controls are run.",
    }


def migrate_plan(args):
    """Caller holds queue.lock continuously through migration and execution."""
    path = args.output / "queue_plan.json"
    plan = read_json(path)
    previous_path = args.output / "queue_status.json"
    previous = read_json(previous_path) if previous_path.exists() else {}
    if identity_alive(previous.get("owner")) or any(
            identity_alive(item.get("worker")) for item in previous.get("active", [])):
        raise RuntimeError("Previous Llama controller or worker is still alive")
    expected = expected_plan(args)
    if plan.get("max_workers") not in (1, 5) or {**plan, "max_workers": 5} != expected:
        raise RuntimeError("Only max_workers 1->5 may change in the frozen Llama queue")
    record_path = args.output / "parallel_scheduler.json"
    fingerprint = file_hash(Path(__file__))
    if record_path.exists():
        record = read_json(record_path)
        if (record["launcher_sha256"] != fingerprint
                or {**record["previous_plan"], "max_workers": 5} != expected):
            raise RuntimeError("Parallel scheduler provenance changed")
    else:
        if plan["max_workers"] != 1:
            raise RuntimeError("Missing migration provenance")
        atomic_json(record_path, {
            "previous_plan": plan, "previous_owner": previous.get("owner"),
            "max_workers": 5, "per_gpu_workers": 2, "gpus": args.gpus,
            "min_free_gib": args.min_free_gib, "launcher_sha256": fingerprint,
            "created": time.time(), "scope": "Scheduling only; frozen worker and checkpoints",
        })
    atomic_json(path, expected)
    return expected, previous


CUDA_OOM = re.compile(
    r"CUDA out of memory|CUDA error:\s*out of memory|torch\.cuda\.OutOfMemoryError",
    re.IGNORECASE,
)


def is_cuda_oom(directory, offset, started):
    """Only inspect this attempt, never an old appended traceback."""
    with (directory / "run.log").open("rb") as handle:
        handle.seek(offset)
        # Tracebacks are at the tail; avoid reading an entire long experiment log.
        handle.seek(max(offset, handle.seek(0, os.SEEK_END) - 65536))
        text = handle.read().decode(errors="replace")
    status_path = directory / "status.json"
    if status_path.exists() and status_path.stat().st_mtime >= started:
        status = read_json(status_path)
        text += "\n" + str(status.get("reason", ""))
    return bool(CUDA_OOM.search(text))


def record_oom(job, gpu, retries, events, limit, gpu_limits):
    name = job["name"]
    retries[name] = retries.get(name, 0) + 1
    reduced = max(1, limit - 1)
    gpu_limits[gpu] = 1
    event = {"job": name, "gpu": gpu, "time": time.time(),
             "oom_attempt": retries[name], "retry_allowed": retries[name] <= MAX_OOM_RETRIES,
             "effective_limit_before": limit, "effective_limit_after": reduced}
    events.append(event)
    print(f"CUDA OOM: {event}", flush=True)
    return reduced, event["retry_allowed"]


def execute(args):
    active, completed, leases, asset_pins = {}, set(), {}, {}
    owner = process_identity(os.getpid())
    with (args.output / "queue.lock").open("a+") as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, previous = migrate_plan(args)
        scheduler_hash = file_hash(Path(__file__))
        peer_hash = file_hash(Path(peers.__file__))
        retries = dict(previous.get("oom_retries", {}))
        events = list(previous.get("oom_events", []))
        effective_limit = min(5, previous.get("effective_max_workers", 5))
        gpu_limits = {int(k): int(v) for k, v in previous.get("gpu_worker_limits", {}).items()}
        peak = previous.get("peak_active_workers", 0)
        slot = 0
        stop_requested = None

        def state(status, **extra):
            atomic_json(args.output / "queue_status.json", {
                "status": status, "owner": owner, "updated": time.time(),
                "completed_jobs": len(completed), "total_jobs": len(plan["jobs"]),
                "completed_main": sum(j["purpose"] == "main" and j["name"] in completed
                                      for j in plan["jobs"]),
                "active": [{"job": r["job"], "gpu": r["gpu"], "worker": r["worker"],
                            "slot": s, "startup_ready": r["startup_ready"]}
                           for s, r in active.items()],
                "requested_max_workers": 5, "effective_max_workers": effective_limit,
                "effective_concurrency": len(active), "peak_active_workers": peak,
                "gpu_worker_limits": gpu_limits, "oom_retries": retries, "oom_events": events,
                "progress": {
                    r["job"]["name"]: read_json(args.output / r["job"]["name"] / "status.json")
                    if (args.output / r["job"]["name"] / "status.json").exists() else
                    {"status": "starting"} for r in active.values()
                }, **extra,
            })

        def interrupted(signum, frame):
            # Defer the exception until a just-created child has an owned record.
            nonlocal stop_requested
            stop_requested = signum

        def check_interrupt():
            if stop_requested is not None:
                raise KeyboardInterrupt(
                    f"Received signal {stop_requested}; stopping only owned workers")

        old_handlers = {sig: signal.signal(sig, interrupted)
                        for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            args.tmp_root.mkdir(parents=True, exist_ok=True)
            cache_paths = {key: args.tmp_root.parent / value for key, value in (
                ("HF_HOME", "hf"), ("HF_DATASETS_CACHE", "datasets"), ("TORCH_HOME", "torch"),
                ("XDG_CACHE_HOME", "xdg"), ("TRITON_CACHE_DIR", "triton"))}
            for path in cache_paths.values():
                path.mkdir(parents=True, exist_ok=True)
            for job in plan["jobs"]:
                path = args.output / job["name"] / "status.json"
                if path.exists() and read_json(path).get("status") == "complete":
                    queue.validate_job(path.parent, job)
                    completed.add(job["name"])
            queue.summarize(args.output, plan, completed)
            while True:
                check_interrupt()
                if (source_gate() != plan["source_gate"]
                        or file_hash(Path(__file__)) != scheduler_hash
                        or file_hash(Path(peers.__file__)) != peer_hash
                        or read_json(args.output / "queue_plan.json") != plan):
                    raise RuntimeError("Source, scheduler, or frozen plan changed during queue")
                ready_models = {key for key, path in args.model_readys.items() if path.exists()}
                for model in ready_models:
                    path = args.model_readys[model]
                    if read_json(path).get("repo") != queue.MODEL_REPOS[model]:
                        raise RuntimeError(f"Wrong model assets: {model}")
                    pin = file_hash(path)
                    pin_path = args.output / f"{model}_asset_pin.json"
                    if model not in asset_pins:
                        if pin_path.exists() and read_json(pin_path)["sha256"] != pin:
                            raise RuntimeError(f"Model ready metadata changed: {model}")
                        asset_pins[model] = pin
                        atomic_json(pin_path, {"sha256": pin, "path": str(path)})
                    elif asset_pins[model] != pin:
                        raise RuntimeError(f"Model ready metadata changed during queue: {model}")
                if set(asset_pins) - ready_models:
                    raise RuntimeError("Pinned model metadata disappeared")
                for key, record in list(active.items()):
                    code = record["process"].poll()
                    if code is None:
                        continue
                    record["log"].close()
                    record["lease"].close()
                    del active[key]
                    job, gpu = record["job"], record["gpu"]
                    if not any(r["gpu"] == gpu for r in active.values()):
                        leases.pop(gpu).close()
                    if code:
                        if not is_cuda_oom(args.output / job["name"], record["log_offset"],
                                           record["started"]):
                            raise RuntimeError(f"Ablation failed ({code}): {job['name']}")
                        effective_limit, retry = record_oom(
                            job, gpu, retries, events, effective_limit, gpu_limits)
                        state("oom_retry_pending" if retry else "oom_retry_exhausted")
                        if not retry:
                            raise RuntimeError(f"CUDA OOM retry limit exhausted: {job['name']}")
                        continue
                    queue.validate_job(args.output / job["name"], job)
                    completed.add(job["name"])
                    if job["purpose"] == "main":
                        queue.summarize(args.output, plan, completed)
                if len(completed) == len(plan["jobs"]):
                    queue.summarize(args.output, plan, completed)
                    state("complete")
                    return
                snapshot = queue.gpu_snapshot()
                if set(args.gpus) - snapshot.keys():
                    raise RuntimeError("Requested physical GPU IDs do not exist")
                eligible = queue.next_jobs(plan["jobs"], completed,
                                           {r["job"]["group"] for r in active.values()}, ready_models)
                for job in eligible:
                    check_interrupt()
                    if retries.get(job["name"], 0) > MAX_OOM_RETRIES:
                        raise RuntimeError(f"CUDA OOM retry limit exhausted: {job['name']}")
                    if len(active) >= effective_limit:
                        break
                    for gpu in args.gpus:
                        if not slots_available(snapshot[gpu], args, active,
                                               effective_limit, gpu_limits):
                            continue
                        new_lease = gpu not in leases
                        if new_lease:
                            lease = (args.output.parent / f"alfworld_gpu{gpu}.lock").open("a")
                            try:
                                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            except BlockingIOError:
                                lease.close()
                                continue
                            leases[gpu] = lease
                        launched = False
                        try:
                            with maintenance_launch_lock(gpu):
                                if maintenance_blockers(maintenance_path(gpu)):
                                    continue
                                devices = queue.gpu_snapshot()
                                device = devices[gpu]
                                if (not slots_available(device, args, active,
                                                        effective_limit, gpu_limits)
                                        or not available(device, args, active, devices=devices)
                                        or assigned_blocker(gpu, device["uuid"], args, active, device,
                                                            devices=devices)):
                                    continue
                                directory = args.output / job["name"]
                                directory.mkdir(parents=True, exist_ok=True)
                                log = (directory / "run.log").open("ab")
                                log_offset = log.tell()
                                worker_lease = os.fdopen(os.dup(leases[gpu].fileno()), "a")
                                env = dict(
                                    os.environ, CUDA_VISIBLE_DEVICES=device["uuid"],
                                    PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
                                    HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                                    TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4",
                                    MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                                    TMPDIR=str(args.tmp_root),
                                )
                                env.update({k: str(v) for k, v in cache_paths.items()})
                                started = time.time()
                                try:
                                    process = subprocess.Popen(
                                        queue.command(args, job), cwd=ROOT, env=env,
                                        stdout=log, stderr=subprocess.STDOUT,
                                        pass_fds=(worker_lease.fileno(), queue_lock.fileno()))
                                except BaseException:
                                    worker_lease.close()
                                    log.close()
                                    raise
                                slot += 1
                                active[slot] = dict(
                                    job=job, gpu=gpu, process=process, worker=process_identity(process.pid),
                                    log=log, lease=worker_lease, started=started, log_offset=log_offset,
                                    startup_ready=False,
                                )
                                launched = True
                                peak = max(peak, len(active))
                                check_interrupt()
                                print(f"Started slot{slot} GPU{gpu}: {job['name']}; "
                                      f"concurrency {len(active)}/{effective_limit}", flush=True)
                                state("running")
                            break
                        finally:
                            if new_lease and not launched:
                                leases.pop(gpu).close()
                missing = sorted(set(args.model_readys) - ready_models)
                state("running" if active else "waiting_for_model_assets" if missing
                      else "waiting_for_gpu", models_waiting_for_assets=missing)
                time.sleep(args.poll_seconds)
        except BaseException as error:
            queue.stop_workers(active)
            active.clear()
            state("failed", error=str(error), traceback=traceback.format_exc())
            raise
        finally:
            for lease in leases.values():
                lease.close()
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model-ready", "data-ready", "output", "tmp-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, choices=[5], default=5)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--min-free-gib", type=float, choices=[32], default=32)
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Controller must be CPU-only: CUDA_VISIBLE_DEVICES=''")
    if args.gpus != [0, 1, 2] or args.poll_seconds <= 0:
        parser.error("Frozen GPUs must be [0,1,2]; require positive polling")
    args.output, args.tmp_root, args.data_ready = (
        args.output.resolve(), args.tmp_root.resolve(), args.data_ready.resolve())
    args.model_readys = {"llama31_8b": args.model_ready.resolve()}
    if args.output.parent != BASE / "results/shared_accuracy":
        parser.error("Require the existing dedicated shared-accuracy namespace")
    execute(args)


if __name__ == "__main__":
    main()
