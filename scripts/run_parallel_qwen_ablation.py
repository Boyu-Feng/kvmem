"""Resume only Qwen component jobs with shared-aware, smoke/full-gated scheduling.

The original queue_plan.json is never rewritten. Four worker slots use physical
GPUs 0,1,2,7, at most two per card and >=32GiB real free memory. This launcher
neither stops matrix loads nor restores them; the external deadline guard owns
that lifecycle. Frozen worker commands, checkpoints, scoring and caches remain.
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
from scripts.run_qa32_rebuttal import atomic_json, file_hash, read_json
from scripts.run_rebuttal_efficiency_queue import (
    identity_alive, maintenance_blockers, maintenance_launch_lock, maintenance_path,
    process_identity,
)

BASE = Path("/mnt/checkpoints/bruli/fengboyu/kvmem_rebuttal_20260929")
QWEN_ROOT = BASE / "results/component_ablation_seeds233_3407_20260930_v1"
LLAMA_ROOT = BASE / "results/shared_accuracy/llama_component_ablation_seeds233_3407_v1"
SENS_ROOT = BASE / "results/sensitivity_shared/qwen7b_seed233_v1"
QWEN_MODEL = BASE / "assets/7b_ready.json"
LLAMA_MODEL = BASE / "ablation_assets/llama31_8b_shared_ready.json"
DATA_READY = BASE / "qa32_assets/data_ready.json"
SENS_WORKER = ROOT / "scripts/run_stepkv_sensitivity.py"
QA32_WORKER = ROOT / "scripts/run_qa32_rebuttal.py"
MAX_OOM_RETRIES = 2
CUDA_OOM = re.compile(
    r"CUDA out of memory|CUDA error:\s*out of memory|torch\.cuda\.OutOfMemoryError",
    re.IGNORECASE,
)


def flag(command, name):
    if command.count(name) != 1:
        return None
    index = command.index(name) + 1
    return command[index] if index < len(command) else None


def process_args(pid, proc=Path("/proc")):
    directory = proc / str(pid)
    try:
        if directory.stat().st_uid != os.getuid():
            return None
        return [os.fsdecode(part) for part in
                directory.joinpath("cmdline").read_bytes().rstrip(b"\0").split(b"\0")]
    except FileNotFoundError:
        return None


def runner(command):
    if not command or not Path(command[0]).name.startswith("python"):
        return None
    index = 2 if len(command) > 1 and command[1] in ("-u", "-B") else 1
    return command[index] if len(command) > index else None


def known_qa32(command):
    if runner(command) != str(QA32_WORKER):
        return False
    assets = str(BASE / "qa32_assets")
    if flag(command, "--assets") != assets or flag(command, "--memory-limit-gib") not in (
            "44", "44.0"):
        return False
    output = flag(command, "--output")
    for dataset in queue.DATASETS:
        root = BASE / f"results/qa32_{dataset}_v1"
        for purpose in ("smoke", "main"):
            for method in ("fullkv", "stepkv", "h2o", "tova", "h2o_step", "tova_step"):
                for label in (("r100",) if method == "fullkv" else ("r20", "r50")):
                    if output != str(root / f"{dataset}/{purpose}/{method}/{label}"):
                        continue
                    fields = {
                        "dataset": dataset, "method": method,
                        "ratio": "0.2" if label == "r20" else "0.5", "purpose": purpose,
                        "samples": "2" if purpose == "smoke" else "500",
                        "sample_start": "500" if purpose == "smoke" else "0",
                        "seed": "233", "max_steps": "7",
                    }
                    expected = [
                        "-u", str(QA32_WORKER), "--assets", assets, "--output", output,
                        "--memory-limit-gib", flag(command, "--memory-limit-gib"),
                    ]
                    for key, value in fields.items():
                        expected.extend(["--" + key.replace("_", "-"), value])
                    return command[1:] == expected
    return False


def known_peer(pid, proc=Path("/proc")):
    """Allow only exact same-UID frozen accuracy commands and official scopes."""
    command = process_args(pid, proc)
    script = runner(command)
    if script == str(QA32_WORKER):
        return known_qa32(command)
    if script not in (str(queue.WORKER), str(SENS_WORKER)) or "--execute" not in command:
        return False
    if flag(command, "--data-ready") != str(DATA_READY):
        return False
    output = flag(command, "--output")
    if not output:
        return False
    if script == str(queue.WORKER):
        for model, root, ready in (
            ("qwen25_7b", QWEN_ROOT, QWEN_MODEL),
            ("llama31_8b", LLAMA_ROOT, LLAMA_MODEL),
        ):
            if flag(command, "--model-ready") != str(ready):
                continue
            for job in queue.jobs({model: ready}):
                if output == str(root / job["name"]):
                    peer_args = argparse.Namespace(
                        model_readys={model: ready}, data_ready=DATA_READY, output=root)
                    return command[1:] == queue.command(peer_args, job)[1:]
        return False
    if flag(command, "--model-ready") != str(QWEN_MODEL):
        return False
    try:
        parts = Path(output).relative_to(SENS_ROOT).parts
    except ValueError:
        return False
    if (len(parts) != 6 or parts[0] != "qwen25_7b" or parts[1] != "seed233"
            or parts[2] not in queue.DATASETS or parts[3] not in ("r20", "r50")
            or parts[4] not in ("smoke", "main") or parts[5] in (".", "..")):
        return False
    expected = [
        "-u", str(SENS_WORKER), "--execute", "--model-ready", str(QWEN_MODEL),
        "--data-ready", str(DATA_READY), "--output", output, "--dataset", parts[2],
        "--seed", "233", "--variant", "full", "--purpose", parts[4],
        "--case", parts[5], "--ratio", "0.2" if parts[3] == "r20" else "0.5",
    ]
    return command[1:] == expected


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


def qa32_contexts_ready(pid, devices, proc=Path("/proc")):
    if devices is None:
        return False
    try:
        environment = (proc / str(pid) / "environ").read_bytes().split(b"\0")
        visible = [part.split(b"=", 1)[1].decode() for part in environment
                   if part.startswith(b"CUDA_VISIBLE_DEVICES=")]
        if len(visible) != 1:
            return False
        selectors = visible[0].split(",")
        by_uuid = {device["uuid"]: index for index, device in devices.items()}
        pair = tuple(int(value) if value.isdigit() else by_uuid.get(value)
                     for value in selectors)
        return (pair in ((3, 4), (5, 6), (0, 7))
                and all(gpu in devices and pid in devices[gpu]["pids"] for gpu in pair))
    except (FileNotFoundError, PermissionError):
        return False


def steady_peer(pid, device, proc=Path("/proc"), devices=None):
    if pid not in device["pids"] or not known_peer(pid, proc):
        return False
    command = process_args(pid, proc)
    directory = Path(flag(command, "--output"))
    if runner(command) == str(QA32_WORKER):
        try:
            status = read_json(directory / "status.json")
            return (status.get("status") == "running" and status.get("pid") == pid
                    and qa32_contexts_ready(pid, devices, proc))
        except FileNotFoundError:
            return False
    return loaded_status(directory, pid)


def assigned_blocker(gpu, uuid, args, active, device, proc=Path("/proc"), devices=None):
    """Skip CPU/non-owned processes before environ; CUDA snapshots still police all UIDs."""
    for directory in proc.iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            command = process_args(int(directory.name), proc)
            if runner(command) is None:
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
            if steady_peer(pid, device, proc, devices):
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
                    or steady_peer(pid, device, proc, devices) for pid in device["pids"]))


def slots_available(device, args, active, effective_limit, gpu_limits):
    records = [r for r in active.values() if r["gpu"] == device["index"]]
    for record in records:
        if (not record["startup_ready"] and record["process"].pid in device["pids"]
                and loaded_status(args.output / record["job"]["name"],
                                  record["process"].pid, record["started"])):
            record["startup_ready"] = True
    return (len(active) < min(4, args.max_workers, effective_limit)
            and len(records) < min(2, gpu_limits.get(device["index"], 2))
            and all(r["startup_ready"] for r in records))


def next_jobs(jobs, completed, active):
    """Each variant needs its smoke; ablation mains also need same-group full."""
    running = {r["job"]["name"] for r in active.values()}
    eligible = []
    for job in jobs:
        if job["name"] in completed or job["name"] in running:
            continue
        if job["purpose"] == "main":
            if f"{job['group']}/smoke/{job['variant']}" not in completed:
                continue
            if job["variant"] != "full" and f"{job['group']}/main/full" not in completed:
                continue
        eligible.append(job)
    # Finish ready formal work before remaining smoke work; plan order stays frozen.
    return sorted(eligible, key=lambda job: job["purpose"] != "main")


def prepare_plan(args):
    """Under queue.lock, pin original bytes and record a separate Qwen-only scope."""
    original_path = args.output / "queue_plan.json"
    original = read_json(original_path)
    models = original.get("model_readys", {})
    if (set(models) != {"qwen25_7b", "llama31_8b"}
            or models["qwen25_7b"] != str(args.model_readys["qwen25_7b"])
            or original.get("jobs") != queue.jobs(models)
            or original.get("data_ready") != str(args.data_ready)
            or original.get("tmp_root") != str(args.tmp_root)
            or original.get("protocol") != "component_ablation_original_shared_gpu_v1"
            or original.get("source_gate") != queue.source_gate()):
        raise RuntimeError("Frozen Qwen plan/source/assets do not match; no migration permitted")
    previous_path = args.output / "queue_status.json"
    previous = read_json(previous_path) if previous_path.exists() else {}
    if identity_alive(previous.get("owner")) or any(
            identity_alive(r.get("worker")) for r in previous.get("active", [])):
        raise RuntimeError("Previous controller or worker is still alive")
    jobs = [job for job in original["jobs"] if job["model"] == "qwen25_7b"]
    if jobs != queue.jobs(args.model_readys):
        raise RuntimeError("Qwen job scope changed")
    plan = {
        "protocol": "qwen_shared_parallel_deadline_v1",
        "original_plan_sha256": file_hash(original_path),
        "launcher_sha256": file_hash(Path(__file__)),
        "source_gate": original["source_gate"], "jobs": jobs,
        "model_readys": {"qwen25_7b": models["qwen25_7b"]},
        "data_ready": original["data_ready"], "tmp_root": original["tmp_root"],
        "gpus": args.gpus, "max_workers": args.max_workers, "per_gpu_workers": 2,
        "min_free_gib": args.min_free_gib, "timing_policy": original["timing_policy"],
        "transferred_models_not_completed_here": ["llama31_8b"],
        "lease_directory": str(args.lease_dir),
    }
    plan_path = args.output / "parallel_qwen_plan.json"
    if plan_path.exists() and read_json(plan_path) != plan:
        raise RuntimeError("Qwen scheduling provenance changed")
    atomic_json(plan_path, plan)
    return plan, previous


def is_cuda_oom(directory, offset, started):
    with (directory / "run.log").open("rb") as handle:
        end = handle.seek(0, os.SEEK_END)
        handle.seek(max(offset, end - 65536))
        text = handle.read().decode(errors="replace")
    path = directory / "status.json"
    if path.exists() and path.stat().st_mtime >= started:
        text += "\n" + str(read_json(path).get("reason", ""))
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
    active, completed, leases = {}, set(), {}
    owner = process_identity(os.getpid())
    with (args.output / "queue.lock").open("a+") as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan, previous = prepare_plan(args)
        retries = dict(previous.get("oom_retries", {}))
        events = list(previous.get("oom_events", []))
        effective_limit = min(4, previous.get("effective_max_workers", 4))
        gpu_limits = {int(k): int(v) for k, v in previous.get("gpu_worker_limits", {}).items()}
        peak = previous.get("peak_active_workers", 0)
        model_pin, slot, stop_requested = None, 0, None
        snapshot, eligible = {}, []

        def state(status, **extra):
            atomic_json(args.output / "queue_status.json", {
                "status": status, "owner": owner, "updated": time.time(),
                "scope": "qwen25_7b_only", "completed_jobs": len(completed),
                "total_jobs": len(plan["jobs"]),
                "completed_main": sum(j["purpose"] == "main" and j["name"] in completed
                                      for j in plan["jobs"]),
                "expected_main": sum(j["purpose"] == "main" for j in plan["jobs"]),
                "active": [{"job": r["job"], "gpu": r["gpu"], "worker": r["worker"],
                            "slot": key, "startup_ready": r["startup_ready"]}
                           for key, r in active.items()],
                "requested_max_workers": 4, "effective_max_workers": effective_limit,
                "effective_concurrency": len(active), "peak_active_workers": peak,
                "gpu_worker_limits": gpu_limits, "oom_retries": retries, "oom_events": events,
                "runnable_jobs": [j["name"] for j in eligible
                                  if j["name"] not in {r["job"]["name"] for r in active.values()}],
                "gpu_snapshot": snapshot,
                "progress": {
                    r["job"]["name"]: read_json(args.output / r["job"]["name"] / "status.json")
                    if (args.output / r["job"]["name"] / "status.json").exists() else
                    {"status": "starting"} for r in active.values()
                }, **extra,
            })

        def interrupted(signum, frame):
            nonlocal stop_requested
            stop_requested = signum

        def check_interrupt():
            if stop_requested is not None:
                raise KeyboardInterrupt(f"Received signal {stop_requested}; stopping own workers")

        old_handlers = {sig: signal.signal(sig, interrupted)
                        for sig in (signal.SIGTERM, signal.SIGINT)}
        try:
            args.tmp_root.mkdir(parents=True, exist_ok=True)
            args.lease_dir.mkdir(parents=True, exist_ok=True)
            caches = {key: args.tmp_root.parent / value for key, value in (
                ("HF_HOME", "hf"), ("HF_DATASETS_CACHE", "datasets"), ("TORCH_HOME", "torch"),
                ("XDG_CACHE_HOME", "xdg"), ("TRITON_CACHE_DIR", "triton"))}
            for path in caches.values():
                path.mkdir(parents=True, exist_ok=True)
            for job in plan["jobs"]:
                path = args.output / job["name"] / "status.json"
                if path.exists() and read_json(path).get("status") == "complete":
                    queue.validate_job(path.parent, job)
                    completed.add(job["name"])
            for job in plan["jobs"]:
                if job["purpose"] == "main" and job["name"] in completed:
                    if f"{job['group']}/smoke/{job['variant']}" not in completed:
                        raise RuntimeError(f"Completed main lacks validated smoke: {job['name']}")
                    if (job["variant"] != "full"
                            and f"{job['group']}/main/full" not in completed):
                        raise RuntimeError(f"Completed main lacks full baseline: {job['name']}")
            queue.summarize(args.output, plan, completed)
            while True:
                check_interrupt()
                if (queue.source_gate() != plan["source_gate"]
                        or file_hash(Path(__file__)) != plan["launcher_sha256"]
                        or file_hash(args.output / "queue_plan.json") != plan["original_plan_sha256"]
                        or read_json(args.output / "parallel_qwen_plan.json") != plan):
                    raise RuntimeError("Frozen source/plan or scheduler provenance changed")
                model_path = args.model_readys["qwen25_7b"]
                if model_path.exists():
                    if read_json(model_path).get("repo") != queue.MODEL_REPOS["qwen25_7b"]:
                        raise RuntimeError("Wrong Qwen model assets")
                    pin = file_hash(model_path)
                    pin_path = args.output / "qwen25_7b_asset_pin.json"
                    if model_pin is None:
                        if pin_path.exists() and read_json(pin_path)["sha256"] != pin:
                            raise RuntimeError("Qwen model asset pin changed")
                        model_pin = pin
                        if not pin_path.exists():
                            atomic_json(pin_path, {"sha256": pin, "path": str(model_path)})
                    elif model_pin != pin:
                        raise RuntimeError("Qwen model asset pin changed during queue")
                elif model_pin is not None:
                    raise RuntimeError("Pinned Qwen model metadata disappeared")
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
                eligible = next_jobs(plan["jobs"], completed, active) if model_pin else []
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
                            lease = (args.lease_dir / f"alfworld_gpu{gpu}.lock").open("a")
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
                                        or assigned_blocker(gpu, device["uuid"], args, active,
                                                            device, devices=devices)):
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
                                env.update({key: str(path) for key, path in caches.items()})
                                started = time.time()
                                try:
                                    process = subprocess.Popen(
                                        queue.command(args, job), cwd=ROOT, env=env, stdout=log,
                                        stderr=subprocess.STDOUT,
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
                                print(f"Started Qwen slot{slot} GPU{gpu}: {job['name']}; "
                                      f"concurrency {len(active)}/{effective_limit}", flush=True)
                                state("running")
                            break
                        finally:
                            if new_lease and not launched:
                                leases.pop(gpu).close()
                state("running" if active else "waiting_for_gpu" if model_pin
                      else "waiting_for_model_assets")
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
    parser.add_argument("--max-workers", type=int, choices=[4], default=4)
    parser.add_argument("--gpus", nargs="+", type=int, default=[0, 1, 2, 7])
    parser.add_argument("--min-free-gib", type=float, choices=[32], default=32)
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        parser.error("Controller must be CPU-only: CUDA_VISIBLE_DEVICES=''")
    if args.gpus != [0, 1, 2, 7] or args.poll_seconds <= 0:
        parser.error("Use physical GPUs [0,1,2,7] and positive polling")
    args.output, args.tmp_root, args.data_ready = (
        args.output.resolve(), args.tmp_root.resolve(), args.data_ready.resolve())
    args.model_readys = {"qwen25_7b": args.model_ready.resolve()}
    if (args.output != QWEN_ROOT or args.model_readys["qwen25_7b"] != QWEN_MODEL
            or args.data_ready != DATA_READY or args.tmp_root != BASE / "ablation_caches/tmp"):
        parser.error("Use the original Qwen output, model, data and cache paths")
    args.lease_dir = args.output.parent / "shared_accuracy/qwen_component_parallel_leases"
    execute(args)


if __name__ == "__main__":
    main()
