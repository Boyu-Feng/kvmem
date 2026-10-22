"""Supplement component ablations on available shared GPUs without stopping loads."""
from __future__ import annotations

import argparse
import csv
import fcntl
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

from scripts.run_qa32_rebuttal import atomic_json, file_hash, read_json
from scripts.run_rebuttal_efficiency_queue import (
    identity_alive, maintenance_blockers, maintenance_launch_lock, maintenance_path,
    process_identity,
)

WORKER = ROOT / "scripts/run_component_ablation_rebuttal.py"
VARIANTS = ("full", "no_repeat", "no_novelty", "no_success", "no_cite", "no_token", "no_step")
DATASETS = ("hotpotqa", "2wiki", "musique")
SEEDS = (233, 3407)
MODEL_REPOS = {
    "qwen25_7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama31_8b": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}
LOAD_SCRIPT = Path("/mnt/3fs/models/bruli/gpu-utilization-20260925/gpu_load.py")


def jobs(models):
    return [
        {
            "name": f"{model}/seed{seed}/{dataset}/{purpose}/{variant}",
            "group": f"{model}/seed{seed}/{dataset}",
            "model": model, "seed": seed, "dataset": dataset,
            "purpose": purpose, "variant": variant,
            "samples": 2 if purpose == "smoke" else 500,
            "ratio": .5, "max_steps": 7,
        }
        for model in models
        for seed in SEEDS
        for dataset in DATASETS
        for purpose in ("smoke", "main")
        for variant in VARIANTS
    ]


def source_gate():
    from scripts.run_component_ablation_rebuttal import source_hashes

    return {
        **source_hashes(),
        "scripts/run_component_ablation_queue.py": file_hash(Path(__file__)),
        "scripts/run_rebuttal_efficiency_queue.py":
            file_hash(ROOT / "scripts/run_rebuttal_efficiency_queue.py"),
    }


def gpu_snapshot():
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.free",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=20)
    devices = {}
    for row in csv.reader(raw.splitlines()):
        index, uuid, free = (value.strip() for value in row)
        devices[int(index)] = {"index": int(index), "uuid": uuid, "free_mib": int(free), "pids": []}
    raw = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=20)
    by_uuid = {device["uuid"]: device for device in devices.values()}
    for row in csv.reader(raw.splitlines()):
        uuid, pid = (value.strip() for value in row)
        if uuid in by_uuid:
            by_uuid[uuid]["pids"].append(int(pid))
    return devices


def own_load(pid, gpu, proc=Path("/proc")):
    directory = proc / str(pid)
    try:
        if directory.stat().st_uid != os.getuid():
            return False
        command = directory.joinpath("cmdline").read_bytes().rstrip(b"\0").split(b"\0")
    except FileNotFoundError:
        return False
    args = [os.fsdecode(part) for part in command]
    return (
        len(args) == 6
        and Path(args[0]).name in ("python3", "python")
        and args[1:] == ["-B", str(LOAD_SCRIPT), "worker", "--gpu", str(gpu)]
    )


def available(device, minimum_gib):
    return (
        device["free_mib"] >= minimum_gib * 1024
        and all(own_load(pid, device["index"]) for pid in device["pids"])
    )


def assigned_experiment(gpu, uuid, proc=Path("/proc")):
    """Reject model-loading processes that have not created a CUDA context yet."""
    for directory in proc.iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            if directory.stat().st_uid != os.getuid():
                continue
            command = directory.joinpath("cmdline").read_bytes().rstrip(b"\0").split(b"\0")
            if not command or "python" not in Path(os.fsdecode(command[0])).name:
                continue
            environment = directory.joinpath("environ").read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        except PermissionError:
            return True
        visible = [
            part.split(b"=", 1)[1].decode()
            for part in environment if part.startswith(b"CUDA_VISIBLE_DEVICES=")
        ]
        if any(str(gpu) in value.split(",") or uuid in value.split(",") for value in visible):
            if not own_load(int(directory.name), gpu, proc):
                return True
    return False


def next_jobs(plan_jobs, completed, running_groups, ready_models):
    """Keep each model/seed/dataset group serial, including its smoke gates."""
    first = {}
    for job in plan_jobs:
        if job["name"] not in completed:
            first.setdefault(job["group"], job)
    return [
        job for group, job in first.items()
        if group not in running_groups and job["model"] in ready_models
    ]


def validate_job(directory, job):
    from scripts.run_component_ablation_rebuttal import validate_completed_run

    record = validate_completed_run(directory)
    fingerprint = read_json(directory / "manifest.json")["fingerprint"]["protocol"]
    for key in ("dataset", "seed", "variant", "purpose", "samples", "ratio", "max_steps"):
        if fingerprint[key] != job[key]:
            raise RuntimeError(f"Completed ablation differs in {key}: {directory}")
    if record["completed_samples"] != job["samples"]:
        raise RuntimeError(f"Wrong completed sample count: {directory}")
    if job["purpose"] == "smoke" and not record["has_actual_eviction"]:
        raise RuntimeError(f"Smoke did not exercise actual KV eviction: {directory}")
    return record


def summarize(output, plan, completed):
    results, identities = [], {}
    for job in plan["jobs"]:
        if job["purpose"] != "main" or job["name"] not in completed:
            continue
        record = validate_job(output / job["name"], job)
        expected = identities.setdefault(job["group"], record["sample_ids"])
        if expected != record["sample_ids"]:
            raise RuntimeError(f"Ablations do not use identical ordered IDs: {job['group']}")
        results.append({**job, **record})
    atomic_json(output / "SUMMARY.json", {
        "completed_main": len(results),
        "expected_main": sum(job["purpose"] == "main" for job in plan["jobs"]),
        "results": results,
        "timing_policy": plan["timing_policy"],
        "statistics": "New seeds233/3407 only. Historical seed42 run provenance is not verified; "
                      "do not compute a three-seed mean until its actual results and IDs are validated.",
    })


def command(args, job):
    return [
        sys.executable, "-u", str(WORKER), "--execute",
        "--model-ready", str(args.model_readys[job["model"]]),
        "--data-ready", str(args.data_ready),
        "--output", str(args.output / job["name"]),
        "--dataset", job["dataset"], "--seed", str(job["seed"]),
        "--variant", job["variant"], "--purpose", job["purpose"],
    ]


def stop_workers(active):
    for record in active.values():
        if record["process"].poll() is None:
            record["process"].terminate()
    for record in active.values():
        process = record["process"]
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
        record["log"].close()
        record["lease"].close()


def execute(args):
    args.output.mkdir(parents=True, exist_ok=True)
    args.tmp_root.mkdir(parents=True, exist_ok=True)
    cache_paths = {
        "HF_HOME": args.tmp_root.parent / "hf",
        "HF_DATASETS_CACHE": args.tmp_root.parent / "datasets",
        "TORCH_HOME": args.tmp_root.parent / "torch",
        "XDG_CACHE_HOME": args.tmp_root.parent / "xdg",
        "TRITON_CACHE_DIR": args.tmp_root.parent / "triton",
    }
    for path in cache_paths.values():
        path.mkdir(parents=True, exist_ok=True)
    plan = {
        "protocol": "component_ablation_original_shared_gpu_v1",
        "model_readys": {key: str(value) for key, value in args.model_readys.items()},
        "data_ready": str(args.data_ready), "gpus": args.gpus, "max_workers": args.max_workers,
        "min_free_gib": args.min_free_gib, "tmp_root": str(args.tmp_root),
        "jobs": jobs(args.model_readys), "source_gate": source_gate(),
        "timing_policy": "Shared GPU accuracy only; existing load workers are never stopped. "
                         "Timings are diagnostic, not latency measurements.",
        "full_control": "full means full StepKV, not FullKV; new same-seed controls are run.",
    }
    owner = process_identity(os.getpid())
    active, completed, asset_pins = {}, set(), {}

    def state(status, **extra):
        atomic_json(args.output / "queue_status.json", {
            "status": status, "owner": owner, "updated": time.time(),
            "completed_jobs": len(completed), "total_jobs": len(plan["jobs"]),
            "completed_main": sum(job["purpose"] == "main" and job["name"] in completed
                                  for job in plan["jobs"]),
            "active": [
                {"job": record["job"], "gpu": gpu, "worker": process_identity(record["process"].pid)}
                for gpu, record in active.items()
            ], **extra,
        })

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    with (args.output / "queue.lock").open("a") as queue_lock:
        fcntl.flock(queue_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = args.output / "queue_plan.json"
        if path.exists() and read_json(path) != plan:
            raise RuntimeError("Changed queue/source protocol; use a new output directory")
        status_path = args.output / "queue_status.json"
        if status_path.exists():
            previous = read_json(status_path)
            if any(identity_alive(record.get("worker")) for record in previous.get("active", [])):
                raise RuntimeError("Previously launched ablation workers are still alive")
        atomic_json(path, plan)
        try:
            for job in plan["jobs"]:
                status = args.output / job["name"] / "status.json"
                if status.exists() and read_json(status).get("status") == "complete":
                    validate_job(status.parent, job)
                    completed.add(job["name"])
            summarize(args.output, plan, completed)
            while len(completed) != len(plan["jobs"]):
                if source_gate() != plan["source_gate"]:
                    raise RuntimeError("Source changed during the ablation queue")
                for gpu, record in list(active.items()):
                    code = record["process"].poll()
                    if code is None:
                        continue
                    record["log"].close()
                    record["lease"].close()
                    del active[gpu]
                    if code:
                        raise RuntimeError(f"Ablation failed ({code}): {record['job']['name']}")
                    validate_job(args.output / record["job"]["name"], record["job"])
                    completed.add(record["job"]["name"])
                    if record["job"]["purpose"] == "main":
                        summarize(args.output, plan, completed)

                ready_models = {key for key, path in args.model_readys.items() if path.exists()}
                for model in ready_models:
                    if read_json(args.model_readys[model]).get("repo") != MODEL_REPOS[model]:
                        raise RuntimeError(f"Model ready metadata is not the requested model: {model}")
                    model_pin = file_hash(args.model_readys[model])
                    pin_path = args.output / f"{model}_asset_pin.json"
                    if model not in asset_pins:
                        if pin_path.exists() and read_json(pin_path)["sha256"] != model_pin:
                            raise RuntimeError(f"Model ready metadata changed: {model}")
                        asset_pins[model] = model_pin
                        atomic_json(pin_path, {"sha256": model_pin, "path": str(args.model_readys[model])})
                    elif asset_pins[model] != model_pin:
                        raise RuntimeError(f"Model ready metadata changed during queue: {model}")
                snapshot = gpu_snapshot()
                if set(args.gpus) - snapshot.keys():
                    raise RuntimeError("Requested GPU IDs do not exist on this node")
                eligible = next_jobs(
                    plan["jobs"], completed, {record["job"]["group"] for record in active.values()},
                    ready_models,
                )
                for job in eligible:
                    if len(active) >= args.max_workers:
                        break
                    for gpu in args.gpus:
                        if (gpu in active or not available(snapshot[gpu], args.min_free_gib)
                                or assigned_experiment(gpu, snapshot[gpu]["uuid"])):
                            continue
                        if maintenance_blockers(maintenance_path(gpu)):
                            continue
                        lease = (args.output.parent / f"alfworld_gpu{gpu}.lock").open("a")
                        try:
                            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        except BlockingIOError:
                            lease.close()
                            continue
                        try:
                            with maintenance_launch_lock(gpu):
                                if maintenance_blockers(maintenance_path(gpu)):
                                    lease.close()
                                    continue
                                device = gpu_snapshot()[gpu]
                                if (not available(device, args.min_free_gib)
                                        or assigned_experiment(gpu, device["uuid"])):
                                    lease.close()
                                    continue
                                directory = args.output / job["name"]
                                directory.mkdir(parents=True, exist_ok=True)
                                log = (directory / "run.log").open("a")
                                environment = dict(
                                    os.environ, CUDA_VISIBLE_DEVICES=device["uuid"],
                                    PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1",
                                    HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                                    TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4",
                                    MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                                    TMPDIR=str(args.tmp_root),
                                )
                                environment.update({key: str(path) for key, path in cache_paths.items()})
                                try:
                                    process = subprocess.Popen(
                                        command(args, job), cwd=ROOT, env=environment,
                                        stdout=log, stderr=subprocess.STDOUT,
                                        pass_fds=(lease.fileno(), queue_lock.fileno()),
                                    )
                                except BaseException:
                                    log.close()
                                    raise
                                active[gpu] = dict(process=process, lease=lease, log=log, job=job)
                                print(f"Started GPU{gpu}: {job['name']} (load unchanged)", flush=True)
                            break
                        except BaseException:
                            lease.close()
                            raise
                missing = sorted(set(args.model_readys) - ready_models)
                state("running" if active else "waiting_for_model_assets" if missing else "waiting_for_gpu",
                      models_waiting_for_assets=missing)
                if len(completed) != len(plan["jobs"]):
                    time.sleep(args.poll_seconds)
            summarize(args.output, plan, completed)
            state("complete")
        except BaseException as error:
            stop_workers(active)
            active.clear()
            state("failed", error=str(error), traceback=traceback.format_exc())
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--qwen-ready", type=Path, required=True)
    parser.add_argument("--llama-ready", type=Path)
    parser.add_argument("--data-ready", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tmp-root", type=Path, required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--min-free-gib", type=float, default=48)
    parser.add_argument("--poll-seconds", type=float, default=10)
    args = parser.parse_args()
    if (len(set(args.gpus)) != len(args.gpus) or not args.gpus or min(args.gpus) < 0
            or not 1 <= args.max_workers <= len(args.gpus)
            or args.min_free_gib < 48 or args.poll_seconds <= 0):
        parser.error("Require distinct GPU IDs, positive polling, 1..N workers and >=48GiB free")
    args.model_readys = {"qwen25_7b": args.qwen_ready.resolve()}
    if args.llama_ready:
        args.model_readys["llama31_8b"] = args.llama_ready.resolve()
    args.data_ready, args.output, args.tmp_root = (
        args.data_ready.resolve(), args.output.resolve(), args.tmp_root.resolve())
    if not args.execute:
        print(f"Planned {len(jobs(args.model_readys))} jobs; no GPU work or load interruption.")
        return
    execute(args)


if __name__ == "__main__":
    main()
