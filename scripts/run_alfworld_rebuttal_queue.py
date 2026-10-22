"""Run matched ALFWorld accuracy conditions without claiming exclusive-GPU latency."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_rebuttal_legacy import CORE_FILES, atomic_json
from scripts.prepare_alfworld_rebuttal_assets import MODELS, sha256
from scripts.alfworld_rebuttal_env import Game, RELEASE_GAME_COUNTS
from scripts.run_alfworld_rebuttal import CheckpointStore, cache_summary
from scripts.run_rebuttal_efficiency_queue import identity_alive, process_identity

RUNNER = ROOT / "scripts/run_alfworld_rebuttal.py"
CONDITIONS = (
    ("fullkv", .5, "r100"),
    ("stepkv", .2, "r20"),
    ("stepkv", .5, "r50"),
    ("h2o_step", .2, "r20"),
    ("h2o_step", .5, "r50"),
)


def source_hashes() -> dict[str, str]:
    paths = [
        Path(__file__), RUNNER, ROOT / "models/AgentKVPolicy.py",
        ROOT / "kv_cache/anchor_utils.py",
        ROOT / "scripts/prepare_alfworld_rebuttal_assets.py",
        ROOT / "config/environments/h200_accuracy_20260929.requirements.txt",
    ]
    optional = ROOT / "scripts/alfworld_rebuttal_env.py"
    if optional.exists():
        paths.append(optional)
    paths.extend(ROOT / name for name in CORE_FILES)
    for directory in ("models", "kv_cache"):
        paths.extend((ROOT / directory).rglob("*.py"))
    paths.extend([
        ROOT / "run_rebuttal_legacy.py",
        ROOT / "scripts/run_rebuttal_efficiency_queue.py",
    ])
    return {str(path.relative_to(ROOT)): sha256(path) for path in paths}


def jobs() -> list[dict]:
    return [
        {
            "name": f"{purpose}/{method}/{label}", "purpose": purpose,
            "method": method, "ratio": ratio,
            "split": "eval_in_distribution" if purpose == "smoke" else "eval_out_of_distribution",
            "samples": 2 if purpose == "smoke" else 0,
            "max_steps": 8 if purpose == "smoke" else 50,
        }
        for purpose in ("smoke", "main")
        for method, ratio, label in CONDITIONS
    ]


def gpu_snapshot(gpus: list[int]) -> list[dict]:
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=20)
    devices = {}
    for line in raw.strip().splitlines():
        index, uuid, total, free, utilization = [part.strip() for part in line.split(",")]
        devices[int(index)] = {
            "index": int(index), "uuid": uuid, "total_mib": int(total),
            "free_mib": int(free), "utilization_percent": int(utilization),
        }
    missing = set(gpus) - devices.keys()
    if missing:
        raise RuntimeError(f"Requested GPUs do not exist: {sorted(missing)}")
    return [devices[index] for index in gpus]


def enough_memory(devices: list[dict], minimum_gib: float) -> bool:
    return bool(devices) and all(device["free_mib"] >= minimum_gib * 1024 for device in devices)


def validate_result(directory: Path, job: dict) -> int:
    status = json.loads((directory / "status.json").read_text())
    if status.get("status") != "complete":
        raise RuntimeError(f"Worker did not complete successfully: {directory}")
    manifest = json.loads((directory / "manifest.json").read_text())
    protocol = manifest["protocol"]
    for key in ("method", "ratio", "purpose", "split", "samples"):
        if protocol[key] != job[key]:
            raise RuntimeError(f"Worker used a different {key}: {directory}")
    games = [
        Game(item["game_id"], item["split"], Path(protocol["data_root"]),
             Path(item["game_path"]), Path(item["trajectory_path"]),
             item["task_type"], item["input_hashes"])
        for item in protocol["selected_games"]
    ]
    expected = [game.game_id for game in games]
    result = json.loads((directory / "results.json").read_text())
    rows = result["episodes"]
    ids = [row["game"]["game_id"] for row in rows]
    if not expected or len(expected) != len(set(expected)) or ids != expected:
        raise RuntimeError(f"Result IDs differ from the registered tasks: {directory}")
    if job["samples"] and len(rows) != job["samples"]:
        raise RuntimeError(f"Wrong number of completed smoke tasks: {directory}")
    if not job["samples"] and len(rows) != RELEASE_GAME_COUNTS[job["split"]]:
        raise RuntimeError(f"Main results do not cover the full released split: {directory}")
    if any(not isinstance(row["success"], bool) for row in rows):
        raise RuntimeError(f"Invalid success values: {directory}")
    store = CheckpointStore(directory, protocol)
    if result["run_sha256"] != store.manifest["run_sha256"] or store.load(games) != rows:
        raise RuntimeError(f"Results do not match the committed episode checkpoints: {directory}")
    for game in games:
        game.verify_inputs()
    if cache_summary(rows)["episodes_missing_cache_audit"]:
        raise RuntimeError(f"Missing actual KV size measurements: {directory}")
    if job["purpose"] == "smoke":
        if not any(row["steps"] >= 2 for row in rows):
            raise RuntimeError(f"Smoke did not exercise environment continuation: {directory}")
        if job["method"] != "fullkv":
            if not any(
                row["audit"].get("prune_event_count", 0) > 0
                and row["audit"]["nonprompt_resident_tokens"] < row["audit"]["nonprompt_logical_tokens"]
                for row in rows
            ):
                raise RuntimeError(f"Smoke did not exercise real KV pruning: {directory}")
    return len(rows)


def summarize(output: Path, plan: dict) -> None:
    completed = []
    expected_ids = None
    for job in plan["jobs"]:
        if job["purpose"] != "main":
            continue
        directory = output / job["name"]
        status_path = directory / "status.json"
        if not status_path.exists() or json.loads(status_path.read_text()).get("status") != "complete":
            continue
        validate_result(directory, job)
        rows = json.loads((directory / "results.json").read_text())["episodes"]
        ids = [row["game"]["game_id"] for row in rows]
        if expected_ids is not None and ids != expected_ids:
            raise RuntimeError("Completed conditions do not have identical registered task IDs")
        expected_ids = ids
        successes = sum(row["success"] for row in rows)
        completed.append({
            "method": job["method"], "keep_ratio": 1.0 if job["method"] == "fullkv" else job["ratio"],
            "n": len(rows), "successes": successes, "success_rate_percent": 100 * successes / len(rows),
            "mean_environment_steps": statistics.mean(row["steps"] for row in rows),
            "kv_cache_statistics": cache_summary(rows),
        })
    atomic_json(output / "SUMMARY.json", {
        "protocol": plan["protocol"], "model": plan["model"],
        "completed_main": len(completed), "expected_main": len(CONDITIONS),
        "task_ids": expected_ids, "results": completed,
        "timing_policy": plan["latency_policy"],
        "statistics": "Single seed, matched task set. No across-seed standard deviation or significance claim.",
    })
    lines = [
        "# ALFWorld shared-GPU accuracy", "",
        "Only complete, exact-ID validated main conditions are included; smoke and partial scores are excluded.",
        "Timings are diagnostic only. This is not a native browser/visual ALFRED or historical QA reproduction.",
        "", "| Method | Retained trajectory fraction | Tasks | Success (%) | Mean steps | Final KV MiB | Mean peak KV MiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in completed:
        lines.append(
            f"| {row['method']} | {row['keep_ratio']:.0%} | {row['n']} | "
            f"{row['success_rate_percent']:.2f} | {row['mean_environment_steps']:.2f} | "
            f"{row['kv_cache_statistics']['mean']['final_total_kv_mib']:.2f} | "
            f"{row['kv_cache_statistics']['mean']['peak_active_total_kv_mib']:.2f} |"
        )
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(MODELS), required=True)
    parser.add_argument("--gpus", type=int, nargs="+", required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--tmp-root", type=Path, required=True,
                        help="Executable local/tmpfs scratch, not NFS (native planner dlopen)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--memory-limit-gib", type=float, required=True)
    parser.add_argument("--min-free-gib", type=float, required=True)
    args = parser.parse_args()
    if len(args.gpus) != len(set(args.gpus)) or min(args.gpus) < 0:
        parser.error("GPU IDs must be distinct nonnegative integers")
    if not 0 < args.memory_limit_gib < args.min_free_gib:
        parser.error("Require positive allocator cap and a larger free-memory threshold")
    args.output = args.output.resolve()
    args.assets = args.assets.resolve()
    args.tmp_root = args.tmp_root.resolve()
    args.tmp_root.mkdir(parents=True, exist_ok=True)
    mount = json.loads(subprocess.check_output([
        "findmnt", "--json", "--target", str(args.tmp_root), "--output", "FSTYPE,OPTIONS",
    ], text=True, timeout=10))["filesystems"][0]
    if mount["fstype"] in {"nfs", "nfs4", "cifs"} or "noexec" in mount["options"].split(","):
        raise RuntimeError("ALFWorld native planner requires executable, non-NFS temporary storage")
    args.output.mkdir(parents=True, exist_ok=True)
    current_hashes = source_hashes()
    validation = json.loads(args.validation.read_text())
    if validation.get("status") != "passed" or validation.get("source_hashes") != current_hashes:
        raise RuntimeError("CPU validation missing or source changed; rerun focused tests")
    plan = {
        "protocol": "alfworld_shared_h200_accuracy_v1", "model": MODELS[args.model],
        "gpus": args.gpus, "assets": str(args.assets),
        "tmp_root": str(args.tmp_root),
        "memory_limit_gib_per_gpu": args.memory_limit_gib,
        "min_free_gib_per_gpu": args.min_free_gib, "jobs": jobs(), "seed": 233,
        "max_new_tokens": 128, "max_context_tokens": 32768,
        "source_hashes": current_hashes,
        "latency_policy": "Shared-GPU diagnostic times only; no latency claims",
    }
    plan = json.loads(json.dumps(plan))
    plan_path = args.output / "queue_plan.json"
    with contextlib.ExitStack() as stack:
        lease_fds = []
        for path in [args.output / "queue.lock", *[
            args.output.parent / f"alfworld_gpu{gpu}.lock" for gpu in sorted(args.gpus)
        ]]:
            lock = stack.enter_context(path.open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lease_fds.append(lock.fileno())
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise RuntimeError("Queue settings changed; use a fresh output directory")
        previous_path = args.output / "queue_status.json"
        if previous_path.exists():
            previous = json.loads(previous_path.read_text())
            if identity_alive(previous.get("worker_identity")):
                raise RuntimeError("A worker from this queue is still running; refusing a duplicate")
        atomic_json(plan_path, plan)
        summarize(args.output, plan)
        job = None
        completed = 0
        worker_identity = None

        def state(status: str, **extra) -> None:
            atomic_json(previous_path, {
                "status": status, "supervisor_pid": os.getpid(), "updated": time.time(),
                "job": job, "completed_jobs": completed, "total_jobs": len(plan["jobs"]),
                "worker_identity": worker_identity, **extra,
            })

        try:
            while True:
                ready_paths = [args.assets / f"{kind}_ready.json" for kind in (args.model, "alfworld")]
                if all(path.exists() for path in ready_paths):
                    break
                for kind in (args.model, "alfworld"):
                    path = args.assets / f"{kind}_status.json"
                    if path.exists() and json.loads(path.read_text()).get("status") == "failed":
                        raise RuntimeError(f"Asset preparation failed: {path}")
                state("waiting_for_assets")
                time.sleep(15)
            model = json.loads(ready_paths[0].read_text())
            data = json.loads(ready_paths[1].read_text())
            if (model["repo"], model["revision"]) != MODELS[args.model]:
                raise RuntimeError("Model assets differ from the pinned official checkpoint")
            cache_root = args.assets.parent / "caches"
            cache_environment = {
                "HF_HOME": str(args.assets / "hf_cache"),
                "HF_HUB_CACHE": str(args.assets / "hf_cache/hub"),
                "TORCH_HOME": str(cache_root / "torch"),
                "TORCH_EXTENSIONS_DIR": str(cache_root / "torch_extensions"),
                "XDG_CACHE_HOME": str(cache_root / "xdg"),
                "TRITON_CACHE_DIR": str(cache_root / "triton"),
                "TORCHINDUCTOR_CACHE_DIR": str(cache_root / "inductor"),
                "TMPDIR": str(args.tmp_root),
            }
            for directory in cache_environment.values():
                Path(directory).mkdir(parents=True, exist_ok=True)
            env = dict(
                os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, args.gpus)),
                OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
                HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", ALFWORLD_DATA=data["path"],
            )
            env.update(cache_environment)
            for job in plan["jobs"]:
                if source_hashes() != current_hashes:
                    raise RuntimeError("Frozen source changed while queue was running")
                directory = args.output / job["name"]
                status_path = directory / "status.json"
                if status_path.exists() and json.loads(status_path.read_text()).get("status") == "complete":
                    validate_result(directory, job)
                    completed += 1
                    continue
                while True:
                    devices = gpu_snapshot(args.gpus)
                    if enough_memory(devices, args.min_free_gib):
                        break
                    state("waiting_for_free_memory", devices=devices)
                    time.sleep(30)
                directory.mkdir(parents=True, exist_ok=True)
                command = [
                    sys.executable, "-u", str(RUNNER),
                    "--model-path", model["path"], "--data-root", data["path"],
                    "--method", job["method"], "--ratio", str(job["ratio"]),
                    "--split", job["split"], "--samples", str(job["samples"]),
                    "--seed", "233", "--max-steps", str(job["max_steps"]),
                    "--max-new-tokens", str(plan["max_new_tokens"]),
                    "--max-context-tokens", str(plan["max_context_tokens"]),
                    "--device", "auto" if len(args.gpus) > 1 else "cuda:0",
                    "--memory-limit-gib", str(args.memory_limit_gib),
                    "--purpose", job["purpose"], "--output", str(directory),
                ]
                log_path = args.output / "logs" / (job["name"].replace("/", "__") + ".log")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a") as log:
                    worker = subprocess.Popen(
                        command, env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                        pass_fds=tuple(lease_fds),
                    )
                    worker_identity = process_identity(worker.pid)
                    while worker.poll() is None:
                        state("running", command=command, log_path=str(log_path),
                              devices_before_loading=devices)
                        time.sleep(10)
                if worker.returncode:
                    raise RuntimeError(f"Worker exited {worker.returncode}; inspect {log_path}")
                validate_result(directory, job)
                summarize(args.output, plan)
                worker_identity = None
                completed += 1
            state("complete")
        except Exception:
            state("failed", traceback=traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
