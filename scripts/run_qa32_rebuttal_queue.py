"""Persistent, memory-bounded Qwen2.5-32B legacy QA accuracy queues."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_rebuttal_legacy import atomic_json
from scripts.prepare_alfworld_rebuttal_assets import sha256
from scripts.run_alfworld_rebuttal_queue import enough_memory, gpu_snapshot
from scripts.run_rebuttal_efficiency_queue import identity_alive, process_identity
from scripts.run_qa32_rebuttal import source_fingerprint, validate_completed_run

RUNNER = ROOT / "scripts/run_qa32_rebuttal.py"
CONDITIONS = (
    ("fullkv", .5, "r100"), ("stepkv", .2, "r20"), ("stepkv", .5, "r50"),
    ("h2o", .2, "r20"), ("h2o", .5, "r50"), ("tova", .2, "r20"), ("tova", .5, "r50"),
    ("h2o_step", .2, "r20"), ("h2o_step", .5, "r50"),
    ("tova_step", .2, "r20"), ("tova_step", .5, "r50"),
)


def source_gate() -> dict:
    return {
        **source_fingerprint(),
        "queue_sha256": sha256(Path(__file__)),
        "resource_helpers_sha256": sha256(ROOT / "scripts/run_alfworld_rebuttal_queue.py"),
        "requirements_sha256": sha256(ROOT / "config/environments/h200_accuracy_20260929.requirements.txt"),
    }


def jobs(datasets: list[str]) -> list[dict]:
    return [
        {
            "name": f"{dataset}/{purpose}/{method}/{label}", "dataset": dataset,
            "purpose": purpose, "method": method, "ratio": ratio,
            "samples": 2 if purpose == "smoke" else 500,
            "sample_start": 500 if purpose == "smoke" else 0,
            "seed": 233, "max_steps": 7,
        }
        for dataset in datasets
        for purpose in ("smoke", "main")
        for method, ratio, label in CONDITIONS
    ]


def validate_job(directory: Path, job: dict) -> dict:
    validated = validate_completed_run(directory)
    manifest = json.loads((directory / "manifest.json").read_text())["fingerprint"]
    for key in ("dataset", "method", "ratio", "purpose", "samples", "sample_start", "seed", "max_steps"):
        if manifest[key] != job[key]:
            raise RuntimeError(f"QA32 worker used a different {key}: {directory}")
    if validated["completed_samples"] != job["samples"]:
        raise RuntimeError(f"QA32 worker did not complete its registered sample count: {directory}")
    if job["purpose"] == "smoke" and job["method"] != "fullkv" and not validated["has_actual_eviction"]:
        raise RuntimeError(f"QA32 smoke did not exercise real budget-driven KV eviction: {directory}")
    return validated


def summarize(root: Path, plan: dict) -> None:
    completed, identities = [], {}
    for job in plan["jobs"]:
        directory = root / job["name"]
        status_path = directory / "status.json"
        if job["purpose"] != "main" or not status_path.exists():
            continue
        if json.loads(status_path.read_text()).get("status") != "complete":
            continue
        record = validate_job(directory, job)
        expected = identities.setdefault(job["dataset"], record["sample_ids"])
        if record["sample_ids"] != expected:
            raise RuntimeError("Completed QA32 conditions do not use identical ordered question IDs")
        completed.append({**job, **record})
    atomic_json(root / "SUMMARY.json", {
        "model": "Qwen/Qwen2.5-32B-Instruct", "completed_main": len(completed),
        "expected_main": len(CONDITIONS) * len(plan["datasets"]), "results": completed,
        "timing_policy": "Shared GPUs and diagnostic hooks; not a latency result",
        "statistics": "Single seed233; no across-seed SD or mixing with historical seven-billion-model tables",
    })
    lines = [
        "# Qwen2.5-32B legacy QA supplement", "",
        "Only complete, ID- and tensor-audit-validated main conditions are included.",
        "K/V payload is measured across both GPUs; shared-GPU timings are diagnostic only.",
        "", "| Dataset | Method | Retained fraction | N | EM (%) | F1 (%) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in completed:
        fraction = 1.0 if row["method"] == "fullkv" else row["ratio"]
        lines.append(
            f"| {row['dataset']} | {row['method']} | {fraction:.0%} | {row['completed_samples']} | "
            f"{row['summary']['exact_match']:.2f} | {row['summary']['f1_score']:.2f} |"
        )
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=["hotpotqa", "2wiki", "musique"], required=True)
    parser.add_argument("--gpus", nargs=2, type=int, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tmp-root", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--memory-limit-gib", type=float, default=44)
    parser.add_argument("--min-free-gib", type=float, default=48)
    parser.add_argument("--wait-for-gpu-lease", action="store_true")
    args = parser.parse_args()
    if len(set(args.gpus)) != 2 or min(args.gpus) < 0 or len(set(args.datasets)) != len(args.datasets):
        parser.error("Select two distinct nonnegative GPU IDs and nonduplicate datasets")
    if not 4 < args.memory_limit_gib <= 44 or not args.memory_limit_gib < args.min_free_gib:
        parser.error("Require allocator cap in (4,44] GiB and a larger free-memory threshold")
    args.output, args.assets, args.tmp_root = args.output.resolve(), args.assets.resolve(), args.tmp_root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.tmp_root.mkdir(parents=True, exist_ok=True)
    gate = source_gate()
    validation = json.loads(args.validation.read_text())
    if validation.get("status") != "passed" or validation.get("source_gate") != gate:
        raise RuntimeError("QA32 CPU validation is missing or source changed")
    plan = {
        "protocol": "qa32_frozen_legacy_shared_h200_v1", "datasets": args.datasets,
        "gpus": args.gpus, "assets": str(args.assets), "tmp_root": str(args.tmp_root),
        "memory_limit_gib_per_gpu": args.memory_limit_gib, "min_free_gib_per_gpu": args.min_free_gib,
        "wait_for_gpu_lease": args.wait_for_gpu_lease,
        "jobs": jobs(args.datasets), "source_gate": gate,
        "timing_policy": "Diagnostic only; latency deferred by user",
    }
    plan = json.loads(json.dumps(plan))
    with contextlib.ExitStack() as stack:
        lease_fds = []
        for path in [args.output / "queue.lock"]:
            lock = stack.enter_context(path.open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            lease_fds.append(lock.fileno())
        plan_path = args.output / "queue_plan.json"
        if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
            raise RuntimeError("QA32 queue configuration changed; use a fresh output directory")
        status_path = args.output / "queue_status.json"
        if status_path.exists() and identity_alive(json.loads(status_path.read_text()).get("worker_identity")):
            raise RuntimeError("A QA32 worker is already alive for this queue")
        atomic_json(plan_path, plan)
        completed, job, identity = 0, None, None

        def state(status: str, **extra) -> None:
            atomic_json(status_path, {
                "status": status, "supervisor_pid": os.getpid(), "updated": time.time(),
                "job": job, "completed_jobs": completed, "total_jobs": len(plan["jobs"]),
                "worker_identity": identity, **extra,
            })

        try:
            while True:
                handles = []
                try:
                    for gpu in sorted(args.gpus):
                        handle = (args.output.parent / f"alfworld_gpu{gpu}.lock").open("a")
                        handles.append(handle)
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    for handle in handles:
                        handle.close()
                    if not args.wait_for_gpu_lease:
                        raise
                    state("waiting_for_gpu_lease", gpus=args.gpus)
                    time.sleep(30)
                    continue
                for handle in handles:
                    stack.enter_context(handle)
                    lease_fds.append(handle.fileno())
                break
            for filename in ("model_ready.json", "data_ready.json"):
                if not (args.assets / filename).is_file():
                    raise RuntimeError(f"Required pinned QA assets missing: {args.assets / filename}")
            root = args.assets.parent
            cache_environment = {
                "HF_HOME": str(root / "assets/hf_cache"), "HF_HUB_CACHE": str(root / "assets/hf_cache/hub"),
                "TORCH_HOME": str(root / "caches/torch"),
                "TORCH_EXTENSIONS_DIR": str(root / "caches/torch_extensions"),
                "XDG_CACHE_HOME": str(root / "caches/xdg"), "TRITON_CACHE_DIR": str(root / "caches/triton"),
                "TORCHINDUCTOR_CACHE_DIR": str(root / "caches/inductor"), "TMPDIR": str(args.tmp_root),
            }
            for value in cache_environment.values():
                Path(value).mkdir(parents=True, exist_ok=True)
            env = dict(
                os.environ, CUDA_VISIBLE_DEVICES=",".join(map(str, args.gpus)),
                OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1",
            )
            env.update(cache_environment)
            summarize(args.output, plan)
            for job in plan["jobs"]:
                if source_gate() != gate:
                    raise RuntimeError("QA32 source changed during the queue; use a new output directory")
                directory = args.output / job["name"]
                worker_status = directory / "status.json"
                if worker_status.exists() and json.loads(worker_status.read_text()).get("status") == "complete":
                    validate_job(directory, job)
                    completed += 1
                    continue
                while True:
                    devices = gpu_snapshot(args.gpus)
                    if enough_memory(devices, args.min_free_gib):
                        break
                    state("waiting_for_free_memory", devices=devices)
                    time.sleep(30)
                command = [
                    sys.executable, "-u", str(RUNNER), "--assets", str(args.assets),
                    "--output", str(directory), "--memory-limit-gib", str(args.memory_limit_gib),
                ]
                for key in ("dataset", "method", "ratio", "purpose", "samples", "sample_start", "seed", "max_steps"):
                    command.extend(["--" + key.replace("_", "-"), str(job[key])])
                log_path = args.output / "logs" / (job["name"].replace("/", "__") + ".log")
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with log_path.open("a") as log:
                    worker = subprocess.Popen(
                        command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                        pass_fds=tuple(lease_fds),
                    )
                    identity = process_identity(worker.pid)
                    while worker.poll() is None:
                        state("running", command=command, log_path=str(log_path), devices_before_loading=devices)
                        time.sleep(10)
                if worker.returncode:
                    raise RuntimeError(f"QA32 worker exited {worker.returncode}; inspect {log_path}")
                validate_job(directory, job)
                identity = None
                completed += 1
                summarize(args.output, plan)
            state("complete")
        except Exception:
            state("failed", traceback=traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
