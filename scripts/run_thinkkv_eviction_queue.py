"""Durable three-seed ThinKV eviction-only QA queue after mechanism preflight."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_rebuttal_legacy import atomic_json, verify_original_sources
from scripts.run_rebuttal_efficiency_queue import maintenance_blockers, maintenance_launch_lock, maintenance_path


def validate_job(directory, job):
    state = json.loads((directory / "status.json").read_text())
    result = json.loads((directory / "results.json").read_text())
    rows = result["results"]
    checkpoint = json.loads((directory / "checkpoint.json").read_text())
    manifest = json.loads((directory / "manifest.json").read_text())
    summary = json.loads((directory / "SUMMARY.json").read_text())
    ids = [row["id"] for row in rows]
    if (state["status"] != "complete" or len(ids) != job["samples"]
            or len(set(ids)) != len(ids) or set(ids) != set(manifest["ids"])
            or checkpoint != rows or manifest["seed"] != job["seed"]
            or manifest["ratio"] != job["ratio"] or summary["status"] != "complete"):
        raise RuntimeError(f"Incomplete ThinkKV result: {directory}")
    if (not any(row["audit"]["evictions"] for row in rows)
            or any(row["audit"]["quantization"] != "none" for row in rows)):
        raise RuntimeError(f"No actual eviction or unexpected quantization: {directory}")
    if job["purpose"] == "smoke" and not any(len(row["trajectory"]) >= 2 for row in rows):
        raise RuntimeError(f"ThinkKV smoke did not exercise tool continuation: {directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--min-free-mb", type=int, default=28000)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_original_sources(ROOT)
    jobs = [dict(seed=seed, dataset=dataset, ratio=ratio, purpose=purpose,
                 samples=count, sample_start=start,
                 name=f"seed{seed}/{dataset}/r{int(ratio*100)}/{purpose}")
            for seed in (233, 42, 3407) for dataset in ("hotpotqa", "2wiki", "musique")
            for ratio in (.2, .5) for purpose, count, start in (("smoke", 20, 500), ("main", 500, 0))]
    code = [Path(__file__), ROOT / "scripts/run_thinkkv_eviction.py",
            ROOT / "scripts/thinkkv_eviction_runtime.py", ROOT / "models/RebuttalLLM.py",
            ROOT / "token_tracker.py"]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in code}
    plan = dict(protocol="thinkkv_eviction_only_qwen_v1", gpu=args.gpu,
                jobs=jobs, source_hashes=hashes, preflight=str(args.preflight.resolve()),
                target_main_conditions=18, target_main_evaluations=9000,
                paper="https://arxiv.org/html/2510.01290v2#A5.SS10",
                scope="Single-category Qwen QA adaptation; eviction only, native-dtype KV")
    plan_path = output / "queue_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError("Changed ThinkKV plan; use a fresh output directory")
    atomic_json(plan_path, plan)

    def state(status, **details):
        atomic_json(output / "queue_status.json", dict(status=status, supervisor_pid=os.getpid(),
                    updated=time.time(), total_jobs=len(jobs), **details))

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED="1",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", OMP_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
    job = None
    try:
        preflight = args.preflight.resolve()
        while True:
            pilot_state = json.loads((preflight / "status.json").read_text()) if (preflight / "status.json").exists() else {}
            if pilot_state.get("status") == "complete":
                pilot = json.loads((preflight / "results.json").read_text())
                rows = pilot["results"]
                if (len(rows) < 10 or not any(len(row["trajectory"]) >= 2 for row in rows)
                        or not any(row["audit"]["evictions"] for row in rows)
                        or any(row["audit"]["quantization"] != "none" for row in rows)):
                    raise RuntimeError("ThinkKV preflight lacks multistep or eviction-only evidence")
                break
            if pilot_state.get("status") == "failed":
                raise RuntimeError("ThinkKV preflight failed; main queue will not start")
            state("waiting_for_preflight", completed_jobs=0, preflight_status=pilot_state.get("status"))
            time.sleep(15)
        for ordinal, job in enumerate(jobs):
            directory = output / job["name"]
            status_path = directory / "status.json"
            if status_path.exists() and json.loads(status_path.read_text()).get("status") == "complete":
                validate_job(directory, job)
                continue
            if any(hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path.relative_to(ROOT))] for path in code):
                raise RuntimeError("ThinkKV eviction source changed after queue registration")
            command = [sys.executable, "-u", str(ROOT / "scripts/run_thinkkv_eviction.py"),
                       "--dataset", job["dataset"], "--seed", str(job["seed"]),
                       "--ratio", str(job["ratio"]), "--purpose", job["purpose"],
                       "--samples", str(job["samples"]), "--sample-start", str(job["sample_start"]),
                       "--output", str(directory)]
            directory.mkdir(parents=True, exist_ok=True)
            while True:
                with maintenance_launch_lock(args.gpu):
                    blockers = maintenance_blockers(maintenance_path(args.gpu))
                    free = int(subprocess.check_output(["nvidia-smi", f"--id={args.gpu}",
                        "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
                    if not blockers and free >= args.min_free_mb:
                        with (directory / "run.log").open("a") as log:
                            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                                     stdout=log, stderr=subprocess.STDOUT)
                        break
                    state("waiting_for_gpu", job=job, completed_jobs=ordinal,
                          free_mb=free, maintenance=blockers)
                time.sleep(15)
            state("running", job=job, completed_jobs=ordinal, child_pid=child.pid)
            if child.wait():
                raise RuntimeError(f"Worker failed: {directory / 'run.log'}")
            validate_job(directory, job)
        state("complete", completed_jobs=len(jobs))
    except BaseException as exc:
        state("failed", job=job, error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
