"""Durable seed-42/3407 queue for the four completed fixed-budget methods."""
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
from run_rebuttal_legacy import atomic_json
from scripts.run_rebuttal_multidataset_queue import SOURCE_ROOT, validate_source, validate_result
from scripts.run_rebuttal_efficiency_queue import maintenance_blockers, maintenance_launch_lock, maintenance_path

METHODS = ("h2o_step", "tova_step", "flowkv_style", "lazyeviction")
DATASETS = ("hotpotqa", "2wiki", "musique")
SOURCES = {method: SOURCE_ROOT / {"lazyeviction": "lazy_source",
                                   "flowkv_style": "flowkv_source"}.get(method, f"{method}_source")
           for method in METHODS}


def validate_job(directory, job):
    rows, audits = validate_result(directory, job["samples"])
    if job["purpose"] == "smoke":
        if not any(row["num_steps"] >= 2 for row in rows):
            raise RuntimeError(f"Smoke did not exercise tool continuation: {directory}")
        events = [event for audit in audits.values()
                  for event in audit.get("cache_measurements", [])]
        if not any(event.get("before_tokens", 0) > event.get("after_tokens", 0) for event in events):
            raise RuntimeError(f"Smoke did not exercise actual KV eviction: {directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, choices=[42, 3407], required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-free-mb", type=int, default=28000)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = [dict(dataset=dataset, method=method, ratio=ratio, purpose=purpose,
                 samples=n, sample_start=start, seed=args.seed,
                 name=f"{dataset}/{method}/seed{args.seed}/{purpose}/r{int(ratio*100)}")
            for dataset in DATASETS for method in METHODS
            for ratio in (.2, .5)
            for purpose, n, start in (("smoke", 10, 500), ("main", 500, 0))]
    code = [ROOT / "run_rebuttal_legacy.py", ROOT / "run_rebuttal_legacy_step.py",
            ROOT / "run_rebuttal_multidataset.py", Path(__file__)]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in code}
    pinned = {method: validate_source(SOURCES[method])[0] for method in METHODS}
    for dataset in ("2wiki", "musique"):
        preflight = ROOT / "results/rebuttal_20260928/multidataset_preflight" / f"{dataset}.json"
        if json.loads(preflight.read_text()).get("status") != "passed":
            raise RuntimeError(f"Missing dataset preflight: {dataset}")
    plan = dict(protocol="restored_original_evaluator_multiseed_v1", seed=args.seed,
                gpu=args.gpu, jobs=jobs, source_hashes=pinned, runner_hashes=hashes,
                target_main_conditions=24, target_main_evaluations=12000)
    plan_path = output / "queue_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError("Changed queue plan; use a new output directory")
    atomic_json(plan_path, plan)

    def state(status, **details):
        atomic_json(output / "queue_status.json", dict(status=status, supervisor_pid=os.getpid(),
                    updated=time.time(), total_jobs=len(jobs), **details))

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    job = None
    try:
        for ordinal, job in enumerate(jobs):
            source = SOURCES[job["method"]]
            if validate_source(source)[0] != pinned[job["method"]]:
                raise RuntimeError(f"Frozen source changed: {source}")
            if any(hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path.relative_to(ROOT))] for path in code):
                raise RuntimeError("Runner changed after queue registration")
            directory = output / job["name"]
            status_path = directory / "status.json"
            if status_path.exists() and json.loads(status_path.read_text()).get("status") == "complete":
                validate_job(directory, job)
                continue
            if job["dataset"] == "hotpotqa":
                runner = ROOT / ("run_rebuttal_legacy_step.py" if job["method"] in ("h2o_step", "tova_step")
                                 else "run_rebuttal_legacy.py")
                dataset_args = []
            else:
                runner = ROOT / "run_rebuttal_multidataset.py"
                dataset_args = ["--dataset", job["dataset"]]
            command = [sys.executable, "-u", str(runner), *dataset_args,
                       "--source", str(source), "--method", job["method"],
                       "--ratio", str(job["ratio"]), "--seed", str(args.seed),
                       "--samples", str(job["samples"]), "--sample-start", str(job["sample_start"]),
                       "--purpose", job["purpose"], "--output", str(directory)]
            directory.mkdir(parents=True, exist_ok=True)
            while True:
                with maintenance_launch_lock(args.gpu):
                    blockers = maintenance_blockers(maintenance_path(args.gpu))
                    free = int(subprocess.check_output(["nvidia-smi", f"--id={args.gpu}",
                        "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
                    if not blockers and free >= args.min_free_mb:
                        if validate_source(source)[0] != pinned[job["method"]]:
                            raise RuntimeError("Frozen source changed while waiting for GPU")
                        with (directory / "run.log").open("a") as log:
                            child = subprocess.Popen(command, cwd=source, env=env, stdin=subprocess.DEVNULL,
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
