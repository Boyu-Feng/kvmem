"""Durable SideQuest adaptation queue for sampling seeds 42 and 3407."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-free-mb", type=int, default=28000)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = (output / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pilot = ROOT / "results/rebuttal_20260928/sidequest_pilot_v1/SUMMARY.json"
    pilot_report = json.loads(pilot.read_text())
    if pilot_report["paired_samples"] != 20 or pilot_report["arms"]["sidequest_untrained"]["applied_deletion_events"] < 1:
        raise RuntimeError("Original SideQuest mechanism pilot did not pass")
    verify_original_sources(ROOT)
    jobs = [dict(seed=seed, dataset=dataset, purpose=purpose, samples=count, sample_start=start,
                 method="sidequest_untrained", name=f"seed{seed}/{dataset}/{purpose}")
            for seed in (42, 3407) for dataset in ("hotpotqa", "2wiki", "musique")
            for purpose, count, start in (("smoke", 10, 500), ("main", 500, 0))]
    code = [Path(__file__), ROOT / "scripts/run_sidequest_baseline.py", ROOT / "scripts/sidequest_runtime.py",
            ROOT / "models/RebuttalLLM.py", ROOT / "token_tracker.py"]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in code}
    plan = dict(protocol="sidequest_untrained_multiseed_v1", gpu=args.gpu, jobs=jobs,
                source_hashes=hashes, target_main_conditions=6, target_main_evaluations=3000,
                independent_decoder=True, trained=False)
    plan_path = output / "queue_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError("Changed SideQuest queue plan; use new output directory")
    atomic_json(plan_path, plan)

    def state(status, **details):
        atomic_json(output / "queue_status.json", dict(status=status, supervisor_pid=os.getpid(),
                    updated=time.time(), total_jobs=len(jobs), **details))

    def validate(directory, job):
        status = json.loads((directory / "status.json").read_text())
        result = json.loads((directory / "results.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        rows = result["results"]
        if (status["status"] != "complete" or len(rows) != job["samples"]
                or len({r["id"] for r in rows}) != job["samples"]
                or {r["id"] for r in rows} != set(manifest["ids"])
                or manifest["seed"] != job["seed"]
                or any(r["method"] != "sidequest_untrained" or not r["audit"]["enabled"] for r in rows)):
            raise RuntimeError(f"Invalid SideQuest output: {directory}")

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED="1",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", OMP_NUM_THREADS="4",
               OPENBLAS_NUM_THREADS="4", TOKENIZERS_PARALLELISM="false")
    job = None
    try:
        for ordinal, job in enumerate(jobs):
            directory = output / job["name"]
            status_path = directory / "status.json"
            if status_path.exists() and json.loads(status_path.read_text()).get("status") == "complete":
                validate(directory, job)
                continue
            if any(hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path.relative_to(ROOT))] for path in code):
                raise RuntimeError("SideQuest source changed after registration")
            command = [sys.executable, "-u", str(ROOT / "scripts/run_sidequest_baseline.py"),
                       "--dataset", job["dataset"], "--seed", str(job["seed"]),
                       "--purpose", job["purpose"], "--samples", str(job["samples"]),
                       "--sample-start", str(job["sample_start"]), "--output", str(directory)]
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
            validate(directory, job)
        state("complete", completed_jobs=len(jobs))
    except BaseException as exc:
        state("failed", job=job, error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
