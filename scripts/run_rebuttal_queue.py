"""Persistent sequential queue: validate all arms, then run paired experiments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_rebuttal_schedule import atomic_json, source_hashes

METHOD_ORDER = ["fullkv", "stepkv", "h2o_step", "tova_step",
                "stepkv_no_step_score", "h2o_token", "tova_token"]


def process_matches(pid, command):
    """Identify an existing worker without sending signals or trusting a stale PID."""
    try:
        actual = (Path("/proc") / str(pid) / "cmdline").read_bytes()
    except FileNotFoundError:
        return False
    return actual.rstrip(b"\0").split(b"\0") == [os.fsencode(part) for part in command]


def existing_worker(status_path, command):
    if not status_path.exists():
        return None
    status = json.loads(status_path.read_text())
    pid = status.get("pid")
    if status.get("status") in ("loading_retriever", "running") and pid and process_matches(pid, command):
        return pid
    return None


def wait_for_existing_worker(pid, command, status_path):
    while process_matches(pid, command):
        time.sleep(5)
    status = json.loads(status_path.read_text())
    if status.get("status") != "complete":
        raise RuntimeError(f"Adopted worker {pid} exited without completing: {status_path}")


def main():
    raise SystemExit("controlled_v2 queue is retired. Use scripts/run_rebuttal_legacy_queue.py.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--assets", type=Path, default=Path("/data/experiment/fengboyu/stepkv"))
    parser.add_argument("--output", type=Path, default=ROOT / "results/rebuttal_20260927/controlled_v2")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--seeds", type=int, nargs="+", default=[233])
    parser.add_argument("--ratios", type=float, nargs="+", default=[.2, .5])
    parser.add_argument("--deadline", default="2026-10-12T18:00:00+00:00")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "queue.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    pinned_sources = source_hashes()
    jobs = []
    for method in METHOD_ORDER:
        jobs.append(dict(method=method, seed=233, ratio=.2, samples=2,
                         sample_start=500, purpose="smoke", name=f"smoke/{method}"))
    for seed in args.seeds:
        for ratio in args.ratios:
            for method in METHOD_ORDER:
                if method == "fullkv" and ratio != args.ratios[0]:
                    continue
                jobs.append(dict(method=method, seed=seed, ratio=ratio,
                                 samples=args.samples, sample_start=0, purpose="main",
                                 name=f"seed{seed}/r{int(ratio*100)}/{method}"))
    protocol = {"sources": pinned_sources, "jobs": jobs, "gpu": args.gpu,
                "assets": str(args.assets), "deadline": args.deadline}
    plan = args.output / "queue_plan.json"
    if plan.exists() and json.loads(plan.read_text()) != protocol:
        raise RuntimeError("Queue source/protocol changed; use a new output directory")
    atomic_json(plan, protocol)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    subprocess.run([sys.executable, "-m", "pip", "freeze"], stdout=(args.output / "pip_freeze.txt").open("w"), check=True)
    state_path = args.output / "queue_status.json"
    def state(status, **fields):
        atomic_json(state_path, dict(status=status, supervisor_pid=os.getpid(),
                                    updated=time.time(), **fields))
    try:
        while not all((args.assets / f"{kind}_ready.json").exists() for kind in ("model", "data")):
            state("waiting_for_assets")
            time.sleep(30)
        for ordinal, job in enumerate(jobs):
            if datetime.now(timezone.utc) >= datetime.fromisoformat(args.deadline):
                state("deadline_reached", completed_jobs=ordinal, total_jobs=len(jobs))
                return
            if source_hashes() != pinned_sources:
                raise RuntimeError("Experiment source changed during queue; refusing mixed-source results")
            directory = args.output / job["name"]
            status_path = directory / "status.json"
            if status_path.exists() and json.loads(status_path.read_text()).get("status") == "complete":
                print(f"Already complete: {job['name']}", flush=True)
                continue
            cmd = [sys.executable, "-u", str(ROOT / "run_rebuttal_schedule.py"),
                   "--method", job["method"], "--ratio", str(job["ratio"]),
                   "--seed", str(job["seed"]), "--samples", str(job["samples"]),
                   "--sample-start", str(job["sample_start"]), "--purpose", job["purpose"],
                   "--assets", str(args.assets), "--output", str(directory)]
            adopted_pid = existing_worker(status_path, cmd)
            while adopted_pid is None:
                query = subprocess.check_output(["nvidia-smi", f"--id={args.gpu}",
                    "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True)
                free_mb = int(query.strip())
                if free_mb >= 28000:
                    break
                state("waiting_for_gpu_memory", next_job=job, free_mb=free_mb)
                if datetime.now(timezone.utc) >= datetime.fromisoformat(args.deadline):
                    state("deadline_reached", completed_jobs=ordinal)
                    return
                time.sleep(30)
            directory.mkdir(parents=True, exist_ok=True)
            if adopted_pid is not None:
                print(f"Continuing {ordinal+1}/{len(jobs)}: {job['name']} (existing PID {adopted_pid})", flush=True)
                state("running", job=job, child_pid=adopted_pid,
                      completed_jobs=ordinal, total_jobs=len(jobs), adopted_worker=True)
                wait_for_existing_worker(adopted_pid, cmd, status_path)
                code = 0
            else:
                print(f"Launching {ordinal+1}/{len(jobs)}: {job['name']}", flush=True)
                with (directory / "run.log").open("a") as log:
                    child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                    state("running", job=job, child_pid=child.pid,
                          completed_jobs=ordinal, total_jobs=len(jobs))
                    code = child.wait()
            if code:
                state("failed", job=job, returncode=code, log=str(directory / "run.log"))
                raise RuntimeError(f"Job failed (exit {code}): {job['name']}")
            if job["purpose"] == "smoke":
                result = json.loads((directory / "results.json").read_text())
                if not any(row["num_steps"] >= 2 for row in result["results"]):
                    raise RuntimeError(f"Smoke did not exercise multi-step tool use: {job['name']}")
                events = [e for row in result["results"] for e in row["runtime_audit"]["cache_measurements"]]
                if job["method"] != "fullkv" and not events:
                    raise RuntimeError(f"Smoke did not exercise pruning: {job['name']}")
            if job["purpose"] == "main":
                subprocess.run([sys.executable, str(ROOT / "scripts/summarize_rebuttal.py"),
                                "--root", str(args.output)], cwd=ROOT, env=env, check=True)
        state("complete", completed_jobs=len(jobs), total_jobs=len(jobs))
    except BaseException as exc:
        state("failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
