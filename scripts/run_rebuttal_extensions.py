"""Run isolated new-baseline worktrees on one GPU, after their validation gates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_rebuttal_schedule import atomic_json
from scripts.run_rebuttal_queue import existing_worker, wait_for_existing_worker

SOURCES = Path("/data/experiment/fengboyu/stepkv/rebuttal_extensions_20260928")
WORKTREES = {"flowkv_style": SOURCES / "source",
             "rkv": SOURCES / "rkv_source",
             "lazyeviction": SOURCES / "lazy_source"}


def hashes(source):
    files = [source / "run_rebuttal_schedule.py", source / "run_all_wiki_experiments_v2.py",
             source / "token_tracker.py"]
    for name in ("models", "kv_cache", "retrievers", "vendor"):
        files.extend(sorted((source / name).rglob("*.py")))
    return {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def validate_smoke(directory, job):
    result = json.loads((directory / "results.json").read_text())
    if len(result["results"]) != job["samples"]:
        raise RuntimeError(f"Smoke result count mismatch: {job['name']}")
    if not any(row["num_steps"] >= 2 for row in result["results"]):
        raise RuntimeError(f"Smoke did not exercise tool continuation: {job['name']}")
    events = [e for row in result["results"]
              for e in row.get("runtime_audit", {}).get("cache_measurements", [])]
    if not any(e.get("before_tokens", 0) > e.get("after_tokens", 0) for e in events):
        raise RuntimeError(f"Smoke did not exercise actual KV eviction: {job['name']}")


def main():
    raise SystemExit("controlled extensions are retired. Use scripts/run_rebuttal_legacy_queue.py.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--methods", nargs="+", choices=sorted(WORKTREES),
                        default=["flowkv_style", "rkv", "lazyeviction"])
    parser.add_argument("--output", type=Path, default=ROOT / "results/rebuttal_20260928/new_baselines")
    parser.add_argument("--assets", type=Path, default=Path("/data/experiment/fengboyu/stepkv"))
    parser.add_argument("--min-free-mb", type=int, default=19000)
    parser.add_argument("--deadline", default="2026-10-12T18:00:00+00:00")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = []
    for method in args.methods:
        for purpose, samples, start in (("smoke", 2, 500), ("main", 500, 0)):
            for ratio in (.2, .5):
                prefix = "smoke" if purpose == "smoke" else "seed233"
                jobs.append(dict(method=method, purpose=purpose, samples=samples,
                                 sample_start=start, seed=233, ratio=ratio,
                                 name=f"{method}/{prefix}/r{int(ratio*100)}/{method}"))
    plan = dict(jobs=jobs, gpu=args.gpu, min_free_mb=args.min_free_mb,
                assets=str(args.assets), deadline=args.deadline,
                sources={name: str(WORKTREES[name]) for name in args.methods},
                model="Qwen/Qwen2.5-7B-Instruct", seeds=[233])
    plan_path = args.output / "queue_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError("Extension queue changed; use a new output directory")
    atomic_json(plan_path, plan)
    deadline = datetime.fromisoformat(args.deadline)

    def state(status, **fields):
        atomic_json(args.output / "queue_status.json", dict(
            status=status, supervisor_pid=os.getpid(), updated=time.time(),
            total_jobs=len(jobs), **fields))

    def expired():
        return datetime.now(timezone.utc) >= deadline

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    try:
        for ordinal, job in enumerate(jobs):
            if expired():
                state("deadline_reached", completed_jobs=ordinal)
                return
            source = WORKTREES[job["method"]]
            ready = SOURCES / f"{job['method']}_ready.json"
            while not ready.exists():
                state("waiting_for_validated_implementation", job=job, completed_jobs=ordinal,
                      readiness_file=str(ready))
                if expired():
                    state("deadline_reached", completed_jobs=ordinal)
                    return
                time.sleep(5)
            gate = json.loads(ready.read_text())
            current_hashes = hashes(source)
            if gate.get("status") != "validated" or gate.get("source_hashes") != current_hashes:
                raise RuntimeError(f"Validation gate/source mismatch: {ready}")
            method_root = args.output / job["method"]
            method_root.mkdir(parents=True, exist_ok=True)
            pin_path = method_root / "source_pin.json"
            if pin_path.exists():
                if json.loads(pin_path.read_text()) != current_hashes:
                    raise RuntimeError(f"Baseline source changed since its first job: {pin_path}")
            else:
                for manifest_path in method_root.rglob("manifest.json"):
                    recorded = json.loads(manifest_path.read_text())["fingerprint"]["source_hashes"]
                    if any(recorded.get(name) != digest for name, digest in current_hashes.items()):
                        raise RuntimeError(f"Existing result source mismatch: {manifest_path}")
                atomic_json(pin_path, current_hashes)
            directory = args.output / job["name"]
            status_path = directory / "status.json"
            if status_path.exists() and json.loads(status_path.read_text()).get("status") == "complete":
                if job["purpose"] == "smoke":
                    validate_smoke(directory, job)
                print(f"Already complete: {job['name']}", flush=True)
                continue
            command = [sys.executable, "-u", str(source / "run_rebuttal_schedule.py"),
                       "--method", job["method"], "--ratio", str(job["ratio"]),
                       "--seed", "233", "--samples", str(job["samples"]),
                       "--sample-start", str(job["sample_start"]), "--purpose", job["purpose"],
                       "--assets", str(args.assets), "--output", str(directory)]
            pid = existing_worker(status_path, command)
            while pid is None:
                free = int(subprocess.check_output(["nvidia-smi", f"--id={args.gpu}",
                           "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
                if free >= args.min_free_mb:
                    break
                state("waiting_for_gpu_memory", job=job, completed_jobs=ordinal, free_mb=free)
                if expired():
                    state("deadline_reached", completed_jobs=ordinal)
                    return
                time.sleep(15)
            directory.mkdir(parents=True, exist_ok=True)
            if pid is not None:
                state("running", job=job, completed_jobs=ordinal, child_pid=pid, adopted_worker=True)
                wait_for_existing_worker(pid, command, status_path)
            else:
                latest_gate = json.loads(ready.read_text())
                if (latest_gate.get("status") != "validated"
                        or latest_gate.get("source_hashes") != current_hashes
                        or hashes(source) != current_hashes):
                    raise RuntimeError(f"Validated source changed while waiting for GPU: {ready}")
                print(f"Launching {ordinal+1}/{len(jobs)}: {job['name']}", flush=True)
                with (directory / "run.log").open("a") as log:
                    child = subprocess.Popen(command, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                             stdout=log, stderr=subprocess.STDOUT)
                    state("running", job=job, completed_jobs=ordinal, child_pid=child.pid)
                    code = child.wait()
                if code:
                    raise RuntimeError(f"Job exited {code}: {directory / 'run.log'}")
            if job["purpose"] == "smoke":
                validate_smoke(directory, job)
            else:
                subprocess.run([sys.executable, str(source / "scripts/summarize_rebuttal.py"),
                                "--root", str(method_root)], cwd=source, env=env, check=True)
                if job["method"] == "flowkv_style":
                    subprocess.run([sys.executable, str(source / "scripts/summarize_flowkv_controls.py"),
                                    "--root", str(method_root), "--controls",
                                    str(ROOT / "results/rebuttal_20260927/controlled_v2")],
                                   cwd=source, env=env, check=True)
        state("complete", completed_jobs=len(jobs))
    except BaseException as exc:
        state("failed", error=str(exc), job=job)
        raise


if __name__ == "__main__":
    main()
