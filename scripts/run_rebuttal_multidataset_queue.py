"""Append 2Wiki/MuSiQue jobs after an existing legacy queue finishes."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_rebuttal_multidataset import atomic_json, source_hashes, verify_original_sources
from scripts.run_rebuttal_queue import existing_worker, wait_for_existing_worker
from scripts.run_rebuttal_efficiency_queue import (
    maintenance_blockers, maintenance_launch_lock, maintenance_path,
)

SOURCE_ROOT = Path("/data/experiment/fengboyu/stepkv/rebuttal_legacy_20260928")
RUNNER = ROOT / "run_rebuttal_multidataset.py"
SOURCES = {"flowkv_style": SOURCE_ROOT / "flowkv_source",
           "rkv": SOURCE_ROOT / "rkv_source", "lazyeviction": SOURCE_ROOT / "lazy_source",
           "h2o_step": SOURCE_ROOT / "h2o_step_source", "tova_step": SOURCE_ROOT / "tova_step_source"}


def validate_source(source):
    verified = verify_original_sources(source)
    gate = json.loads((source / "LEGACY_VALIDATION.json").read_text())
    hashes = source_hashes(source)
    baseline_hashes = {k: v for k, v in hashes.items() if k != "__legacy_runner__"}
    if gate.get("status") != "validated" or gate.get("source_hashes") != baseline_hashes:
        raise RuntimeError(f"CPU validation gate/source mismatch: {source}")
    return hashes, verified


def validate_result(directory, expected_samples):
    rows = json.loads((directory / "results.json").read_text())["results"]
    expected_ids = set(json.loads((directory / "manifest.json").read_text())["fingerprint"]["sample_ids"])
    if len(rows) != expected_samples or {r["id"] for r in rows} != expected_ids:
        raise RuntimeError(f"Incomplete result IDs: {directory}")
    audits = json.loads((directory / "baseline_audit.json").read_text())
    if set(audits) != {r["id"] for r in rows}:
        raise RuntimeError(f"Missing per-question baseline audit: {directory}")
    log = (directory / "run.log").read_text(errors="replace")
    if "Traceback (most recent call last)" in log or "[ERROR]" in log:
        status_path = directory / "status.json"
        status = json.loads(status_path.read_text())
        status.update(status="invalid_runtime", validation_error="Original evaluator logged a runtime error")
        atomic_json(status_path, status)
        raise RuntimeError(f"Original evaluator logged an error; excluded from summary: {directory}")
    return rows, audits


def validate_smoke(directory):
    rows, audits = validate_result(directory, 2)
    if not any(r["num_steps"] >= 2 for r in rows):
        raise RuntimeError(f"Smoke did not exercise tool continuation: {directory}")
    events = [e for audit in audits.values() for e in audit.get("cache_measurements", [])]
    if not any(e.get("before_tokens", 0) > e.get("after_tokens", 0) for e in events):
        raise RuntimeError(f"Smoke did not exercise actual KV eviction: {directory}")


def summarize(root, jobs):
    lines = ["# 新增 baseline：恢复后的原评测路径", "",
             "仅列已完成主实验。原表结果继续保留；本轮不重跑原方法的 500 题，也不合并已停止的 controlled_v2 数字。",
             "新增实验为 seed=233 单次结果，不能当作原表多 seed 均值；原表逐题数据尚待溯源。", "",
             "跨数据集固定当前 HotpotQA adapter 参数；不自动采用历史 MuSiQue CLI 的 attention 模式覆盖。", "",
             "| 数据集 | 方法 | 预算 | N | EM (%) | F1 (%) |", "|---|---|---|---:|---:|---:|"]
    for job in jobs:
        directory = root / job["name"]
        status_path = directory / "status.json"
        if job["purpose"] != "main" or not status_path.exists():
            continue
        if json.loads(status_path.read_text()).get("status") != "complete":
            continue
        data = json.loads((directory / "results.json").read_text())
        summary = data["summary"]
        lines.append(f"| {job['dataset']} | {job['method']} | {job['ratio']:.0%} | {len(data['results'])} | {summary['exact_match']:.2f} | {summary['f1_score']:.2f} |")
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--after", type=Path, required=True, help="Predecessor queue directory; must finish successfully")
    parser.add_argument("--datasets", nargs="+", choices=["2wiki", "musique"], default=["2wiki", "musique"])
    parser.add_argument("--methods", nargs="+", choices=list(SOURCES), default=list(SOURCES))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-free-mb", type=int, default=28000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = []
    for dataset in args.datasets:
        for method in args.methods:
            for purpose, count, start in (("smoke", 2, 500), ("main", 500, 0)):
                for ratio in (.2, .5):
                    prefix = "smoke" if purpose == "smoke" else "seed233"
                    jobs.append(dict(dataset=dataset, method=method, purpose=purpose, samples=count,
                                     sample_start=start, ratio=ratio, seed=233,
                                     name=f"{dataset}/{method}/{prefix}/r{int(ratio*100)}/{method}"))
    plan = {"runtime": "restored_original_evaluator_multidataset_v1", "after": str(args.after.resolve()), "jobs": jobs, "gpu": args.gpu,
            "sources": {m: str(SOURCES[m]) for m in args.methods}, "seeds": [233],
            "model": "Qwen2.5-7B-Instruct", "original_results": "retain user-provided paper table; no original-method main reruns"}
    plan_path = args.output / "queue_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError("Changed queue plan; use a new output directory")
    atomic_json(plan_path, plan)
    for method in args.methods:
        validate_source(SOURCES[method])
    for dataset in args.datasets:
        ready = ROOT / "results/rebuttal_20260928/multidataset_preflight" / f"{dataset}.json"
        if not ready.exists() or json.loads(ready.read_text()).get("status") != "passed":
            raise RuntimeError(f"Dataset preflight missing: {ready}")
    summarize(args.output, jobs)
    subprocess.run([sys.executable, "-m", "pip", "freeze"],
                   stdout=(args.output / "pip_freeze.txt").open("w"), check=True)

    def state(status, **extra):
        atomic_json(args.output / "queue_status.json", {"status": status, "supervisor_pid": os.getpid(),
                    "updated": time.time(), "total_jobs": len(jobs), **extra})

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    job = None
    try:
        while True:
            predecessor = args.after / "queue_status.json"
            previous = json.loads(predecessor.read_text()) if predecessor.exists() else {}
            if previous.get("status") == "complete":
                break
            failed = previous.get("status") in ("failed", "invalid_runtime", "deadline_reached")
            state("blocked_by_previous_queue" if failed else "waiting_for_previous_queue",
                  predecessor=str(args.after), predecessor_status=previous.get("status"),
                  completed_jobs=0)
            time.sleep(15)
        for ordinal, job in enumerate(jobs):
            source = SOURCES[job["method"]]
            while not (source / "LEGACY_VALIDATION.json").exists():
                state("waiting_for_legacy_adapter_validation", job=job, completed_jobs=ordinal)
                time.sleep(5)
            hashes, _ = validate_source(source)
            method_root = args.output / job["method"]
            pin_path = method_root / "source_pin.json"
            if pin_path.exists() and json.loads(pin_path.read_text()) != hashes:
                raise RuntimeError(f"Source changed after first job: {source}")
            atomic_json(pin_path, hashes)
            directory = args.output / job["name"]
            status_path = directory / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            if status.get("status") == "complete":
                if job["purpose"] == "smoke":
                    validate_smoke(directory)
                else:
                    validate_result(directory, job["samples"])
                continue
            cmd = [sys.executable, "-u", str(RUNNER),
                   "--source", str(source), "--dataset", job["dataset"], "--method", job["method"], "--ratio", str(job["ratio"]),
                   "--seed", "233", "--samples", str(job["samples"]), "--sample-start", str(job["sample_start"]),
                   "--purpose", job["purpose"], "--output", str(directory)]
            adopted_pid = existing_worker(status_path, cmd)
            if adopted_pid is None and status.get("pid") and (Path("/proc") / str(status["pid"]) / "cmdline").exists():
                raise RuntimeError(f"Existing process does not match requested worker: {status_path}")
            if validate_source(source)[0] != hashes:
                raise RuntimeError("Source changed while waiting for GPU")
            directory.mkdir(parents=True, exist_ok=True)
            if adopted_pid is not None:
                state("running", job=job, completed_jobs=ordinal, child_pid=adopted_pid, adopted_worker=True)
                wait_for_existing_worker(adopted_pid, cmd, status_path)
            else:
                with (directory / "run.log").open("a") as log:
                    # Existing workers are adopted above without being paused.
                    # Serialize this final check+launch with the efficiency
                    # owner's publication so a new worker cannot slip past it.
                    while True:
                        with maintenance_launch_lock(args.gpu):
                            blockers = maintenance_blockers(maintenance_path(args.gpu))
                            if blockers:
                                state("waiting_for_efficiency_maintenance", job=job,
                                      completed_jobs=ordinal, maintenance=blockers)
                            else:
                                free = int(subprocess.check_output(["nvidia-smi", f"--id={args.gpu}",
                                    "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
                                if free >= args.min_free_mb:
                                    if validate_source(source)[0] != hashes:
                                        raise RuntimeError("Source changed while waiting for GPU")
                                    child = subprocess.Popen(cmd, cwd=source, env=env, stdin=subprocess.DEVNULL,
                                                             stdout=log, stderr=subprocess.STDOUT)
                                    break
                                state("waiting_for_gpu_memory", job=job, completed_jobs=ordinal, free_mb=free)
                        time.sleep(5)
                    state("running", job=job, completed_jobs=ordinal, child_pid=child.pid)
                    if child.wait():
                        raise RuntimeError(f"Worker failed: {directory / 'run.log'}")
            if job["purpose"] == "smoke":
                validate_smoke(directory)
            else:
                validate_result(directory, job["samples"])
            summarize(args.output, jobs)
        state("complete", completed_jobs=len(jobs))
    except BaseException as exc:
        state("failed", job=job, error=str(exc))
        raise


if __name__ == "__main__":
    main()
