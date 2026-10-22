"""Defer all MuSiQue jobs while GPU7 maintenance holds; retain frozen QA32 code."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_parallel_qa32_queue as scheduler
from scripts.latency_runtime import file_hash, patch_attribute
from scripts.run_rebuttal_efficiency_queue import (
    maintenance_blockers, maintenance_path, read_maintenance,
)


def priority_state():
    blockers = maintenance_blockers(maintenance_path(7))
    record = read_maintenance(maintenance_path(7))
    return {"held_datasets": ["musique"] if blockers else [],
            "gpu7_maintenance": blockers,
            "maintenance_output": record.get("output") if record is not None else None,
            "resume_policy": "All original candidates resume after maintenance owner/child release"}


def selected_dataset(roots, completed, order):
    for dataset in order:
        jobs = [(root, job) for root, plan in roots.items() for job in plan["jobs"]
                if job["dataset"] == dataset and job["purpose"] == "main"]
        if not jobs:
            raise RuntimeError(f"Priority dataset has no formal jobs: {dataset}")
        if any((str(root), job["name"]) not in completed for root, job in jobs):
            return dataset
    return None


def install_priority(scope, verify, order=(), target=None):
    target = scheduler if target is None else target
    original_ready, original_publish = target.ready_jobs, target.publish

    def ready(*args, **kwargs):
        verify()
        candidates = original_ready(*args, **kwargs)
        if order:
            roots = kwargs["roots"] if "roots" in kwargs else args[0]
            completed = kwargs["completed"] if "completed" in kwargs else args[1]
            target = selected_dataset(roots, completed, order)
            candidates = [(root, job) for root, job in candidates if job["dataset"] == target]
        if priority_state()["held_datasets"]:
            return [(root, job) for root, job in candidates if job["dataset"] != "musique"]
        return candidates

    def publish(*args, **kwargs):
        verify()
        priority = {**priority_state(), "dataset_order": list(order)}
        if order:
            roots = kwargs["roots"] if "roots" in kwargs else args[1]
            completed = kwargs["completed"] if "completed" in kwargs else args[3]
            priority["selected_dataset"] = selected_dataset(roots, completed, order)
        return original_publish(*args, **kwargs, latency_priority=priority)

    scope.enter_context(patch_attribute(target, "ready_jobs", ready))
    scope.enter_context(patch_attribute(target, "publish", publish))


def main():
    import contextlib
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-scheduler", type=Path, required=True)
    parser.add_argument("--dataset-order", nargs="+", choices=scheduler.DATASETS)
    wrapper, remaining = parser.parse_known_args()
    original_path = Path(scheduler.__file__).resolve()
    if wrapper.source_scheduler.resolve() != original_path:
        raise ValueError("Require the unchanged original QA32 scheduler")
    order = tuple(wrapper.dataset_order or ())
    if order and (len(order) != len(scheduler.DATASETS) or set(order) != set(scheduler.DATASETS)):
        raise ValueError("Priority order must contain each frozen dataset exactly once")
    original_sha, overlay_sha = file_hash(original_path), file_hash(__file__)

    def verify():
        if file_hash(original_path) != original_sha or file_hash(__file__) != overlay_sha:
            raise RuntimeError("QA32/ThinkKV handoff source changed during execution")

    args = scheduler.parser().parse_args(remaining)
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=True)
    provenance = {"protocol": "qa32_gpu7_latency_priority_overlay_v1",
                  "source_scheduler": str(original_path), "source_sha256": original_sha,
                  "overlay": str(Path(__file__).resolve()), "overlay_sha256": overlay_sha,
                  "held_dataset": "musique", "maintenance_gpu": 7,
                  "dataset_order": list(order),
                  "experimental_changes": None,
                  "scope": "Scheduling only; unchanged workers, fingerprints, data, seeds and precision"}
    path = output / "thinkkv_priority_overlay.json"
    if path.exists() and scheduler.runner.read_json(path) != provenance:
        raise RuntimeError("Handoff scheduling provenance changed; use a new output")
    scheduler.runner.atomic_json(path, provenance)
    with contextlib.ExitStack() as scope:
        install_priority(scope, verify, order)
        scope.enter_context(patch_attribute(sys, "argv", [str(original_path), *remaining]))
        scheduler.main()


if __name__ == "__main__":
    main()
