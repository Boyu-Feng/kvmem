"""Read-only status for the October 1 three-seed extension queues."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "results/rebuttal_multiseed_20261001"
QUEUES = ("seed42/fixed_v2", "seed3407/fixed_v2", "sidequest", "thinkkv_eviction_only")


def show(directory):
    plan_path = directory / "queue_plan.json"
    status_path = directory / "queue_status.json"
    if not plan_path.exists():
        print(f"{directory.name}: not registered")
        return
    plan = json.loads(plan_path.read_text())
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    totals = dict(completed=0, planned=0, saved=0, expected=0)
    for job in plan["jobs"]:
        if job["purpose"] != "main":
            continue
        totals["planned"] += 1
        totals["expected"] += job["samples"]
        job_dir = directory / job["name"]
        job_status = json.loads((job_dir / "status.json").read_text()) if (job_dir / "status.json").exists() else {}
        if job_status.get("status") == "complete":
            totals["completed"] += 1
        cp = job_dir / "checkpoint.json"
        if cp.exists():
            totals["saved"] += len(json.loads(cp.read_text()))
    print(f"{directory}: {status.get('status', 'pending')}; "
          f"main {totals['completed']}/{totals['planned']}; "
          f"saved {totals['saved']}/{totals['expected']}; "
          f"current={status.get('job', {}).get('name', '—')}")
    if status.get("error"):
        print(f"  error: {status['error']}")
    if status.get("preflight_status"):
        print(f"  preflight: {status['preflight_status']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT)
    args = parser.parse_args()
    for name in QUEUES:
        show(args.root / name)


if __name__ == "__main__":
    main()
