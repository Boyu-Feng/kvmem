"""Show the persistent queue and per-condition checkpoint progress."""
import argparse
import json
from pathlib import Path


def show_queue(root):
    if not (root / "queue_status.json").exists():
        print(f"{root.name}: no queue status yet")
        return
    status = json.loads((root / "queue_status.json").read_text())
    print(f"Results: {root}")
    print(f"Queue: {status['status']}; supervisor PID: {status['supervisor_pid']}")
    if "job" in status:
        print(f"Current: {status['job']['name']}; process: {status.get('child_pid')}")
    if "error" in status:
        print(f"Error: {status['error']}")
    if "predecessor" in status:
        print(f"After: {status['predecessor']} ({status.get('predecessor_status')})")
    plan = json.loads((root / "queue_plan.json").read_text())
    complete_main_samples = 0
    for job in plan["jobs"]:
        directory = root / job["name"]
        cp = directory / "checkpoint.json"
        st = directory / "status.json"
        if not st.exists():
            continue
        state = json.loads(st.read_text())["status"]
        count = len(json.loads(cp.read_text())) if cp.exists() else 0
        if job["purpose"] == "main":
            complete_main_samples += count
        print(f"{job['name']}: {state}; {count}/{job['samples']} samples")
    print(f"Saved main question-condition evaluations: {complete_main_samples}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    args = parser.parse_args()
    base = Path(__file__).resolve().parents[1] / "results/rebuttal_20260928"
    roots = [args.root] if args.root else [base / name for name in (
        "legacy_baselines", "legacy_step_controls", "multidataset_baselines", "multidataset_step_controls",
        "sidequest_adaptive_v1")]
    for root in roots:
        show_queue(root)
    if not args.root:
        pilot = base / "sidequest_pilot_v1"
        if (pilot / "status.json").exists():
            state = json.loads((pilot / "status.json").read_text())
            print(f"SideQuest pilot: {state['status']}; PID: {state.get('pid')}")
            if (pilot / "SUMMARY.json").exists():
                summary = json.loads((pilot / "SUMMARY.json").read_text())
                print(f"Paired samples: {summary['paired_samples']}/{summary['expected_pairs']}")
        thinkkv = base / 'thinkkv_candidate_v1'
        if (thinkkv / 'experiment_plan.json').exists():
            calibration = base / 'thinkkv_preflight/calibration_qa20'
            state = json.loads((calibration / 'status.json').read_text()) if (calibration / 'status.json').exists() else {}
            checkpoint = calibration / 'checkpoint.json'
            count = len(json.loads(checkpoint.read_text())) if checkpoint.exists() else 0
            print(f"ThinkKV candidate: calibration {state.get('status', 'pending')}; {count}/20 prompts")
            if (calibration / 'calibration.json').exists():
                report = json.loads((calibration / 'calibration.json').read_text())
                print(f"Calibration gate: {report['status']}; common three-mode layers: {report['eligible_layers']}")
            print('ThinkKV main: 6 registered conditions (3 datasets x 20%/50%); adapter/controller pending; 0 completed')


if __name__ == "__main__":
    main()
