"""Explicit dataset-only extension; leave the live HotpotQA latency sources untouched."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_latency_experiment as runner
from scripts.latency_datasets import ADDITIONAL_DATASETS, dataset_scope


def main(argv=None):
    parser = runner.build_parser()
    parser.description = __doc__
    parser.add_argument("--dataset", choices=ADDITIONAL_DATASETS, required=True)
    args = parser.parse_args(argv)
    plan = {**runner.prepare_plan(args), "dataset": args.dataset,
            "dataset_config_policy": "Frozen common QA defaults; pinned normalized dev rows"}
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    with dataset_scope(args.dataset):
        runner.execute(args, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
