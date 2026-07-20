#!/usr/bin/env python3
"""
Replot H2O / TOVA / StepKV drop-token figures from saved analyze JSON.

No GPU, no model, no re-running episodes — only reads the JSON written by
analyze_h2o_dropped_tokens_hotpotqa.py and regenerates PDFs.

Usage:
    python scripts/replot_h2o_drop_figures.py \\
        --input-json results/h2o_drop_analysis/hotpot_drop_tokens_sample0_20250720_120000.json

    python scripts/replot_h2o_drop_figures.py \\
        --input-json results/h2o_drop_analysis/hotpot_drop_tokens_sample0_20250720_120000.json \\
        --only dropped cohort \\
        --output-dir assets/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

METHOD_ORDER: List[Tuple[str, str]] = [
    ("h2o", "H2O"),
    ("tova", "TOVA"),
    ("step_aware_h2o", "StepKV"),
]


def _load_method_plot_data(path: str) -> Tuple[List[Tuple[str, Dict[str, Any]]], int, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        blob = json.load(f)

    meta = blob.get("meta") or {}
    results = blob.get("method_results") or {}
    method_plot_data: List[Tuple[str, Dict[str, Any]]] = []

    for mode, default_name in METHOD_ORDER:
        row = results.get(mode)
        if not row:
            continue
        display_name = str(row.get("display_name") or default_name)
        plot_data = row.get("plot_data")
        if not isinstance(plot_data, dict):
            raise ValueError(f"Missing plot_data for method {mode!r} in {path}")
        method_plot_data.append((display_name, plot_data))

    if not method_plot_data:
        raise ValueError(f"No method_results with plot_data found in {path}")

    max_steps = int(meta.get("max_steps") or 7)
    return method_plot_data, max_steps, meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replot evicted/cohort figures from saved h2o drop analysis JSON."
    )
    parser.add_argument(
        "--input-json",
        required=True,
        help="Path to hotpot_drop_tokens_sample*.json from analyze_h2o_dropped_tokens_hotpotqa.py",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output PDFs (default: same dir as input JSON)",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix (default: input basename without .json)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=("dropped", "kept", "cohort", "survival", "dynamics", "all"),
        default=("dropped", "cohort"),
        help="Which figures to regenerate (default: dropped cohort)",
    )
    args = parser.parse_args()

    input_json = os.path.abspath(args.input_json)
    if not os.path.isfile(input_json):
        raise FileNotFoundError(input_json)

    method_plot_data, max_steps, _meta = _load_method_plot_data(input_json)
    output_dir = os.path.abspath(args.output_dir or os.path.dirname(input_json))
    os.makedirs(output_dir, exist_ok=True)

    base_prefix = args.prefix or os.path.splitext(os.path.basename(input_json))[0]
    only = set(args.only)
    if "all" in only:
        only = {"dropped", "kept", "cohort", "survival", "dynamics"}

    from analyze_h2o_dropped_tokens_hotpotqa import (
        _plot_cohort_survival,
        _plot_dropped_three_methods,
        _plot_final_survival_line,
        _plot_kept_three_methods,
        _plot_survival_dynamics,
    )

    outputs: List[str] = []
    if "dropped" in only:
        path = os.path.join(output_dir, f"{base_prefix}_dropped.pdf")
        _plot_dropped_three_methods(method_plot_data, path, max_react_steps=max_steps)
        outputs.append(path)
    if "kept" in only:
        path = os.path.join(output_dir, f"{base_prefix}_kept.pdf")
        _plot_kept_three_methods(method_plot_data, path, max_react_steps=max_steps)
        outputs.append(path)
    if "cohort" in only:
        path = os.path.join(output_dir, f"{base_prefix}_cohort_survival.pdf")
        _plot_cohort_survival(method_plot_data, path, max_react_steps=max_steps)
        outputs.append(path)
    if "survival" in only:
        path = os.path.join(output_dir, f"{base_prefix}_final_survival.pdf")
        _plot_final_survival_line(method_plot_data, path, max_react_steps=max_steps)
        outputs.append(path)
    if "dynamics" in only:
        path = os.path.join(output_dir, f"{base_prefix}_survival_dynamics.pdf")
        _plot_survival_dynamics(method_plot_data, path, max_react_steps=max_steps)
        outputs.append(path)

    for path in outputs:
        print(f"[INFO] Wrote {path}")


if __name__ == "__main__":
    main()
