#!/usr/bin/env python3
"""
Replot H2O / TOVA / StepKV drop-token figures from saved analyze JSON.

No GPU, no model, no re-running episodes — only reads the JSON written by
analyze_h2o_dropped_tokens_hotpotqa.py and regenerates PDFs.

Usage:
    # list available data files under the default output dir
    python scripts/replot_h2o_drop_figures.py --list

    # replot the newest sample JSON in results/h2o_drop_analysis/
    python scripts/replot_h2o_drop_figures.py --latest

    # replot every sample JSON in that directory
    python scripts/replot_h2o_drop_figures.py --input-dir results/h2o_drop_analysis --all

    # replot one specific JSON
    python scripts/replot_h2o_drop_figures.py \\
        --input-json results/h2o_drop_analysis/hotpot_drop_tokens_sample0_20250720_120000.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

DEFAULT_INPUT_DIR = "results/h2o_drop_analysis"
DATA_JSON_GLOB = "hotpot_drop_tokens_sample*.json"

METHOD_ORDER: List[Tuple[str, str]] = [
    ("h2o", "H2O"),
    ("tova", "TOVA"),
    ("step_aware_h2o", "StepKV"),
]


def _is_data_json(path: str) -> bool:
    name = os.path.basename(path)
    if not name.startswith("hotpot_drop_tokens_sample") or not name.endswith(".json"):
        return False
    # Exclude sidecar artifacts if any were named similarly.
    if name.endswith("_plot_error.json"):
        return False
    return True


def _find_data_jsons(input_dir: str) -> List[str]:
    pattern = os.path.join(input_dir, DATA_JSON_GLOB)
    paths = [p for p in glob.glob(pattern) if _is_data_json(p)]
    return sorted(paths, key=lambda p: os.path.getmtime(p))


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


def _replot_one(
    input_json: str,
    *,
    output_dir: str | None,
    prefix: str | None,
    only: set[str],
) -> List[str]:
    from analyze_h2o_dropped_tokens_hotpotqa import (
        _plot_cohort_survival,
        _plot_dropped_three_methods,
        _plot_final_survival_line,
        _plot_kept_three_methods,
        _plot_survival_dynamics,
    )

    input_json = os.path.abspath(input_json)
    method_plot_data, max_steps, _meta = _load_method_plot_data(input_json)
    out_dir = os.path.abspath(output_dir or os.path.dirname(input_json))
    os.makedirs(out_dir, exist_ok=True)

    base_prefix = prefix or os.path.splitext(os.path.basename(input_json))[0]
    outputs: List[str] = []

    if "dropped" in only:
        path = os.path.join(out_dir, f"{base_prefix}_dropped.pdf")
        _plot_dropped_three_methods(method_plot_data, path, max_react_steps=max_steps)
        outputs.append(path)
    if "kept" in only:
        path = os.path.join(out_dir, f"{base_prefix}_kept.pdf")
        _plot_kept_three_methods(method_plot_data, path, max_react_steps=max_steps)
        outputs.append(path)
    if "cohort" in only:
        path = os.path.join(out_dir, f"{base_prefix}_cohort_survival.pdf")
        _plot_cohort_survival(method_plot_data, path, max_react_steps=max_steps)
        outputs.append(path)
    if "survival" in only:
        path = os.path.join(out_dir, f"{base_prefix}_final_survival.pdf")
        _plot_final_survival_line(method_plot_data, path, max_react_steps=max_steps)
        outputs.append(path)
    if "dynamics" in only:
        path = os.path.join(out_dir, f"{base_prefix}_survival_dynamics.pdf")
        _plot_survival_dynamics(method_plot_data, path, max_react_steps=max_steps)
        outputs.append(path)

    return outputs


def _print_available_jsons(input_dir: str) -> None:
    paths = _find_data_jsons(input_dir)
    if not paths:
        print(f"[INFO] No data JSON found under {input_dir}")
        print(f"[INFO] Expected files like: {DATA_JSON_GLOB}")
        print("[INFO] PDF/txt/jsonl in the same folder are outputs; replot needs the main .json.")
        return
    print(f"[INFO] Found {len(paths)} data JSON(s) under {input_dir}:")
    for path in paths:
        mtime = os.path.getmtime(path)
        print(f"  - {path}  (mtime={mtime:.0f})")
    print("[INFO] Use --latest to replot the newest one, or --all to replot all.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replot evicted/cohort figures from saved h2o drop analysis JSON."
    )
    parser.add_argument(
        "--input-json",
        default=None,
        help="Path to one hotpot_drop_tokens_sample*.json",
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Directory to scan for data JSON (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Replot the newest data JSON under --input-dir",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Replot every data JSON under --input-dir",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available data JSON files and exit",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output PDFs (default: same dir as each input JSON)",
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

    input_dir = os.path.abspath(args.input_dir)

    if args.list:
        _print_available_jsons(input_dir)
        return

    only = set(args.only)
    if "all" in only:
        only = {"dropped", "kept", "cohort", "survival", "dynamics"}

    if args.input_json:
        input_jsons = [os.path.abspath(args.input_json)]
        if not os.path.isfile(input_jsons[0]):
            raise FileNotFoundError(input_jsons[0])
    elif args.all:
        input_jsons = _find_data_jsons(input_dir)
        if not input_jsons:
            raise FileNotFoundError(
                f"No data JSON under {input_dir}. Run with --list to inspect the folder."
            )
    elif args.latest:
        input_jsons = _find_data_jsons(input_dir)
        if not input_jsons:
            raise FileNotFoundError(
                f"No data JSON under {input_dir}. Run with --list to inspect the folder."
            )
        input_jsons = [input_jsons[-1]]
        print(f"[INFO] Using latest: {input_jsons[0]}")
    else:
        parser.error("Specify --input-json, --latest, --all, or --list")

    for input_json in input_jsons:
        print(f"[INFO] Replot from {input_json}")
        outputs = _replot_one(
            input_json,
            output_dir=args.output_dir,
            prefix=args.prefix,
            only=only,
        )
        for path in outputs:
            print(f"[INFO] Wrote {path}")


if __name__ == "__main__":
    main()
