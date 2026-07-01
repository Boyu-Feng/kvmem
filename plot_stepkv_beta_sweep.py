#!/usr/bin/env python3
"""
Re-plot StepKV step-score weight (beta) sweep from summary.json.

Expected JSON: output of run_stepkv_stepscore_weight_sweep.py
  results/stepkv_stepscore_weight_sweep/summary.json

Usage:
  python plot_stepkv_beta_sweep.py

  python plot_stepkv_beta_sweep.py \\
    --input_json results/stepkv_stepscore_weight_sweep/summary.json \\
    --labelsize 18 --ticksize 14
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required. Install with: pip install matplotlib"
    ) from exc


DEFAULT_INPUT_JSON = "results/stepkv_stepscore_weight_sweep/summary.json"
DEFAULT_OUTPUT_PNG = "results/stepkv_stepscore_weight_sweep/stepscore_weight_curve_replot.png"
DEFAULT_COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]
RATIO_TAGS = [("0.2", "r20"), ("0.5", "r50")]
MARKERS = {"r20": "o", "r50": "s"}


def _resolve_json_path(path: str) -> str:
    if os.path.isfile(path):
        return path
    if not path.endswith(".json") and os.path.isfile(path + ".json"):
        return path + ".json"
    return path


def load_summary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "datasets" not in data:
        raise ValueError(f"Invalid beta sweep summary JSON: {path}")
    return data


def _collect_betas(summary: Dict[str, Any]) -> List[float]:
    if isinstance(summary.get("betas"), list) and summary["betas"]:
        return sorted(float(b) for b in summary["betas"])

    found = set()
    for ds_data in summary.get("datasets", {}).values():
        if not isinstance(ds_data, dict):
            continue
        for rdata in ds_data.values():
            if not isinstance(rdata, dict):
                continue
            for key in (rdata.get("betas") or {}).keys():
                try:
                    found.add(float(key))
                except ValueError:
                    continue
    if not found:
        raise ValueError("No beta values found in summary JSON.")
    return sorted(found)


def _collect_ratio_tags(summary: Dict[str, Any]) -> List[Tuple[str, str]]:
    found = set()
    for ds_data in summary.get("datasets", {}).values():
        if not isinstance(ds_data, dict):
            continue
        for rtag, rdata in ds_data.items():
            if not isinstance(rdata, dict):
                continue
            ratio = rdata.get("cache_ratio")
            if ratio is not None:
                found.add((f"{float(ratio):g}", str(rtag)))
            else:
                found.add((str(rtag), str(rtag)))
    if found:
        return sorted(found, key=lambda x: float(x[0]))
    return list(RATIO_TAGS)


def _setup_style(labelsize: float, ticksize: float, legend_size: float) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": ticksize,
            "axes.labelsize": labelsize,
            "xtick.labelsize": ticksize,
            "ytick.labelsize": ticksize,
            "legend.fontsize": legend_size,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _series_list(summary: Dict[str, Any], betas: List[float], ratio_tags: List[Tuple[str, str]]):
    series = []
    ci = 0
    for ds, ds_data in summary.get("datasets", {}).items():
        if not isinstance(ds_data, dict):
            continue
        for ratio_str, rtag in ratio_tags:
            if rtag not in ds_data:
                continue
            rblock = ds_data[rtag]
            beta_map = rblock.get("betas") or {}
            em_vals: List[float] = []
            f1_vals: List[float] = []
            for b in betas:
                row = beta_map.get(f"{b:g}")
                if row is None:
                    em_vals.append(float("nan"))
                    f1_vals.append(float("nan"))
                else:
                    em_vals.append(float(row.get("em", float("nan"))))
                    f1_vals.append(float(row.get("f1", float("nan"))))
            series.append(
                {
                    "label": f"{ds} (ratio={ratio_str})",
                    "rtag": rtag,
                    "color": DEFAULT_COLORS[ci % len(DEFAULT_COLORS)],
                    "em": em_vals,
                    "f1": f1_vals,
                }
            )
            ci += 1
    return series


def plot_beta_sweep(
    summary: Dict[str, Any],
    output_png: str,
    *,
    style: str = "line",
    labelsize: float = 18,
    ticksize: float = 14,
    legend_size: float = 12,
    show_title: bool = False,
) -> None:
    betas = _collect_betas(summary)
    ratio_tags = _collect_ratio_tags(summary)
    series = _series_list(summary, betas, ratio_tags)
    if not series:
        raise ValueError("No plottable series found in summary JSON.")

    _setup_style(labelsize, ticksize, legend_size)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))
    x = list(range(len(betas)))
    x_labels = [f"{b:g}" for b in betas]

    for ax, metric in zip(axes, ("em", "f1")):
        if style == "bar":
            n_series = len(series)
            total_w = 0.78
            bar_w = total_w / max(1, n_series)
            for i, s in enumerate(series):
                vals = s[metric]
                shift = (i - (n_series - 1) / 2.0) * bar_w
                pos = [xi + shift for xi in x]
                ax.bar(
                    pos,
                    vals,
                    width=bar_w * 0.88,
                    label=s["label"],
                    color=s["color"],
                    edgecolor="white",
                    linewidth=0.6,
                )
        else:
            for s in series:
                vals = s[metric]
                ax.plot(
                    x,
                    vals,
                    marker=MARKERS.get(s["rtag"], "o"),
                    color=s["color"],
                    linewidth=2.0,
                    markersize=8,
                    label=s["label"],
                )

        ax.set_xticks(x, x_labels, fontsize=ticksize)
        ax.set_xlabel(r"step-score weight $\beta$", fontsize=labelsize)
        ax.set_ylabel(metric.upper(), fontsize=labelsize)
        ax.tick_params(axis="y", labelsize=ticksize)
        ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.6)
        ax.set_axisbelow(True)
        if show_title:
            ax.set_title(f"{metric.upper()} vs $\\beta$", fontsize=labelsize)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        ncol=min(4, max(1, len(series))),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=legend_size,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)

    out_dir = os.path.dirname(os.path.abspath(output_png)) or "."
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_png)
    pdf_path = os.path.splitext(output_png)[0] + ".pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[INFO] Saved: {output_png}")
    print(f"[INFO] Saved: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-plot StepKV beta sweep from summary.json.")
    parser.add_argument(
        "--input_json",
        type=str,
        default=DEFAULT_INPUT_JSON,
        help=f"Path to beta sweep summary.json (default: {DEFAULT_INPUT_JSON})",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default=DEFAULT_OUTPUT_PNG,
        help=f"Output figure path (default: {DEFAULT_OUTPUT_PNG})",
    )
    parser.add_argument(
        "--style",
        choices=["line", "bar"],
        default="line",
        help="line: marker+line (default); bar: grouped bars per beta (discrete).",
    )
    parser.add_argument("--labelsize", type=float, default=18, help="Axis label font size.")
    parser.add_argument("--ticksize", type=float, default=14, help="Tick label font size.")
    parser.add_argument("--legend_size", type=float, default=12, help="Legend font size.")
    parser.add_argument("--show_title", action="store_true", help="Show subplot titles.")
    args = parser.parse_args()

    input_json = _resolve_json_path(os.path.abspath(args.input_json))
    if not os.path.isfile(input_json):
        raise FileNotFoundError(f"JSON not found: {input_json}")

    summary = load_summary(input_json)
    betas = _collect_betas(summary)
    print(f"[INFO] Loaded beta sweep from {input_json}")
    print(f"[INFO] betas={betas}, datasets={list(summary.get('datasets', {}).keys())}")

    plot_beta_sweep(
        summary,
        args.output_png,
        style=args.style,
        labelsize=args.labelsize,
        ticksize=args.ticksize,
        legend_size=args.legend_size,
        show_title=args.show_title,
    )


if __name__ == "__main__":
    main()
