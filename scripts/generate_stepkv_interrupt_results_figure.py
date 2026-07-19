#!/usr/bin/env python3
"""
Generate bar-chart figure from StepKV step-interruption ablation results.

Reads results/stepkv_interrupt_ablation_2wiki/summary.json and writes PNG + PDF.

Usage:
    python scripts/generate_stepkv_interrupt_results_figure.py
    python scripts/generate_stepkv_interrupt_results_figure.py \\
        --summary results/stepkv_interrupt_ablation_2wiki/summary.json \\
        -o assets/stepkv_interrupt_results
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


# Palette aligned with other StepKV figures
C_EM = "#4C72B0"
C_F1 = "#55A868"
C_BASELINE = "#888888"
C_TEXT = "#333333"
C_MUTED = "#666666"

MODE_LABELS = {
    "none": "Full StepKV",
    "lag1": "lag-1",
    "lag2": "lag-2",
    "lag3": "lag-3",
    "window2": "window-2",
}


def _load_summary(path: str) -> Dict[str, Any]:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Summary not found: {path}\n"
            "Run interrupt ablation first:\n"
            "  python run_stepkv_interrupt_ablation_2wiki.py --num_samples 100 --seed 42"
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _extract_rows(summary: Dict[str, Any]) -> Tuple[List[str], List[float], List[float]]:
    mode_order: List[str] = list(summary.get("mode_order") or [])
    runs: Dict[str, Any] = summary.get("runs") or {}
    if not mode_order:
        mode_order = sorted(runs.keys(), key=lambda m: (m != "none", m))

    labels: List[str] = []
    ems: List[float] = []
    f1s: List[float] = []
    for mode in mode_order:
        row = runs.get(mode)
        if not row:
            continue
        labels.append(MODE_LABELS.get(mode, mode))
        ems.append(float(row["em"]))
        f1s.append(float(row["f1"]))
    if not labels:
        raise ValueError(f"No runs found in summary: {summary}")
    return labels, ems, f1s


def generate_results_figure(summary: Dict[str, Any], output_base: str) -> None:
    labels, ems, f1s = _extract_rows(summary)
    n = len(labels)
    x = np.arange(n)
    width = 0.34

    fig, ax = plt.subplots(figsize=(max(6.0, 1.35 * n + 2.0), 4.6))

    bars_em = ax.bar(x - width / 2, ems, width, label="EM", color=C_EM, edgecolor="white", linewidth=0.8)
    bars_f1 = ax.bar(x + width / 2, f1s, width, label="F1", color=C_F1, edgecolor="white", linewidth=0.8)

    ymax = max(max(ems), max(f1s))
    ax.set_ylim(0, min(100, ymax + 12))
    ax.set_ylabel("Score (%)", fontsize=11, color=C_TEXT)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.35, linewidth=0.8)
    ax.set_axisbelow(True)

    for bar in list(bars_em) + list(bars_f1):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + 0.8,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=C_TEXT,
        )

    baseline_em = ems[0] if labels[0] == MODE_LABELS["none"] else None
    if baseline_em is not None:
        ax.axhline(baseline_em, color=C_BASELINE, linestyle=":", linewidth=1.0, alpha=0.7)
        for i in range(1, n):
            dem = ems[i] - baseline_em
            ax.text(
                x[i],
                max(ems[i], f1s[i]) + 5.5,
                f"ΔEM {dem:+.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color=C_MUTED,
            )

    ax.legend(loc="upper right", frameon=False, fontsize=10)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_base) or ".", exist_ok=True)
    for ext in ("png", "pdf"):
        path = f"{output_base}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300 if ext == "png" else None)
        print(f"[INFO] Wrote {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate StepKV interrupt ablation results bar chart."
    )
    parser.add_argument(
        "--summary",
        default="results/stepkv_interrupt_ablation_2wiki/summary.json",
        help="Path to summary.json from interrupt ablation",
    )
    parser.add_argument(
        "-o",
        "--output-base",
        default="assets/stepkv_interrupt_results",
        help="Output path without extension (writes .png and .pdf)",
    )
    args = parser.parse_args()

    summary = _load_summary(args.summary)
    generate_results_figure(summary, args.output_base)


if __name__ == "__main__":
    main()
