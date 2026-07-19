#!/usr/bin/env python3
"""
Generate paper-ready figure for StepKV step-interruption ablation (lag2 vs none).

Outputs PNG + PDF under assets/ by default.

Usage:
    python scripts/generate_stepkv_interrupt_figure.py
    python scripts/generate_stepkv_interrupt_figure.py -o assets/stepkv_interrupt_lag2
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# Palette aligned with other StepKV figures
C_Q = "#CCCCCC"
C_STEP = "#FFF2CC"
C_STEP_STROKE = "#D6B656"
C_DROP = "#F4CCCC"
C_DROP_STROKE = "#C00000"
C_AGENT = "#B4A7D6"
C_LLM = "#E8E8E8"
C_TEXT = "#222222"
C_MUTED = "#666666"
C_ARROW = "#9673A6"


def _lag2_dropped_through(enter_step: int) -> List[int]:
    """Cumulative step ids hard-dropped just before entering enter_step."""
    dropped = []
    for t in range(3, enter_step + 1):
        sid = t - 2
        if sid >= 1 and sid not in dropped:
            dropped.append(sid)
    return dropped


def _visible_steps(enter_step: int, max_step: int, mode: str) -> Tuple[List[int], List[int]]:
    all_steps = list(range(1, max_step + 1))
    if mode == "none":
        return all_steps, []
    dropped = _lag2_dropped_through(enter_step)
    visible = [s for s in all_steps if s not in dropped]
    return visible, dropped


def _draw_step_chip(ax, x: float, y: float, w: float, h: float, label: str, *,
                    face: str, edge: str, alpha: float = 1.0, strike: bool = False) -> None:
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=edge,
        facecolor=face,
        alpha=alpha,
    )
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=9, color=C_TEXT, fontweight="bold" if not strike else "normal")
    if strike:
        ax.plot([x + 0.05, x + w - 0.05], [y + h / 2, y + h / 2],
                color=C_DROP_STROKE, linewidth=2.2, solid_capstyle="round")


def _draw_kv_row(ax, y: float, row_h: float, enter_step: int, max_step: int, mode: str,
                 row_label: str) -> None:
    q_w, step_w, gap = 0.55, 0.72, 0.08
    x0 = 0.2
    ax.text(0.02, y + row_h / 2, row_label, ha="left", va="center",
            fontsize=10, fontweight="bold", color=C_TEXT)

    visible, dropped = _visible_steps(enter_step, max_step, mode)

    x = x0
    _draw_step_chip(ax, x, y, q_w, row_h, "Q", face=C_Q, edge="#888888")
    x += q_w + gap

    for sid in range(1, max_step + 1):
        if sid in visible:
            _draw_step_chip(ax, x, y, step_w, row_h, f"Step {sid}",
                            face=C_STEP, edge=C_STEP_STROKE)
        else:
            _draw_step_chip(ax, x, y, step_w, row_h, f"Step {sid}",
                            face=C_DROP, edge=C_DROP_STROKE, alpha=0.85, strike=True)
        x += step_w + gap

    if mode == "lag2" and enter_step >= 3:
        drop_sid = enter_step - 2
        ax.annotate(
            f"drop Step {drop_sid}",
            xy=(x0 + q_w + gap + (drop_sid - 1) * (step_w + gap) + step_w / 2, y + row_h),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center", va="bottom",
            fontsize=8, color=C_DROP_STROKE,
            arrowprops=dict(arrowstyle="->", color=C_DROP_STROKE, lw=1.0),
        )


def generate_main_figure(output_base: str, max_step: int = 5) -> None:
    """Snapshot figure: none vs lag2 at enter steps 3..max_step."""
    snapshots = list(range(3, max_step + 1))
    n = len(snapshots)
    fig_w = 2.2 * n + 1.0
    fig, axes = plt.subplots(2, n, figsize=(fig_w, 4.8), sharex=False, sharey=True)
    if n == 1:
        axes = axes.reshape(2, 1)

    fig.suptitle("Continuous Step Interruption vs Full StepKV", fontsize=14, fontweight="bold", y=0.98)

    for col, enter_step in enumerate(snapshots):
        for row, (mode, title) in enumerate(
            [("none", "Full StepKV (none)"), ("lag2", "Interrupt lag2")]
        ):
            ax = axes[row, col]
            ax.set_xlim(0, 6.2)
            ax.set_ylim(0, 1.2)
            ax.axis("off")
            _draw_kv_row(ax, 0.35, 0.55, enter_step, max_step, mode, title)
            if row == 0:
                ax.text(3.1, 1.05, f"Enter Step {enter_step}", ha="center", va="bottom",
                        fontsize=11, fontweight="bold", color=C_ARROW)
            if col == 0 and row == 1:
                ax.text(-0.05, 0.65, "Agent still\ndecoding →",
                        ha="right", va="center", fontsize=9, color=C_MUTED)

    legend_handles = [
        mpatches.Patch(facecolor=C_STEP, edgecolor=C_STEP_STROKE, label="Retained step KV"),
        mpatches.Patch(facecolor=C_DROP, edgecolor=C_DROP_STROKE, label="Hard-dropped step (lag2)"),
        mpatches.Patch(facecolor=C_Q, edgecolor="#888888", label="Question (protected)"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.02))

    note = (
        "lag2: on entering step t, forcibly remove entire span of step (t−2); "
        "generation continues (no backfill)."
    )
    fig.text(0.5, 0.07, note, ha="center", va="bottom", fontsize=9, color=C_MUTED)

    os.makedirs(os.path.dirname(output_base) or ".", exist_ok=True)
    fig.tight_layout(rect=[0, 0.10, 1, 0.94])
    for ext in ("png", "pdf"):
        path = f"{output_base}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300 if ext == "png" else None)
        print(f"[INFO] Wrote {path}")
    plt.close(fig)


def generate_policy_figure(output_base: str, enter_step: int = 4, max_step: int = 5) -> None:
    """Compare none / lag1 / lag2 / window2 at one time point."""
    policies: Dict[str, str] = {
        "none": "none",
        "lag1": "lag1",
        "lag2": "lag2",
        "window2": "window2",
    }

    def cumulative_lag_drops(enter_step: int, lag: int) -> List[int]:
        dropped: List[int] = []
        for t in range(2, enter_step + 1):
            sid = t - lag
            if sid >= 1 and sid not in dropped:
                dropped.append(sid)
        return dropped

    def visible_for(mode: str) -> Tuple[List[int], List[int]]:
        if mode == "none":
            return list(range(1, max_step + 1)), []
        if mode.startswith("lag"):
            lag = int(mode[3:])
            dropped = cumulative_lag_drops(enter_step, lag)
            visible = [s for s in range(1, max_step + 1) if s not in dropped]
            return visible, dropped
        if mode.startswith("window"):
            w = int(mode[6:])
            dropped = [s for s in range(1, max_step + 1) if s < enter_step - w]
            visible = [s for s in range(1, max_step + 1) if s not in dropped]
            return visible, dropped
        return list(range(1, max_step + 1)), []

    fig, axes = plt.subplots(len(policies), 1, figsize=(8.5, 5.2))
    fig.suptitle(f"Interrupt Policies at Enter Step {enter_step}", fontsize=13, fontweight="bold")

    for ax, (name, mode) in zip(axes, policies.items()):
        ax.set_xlim(0, 6.2)
        ax.set_ylim(0, 1.0)
        ax.axis("off")
        visible, dropped = visible_for(mode)
        q_w, step_w, gap = 0.55, 0.72, 0.08
        x0 = 1.05
        ax.text(0.02, 0.5, name, ha="left", va="center", fontsize=10, fontweight="bold")
        x = x0
        _draw_step_chip(ax, x, 0.25, q_w, 0.5, "Q", face=C_Q, edge="#888888")
        x += q_w + gap
        for sid in range(1, max_step + 1):
            if sid in visible:
                _draw_step_chip(ax, x, 0.25, step_w, 0.5, str(sid), face=C_STEP, edge=C_STEP_STROKE)
            else:
                _draw_step_chip(ax, x, 0.25, step_w, 0.5, str(sid),
                                face=C_DROP, edge=C_DROP_STROKE, strike=True)
            x += step_w + gap

    fig.text(
        0.5, 0.02,
        "lag1: drop t−1  |  lag2: drop t−2  |  window2: drop all id < t−2",
        ha="center", fontsize=9, color=C_MUTED,
    )
    os.makedirs(os.path.dirname(output_base) or ".", exist_ok=True)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    for ext in ("png", "pdf"):
        path = f"{output_base}.{ext}"
        fig.savefig(path, bbox_inches="tight", dpi=300 if ext == "png" else None)
        print(f"[INFO] Wrote {path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate StepKV interrupt figures.")
    parser.add_argument(
        "-o", "--output-base",
        default="assets/stepkv_interrupt_lag2_timeline",
        help="Output path without extension (writes .png and .pdf)",
    )
    parser.add_argument(
        "--policies-output-base",
        default="assets/stepkv_interrupt_policies",
        help="Second figure comparing policies",
    )
    parser.add_argument("--max-step", type=int, default=5)
    parser.add_argument("--enter-step", type=int, default=4)
    parser.add_argument("--no-policies", action="store_true")
    args = parser.parse_args()

    generate_main_figure(args.output_base, max_step=args.max_step)
    if not args.no_policies:
        generate_policy_figure(args.policies_output_base, enter_step=args.enter_step, max_step=args.max_step)


if __name__ == "__main__":
    main()
