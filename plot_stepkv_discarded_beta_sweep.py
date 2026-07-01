#!/usr/bin/env python3
"""
Re-plot StepKV discarded-token beta sweep from saved JSON (no re-run).

Expected JSON: output of analyze_stepkv_discarded_tokens.py (beta_sweep)
  results/stepkv_discarded_token_analysis/hotpotqa_sample0_*_beta_sweep.json

Or the combined summary JSON with analyses.beta_sweep.

Usage:
  python plot_stepkv_discarded_beta_sweep.py \\
    --input_json results/stepkv_discarded_token_analysis/hotpotqa_sample0_20260101_120000_beta_sweep.json

  python plot_stepkv_discarded_beta_sweep.py \\
    --input_json path/to/*_beta_sweep.json \\
    --labelsize 20 --ticksize 16 --marker_size 28
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required. Install with: pip install matplotlib"
    ) from exc


DEFAULT_SEARCH_DIRS = [
    "results/stepkv_drop_token_analysis",
    "results/stepkv_discarded_token_analysis",
]


def _fix_duplicate_repo_prefix(path: str) -> str:
    dup = f"{os.sep}autodl-tmp{os.sep}kvmem{os.sep}autodl-tmp{os.sep}kvmem{os.sep}"
    if dup in path:
        path = path.replace(dup, f"{os.sep}autodl-tmp{os.sep}kvmem{os.sep}", 1)
    dup2 = "autodl-tmp/kvmem/autodl-tmp/kvmem/"
    if dup2 in path.replace("\\", "/"):
        path = path.replace("autodl-tmp/kvmem/autodl-tmp/kvmem/", "autodl-tmp/kvmem/", 1)
    return path


def _resolve_json_path(path: str) -> str:
    path = _fix_duplicate_repo_prefix(os.path.expanduser(path.strip()))
    candidates = [path]
    if not os.path.isabs(path):
        candidates.append(os.path.join(os.getcwd(), path))
    if not path.endswith(".json"):
        candidates.append(path + ".json")
        if not os.path.isabs(path):
            candidates.append(os.path.join(os.getcwd(), path + ".json"))

    seen = set()
    for cand in candidates:
        cand = os.path.abspath(cand)
        if cand in seen:
            continue
        seen.add(cand)
        if os.path.isfile(cand):
            return cand

    matches = sorted(glob.glob(path))
    if len(matches) == 1:
        return os.path.abspath(matches[0])
    if len(matches) > 1:
        raise FileNotFoundError(
            f"Multiple JSON files match {path!r}; pass the exact file path."
        )
    return os.path.abspath(path)


def _json_has_beta_runs(path: str) -> bool:
    try:
        load_beta_sweep_payload(path)
        return True
    except (ValueError, json.JSONDecodeError, OSError):
        return False


def discover_input_json(search_dirs: List[str]) -> str:
    candidates: List[str] = []
    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        candidates.extend(sorted(glob.glob(os.path.join(directory, "*_beta_sweep.json"))))
        summary = os.path.join(directory, "summary.json")
        if os.path.isfile(summary):
            candidates.append(summary)
        candidates.extend(sorted(glob.glob(os.path.join(directory, "*_summary.json"))))

    valid = [p for p in candidates if _json_has_beta_runs(p)]
    if not valid:
        tried = ", ".join(search_dirs)
        raise FileNotFoundError(
            f"No JSON with analyses.beta_sweep / beta_runs found under: {tried}. "
            "Look for *_beta_sweep.json or a summary.json from analyze_stepkv_discarded_tokens.py."
        )
    # Prefer dedicated beta_sweep json over combined summary.
    beta_only = [p for p in valid if p.endswith("_beta_sweep.json")]
    return (beta_only or valid)[-1]


def _max_plotted_decode_index(beta_runs: Dict[str, Dict[str, Any]]) -> int:
    max_idx = -1
    for run in beta_runs.values():
        discarded = run.get("discarded", {}) or {}
        for idx in discarded.get("discarded_decode_indices", []) or []:
            max_idx = max(max_idx, int(idx))
        for bd in discarded.get("step_boundaries", []) or []:
            max_idx = max(max_idx, int(float(bd.get("x", -1))))
    return max(max_idx, 0)


def _heatmap_figsize(plot_len: int) -> float:
    return float(min(14.0, max(5.0, plot_len * 0.09)))


def load_beta_sweep_payload(path: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta: Dict[str, Any] = {}
    if isinstance(data.get("beta_runs"), dict):
        beta_runs = data["beta_runs"]
        meta = {k: v for k, v in data.items() if k != "beta_runs"}
    elif isinstance(data.get("analyses"), dict) and isinstance(data["analyses"].get("beta_sweep"), dict):
        block = data["analyses"]["beta_sweep"]
        beta_runs = block.get("beta_runs") or {}
        meta = data.get("meta", {})
        meta.update({k: v for k, v in block.items() if k != "beta_runs"})
    else:
        raise ValueError(
            f"Unsupported beta sweep JSON format: {path}. "
            "Expected top-level beta_runs or analyses.beta_sweep.beta_runs."
        )

    if not beta_runs:
        raise ValueError(f"No beta_runs found in {path}")
    return beta_runs, meta


def plot_discarded_beta_sweep(
    beta_runs: Dict[str, Dict[str, Any]],
    output_png: str,
    *,
    meta: Optional[Dict[str, Any]] = None,
    labelsize: float = 18,
    ticksize: float = 14,
    title_size: float = 18,
    marker_size: float = 22,
    show_title: bool = False,
) -> None:
    betas = sorted(beta_runs.keys(), key=lambda x: float(x), reverse=True)
    plot_x_max = _max_plotted_decode_index(beta_runs)

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": ticksize,
            "axes.labelsize": labelsize,
            "xtick.labelsize": ticksize,
            "ytick.labelsize": ticksize,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    fig_h = max(4.5, 0.95 * len(betas) + 2.0)
    fig, ax = plt.subplots(figsize=(_heatmap_figsize(plot_x_max + 1), fig_h))
    cmap = plt.get_cmap("viridis")

    boundaries: List[Dict[str, Any]] = []
    for i, beta in enumerate(betas):
        run = beta_runs[beta]
        discarded = run.get("discarded", {}) or {}
        boundaries = discarded.get("step_boundaries", []) or boundaries
        xs = discarded.get("discarded_decode_indices", []) or []
        ys = [i] * len(xs)
        color = cmap(i / max(1, len(betas) - 1))
        ax.scatter(xs, ys, s=marker_size, alpha=0.88, c=[color], edgecolors="none")

    for bd in boundaries:
        x = float(bd.get("x", -1))
        if x <= plot_x_max:
            ax.axvline(x, linestyle="--", linewidth=1.0, color="gray", alpha=0.55)

    ax.set_yticks(range(len(betas)))
    ax.set_yticklabels([f"beta={b}" for b in betas], fontsize=ticksize)
    ax.set_xlabel("Decode Token Index (prompt excluded)", fontsize=labelsize)
    ax.set_xlim(-0.5, plot_x_max + 0.5)
    ax.tick_params(axis="x", labelsize=ticksize)
    ax.grid(True, axis="x", alpha=0.25, linestyle=":", linewidth=0.7)
    ax.set_axisbelow(True)

    if show_title:
        sample_id = ""
        if meta:
            sample_id = str(meta.get("sample_id") or "").strip()
        title = "StepKV Discarded Tokens vs step-score weight"
        if sample_id:
            title += f" | sample={sample_id}"
        ax.set_title(title, fontsize=title_size, pad=10)

    fig.tight_layout()
    out_dir = os.path.dirname(os.path.abspath(output_png)) or "."
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(output_png)
    pdf_path = os.path.splitext(output_png)[0] + ".pdf"
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"[INFO] Saved: {output_png}")
    print(f"[INFO] Saved: {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-plot StepKV discarded-token beta sweep from saved JSON."
    )
    parser.add_argument(
        "--input_json",
        type=str,
        default=None,
        help="Path to *_beta_sweep.json or summary.json with analyses.beta_sweep",
    )
    parser.add_argument(
        "--search_dir",
        action="append",
        default=None,
        help="Directory to auto-search when --input_json is omitted "
             "(default: results/stepkv_drop_token_analysis and "
             "results/stepkv_discarded_token_analysis).",
    )
    parser.add_argument(
        "--output_png",
        type=str,
        default=None,
        help="Output PNG path (default: <input>_replot.png)",
    )
    parser.add_argument("--labelsize", type=float, default=18, help="Axis label font size.")
    parser.add_argument("--ticksize", type=float, default=14, help="Tick label font size.")
    parser.add_argument("--title_size", type=float, default=18, help="Title font size.")
    parser.add_argument("--marker_size", type=float, default=22, help="Scatter marker size.")
    parser.add_argument("--show_title", action="store_true", help="Show plot title.")
    args = parser.parse_args()

    if args.input_json:
        input_json = _resolve_json_path(args.input_json)
        if not os.path.isfile(input_json):
            raise FileNotFoundError(
                f"JSON not found: {input_json}\n"
                "Tip: run from repo root (~/autodl-tmp/kvmem) and use e.g.\n"
                "  --input_json results/stepkv_drop_token_analysis/summary.json"
            )
    else:
        search_dirs = args.search_dir or DEFAULT_SEARCH_DIRS
        input_json = discover_input_json(search_dirs)
        print(f"[INFO] Auto-selected JSON: {input_json}")

    beta_runs, meta = load_beta_sweep_payload(input_json)
    print(f"[INFO] Loaded betas={sorted(beta_runs.keys(), key=float, reverse=True)}")

    if args.output_png:
        output_png = args.output_png
    else:
        base, _ext = os.path.splitext(input_json)
        output_png = f"{base}_replot.png"

    plot_discarded_beta_sweep(
        beta_runs,
        output_png,
        meta=meta,
        labelsize=args.labelsize,
        ticksize=args.ticksize,
        title_size=args.title_size,
        marker_size=args.marker_size,
        show_title=args.show_title,
    )


if __name__ == "__main__":
    main()
