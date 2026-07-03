#!/usr/bin/env python3
"""
BrowseComp StepKV radar chart vs FullKV baseline (single figure).

Plots FullKV + StepKV@50% + StepKV@20% on one radar with six axes:
  EM, F1, Avg Time, Max Time, Avg Cache, Max Cache
All normalized to FullKV = 1.0 (larger = better on every axis).

Expected layout:
  {run_dir}/fullkv/react_kv_none_browsecomp*.json
  {run_dir}/stepkv_r50/react_kv_step_aware_h2o_browsecomp_r50.json
  {run_dir}/stepkv_r20/react_kv_step_aware_h2o_browsecomp_r20.json

Example:
  python plot_browsecomp_radar.py \\
    --run_dir results/browsecomp_v2
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from analyze_run_kv_metrics import (
    MethodRunStats,
    detect_dataset_suffix,
    resolve_result_json,
)

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib and numpy are required. Install with: pip install matplotlib numpy"
    ) from exc


DEFAULT_RUN_DIR = "results/browsecomp_v2"
DATASET_SUFFIX = "browsecomp"

# (subdir candidates, json stem, ratio tag, display label)
BROWSECOMP_STEPKV_SERIES: List[Tuple[List[str], str, Optional[str], str]] = [
    (["fullkv"], "react_kv_none", None, "FullKV"),
    (["stepkv_r50", "stepaware_r50"], "react_kv_step_aware_h2o", "r50", "StepKV (50%)"),
    (["stepkv_r20", "stepaware_r20"], "react_kv_step_aware_h2o", "r20", "StepKV (20%)"),
]

AXIS_SPECS: List[Tuple[str, str, str, bool]] = [
    ("em", "EM", "em", True),
    ("f1", "F1", "f1", True),
    ("avg_sample_time_s", "Avg Time", "avg_sample_time_s", False),
    ("max_sample_time_s", "Max Time", "max_sample_time_s", False),
    ("avg_cache", "Avg Cache", "avg_cache", False),
    ("max_cache", "Max Cache", "max_cache", False),
]

SERIES_COLORS = {
    "FullKV": "#009E73",
    "StepKV (50%)": "#0072B2",
    "StepKV (20%)": "#E69F00",
}

SERIES_ORDER = {"FullKV": 0, "StepKV (50%)": 1, "StepKV (20%)": 2}


def _resolve_first_subdir(
    run_dir: str,
    subdir_candidates: Sequence[str],
    stem: str,
    dataset_suffix: str,
    ratio_tag: Optional[str],
) -> Optional[str]:
    for subdir in subdir_candidates:
        path = resolve_result_json(run_dir, subdir, stem, dataset_suffix, ratio_tag)
        if path:
            if subdir != subdir_candidates[0]:
                print(f"[INFO] Resolved via fallback subdir '{subdir}': {path}")
            return path
    return None


def _load_one_stats(
    run_dir: str,
    dataset_suffix: str,
    subdir_candidates: Sequence[str],
    stem: str,
    ratio_tag: Optional[str],
    method_name: str,
    display_label: str,
) -> Optional[MethodRunStats]:
    from analyze_run_kv_metrics import (
        _as_float,
        _max,
        _max_int,
        _mean,
        _per_sample_final_kv,
        _per_sample_peak_kv,
        compute_derived_stats,
        load_result_json,
    )

    json_path = _resolve_first_subdir(run_dir, subdir_candidates, stem, dataset_suffix, ratio_tag)
    if not json_path:
        print(f"[WARN] Missing: {' / '.join(subdir_candidates)}/{stem}_{dataset_suffix}*.json")
        return None

    data = load_result_json(json_path)
    summary = data.get("summary", {})
    results = data.get("results", [])
    derived = compute_derived_stats(data)

    sample_times: List[float] = []
    peak_kvs: List[int] = []
    final_kvs: List[int] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        st = r.get("sample_time")
        if isinstance(st, (int, float)) and st > 0:
            sample_times.append(float(st))
        peak_kvs.append(_per_sample_peak_kv(r))
        final_kvs.append(_per_sample_final_kv(r))

    ratio_label = ratio_tag if ratio_tag else "full"
    row = MethodRunStats(
        key=f"{display_label}_{ratio_label}",
        method=method_name,
        ratio=ratio_label,
        subdir=subdir_candidates[0],
        result_json=json_path,
        n_samples=int(summary.get("total_samples", len(results)) or len(results)),
        em=_as_float(summary.get("exact_match")),
        f1=_as_float(summary.get("f1_score")),
        avg_sample_time_s=_mean(sample_times) or _as_float(derived.get("avg_sample_time_seconds")),
        max_sample_time_s=_max(sample_times) or _as_float(derived.get("max_sample_time_seconds")),
        avg_peak_kv_tokens=_mean(peak_kvs) or _as_float(derived.get("avg_step_decode_cache_len")),
        max_peak_kv_tokens=float(_max_int(peak_kvs)) if peak_kvs else _as_float(derived.get("max_step_decode_cache_len")),
        avg_final_kv_tokens=_mean(final_kvs) or _as_float(derived.get("avg_final_decode_cache_len")),
        max_final_kv_tokens=float(_max_int(final_kvs)) if final_kvs else _as_float(derived.get("max_final_decode_cache_len")),
    )
    print(f"[OK] {display_label}: n={row.n_samples} json={json_path}")
    return row


def load_browsecomp_stepkv_rows(run_dir: str, dataset_suffix: str) -> List[MethodRunStats]:
    """Load FullKV + StepKV r50/r20 (stepkv_* subdirs, fallback stepaware_*)."""
    rows: List[MethodRunStats] = []
    for candidates, stem, ratio_tag, label in BROWSECOMP_STEPKV_SERIES:
        method_name = "FullKV" if label == "FullKV" else "StepKV"
        row = _load_one_stats(
            run_dir, dataset_suffix, candidates, stem, ratio_tag, method_name, label
        )
        if row is not None:
            rows.append(row)
    return rows


def _cache_fields(cache_metric: str) -> Tuple[str, str]:
    if cache_metric == "final":
        return "avg_final_kv_tokens", "max_final_kv_tokens"
    return "avg_peak_kv_tokens", "max_peak_kv_tokens"


def _row_cache_stats(row: MethodRunStats, cache_metric: str) -> Tuple[Optional[float], Optional[float]]:
    avg_field, max_field = _cache_fields(cache_metric)
    return getattr(row, avg_field), getattr(row, max_field)


def _normalize_vs_fullkv(
    method_val: Optional[float],
    fullkv_val: Optional[float],
    higher_is_better: bool,
) -> Optional[float]:
    if method_val is None or fullkv_val is None:
        return None
    if fullkv_val <= 0 or method_val <= 0:
        return None
    if higher_is_better:
        return float(method_val) / float(fullkv_val)
    return float(fullkv_val) / float(method_val)


def _display_label(row: MethodRunStats) -> str:
    if row.method == "FullKV":
        return "FullKV"
    pct = row.ratio.replace("r", "")
    return f"StepKV ({pct}%)"


def _build_normalized_series(
    rows: List[MethodRunStats],
    baseline: MethodRunStats,
    cache_metric: str,
) -> List[Dict[str, Any]]:
    baseline_avg_cache, baseline_max_cache = _cache_fields(cache_metric)
    baseline_raw = {
        "em": baseline.em,
        "f1": baseline.f1,
        "avg_sample_time_s": baseline.avg_sample_time_s,
        "max_sample_time_s": baseline.max_sample_time_s,
        "avg_cache": getattr(baseline, baseline_avg_cache),
        "max_cache": getattr(baseline, baseline_max_cache),
    }

    out: List[Dict[str, Any]] = []
    for row in rows:
        label = _display_label(row)
        row_avg_cache, row_max_cache = _row_cache_stats(row, cache_metric)
        raw = {
            "em": row.em,
            "f1": row.f1,
            "avg_sample_time_s": row.avg_sample_time_s,
            "max_sample_time_s": row.max_sample_time_s,
            "avg_cache": row_avg_cache,
            "max_cache": row_max_cache,
        }

        norm: Dict[str, Optional[float]] = {}
        for key, _, field, higher_is_better in AXIS_SPECS:
            norm[key] = _normalize_vs_fullkv(
                raw.get(field),
                baseline_raw.get(field),
                higher_is_better,
            )

        if any(v is None for v in norm.values()):
            print(f"[WARN] Skip incomplete series: {label}")
            continue

        out.append(
            {
                "method": row.method,
                "ratio": row.ratio,
                "label": label,
                "raw": raw,
                "normalized": norm,
            }
        )

    out.sort(key=lambda s: SERIES_ORDER.get(s["label"], 99))
    return out


def _setup_style(labelsize: int, ticksize: int) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": ticksize,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def plot_radar_single(
    series_list: List[Dict[str, Any]],
    output_prefix: str,
    *,
    title: Optional[str] = None,
    labelsize: int = 14,
    ticksize: int = 12,
    fill_alpha: float = 0.10,
) -> None:
    """One polar plot: FullKV + StepKV r50 + StepKV r20 together."""
    _setup_style(labelsize, ticksize)

    axis_labels = [spec[1] for spec in AXIS_SPECS]
    n_axes = len(axis_labels)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(6.8, 6.8), subplot_kw={"polar": True})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(axis_labels, fontsize=labelsize)

    all_norm_vals = [
        float(v) for spec in series_list for v in spec["normalized"].values()
    ]
    radial_max = max(1.05, max(all_norm_vals) * 1.10 if all_norm_vals else 1.2)
    ax.set_ylim(0, radial_max)
    tick_vals = [0.5, 1.0]
    if radial_max >= 1.45:
        tick_vals.append(1.5)
    if radial_max >= 1.95:
        tick_vals.append(2.0)
    ax.set_yticks(tick_vals)
    ax.set_yticklabels([f"{t:.1f}" for t in tick_vals], fontsize=ticksize - 1, color="#666666")
    ax.grid(color="#CCCCCC", linestyle=":", linewidth=0.8)
    ax.spines["polar"].set_color("#AAAAAA")

    # FullKV reference ring at 1.0
    ax.plot(
        angles_closed,
        [1.0] * (n_axes + 1),
        color="#444444",
        linewidth=1.2,
        linestyle=":",
        zorder=1,
        label="_nolegend_",
    )

    for spec in series_list:
        label = spec["label"]
        vals = [float(spec["normalized"][key]) for key, _, _, _ in AXIS_SPECS]
        vals_closed = vals + vals[:1]
        color = SERIES_COLORS.get(label, "#333333")
        lw = 2.6 if label == "FullKV" else 2.0
        zorder = 3 if label == "FullKV" else 2

        ax.plot(
            angles_closed,
            vals_closed,
            color=color,
            linewidth=lw,
            label=label,
            zorder=zorder,
        )
        ax.fill(angles_closed, vals_closed, color=color, alpha=fill_alpha, zorder=1)

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.28, 1.12),
        frameon=False,
        fontsize=ticksize,
    )

    if title:
        ax.set_title(title, fontsize=labelsize + 2, pad=20)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = f"{output_prefix}_radar.{ext}"
        fig.savefig(out)
        print(f"[INFO] Saved figure: {out}")
    plt.close(fig)


def write_radar_json(payload: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[INFO] Saved JSON: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="BrowseComp StepKV radar: FullKV + r50 + r20 on one chart."
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=DEFAULT_RUN_DIR,
        help=f"Experiment run directory (default: {DEFAULT_RUN_DIR})",
    )
    parser.add_argument(
        "--dataset_suffix",
        type=str,
        default=None,
        help="Result JSON suffix (default: browsecomp, or auto-detect).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: {run_dir}/analysis).",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default=None,
        help="Figure filename prefix (default: {output_dir}/browsecomp_stepkv_radar).",
    )
    parser.add_argument(
        "--cache_metric",
        choices=("peak", "final"),
        default="peak",
        help="Cache axes: peak = max per-step cache; final = end-of-sample cache.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional figure title.",
    )
    parser.add_argument(
        "--labelsize",
        type=int,
        default=14,
    )
    parser.add_argument(
        "--ticksize",
        type=int,
        default=12,
    )
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    dataset_suffix = args.dataset_suffix or detect_dataset_suffix(run_dir) or DATASET_SUFFIX
    output_dir = args.output_dir or os.path.join(run_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    output_prefix = args.output_prefix or os.path.join(output_dir, "browsecomp_stepkv_radar")

    rows = load_browsecomp_stepkv_rows(run_dir, dataset_suffix)
    baseline = next((r for r in rows if r.method == "FullKV"), None)
    if baseline is None:
        raise RuntimeError("FullKV baseline (fullkv/react_kv_none) is required.")

    series_list = _build_normalized_series(rows, baseline, args.cache_metric)
    if len(series_list) < 2:
        raise RuntimeError(
            "Need at least FullKV + one StepKV ratio. "
            "Expected subdirs: fullkv, stepkv_r50, stepkv_r20"
        )

    baseline_avg, baseline_max = _row_cache_stats(baseline, args.cache_metric)
    payload = {
        "run_dir": run_dir,
        "dataset_suffix": dataset_suffix,
        "cache_metric": args.cache_metric,
        "baseline": {
            "method": "FullKV",
            "em": baseline.em,
            "f1": baseline.f1,
            "avg_sample_time_s": baseline.avg_sample_time_s,
            "max_sample_time_s": baseline.max_sample_time_s,
            "avg_cache": baseline_avg,
            "max_cache": baseline_max,
        },
        "normalization": {
            "note": "All axes normalized vs FullKV; larger is better on every axis.",
        },
        "series": series_list,
    }
    write_radar_json(payload, f"{output_prefix}_data.json")

    plot_radar_single(
        series_list,
        output_prefix,
        title=args.title,
        labelsize=args.labelsize,
        ticksize=args.ticksize,
    )
    print(f"[DONE] Radar chart written to {output_dir}")


if __name__ == "__main__":
    main()
