#!/usr/bin/env python3
"""
BrowseComp StepKV radar chart vs FullKV baseline (single figure).

Plots FullKV + StepKV@50% + StepKV@20% on one radar with six axes:
  EM, F1, Avg Time, Max Time, Avg Cache, Max Cache
All normalized to FullKV = 1.0 (larger = better on every axis).

Expected layout (either works):

  A) metrics markdown (from record_experiment_metrics.py):
     {run_dir}/metrics_react_kv_none.md
     {run_dir}/metrics_react_kv_step_aware_h2o_r50.md   # or _r05
     {run_dir}/metrics_react_kv_step_aware_h2o_r20.md   # or _r02

  B) result JSON:
     {run_dir}/fullkv/react_kv_none_browsecomp*.json
     {run_dir}/stepkv_r50/react_kv_step_aware_h2o_browsecomp_r50.json
     {run_dir}/stepkv_r20/react_kv_step_aware_h2o_browsecomp_r20.json

Example (metrics md):
  python plot_browsecomp_radar.py \\
    --run_dir results/browsecomp_v2/stepkv_r50 \\
    --source metrics
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from analyze_run_kv_metrics import (
    MethodRunStats,
    detect_dataset_suffix,
    resolve_result_json,
)
from record_experiment_metrics import parse_metrics_markdown

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

# metrics_*.md filename candidates (shell may write r05/r02 or r50/r20)
BROWSECOMP_METRICS_SERIES: List[Tuple[str, str, List[str], str]] = [
    ("FullKV", "full", ["metrics_react_kv_none.md"], "FullKV"),
    (
        "StepKV (50%)",
        "r50",
        [
            "metrics_react_kv_step_aware_h2o_r50.md",
            "metrics_react_kv_step_aware_h2o_r05.md",
            "metrics_react_kv_step_aware_h2o_r5.md",
        ],
        "StepKV",
    ),
    (
        "StepKV (20%)",
        "r20",
        [
            "metrics_react_kv_step_aware_h2o_r20.md",
            "metrics_react_kv_step_aware_h2o_r02.md",
            "metrics_react_kv_step_aware_h2o_r2.md",
        ],
        "StepKV",
    ),
]

AXIS_SPECS: List[Tuple[str, str, str]] = [
    ("em", "EM", "em"),
    ("f1", "F1", "f1"),
    ("avg_sample_time_s", "Avg Time", "avg_sample_time_s"),
    ("max_sample_time_s", "Max Time", "max_sample_time_s"),
    ("avg_cache", "Avg Cache", "avg_cache"),
    ("max_cache", "Max Cache", "max_cache"),
]

SERIES_COLORS = {
    "FullKV": "#009E73",
    "StepKV (50%)": "#0072B2",
    "StepKV (20%)": "#E69F00",
}

SERIES_ORDER = {"FullKV": 0, "StepKV (50%)": 1, "StepKV (20%)": 2}


def _resolve_metrics_file(run_dir: str, filename: str) -> Optional[str]:
    """Find metrics md under run_dir (flat or nested subdirs)."""
    direct = os.path.join(run_dir, filename)
    if os.path.isfile(direct):
        return direct

    subdir_hints = {
        "metrics_react_kv_none.md": ["fullkv", "."],
        "metrics_react_kv_step_aware_h2o_r50.md": ["stepkv_r50", "stepaware_r50", "."],
        "metrics_react_kv_step_aware_h2o_r05.md": ["stepkv_r50", "stepaware_r50", "."],
        "metrics_react_kv_step_aware_h2o_r5.md": ["stepkv_r50", "stepaware_r50", "."],
        "metrics_react_kv_step_aware_h2o_r20.md": ["stepkv_r20", "stepaware_r20", "."],
        "metrics_react_kv_step_aware_h2o_r02.md": ["stepkv_r20", "stepaware_r20", "."],
        "metrics_react_kv_step_aware_h2o_r2.md": ["stepkv_r20", "stepaware_r20", "."],
    }
    for sub in subdir_hints.get(filename, ["."]):
        if sub == ".":
            continue
        candidate = os.path.join(run_dir, sub, filename)
        if os.path.isfile(candidate):
            return candidate

    matches = sorted(glob.glob(os.path.join(run_dir, "**", filename), recursive=True))
    if matches:
        return matches[0]
    return None


def _as_float(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _stats_from_metrics_md(path: str, method_name: str, ratio: str, display_label: str) -> MethodRunStats:
    parsed = parse_metrics_markdown(path)
    avg_peak = _as_float(parsed.get("avg_step_decode_cache_len"))
    max_peak = _as_float(parsed.get("max_step_decode_cache_len"))
    avg_final = _as_float(parsed.get("avg_final_decode_cache_len"))
    max_final = _as_float(parsed.get("max_final_decode_cache_len"))
    return MethodRunStats(
        key=f"{display_label}_{ratio}",
        method=method_name,
        ratio=ratio,
        subdir=os.path.dirname(path) or ".",
        result_json=path,
        n_samples=int(parsed.get("n_samples", 0) or 0),
        em=_as_float(parsed.get("EM")),
        f1=_as_float(parsed.get("F1")),
        avg_sample_time_s=_as_float(parsed.get("avg_sample_time_seconds")),
        max_sample_time_s=_as_float(parsed.get("max_sample_time_seconds")),
        avg_peak_kv_tokens=avg_peak,
        max_peak_kv_tokens=max_peak,
        avg_final_kv_tokens=avg_final,
        max_final_kv_tokens=max_final,
    )


def load_browsecomp_stepkv_rows_from_metrics(run_dir: str) -> List[MethodRunStats]:
    rows: List[MethodRunStats] = []
    for display_label, ratio, filenames, method_name in BROWSECOMP_METRICS_SERIES:
        found_path = None
        for name in filenames:
            found_path = _resolve_metrics_file(run_dir, name)
            if found_path:
                break
        if not found_path:
            print(f"[WARN] Missing metrics md (tried): {', '.join(filenames)}")
            continue
        row = _stats_from_metrics_md(found_path, method_name, ratio, display_label)
        print(f"[OK] {display_label}: n={row.n_samples} md={found_path}")
        rows.append(row)
    return rows


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


def _first_float(*candidates: Any) -> Optional[float]:
    for c in candidates:
        v = _as_float(c)
        if v is not None and v >= 0:
            return v
    return None


def _stats_from_summary_and_results(
    data: Dict[str, Any],
    sample_times: List[float],
    peak_kvs: List[int],
    final_kvs: List[int],
) -> Dict[str, Optional[float]]:
    """Merge per-sample aggregates with summary / timing_stats fallbacks."""
    from analyze_run_kv_metrics import (
        _max,
        _max_int,
        _mean,
        compute_derived_stats,
    )

    summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
    derived = compute_derived_stats(data)
    timing = summary.get("timing_stats") if isinstance(summary.get("timing_stats"), dict) else {}

    avg_time = _first_float(
        _mean(sample_times),
        derived.get("avg_sample_time_seconds"),
        summary.get("avg_sample_time_seconds"),
        summary.get("avg_time_per_sample"),
    )
    max_time = _first_float(
        _max(sample_times),
        derived.get("max_sample_time_seconds"),
        summary.get("max_sample_time_seconds"),
    )
    if max_time is None:
        max_time = avg_time

    avg_peak = _first_float(
        _mean([float(v) for v in peak_kvs if v > 0]) if any(v > 0 for v in peak_kvs) else None,
        derived.get("avg_step_decode_cache_len"),
        timing.get("avg_kv_cache_length"),
        summary.get("avg_step_decode_cache_len"),
        derived.get("avg_final_decode_cache_len"),
        summary.get("avg_final_decode_cache_len"),
    )
    max_peak = _first_float(
        float(_max_int([v for v in peak_kvs if v > 0])) if any(v > 0 for v in peak_kvs) else None,
        derived.get("max_step_decode_cache_len"),
        timing.get("max_kv_cache_length"),
        summary.get("max_step_decode_cache_len"),
        derived.get("max_final_decode_cache_len"),
        summary.get("max_final_decode_cache_len"),
    )
    if max_peak is None:
        max_peak = avg_peak

    avg_final = _first_float(
        _mean([float(v) for v in final_kvs if v > 0]) if any(v > 0 for v in final_kvs) else None,
        derived.get("avg_final_decode_cache_len"),
        summary.get("avg_final_decode_cache_len"),
        avg_peak,
    )
    max_final = _first_float(
        float(_max_int([v for v in final_kvs if v > 0])) if any(v > 0 for v in final_kvs) else None,
        derived.get("max_final_decode_cache_len"),
        summary.get("max_final_decode_cache_len"),
        max_peak,
    )

    return {
        "em": _as_float(summary.get("exact_match")),
        "f1": _as_float(summary.get("f1_score")),
        "avg_sample_time_s": avg_time,
        "max_sample_time_s": max_time,
        "avg_peak_kv_tokens": avg_peak,
        "max_peak_kv_tokens": max_peak,
        "avg_final_kv_tokens": avg_final,
        "max_final_kv_tokens": max_final,
    }


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
        _per_sample_final_kv,
        _per_sample_peak_kv,
        load_result_json,
    )

    json_path = _resolve_first_subdir(run_dir, subdir_candidates, stem, dataset_suffix, ratio_tag)
    if not json_path:
        print(f"[WARN] Missing: {' / '.join(subdir_candidates)}/{stem}_{dataset_suffix}*.json")
        return None

    data = load_result_json(json_path)
    summary = data.get("summary", {})
    results = data.get("results", [])

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

    stats = _stats_from_summary_and_results(data, sample_times, peak_kvs, final_kvs)
    ratio_label = ratio_tag if ratio_tag else "full"
    row = MethodRunStats(
        key=f"{display_label}_{ratio_label}",
        method=method_name,
        ratio=ratio_label,
        subdir=subdir_candidates[0],
        result_json=json_path,
        n_samples=int(summary.get("total_samples", len(results)) or len(results)),
        em=stats["em"],
        f1=stats["f1"],
        avg_sample_time_s=stats["avg_sample_time_s"],
        max_sample_time_s=stats["max_sample_time_s"],
        avg_peak_kv_tokens=stats["avg_peak_kv_tokens"],
        max_peak_kv_tokens=stats["max_peak_kv_tokens"],
        avg_final_kv_tokens=stats["avg_final_kv_tokens"],
        max_final_kv_tokens=stats["max_final_kv_tokens"],
    )
    print(
        f"[OK] {display_label}: n={row.n_samples} json={json_path} "
        f"EM={row.em} F1={row.f1} avg_t={row.avg_sample_time_s} avg_cache={row.avg_peak_kv_tokens}"
    )
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
) -> Optional[float]:
    """Always method / FullKV so FullKV = 1.0 (largest baseline on each axis)."""
    if method_val is None or fullkv_val is None:
        return None
    if fullkv_val == 0:
        return 1.0 if method_val == 0 else None
    return float(method_val) / float(fullkv_val)


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
        for key, _, field in AXIS_SPECS:
            norm[key] = _normalize_vs_fullkv(
                raw.get(field),
                baseline_raw.get(field),
            )

        if any(v is None for v in norm.values()):
            missing = [
                axis_key
                for axis_key, v in norm.items()
                if v is None
            ]
            print(f"[WARN] Skip incomplete series: {label} missing={missing} raw={raw}")
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
    radial_max = max(1.05, max(all_norm_vals) * 1.05 if all_norm_vals else 1.05)
    ax.set_ylim(0, radial_max)
    tick_vals = [0.25, 0.5, 0.75, 1.0]
    if radial_max > 1.05:
        tick_vals.append(round(radial_max, 2))
    ax.set_yticks(tick_vals)
    ax.set_yticklabels([f"{t:.1f}" for t in tick_vals], fontsize=ticksize - 1, color="#666666")
    ax.grid(color="#CCCCCC", linestyle=":", linewidth=0.8)
    ax.spines["polar"].set_color("#AAAAAA")

    # FullKV baseline ring at 1.0
    ax.plot(
        angles_closed,
        [1.0] * (n_axes + 1),
        color="#009E73",
        linewidth=2.6,
        linestyle="-",
        zorder=2,
        label="_nolegend_",
    )

    for spec in series_list:
        label = spec["label"]
        vals = [float(spec["normalized"][key]) for key, _, _ in AXIS_SPECS]
        vals_closed = vals + vals[:1]
        color = SERIES_COLORS.get(label, "#333333")
        lw = 2.6 if label == "FullKV" else 2.0
        zorder = 3 if label == "FullKV" else 2
        fill_a = 0.12 if label == "FullKV" else fill_alpha

        ax.plot(
            angles_closed,
            vals_closed,
            color=color,
            linewidth=lw,
            label=label,
            zorder=zorder,
        )
        ax.fill(angles_closed, vals_closed, color=color, alpha=fill_a, zorder=1)

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
        "--source",
        choices=("auto", "metrics", "json"),
        default="auto",
        help="Input type: metrics=metrics_*.md, json=result JSON, auto=try metrics then json.",
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

    rows: List[MethodRunStats] = []
    if args.source in ("auto", "metrics"):
        rows = load_browsecomp_stepkv_rows_from_metrics(run_dir)
    if args.source == "json" or (args.source == "auto" and len(rows) < 2):
        if args.source == "auto" and rows:
            print("[INFO] Fewer than 2 metrics files found; also trying result JSON...")
        json_rows = load_browsecomp_stepkv_rows(run_dir, dataset_suffix)
        if args.source == "json" or len(json_rows) > len(rows):
            rows = json_rows

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
