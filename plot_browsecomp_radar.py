#!/usr/bin/env python3
"""
BrowseComp StepKV radar chart vs FullKV baseline (single figure).

Plots FullKV + StepKV@50% + StepKV@20% on one radar (six axes).

Normalization (hybrid):
  - EM / F1: if FullKV > 0 → value / FullKV (FullKV = 1, StepKV relative, can exceed 1).
             if FullKV = 0 → (value + floor) / (scale + floor) so FullKV is not pinned to axis.
  - Time / Cache: value / max(all methods) — FullKV usually largest on cache.

Expected JSON layout:
  {run_dir}/fullkv/react_kv_none_browsecomp*.json
  {run_dir}/stepkv_r50/react_kv_step_aware_h2o_browsecomp_r50.json
  {run_dir}/stepkv_r20/react_kv_step_aware_h2o_browsecomp_r20.json

Example:
  python plot_browsecomp_radar.py \\
    --run_dir /root/autodl-tmp/kvmem/results/browsecomp_v2
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

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

AXIS_SPECS: List[Tuple[str, str, str, str]] = [
    ("em", "EM", "em", "quality"),
    ("f1", "F1", "f1", "quality"),
    ("avg_sample_time_s", "Avg Time", "avg_sample_time_s", "cost"),
    ("max_sample_time_s", "Max Time", "max_sample_time_s", "cost"),
    ("avg_cache", "Avg Cache", "avg_cache", "cost"),
    ("max_cache", "Max Cache", "max_cache", "cost"),
]

QUALITY_AXIS_KEYS = frozenset(k for k, _, _, g in AXIS_SPECS if g == "quality")

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


def load_result_summary_only(path: str) -> Dict[str, Any]:
    """
    Load only JSON['summary'] and skip the huge results[] array (avoids OOM).
    Experiment outputs write summary before results, so streaming the head suffices.
    """
    size_mb = os.path.getsize(path) / (1024.0 * 1024)
    small_file_mb = 8.0
    max_head_scan_mb = 48.0

    if size_mb <= small_file_mb:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        summary = data.get("summary", {}) if isinstance(data.get("summary"), dict) else {}
        del data
        return {"summary": summary, "results": []}

    decoder = json.JSONDecoder()
    buffer = ""
    with open(path, "r", encoding="utf-8") as f:
        while len(buffer) < int(max_head_scan_mb * 1024 * 1024):
            piece = f.read(1024 * 1024)
            if not piece:
                break
            buffer += piece
            match = re.search(r'"summary"\s*:\s*\{', buffer)
            if not match:
                continue
            start = match.end() - 1
            try:
                summary, _end = decoder.raw_decode(buffer, start)
            except json.JSONDecodeError:
                continue
            if isinstance(summary, dict):
                print(f"[INFO] Summary-only load ({size_mb:.1f} MB): {os.path.basename(path)}")
                return {"summary": summary, "results": []}

    raise RuntimeError(
        f"Could not stream summary from {path} ({size_mb:.1f} MB). "
        "Check that the file contains a top-level 'summary' object."
    )


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
    )

    json_path = _resolve_first_subdir(run_dir, subdir_candidates, stem, dataset_suffix, ratio_tag)
    if not json_path:
        print(f"[WARN] Missing: {' / '.join(subdir_candidates)}/{stem}_{dataset_suffix}*.json")
        return None

    data = load_result_summary_only(json_path)
    summary = data.get("summary", {})
    results = data.get("results", [])

    sample_times: List[float] = []
    peak_kvs: List[int] = []
    final_kvs: List[int] = []
    if results:
        for r in results:
            if not isinstance(r, dict):
                continue
            st = r.get("sample_time")
            if isinstance(st, (int, float)) and st > 0:
                sample_times.append(float(st))
            peak_kvs.append(_per_sample_peak_kv(r))
            final_kvs.append(_per_sample_final_kv(r))

    stats = _stats_from_summary_and_results(data, sample_times, peak_kvs, final_kvs)
    del data
    gc.collect()
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


def _raw_stats_for_row(row: MethodRunStats, cache_metric: str) -> Dict[str, Optional[float]]:
    row_avg_cache, row_max_cache = _row_cache_stats(row, cache_metric)
    return {
        "em": row.em,
        "f1": row.f1,
        "avg_sample_time_s": row.avg_sample_time_s,
        "max_sample_time_s": row.max_sample_time_s,
        "avg_cache": row_avg_cache,
        "max_cache": row_max_cache,
    }


def _axis_values(all_raw: List[Dict[str, Optional[float]]], field: str) -> List[float]:
    return [
        float(r[field])
        for r in all_raw
        if r.get(field) is not None and float(r[field]) >= 0
    ]


def _compute_axis_scales(
    all_raw: List[Dict[str, Optional[float]]],
    fullkv_raw: Dict[str, Optional[float]],
    *,
    acc_display_floor: float,
    acc_scale_override: Optional[float],
    min_acc_scale: float,
) -> Tuple[Dict[str, float], Dict[str, str]]:
    """
    Return per-axis denominator and a short note on how it was chosen.
    Quality axes use FullKV-relative when FullKV>0, else soft-floor display scale.
    Cost axes use max across all methods.
    """
    scales: Dict[str, float] = {}
    modes: Dict[str, str] = {}

    for key, _, field, group in AXIS_SPECS:
        vals = _axis_values(all_raw, field)
        max_v = max(vals) if vals else 0.0
        fullkv_v = float(fullkv_raw.get(field) or 0.0)

        if group == "quality":
            if fullkv_v > 1e-9:
                scales[key] = fullkv_v
                modes[key] = "fullkv_relative"
            else:
                auto_scale = max(max_v * 1.15, min_acc_scale, 1e-9)
                if acc_scale_override is not None and acc_scale_override > 0:
                    auto_scale = max(float(acc_scale_override), min_acc_scale)
                scales[key] = auto_scale
                modes[key] = f"zero_fullkv_floor(floor={acc_display_floor})"
        else:
            scales[key] = max(max_v, 1e-9) if vals else 1.0
            modes[key] = "max_all"

    return scales, modes


def _normalize_row(
    raw: Dict[str, Optional[float]],
    axis_scales: Dict[str, float],
    axis_modes: Dict[str, str],
    *,
    acc_display_floor: float,
) -> Dict[str, Optional[float]]:
    norm: Dict[str, Optional[float]] = {}
    for key, _, field, group in AXIS_SPECS:
        v = raw.get(field)
        if v is None:
            norm[key] = None
            continue
        v = float(v)
        scale = axis_scales[key]
        if group == "quality" and axis_modes[key].startswith("zero_fullkv_floor"):
            norm[key] = (v + acc_display_floor) / (scale + acc_display_floor)
        else:
            norm[key] = v / scale
    return norm


def _display_label(row: MethodRunStats) -> str:
    if row.method == "FullKV":
        return "FullKV"
    pct = row.ratio.replace("r", "")
    return f"StepKV ({pct}%)"


def _build_normalized_series(
    rows: List[MethodRunStats],
    cache_metric: str,
    *,
    acc_display_floor: float = 0.5,
    acc_scale_override: Optional[float] = None,
    min_acc_scale: float = 1.0,
) -> List[Dict[str, Any]]:
    labeled_raw: List[Tuple[MethodRunStats, str, Dict[str, Optional[float]]]] = []
    fullkv_raw: Optional[Dict[str, Optional[float]]] = None
    for row in rows:
        raw = _raw_stats_for_row(row, cache_metric)
        if row.method == "FullKV":
            fullkv_raw = raw
        labeled_raw.append((row, _display_label(row), raw))

    if fullkv_raw is None:
        raise RuntimeError("FullKV row required for normalization.")

    all_raw = [raw for _, _, raw in labeled_raw]
    axis_scales, axis_modes = _compute_axis_scales(
        all_raw,
        fullkv_raw,
        acc_display_floor=acc_display_floor,
        acc_scale_override=acc_scale_override,
        min_acc_scale=min_acc_scale,
    )

    out: List[Dict[str, Any]] = []
    for row, label, raw in labeled_raw:
        norm = _normalize_row(
            raw,
            axis_scales,
            axis_modes,
            acc_display_floor=acc_display_floor,
        )

        if any(v is None for v in norm.values()):
            missing = [axis_key for axis_key, v in norm.items() if v is None]
            print(f"[WARN] Skip incomplete series: {label} missing={missing} raw={raw}")
            continue

        out.append(
            {
                "method": row.method,
                "ratio": row.ratio,
                "label": label,
                "raw": raw,
                "axis_scale": {k: axis_scales[k] for k in axis_scales},
                "axis_scale_mode": {k: axis_modes[k] for k in axis_modes},
                "normalized": norm,
            }
        )

    out.sort(key=lambda s: SERIES_ORDER.get(s["label"], 99))
    return out


def _style_radar_spokes(ax: plt.Axes, *, linewidth: float = 2.4) -> None:
    """Bold radial spokes (one per dimension)."""
    for line in ax.xaxis.get_gridlines():
        line.set_linestyle("-")
        line.set_color("#B0B0B0")
        line.set_linewidth(linewidth)
        line.set_alpha(0.95)


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
    labelsize: int = 24,
    ticksize: int = 20,
    fill_alpha: float = 0.10,
) -> None:
    """One polar plot: FullKV + StepKV r50 + StepKV r20 together."""
    _setup_style(labelsize, ticksize)

    axis_labels = [spec[1] for spec in AXIS_SPECS]
    n_axes = len(axis_labels)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(8.0, 8.0), subplot_kw={"polar": True})
    fig.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles)
    ax.set_xticklabels(axis_labels, fontsize=labelsize, fontweight="bold")

    all_norm_vals = [
        float(v) for spec in series_list for v in spec["normalized"].values()
    ]
    radial_max = max(1.05, max(all_norm_vals) * 1.05 if all_norm_vals else 1.05)
    ax.set_ylim(0, radial_max)
    tick_vals = [t for t in (0.25, 0.5, 0.75, 1.0) if t <= radial_max + 1e-9]
    outer = round(radial_max, 2)
    if outer > 1.0 and outer not in tick_vals:
        tick_vals.append(outer)
    ax.set_yticks(tick_vals)
    ax.set_yticklabels([""] * len(tick_vals))
    ax.grid(True, linestyle=":", color="#CCCCCC", linewidth=0.8)
    for line in ax.yaxis.get_gridlines():
        line.set_linestyle(":")
        line.set_color("#CCCCCC")
        line.set_linewidth(0.8)
    _style_radar_spokes(ax, linewidth=2.4)
    ax.tick_params(axis="y", labelleft=False, labelright=False)
    ax.spines["polar"].set_color("#AAAAAA")
    ax.spines["polar"].set_linewidth(1.0)

    for spec in series_list:
        label = spec["label"]
        vals = [float(spec["normalized"][key]) for key, _, _, _ in AXIS_SPECS]
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
        ax.scatter(
            angles,
            vals,
            s=46,
            facecolors=color,
            edgecolors="white",
            linewidths=0.9,
            zorder=zorder + 2,
        )

    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.22, 1.10),
        frameon=False,
        prop={"size": ticksize, "weight": "bold"},
    )

    if title:
        ax.set_title(title, fontsize=labelsize + 2, fontweight="bold", pad=24)

    for ext in ("pdf", "png"):
        out = f"{output_prefix}_radar.{ext}"
        fig.savefig(out, pad_inches=0.08)
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
        choices=("json", "metrics", "auto"),
        default="json",
        help="Input: json=result JSON (default), metrics=metrics_*.md, auto=metrics then json.",
    )
    parser.add_argument(
        "--cache_metric",
        choices=("peak", "final"),
        default="peak",
        help="Cache axes: peak = max per-step cache; final = end-of-sample cache.",
    )
    parser.add_argument(
        "--acc_display_floor",
        type=float,
        default=0.5,
        help="When FullKV EM/F1=0, add this floor (percent pts) for radar display only.",
    )
    parser.add_argument(
        "--acc_scale",
        type=float,
        default=None,
        help="Fixed EM/F1 scale (%%) when FullKV accuracy is 0; default auto from data.",
    )
    parser.add_argument(
        "--min_acc_scale",
        type=float,
        default=1.0,
        help="Minimum EM/F1 scale (%%) when FullKV accuracy is 0.",
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
        default=24,
    )
    parser.add_argument(
        "--ticksize",
        type=int,
        default=20,
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
    if args.source == "json":
        rows = load_browsecomp_stepkv_rows(run_dir, dataset_suffix)
    elif args.source == "metrics":
        rows = load_browsecomp_stepkv_rows_from_metrics(run_dir)
    else:
        rows = load_browsecomp_stepkv_rows_from_metrics(run_dir)
        if len(rows) < 3:
            print("[INFO] metrics incomplete; falling back to result JSON...")
            rows = load_browsecomp_stepkv_rows(run_dir, dataset_suffix)

    baseline = next((r for r in rows if r.method == "FullKV"), None)
    if baseline is None:
        raise RuntimeError("FullKV baseline (fullkv/react_kv_none) is required.")

    series_list = _build_normalized_series(
        rows,
        args.cache_metric,
        acc_display_floor=args.acc_display_floor,
        acc_scale_override=args.acc_scale,
        min_acc_scale=args.min_acc_scale,
    )
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
            "quality_axes": "FullKV>0: value/FullKV; FullKV=0: (value+floor)/(scale+floor)",
            "cost_axes": "value / max(all methods)",
            "acc_display_floor": args.acc_display_floor,
            "acc_scale": args.acc_scale,
            "axis_scale": series_list[0].get("axis_scale") if series_list else {},
            "axis_scale_mode": series_list[0].get("axis_scale_mode") if series_list else {},
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
