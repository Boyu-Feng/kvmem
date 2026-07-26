#!/usr/bin/env python3
"""
Analyze KV-method efficiency for one experiment run (run2 / run3 / ...).

Reads result JSONs under a run directory, aggregates per-sample wall-clock time and
decode-only KV cache size, and writes tables + publication-style figures.

Expected layout (same as run_*_experiments.sh):
  {run_dir}/fullkv/react_kv_none_{dataset}.json
  {run_dir}/h2o_r50/react_kv_h2o_{dataset}_r50.json
  {run_dir}/h2o_r20/react_kv_h2o_{dataset}_r20.json
  {run_dir}/tova_r50/...
  {run_dir}/stepaware_r50/...

Example:
  python analyze_run_kv_metrics.py \\
    --run_dir results/musique_qwen25_7b_v2/run2 \\
    --output_dir results/musique_qwen25_7b_v2/run2/analysis

  # Combined 2×3 figure for all Qwen datasets (run2):
  python analyze_run_kv_metrics.py \\
    --combine_qwen --results_root results --run_tag run2 \\
    --output_dir results/qwen25_7b_v2_run2/analysis
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

from record_experiment_metrics import _step_decode_lens_from_result, compute_derived_stats

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
except ImportError as exc:  # pragma: no cover
    plt = None
    mpatches = None
    _MPL_IMPORT_ERROR = exc
else:
    _MPL_IMPORT_ERROR = None


# (subdir under run_dir, json stem without dataset/ratio, ratio tag or None, display method)
METHOD_CONFIGS: List[Tuple[str, str, Optional[str], str]] = [
    ("fullkv", "react_kv_none", None, "FullKV"),
    ("h2o_r50", "react_kv_h2o", "r50", "H2O"),
    ("h2o_r20", "react_kv_h2o", "r20", "H2O"),
    ("tova_r50", "react_kv_tova", "r50", "TOVA"),
    ("tova_r20", "react_kv_tova", "r20", "TOVA"),
    ("stepaware_r50", "react_kv_step_aware_h2o", "r50", "StepKV"),
    ("stepaware_r20", "react_kv_step_aware_h2o", "r20", "StepKV"),
]

METHOD_ORDER = ["FullKV", "H2O", "TOVA", "StepKV"]
CACHE_METHOD_ORDER = ["FullKV", "H2O", "TOVA", "TokenSkipping", "StepKV"]
METHOD_DISPLAY = {
    "FullKV": "FullKV",
    "H2O": r"H$_2$O",
    "TOVA": "TOVA",
    "StepKV": "StepKV",
    "TokenSkipping": "TSkip",
}
RATIO_ORDER = ["r50", "r20", "full"]

# TokenSkipping was run on another machine. These final decode-cache statistics
# are embedded here so the cache-size figures remain reproducible locally.
TOKEN_SKIPPING_FINAL_CACHE = {
    "hotpotqa": {
        "r50": (404.6, 977.0),
        "r20": (333.6, 394.0),
    },
    "2wiki": {
        "r50": (421.3, 1024.0),
        "r20": (336.0, 443.0),
    },
    "musique": {
        "r50": (658.6, 1612.0),
        "r20": (594.0, 683.0),
    },
}

# Preferred column order when combining Qwen multi-dataset figures.
# hotpotqa uses results/wiki_qwen25_7b_v2 (folder key "wiki").
QWEN_DATASET_ORDER = ["wiki", "2wiki", "musique"]
DATASET_DISPLAY = {
    "wiki": "HotpotQA",
    "hotpotqa": "HotpotQA",
    "2wiki": "2Wiki",
    "musique": "MuSiQue",
    "browsecomp": "BrowseComp",
}

RESULT_JSON_RE = re.compile(
    r"^(react_kv_[a-z0-9_]+)_(.+?)(?:_(r\d+))?\.json$", re.IGNORECASE
)


@dataclass
class MethodRunStats:
    key: str
    method: str
    ratio: str
    subdir: str
    result_json: str
    n_samples: int
    em: Optional[float]
    f1: Optional[float]
    avg_sample_time_s: Optional[float]
    max_sample_time_s: Optional[float]
    avg_peak_kv_tokens: Optional[float]
    max_peak_kv_tokens: Optional[float]
    avg_final_kv_tokens: Optional[float]
    max_final_kv_tokens: Optional[float]

    @property
    def label(self) -> str:
        if self.ratio == "full":
            return self.method
        return f"{self.method} ({self.ratio.replace('r', '')}%)"


def _add_hardcoded_tokenskipping_rows(
    rows: List[MethodRunStats],
    dataset_suffix: str,
    run_dir: str,
) -> None:
    """Append externally measured TokenSkipping final-cache statistics."""
    suffix = dataset_suffix.lower()
    run_dir_lower = os.path.abspath(run_dir).lower()
    if suffix == "2wiki" or "2wiki_qwen25_7b_v2" in run_dir_lower:
        dataset_key = "2wiki"
    elif suffix == "musique" or "musique_qwen25_7b_v2" in run_dir_lower:
        dataset_key = "musique"
    elif suffix in ("wiki", "hotpotqa") or "wiki_qwen25_7b_v2" in run_dir_lower:
        dataset_key = "hotpotqa"
    else:
        dataset_key = suffix
    stats = TOKEN_SKIPPING_FINAL_CACHE.get(dataset_key)
    if not stats:
        return

    rows[:] = [row for row in rows if row.method != "TokenSkipping"]
    for ratio in ("r50", "r20"):
        avg_final, max_final = stats[ratio]
        rows.append(
            MethodRunStats(
                key=f"TokenSkipping_{ratio}",
                method="TokenSkipping",
                ratio=ratio,
                subdir="hardcoded_external",
                result_json="hardcoded: external TokenSkipping run",
                n_samples=0,
                em=None,
                f1=None,
                avg_sample_time_s=None,
                max_sample_time_s=None,
                avg_peak_kv_tokens=None,
                max_peak_kv_tokens=None,
                avg_final_kv_tokens=avg_final,
                max_final_kv_tokens=max_final,
            )
        )


def _decode_cache_len(total_len: int, prompt_len: int) -> int:
    return max(0, int(total_len) - int(prompt_len))


def _per_sample_peak_kv(result: Dict[str, Any]) -> int:
    step_lens = _step_decode_lens_from_result(result)
    if step_lens:
        return int(max(step_lens))
    final_len = int(result.get("llm_stats", {}).get("final_cache_len", 0) or 0)
    if final_len > 0:
        return final_len
    prompt_len = int(result.get("prompt_token_count", 0) or 0)
    for t in result.get("step_timings") or []:
        raw = int(t.get("kv_cache_length", 0) or 0)
        if raw > 0:
            return _decode_cache_len(raw, prompt_len)
    return 0


def _per_sample_final_kv(result: Dict[str, Any]) -> int:
    final_len = int(result.get("llm_stats", {}).get("final_cache_len", 0) or 0)
    if final_len > 0:
        return final_len
    step_lens = _step_decode_lens_from_result(result)
    return int(step_lens[-1]) if step_lens else 0


def detect_all_dataset_suffixes(run_dir: str) -> List[str]:
    """Collect all dataset suffixes present under run_dir."""
    found: set[str] = set()
    for path in glob.glob(os.path.join(run_dir, "*", "react_kv_*.json")):
        name = os.path.basename(path)
        m = RESULT_JSON_RE.match(name)
        if m:
            found.add(m.group(2))
    return sorted(found)


def detect_dataset_suffix(run_dir: str) -> Optional[str]:
    for path in glob.glob(os.path.join(run_dir, "*", "react_kv_*.json")):
        name = os.path.basename(path)
        m = RESULT_JSON_RE.match(name)
        if m:
            return m.group(2)
    return None


def discover_qwen_dataset_runs(
    results_root: str,
    run_tag: str = "run2",
) -> List[Tuple[str, str, str]]:
    """Find Qwen2.5-7B experiment dirs under results_root.

    Returns list of (display_name, dataset_suffix, run_dir), sorted by QWEN_DATASET_ORDER.
    """
    results_root = os.path.abspath(results_root)
    if not os.path.isdir(results_root):
        raise FileNotFoundError(f"results_root not found: {results_root}")

    order_index = {name: i for i, name in enumerate(QWEN_DATASET_ORDER)}
    found: List[Tuple[int, str, str, str]] = []

    for entry in sorted(glob.glob(os.path.join(results_root, "*_qwen25_7b_v2"))):
        base = os.path.basename(entry)
        m = re.match(r"^(.+?)_qwen25_7b_v2$", base)
        if not m:
            continue
        folder_key = m.group(1)
        if folder_key not in order_index:
            continue
        run_dir = os.path.join(entry, run_tag)
        if not os.path.isdir(run_dir):
            print(f"[WARN] Skip {folder_key}: missing {run_dir}")
            continue
        dataset_suffix = detect_dataset_suffix(run_dir) or folder_key
        display = DATASET_DISPLAY.get(folder_key, DATASET_DISPLAY.get(dataset_suffix, folder_key))
        found.append((order_index[folder_key], display, dataset_suffix, run_dir))

    found.sort(key=lambda item: item[0])
    return [(display, suffix, run_dir) for _, display, suffix, run_dir in found]


def resolve_result_json(
    run_dir: str,
    subdir: str,
    stem: str,
    dataset_suffix: str,
    ratio_tag: Optional[str],
) -> Optional[str]:
    folder = os.path.join(run_dir, subdir)
    if not os.path.isdir(folder):
        return None

    candidates: List[str] = []
    if ratio_tag:
        candidates.append(f"{stem}_{dataset_suffix}_{ratio_tag}.json")
    candidates.append(f"{stem}_{dataset_suffix}.json")
    candidates.append(f"{stem}.json")

    for name in candidates:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            return path

    pattern = os.path.join(folder, f"{stem}_*.json")
    matches = sorted(glob.glob(pattern))
    if ratio_tag:
        tagged = [p for p in matches if f"_{ratio_tag}.json" in p]
        if tagged:
            return tagged[0]
    if matches:
        return matches[0]
    return None


def load_result_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object JSON at {path}")
    return data


def analyze_one_run(
    run_dir: str,
    dataset_suffix: str,
    method_configs: Optional[List[Tuple[str, str, Optional[str], str]]] = None,
) -> List[MethodRunStats]:
    rows: List[MethodRunStats] = []
    configs = method_configs or METHOD_CONFIGS
    for subdir, stem, ratio_tag, method_name in configs:
        json_path = resolve_result_json(run_dir, subdir, stem, dataset_suffix, ratio_tag)
        if not json_path:
            print(f"[WARN] Missing result: {subdir}/{stem}_{dataset_suffix}*.json")
            continue

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
        key = f"{method_name}_{ratio_label}"

        rows.append(
            MethodRunStats(
                key=key,
                method=method_name,
                ratio=ratio_label,
                subdir=subdir,
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
        )
        print(f"[OK] {key}: n={rows[-1].n_samples} json={json_path}")
    return rows


def _as_float(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _mean(values: List[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _max(values: List[float]) -> Optional[float]:
    vals = [float(v) for v in values if v is not None]
    return max(vals) if vals else None


def _max_int(values: List[int]) -> Optional[int]:
    vals = [int(v) for v in values if v is not None and v >= 0]
    return max(vals) if vals else None


def write_csv(rows: List[MethodRunStats], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json_summary(rows: List[MethodRunStats], path: str, run_dir: str, dataset_suffix: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "run_dir": os.path.abspath(run_dir),
        "dataset_suffix": dataset_suffix,
        "methods": [asdict(r) for r in rows],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def write_markdown_table(rows: List[MethodRunStats], path: str) -> None:
    lines = [
        "# KV Efficiency Summary",
        "",
        "| Method | Ratio | N | EM | F1 | Avg Time (s) | Max Time (s) | Avg Peak KV | Max Peak KV |",
        "|--------|-------|---|----|----|--------------|--------------|-------------|-------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r.method} | {r.ratio} | {r.n_samples} | "
            f"{_fmt(r.em, 2)} | {_fmt(r.f1, 2)} | "
            f"{_fmt(r.avg_sample_time_s, 1)} | {_fmt(r.max_sample_time_s, 1)} | "
            f"{_fmt(r.avg_peak_kv_tokens, 0)} | {_fmt(r.max_peak_kv_tokens, 0)} |"
        )
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _fmt(v: Optional[float], digits: int) -> str:
    if v is None:
        return "—"
    if digits == 0:
        return f"{int(round(v))}"
    return f"{v:.{digits}f}"


def _lookup(rows: List[MethodRunStats], method: str, ratio: str) -> Optional[MethodRunStats]:
    for r in rows:
        if r.method == method and r.ratio == ratio:
            return r
    return None


def _method_tick_labels(method_order: List[str]) -> List[str]:
    return [METHOD_DISPLAY[m] for m in method_order]


def _setup_matplotlib_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _draw_method_ratio_bars(
    ax,
    rows: List[MethodRunStats],
    field: str,
    ylabel: str,
    *,
    color_r50: str = "#0072B2",
    color_r20: str = "#E69F00",
    color_full: str = "#009E73",
    show_xticklabels: bool = True,
    show_ylabel: bool = True,
    xtick_rotation: float = 0,
    method_order: Optional[List[str]] = None,
) -> None:
    """Grouped bars: FullKV + r50/r20 per method on one axes."""
    methods = method_order or METHOD_ORDER
    n_methods = len(methods)
    group_width = 0.34
    x = list(range(n_methods))

    vals_r50: List[float] = []
    vals_r20: List[float] = []
    vals_full: List[float] = []

    for method in methods:
        full_row = _lookup(rows, method, "full")
        r50_row = _lookup(rows, method, "r50")
        r20_row = _lookup(rows, method, "r20")

        if method == "FullKV" and full_row:
            v = getattr(full_row, field)
            val = float(v) if v is not None else 0.0
            vals_full.append(val)
            vals_r50.append(val)
            vals_r20.append(val)
        else:
            v50 = getattr(r50_row, field) if r50_row else None
            v20 = getattr(r20_row, field) if r20_row else None
            vals_r50.append(float(v50) if v50 is not None else 0.0)
            vals_r20.append(float(v20) if v20 is not None else 0.0)
            vals_full.append(0.0)

    bar_r50 = ax.bar(
        [xi - group_width / 2 for xi in x],
        vals_r50,
        width=group_width,
        color=color_r50,
        label="keep ratio 0.5",
        edgecolor="white",
        linewidth=0.6,
    )
    bar_r20 = ax.bar(
        [xi + group_width / 2 for xi in x],
        vals_r20,
        width=group_width,
        color=color_r20,
        label="keep ratio 0.2",
        edgecolor="white",
        linewidth=0.6,
    )

    full_idx = methods.index("FullKV")
    if vals_full[full_idx] > 0:
        for b in (bar_r50[full_idx], bar_r20[full_idx]):
            b.set_visible(False)
        ax.bar(
            x[full_idx],
            vals_full[full_idx],
            width=group_width,
            color=color_full,
            edgecolor="white",
            linewidth=0.6,
            hatch="///",
        )

    ax.set_xticks(x)
    if show_xticklabels:
        labels = ax.set_xticklabels(
            _method_tick_labels(methods),
            rotation=xtick_rotation,
            ha="center",
            rotation_mode="anchor",
        )
        # Slight right shift so rotated labels align with bar-group centers.
        if xtick_rotation:
            for lbl in labels:
                lbl.set_x(lbl.get_position()[0] + 0.08)
    else:
        ax.set_xticklabels([])
    if show_ylabel and ylabel:
        ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    if show_xticklabels:
        ax.tick_params(axis="x", labelsize=11)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)


def _legend_patches(
    color_r50: str = "#0072B2",
    color_r20: str = "#E69F00",
    color_full: str = "#009E73",
):
    return [
        mpatches.Patch(facecolor=color_r50, edgecolor="white", label="keep ratio 0.5"),
        mpatches.Patch(facecolor=color_r20, edgecolor="white", label="keep ratio 0.2"),
        mpatches.Patch(facecolor=color_full, edgecolor="white", hatch="///", label="FullKV (no prune)"),
    ]


def _add_bottom_legend(
    fig,
    color_r50: str,
    color_r20: str,
    color_full: str,
    *,
    fontsize: int = 10,
    y_anchor: float = 0.02,
) -> None:
    """Shared legend below all panels, kept close to the figure."""
    fig.legend(
        handles=_legend_patches(color_r50, color_r20, color_full),
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=fontsize,
        bbox_to_anchor=(0.5, y_anchor),
    )


def plot_multi_dataset_grid(
    rows_by_dataset: List[Tuple[str, str, List[MethodRunStats]]],
    *,
    avg_field: str,
    avg_ylabel: str,
    max_field: str,
    max_ylabel: str,
    output_prefix: str,
    stem: str,
    method_order: Optional[List[str]] = None,
    figure_height: float = 6.6,
) -> None:
    """2×N grid: top row = avg, bottom row = max; one column per dataset."""
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from _MPL_IMPORT_ERROR
    if not rows_by_dataset:
        print("[WARN] Skip multi-dataset plot: no datasets.")
        return

    _setup_matplotlib_style()
    color_r50, color_r20, color_full = "#0072B2", "#E69F00", "#009E73"
    n_cols = len(rows_by_dataset)

    fig, axes = plt.subplots(
        2,
        n_cols,
        figsize=(3.4 * n_cols + 1.4, figure_height),
        squeeze=False,
    )

    for col, (display_name, _suffix, rows) in enumerate(rows_by_dataset):
        ax_avg = axes[0, col]
        ax_max = axes[1, col]

        _draw_method_ratio_bars(
            ax_avg,
            rows,
            avg_field,
            avg_ylabel,
            color_r50=color_r50,
            color_r20=color_r20,
            color_full=color_full,
            show_xticklabels=True,
            show_ylabel=(col == 0),
            xtick_rotation=12,
            method_order=method_order,
        )
        ax_avg.set_title(display_name, fontsize=13, pad=6)

        _draw_method_ratio_bars(
            ax_max,
            rows,
            max_field,
            max_ylabel,
            color_r50=color_r50,
            color_r20=color_r20,
            color_full=color_full,
            show_xticklabels=True,
            show_ylabel=(col == 0),
            xtick_rotation=12,
            method_order=method_order,
        )

    _add_bottom_legend(fig, color_r50, color_r20, color_full, fontsize=13, y_anchor=-0.002)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.subplots_adjust(hspace=0.18, wspace=0.28)

    out_stem = f"{output_prefix}_{stem}"
    for ext in ("pdf", "png"):
        out = f"{out_stem}.{ext}"
        fig.savefig(out)
        print(f"[INFO] Saved figure: {out}")
    plt.close(fig)


def plot_combined_qwen_figures(
    rows_by_dataset: List[Tuple[str, str, List[MethodRunStats]]],
    output_prefix: str,
) -> None:
    plot_multi_dataset_grid(
        rows_by_dataset,
        avg_field="avg_sample_time_s",
        avg_ylabel="Avg. Sample Time (s)",
        max_field="max_sample_time_s",
        max_ylabel="Max Sample Time (s)",
        output_prefix=output_prefix,
        stem="qwen_multi_time",
    )
    plot_multi_dataset_grid(
        rows_by_dataset,
        avg_field="avg_final_kv_tokens",
        avg_ylabel="Avg Cache",
        max_field="max_final_kv_tokens",
        max_ylabel="Max Cache",
        output_prefix=output_prefix,
        stem="qwen_multi_cache",
        method_order=CACHE_METHOD_ORDER,
        figure_height=5.2,
    )


TIME_BAR_METRICS: List[Tuple[str, str]] = [
    ("avg_sample_time_s", "Avg. Sample Time (s)"),
    ("max_sample_time_s", "Max Sample Time (s)"),
]

CACHE_BAR_METRICS: List[Tuple[str, str]] = [
    ("avg_final_kv_tokens", "Avg Cache"),
    ("max_final_kv_tokens", "Max Cache"),
]


def plot_time_bars(
    rows: List[MethodRunStats],
    output_prefix: str,
    dataset_suffix: str,
) -> None:
    """One figure per dataset: sample time only (avg + max as side-by-side panels)."""
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from _MPL_IMPORT_ERROR

    _setup_matplotlib_style()
    color_r50, color_r20, color_full = "#0072B2", "#E69F00", "#009E73"

    fig, axes = plt.subplots(1, len(TIME_BAR_METRICS), figsize=(7.2, 4.0))
    if len(TIME_BAR_METRICS) == 1:
        axes = [axes]
    for ax, (field, ylabel) in zip(axes, TIME_BAR_METRICS):
        _draw_method_ratio_bars(
            ax, rows, field, ylabel,
            color_r50=color_r50, color_r20=color_r20, color_full=color_full,
        )

    _add_bottom_legend(fig, color_r50, color_r20, color_full)
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    stem = f"{output_prefix}_{dataset_suffix}_time"
    for ext in ("pdf", "png"):
        out = f"{stem}.{ext}"
        fig.savefig(out)
        print(f"[INFO] Saved figure: {out}")
    plt.close(fig)


def plot_cache_bars(
    rows: List[MethodRunStats],
    output_prefix: str,
    dataset_suffix: str,
) -> None:
    """One figure per dataset: KV cache size only (avg + max as side-by-side panels)."""
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from _MPL_IMPORT_ERROR

    _setup_matplotlib_style()
    color_r50, color_r20, color_full = "#0072B2", "#E69F00", "#009E73"

    fig, axes = plt.subplots(1, len(CACHE_BAR_METRICS), figsize=(7.2, 4.0))
    if len(CACHE_BAR_METRICS) == 1:
        axes = [axes]
    for ax, (field, ylabel) in zip(axes, CACHE_BAR_METRICS):
        _draw_method_ratio_bars(
            ax, rows, field, ylabel,
            color_r50=color_r50, color_r20=color_r20, color_full=color_full,
            method_order=CACHE_METHOD_ORDER,
        )

    _add_bottom_legend(fig, color_r50, color_r20, color_full)
    fig.tight_layout(rect=(0, 0.07, 1, 1))

    stem = f"{output_prefix}_{dataset_suffix}_cache"
    for ext in ("pdf", "png"):
        out = f"{stem}.{ext}"
        fig.savefig(out)
        print(f"[INFO] Saved figure: {out}")
    plt.close(fig)


def plot_grouped_bars(
    rows: List[MethodRunStats],
    output_prefix: str,
    dataset_suffix: str,
) -> None:
    """Plot separate time and cache figures for one dataset."""
    plot_time_bars(rows, output_prefix, dataset_suffix)
    plot_cache_bars(rows, output_prefix, dataset_suffix)


def plot_normalized_efficiency(
    rows: List[MethodRunStats],
    output_prefix: str,
    dataset_suffix: str,
) -> None:
    """Normalized bars relative to FullKV — time and cache in separate figures."""
    if plt is None:
        return

    _setup_matplotlib_style()
    baseline = _lookup(rows, "FullKV", "full")
    if not baseline or not baseline.avg_sample_time_s or not baseline.avg_peak_kv_tokens:
        print(f"[WARN] Skip normalized plot for {dataset_suffix}: FullKV baseline missing.")
        return

    entries: List[Tuple[str, str, float, float]] = []
    for method in ["H2O", "TOVA", "StepKV"]:
        for ratio in ["r50", "r20"]:
            row = _lookup(rows, method, ratio)
            if not row or row.avg_sample_time_s is None or row.avg_peak_kv_tokens is None:
                continue
            time_ratio = row.avg_sample_time_s / baseline.avg_sample_time_s
            kv_ratio = row.avg_peak_kv_tokens / baseline.avg_peak_kv_tokens
            label = f"{METHOD_DISPLAY.get(method, method)}\n{ratio.replace('r', '')}%"
            entries.append((label, ratio, time_ratio, kv_ratio))

    if not entries:
        return

    labels = [e[0] for e in entries]
    time_vals = [e[2] for e in entries]
    kv_vals = [e[3] for e in entries]
    colors = ["#0072B2" if e[1] == "r50" else "#E69F00" for e in entries]

    for category, vals, ylabel, stem_suffix in (
        ("Time / FullKV", time_vals, "Ratio vs FullKV (time)", "time"),
        ("Peak KV / FullKV", kv_vals, "Ratio vs FullKV (cache)", "cache"),
    ):
        fig, ax = plt.subplots(figsize=(7.0, 3.6))
        x = list(range(len(labels)))
        ax.bar(x, vals, width=0.55, color=colors, alpha=0.92)
        ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.set_axisbelow(True)
        fig.tight_layout()

        for ext in ("pdf", "png"):
            out = f"{output_prefix}_{dataset_suffix}_normalized_{stem_suffix}.{ext}"
            fig.savefig(out)
            print(f"[INFO] Saved figure: {out}")
        plt.close(fig)


def _analyze_and_write_dataset(
    run_dir: str,
    dataset_suffix: str,
    output_dir: str,
    *,
    no_plot: bool,
    skip_individual_plots: bool = False,
) -> Optional[List[MethodRunStats]]:
    rows = analyze_one_run(run_dir, dataset_suffix)
    _add_hardcoded_tokenskipping_rows(rows, dataset_suffix, run_dir)
    if not rows:
        print(f"[WARN] No results for dataset={dataset_suffix}")
        return None

    prefix = os.path.join(output_dir, "kv_efficiency")
    write_csv(rows, f"{prefix}_{dataset_suffix}_summary.csv")
    write_json_summary(rows, f"{prefix}_{dataset_suffix}_summary.json", run_dir, dataset_suffix)
    write_markdown_table(rows, f"{prefix}_{dataset_suffix}_summary.md")

    if not no_plot and not skip_individual_plots:
        plot_grouped_bars(rows, prefix, dataset_suffix)
        plot_normalized_efficiency(rows, prefix, dataset_suffix)
    return rows


def _run_combine_qwen(
    results_root: str,
    run_tag: str,
    output_dir: str,
    *,
    no_plot: bool,
) -> None:
    datasets = discover_qwen_dataset_runs(results_root, run_tag=run_tag)
    if not datasets:
        raise RuntimeError(
            f"No Qwen run dirs found under {results_root}/*_qwen25_7b_v2/{run_tag}"
        )

    rows_by_dataset: List[Tuple[str, str, List[MethodRunStats]]] = []
    for display_name, dataset_suffix, run_dir in datasets:
        print(f"[INFO] Analyzing Qwen dataset={dataset_suffix} run_dir={run_dir}")
        rows = _analyze_and_write_dataset(
            run_dir,
            dataset_suffix,
            output_dir,
            no_plot=no_plot,
            skip_individual_plots=True,
        )
        if rows:
            rows_by_dataset.append((display_name, dataset_suffix, rows))

    if not no_plot and rows_by_dataset:
        prefix = os.path.join(output_dir, "kv_efficiency")
        plot_combined_qwen_figures(rows_by_dataset, prefix)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze KV time/cache metrics for one experiment run.")
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Path to one run, e.g. results/musique_qwen25_7b_v2/run2",
    )
    parser.add_argument(
        "--combine_qwen",
        action="store_true",
        help="Discover all Qwen datasets under --results_root, read --run_tag (default run2), "
        "and write combined 2×N time/cache figures.",
    )
    parser.add_argument(
        "--results_root",
        type=str,
        default="results",
        help="Root folder for --combine_qwen (default: results)",
    )
    parser.add_argument(
        "--run_tag",
        type=str,
        default="run2",
        help="Run subfolder name for --combine_qwen (default: run2)",
    )
    parser.add_argument(
        "--dataset_suffix",
        type=str,
        default=None,
        help="JSON dataset suffix (musique, browsecomp, 2wiki, ...). "
        "Use 'all' for every dataset found under run_dir. Auto-detect if omitted.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to write tables/figures (default: {run_dir}/analysis)",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Only write CSV/JSON/Markdown, skip figures.",
    )
    args = parser.parse_args()

    if args.combine_qwen:
        output_dir = args.output_dir or os.path.join(
            os.path.abspath(args.results_root),
            f"qwen25_7b_{args.run_tag}_analysis",
        )
        os.makedirs(output_dir, exist_ok=True)
        _run_combine_qwen(
            args.results_root,
            args.run_tag,
            output_dir,
            no_plot=args.no_plot,
        )
        print(f"[DONE] Combined Qwen analysis written to {output_dir}")
        return

    if not args.run_dir:
        parser.error("--run_dir is required unless --combine_qwen is set.")

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    output_dir = args.output_dir or os.path.join(run_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    if args.dataset_suffix and args.dataset_suffix.lower() == "all":
        datasets = detect_all_dataset_suffixes(run_dir)
    elif args.dataset_suffix:
        datasets = [args.dataset_suffix]
    else:
        one = detect_dataset_suffix(run_dir)
        datasets = [one] if one else []

    if not datasets:
        raise RuntimeError(
            "Could not detect dataset suffix. Pass --dataset_suffix explicitly "
            "(e.g. musique, browsecomp, 2wiki) or --dataset_suffix all."
        )

    for dataset_suffix in datasets:
        print(f"[INFO] Analyzing dataset={dataset_suffix}")
        _analyze_and_write_dataset(
            run_dir, dataset_suffix, output_dir, no_plot=args.no_plot
        )

    print(f"[DONE] Analysis written to {output_dir}")


if __name__ == "__main__":
    main()
