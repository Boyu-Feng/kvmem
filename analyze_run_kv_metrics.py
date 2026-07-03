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
METHOD_DISPLAY = {
    "FullKV": "FullKV",
    "H2O": r"H$_2$O",
    "TOVA": "TOVA",
    "StepKV": "StepKV",
}
RATIO_ORDER = ["r50", "r20", "full"]

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


def detect_dataset_suffix(run_dir: str) -> Optional[str]:
    for path in glob.glob(os.path.join(run_dir, "*", "react_kv_*.json")):
        name = os.path.basename(path)
        m = RESULT_JSON_RE.match(name)
        if m:
            return m.group(2)
    return None


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


def _method_tick_labels() -> List[str]:
    return [METHOD_DISPLAY[m] for m in METHOD_ORDER]


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


def plot_grouped_bars(
    rows: List[MethodRunStats],
    output_prefix: str,
) -> None:
    if plt is None:
        raise RuntimeError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        ) from _MPL_IMPORT_ERROR

    _setup_matplotlib_style()

    # Colors: colorblind-friendly (Okabe-Ito inspired)
    color_r50 = "#0072B2"
    color_r20 = "#E69F00"
    color_full = "#009E73"

    metrics = [
        ("avg_sample_time_s", "Avg. Sample Time (s)", "Time"),
        ("max_sample_time_s", "Max Sample Time (s)", "Time"),
        ("avg_peak_kv_tokens", "Avg. Peak KV Cache", "KV Cache"),
        ("max_peak_kv_tokens", "Max Peak KV Cache", "KV Cache"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    axes_flat = axes.flatten()

    n_methods = len(METHOD_ORDER)
    group_width = 0.34
    x = list(range(n_methods))

    for ax, (field, ylabel, _) in zip(axes_flat, metrics):
        vals_r50: List[float] = []
        vals_r20: List[float] = []
        vals_full: List[float] = []

        for method in METHOD_ORDER:
            full_row = _lookup(rows, method, "full")
            r50_row = _lookup(rows, method, "r50")
            r20_row = _lookup(rows, method, "r20")

            if method == "FullKV" and full_row:
                v = getattr(full_row, field)
                vals_full.append(float(v) if v is not None else 0.0)
                vals_r50.append(float(v) if v is not None else 0.0)
                vals_r20.append(float(v) if v is not None else 0.0)
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

        # FullKV: single centered bar (same height for both ratios visually -> use r50 bar only)
        full_idx = METHOD_ORDER.index("FullKV")
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
        ax.set_xticklabels(_method_tick_labels())
        ax.set_ylabel(ylabel, fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.set_axisbelow(True)

    handles = [
        mpatches.Patch(facecolor=color_r50, edgecolor="white", label="keep ratio 0.5"),
        mpatches.Patch(facecolor=color_r20, edgecolor="white", label="keep ratio 0.2"),
        mpatches.Patch(facecolor=color_full, edgecolor="white", hatch="///", label="FullKV (no prune)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.08),
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)

    for ext in ("pdf", "png"):
        out = f"{output_prefix}_grouped.{ext}"
        fig.savefig(out)
        print(f"[INFO] Saved figure: {out}")
    plt.close(fig)


def plot_normalized_efficiency(
    rows: List[MethodRunStats],
    output_prefix: str,
) -> None:
    """Normalized bars relative to FullKV."""
    if plt is None:
        return

    _setup_matplotlib_style()
    baseline = _lookup(rows, "FullKV", "full")
    if not baseline or not baseline.avg_sample_time_s or not baseline.avg_peak_kv_tokens:
        print("[WARN] Skip normalized plot: FullKV baseline missing.")
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

    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    x = list(range(len(labels)))
    width = 0.36
    ax.bar([i - width / 2 for i in x], time_vals, width=width, color=colors, alpha=0.92, label="Time / FullKV")
    ax.bar([i + width / 2 for i in x], kv_vals, width=width, color=colors, alpha=0.55, hatch="//", label="Peak KV / FullKV")
    ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Ratio vs FullKV", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper right", fontsize=10)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = f"{output_prefix}_normalized.{ext}"
        fig.savefig(out)
        print(f"[INFO] Saved figure: {out}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze KV time/cache metrics for one experiment run.")
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to one run, e.g. results/musique_qwen25_7b_v2/run2",
    )
    parser.add_argument(
        "--dataset_suffix",
        type=str,
        default=None,
        help="JSON dataset suffix (musique, browsecomp, 2wiki, ...). Auto-detect if omitted.",
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

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run_dir not found: {run_dir}")

    dataset_suffix = args.dataset_suffix or detect_dataset_suffix(run_dir)
    if not dataset_suffix:
        raise RuntimeError(
            "Could not detect dataset suffix. Pass --dataset_suffix explicitly "
            "(e.g. musique, browsecomp, 2wiki)."
        )

    output_dir = args.output_dir or os.path.join(run_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    rows = analyze_one_run(run_dir, dataset_suffix)
    if not rows:
        raise RuntimeError(f"No result JSONs found under {run_dir}")

    prefix = os.path.join(output_dir, "kv_efficiency")
    write_csv(rows, f"{prefix}_summary.csv")
    write_json_summary(rows, f"{prefix}_summary.json", run_dir, dataset_suffix)
    write_markdown_table(rows, f"{prefix}_summary.md")

    if not args.no_plot:
        plot_grouped_bars(rows, prefix)
        plot_normalized_efficiency(rows, prefix)

    print(f"[DONE] Analysis written to {output_dir}")


if __name__ == "__main__":
    main()
