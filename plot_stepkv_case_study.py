#!/usr/bin/env python3
"""
StepKV gap case study outputs (baseline fail / StepKV success):

  - critical_tokens.json : baseline discarded but StepKV kept (all such tokens)
  - *_score_decomp.pdf    : grouped token vs step score bars (raw values)
  - *_global_heatmap.pdf  : full-decode causal score heatmap (log + percentile norm)

Example:
  # Scan qualifying gap samples and save manifest (run once)
  python plot_stepkv_case_study.py --dataset hotpotqa --scan_gap_only --scan_limit 100

  # Run case study from saved manifest (no rescan)
  python plot_stepkv_case_study.py --dataset hotpotqa --gap_index 0

  # Or pick by saved pos directly
  python plot_stepkv_case_study.py --dataset hotpotqa --sample_pos 12
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import run_all_wiki_experiments_v2 as wiki_base
from analyze_stepkv_discarded_tokens import (
    BASELINE_METHODS,
    METHOD_LABELS,
    OURS_METHOD,
    SCORE_METHODS,
    _apply_causal_display_mask,
    _build_decode_score_square,
    _build_success_gap_token_text_report,
    _discard_sets_from_run,
    _evaluate_run,
    _latest_token_score_snapshot,
    _prepare_dataset,
    _run_one_sample,
    _save_json,
    _save_success_gap_token_detail_files,
    _save_tokens_jsonl,
    _scored_decode_len,
    _summarize_discard_diff,
    _get_analysis_tokenizer,
    _token_text_record,
    extract_decode_token_scores,
    run_success_gap_one_sample,
)

HEATMAP_MAX_LEN = 140


@dataclass
class CriticalTokenRow:
    decode_idx: int
    label: str
    owner_step: int
    h2o_discarded: bool
    tova_discarded: bool
    stepkv_discarded: bool
    token_score: float
    step_score: float
    combined_score: float


def _minmax_norm(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi <= lo:
        return np.ones_like(arr) if hi > 0 else np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _normalize_heatmap_display(mat: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Log1p + percentile stretch on lower triangle for visible contrast."""
    out = np.array(mat, dtype=float)
    n = out.shape[0]
    if n <= 0:
        return out, {"lo": 0.0, "hi": 1.0, "p_low": 2.0, "p_high": 98.0}

    tril = np.tril(np.ones((n, n), dtype=bool), k=0)
    vals = out[tril & np.isfinite(out)]
    pos = vals[vals > 0] if np.any(vals > 0) else vals
    if pos.size == 0:
        display = np.full_like(out, np.nan)
        return display, {"lo": 0.0, "hi": 1.0, "p_low": 2.0, "p_high": 98.0}

    logged = np.log1p(pos)
    p_low, p_high = 2.0, 98.0
    lo = float(np.percentile(logged, p_low))
    hi = float(np.percentile(logged, p_high))
    if hi <= lo:
        hi = lo + 1e-8

    display = np.full_like(out, np.nan)
    for i in range(n):
        for j in range(i + 1):
            v = out[i, j]
            if not np.isfinite(v):
                continue
            lv = float(np.log1p(max(0.0, v)))
            display[i, j] = float(np.clip((lv - lo) / (hi - lo), 0.0, 1.0))

    meta = {"lo": lo, "hi": hi, "p_low": p_low, "p_high": p_high, "log1p": 1.0}
    return display, meta


def _decode_token_label(
    debug_payload: Dict[str, Any],
    decode_idx: int,
    tokenizer,
    *,
    max_len: int = 14,
) -> str:
    prompt_len = int(debug_payload.get("prompt_token_count", 0) or 0)
    global_ids = list(debug_payload.get("global_token_ids", []) or [])
    gid = prompt_len + int(decode_idx)
    rec = _token_text_record(tokenizer, global_ids, prompt_len, gid)
    text = (rec.get("text_clean") or rec.get("text") or "").replace("\n", " ").replace("\r", " ").strip()
    text = text.replace("▁", " ").strip()
    if not text:
        tid = rec.get("token_id")
        text = f"<{tid}>" if tid is not None else "?"
    if len(text) > max_len:
        text = text[: max_len - 1] + "…"
    return text


def _owner_step_for_decode(debug_payload: Dict[str, Any], decode_idx: int) -> int:
    prompt_len = int(debug_payload.get("prompt_token_count", 0) or 0)
    gid = prompt_len + int(decode_idx)
    for sid_str, rng in (debug_payload.get("step_token_ranges", {}) or {}).items():
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            continue
        if int(rng[0]) <= gid <= int(rng[1]):
            return int(sid_str)
    return -1


def _score_at(score_info: Dict[str, Any], field: str, idx: int) -> float:
    vals = score_info.get(field) or []
    if 0 <= idx < len(vals) and vals[idx] is not None:
        return float(vals[idx])
    return 0.0


def _all_critical_indices(
    method_runs: Dict[str, Dict[str, Any]],
    discard_diff: Dict[str, Any],
) -> List[int]:
    """All decode indices discarded by any baseline but kept by StepKV."""
    ours_discarded = _discard_sets_from_run(method_runs[OURS_METHOD])
    return sorted(
        {
            int(i)
            for i in (discard_diff.get("only_baseline_not_ours", []) or [])
            if int(i) not in ours_discarded
        }
    )


def _pick_score_contrast_tokens(
    rows: Sequence[CriticalTokenRow],
    *,
    max_tokens: int = 12,
) -> List[CriticalTokenRow]:
    """Pick tokens where token score and step score differ most (easier to read in panel B)."""

    def _key(row: CriticalTokenRow) -> Tuple[float, float, float]:
        diff = abs(float(row.token_score) - float(row.step_score))
        denom = max(1e-8, max(abs(row.token_score), abs(row.step_score)))
        rel = diff / denom
        return (rel, diff, float(row.combined_score))

    ranked = sorted(rows, key=_key, reverse=True)
    return list(ranked[: int(max_tokens)])


def _build_critical_rows(
    method_runs: Dict[str, Dict[str, Any]],
    decode_indices: Sequence[int],
    tokenizer,
) -> List[CriticalTokenRow]:
    discard_sets = {m: _discard_sets_from_run(method_runs[m]) for m in SCORE_METHODS}
    ours_payload = method_runs[OURS_METHOD].get("debug_payload", {}) or {}
    score_info = extract_decode_token_scores(ours_payload)

    rows: List[CriticalTokenRow] = []
    for idx in decode_indices:
        token_s = _score_at(score_info, "hh_scores", idx)
        step_s = _score_at(score_info, "step_scores", idx)
        combined = _score_at(score_info, "combined_scores", idx)
        if combined <= 0:
            combined = token_s + step_s
        rows.append(
            CriticalTokenRow(
                decode_idx=int(idx),
                label=_decode_token_label(ours_payload, int(idx), tokenizer),
                owner_step=_owner_step_for_decode(ours_payload, int(idx)),
                h2o_discarded=int(idx) in discard_sets["h2o"],
                tova_discarded=int(idx) in discard_sets["tova"],
                stepkv_discarded=int(idx) in discard_sets[OURS_METHOD],
                token_score=token_s,
                step_score=step_s,
                combined_score=combined,
            )
        )
    return rows


def _format_header(gap_row: Dict[str, Any]) -> str:
    evals = gap_row.get("evaluations", {}) or {}
    em_parts = []
    for method in BASELINE_METHODS + [OURS_METHOD]:
        label = METHOD_LABELS.get(method, method)
        ok = bool((evals.get(method) or {}).get("exact_match"))
        em_parts.append(f"{label}: {'✓' if ok else '✗'}")
    q = str(gap_row.get("question", "") or "").replace("\n", " ")
    gold = str(gap_row.get("gold_answer", "") or "").replace("\n", " ")
    if len(q) > 110:
        q = q[:107] + "..."
    if len(gold) > 60:
        gold = gold[:57] + "..."
    return f"Q: {q}    Gold: {gold}    |    " + "    ".join(em_parts)


def _save_figure(fig: plt.Figure, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".pdf":
        fig.savefig(output_path, bbox_inches="tight")
    else:
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _export_critical_tokens_file(
    output_path: str,
    gap_row: Dict[str, Any],
    all_rows: List[CriticalTokenRow],
) -> None:
    payload = {
        "header": _format_header(gap_row),
        "question": gap_row.get("question", ""),
        "gold_answer": gap_row.get("gold_answer", ""),
        "evaluations": gap_row.get("evaluations", {}),
        "description": "Tokens discarded by H2O and/or TOVA but kept by StepKV.",
        "count": len(all_rows),
        "tokens": [asdict(r) for r in all_rows],
    }
    _save_json(output_path, payload)


def _save_panel_b_figure(
    output_path: str,
    gap_row: Dict[str, Any],
    rows: List[CriticalTokenRow],
) -> None:
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13})
    fig, ax = plt.subplots(figsize=(14, 5.5))

    fig.suptitle(_format_header(gap_row), fontsize=10, y=1.02)
    ax.set_title(
        "StepKV score decomposition — token (H2O) vs step (raw scores, contrast-picked tokens)",
        loc="left",
        fontsize=13,
        pad=10,
    )

    if not rows:
        ax.text(0.5, 0.5, "No tokens selected", ha="center", va="center")
        ax.axis("off")
        _save_figure(fig, output_path)
        return

    x = np.arange(len(rows))
    width = 0.36
    token_vals = [r.token_score for r in rows]
    step_vals = [r.step_score for r in rows]
    labels = [r.label for r in rows]

    ax.bar(x - width / 2, token_vals, width, color="#6BAED6", edgecolor="white", linewidth=0.5, label="Token score (H2O)")
    ax.bar(x + width / 2, step_vals, width, color="#FC9272", edgecolor="white", linewidth=0.5, label="Step score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Raw score", fontsize=12)
    ymax = max(token_vals + step_vals + [1e-6]) * 1.15
    ax.set_ylim(0, ymax)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=10)
    fig.tight_layout()
    _save_figure(fig, output_path)


def _downsample_matrix(mat: np.ndarray, max_len: int) -> Tuple[np.ndarray, int]:
    n = int(mat.shape[0])
    if n <= max_len:
        return mat, 1
    step = int(np.ceil(n / max_len))
    return mat[::step, ::step], step


def _save_global_heatmap_figure(
    output_path: str,
    gap_row: Dict[str, Any],
    debug_payload: Dict[str, Any],
    critical_indices: Sequence[int],
) -> Dict[str, Any]:
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13})

    score_info = extract_decode_token_scores(debug_payload)
    snap = _latest_token_score_snapshot(debug_payload) or {}
    prompt_len = int(score_info.get("prompt_token_count", debug_payload.get("prompt_token_count", 0)) or 0)
    plot_len = max(int(score_info.get("decode_len", 0) or 0), _scored_decode_len(score_info))
    plot_len = max(plot_len, 1)

    mat, source = _build_decode_score_square(snap, prompt_len, plot_len, score_info)
    mat = _apply_causal_display_mask(mat)
    mat, stride = _downsample_matrix(mat, HEATMAP_MAX_LEN)
    display, norm_meta = _normalize_heatmap_display(mat)

    fig_h = float(min(12.0, max(4.5, display.shape[0] * 0.07)))
    fig, ax = plt.subplots(figsize=(14, fig_h))
    fig.suptitle(_format_header(gap_row), fontsize=10, y=1.02)
    ax.set_title(
        f"Global causal score heatmap (decode 0–{plot_len - 1}, log+percentile norm)",
        loc="left",
        fontsize=13,
        pad=10,
    )

    im = ax.imshow(
        display,
        cmap="viridis",
        interpolation="nearest",
        aspect="auto",
        origin="upper",
        vmin=0.0,
        vmax=1.0,
    )

    for idx in critical_indices:
        local = int(idx) // stride
        if 0 <= local < display.shape[0]:
            ax.axvline(local, color="#FF6B6B", linewidth=0.5, alpha=0.55)
            ax.axhline(local, color="#FF6B6B", linewidth=0.5, alpha=0.55)

    ax.set_xlabel("Key token index (decode, downsampled)" if stride > 1 else "Key token index (decode)", fontsize=11)
    ax.set_ylabel("Query token index (decode)", fontsize=11)
    ax.text(
        0.02,
        0.98,
        f"source={source}; norm=log1p + p{norm_meta['p_low']:.0f}–p{norm_meta['p_high']:.0f}; stride={stride}",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        color="#555555",
    )
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Relative score")
    fig.tight_layout()
    _save_figure(fig, output_path)

    return {
        "plot_len": int(plot_len),
        "matrix_source": source,
        "downsample_stride": int(stride),
        "norm": norm_meta,
    }


def _export_decode_token_status(
    method_runs: Dict[str, Dict[str, Any]],
    tokenizer,
    critical_indices: Sequence[int],
    output_jsonl: str,
) -> None:
    discard_sets = {m: _discard_sets_from_run(method_runs[m]) for m in SCORE_METHODS}
    critical_set = {int(i) for i in critical_indices}

    decode_lens = []
    for method in SCORE_METHODS:
        payload = method_runs[method].get("debug_payload", {}) or {}
        tracker = payload.get("token_tracker", {}) or {}
        prompt_len = int(payload.get("prompt_token_count", 0) or 0)
        next_gid = int(tracker.get("next_global_id", prompt_len) or prompt_len)
        decode_lens.append(max(0, next_gid - prompt_len))
    max_decode = max(decode_lens) if decode_lens else 0

    rows: List[Dict[str, Any]] = []
    ref_payload = method_runs[OURS_METHOD].get("debug_payload", {}) or {}
    for decode_idx in range(max_decode):
        label = _decode_token_label(ref_payload, decode_idx, tokenizer)
        row: Dict[str, Any] = {
            "decode_index": int(decode_idx),
            "label": label,
            "is_critical_baseline_drop_stepkv_keep": int(decode_idx) in critical_set,
        }
        for method in SCORE_METHODS:
            label_m = METHOD_LABELS.get(method, method)
            discarded = int(decode_idx) in discard_sets[method]
            row[f"{label_m}_status"] = "discarded" if discarded else "kept"
        rows.append(row)

    os.makedirs(os.path.dirname(output_jsonl) or ".", exist_ok=True)
    _save_tokens_jsonl(output_jsonl, rows)


def save_case_study_outputs(
    gap_row: Dict[str, Any],
    method_runs: Dict[str, Dict[str, Any]],
    output_prefix: str,
    *,
    max_score_tokens: int = 12,
    tokenizer=None,
) -> Dict[str, Any]:
    discard_diff = _summarize_discard_diff(
        {m: method_runs[m] for m in BASELINE_METHODS},
        method_runs[OURS_METHOD],
    )
    critical_indices = _all_critical_indices(method_runs, discard_diff)
    if not critical_indices:
        raise RuntimeError(
            "No critical tokens (only_baseline_not_ours) found. "
            "Need a baseline_fail_ours_success sample with discard differences."
        )

    if tokenizer is None:
        tokenizer = _get_analysis_tokenizer(wiki_base.MODEL_PATH)

    all_rows = _build_critical_rows(method_runs, critical_indices, tokenizer)
    score_rows = _pick_score_contrast_tokens(all_rows, max_tokens=max_score_tokens)
    ours_payload = method_runs[OURS_METHOD].get("debug_payload", {}) or {}

    critical_json = f"{output_prefix}_critical_tokens.json"
    panel_b_pdf = f"{output_prefix}_score_decomp.pdf"
    panel_c_pdf = f"{output_prefix}_global_heatmap.pdf"

    _export_critical_tokens_file(critical_json, gap_row, all_rows)
    _save_panel_b_figure(panel_b_pdf, gap_row, score_rows)
    heatmap_meta = _save_global_heatmap_figure(panel_c_pdf, gap_row, ours_payload, critical_indices)

    return {
        "critical_tokens_json": critical_json,
        "score_decomp_pdf": panel_b_pdf,
        "global_heatmap_pdf": panel_c_pdf,
        "all_critical_count": len(all_rows),
        "score_plot_tokens": [asdict(r) for r in score_rows],
        "heatmap_meta": heatmap_meta,
    }


def _pick_case_study_sample(
    selected: List[Tuple[int, Dict[str, Any]]],
    *,
    sample_pos: Optional[int] = None,
    orig_idx: Optional[int] = None,
    sample_id: str = "",
) -> Optional[Tuple[int, int, Dict[str, Any]]]:
    """Resolve one sample by pos / orig_idx / id. Returns None when none specified."""
    if sample_id:
        for pos, (oidx, sample) in enumerate(selected):
            if str(sample.get("id", "")) == str(sample_id):
                return pos, oidx, sample
        raise ValueError(f"sample_id not found in selected set: {sample_id}")
    if orig_idx is not None:
        for pos, (oidx, sample) in enumerate(selected):
            if int(oidx) == int(orig_idx):
                return pos, oidx, sample
        raise ValueError(f"orig_idx not found in selected set: {orig_idx}")
    if sample_pos is not None:
        pos = int(sample_pos)
        if pos < 0 or pos >= len(selected):
            raise IndexError(f"sample_pos out of range: {pos} (total {len(selected)})")
        oidx, sample = selected[pos]
        return pos, oidx, sample
    return None


def _print_sample_catalog(selected: List[Tuple[int, Dict[str, Any]]], args) -> None:
    print(
        f"[INFO] dataset={args.dataset} seed={args.seed} num_samples={args.num_samples} "
        f"selected={len(selected)}"
    )
    print(f"{'pos':>5}  {'orig_idx':>8}  {'id':<24}  question")
    print("-" * 100)
    for pos, (orig_idx, sample) in enumerate(selected):
        sid = str(sample.get("id", "") or "")[:24]
        q = str(sample.get("question", "") or "").replace("\n", " ").strip()
        if len(q) > 72:
            q = q[:69] + "..."
        print(f"{pos:5d}  {orig_idx:8d}  {sid:<24}  {q}")


def _print_gap_sample_catalog(
    selected: List[Tuple[int, Dict[str, Any]]],
    found: List[Tuple[int, int, Dict[str, Any], Dict[str, Any]]],
) -> None:
    print(f"[INFO] baseline_fail_ours_success samples: {len(found)}")
    print(f"{'gap_idx':>7}  {'pos':>5}  {'orig_idx':>8}  {'id':<24}  question")
    print("-" * 100)
    for gap_idx, (pos, orig_idx, sample, row) in enumerate(found):
        sid = str(sample.get("id", "") or "")[:24]
        q = str(sample.get("question", "") or "").replace("\n", " ").strip()
        if len(q) > 60:
            q = q[:57] + "..."
        evals = row.get("evaluations", {}) or {}
        marks = []
        for method in BASELINE_METHODS + [OURS_METHOD]:
            label = METHOD_LABELS.get(method, method)
            ok = bool((evals.get(method) or {}).get("exact_match"))
            marks.append(f"{label}:{'✓' if ok else '✗'}")
        print(f"{gap_idx:7d}  {pos:5d}  {orig_idx:8d}  {sid:<24}  {q}  | {' '.join(marks)}")


HEATMAP_MAX_LEN = 140
DEFAULT_CASE_STUDY_DIR = os.path.join("results", "stepkv_case_study")


def _gap_manifest_path(args, *, explicit: str = "") -> str:
    if explicit:
        return explicit
    return os.path.join(
        DEFAULT_CASE_STUDY_DIR,
        f"{args.dataset}_seed{int(args.seed)}_ns{int(args.num_samples)}_gap_manifest.json",
    )


def _serialize_gap_sample(
    gap_index: int,
    pos: int,
    orig_idx: int,
    sample: Dict[str, Any],
    row: Dict[str, Any],
) -> Dict[str, Any]:
    evals = row.get("evaluations", {}) or {}
    diff = row.get("discard_diff", {}) or {}
    counts = diff.get("counts", {}) or {}
    question = str(sample.get("question", "") or "").replace("\n", " ").strip()
    return {
        "gap_index": int(gap_index),
        "sample_pos": int(pos),
        "orig_idx": int(orig_idx),
        "sample_id": sample.get("id", ""),
        "question": question,
        "gold_answer": sample.get("answer", ""),
        "category": row.get("category", ""),
        "evaluations": evals,
        "discard_diff_counts": counts,
        "only_baseline_not_ours_count": int(counts.get("only_baseline_not_ours", 0)),
    }


def _build_gap_manifest(
    args,
    found: List[Tuple[int, int, Dict[str, Any], Dict[str, Any]]],
    *,
    scan_start: int,
    scan_end: int,
) -> Dict[str, Any]:
    samples = [
        _serialize_gap_sample(gap_idx, pos, orig_idx, sample, row)
        for gap_idx, (pos, orig_idx, sample, row) in enumerate(found)
    ]
    positions = [int(s["sample_pos"]) for s in samples]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        "manifest_version": 1,
        "created_at": stamp,
        "updated_at": stamp,
        "dataset": args.dataset,
        "seed": int(args.seed),
        "num_samples": int(args.num_samples),
        "max_steps": str(args.max_steps),
        "cache_ratio": float(args.cache_ratio),
        "baseline_fail_mode": str(args.baseline_fail_mode),
        "criteria": "baseline_fail_ours_success",
        "scan": {
            "scan_start": int(scan_start),
            "scan_end": int(scan_end),
            "scan_limit": int(args.scan_limit),
            "found_count": len(samples),
        },
        "aggregate": {
            "baseline_fail_ours_success_count": len(samples),
            "baseline_fail_ours_success_positions": positions,
        },
        "samples": samples,
    }


def _save_gap_manifest(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    _save_json(path, payload)


def _load_gap_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _manifest_matches_args(manifest: Dict[str, Any], args) -> bool:
    return (
        str(manifest.get("dataset", "")) == str(args.dataset)
        and int(manifest.get("seed", -1)) == int(args.seed)
        and int(manifest.get("num_samples", -1)) == int(args.num_samples)
    )


def _resolve_gap_manifest_path(args) -> str:
    if args.success_gap_json:
        return args.success_gap_json
    return _gap_manifest_path(args, explicit=args.gap_manifest)


def _load_success_gap_positions(path: str) -> List[int]:
    data = _load_gap_manifest(path)
    positions = data.get("aggregate", {}).get("baseline_fail_ours_success_positions")
    if positions:
        return [int(p) for p in positions]
    return [
        int(row["sample_pos"])
        for row in data.get("samples", []) or []
        if row.get("category") == "baseline_fail_ours_success"
    ]


def _try_load_saved_gap_manifest(args) -> Optional[Dict[str, Any]]:
    path = _resolve_gap_manifest_path(args)
    if not os.path.isfile(path):
        return None
    manifest = _load_gap_manifest(path)
    if not _manifest_matches_args(manifest, args):
        print(
            f"[WARN] Gap manifest exists but config mismatch, ignore: {path} "
            f"(dataset/seed/num_samples differ from current args)."
        )
        return None
    return manifest


def _save_found_gap_manifest(
    args,
    found: List[Tuple[int, int, Dict[str, Any], Dict[str, Any]]],
    *,
    scan_start: int,
    scan_end: int,
) -> str:
    path = _resolve_gap_manifest_path(args)
    payload = _build_gap_manifest(args, found, scan_start=scan_start, scan_end=scan_end)
    _save_gap_manifest(path, payload)
    return path


def _resolve_from_manifest(
    manifest: Dict[str, Any],
    gap_index: int,
    selected: List[Tuple[int, Dict[str, Any]]],
) -> Tuple[int, int, Dict[str, Any], Dict[str, Any]]:
    samples = list(manifest.get("samples", []) or [])
    if samples:
        if gap_index < 0 or gap_index >= len(samples):
            raise IndexError(f"gap_index={gap_index} out of range for {len(samples)} saved gap sample(s)")
        entry = samples[gap_index]
        pos = int(entry["sample_pos"])
    else:
        positions = _load_success_gap_positions_from_data(manifest)
        if gap_index < 0 or gap_index >= len(positions):
            raise IndexError(f"gap_index={gap_index} out of range for {len(positions)} saved gap sample(s)")
        pos = int(positions[gap_index])
        entry = {"sample_pos": pos, "category": "baseline_fail_ours_success"}

    if pos < 0 or pos >= len(selected):
        raise IndexError(f"sample_pos={pos} from manifest out of range (total {len(selected)})")
    orig_idx, sample = selected[pos]
    row = {
        **entry,
        "sample_pos": pos,
        "orig_idx": orig_idx,
        "discard_diff": {"counts": entry.get("discard_diff_counts", {})},
    }
    return pos, orig_idx, sample, row


def _load_success_gap_positions_from_data(data: Dict[str, Any]) -> List[int]:
    positions = data.get("aggregate", {}).get("baseline_fail_ours_success_positions")
    if positions:
        return [int(p) for p in positions]
    return [
        int(row["sample_pos"])
        for row in data.get("samples", []) or []
        if row.get("category") == "baseline_fail_ours_success"
    ]


def _scan_success_gap_samples(selected, retriever, args):
    scan_start = int(args.scan_start)
    scan_limit = int(args.scan_limit)
    end = min(len(selected), scan_start + scan_limit)
    found = []
    for pos in range(scan_start, end):
        orig_idx, sample = selected[pos]
        print(f"[INFO] success_gap scan pos={pos} id={sample.get('id', '')} ...")
        row = run_success_gap_one_sample(sample, retriever, args, pos, orig_idx)
        if row.get("category") == "baseline_fail_ours_success":
            found.append((pos, orig_idx, sample, row))
    return found, scan_start, end


def _resolve_success_gap_sample(selected, retriever, args):
    picked = _pick_case_study_sample(
        selected,
        sample_pos=args.sample_pos,
        orig_idx=args.orig_idx,
        sample_id=args.sample_id or "",
    )
    if picked is not None:
        pos, orig_idx, sample = picked
        print(
            f"[INFO] using sample pos={pos} orig_idx={orig_idx} id={sample.get('id', '')}"
        )
        row = run_success_gap_one_sample(sample, retriever, args, pos, orig_idx)
        if args.require_success_gap and row.get("category") != "baseline_fail_ours_success":
            raise RuntimeError(
                f"Selected sample pos={pos} is not baseline_fail_ours_success "
                f"(category={row.get('category')}). Use --no_require_success_gap to force run."
            )
        return pos, orig_idx, sample, row

    manifest = None
    if args.success_gap_json or args.use_saved_gap_manifest:
        manifest = _try_load_saved_gap_manifest(args)
    if manifest is not None:
        gap_index = int(args.gap_index)
        path = _resolve_gap_manifest_path(args)
        pos, orig_idx, sample, row = _resolve_from_manifest(manifest, gap_index, selected)
        print(
            f"[INFO] using saved gap manifest gap_index={gap_index} -> pos={pos} "
            f"orig_idx={orig_idx} id={sample.get('id', '')} ({path})"
        )
        return pos, orig_idx, sample, row

    found, scan_start, scan_end = _scan_success_gap_samples(selected, retriever, args)
    if not found:
        raise RuntimeError(
            "No baseline_fail_ours_success sample found in scan range. "
            "Increase --scan_limit or run --scan_gap_only first to build a manifest."
        )
    manifest_path = _save_found_gap_manifest(args, found, scan_start=scan_start, scan_end=scan_end)
    print(f"[INFO] saved gap manifest ({len(found)} samples): {manifest_path}")
    gap_index = int(args.gap_index)
    if gap_index < 0 or gap_index >= len(found):
        raise IndexError(f"gap_index={gap_index} but only {len(found)} gap sample(s) found")
    pos, orig_idx, sample, row = found[gap_index]
    print(
        f"[INFO] auto-selected gap_index={gap_index} -> pos={pos} orig_idx={orig_idx} "
        f"id={sample.get('id', '')}"
    )
    return pos, orig_idx, sample, row


def _run_all_methods_for_case_study(sample, retriever, args) -> Dict[str, Dict[str, Any]]:
    runs: Dict[str, Dict[str, Any]] = {}
    for method in SCORE_METHODS:
        print(f"[INFO] case study run -> {METHOD_LABELS.get(method, method)} ...")
        runs[method] = _run_one_sample(
            sample,
            retriever,
            pruning_mode=method,
            cache_ratio=float(args.cache_ratio),
            max_steps=wiki_base.MAX_STEPS,
            attention_viz=(method == OURS_METHOD),
        )
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="StepKV gap case study outputs (critical JSON + B/C PDFs).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Sample selection (pick one):\n"
            "  --sample_pos N   index in the selected subset (0 .. num_samples-1)\n"
            "  --orig_idx N     original dataset index after seed shuffle\n"
            "  --sample_id ID   dataset sample id string\n"
            "  --gap_index N    when auto-scanning, use the N-th gap sample (default 0)\n"
            "\n"
            "If none of sample_pos/orig_idx/sample_id is given, the script uses a saved gap\n"
            "manifest when available; otherwise it scans and writes one."
        ),
    )
    parser.add_argument("--dataset", choices=["hotpotqa", "2wiki", "musique"], default="hotpotqa")
    sample_group = parser.add_mutually_exclusive_group()
    sample_group.add_argument(
        "--sample_pos",
        type=int,
        default=None,
        help="Sample position in the selected subset (see --list_samples).",
    )
    sample_group.add_argument(
        "--orig_idx",
        type=int,
        default=None,
        help="Original dataset index within the selected subset.",
    )
    sample_group.add_argument(
        "--sample_id",
        type=str,
        default="",
        help="Dataset sample id (exact match in selected subset).",
    )
    parser.add_argument(
        "--list_samples",
        action="store_true",
        help="Print the selectable sample catalog and exit.",
    )
    parser.add_argument(
        "--list_gap_samples",
        action="store_true",
        help="Scan (or read saved manifest) and list gap samples, then exit.",
    )
    parser.add_argument(
        "--scan_gap_only",
        action="store_true",
        help="Only scan for gap samples, save manifest, and exit (no case study figures).",
    )
    parser.add_argument(
        "--gap_manifest",
        type=str,
        default="",
        help=(
            "Path to gap manifest JSON. Default: "
            "results/stepkv_case_study/{dataset}_seed{seed}_ns{num_samples}_gap_manifest.json"
        ),
    )
    parser.add_argument(
        "--use_saved_gap_manifest",
        action="store_true",
        default=True,
        help="When no explicit sample is given, reuse saved gap manifest if present (default: on).",
    )
    parser.add_argument(
        "--no_use_saved_gap_manifest",
        action="store_false",
        dest="use_saved_gap_manifest",
        help="Force rescan instead of reading saved gap manifest.",
    )
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--max_steps", type=str, default="7")
    parser.add_argument("--cache_ratio", type=float, default=0.5)
    parser.add_argument("--model_path", type=str, default="auto")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--wiki_index_dir", type=str, default=wiki_base.WIKI_INDEX_DIR)
    parser.add_argument("--bm25_top_k", type=int, default=wiki_base.BM25_TOP_K)
    parser.add_argument("--max_critical_tokens", type=int, default=12, help="Max tokens in score-decomp figure (B).")
    parser.add_argument("--require_success_gap", action="store_true", default=True)
    parser.add_argument("--no_require_success_gap", action="store_false", dest="require_success_gap")
    parser.add_argument("--scan_start", type=int, default=0)
    parser.add_argument("--scan_limit", type=int, default=30)
    parser.add_argument(
        "--gap_index",
        type=int,
        default=0,
        help="When auto-scanning or using --success_gap_json, pick the N-th gap sample.",
    )
    parser.add_argument("--baseline_fail_mode", choices=["all", "any"], default="all")
    parser.add_argument(
        "--success_gap_json",
        type=str,
        default="",
        help="Alias of --gap_manifest (legacy). Path to gap manifest JSON.",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="",
        help="Output path prefix. Default: results/stepkv_case_study/{dataset}_sample{N}_gap",
    )
    args = parser.parse_args()

    from models.model_paths import resolve_local_model_path

    args.model_path = resolve_local_model_path(args.model_path)

    selected, retriever = _prepare_dataset(args)

    if args.list_samples:
        _print_sample_catalog(selected, args)
        return

    if args.list_gap_samples or args.scan_gap_only:
        manifest = _try_load_saved_gap_manifest(args) if args.use_saved_gap_manifest else None
        if manifest is not None and not args.scan_gap_only:
            found = []
            for entry in manifest.get("samples", []) or []:
                pos = int(entry["sample_pos"])
                if pos < 0 or pos >= len(selected):
                    continue
                orig_idx, sample = selected[pos]
                found.append((pos, orig_idx, sample, entry))
            print(f"[INFO] loaded gap manifest: {_resolve_gap_manifest_path(args)}")
            _print_gap_sample_catalog(selected, found)
            if found:
                positions = [p for p, _, _, _ in found]
                print(f"[INFO] qualifying sample_pos: {positions}")
            return

        print(
            f"[INFO] scanning pos [{args.scan_start}, "
            f"{min(len(selected), args.scan_start + args.scan_limit)}) for gap samples ..."
        )
        found, scan_start, scan_end = _scan_success_gap_samples(selected, retriever, args)
        _print_gap_sample_catalog(selected, found)
        if found:
            manifest_path = _save_found_gap_manifest(
                args, found, scan_start=scan_start, scan_end=scan_end
            )
            positions = [p for p, _, _, _ in found]
            print(f"[INFO] saved gap manifest: {manifest_path}")
            print(f"[INFO] qualifying sample_pos: {positions}")
            print(
                "[INFO] next: python plot_stepkv_case_study.py "
                f"--dataset {args.dataset} --seed {args.seed} --gap_index 0"
            )
            print(
                "[INFO]   or: python plot_stepkv_case_study.py "
                f"--dataset {args.dataset} --seed {args.seed} --sample_pos {positions[0]}"
            )
        else:
            print("[WARN] No qualifying gap samples found in scan range.")
        return

    pos, orig_idx, sample, gap_row = _resolve_success_gap_sample(selected, retriever, args)

    method_runs = _run_all_methods_for_case_study(sample, retriever, args)
    gold = str(sample.get("answer", "") or "")
    gap_row = {
        **gap_row,
        "sample_pos": pos,
        "orig_idx": orig_idx,
        "sample_id": sample.get("id", ""),
        "question": sample.get("question", ""),
        "gold_answer": gold,
        "evaluations": {m: _evaluate_run(method_runs[m], gold) for m in SCORE_METHODS},
        "discard_diff": _summarize_discard_diff(
            {m: method_runs[m] for m in BASELINE_METHODS},
            method_runs[OURS_METHOD],
        ),
        "token_text_detail": _build_success_gap_token_text_report(method_runs),
    }

    tag = f"{args.dataset}_sample{pos}_gap"
    out_dir = DEFAULT_CASE_STUDY_DIR
    prefix = args.output_prefix or os.path.join(out_dir, tag)
    gap_manifest_path = _resolve_gap_manifest_path(args)

    outputs = save_case_study_outputs(
        gap_row,
        method_runs,
        prefix,
        max_score_tokens=int(args.max_critical_tokens),
    )

    detail_paths = _save_success_gap_token_detail_files(gap_row, out_dir, f"{args.dataset}_gap")
    status_jsonl = os.path.join(out_dir, f"{tag}_decode_status.jsonl")
    tokenizer = _get_analysis_tokenizer(wiki_base.MODEL_PATH)
    critical_indices = _all_critical_indices(method_runs, gap_row["discard_diff"])
    _export_decode_token_status(method_runs, tokenizer, critical_indices, status_jsonl)

    meta = {
        "dataset": args.dataset,
        "seed": int(args.seed),
        "num_samples": int(args.num_samples),
        "sample_pos": pos,
        "orig_idx": orig_idx,
        "sample_id": sample.get("id", ""),
        "gap_index": int(args.gap_index),
        "gap_manifest": gap_manifest_path if os.path.isfile(gap_manifest_path) else "",
        "category": gap_row.get("category"),
        "evaluations": gap_row.get("evaluations"),
        "discard_diff_counts": gap_row.get("discard_diff", {}).get("counts", {}),
        "outputs": outputs,
        "token_detail_files": detail_paths,
        "decode_status_jsonl": status_jsonl,
    }
    meta_path = f"{prefix}_meta.json"
    _save_json(meta_path, meta)

    print(f"[DONE] critical tokens JSON: {outputs['critical_tokens_json']} ({outputs['all_critical_count']} tokens)")
    print(f"[DONE] score decomp figure: {outputs['score_decomp_pdf']}")
    print(f"[DONE] global heatmap: {outputs['global_heatmap_pdf']}")
    print(f"[DONE] metadata: {meta_path}")
    print(f"[DONE] decode status jsonl: {status_jsonl}")
    if detail_paths:
        print(f"[DONE] token detail files: {detail_paths}")


if __name__ == "__main__":
    main()
