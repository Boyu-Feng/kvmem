#!/usr/bin/env python3
"""
StepKV gap case study figure (baseline fail / StepKV success):

  Header : Q, Gold, H2O/TOVA/StepKV EM
  (A)      Critical tokens — discard status per method + combined score bar
  (B)      StepKV token score + step score (stacked) for those tokens
  (C)      Mini causal score heatmap over the critical-token window

Example:
  python plot_stepkv_case_study.py --dataset hotpotqa
  python plot_stepkv_case_study.py --dataset hotpotqa --gap_index 1 --scan_limit 50
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
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
    _pick_sample,
    _prepare_dataset,
    _resolve_heatmap_plot_len,
    _run_one_sample,
    _save_json,
    _save_success_gap_token_detail_files,
    _save_tokens_jsonl,
    _summarize_discard_diff,
    _get_analysis_tokenizer,
    _token_text_record,
    extract_decode_token_scores,
    run_success_gap_one_sample,
)


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


def _minmax_normalize_matrix(mat: np.ndarray) -> np.ndarray:
    """Stretch matrix values to [0, 1] using min/max over the lower triangle."""
    out = np.array(mat, dtype=float)
    if out.size == 0:
        return out
    tril = np.tril(np.ones_like(out, dtype=bool), k=0)
    vals = out[tril & np.isfinite(out)]
    if vals.size == 0:
        return np.zeros_like(out)
    pos = vals[vals > 0]
    use = pos if pos.size > 0 else vals
    lo, hi = float(np.min(use)), float(np.max(use))
    if hi <= lo:
        norm = np.zeros_like(out)
    else:
        norm = (out - lo) / (hi - lo)
        norm = np.clip(norm, 0.0, 1.0)
    norm[~tril] = np.nan
    return norm


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


def _pick_critical_tokens(
    method_runs: Dict[str, Dict[str, Any]],
    discard_diff: Dict[str, Any],
    *,
    max_tokens: int = 12,
) -> List[int]:
    """Tokens discarded by any baseline but kept by StepKV."""
    ours_discarded = _discard_sets_from_run(method_runs[OURS_METHOD])
    candidates = [
        int(i)
        for i in (discard_diff.get("only_baseline_not_ours", []) or [])
        if int(i) not in ours_discarded
    ]
    if not candidates:
        return []

    ours_payload = method_runs[OURS_METHOD].get("debug_payload", {}) or {}
    score_info = extract_decode_token_scores(ours_payload)

    def _rank_key(idx: int) -> Tuple[float, float, int]:
        step_s = _score_at(score_info, "step_scores", idx)
        hh_s = _score_at(score_info, "hh_scores", idx)
        return (step_s, hh_s, -int(idx))

    ranked = sorted(set(candidates), key=_rank_key, reverse=True)
    return ranked[: int(max_tokens)]


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


def _plot_panel_a(ax, rows: List[CriticalTokenRow]) -> None:
    ax.set_title(
        "(A) Critical tokens: baseline discarded (✗) vs StepKV kept (✓)",
        loc="left",
        fontsize=13,
        pad=8,
    )
    if not rows:
        ax.text(0.5, 0.5, "No baseline-discarded / StepKV-kept tokens", ha="center", va="center")
        ax.axis("off")
        return

    n = len(rows)
    ax.set_xlim(-0.5, 4.6)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()

    combined_scores = [r.combined_score for r in rows]
    combined_norm = _minmax_norm(combined_scores)
    cmap = plt.get_cmap("Blues")

    for i, row in enumerate(rows):
        ax.text(-0.05, i, row.label, ha="right", va="center", fontsize=10, transform=ax.get_yaxis_transform())
        ax.text(0.35, i, f"st{row.owner_step}", ha="center", va="center", fontsize=9, color="#666666")

        for col, discarded in enumerate((row.h2o_discarded, row.tova_discarded, row.stepkv_discarded)):
            mark = "✗" if discarded else "✓"
            color = "#E45756" if discarded else "#54A24B"
            ax.text(1.0 + col * 0.55, i, mark, ha="center", va="center", fontsize=13, color=color, fontweight="bold")

        frac = float(combined_norm[i])
        bar_w = 2.2 * frac
        ax.barh(
            i,
            bar_w,
            left=2.85,
            height=0.55,
            color=cmap(0.35 + 0.6 * frac),
            edgecolor="white",
            linewidth=0.5,
        )

    ax.text(1.0, -0.9, "H2O", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(1.55, -0.9, "TOVA", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(2.1, -0.9, "StepKV", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(3.95, -0.9, "Combined score (min-max)", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.text(-0.05, -0.9, "Token", ha="right", va="bottom", fontsize=11, fontweight="bold", transform=ax.get_yaxis_transform())

    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_panel_b(ax, rows: List[CriticalTokenRow]) -> None:
    ax.set_title("(B) StepKV score decomposition (token + step)", loc="left", fontsize=13, pad=8)
    if not rows:
        ax.axis("off")
        return

    x = np.arange(len(rows))
    token_vals = _minmax_norm([r.token_score for r in rows])
    step_vals = _minmax_norm([r.step_score for r in rows])
    labels = [r.label for r in rows]

    ax.bar(x, token_vals, color="#6BAED6", edgecolor="white", linewidth=0.5, label="Token score (H2O, min-max)")
    ax.bar(x, step_vals, bottom=token_vals, color="#FC9272", edgecolor="white", linewidth=0.5, label="Step score (min-max)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Relative score (min-max within window)", fontsize=12)
    ax.set_ylim(0, max(1.05, float(np.max(token_vals + step_vals)) + 0.05))
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", frameon=False, fontsize=10)


def _plot_panel_c(
    ax,
    debug_payload: Dict[str, Any],
    rows: List[CriticalTokenRow],
) -> None:
    ax.set_title("(C) Mini causal score heatmap (critical window)", loc="left", fontsize=13, pad=8)
    if not rows:
        ax.axis("off")
        return

    indices = [r.decode_idx for r in rows]
    seg_start = max(0, min(indices) - 1)
    seg_end = max(indices) + 2

    score_info = extract_decode_token_scores(debug_payload)
    snap = _latest_token_score_snapshot(debug_payload) or {}
    prompt_len = int(score_info.get("prompt_token_count", debug_payload.get("prompt_token_count", 0)) or 0)
    plot_len = _resolve_heatmap_plot_len(score_info, snap, segment_end=seg_end)

    mat, source = _build_decode_score_square(snap, prompt_len, plot_len, score_info)
    mat = _apply_causal_display_mask(mat)

    seg_end = min(seg_end, mat.shape[0])
    sub = mat[seg_start:seg_end, seg_start:seg_end]
    sub_norm = _minmax_normalize_matrix(sub)
    im = ax.imshow(
        sub_norm,
        cmap="Reds",
        interpolation="nearest",
        aspect="auto",
        origin="upper",
        vmin=0.0,
        vmax=1.0,
    )

    for idx in indices:
        if seg_start <= idx < seg_end:
            local = idx - seg_start
            ax.axvline(local, color="#333333", linewidth=0.6, alpha=0.35)
            ax.axhline(local, color="#333333", linewidth=0.6, alpha=0.35)

    ax.set_xlabel("Key token index (local)", fontsize=11)
    ax.set_ylabel("Query token index (local)", fontsize=11)
    ax.text(0.02, 0.98, f"source={source}; color=min-max stretch", transform=ax.transAxes, va="top", fontsize=9, color="#555555")
    fig = ax.figure
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _export_decode_token_status(
    method_runs: Dict[str, Dict[str, Any]],
    tokenizer,
    critical_indices: Sequence[int],
    output_jsonl: str,
) -> None:
    """Flat per-decode-index kept/discarded status for all methods."""
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


def plot_gap_case_study_figure(
    gap_row: Dict[str, Any],
    method_runs: Dict[str, Dict[str, Any]],
    output_path: str,
    *,
    max_critical_tokens: int = 12,
    tokenizer=None,
) -> Tuple[str, List[CriticalTokenRow]]:
    discard_diff = _summarize_discard_diff(
        {m: method_runs[m] for m in BASELINE_METHODS},
        method_runs[OURS_METHOD],
    )
    critical_indices = _pick_critical_tokens(
        method_runs, discard_diff, max_tokens=max_critical_tokens
    )
    if not critical_indices:
        raise RuntimeError(
            "No critical tokens (only_baseline_not_ours) found. "
            "Need a baseline_fail_ours_success sample with discard differences."
        )

    if tokenizer is None:
        from transformers import AutoTokenizer
        from models.model_paths import resolve_local_model_path

        model_path = resolve_local_model_path(wiki_base.MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    rows = _build_critical_rows(method_runs, critical_indices, tokenizer)
    ours_payload = method_runs[OURS_METHOD].get("debug_payload", {}) or {}

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
        }
    )

    fig = plt.figure(figsize=(14, 11))
    gs = fig.add_gridspec(4, 1, height_ratios=[0.10, 0.34, 0.30, 0.30], hspace=0.42)

    ax_header = fig.add_subplot(gs[0])
    ax_header.axis("off")
    ax_header.text(0.5, 0.5, _format_header(gap_row), ha="center", va="center", fontsize=11, wrap=True)

    _plot_panel_a(fig.add_subplot(gs[1]), rows)
    _plot_panel_b(fig.add_subplot(gs[2]), rows)
    _plot_panel_c(fig.add_subplot(gs[3]), ours_payload, rows)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".pdf":
        fig.savefig(output_path, bbox_inches="tight")
    else:
        fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path, rows


def _load_success_gap_positions(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
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
    return found


def _resolve_success_gap_sample(selected, retriever, args):
    if args.success_gap_json:
        positions = _load_success_gap_positions(args.success_gap_json)
        if not positions:
            raise RuntimeError(f"No gap samples in {args.success_gap_json}")
        pos = positions[int(args.gap_index)]
        orig_idx, sample = selected[pos]
        row = run_success_gap_one_sample(sample, retriever, args, pos, orig_idx)
        return pos, orig_idx, sample, row

    if args.sample_pos is not None:
        pos, orig_idx, sample = _pick_sample(selected, args)
        row = run_success_gap_one_sample(sample, retriever, args, pos, orig_idx)
        if args.require_success_gap and row.get("category") != "baseline_fail_ours_success":
            raise RuntimeError(f"sample_pos={pos} is not a gap sample ({row.get('category')})")
        return pos, orig_idx, sample, row

    found = _scan_success_gap_samples(selected, retriever, args)
    if not found:
        raise RuntimeError("No baseline_fail_ours_success sample found; increase --scan_limit")
    gap_index = int(args.gap_index)
    if gap_index >= len(found):
        raise IndexError(f"gap_index={gap_index} but only {len(found)} gap sample(s) found")
    return found[gap_index]


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
    parser = argparse.ArgumentParser(description="Plot StepKV gap case study (A/B/C panels).")
    parser.add_argument("--dataset", choices=["hotpotqa", "2wiki", "musique"], default="hotpotqa")
    parser.add_argument("--sample_pos", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--max_steps", type=str, default="7")
    parser.add_argument("--cache_ratio", type=float, default=0.5)
    parser.add_argument("--model_path", type=str, default="auto")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--wiki_index_dir", type=str, default=wiki_base.WIKI_INDEX_DIR)
    parser.add_argument("--bm25_top_k", type=int, default=wiki_base.BM25_TOP_K)
    parser.add_argument("--max_critical_tokens", type=int, default=12)
    parser.add_argument("--require_success_gap", action="store_true", default=True)
    parser.add_argument("--no_require_success_gap", action="store_false", dest="require_success_gap")
    parser.add_argument("--scan_start", type=int, default=0)
    parser.add_argument("--scan_limit", type=int, default=30)
    parser.add_argument("--gap_index", type=int, default=0)
    parser.add_argument("--baseline_fail_mode", choices=["all", "any"], default="all")
    parser.add_argument("--success_gap_json", type=str, default="")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output path (.pdf recommended). Default: results/stepkv_case_study/...pdf",
    )
    args = parser.parse_args()

    from models.model_paths import resolve_local_model_path

    args.model_path = resolve_local_model_path(args.model_path)

    selected, retriever = _prepare_dataset(args)
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
    out_dir = os.path.join("results", "stepkv_case_study")
    out = args.output or os.path.join(out_dir, f"{tag}_case_study.pdf")

    path, rows = plot_gap_case_study_figure(
        gap_row,
        method_runs,
        out,
        max_critical_tokens=int(args.max_critical_tokens),
    )

    detail_paths = _save_success_gap_token_detail_files(gap_row, out_dir, f"{args.dataset}_gap")
    status_jsonl = os.path.join(out_dir, f"{tag}_decode_status.jsonl")
    tokenizer = _get_analysis_tokenizer(wiki_base.MODEL_PATH)
    critical_indices = [r.decode_idx for r in rows]
    _export_decode_token_status(method_runs, tokenizer, critical_indices, status_jsonl)

    meta = {
        "figure": path,
        "dataset": args.dataset,
        "sample_pos": pos,
        "sample_id": sample.get("id", ""),
        "category": gap_row.get("category"),
        "evaluations": gap_row.get("evaluations"),
        "critical_tokens": [
            {
                "decode_idx": r.decode_idx,
                "label": r.label,
                "owner_step": r.owner_step,
                "h2o_discarded": r.h2o_discarded,
                "tova_discarded": r.tova_discarded,
                "stepkv_discarded": r.stepkv_discarded,
                "token_score": r.token_score,
                "step_score": r.step_score,
                "combined_score": r.combined_score,
            }
            for r in rows
        ],
        "discard_diff_counts": gap_row.get("discard_diff", {}).get("counts", {}),
        "token_detail_files": detail_paths,
        "decode_status_jsonl": status_jsonl,
    }
    meta_path = os.path.splitext(path)[0] + ".json"
    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[DONE] case study figure: {path}")
    print(f"[DONE] metadata: {meta_path}")
    print(f"[DONE] decode status jsonl: {status_jsonl}")
    if detail_paths:
        print(f"[DONE] token detail files: {detail_paths}")
    print(f"[INFO] critical tokens: {len(rows)}")


if __name__ == "__main__":
    main()
