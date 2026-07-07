#!/usr/bin/env python3
"""
StepKV case study figure (3 columns):

  Top row    : causal attention / score heatmap for a short decode segment
  Bottom row : stacked bars per token (bottom = token/H2O score, top = step score)

Reuses score extraction + attention helpers from analyze_stepkv_discarded_tokens.py.

Examples:
  # Default: scan for baseline-fail / StepKV-success samples, then plot
  python plot_stepkv_case_study.py --dataset hotpotqa

  # Use the 2nd gap sample found in the scan window
  python plot_stepkv_case_study.py --dataset hotpotqa --gap_index 1 --scan_limit 50

  # Explicit decode-index windows (end exclusive)
  python plot_stepkv_case_study.py --dataset hotpotqa --gap_index 0 \\
    --segments 12:28 40:56 72:88

  # From a saved StepKV result JSON (needs debug_payload with token_score_snapshot)
  python plot_stepkv_case_study.py \\
    --result_json results/wiki_qwen25_7b_v2/run2/stepaware_r50/react_kv_step_aware_h2o_wiki_500_0502.json \\
    --sample_idx 0 --segments 10:26 30:46 55:71
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
    _apply_causal_display_mask,
    _build_decode_score_square,
    _enhance_attention_contrast,
    _latest_token_score_snapshot,
    _pick_sample,
    _prepare_dataset,
    _resolve_heatmap_plot_len,
    _run_one_sample,
    _scored_decode_len,
    _token_text_record,
    extract_decode_token_scores,
    run_success_gap_one_sample,
)


def _parse_segment(spec: str) -> Tuple[int, int]:
    m = re.match(r"^(\d+)\s*:\s*(\d+)$", spec.strip())
    if not m:
        raise ValueError(f"Invalid segment spec {spec!r}; use START:END (end exclusive)")
    start, end = int(m.group(1)), int(m.group(2))
    if end <= start:
        raise ValueError(f"Segment end must be > start: {spec}")
    return start, end


def _scored_decode_span(score_info: Dict[str, Any]) -> Tuple[int, int]:
    """Return [start, end) decode indices that have any token or step score."""
    hh = score_info.get("hh_scores") or score_info.get("scores") or []
    step = score_info.get("step_scores") or []
    n = max(len(hh), len(step))
    first, last = None, None
    for i in range(n):
        hv = hh[i] if i < len(hh) else None
        sv = step[i] if i < len(step) else None
        if hv is not None or sv is not None:
            if first is None:
                first = i
            last = i
    if first is None:
        return 0, 0
    return int(first), int(last) + 1


def _auto_three_segments(
    score_info: Dict[str, Any],
    window: int,
) -> List[Tuple[int, int]]:
    start, end = _scored_decode_span(score_info)
    if end <= start:
        raise RuntimeError("No scored decode tokens found for auto segment selection.")
    span = end - start
    win = min(int(window), max(4, span // 3))
    if span <= win:
        return [(start, end), (start, end), (start, end)]

    # Three evenly spaced windows inside the scored span.
    if span >= 3 * win:
        gap = (span - 3 * win) // 2
        s0 = start
        s1 = start + win + gap
        s2 = start + 2 * (win + gap)
        return [(s0, s0 + win), (s1, s1 + win), (s2, s2 + win)]

    # Short span: sliding windows with overlap.
    step = max(1, (span - win) // 2)
    return [
        (start, start + win),
        (start + step, start + step + win),
        (min(start + 2 * step, end - win), end),
    ]


def _segment_column_title(
    debug_payload: Dict[str, Any],
    seg_start: int,
    seg_end: int,
) -> str:
    prompt_len = int(debug_payload.get("prompt_token_count", 0) or 0)
    step_ids: List[int] = []
    for sid_str, rng in (debug_payload.get("step_token_ranges", {}) or {}).items():
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            continue
        dec_s = int(rng[0]) - prompt_len
        dec_e = int(rng[1]) - prompt_len
        if dec_e >= seg_start and dec_s < seg_end:
            step_ids.append(int(sid_str))
    if step_ids:
        if len(step_ids) == 1:
            return f"Step {step_ids[0]}"
        return f"Steps {min(step_ids)}–{max(step_ids)}"
    return f"Tokens {seg_start}–{seg_end - 1}"


def _decode_token_label(
    debug_payload: Dict[str, Any],
    decode_idx: int,
    tokenizer,
) -> str:
    prompt_len = int(debug_payload.get("prompt_token_count", 0) or 0)
    global_ids = list(debug_payload.get("global_token_ids", []) or [])
    gid = prompt_len + int(decode_idx)
    rec = _token_text_record(tokenizer, global_ids, prompt_len, gid)
    text = (rec.get("text_clean") or rec.get("text") or "").replace("\n", " ").replace("\r", " ").strip()
    text = text.replace("▁", " ").strip()
    if not text:
        tid = rec.get("token_id")
        return f"<{tid}>" if tid is not None else "?"
    if len(text) > 10:
        text = text[:9] + "…"
    return text


def _segment_arrays(
    score_info: Dict[str, Any],
    seg_start: int,
    seg_end: int,
) -> Tuple[np.ndarray, np.ndarray]:
    hh = score_info.get("hh_scores") or score_info.get("scores") or []
    step = score_info.get("step_scores") or []
    n = seg_end - seg_start
    token_vals = np.zeros(n, dtype=float)
    step_vals = np.zeros(n, dtype=float)
    for i in range(n):
        idx = seg_start + i
        if idx < len(hh) and hh[idx] is not None:
            token_vals[i] = max(0.0, float(hh[idx]))
        if idx < len(step) and step[idx] is not None:
            step_vals[i] = max(0.0, float(step[idx]))
    return token_vals, step_vals


def _build_full_score_matrix(
    snap: Dict[str, Any],
    prompt_len: int,
    score_info: Dict[str, Any],
    plot_len: int,
) -> Tuple[np.ndarray, str]:
    mat, source = _build_decode_score_square(
        snap,
        prompt_len=prompt_len,
        plot_len=plot_len,
        score_info=score_info,
    )
    mat = _enhance_attention_contrast(mat)
    mat = _apply_causal_display_mask(mat)
    return mat, source


def _plot_full_score_heatmap(
    debug_payload: Dict[str, Any],
    output_path: str,
) -> Tuple[str, str]:
    """Save the full Q×K score heatmap (same style as token_score_heatmap)."""
    score_info = extract_decode_token_scores(debug_payload)
    snap = _latest_token_score_snapshot(debug_payload) or {}
    prompt_len = int(score_info.get("prompt_token_count", debug_payload.get("prompt_token_count", 0)) or 0)
    plot_len = _resolve_heatmap_plot_len(score_info, snap)
    mat, source = _build_full_score_matrix(snap, prompt_len, score_info, plot_len)

    fig_side = float(min(12.0, max(4.0, plot_len * 0.08)))
    fig, ax = plt.subplots(figsize=(fig_side, fig_side))
    im = ax.imshow(
        mat,
        cmap="Reds",
        interpolation="nearest",
        aspect="equal",
        origin="upper",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xlabel("Key Token Index")
    ax.set_ylabel("Query Token Index")
    ax.set_xlim(-0.5, plot_len - 0.5)
    ax.set_ylim(-0.5, plot_len - 0.5)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(f"Full score matrix ({source})", fontsize=10)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path, source


def plot_stepkv_case_study(
    debug_payload: Dict[str, Any],
    segments: Sequence[Tuple[int, int]],
    output_path: str,
    *,
    tokenizer=None,
    column_titles: Optional[List[str]] = None,
    suptitle: Optional[str] = None,
) -> str:
    if len(segments) != 3:
        raise ValueError(f"Expected exactly 3 segments, got {len(segments)}")

    score_info = extract_decode_token_scores(debug_payload)
    snap = _latest_token_score_snapshot(debug_payload) or {}
    if not score_info.get("has_snapshot"):
        raise RuntimeError(
            "debug_payload has no token_score_snapshot. Re-run with attention_viz enabled "
            "(see plot_stepkv_case_study.py --rerun)."
        )

    prompt_len = int(score_info.get("prompt_token_count", debug_payload.get("prompt_token_count", 0)) or 0)
    plot_len = _resolve_heatmap_plot_len(
        score_info,
        snap,
        segment_end=max(end for _, end in segments),
    )

    full_mat, matrix_source = _build_full_score_matrix(snap, prompt_len, score_info, plot_len)
    print(f"[INFO] Score matrix source={matrix_source}, plot_len={plot_len}")

    if tokenizer is None:
        from transformers import AutoTokenizer

        from models.model_paths import resolve_local_model_path

        model_path = resolve_local_model_path(wiki_base.MODEL_PATH)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.6), gridspec_kw={"height_ratios": [1.35, 1.0], "hspace": 0.38, "wspace": 0.28})
    if suptitle:
        fig.suptitle(suptitle, fontsize=11, y=0.995)

    token_color = "#6BAED6"
    step_color = "#FC9272"

    for col, (seg_start, seg_end) in enumerate(segments):
        ax_hm = axes[0, col]
        ax_bar = axes[1, col]
        seg_len = seg_end - seg_start
        seg_end_clamped = min(seg_end, full_mat.shape[0])
        mat = full_mat[seg_start:seg_end_clamped, seg_start:seg_end_clamped]
        if mat.shape[0] < seg_len:
            pad = seg_len - mat.shape[0]
            mat = np.pad(mat, ((0, pad), (0, pad)), constant_values=np.nan)

        im = ax_hm.imshow(
            mat,
            cmap="Reds",
            interpolation="nearest",
            aspect="equal",
            origin="upper",
            vmin=0.0,
            vmax=1.0,
        )
        title = (column_titles[col] if column_titles and col < len(column_titles) else None)
        if not title:
            title = _segment_column_title(debug_payload, seg_start, seg_end)
        ax_hm.set_title(title, fontsize=12, pad=6)
        ax_hm.set_xlabel("Key Token Index", fontsize=10)
        ax_hm.set_ylabel("Query Token Index", fontsize=10)
        tick_step = max(1, seg_len // 6)
        local_ticks = list(range(0, seg_len, tick_step))
        ax_hm.set_xticks(local_ticks)
        ax_hm.set_xticklabels([str(seg_start + t) for t in local_ticks], fontsize=8)
        ax_hm.set_yticks(local_ticks)
        ax_hm.set_yticklabels([str(seg_start + t) for t in local_ticks], fontsize=8)
        if col == 2:
            fig.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)

        token_vals, step_vals = _segment_arrays(score_info, seg_start, seg_end)
        x = np.arange(seg_len)
        labels = [_decode_token_label(debug_payload, seg_start + i, tokenizer) for i in range(seg_len)]

        ax_bar.bar(x, token_vals, color=token_color, edgecolor="white", linewidth=0.4, label="Token score")
        ax_bar.bar(
            x,
            step_vals,
            bottom=token_vals,
            color=step_color,
            edgecolor="white",
            linewidth=0.4,
            label="Step score",
        )
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
        ax_bar.set_ylabel("Score" if col == 0 else "", fontsize=10)
        ax_bar.grid(axis="y", linestyle=":", alpha=0.35)
        ax_bar.set_axisbelow(True)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=token_color, label="Token score (H2O)"),
        plt.Rectangle((0, 0), 1, 1, color=step_color, label="Step score"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.01))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _load_debug_from_result_json(path: str, sample_idx: int) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data.get("debug_payload"), dict):
        return data["debug_payload"]
    results = data.get("results", [])
    if not results:
        raise ValueError(f"No debug_payload or results[] in {path}")
    if sample_idx < 0 or sample_idx >= len(results):
        raise IndexError(f"sample_idx={sample_idx} out of range (n={len(results)})")
    row = results[sample_idx]
    payload = row.get("debug_payload")
    if not isinstance(payload, dict):
        raise ValueError(f"results[{sample_idx}] has no debug_payload in {path}")
    return payload


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


def _format_gap_suptitle(gap_row: Dict[str, Any]) -> str:
    evals = gap_row.get("evaluations", {}) or {}
    parts = []
    for method in BASELINE_METHODS + [OURS_METHOD]:
        label = METHOD_LABELS.get(method, method)
        em = bool((evals.get(method) or {}).get("exact_match"))
        parts.append(f"{label}: {'✓' if em else '✗'}")
    q = str(gap_row.get("question", "") or "").replace("\n", " ")
    if len(q) > 90:
        q = q[:87] + "..."
    return f"sample_pos={gap_row.get('sample_pos')} | " + "  ".join(parts) + f"\n{q}"


def _scan_success_gap_samples(
    selected: List[Tuple[int, Dict[str, Any]]],
    retriever,
    args,
) -> List[Tuple[int, int, Dict[str, Any], Dict[str, Any]]]:
    """Scan samples; return list of (pos, orig_idx, sample, gap_row) for baseline-fail/StepKV-success."""
    scan_start = int(args.scan_start)
    scan_limit = int(args.scan_limit)
    end = min(len(selected), scan_start + scan_limit)
    if scan_start < 0 or scan_start >= len(selected):
        raise IndexError(f"scan_start out of range: {scan_start} (total {len(selected)})")

    found: List[Tuple[int, int, Dict[str, Any], Dict[str, Any]]] = []
    for pos in range(scan_start, end):
        orig_idx, sample = selected[pos]
        print(
            f"[INFO] success_gap scan pos={pos}/{end - 1} "
            f"id={sample.get('id', '')} ..."
        )
        row = run_success_gap_one_sample(sample, retriever, args, pos, orig_idx)
        cat = str(row.get("category", ""))
        evals = row.get("evaluations", {}) or {}
        print(
            f"[INFO]   category={cat}  "
            + "  ".join(
                f"{METHOD_LABELS.get(m, m)}={'EM' if (evals.get(m) or {}).get('exact_match') else 'miss'}"
                for m in BASELINE_METHODS + [OURS_METHOD]
            )
        )
        if cat == "baseline_fail_ours_success":
            found.append((pos, orig_idx, sample, row))
    return found


def _resolve_success_gap_sample(
    selected: List[Tuple[int, Dict[str, Any]]],
    retriever,
    args,
) -> Tuple[int, int, Dict[str, Any], Dict[str, Any]]:
    if args.success_gap_json:
        positions = _load_success_gap_positions(args.success_gap_json)
        if not positions:
            raise RuntimeError(f"No baseline_fail_ours_success samples in {args.success_gap_json}")
        gap_index = int(args.gap_index)
        if gap_index < 0 or gap_index >= len(positions):
            raise IndexError(
                f"gap_index={gap_index} out of range ({len(positions)} gap samples in JSON)"
            )
        pos = positions[gap_index]
        if pos < 0 or pos >= len(selected):
            raise IndexError(f"sample_pos={pos} from success_gap JSON out of range")
        orig_idx, sample = selected[pos]
        print(f"[INFO] Using gap sample_pos={pos} from {args.success_gap_json}")
        row = run_success_gap_one_sample(sample, retriever, args, pos, orig_idx)
        if row.get("category") != "baseline_fail_ours_success":
            print(
                f"[WARN] sample_pos={pos} is no longer baseline_fail_ours_success "
                f"(now {row.get('category')}); continuing anyway."
            )
        return pos, orig_idx, sample, row

    if args.sample_pos is not None and not args.require_success_gap:
        pos, orig_idx, sample = _pick_sample(selected, args)
        row = run_success_gap_one_sample(sample, retriever, args, pos, orig_idx)
        return pos, orig_idx, sample, row

    if not args.require_success_gap and args.sample_pos is None:
        args.sample_pos = 0
        pos, orig_idx, sample = _pick_sample(selected, args)
        row = run_success_gap_one_sample(sample, retriever, args, pos, orig_idx)
        return pos, orig_idx, sample, row

    if args.sample_pos is not None and args.require_success_gap:
        pos, orig_idx, sample = _pick_sample(selected, args)
        row = run_success_gap_one_sample(sample, retriever, args, pos, orig_idx)
        if row.get("category") != "baseline_fail_ours_success":
            raise RuntimeError(
                f"sample_pos={pos} is '{row.get('category')}', not baseline_fail_ours_success. "
                "Pick another --sample_pos, increase --scan_limit, or pass --no_require_success_gap."
            )
        return pos, orig_idx, sample, row

    found = _scan_success_gap_samples(selected, retriever, args)
    if not found:
        raise RuntimeError(
            f"No baseline_fail_ours_success sample in scan window "
            f"[{args.scan_start}, {args.scan_start + args.scan_limit}). "
            "Increase --scan_limit or relax --baseline_fail_mode any."
        )
    gap_index = int(args.gap_index)
    if gap_index < 0 or gap_index >= len(found):
        raise IndexError(
            f"gap_index={gap_index} out of range: found {len(found)} gap sample(s). "
            f"Positions: {[p for p, _, _, _ in found]}"
        )
    pos, orig_idx, sample, row = found[gap_index]
    print(f"[INFO] Selected gap sample {gap_index + 1}/{len(found)} at sample_pos={pos}")
    return pos, orig_idx, sample, row


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot StepKV 3-column case study (heatmap + stacked scores).")
    parser.add_argument("--dataset", choices=["hotpotqa", "2wiki", "musique"], default="hotpotqa")
    parser.add_argument(
        "--sample_pos",
        type=int,
        default=None,
        help="Use a specific sample position. With default --require_success_gap, it must be a gap sample.",
    )
    parser.add_argument("--sample_idx", type=int, default=0, help="Index inside result JSON when using --result_json.")
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--max_steps", type=str, default="7")
    parser.add_argument("--cache_ratio", type=float, default=0.5)
    parser.add_argument("--model_path", type=str, default="auto")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--wiki_index_dir", type=str, default=wiki_base.WIKI_INDEX_DIR)
    parser.add_argument("--bm25_top_k", type=int, default=wiki_base.BM25_TOP_K)
    parser.add_argument(
        "--segments",
        nargs=3,
        metavar="START:END",
        default=None,
        help="Three decode-index windows (end exclusive), e.g. 12:28 40:56 72:88",
    )
    parser.add_argument("--window_size", type=int, default=24, help="Auto-segment window length when --segments omitted.")
    parser.add_argument(
        "--require_success_gap",
        action="store_true",
        default=True,
        help="Only use samples where baselines fail and StepKV succeeds (default: on).",
    )
    parser.add_argument(
        "--no_require_success_gap",
        action="store_false",
        dest="require_success_gap",
        help="Allow any sample_pos without gap filtering.",
    )
    parser.add_argument(
        "--scan_start",
        type=int,
        default=0,
        help="First sample_pos when scanning for gap samples.",
    )
    parser.add_argument(
        "--scan_limit",
        type=int,
        default=30,
        help="How many samples to scan when searching for gap cases.",
    )
    parser.add_argument(
        "--gap_index",
        type=int,
        default=0,
        help="Which baseline_fail_ours_success sample to plot (0 = first found).",
    )
    parser.add_argument(
        "--baseline_fail_mode",
        choices=["all", "any"],
        default="all",
        help="'all' = H2O and TOVA both wrong; 'any' = at least one baseline wrong.",
    )
    parser.add_argument(
        "--success_gap_json",
        type=str,
        default="",
        help="Optional precomputed success_gap JSON/checkpoint; sample_pos list is read from it.",
    )
    parser.add_argument(
        "--result_json",
        type=str,
        default="",
        help="Optional saved react result JSON with debug_payload.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Re-run StepKV on one sample with attention_viz (default unless --result_json).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Output PNG path (default: results/stepkv_case_study/<dataset>_sample<N>_case_study.png)",
    )
    args = parser.parse_args()

    from models.model_paths import resolve_local_model_path

    args.model_path = resolve_local_model_path(args.model_path)

    debug_payload: Dict[str, Any]
    tag = f"{args.dataset}_gap"
    gap_row: Optional[Dict[str, Any]] = None
    suptitle: Optional[str] = None

    if args.result_json:
        debug_payload = _load_debug_from_result_json(args.result_json, args.sample_idx)
        tag = os.path.splitext(os.path.basename(args.result_json))[0]
    else:
        selected, retriever = _prepare_dataset(args)
        pos, orig_idx, sample, gap_row = _resolve_success_gap_sample(selected, retriever, args)
        suptitle = _format_gap_suptitle(gap_row)
        print(f"[INFO] Re-running StepKV with attention_viz for sample_pos={pos} ...")
        run = _run_one_sample(
            sample,
            retriever,
            pruning_mode=OURS_METHOD,
            cache_ratio=float(args.cache_ratio),
            max_steps=wiki_base.MAX_STEPS,
            attention_viz=True,
        )
        debug_payload = run.get("debug_payload", {}) or {}
        tag = f"{args.dataset}_sample{pos}_gap"

    score_info = extract_decode_token_scores(debug_payload)
    if args.segments:
        segments = [_parse_segment(s) for s in args.segments]
    else:
        segments = _auto_three_segments(score_info, window=int(args.window_size))
        print(f"[INFO] Auto segments: {segments}")

    out = args.output or os.path.join("results", "stepkv_case_study", f"{tag}_case_study.png")
    path = plot_stepkv_case_study(debug_payload, segments, out, suptitle=suptitle)
    full_heatmap_path = os.path.splitext(path)[0] + "_full_heatmap.png"
    full_path, matrix_source = _plot_full_score_heatmap(debug_payload, full_heatmap_path)
    meta = {
        "segments": [{"start": s, "end": e} for s, e in segments],
        "figure": path,
        "full_heatmap": full_path,
        "matrix_source": matrix_source,
        "plot_len": _resolve_heatmap_plot_len(
            extract_decode_token_scores(debug_payload),
            _latest_token_score_snapshot(debug_payload) or {},
        ),
        "dataset": args.dataset,
        "category": (gap_row or {}).get("category", "unknown"),
        "sample_pos": (gap_row or {}).get("sample_pos"),
        "sample_id": (gap_row or {}).get("sample_id"),
        "evaluations": (gap_row or {}).get("evaluations"),
        "gold_answer": (gap_row or {}).get("gold_answer"),
        "question": (gap_row or {}).get("question"),
    }
    meta_path = os.path.splitext(path)[0] + ".json"
    os.makedirs(os.path.dirname(meta_path) or ".", exist_ok=True)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[DONE] case study figure: {path}")
    print(f"[DONE] full score heatmap: {full_path}")
    print(f"[DONE] metadata: {meta_path}")


if __name__ == "__main__":
    main()
