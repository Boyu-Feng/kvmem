import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import run_all_wiki_experiments_v2 as base
from analyze_stepkv_discarded_tokens import _build_kv_config
from models.QwenLLMWithKVCache import QwenLLMWithKVCache
from models.model_paths import resolve_local_model_path
from retrievers.WikiBM25Retriever import WikiBM25Retriever
from token_tracker import TokenTracker

METHOD_SERIES: List[Tuple[str, str]] = [
    ("h2o", "H2O"),
    ("tova", "TOVA"),
    ("step_aware_h2o", "StepKV"),
]


def _build_full_step_ranges(
    step_token_ranges: Dict[str, Any],
    obs_step_ranges: Dict[str, Any],
) -> List[Tuple[int, int, int]]:
    """Merge Think+Action with the following Observation into one ReAct step span."""
    full_ranges: List[Tuple[int, int, int]] = []
    step_ids = sorted(int(k) for k in step_token_ranges.keys())
    for sid in step_ids:
        rng = step_token_ranges.get(str(sid)) or step_token_ranges.get(sid)
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            continue
        s, e = int(rng[0]), int(rng[1])
        if e < s:
            continue
        obs_rng = obs_step_ranges.get(str(sid + 1)) or obs_step_ranges.get(sid + 1)
        if isinstance(obs_rng, (list, tuple)) and len(obs_rng) == 2:
            e = max(e, int(obs_rng[1]))
        full_ranges.append((sid, s, e))
    return full_ranges


def _resolve_owner_step(
    global_id: int,
    step_ranges: List[Tuple[int, int, int]],
    obs_step_ranges: Dict[str, Any],
) -> int:
    """Map a global token id to the ReAct step that produced it."""
    gid = int(global_id)
    for sid, s, e in step_ranges:
        if s <= gid <= e:
            return sid
    for obs_step_str, rng in sorted(obs_step_ranges.items(), key=lambda kv: int(kv[0])):
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            continue
        obs_at_loop_step = int(obs_step_str)
        s, e = int(rng[0]), int(rng[1])
        if s <= gid <= e:
            # Observation {k} is prefilled at loop step k+1.
            return max(1, obs_at_loop_step - 1)
    return -1


def _extract_plot_data(debug_payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt_token_count = int(debug_payload.get("prompt_token_count", 0))
    pruning_history = debug_payload.get("pruning_history", []) or []
    step_token_ranges = debug_payload.get("step_token_ranges", {}) or {}
    obs_step_ranges = debug_payload.get("obs_step_ranges", {}) or {}
    token_tracker = debug_payload.get("token_tracker", {}) or {}
    step_pruning_events = token_tracker.get("step_pruning_events", {}) or {}

    event_rows = []
    scatter_points = []
    event_id = 0
    owner_rows: Dict[str, Dict[str, Any]] = {}

    step_ranges = _build_full_step_ranges(step_token_ranges, obs_step_ranges)

    def _owner_step(global_id: int) -> int:
        return _resolve_owner_step(global_id, step_ranges, obs_step_ranges)

    for ev in pruning_history:
        if not isinstance(ev, dict):
            continue
        dropped = ev.get("evicted_abs_indices", []) or []
        if not dropped:
            continue
        dropped_no_prefill = [int(x) for x in dropped if int(x) >= prompt_token_count]
        if not dropped_no_prefill:
            continue

        event_id += 1
        react_step = ev.get("react_step")
        react_step = int(react_step) if react_step is not None else -1
        shifted = [int(x - prompt_token_count) for x in dropped_no_prefill]

        row = {
            "event_id": int(event_id),
            "react_step": int(react_step),
            "tokens_evicted": int(len(shifted)),
            "evicted_abs_indices_no_prefill": shifted,
            "cache_before": int(ev.get("cache_before", 0)),
            "new_total_len": int(ev.get("new_total_len", 0)),
            "single_token_mode": bool(ev.get("single_token_mode", False)),
        }
        event_rows.append(row)

        for x in shifted:
            scatter_points.append(
                {
                    "event_id": int(event_id),
                    "react_step": int(react_step),
                    "x": int(x),
                }
            )

    final_points = []
    final_event_rows = []
    final_event_id = 0
    init_dropped, react_pruning_events = _split_pruning_events(step_pruning_events)
    if init_dropped:
        init_decode = sorted(set(int(x) for x in init_dropped if int(x) >= prompt_token_count))
        if init_decode:
            final_event_id += 1
            owner_counts: Dict[int, int] = {}
            shifted_ids = []
            for gid in init_decode:
                shifted_x = int(gid - prompt_token_count)
                shifted_ids.append(shifted_x)
                owner = int(_owner_step(gid))
                owner_counts[owner] = int(owner_counts.get(owner, 0) + 1)
                final_points.append(
                    {
                        "event_id": int(final_event_id),
                        "prune_step": 0,
                        "owner_step": int(owner),
                        "x": shifted_x,
                        "global_id": int(gid),
                    }
                )
            final_event_rows.append(
                {
                    "event_id": int(final_event_id),
                    "prune_step": 0,
                    "tokens_evicted": int(len(shifted_ids)),
                    "evicted_abs_indices_no_prefill": shifted_ids,
                    "owner_step_counts": {str(int(k)): int(v) for k, v in sorted(owner_counts.items())},
                }
            )

    for prune_step, dropped_ids in sorted(react_pruning_events.items(), key=lambda kv: int(kv[0])):
        dropped_ids = sorted(set(int(x) for x in (dropped_ids or [])))
        dropped_ids = [gid for gid in dropped_ids if gid >= prompt_token_count]
        if not dropped_ids:
            continue
        final_event_id += 1
        owner_counts = {}
        shifted_ids = []
        for gid in dropped_ids:
            shifted_x = int(gid - prompt_token_count)
            shifted_ids.append(shifted_x)
            owner = int(_owner_step(gid))
            owner_counts[owner] = int(owner_counts.get(owner, 0) + 1)
            key = str(owner)
            if key not in owner_rows:
                owner_rows[key] = {"owner_step": owner, "dropped_count": 0}
            owner_rows[key]["dropped_count"] = int(owner_rows[key]["dropped_count"] + 1)
            final_points.append(
                {
                    "event_id": int(final_event_id),
                    "prune_step": int(prune_step),
                    "owner_step": int(owner),
                    "x": shifted_x,
                    "global_id": int(gid),
                }
            )
        final_event_rows.append(
            {
                "event_id": int(final_event_id),
                "prune_step": int(prune_step),
                "tokens_evicted": int(len(shifted_ids)),
                "evicted_abs_indices_no_prefill": shifted_ids,
                "owner_step_counts": {str(int(k)): int(v) for k, v in sorted(owner_counts.items())},
            }
        )

    step_boundaries = []
    for sid, _start, end_abs in step_ranges:
        end_shifted = end_abs - prompt_token_count
        if end_shifted >= 0:
            step_boundaries.append({"step": sid, "x": float(end_shifted) + 0.5})

    kept_points = _build_kept_points(
        prompt_token_count=prompt_token_count,
        step_ranges=step_ranges,
        step_pruning_events=step_pruning_events,
        token_tracker=token_tracker,
        owner_step_fn=_owner_step,
    )

    return {
        "prompt_token_count": prompt_token_count,
        "events": event_rows,
        "points": scatter_points,
        "final_events": final_event_rows,
        "final_points": final_points,
        "kept_points": kept_points,
        "dropped_by_owner_step": [owner_rows[k] for k in sorted(owner_rows.keys(), key=lambda x: int(x))],
        "step_boundaries": step_boundaries,
    }


def _split_pruning_events(step_pruning_events: Dict[str, Any]) -> Tuple[List[int], Dict[int, List[int]]]:
    """Split pre-step (init) drops from numbered ReAct-step drops."""
    init_dropped: List[int] = []
    react_events: Dict[int, List[int]] = {}
    for key, dropped_ids in (step_pruning_events or {}).items():
        ids = [int(x) for x in (dropped_ids or [])]
        if key is None or str(key).lower() in {"init", "none", "null"}:
            init_dropped.extend(ids)
            continue
        try:
            step_id = int(key)
        except (TypeError, ValueError):
            continue
        react_events.setdefault(step_id, []).extend(ids)
    return init_dropped, react_events


def _prune_step_sort_key(item: Tuple[Any, Any]) -> Tuple[int, int]:
    key = item[0]
    if key is None or str(key).lower() in {"init", "none", "null"}:
        return (0, -1)
    try:
        return (1, int(key))
    except (TypeError, ValueError):
        return (2, 0)


def _decode_end_global_id_through_step(
    step_ranges: List[Tuple[int, int, int]],
    through_step: int,
    prompt_token_count: int,
    next_global_id: int,
) -> int:
    """Last global token id that exists after finishing the given ReAct step."""
    end = int(prompt_token_count) - 1
    for sid, _s, e in step_ranges:
        if int(sid) <= int(through_step):
            end = max(end, int(e))
    return min(end, int(next_global_id) - 1)


def _build_kept_points(
    prompt_token_count: int,
    step_ranges: List[Tuple[int, int, int]],
    step_pruning_events: Dict[str, Any],
    token_tracker: Dict[str, Any],
    owner_step_fn,
) -> List[Dict[str, Any]]:
    """Tokens still in cache after each ReAct step (inverse of cumulative drops)."""
    next_global_id = int(token_tracker.get("next_global_id", prompt_token_count))
    if next_global_id <= prompt_token_count:
        return []

    init_dropped, react_pruning_events = _split_pruning_events(step_pruning_events)

    prune_steps = sorted(react_pruning_events.keys())
    max_range_step = max((sid for sid, _, _ in step_ranges), default=1)
    max_step = max(max(prune_steps) if prune_steps else 1, max_range_step)

    cumulative_dropped: set[int] = set(
        int(x) for x in init_dropped if int(x) >= prompt_token_count
    )
    kept_points: List[Dict[str, Any]] = []

    for snapshot_step in range(1, max_step + 1):
        dropped_now = react_pruning_events.get(snapshot_step, [])
        cumulative_dropped.update(int(x) for x in dropped_now if int(x) >= prompt_token_count)

        decode_end = _decode_end_global_id_through_step(
            step_ranges,
            snapshot_step,
            prompt_token_count,
            next_global_id,
        )
        if decode_end < prompt_token_count:
            continue

        for gid in range(int(prompt_token_count), decode_end + 1):
            if gid in cumulative_dropped:
                continue
            kept_points.append(
                {
                    "snapshot_step": int(snapshot_step),
                    "owner_step": int(owner_step_fn(gid)),
                    "x": int(gid - prompt_token_count),
                    "global_id": int(gid),
                }
            )

    return kept_points


def _step_label(step: int) -> str:
    return f"token from step {step}"


def _display_step_from_point(point: Dict[str, Any]) -> int:
    """Map orphan tokens (owner_step=-1) to a valid ReAct step id starting at 1."""
    owner = int(point.get("owner_step", -1))
    if owner >= 1:
        return owner
    prune = int(point.get("prune_step", 0) or 0)
    return max(1, prune)


def _point_display_step(point: Dict[str, Any], step_key: str) -> int:
    if step_key == "owner_step":
        return _display_step_from_point(point)
    return max(1, int(point.get(step_key, 1)))


def _collect_plot_points(plot_data: Dict[str, Any], mode: str) -> Tuple[List[Dict[str, Any]], str]:
    if mode == "kept":
        return list(plot_data.get("kept_points", []) or []), "owner_step"
    points = plot_data.get("final_points", []) or plot_data.get("points", []) or []
    step_key = "owner_step" if points and "owner_step" in points[0] else "react_step"
    return points, step_key


def _point_y_value(point: Dict[str, Any], mode: str, step_key: str) -> int:
    if mode == "kept":
        return max(1, int(point.get("snapshot_step", 1) or 1))
    return max(
        1,
        int(point.get("prune_step", point.get("react_step", point.get("event_id", 1))) or 1),
    )


def _max_react_step_in_plot_data(plot_data: Dict[str, Any], mode: str) -> int:
    max_step = 1
    for bd in plot_data.get("step_boundaries", []) or []:
        max_step = max(max_step, int(bd.get("step", 1)))
    points, step_key = _collect_plot_points(plot_data, mode)
    for p in points:
        max_step = max(max_step, _point_y_value(p, mode, step_key))
    return max_step


def _plot_three_methods(
    method_plot_data: List[Tuple[str, Dict[str, Any]]],
    output_pdf: str,
    mode: str = "dropped",
    max_react_steps: Optional[int] = None,
) -> None:
    """Three rows: H2O / TOVA / StepKV token scatter (per-method x-axis)."""
    if mode not in ("dropped", "kept"):
        raise ValueError(f"Unknown plot mode: {mode}")

    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 15,
            "axes.titlesize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )

    all_owner_steps: set[int] = set()
    for _, plot_data in method_plot_data:
        points, step_key = _collect_plot_points(plot_data, mode)
        for p in points:
            all_owner_steps.add(_point_display_step(p, step_key))
    owner_steps = sorted(s for s in all_owner_steps if s >= 1)
    cmap = plt.get_cmap("tab10")
    owner_to_color = {s: cmap(i % 10) for i, s in enumerate(owner_steps)}
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=owner_to_color[s],
            markersize=8,
            linestyle="None",
        )
        for s in owner_steps
    ]
    legend_labels = [_step_label(s) for s in owner_steps]

    global_y_max = max(
        (_max_react_step_in_plot_data(plot_data, mode) for _, plot_data in method_plot_data),
        default=1,
    )
    if max_react_steps is not None and int(max_react_steps) > 0:
        global_y_max = min(global_y_max, int(max_react_steps))

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=False)
    fig.subplots_adjust(hspace=0.22, bottom=0.13, top=0.98)

    y_label = "Retained after ReAct step" if mode == "kept" else "Evicted at ReAct step"
    empty_msg = (
        "No retained-token points under current config"
        if mode == "kept"
        else "No dropped-token points under current config"
    )
    legend_title = "Retained token origin" if mode == "kept" else "Dropped token origin"
    footer = (
        "Y-axis = cache snapshot after each ReAct step finishes (tokens not yet evicted)"
        if mode == "kept"
        else "Y-axis = when token was evicted; red ring = evicted earlier than token origin (cross-step drop)"
    )

    for ax, (method_label, plot_data) in zip(axes, method_plot_data):
        points, step_key = _collect_plot_points(plot_data, mode)
        boundaries = plot_data.get("step_boundaries", []) or []
        x_vals: List[int] = []

        if points:
            for owner_step in owner_steps:
                matched = [p for p in points if _point_display_step(p, step_key) == owner_step]
                xs = [p["x"] for p in matched]
                ys = [_point_y_value(p, mode, step_key) for p in matched]
                if not xs:
                    continue
                x_vals.extend(int(x) for x in xs)
                if mode == "kept":
                    ax.scatter(
                        xs,
                        ys,
                        s=12,
                        alpha=0.75,
                        c=[owner_to_color[owner_step]],
                        edgecolors="none",
                    )
                else:
                    edgecolors = []
                    for p in matched:
                        prune_step = max(1, int(p.get("prune_step", p.get("react_step", 1)) or 1))
                        owner = _point_display_step(p, step_key)
                        edgecolors.append("#d62728" if prune_step < owner else "none")
                    ax.scatter(
                        xs,
                        ys,
                        s=16,
                        alpha=0.85,
                        c=[owner_to_color[owner_step]],
                        edgecolors=edgecolors,
                        linewidths=[1.2 if ec != "none" else 0.0 for ec in edgecolors],
                    )
        else:
            ax.text(
                0.5,
                0.5,
                empty_msg,
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=13,
                color="gray",
            )

        for bd in boundaries:
            ax.axvline(float(bd["x"]), linestyle="--", linewidth=1.0, color="gray", alpha=0.7)

        ax.set_ylabel(y_label, fontsize=15)
        ax.set_title(method_label, loc="left", fontsize=15, pad=6)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
        ax.set_ylim(0.5, global_y_max + 0.5)
        ax.set_yticks(list(range(1, global_y_max + 1)))
        if x_vals:
            ax.set_xlim(-0.5, max(x_vals) + 0.5)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Key Position Index (No Prefill)", fontsize=15)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.08),
            ncol=min(4, max(1, len(legend_labels))),
            frameon=False,
            title=legend_title,
        )
        fig.text(
            0.5,
            0.115,
            footer,
            ha="center",
            va="center",
            fontsize=11,
            color="#555555",
        )

    os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def _plot_dropped_three_methods(
    method_plot_data: List[Tuple[str, Dict[str, Any]]],
    output_pdf: str,
    max_react_steps: Optional[int] = None,
) -> None:
    _plot_three_methods(method_plot_data, output_pdf, mode="dropped", max_react_steps=max_react_steps)


def _plot_kept_three_methods(
    method_plot_data: List[Tuple[str, Dict[str, Any]]],
    output_pdf: str,
    max_react_steps: Optional[int] = None,
) -> None:
    _plot_three_methods(method_plot_data, output_pdf, mode="kept", max_react_steps=max_react_steps)


def _run_one_method(
    sample: Dict[str, Any],
    retriever,
    pruning_mode: str,
    cache_ratio: float,
    max_steps: int,
) -> Tuple[str, List[Any], List[Any], Dict[str, Any]]:
    kv_config = _build_kv_config(pruning_mode, cache_ratio=cache_ratio)
    token_tracker = TokenTracker()
    llm = QwenLLMWithKVCache(base.MODEL_PATH, kv_config, token_tracker=token_tracker)
    try:
        pred_answer, trajectory_log, step_timings, debug_payload = base._run_react_kv_episode(
            sample["question"],
            llm,
            retriever,
            pruning_mode=pruning_mode,
            max_steps=max_steps,
            return_debug=True,
        )
    finally:
        del llm
    return pred_answer, trajectory_log, step_timings, debug_payload if isinstance(debug_payload, dict) else {}


def _has_dropped_points(plot_data: Dict[str, Any]) -> bool:
    return bool(plot_data.get("final_points") or plot_data.get("points"))



def main():
    parser = argparse.ArgumentParser(
        description="Plot dropped/kept decode tokens for H2O / TOVA / StepKV on HotpotQA."
    )
    parser.add_argument("--sample_pos", type=int, default=0, help="Position in shuffled selected samples.")
    parser.add_argument(
        "--auto_find_nonempty",
        action="store_true",
        help="Auto scan next samples until at least one method has dropped points.",
    )
    parser.add_argument(
        "--max_auto_tries",
        type=int,
        default=20,
        help="Max samples to try when --auto_find_nonempty is set.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=int(base.MAX_STEPS),
        help=f"Max ReAct steps per episode (default: {base.MAX_STEPS}, same as main experiments).",
    )
    parser.add_argument("--num_samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--bm25_top_k", type=int, default=5)
    parser.add_argument("--wiki_index_dir", type=str, default=base.WIKI_INDEX_DIR)
    parser.add_argument(
        "--model_path",
        type=str,
        default="auto",
        help="Local model dir, or 'auto' to use KVMEM_MODEL_PATH / hf_cache/models/Qwen2.5-7B-Instruct.",
    )
    parser.add_argument("--output_dir", type=str, default="results/h2o_drop_analysis")
    parser.add_argument("--cache_ratio", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    base.NUM_SAMPLES = int(args.num_samples)
    base.RANDOM_SEED = int(args.seed)
    base.MAX_STEPS = int(args.max_steps)
    base.BM25_TOP_K = int(args.bm25_top_k)
    base.WIKI_INDEX_DIR = args.wiki_index_dir
    args.model_path = resolve_local_model_path(args.model_path)
    base.MODEL_PATH = args.model_path
    print(f"[INFO] Analysis model (local): {base.MODEL_PATH}")
    print(f"[INFO] Max ReAct steps: {base.format_max_steps(base.MAX_STEPS)}")

    if not os.path.exists(args.wiki_index_dir):
        raise FileNotFoundError(f"Wiki index not found: {args.wiki_index_dir}")

    print("[INFO] Loading HotpotQA and selecting samples...")
    val_data = base.load_hotpotqa_data()
    selected_samples = base.select_samples(val_data)
    if args.sample_pos < 0 or args.sample_pos >= len(selected_samples):
        raise IndexError(f"--sample_pos out of range: {args.sample_pos} (total {len(selected_samples)})")

    retriever = WikiBM25Retriever(index_dir=args.wiki_index_dir, load_corpus=True)

    start_pos = int(args.sample_pos)
    tries = int(args.max_auto_tries) if args.auto_find_nonempty else 1
    chosen_pos = None
    chosen_orig_idx = None
    chosen_sample = None
    method_results: Dict[str, Dict[str, Any]] = {}
    method_plot_data: List[Tuple[str, Dict[str, Any]]] = []

    for off in range(max(1, tries)):
        pos = start_pos + off
        if pos >= len(selected_samples):
            break
        orig_idx, sample = selected_samples[pos]
        print(f"[INFO] Trying sample_pos={pos}, orig_idx={orig_idx}, id={sample['id']}")

        method_results = {}
        method_plot_data = []
        any_points = False
        for pruning_mode, display_name in METHOD_SERIES:
            print(f"[INFO]   Running {display_name} ({pruning_mode}) ...")
            pred, traj, timings, debug_payload = _run_one_method(
                sample,
                retriever,
                pruning_mode=pruning_mode,
                cache_ratio=float(args.cache_ratio),
                max_steps=int(args.max_steps),
            )
            plot_data = _extract_plot_data(debug_payload)
            method_results[pruning_mode] = {
                "display_name": display_name,
                "predicted_answer": pred,
                "trajectory": traj,
                "step_timings": timings,
                "plot_data": plot_data,
            }
            method_plot_data.append((display_name, plot_data))
            if _has_dropped_points(plot_data):
                any_points = True

        chosen_pos = pos
        chosen_orig_idx = orig_idx
        chosen_sample = sample
        if any_points or not args.auto_find_nonempty:
            if any_points:
                print(f"[INFO] Found dropped-token points at sample_pos={pos}")
            break

    if chosen_sample is None:
        raise RuntimeError("No valid sample could be executed.")

    if not any(_has_dropped_points(pd) for _, pd in method_plot_data):
        print(
            "[WARN] No dropped-token points found in selected sample range. "
            "Output files will still be generated."
        )

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"hotpot_drop_tokens_sample{chosen_pos}_{stamp}"
    json_path = os.path.join(args.output_dir, f"{prefix}.json")
    dropped_pdf_path = os.path.join(args.output_dir, f"{prefix}_dropped.pdf")
    kept_pdf_path = os.path.join(args.output_dir, f"{prefix}_kept.pdf")
    dropped_points_jsonl_path = os.path.join(args.output_dir, f"{prefix}_dropped_points.jsonl")
    kept_points_jsonl_path = os.path.join(args.output_dir, f"{prefix}_kept_points.jsonl")
    plot_error_path = os.path.join(args.output_dir, f"{prefix}_plot_error.txt")

    output_blob = {
        "meta": {
            "created_at": stamp,
            "sample_pos": int(chosen_pos),
            "requested_sample_pos": int(args.sample_pos),
            "orig_idx": int(chosen_orig_idx),
            "sample_id": chosen_sample["id"],
            "question": chosen_sample["question"],
            "gold_answer": chosen_sample["answer"],
            "max_steps": int(args.max_steps),
            "cache_ratio": float(args.cache_ratio),
            "methods": [m for m, _ in METHOD_SERIES],
            "auto_find_nonempty": bool(args.auto_find_nonempty),
            "max_auto_tries": int(args.max_auto_tries),
        },
        "method_results": method_results,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_blob, f, ensure_ascii=False, indent=2)

    with open(dropped_points_jsonl_path, "w", encoding="utf-8") as f:
        for display_name, plot_data in method_plot_data:
            for p in plot_data.get("final_points", []) or plot_data.get("points", []) or []:
                f.write(json.dumps({"method": display_name, **p}, ensure_ascii=False) + "\n")

    with open(kept_points_jsonl_path, "w", encoding="utf-8") as f:
        for display_name, plot_data in method_plot_data:
            for p in plot_data.get("kept_points", []) or []:
                f.write(json.dumps({"method": display_name, **p}, ensure_ascii=False) + "\n")

    try:
        _plot_dropped_three_methods(method_plot_data, dropped_pdf_path, max_react_steps=int(args.max_steps))
        _plot_kept_three_methods(method_plot_data, kept_pdf_path, max_react_steps=int(args.max_steps))
    except Exception as e:
        with open(plot_error_path, "w", encoding="utf-8") as f:
            f.write(str(e))
        print(f"[WARN] Plot generation failed, see: {plot_error_path}")
        raise

    print(f"[DONE] Data saved: {json_path}")
    print(f"[DONE] Dropped points saved: {dropped_points_jsonl_path}")
    print(f"[DONE] Kept points saved: {kept_points_jsonl_path}")
    print(f"[DONE] Dropped figure saved: {dropped_pdf_path}")
    print(f"[DONE] Kept figure saved: {kept_pdf_path}")


if __name__ == "__main__":
    main()
