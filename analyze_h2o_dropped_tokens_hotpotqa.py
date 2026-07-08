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


def _extract_plot_data(debug_payload: Dict[str, Any]) -> Dict[str, Any]:
    prompt_token_count = int(debug_payload.get("prompt_token_count", 0))
    pruning_history = debug_payload.get("pruning_history", []) or []
    step_token_ranges = debug_payload.get("step_token_ranges", {}) or {}
    token_tracker = debug_payload.get("token_tracker", {}) or {}
    step_pruning_events = token_tracker.get("step_pruning_events", {}) or {}

    event_rows = []
    scatter_points = []
    event_id = 0
    owner_rows: Dict[str, Dict[str, Any]] = {}

    step_ranges: List[Tuple[int, int, int]] = []
    for sid_str, rng in sorted(step_token_ranges.items(), key=lambda kv: int(kv[0])):
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            continue
        sid = int(sid_str)
        s, e = int(rng[0]), int(rng[1])
        if e >= s:
            step_ranges.append((sid, s, e))

    def _owner_step(global_id: int) -> int:
        gid = int(global_id)
        for sid, s, e in step_ranges:
            if s <= gid <= e:
                return sid
        return -1

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
    for prune_step_str, dropped_ids in sorted(step_pruning_events.items(), key=lambda kv: int(kv[0])):
        prune_step = int(prune_step_str)
        dropped_ids = sorted(set(int(x) for x in (dropped_ids or [])))
        dropped_ids = [gid for gid in dropped_ids if gid >= prompt_token_count]
        if not dropped_ids:
            continue
        final_event_id += 1
        owner_counts: Dict[int, int] = {}
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
    for sid_str, rng in sorted(step_token_ranges.items(), key=lambda kv: int(kv[0])):
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            continue
        sid = int(sid_str)
        end_abs = int(rng[1])
        end_shifted = end_abs - prompt_token_count
        if end_shifted >= 0:
            step_boundaries.append({"step": sid, "x": float(end_shifted) + 0.5})

    return {
        "prompt_token_count": prompt_token_count,
        "events": event_rows,
        "points": scatter_points,
        "final_events": final_event_rows,
        "final_points": final_points,
        "dropped_by_owner_step": [owner_rows[k] for k in sorted(owner_rows.keys(), key=lambda x: int(x))],
        "step_boundaries": step_boundaries,
    }


def _step_label(step: int) -> str:
    return f"step {step}"


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


def _plot_three_methods(
    method_plot_data: List[Tuple[str, Dict[str, Any]]],
    output_pdf: str,
) -> None:
    """Three rows: H2O / TOVA / StepKV dropped-token scatter (shared x-axis)."""
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

    boundaries = method_plot_data[0][1].get("step_boundaries", []) if method_plot_data else []
    all_steps: set[int] = set()
    for _, plot_data in method_plot_data:
        points = plot_data.get("final_points", []) or plot_data.get("points", []) or []
        step_key = "owner_step" if points and "owner_step" in points[0] else "react_step"
        for p in points:
            all_steps.add(_point_display_step(p, step_key))
    steps = sorted(s for s in all_steps if s >= 1)
    cmap = plt.get_cmap("tab10")
    step_to_color = {s: cmap(i % 10) for i, s in enumerate(steps)}
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=step_to_color[s],
            markersize=8,
            linestyle="None",
        )
        for s in steps
    ]
    legend_labels = [_step_label(s) for s in steps]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.subplots_adjust(hspace=0.22, bottom=0.11, top=0.98)

    for ax, (method_label, plot_data) in zip(axes, method_plot_data):
        points = plot_data.get("final_points", []) or plot_data.get("points", []) or []
        step_key = "owner_step" if points and "owner_step" in points[0] else "react_step"
        y_vals: List[int] = []

        if points:
            for step in steps:
                matched = [p for p in points if _point_display_step(p, step_key) == step]
                xs = [p["x"] for p in matched]
                ys = [p["event_id"] for p in matched]
                if not xs:
                    continue
                y_vals.extend(int(y) for y in ys)
                ax.scatter(xs, ys, s=16, alpha=0.85, c=[step_to_color[step]], edgecolors="none")
        else:
            ax.text(
                0.5,
                0.5,
                "No dropped-token points under current config",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=13,
                color="gray",
            )

        for bd in boundaries:
            ax.axvline(float(bd["x"]), linestyle="--", linewidth=1.0, color="gray", alpha=0.7)

        ax.set_ylabel("Prune Event", fontsize=15)
        ax.set_title(method_label, loc="left", fontsize=15, pad=6)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        if y_vals:
            y_min, y_max = min(y_vals), max(y_vals)
            pad = 1 if y_max > y_min else 0
            ax.set_ylim(y_min - pad, y_max + pad)
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Key Position Index (No Prefill)", fontsize=15)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.06),
            ncol=min(8, max(1, len(legend_labels))),
            frameon=False,
        )

    os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


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
        description="Plot dropped decode tokens for H2O / TOVA / StepKV on HotpotQA."
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
    parser.add_argument("--max_steps", type=int, default=12)
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
    pdf_path = os.path.join(args.output_dir, f"{prefix}.pdf")
    points_jsonl_path = os.path.join(args.output_dir, f"{prefix}_points.jsonl")
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

    with open(points_jsonl_path, "w", encoding="utf-8") as f:
        for display_name, plot_data in method_plot_data:
            for p in plot_data.get("final_points", []) or plot_data.get("points", []) or []:
                f.write(json.dumps({"method": display_name, **p}, ensure_ascii=False) + "\n")

    try:
        _plot_three_methods(method_plot_data, pdf_path)
    except Exception as e:
        with open(plot_error_path, "w", encoding="utf-8") as f:
            f.write(str(e))
        print(f"[WARN] Plot generation failed, see: {plot_error_path}")
        raise

    print(f"[DONE] Data saved: {json_path}")
    print(f"[DONE] Point data saved: {points_jsonl_path}")
    print(f"[DONE] Figure saved: {pdf_path}")


if __name__ == "__main__":
    main()
