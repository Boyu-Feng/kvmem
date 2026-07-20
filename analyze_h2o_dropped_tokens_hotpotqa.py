import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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


def _step_ranges_from_token_ranges(
    step_token_ranges: Dict[str, Any],
) -> List[Tuple[int, int, int]]:
    """Use raw step_token_ranges from the episode debug payload."""
    out: List[Tuple[int, int, int]] = []
    for sid in sorted(int(k) for k in step_token_ranges.keys()):
        rng = step_token_ranges.get(str(sid)) or step_token_ranges.get(sid)
        if not isinstance(rng, (list, tuple)) or len(rng) != 2:
            continue
        s, e = int(rng[0]), int(rng[1])
        if e < s:
            continue
        out.append((sid, s, e))
    return out


def _build_full_step_ranges(
    step_token_ranges: Dict[str, Any],
    obs_step_ranges: Dict[str, Any],
) -> List[Tuple[int, int, int]]:
    """Merge Think+Action with the following Observation into one ReAct step span."""
    full_ranges: List[Tuple[int, int, int]] = []
    for sid, s, e in _step_ranges_from_token_ranges(step_token_ranges):
        obs_rng = obs_step_ranges.get(str(sid + 1)) or obs_step_ranges.get(sid + 1)
        if isinstance(obs_rng, (list, tuple)) and len(obs_rng) == 2:
            e = max(e, int(obs_rng[1]))
        full_ranges.append((sid, s, e))
    return full_ranges


def _resolve_owner_step(
    global_id: int,
    step_ranges: List[Tuple[int, int, int]],
) -> int:
    """Map a global token id to the ReAct step span in step_token_ranges."""
    gid = int(global_id)
    for sid, s, e in step_ranges:
        if s <= gid <= e:
            return sid
    return -1


def _max_global_id_at_prune_step(
    step_token_ranges: Dict[str, Any],
    prune_step: int,
    next_global_id: int,
) -> int:
    """
    Last global token id present in cache when step ``prune_step`` finishes pruning.

    Loop k+1 always starts at step_token_ranges[k+1][0] (obs prefill + new decode),
    so the cache at the end of step k ends one token before that.
    """
    cap = int(prune_step)
    if cap <= 0:
        rng2 = step_token_ranges.get("2") or step_token_ranges.get(2)
        if isinstance(rng2, (list, tuple)) and len(rng2) == 2:
            return int(rng2[0]) - 1
        rng1 = step_token_ranges.get("1") or step_token_ranges.get(1)
        if isinstance(rng1, (list, tuple)) and len(rng1) == 2:
            return int(rng1[1])
        return int(next_global_id) - 1

    next_sid = cap + 1
    rng = step_token_ranges.get(str(next_sid)) or step_token_ranges.get(next_sid)
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        return int(rng[0]) - 1
    rng = step_token_ranges.get(str(cap)) or step_token_ranges.get(cap)
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        return int(rng[1])
    return int(next_global_id) - 1


def _resolve_owner_step_at_prune(
    global_id: int,
    step_ranges: List[Tuple[int, int, int]],
    prune_step: int,
) -> int:
    """
    Owner step for a token evicted at ``prune_step``.

    Only steps that have already generated tokens by that prune can own the token
    (owner_step <= prune_step). This avoids labeling step-3/4 tokens on y=2.
    """
    gid = int(global_id)
    cap = max(0, int(prune_step))
    for sid, s, e in step_ranges:
        if int(sid) > cap:
            break
        if s <= gid <= e:
            return int(sid)
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
    next_global_id = int(token_tracker.get("next_global_id", prompt_token_count))

    def _owner_step(global_id: int) -> int:
        return _resolve_owner_step(global_id, step_ranges)

    def _owner_step_when_pruned(global_id: int, prune_step: int) -> int:
        return _resolve_owner_step_at_prune(global_id, step_ranges, prune_step)

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
                max_gid = _max_global_id_at_prune_step(
                    step_token_ranges,
                    0,
                    next_global_id,
                )
                if int(gid) > int(max_gid):
                    continue
                owner = int(_owner_step_when_pruned(gid, 0))
                if owner < 1:
                    continue
                shifted_ids.append(shifted_x)
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
            if shifted_ids:
                final_event_rows.append(
                    {
                        "event_id": int(final_event_id),
                        "prune_step": 0,
                        "tokens_evicted": int(len(shifted_ids)),
                        "evicted_abs_indices_no_prefill": shifted_ids,
                        "owner_step_counts": {str(int(k)): int(v) for k, v in sorted(owner_counts.items())},
                    }
                )
            else:
                final_event_id -= 1

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
            max_gid = _max_global_id_at_prune_step(
                step_token_ranges,
                int(prune_step),
                next_global_id,
            )
            if int(gid) > int(max_gid):
                continue
            owner = int(_owner_step_when_pruned(gid, int(prune_step)))
            if owner < 1:
                continue
            shifted_ids.append(shifted_x)
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
        if not shifted_ids:
            final_event_id -= 1
            continue
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

    kept_points, kept_owner_counts_by_snapshot = _build_kept_points(
        prompt_token_count=prompt_token_count,
        step_token_ranges=step_token_ranges,
        step_ranges=step_ranges,
        step_pruning_events=step_pruning_events,
        token_tracker=token_tracker,
    )

    owner_step_totals = _count_owner_step_totals(step_ranges, prompt_token_count, next_global_id)
    eviction_stats = _build_eviction_stats(final_points, owner_step_totals)
    survival_dynamics = _build_survival_dynamics(
        kept_owner_counts_by_snapshot,
        owner_step_totals,
    )
    cohort_survival = _build_cohort_survival_dynamics(
        kept_owner_counts_by_snapshot,
        owner_step_totals,
    )

    return {
        "prompt_token_count": prompt_token_count,
        "events": event_rows,
        "points": scatter_points,
        "final_events": final_event_rows,
        "final_points": final_points,
        "kept_points": kept_points,
        "kept_owner_counts_by_snapshot": kept_owner_counts_by_snapshot,
        "dropped_by_owner_step": [owner_rows[k] for k in sorted(owner_rows.keys(), key=lambda x: int(x))],
        "step_boundaries": step_boundaries,
        "owner_step_totals": owner_step_totals,
        "eviction_stats": eviction_stats,
        "survival_dynamics": survival_dynamics,
        "cohort_survival": cohort_survival,
    }


def _count_owner_step_totals(
    step_ranges: List[Tuple[int, int, int]],
    prompt_token_count: int,
    next_global_id: int,
) -> Dict[int, int]:
    """Decode tokens per ReAct owner step (Think+Action+Observation span)."""
    totals: Dict[int, int] = {}
    decode_end = int(next_global_id) - 1
    for sid, start_abs, end_abs in step_ranges:
        start_abs = max(int(start_abs), int(prompt_token_count))
        end_abs = min(int(end_abs), decode_end)
        if end_abs < start_abs:
            continue
        totals[int(sid)] = int(totals.get(int(sid), 0) + (end_abs - start_abs + 1))
    return totals


def _build_survival_dynamics(
    kept_owner_counts_by_snapshot: Dict[str, Dict[str, int]],
    owner_step_totals: Dict[int, int],
) -> List[Dict[str, Any]]:
    """
    After each ReAct step k, track how many tokens remain from:
    - prior steps (1..k-1)
    - the current/new step k
    """
    owner_steps = sorted(int(s) for s in owner_step_totals.keys())
    if not owner_steps:
        return []

    rows: List[Dict[str, Any]] = []
    for snapshot_step in owner_steps:
        counts = kept_owner_counts_by_snapshot.get(str(snapshot_step), {}) or {}
        prior_kept = 0
        for owner_key, kept_n in counts.items():
            try:
                owner = int(owner_key)
            except (TypeError, ValueError):
                continue
            if owner < int(snapshot_step):
                prior_kept += int(kept_n)

        current_kept = int(counts.get(str(snapshot_step), counts.get(int(snapshot_step), 0)) or 0)
        prior_total = sum(int(owner_step_totals.get(step, 0)) for step in owner_steps if step < snapshot_step)
        current_total = int(owner_step_totals.get(int(snapshot_step), 0))
        total_kept = int(prior_kept + current_kept)

        prior_frac = (float(prior_kept) / float(prior_total)) if prior_total > 0 else None
        current_frac = (float(current_kept) / float(current_total)) if current_total > 0 else None
        prior_share = (float(prior_kept) / float(total_kept)) if total_kept > 0 else None
        current_share = (float(current_kept) / float(total_kept)) if total_kept > 0 else None

        rows.append(
            {
                "snapshot_step": int(snapshot_step),
                "prior_kept": int(prior_kept),
                "current_kept": int(current_kept),
                "total_kept": int(total_kept),
                "prior_total": int(prior_total),
                "current_total": int(current_total),
                "prior_kept_frac": prior_frac,
                "current_kept_frac": current_frac,
                "prior_share_of_cache": prior_share,
                "current_share_of_cache": current_share,
            }
        )
    return rows


def _build_cohort_survival_dynamics(
    kept_owner_counts_by_snapshot: Dict[str, Dict[str, int]],
    owner_step_totals: Dict[int, int],
) -> List[Dict[str, Any]]:
    """
    Track each owner-step cohort separately across later snapshots.

    Example: step-2 tokens may survive step-2 prune but keep dropping at steps 3..7.
    """
    owner_steps = sorted(int(s) for s in owner_step_totals.keys())
    rows: List[Dict[str, Any]] = []
    for owner_step in owner_steps:
        total = int(owner_step_totals.get(owner_step, 0))
        if total <= 0:
            continue
        for snapshot_step in owner_steps:
            if int(snapshot_step) < int(owner_step):
                continue
            counts = kept_owner_counts_by_snapshot.get(str(snapshot_step), {}) or {}
            kept = int(counts.get(str(owner_step), counts.get(owner_step, 0)) or 0)
            rows.append(
                {
                    "owner_step": int(owner_step),
                    "snapshot_step": int(snapshot_step),
                    "kept": int(kept),
                    "total": int(total),
                    "kept_frac": float(kept) / float(total),
                }
            )
    return rows


def _build_eviction_stats(
    final_points: Sequence[Dict[str, Any]],
    owner_step_totals: Dict[int, int],
) -> Dict[str, Any]:
    """
    Quantify (prune_step, owner_step) evictions vs each owner step's total decode tokens.

    Helps separate global cache_ratio budget effects from cross-step score competition.
    """
    matrix: Dict[str, int] = {}
    evicted_global_ids: set[int] = set()
    evicted_by_owner: Dict[int, int] = {}

    for point in final_points or []:
        owner = _display_step_from_point(point)
        if owner < 1:
            continue
        prune_step = max(0, int(point.get("prune_step", point.get("react_step", 0)) or 0))
        matrix_key = f"{int(prune_step)}:{int(owner)}"
        matrix[matrix_key] = int(matrix.get(matrix_key, 0) + 1)
        evicted_by_owner[int(owner)] = int(evicted_by_owner.get(int(owner), 0) + 1)
        gid = point.get("global_id")
        if gid is not None:
            evicted_global_ids.add(int(gid))

    owner_steps = sorted(
        set(owner_step_totals.keys()) | set(evicted_by_owner.keys()),
        key=lambda x: int(x),
    )
    prune_steps = sorted(
        set(
            max(0, int(p.get("prune_step", p.get("react_step", 0)) or 0))
            for p in (final_points or [])
        )
        | set(int(s) for s in owner_steps)
    )

    cross_tab: List[Dict[str, Any]] = []
    for prune_step in prune_steps:
        for owner in owner_steps:
            count = int(matrix.get(f"{int(prune_step)}:{int(owner)}", 0))
            if count <= 0:
                continue
            owner_total = int(owner_step_totals.get(int(owner), 0))
            cross_tab.append(
                {
                    "prune_step": int(prune_step),
                    "owner_step": int(owner),
                    "evicted": int(count),
                    "owner_total_tokens": int(owner_total),
                    "evicted_frac_of_owner_total": (
                        float(count) / float(owner_total) if owner_total > 0 else None
                    ),
                    "same_owner_as_prune": bool(int(owner) == int(prune_step)),
                }
            )

    per_owner: List[Dict[str, Any]] = []
    for owner in owner_steps:
        owner_total = int(owner_step_totals.get(int(owner), 0))
        evicted = int(evicted_by_owner.get(int(owner), 0))
        per_owner.append(
            {
                "owner_step": int(owner),
                "total_tokens": int(owner_total),
                "total_evicted": int(evicted),
                "evicted_frac": (float(evicted) / float(owner_total) if owner_total > 0 else None),
                "survived_frac": (
                    float(owner_total - evicted) / float(owner_total) if owner_total > 0 else None
                ),
            }
        )

    per_prune: List[Dict[str, Any]] = []
    prune_totals: Dict[int, int] = {}
    for row in cross_tab:
        ps = int(row["prune_step"])
        prune_totals[ps] = int(prune_totals.get(ps, 0) + int(row["evicted"]))
    for prune_step in prune_steps:
        total_evicted = int(prune_totals.get(int(prune_step), 0))
        same_owner = int(matrix.get(f"{int(prune_step)}:{int(prune_step)}", 0))
        cross_owner = int(total_evicted - same_owner)
        owner_total = int(owner_step_totals.get(int(prune_step), 0))
        per_prune.append(
            {
                "prune_step": int(prune_step),
                "total_evicted": int(total_evicted),
                "same_owner_evicted": int(same_owner),
                "cross_owner_evicted": int(cross_owner),
                "same_owner_frac_of_event": (
                    float(same_owner) / float(total_evicted) if total_evicted > 0 else None
                ),
                "cross_owner_frac_of_event": (
                    float(cross_owner) / float(total_evicted) if total_evicted > 0 else None
                ),
                "same_owner_evicted_frac_of_owner_total": (
                    float(same_owner) / float(owner_total) if owner_total > 0 else None
                ),
            }
        )

    return {
        "cross_tab": cross_tab,
        "per_owner": per_owner,
        "per_prune_step": per_prune,
        "owner_steps": [int(s) for s in owner_steps],
        "prune_steps": [int(s) for s in prune_steps],
        "unique_evicted_tokens": int(len(evicted_global_ids) if evicted_global_ids else sum(evicted_by_owner.values())),
    }


def _annotate_expected_budget_evict_frac(
    eviction_stats: Dict[str, Any],
    cache_ratio: float,
) -> None:
    expected = float(max(0.0, min(1.0, 1.0 - float(cache_ratio))))
    for row in eviction_stats.get("per_prune_step", []) or []:
        row["expected_budget_evict_frac"] = expected


def _format_eviction_stats_text(
    method_label: str,
    eviction_stats: Dict[str, Any],
    cache_ratio: float,
) -> str:
    """Render human-readable tables for console / txt export."""
    lines: List[str] = []
    lines.append(f"=== Eviction stats: {method_label} (cache_ratio={cache_ratio:.3f}) ===")
    lines.append("")

    lines.append("Table A: (prune_step x owner_step) evicted count / owner total / frac")
    lines.append("prune | owner | evicted | owner_total | evicted/owner | same?")
    lines.append("------+-------+---------+-------------+---------------+------")
    for row in sorted(
        eviction_stats.get("cross_tab", []) or [],
        key=lambda r: (int(r["prune_step"]), int(r["owner_step"])),
    ):
        frac = row.get("evicted_frac_of_owner_total")
        frac_s = f"{float(frac):.3f}" if frac is not None else "n/a"
        same = "Y" if row.get("same_owner_as_prune") else "N"
        lines.append(
            f"{int(row['prune_step']):5d} | {int(row['owner_step']):5d} | "
            f"{int(row['evicted']):7d} | {int(row['owner_total_tokens']):11d} | "
            f"{frac_s:13s} | {same:4s}"
        )
    lines.append("")

    lines.append("Table B: per owner_step cumulative evicted vs total decode tokens")
    lines.append("owner | total | evicted | evicted_frac | survived_frac")
    lines.append("------+-------+---------+--------------+--------------")
    for row in eviction_stats.get("per_owner", []) or []:
        ev_frac = row.get("evicted_frac")
        sv_frac = row.get("survived_frac")
        ev_s = f"{float(ev_frac):.3f}" if ev_frac is not None else "n/a"
        sv_s = f"{float(sv_frac):.3f}" if sv_frac is not None else "n/a"
        lines.append(
            f"{int(row['owner_step']):5d} | {int(row['total_tokens']):5d} | "
            f"{int(row['total_evicted']):7d} | {ev_s:12s} | {sv_s:12s}"
        )
    lines.append("")

    expected = float(max(0.0, min(1.0, 1.0 - float(cache_ratio))))
    lines.append(
        "Table C: per prune_step — same-owner vs cross-owner evictions "
        f"(expected same/owner_total ~{expected:.3f} if budget-only)"
    )
    lines.append(
        "prune | total | same_owner | cross_owner | same/event | same/owner_total | expected"
    )
    lines.append(
        "------+-------+------------+-------------+------------+----------------+----------"
    )
    for row in eviction_stats.get("per_prune_step", []) or []:
        same_event = row.get("same_owner_frac_of_event")
        same_owner_total = row.get("same_owner_evicted_frac_of_owner_total")
        expected_row = row.get("expected_budget_evict_frac", expected)
        se_s = f"{float(same_event):.3f}" if same_event is not None else "n/a"
        so_s = f"{float(same_owner_total):.3f}" if same_owner_total is not None else "n/a"
        ex_s = f"{float(expected_row):.3f}" if expected_row is not None else "n/a"
        lines.append(
            f"{int(row['prune_step']):5d} | {int(row['total_evicted']):5d} | "
            f"{int(row['same_owner_evicted']):10d} | {int(row['cross_owner_evicted']):11d} | "
            f"{se_s:10s} | {so_s:14s} | {ex_s:8s}"
        )
    lines.append("")
    return "\n".join(lines)


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
    """Last global token id generated after finishing the given ReAct step."""
    end = int(prompt_token_count) - 1
    for sid, _s, e in step_ranges:
        if int(sid) <= int(through_step):
            end = max(end, int(e))
    return min(end, int(next_global_id) - 1)


def _build_kept_points(
    prompt_token_count: int,
    step_token_ranges: Dict[str, Any],
    step_ranges: List[Tuple[int, int, int]],
    step_pruning_events: Dict[str, Any],
    token_tracker: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """Cumulative cache snapshot: tokens still present after each ReAct step."""
    next_global_id = int(token_tracker.get("next_global_id", prompt_token_count))
    if next_global_id <= prompt_token_count:
        return [], {}

    init_dropped, react_pruning_events = _split_pruning_events(step_pruning_events)
    prune_steps = sorted(react_pruning_events.keys())
    max_range_step = max((sid for sid, _, _ in step_ranges), default=1)
    max_step = max(max(prune_steps) if prune_steps else 1, max_range_step)

    cumulative_dropped: set[int] = set(
        int(x) for x in init_dropped if int(x) >= prompt_token_count
    )
    kept_points: List[Dict[str, Any]] = []
    kept_owner_counts_by_snapshot: Dict[str, Dict[str, int]] = {}

    for snapshot_step in range(1, max_step + 1):
        dropped_now = react_pruning_events.get(snapshot_step, [])
        cumulative_dropped.update(int(x) for x in dropped_now if int(x) >= prompt_token_count)

        decode_end = _max_global_id_at_prune_step(
            step_token_ranges,
            int(snapshot_step),
            next_global_id,
        )
        if decode_end < prompt_token_count:
            continue

        owner_counts: Dict[str, int] = {}
        for gid in range(int(prompt_token_count), decode_end + 1):
            if gid in cumulative_dropped:
                continue
            owner = int(
                _resolve_owner_step_at_prune(
                    gid,
                    step_ranges,
                    int(snapshot_step),
                )
            )
            if owner < 1:
                continue
            owner_key = str(owner)
            owner_counts[owner_key] = int(owner_counts.get(owner_key, 0) + 1)
            kept_points.append(
                {
                    "snapshot_step": int(snapshot_step),
                    "owner_step": owner,
                    "x": int(gid - prompt_token_count),
                    "global_id": int(gid),
                }
            )
        kept_owner_counts_by_snapshot[str(snapshot_step)] = owner_counts

    return kept_points, kept_owner_counts_by_snapshot


def _step_label(step: int) -> str:
    return f"step{step} token"


def _display_step_from_point(point: Dict[str, Any]) -> int:
    """Return owner_step as recorded; orphans stay at -1 (not remapped)."""
    return int(point.get("owner_step", -1))


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
        return max(1, int(point.get("snapshot_step", point.get("keep_step", 1)) or 1))
    return max(
        1,
        int(point.get("prune_step", point.get("react_step", point.get("event_id", 1))) or 1),
    )


def _marker_size_for_panel(n_points: int, x_span: int) -> float:
    """Scatter area (points^2). Use x-span density so markers can be larger when spread out."""
    span = max(1, int(x_span))
    avg_per_x = float(n_points) / span
    if avg_per_x > 8:
        return 18.0
    if avg_per_x > 4:
        return 30.0
    if avg_per_x > 2:
        return 44.0
    if avg_per_x > 1:
        return 58.0
    return 72.0


def _panel_x_span(points: Sequence[Dict[str, Any]]) -> int:
    if not points:
        return 1
    xs = [int(p["x"]) for p in points]
    return max(1, max(xs) - min(xs) + 1)


def _plot_style(mode: str) -> Dict[str, float]:
    from scripts.paper_figure_style import (
        FONT_AXIS_LABEL,
        FONT_LEGEND,
        FONT_METHOD_TITLE,
        FONT_TICK,
        panel_height,
    )

    panel_h = panel_height(3)
    if mode == "dropped":
        return {
            "axis_label": FONT_AXIS_LABEL,
            "tick": FONT_TICK,
            "title": 18,
            "method_title": FONT_METHOD_TITLE,
            "legend_font": FONT_LEGEND,
            "legend_marker": 16,
            "legend_ncol": 4,
            "panel_h": panel_h,
            "fig_w": 16.0,
            "labelpad": 6,
            "hspace": 0.38,
        }
    return {
        "axis_label": FONT_AXIS_LABEL,
        "tick": FONT_TICK,
        "title": 18,
        "method_title": FONT_METHOD_TITLE,
        "legend_font": FONT_LEGEND,
        "legend_marker": 14,
        "legend_ncol": 4,
        "panel_h": panel_h,
        "fig_w": 14.0,
        "fig_w_min": 16.0,
        "fig_w_max": 32.0,
        "fig_w_scale": 0.045,
        "labelpad": 4,
        "hspace": 0.12,
    }


def _legend_bottom_margin(n_items: int, ncol: int) -> float:
    rows = max(1, (int(n_items) + int(ncol) - 1) // int(ncol))
    return 0.04 + 0.032 * rows


def _bottom_axis_labelpad() -> int:
    return 2


def _multi_panel_chart_style() -> Dict[str, float]:
    from scripts.paper_figure_style import (
        FONT_AXIS_LABEL,
        FONT_LEGEND,
        FONT_METHOD_TITLE,
        FONT_TICK,
        panel_height,
    )

    return {
        "axis_label": FONT_AXIS_LABEL,
        "tick": FONT_TICK,
        "legend": FONT_LEGEND,
        "method_title": FONT_METHOD_TITLE,
        "panel_h": panel_height(3),
        "hspace": 0.28,
        "labelpad": 6,
    }


def _set_middle_ylabel(
    axes: Sequence[Any],
    ax_idx: int,
    label: str,
    fontsize: float,
    labelpad: float = 6,
) -> None:
    if ax_idx == len(axes) // 2:
        axes[ax_idx].set_ylabel(label, fontsize=fontsize, labelpad=labelpad)
    else:
        axes[ax_idx].set_ylabel("")


def _step_legend_ncol(n_items: int) -> int:
    """Lay out step legend in two rows (e.g. step1..step7 -> 4 + 3)."""
    n = max(1, int(n_items))
    if n <= 4:
        return n
    return (n + 1) // 2


def _global_react_step_count(
    method_plot_data: List[Tuple[str, Dict[str, Any]]],
    max_react_steps: Optional[int] = None,
) -> int:
    """Max ReAct step count across all methods (for shared axes/legends)."""
    max_step = 1
    for _, plot_data in method_plot_data:
        owner_totals = plot_data.get("owner_step_totals") or {}
        if owner_totals:
            max_step = max(max_step, max(int(k) for k in owner_totals.keys()))
        for bd in plot_data.get("step_boundaries", []) or []:
            max_step = max(max_step, int(bd.get("step", 1)))
        for row in plot_data.get("cohort_survival", []) or []:
            max_step = max(
                max_step,
                int(row.get("snapshot_step", 1)),
                int(row.get("owner_step", 1)),
            )
        for row in plot_data.get("survival_dynamics", []) or []:
            max_step = max(max_step, int(row.get("snapshot_step", 1)))
    if max_react_steps is not None and int(max_react_steps) > 0:
        max_step = min(max_step, int(max_react_steps))
    return max(1, int(max_step))


def _global_owner_steps(
    method_plot_data: List[Tuple[str, Dict[str, Any]]],
    max_react_steps: Optional[int] = None,
) -> List[int]:
    """Union of owner steps across methods, up to the global max step count."""
    steps: set[int] = set()
    for _, plot_data in method_plot_data:
        owner_totals = plot_data.get("owner_step_totals") or {}
        steps.update(int(k) for k in owner_totals.keys())
    max_step = _global_react_step_count(method_plot_data, max_react_steps)
    if not steps:
        steps = set(range(1, max_step + 1))
    steps = {s for s in steps if 1 <= s <= max_step}
    return sorted(steps) if steps else list(range(1, max_step + 1))


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

    style = _plot_style(mode)
    plt.rcParams.update(
        {
            "font.size": style["tick"],
            "axes.labelsize": style["axis_label"],
            "axes.titlesize": style["title"],
            "xtick.labelsize": style["tick"],
            "ytick.labelsize": style["tick"],
            "legend.fontsize": style["legend_font"],
        }
    )

    all_owner_steps: set[int] = set()
    panel_max_x = 0
    for _, plot_data in method_plot_data:
        points, step_key = _collect_plot_points(plot_data, mode)
        for p in points:
            all_owner_steps.add(_point_display_step(p, step_key))
            panel_max_x = max(panel_max_x, int(p["x"]))
    owner_steps = _global_owner_steps(method_plot_data, max_react_steps)
    cmap = plt.get_cmap("tab10")
    owner_to_color = {s: cmap((s - 1) % 10) for s in owner_steps}
    legend_marker = style.get("legend_marker", 9)
    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=owner_to_color[s],
            markersize=legend_marker,
            linestyle="None",
        )
        for s in owner_steps
    ]
    legend_labels = [_step_label(s) for s in owner_steps]

    global_y_max = _global_react_step_count(method_plot_data, max_react_steps)
    global_y_max = max(
        global_y_max,
        max(
            (_max_react_step_in_plot_data(plot_data, mode) for _, plot_data in method_plot_data),
            default=1,
        ),
    )

    n_panels = len(method_plot_data)
    from scripts.paper_figure_style import FIG_H, FIG_W, SUBPLOTS_LEFT, SUBPLOTS_RIGHT, SUBPLOTS_TOP, apply_top_legend

    if mode == "dropped":
        fig_w = FIG_W
        fig_h = FIG_H
    else:
        fig_w = max(style["fig_w_min"], min(style["fig_w_max"], 10.0 + panel_max_x * style["fig_w_scale"]))
        fig_h = style["panel_h"] * n_panels
    fig, axes = plt.subplots(n_panels, 1, figsize=(fig_w, fig_h), sharex=False)
    if n_panels == 1:
        axes = [axes]
    has_top_legend = bool(legend_handles)
    legend_ncol = _step_legend_ncol(len(legend_labels)) if has_top_legend else 4
    panel_hspace = float(style.get("hspace", 0.12))
    fig.subplots_adjust(
        hspace=panel_hspace,
        bottom=0.06,
        top=SUBPLOTS_TOP if has_top_legend else 0.96,
        left=SUBPLOTS_LEFT,
        right=SUBPLOTS_RIGHT,
    )

    y_label = "Cumulative keep after step" if mode == "kept" else "Evicted at ReAct step"
    empty_msg = (
        "No cumulative kept tokens under current config"
        if mode == "kept"
        else "No dropped-token points under current config"
    )
    x_labelpad = _bottom_axis_labelpad()

    for ax_idx, (method_label, plot_data) in enumerate(method_plot_data):
        ax = axes[ax_idx]
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
                size = _marker_size_for_panel(len(xs), _panel_x_span(matched))
                face = owner_to_color[owner_step]
                if mode == "kept":
                    ax.scatter(
                        xs,
                        ys,
                        s=size,
                        c=[face],
                        edgecolors="none",
                        linewidths=0,
                        rasterized=True,
                    )
                else:
                    edgecolors = []
                    linewidths = []
                    for p in matched:
                        prune_step = max(1, int(p.get("prune_step", p.get("react_step", 1)) or 1))
                        owner = _point_display_step(p, step_key)
                        if prune_step < owner:
                            edgecolors.append("#d62728")
                            linewidths.append(0.5)
                        else:
                            edgecolors.append("none")
                            linewidths.append(0.0)
                    ax.scatter(
                        xs,
                        ys,
                        s=size,
                        c=[face],
                        edgecolors=edgecolors,
                        linewidths=linewidths,
                        rasterized=True,
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

        if ax_idx == len(axes) // 2:
            ax.set_ylabel(y_label, fontsize=style["axis_label"], labelpad=style["labelpad"])
        else:
            ax.set_ylabel("")
        ax.set_title(
            method_label,
            loc="left",
            fontsize=float(style.get("method_title", style["axis_label"])),
            pad=4,
        )
        ax.tick_params(axis="both", which="major", labelsize=style["tick"], pad=3)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
        ax.set_ylim(0.5, global_y_max + 0.5)
        ax.set_yticks(list(range(1, global_y_max + 1)))
        if x_vals:
            ax.set_xlim(-0.5, max(x_vals) + 0.5)
        ax.grid(True, alpha=0.25)

    x_labelpad = _bottom_axis_labelpad()
    axes[-1].set_xlabel(
        "Key Position Index (No Prefill)",
        fontsize=style["axis_label"],
        labelpad=x_labelpad,
    )
    axes[-1].tick_params(axis="x", which="major", labelsize=style["tick"], pad=3)

    if has_top_legend:
        apply_top_legend(fig, legend_handles, legend_labels, ncol=legend_ncol)

    os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def _extract_final_survival_series(
    plot_data: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Per owner step: fraction of decode tokens still present after final inference."""
    rows = (plot_data.get("eviction_stats", {}) or {}).get("per_owner", []) or []
    series: List[Dict[str, Any]] = []
    for row in rows:
        owner = int(row.get("owner_step", 0))
        if owner < 1:
            continue
        total = int(row.get("total_tokens", 0))
        evicted = int(row.get("total_evicted", 0))
        survived = max(0, total - evicted)
        survived_frac = row.get("survived_frac")
        if survived_frac is None and total > 0:
            survived_frac = float(survived) / float(total)
        series.append(
            {
                "owner_step": owner,
                "total_tokens": total,
                "survived_tokens": survived,
                "survived_frac": float(survived_frac) if survived_frac is not None else None,
            }
        )
    return sorted(series, key=lambda r: int(r["owner_step"]))


def _plot_final_survival_line(
    method_plot_data: List[Tuple[str, Dict[str, Any]]],
    output_pdf: str,
    max_react_steps: Optional[int] = None,
) -> None:
    """Line chart: after full inference, remaining token fraction per ReAct step."""
    method_colors = {
        "H2O": "#1f77b4",
        "TOVA": "#ff7f0e",
        "StepKV": "#2ca02c",
    }
    fallback_cmap = plt.get_cmap("tab10")

    all_steps = set(_global_owner_steps(method_plot_data, max_react_steps))
    method_series: List[Tuple[str, List[Dict[str, Any]]]] = []
    for idx, (method_label, plot_data) in enumerate(method_plot_data):
        series = _extract_final_survival_series(plot_data)
        method_series.append((method_label, series))
        for row in series:
            all_steps.add(int(row["owner_step"]))

    if not all_steps:
        fig, ax = plt.subplots(figsize=(8.0, 5.0))
        ax.text(
            0.5,
            0.5,
            "No survival data under current config",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=13,
            color="gray",
        )
        os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
        fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.10)
        plt.close(fig)
        return

    x_max = _global_react_step_count(method_plot_data, max_react_steps)
    x_ticks = list(range(1, x_max + 1))

    fig, ax = plt.subplots(figsize=(9.0, 5.5))
    for idx, (method_label, series) in enumerate(method_series):
        by_step = {int(r["owner_step"]): r for r in series}
        xs: List[int] = []
        ys: List[float] = []
        for step in x_ticks:
            row = by_step.get(step)
            if not row or row.get("survived_frac") is None:
                continue
            xs.append(step)
            ys.append(float(row["survived_frac"]))
        if not xs:
            continue
        color = method_colors.get(method_label, fallback_cmap(idx % 10))
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=7,
            linewidth=2.0,
            label=method_label,
            color=color,
        )

    x_labelpad = _bottom_axis_labelpad()
    ax.set_xlabel("ReAct step (token origin)", fontsize=24, labelpad=x_labelpad)
    ax.set_ylabel("Remain tokens", fontsize=24, labelpad=6)
    ax.set_xticks(x_ticks)
    ax.set_xlim(0.5, x_max + 0.5)
    ax.set_ylim(0.0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis="both", which="major", labelsize=22)
    handles, labels = ax.get_legend_handles_labels()
    legend_ncol = max(1, len(labels))
    bottom = _legend_bottom_margin(len(labels), legend_ncol) + 0.01 if labels else 0.06
    fig.subplots_adjust(left=0.10, right=0.98, top=0.98, bottom=bottom)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=legend_ncol,
            frameon=False,
            fontsize=21,
            handlelength=2.0,
            handletextpad=0.8,
        )

    os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.10)
    plt.close(fig)


def _plot_survival_dynamics(
    method_plot_data: List[Tuple[str, Dict[str, Any]]],
    output_pdf: str,
    max_react_steps: Optional[int] = None,
) -> None:
    """Stacked area: prior-step vs current-step tokens still in cache after each prune."""
    prior_color = "#4C72B0"
    current_color = "#DD8452"
    style = _multi_panel_chart_style()

    plt.rcParams.update(
        {
            "font.size": style["tick"],
            "axes.labelsize": style["axis_label"],
            "legend.fontsize": style["legend"],
        }
    )

    n_panels = len(method_plot_data)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14.0, style["panel_h"] * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    global_x_max = _global_react_step_count(method_plot_data, max_react_steps)
    x_labelpad = _bottom_axis_labelpad()
    has_any = False
    dynamics_legend = [
        Patch(facecolor=prior_color, alpha=0.78, label="Prior steps in cache"),
        Patch(facecolor=current_color, alpha=0.78, label="Current step in cache"),
        Line2D(
            [0],
            [0],
            color=prior_color,
            linewidth=3.0,
            marker="s",
            markersize=8,
            linestyle="--",
            label="Prior kept tokens",
        ),
        Line2D(
            [0],
            [0],
            color=current_color,
            linewidth=3.0,
            marker="o",
            markersize=9,
            linestyle="-",
            label="Current kept tokens",
        ),
        Line2D(
            [0],
            [0],
            color=prior_color,
            linewidth=2.2,
            marker="s",
            markersize=7,
            linestyle=":",
            label="Prior kept %",
        ),
        Line2D(
            [0],
            [0],
            color=current_color,
            linewidth=2.2,
            marker="o",
            markersize=7,
            linestyle="-.",
            label="Current kept %",
        ),
    ]
    legend_ncol = 3

    for ax_idx, (ax, (_method_label, plot_data)) in enumerate(zip(axes, method_plot_data)):
        rows = list(plot_data.get("survival_dynamics", []) or [])
        if not rows:
            ax.text(
                0.5,
                0.5,
                "No survival dynamics under current config",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=13,
                color="gray",
            )
            _set_middle_ylabel(
                axes,
                ax_idx,
                "Kept tokens in cache",
                style["axis_label"],
                style["labelpad"],
            )
            continue

        has_any = True
        row_by_step = {int(r["snapshot_step"]): r for r in rows}
        xs = list(range(1, global_x_max + 1))
        prior_kept = [int(row_by_step[x]["prior_kept"]) if x in row_by_step else 0 for x in xs]
        current_kept = [int(row_by_step[x]["current_kept"]) if x in row_by_step else 0 for x in xs]
        total_kept = [int(row_by_step[x]["total_kept"]) if x in row_by_step else 0 for x in xs]
        prior_frac = [
            float(row_by_step[x]["prior_kept_frac"])
            if x in row_by_step and row_by_step[x].get("prior_kept_frac") is not None
            else float("nan")
            for x in xs
        ]
        current_frac = [
            float(row_by_step[x]["current_kept_frac"])
            if x in row_by_step and row_by_step[x].get("current_kept_frac") is not None
            else float("nan")
            for x in xs
        ]

        plotted_xs = [x for x in xs if x in row_by_step]
        if not plotted_xs:
            continue

        ax.stackplot(
            plotted_xs,
            [prior_kept[xs.index(x)] for x in plotted_xs],
            [current_kept[xs.index(x)] for x in plotted_xs],
            colors=[prior_color, current_color],
            alpha=0.78,
        )
        ax.plot(
            plotted_xs,
            [prior_kept[xs.index(x)] for x in plotted_xs],
            color=prior_color,
            linewidth=3.0,
            marker="s",
            markersize=8,
            linestyle="--",
            zorder=5,
        )
        ax.plot(
            plotted_xs,
            [current_kept[xs.index(x)] for x in plotted_xs],
            color=current_color,
            linewidth=3.0,
            marker="o",
            markersize=9,
            linestyle="-",
            zorder=5,
        )
        _set_middle_ylabel(
            axes,
            ax_idx,
            "Kept tokens in cache",
            style["axis_label"],
            style["labelpad"],
        )
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="both", which="major", labelsize=style["tick"])

        plotted_totals = [total_kept[xs.index(x)] for x in plotted_xs]
        for x, total in zip(plotted_xs, plotted_totals):
            if total > 0:
                ax.text(
                    x,
                    total + max(1, 0.02 * max(plotted_totals)),
                    f"{total}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    color="#333333",
                )

        ax2 = ax.twinx()
        ax2.plot(
            plotted_xs,
            [prior_frac[xs.index(x)] for x in plotted_xs],
            color=prior_color,
            linewidth=2.2,
            marker="s",
            markersize=7,
            linestyle=":",
            zorder=6,
        )
        ax2.plot(
            plotted_xs,
            [current_frac[xs.index(x)] for x in plotted_xs],
            color=current_color,
            linewidth=2.2,
            marker="o",
            markersize=7,
            linestyle="-.",
            zorder=6,
        )
        ax2.set_ylim(0.0, 1.05)
        ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        if ax_idx == len(axes) // 2:
            ax2.set_ylabel("% kept", fontsize=style["axis_label"], labelpad=10)
        else:
            ax2.set_ylabel("")
        ax2.tick_params(axis="y", which="major", labelsize=style["tick"])

    for ax in axes:
        ax.set_xlim(0.5, global_x_max + 0.5)
        ax.set_xticks(list(range(1, global_x_max + 1)))

    axes[-1].set_xlabel("ReAct step", fontsize=style["axis_label"], labelpad=x_labelpad)

    bottom = _legend_bottom_margin(len(dynamics_legend), legend_ncol) + 0.01 if has_any else 0.06
    fig.subplots_adjust(
        hspace=style["hspace"],
        top=0.98,
        bottom=bottom,
        left=0.10,
        right=0.90,
    )
    if has_any:
        fig.legend(
            handles=dynamics_legend,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.02),
            ncol=legend_ncol,
            frameon=False,
            fontsize=style["legend"],
            handlelength=2.0,
            handletextpad=0.8,
            columnspacing=1.8,
        )
    os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def _plot_cohort_survival(
    method_plot_data: List[Tuple[str, Dict[str, Any]]],
    output_pdf: str,
    max_react_steps: Optional[int] = None,
) -> None:
    """One line per owner-step cohort: survival fraction after each subsequent prune."""
    style = _multi_panel_chart_style()
    plt.rcParams.update(
        {
            "font.size": style["tick"],
            "axes.labelsize": style["axis_label"],
            "legend.fontsize": style["legend"],
        }
    )

    from scripts.paper_figure_style import (
        FIG_H,
        FIG_W,
        FONT_ANNOT,
        SUBPLOTS_LEFT,
        SUBPLOTS_RIGHT,
        SUBPLOTS_TOP,
        apply_top_legend,
    )

    n_panels = len(method_plot_data)
    fig, axes = plt.subplots(n_panels, 1, figsize=(FIG_W, FIG_H), sharex=True)
    if n_panels == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab10")
    global_x_max = _global_react_step_count(method_plot_data, max_react_steps)
    global_owner_steps = _global_owner_steps(method_plot_data, max_react_steps)
    owner_to_color = {s: cmap((s - 1) % 10) for s in global_owner_steps}
    x_labelpad = _bottom_axis_labelpad()
    has_any = False

    for ax_idx, (ax, (method_label, plot_data)) in enumerate(zip(axes, method_plot_data)):
        rows = list(plot_data.get("cohort_survival", []) or [])
        if not rows:
            ax.text(
                0.5,
                0.5,
                "No cohort survival data under current config",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=13,
                color="gray",
            )
            _set_middle_ylabel(
                axes,
                ax_idx,
                "Cohort remaining %",
                style["axis_label"],
                style["labelpad"],
            )
            ax.set_title(
                method_label,
                loc="left",
                fontsize=float(style.get("method_title", style["axis_label"])),
                pad=4,
            )
            continue

        has_any = True
        by_owner: Dict[int, List[Dict[str, Any]]] = {}
        for row in rows:
            owner = int(row["owner_step"])
            by_owner.setdefault(owner, []).append(row)

        for owner in global_owner_steps:
            series = sorted(by_owner.get(owner, []), key=lambda r: int(r["snapshot_step"]))
            if not series:
                continue
            xs = [int(r["snapshot_step"]) for r in series]
            ys = [float(r["kept_frac"]) for r in series]
            color = owner_to_color[owner]
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=2.8,
                marker="o",
                markersize=8,
            )
            if xs and ys:
                ax.annotate(
                    f"{ys[-1] * 100.0:.0f}%",
                    xy=(xs[-1], ys[-1]),
                    xytext=(6, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=FONT_ANNOT,
                    color=color,
                    fontweight="bold",
                )

        _set_middle_ylabel(
            axes,
            ax_idx,
            "Cohort remaining %",
            style["axis_label"],
            style["labelpad"],
        )
        ax.set_title(
            method_label,
            loc="left",
            fontsize=float(style.get("method_title", style["axis_label"])),
            pad=4,
        )
        ax.set_ylim(0.0, 1.08)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="both", which="major", labelsize=style["tick"])

    for ax in axes:
        ax.set_xlim(0.5, global_x_max + 0.5)
        ax.set_xticks(list(range(1, global_x_max + 1)))

    axes[-1].set_xlabel("ReAct step", fontsize=style["axis_label"], labelpad=x_labelpad)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=owner_to_color[s],
            linewidth=2.8,
            marker="o",
            markersize=8,
            label=f"Step {s}",
        )
        for s in global_owner_steps
    ]
    legend_ncol = _step_legend_ncol(len(global_owner_steps))
    fig.subplots_adjust(
        hspace=style["hspace"],
        top=SUBPLOTS_TOP if has_any else 0.96,
        bottom=0.06,
        left=SUBPLOTS_LEFT,
        right=SUBPLOTS_RIGHT,
    )
    if has_any:
        apply_top_legend(
            fig,
            legend_handles,
            [f"step{s} token" for s in global_owner_steps],
            ncol=legend_ncol,
            left=SUBPLOTS_LEFT,
            right=SUBPLOTS_RIGHT,
        )
    os.makedirs(os.path.dirname(output_pdf) or ".", exist_ok=True)
    fig.savefig(output_pdf, bbox_inches="tight", pad_inches=0.12)
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
            _annotate_expected_budget_evict_frac(
                plot_data.get("eviction_stats", {}),
                cache_ratio=float(args.cache_ratio),
            )
            method_results[pruning_mode] = {
                "display_name": display_name,
                "predicted_answer": pred,
                "trajectory": traj,
                "step_timings": timings,
                "plot_data": plot_data,
                "eviction_stats": plot_data.get("eviction_stats", {}),
                "owner_step_totals": plot_data.get("owner_step_totals", {}),
                "final_survival_by_step": _extract_final_survival_series(plot_data),
                "survival_dynamics": plot_data.get("survival_dynamics", []),
                "cohort_survival": plot_data.get("cohort_survival", []),
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
    survival_pdf_path = os.path.join(args.output_dir, f"{prefix}_final_survival.pdf")
    dynamics_pdf_path = os.path.join(args.output_dir, f"{prefix}_survival_dynamics.pdf")
    cohort_pdf_path = os.path.join(args.output_dir, f"{prefix}_cohort_survival.pdf")
    eviction_stats_path = os.path.join(args.output_dir, f"{prefix}_eviction_stats.txt")
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

    stats_lines: List[str] = []
    for display_name, plot_data in method_plot_data:
        stats = plot_data.get("eviction_stats", {}) or {}
        _annotate_expected_budget_evict_frac(stats, cache_ratio=float(args.cache_ratio))
        stats_lines.append(
            _format_eviction_stats_text(display_name, stats, cache_ratio=float(args.cache_ratio))
        )
    stats_text = "\n".join(stats_lines).rstrip() + "\n"
    with open(eviction_stats_path, "w", encoding="utf-8") as f:
        f.write(stats_text)
    print(stats_text)

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
        _plot_final_survival_line(
            method_plot_data,
            survival_pdf_path,
            max_react_steps=int(args.max_steps),
        )
        _plot_survival_dynamics(
            method_plot_data,
            dynamics_pdf_path,
            max_react_steps=int(args.max_steps),
        )
        _plot_cohort_survival(
            method_plot_data,
            cohort_pdf_path,
            max_react_steps=int(args.max_steps),
        )
    except Exception as e:
        with open(plot_error_path, "w", encoding="utf-8") as f:
            f.write(str(e))
        print(f"[WARN] Plot generation failed, see: {plot_error_path}")
        raise

    print(f"[DONE] Data saved: {json_path}")
    print(f"[DONE] Eviction stats saved: {eviction_stats_path}")
    print(f"[DONE] Dropped points saved: {dropped_points_jsonl_path}")
    print(f"[DONE] Kept points saved: {kept_points_jsonl_path}")
    print(f"[DONE] Dropped figure saved: {dropped_pdf_path}")
    print(f"[DONE] Kept figure saved: {kept_pdf_path}")
    print(f"[DONE] Final survival figure saved: {survival_pdf_path}")
    print(f"[DONE] Survival dynamics figure saved: {dynamics_pdf_path}")
    print(f"[DONE] Cohort survival figure saved: {cohort_pdf_path}")


if __name__ == "__main__":
    main()
