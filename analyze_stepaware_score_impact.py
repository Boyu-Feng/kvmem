#!/usr/bin/env python3
"""
Offline analysis for Step-aware step scores.

This script reconstructs step utility scores from saved trajectory logs
and compares Top-1 / Bottom-1 / Random-1 step removal *proxy impact*:

- answer_hit_rate: dropped step observation contains gold-answer tokens
- em_true_answer_hit_rate: among EM=True samples, dropped step hits gold tokens

Outputs:
- JSON summary
- Markdown table
- Bar chart PNG
"""

import argparse
import json
import math
import os
import random
import re
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt


def normalize_answer(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


from kv_cache.anchor_utils import extract_anchor_terms as _extract_anchor_terms


def _normalize_action_arg(action_arg: str) -> str:
    s = (action_arg or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _compute_step_score(
    reward: float,
    citation: float,
    reward_weight: float = 0.85,
    citation_weight: float = 0.15,
) -> float:
    reward_clamped = max(-1.0, min(2.0, float(reward)))
    citation_sat = math.log1p(max(0.0, float(citation)))
    score = reward_weight * reward_clamped + citation_weight * citation_sat
    return float(max(0.0, min(8.0, score)))


def _gold_tokens(gold_answer: str) -> Set[str]:
    return set(normalize_answer(gold_answer).split())


def _answer_hit(observation: str, gold_answer: str) -> bool:
    g = _gold_tokens(gold_answer)
    if not g:
        return False
    obs_tokens = set(normalize_answer(observation).split())
    return len(g.intersection(obs_tokens)) > 0


def _rebuild_step_scores(trajectory: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    step_meta: Dict[int, Dict[str, Any]] = {}
    step_scores: Dict[int, float] = {}

    seen_obs_anchor_terms: Set[str] = set()
    seen_action_args: Counter = Counter()

    # Pass 1: reward + anchors per step from each observation.
    for step_item in trajectory:
        step = int(step_item.get("step", 0))
        if step <= 0:
            continue
        obs = str(step_item.get("observation", "") or "")
        action_arg = str(step_item.get("action_arg", "") or "")
        action_norm = _normalize_action_arg(action_arg)
        action_repeat = seen_action_args.get(action_norm, 0) if action_norm else 0
        if action_norm:
            seen_action_args[action_norm] += 1

        obs_terms = _extract_anchor_terms(obs)
        novelty_terms = obs_terms.difference(seen_obs_anchor_terms)
        novelty_ratio = (len(novelty_terms) / max(1, len(obs_terms))) if obs_terms else 0.0
        seen_obs_anchor_terms.update(obs_terms)

        obs_lower = obs.lower()
        success_flag = 1.0
        if ("could not find" in obs_lower) or ("invalid action" in obs_lower) or ("no results" in obs_lower):
            success_flag = 0.0
        reward_val = success_flag + novelty_ratio - 0.3 * float(action_repeat)

        step_meta[step] = {
            "anchors": obs_terms,
            "reward": float(reward_val),
            "citation": 0.0,
            "observation": obs,
        }
        step_scores[step] = _compute_step_score(reward_val, 0.0)

    # Pass 2: citation increment from later-step text reference overlap.
    # Approximate reference text by: current step thought + action + observation.
    sorted_steps = sorted(step_meta.keys())
    for t in sorted_steps:
        item = next((x for x in trajectory if int(x.get("step", 0)) == t), None)
        if item is None:
            continue
        ref_text = " ".join(
            [
                str(item.get("thought", "") or ""),
                str(item.get("action_arg", "") or ""),
                str(item.get("observation", "") or ""),
            ]
        )
        ref_terms = _extract_anchor_terms(ref_text)
        if not ref_terms:
            continue
        for s in sorted_steps:
            if s >= t:
                break
            anchors = step_meta[s].get("anchors", set())
            if not anchors:
                continue
            overlap = len(ref_terms.intersection(anchors))
            if overlap <= 0:
                continue
            base = float(min(len(ref_terms), len(anchors)))
            overlap_ratio = float(overlap) / max(1.0, base)
            citation_inc = min(1.0, overlap_ratio * 1.5)
            step_meta[s]["citation"] = float(step_meta[s]["citation"]) + float(citation_inc)
            step_scores[s] = _compute_step_score(step_meta[s]["reward"], step_meta[s]["citation"])

    out = {}
    for s in sorted_steps:
        out[s] = {
            "score": float(step_scores[s]),
            "reward": float(step_meta[s]["reward"]),
            "citation": float(step_meta[s]["citation"]),
            "observation": str(step_meta[s]["observation"]),
        }
    return out


def _pick_step(step_scores: Dict[int, Dict[str, Any]], mode: str, rnd: random.Random) -> Optional[int]:
    if not step_scores:
        return None
    items = [(sid, meta["score"]) for sid, meta in step_scores.items()]
    if mode == "top1":
        return max(items, key=lambda x: x[1])[0]
    if mode == "bottom1":
        return min(items, key=lambda x: x[1])[0]
    if mode == "random1":
        return rnd.choice([sid for sid, _ in items])
    raise ValueError(f"Unsupported mode: {mode}")


def _aggregate(result_rows: List[Dict[str, Any]], seed: int = 233) -> Dict[str, Any]:
    modes = ["top1", "bottom1", "random1"]
    rnd = random.Random(seed)
    agg = {m: {"n": 0, "answer_hit": 0, "em_true": 0, "em_true_answer_hit": 0, "avg_score": 0.0} for m in modes}

    for row in result_rows:
        trajectory = row.get("trajectory", [])
        if not isinstance(trajectory, list) or not trajectory:
            continue
        gold = str(row.get("gold_answer", "") or "")
        em = bool(row.get("em", False))
        step_scores = _rebuild_step_scores(trajectory)
        if not step_scores:
            continue
        for mode in modes:
            sid = _pick_step(step_scores, mode, rnd)
            if sid is None:
                continue
            dropped_obs = step_scores[sid]["observation"]
            hit = _answer_hit(dropped_obs, gold)
            agg[mode]["n"] += 1
            agg[mode]["answer_hit"] += int(hit)
            agg[mode]["avg_score"] += float(step_scores[sid]["score"])
            if em:
                agg[mode]["em_true"] += 1
                agg[mode]["em_true_answer_hit"] += int(hit)

    for mode in modes:
        n = max(1, agg[mode]["n"])
        em_n = max(1, agg[mode]["em_true"])
        agg[mode]["answer_hit_rate"] = agg[mode]["answer_hit"] / n
        agg[mode]["em_true_answer_hit_rate"] = agg[mode]["em_true_answer_hit"] / em_n
        agg[mode]["avg_score"] = agg[mode]["avg_score"] / n

    return agg


def _plot_bar(all_metrics: Dict[str, Dict[str, Any]], output_png: str) -> None:
    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    modes = ["top1", "bottom1", "random1"]
    mode_labels = ["Top-1", "Bottom-1", "Random-1"]

    labelsize = 16
    ticksize = 13
    legend_size = 11
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": ticksize,
            "axes.labelsize": labelsize,
            "xtick.labelsize": ticksize,
            "ytick.labelsize": ticksize,
            "legend.fontsize": legend_size,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    labels = list(all_metrics.keys())
    n_groups = len(labels)
    x = list(range(len(modes)))
    total_w = 0.72
    bar_w = total_w / max(1, n_groups)
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2", "#FF9DA6", "#9D755D"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for i, label in enumerate(labels):
        agg = all_metrics[label]
        vals = [agg[m]["answer_hit_rate"] * 100.0 for m in modes]
        vals_em = [agg[m]["em_true_answer_hit_rate"] * 100.0 for m in modes]
        shift = (i - (n_groups - 1) / 2.0) * bar_w
        pos = [v + shift for v in x]
        color = colors[i % len(colors)]
        axes[0].bar(pos, vals, width=bar_w * 0.86, label=label, color=color, edgecolor="white", linewidth=0.6)
        axes[1].bar(pos, vals_em, width=bar_w * 0.86, label=label, color=color, edgecolor="white", linewidth=0.6)

    for ax in axes:
        ax.set_xticks(x, mode_labels, fontsize=ticksize)
        ax.set_ylabel("Hit Rate (%)", fontsize=labelsize)
        ax.tick_params(axis="y", labelsize=ticksize)
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.6)
        ax.set_axisbelow(True)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        ncol=min(4, max(1, n_groups)),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=legend_size,
    )
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)
    plt.savefig(output_png)
    plt.close()


def _write_md(path: str, all_metrics: Dict[str, Dict[str, Any]], meta: Dict[str, Any]) -> None:
    lines = [
        "# StepKV High-Score Step Impact (Proxy)",
        "",
        f"- generated_at_utc: {meta['generated_at_utc']}",
        f"- groups: {', '.join(meta['sources'].keys())}",
        "",
        "| Group | Mode | N | Avg Score | Answer-hit Rate | Answer-hit Rate (EM=True) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for group, agg in all_metrics.items():
        for mode in ("top1", "bottom1", "random1"):
            r = agg[mode]
            lines.append(
                f"| {group} | {mode} | {r['n']} | {r['avg_score']:.4f} | {r['answer_hit_rate']:.2%} | {r['em_true_answer_hit_rate']:.2%} |"
            )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze Step-aware high-score step impact (proxy).")
    parser.add_argument(
        "--result",
        action="append",
        nargs=2,
        metavar=("GROUP", "JSON_PATH"),
        required=True,
        help="Repeatable pair. Example: --result r20 path1.json --result r50 path2.json",
    )
    parser.add_argument("--output_json", default="results/stepkv_score_impact_proxy.json", type=str)
    parser.add_argument("--output_md", default="results/stepkv_score_impact_proxy.md", type=str)
    parser.add_argument("--output_png", default="results/stepkv_score_impact_proxy.png", type=str)
    parser.add_argument("--seed", default=233, type=int)
    args = parser.parse_args()

    all_metrics: Dict[str, Dict[str, Any]] = {}
    sources: Dict[str, Dict[str, Any]] = {}
    for group, path in args.result:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = data["results"] if isinstance(data, dict) and isinstance(data.get("results"), list) else data
        if not isinstance(rows, list):
            raise ValueError(f"Unsupported result format for group={group}: {path}")
        all_metrics[group] = _aggregate(rows, seed=int(args.seed))
        sources[group] = {"path": path, "n_results": len(rows)}

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sources": sources,
        "seed": int(args.seed),
    }
    payload = {"meta": meta, "metrics": all_metrics}

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    _write_md(args.output_md, all_metrics, meta)
    _plot_bar(all_metrics, args.output_png)

    print(f"[INFO] Wrote JSON: {args.output_json}")
    print(f"[INFO] Wrote MD:   {args.output_md}")
    print(f"[INFO] Wrote PNG:  {args.output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
