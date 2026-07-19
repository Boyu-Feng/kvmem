#!/usr/bin/env python3
"""
StepKV continuous step-interruption ablation on 2WikiMultihopQA.

Simulates hard loss of entire step KV spans during ongoing ReAct inference.
Each interrupt mode force-drops whole steps at step boundaries (no backfill).

Modes (step_interrupt_mode):
  none     - Full StepKV baseline (no forced drops)
  lag1     - Enter step t: drop step t-1
  lag2     - Enter step t: drop step t-2  (enter step 3 -> drop step 1)
  lag3     - Enter step t: drop step t-3
  window2  - Enter step t: drop all steps with id < t-2

Usage:
    python run_stepkv_interrupt_ablation_2wiki.py
    python run_stepkv_interrupt_ablation_2wiki.py --num_samples 100 --seed 42
    python run_stepkv_interrupt_ablation_2wiki.py --modes none lag2 --skip_existing

Output:
    results/stepkv_interrupt_ablation_2wiki/
      none/result.json
      lag2/result.json
      summary.json
      ablation_table.md
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List

import run_all_wiki_experiments_v2 as base
import run_all_2wiki_experiments_v2 as runner_2wiki
from kv_cache.step_interrupt import INTERRUPT_MODES, describe_interrupt_mode
from models.model_paths import ensure_local_model_path


DEFAULT_MODES = ("none", "lag1", "lag2", "window2")


def _save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_metrics(result_json: str) -> Dict[str, float]:
    with open(result_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    summary = data.get("summary", {}) if isinstance(data, dict) else {}
    return {
        "em": float(summary.get("exact_match", 0.0)),
        "f1": float(summary.get("f1_score", 0.0)),
        "time_s": float(summary.get("total_time_seconds", 0.0)),
    }


def _prepare_2wiki(num_samples: int, seed: int):
    base.NUM_SAMPLES = int(num_samples)
    base.RANDOM_SEED = int(seed)
    val_data = runner_2wiki.load_2wiki_data(runner_2wiki.DEFAULT_2WIKI_LOCAL_PATH)
    selected = base.select_samples(val_data)
    from retrievers.WikiBM25Retriever import WikiBM25Retriever

    retriever = WikiBM25Retriever(index_dir=base.WIKI_INDEX_DIR, load_corpus=True)
    return selected, retriever


def _run_one(
    selected_samples,
    retriever,
    out_json: str,
    ckpt_json: str,
    interrupt_mode: str,
    cache_ratio: float,
) -> Dict[str, Any]:
    kv_override = {
        "cache_ratio": float(cache_ratio),
        "attn_mode": "piggyback",
        "observation_window": 0,
        "step_poolwise_prune": True,
        "step_interrupt_mode": interrupt_mode,
        "step_force_drop_map": {},
    }
    em, f1, total_time = base.run_react_kv_experiment(
        val_data=None,
        selected_samples=selected_samples,
        retriever=retriever,
        pruning_mode="step_aware_h2o",
        output_path=out_json,
        checkpoint_path=ckpt_json,
        kv_config_override=kv_override,
        metrics_dataset="2wiki",
        metrics_method=f"stepkv_interrupt_{interrupt_mode}",
    )
    return {
        "interrupt_mode": interrupt_mode,
        "description": describe_interrupt_mode(interrupt_mode),
        "em": float(em),
        "f1": float(f1),
        "time_s": float(total_time),
        "output_json": out_json,
        "checkpoint_json": ckpt_json,
    }


def _write_md(summary: Dict[str, Any], path: str) -> None:
    baseline = summary["runs"]["none"]
    lines = [
        "# StepKV step-interruption ablation (2WikiMultihopQA)",
        "",
        f"- generated_at_utc: {summary['generated_at_utc']}",
        f"- seed: {summary['seed']}",
        f"- num_samples: {summary['num_samples']}",
        f"- cache_ratio: {summary['cache_ratio']}",
        f"- model: {summary['model_path']}",
        "",
        f"Baseline (none): EM={baseline['em']:.2f}, F1={baseline['f1']:.2f}",
        "",
        "| Mode | Description | EM | F1 | ΔEM | ΔF1 | Time(s) |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for mode in summary["mode_order"]:
        row = summary["runs"][mode]
        if mode == "none":
            dem = "--"
            df1 = "--"
        else:
            dem = f"{row['em'] - baseline['em']:+.2f}"
            df1 = f"{row['f1'] - baseline['f1']:+.2f}"
        desc = row["description"].replace("|", "\\|")
        lines.append(
            f"| {mode} | {desc} | {row['em']:.2f} | {row['f1']:.2f} | "
            f"{dem} | {df1} | {row['time_s']:.1f} |"
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run StepKV step-interruption ablation on 2WikiMultihopQA."
    )
    parser.add_argument(
        "--output_root",
        default="results/stepkv_interrupt_ablation_2wiki",
        type=str,
    )
    parser.add_argument("--num_samples", default=100, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--cache_ratio", default=0.5, type=float)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=list(DEFAULT_MODES),
        choices=list(INTERRUPT_MODES),
    )
    parser.add_argument("--model_path", default="auto", type=str)
    parser.add_argument(
        "--model_family",
        choices=["auto", "qwen", "llama"],
        default="auto",
    )
    parser.add_argument("--no_download_model", action="store_true")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip mode if result.json already exists.",
    )
    args = parser.parse_args()

    base.MODEL_PATH = ensure_local_model_path(
        args.model_path,
        model_family=args.model_family,
        allow_download=not args.no_download_model,
    )
    print(f"[INFO] Using model: {base.MODEL_PATH}")

    ordered_modes = list(dict.fromkeys(args.modes))
    if "none" not in ordered_modes:
        ordered_modes = ["none"] + ordered_modes

    selected, retriever = _prepare_2wiki(args.num_samples, args.seed)

    summary: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "dataset": "2wiki",
        "seed": int(args.seed),
        "num_samples": int(args.num_samples),
        "cache_ratio": float(args.cache_ratio),
        "model_path": base.MODEL_PATH,
        "mode_order": ordered_modes,
        "runs": {},
    }

    for mode in ordered_modes:
        out_dir = os.path.join(args.output_root, mode)
        out_json = os.path.join(out_dir, "result.json")
        ckpt_json = os.path.join(out_dir, "result_checkpoint.json")

        if args.skip_existing and os.path.isfile(out_json):
            metrics = _load_metrics(out_json)
            row = {
                "interrupt_mode": mode,
                "description": describe_interrupt_mode(mode),
                "em": metrics["em"],
                "f1": metrics["f1"],
                "time_s": metrics["time_s"],
                "output_json": out_json,
                "checkpoint_json": ckpt_json,
                "skipped_rerun": True,
            }
            print(
                f"[INFO] [{mode}] loaded existing: EM={row['em']:.2f}, F1={row['f1']:.2f}"
            )
        else:
            print(f"\n[INFO] ===== interrupt_mode={mode} =====")
            print(f"[INFO] {describe_interrupt_mode(mode)}")
            row = _run_one(
                selected_samples=selected,
                retriever=retriever,
                out_json=out_json,
                ckpt_json=ckpt_json,
                interrupt_mode=mode,
                cache_ratio=args.cache_ratio,
            )
            row["skipped_rerun"] = False
            print(
                f"[INFO] [{mode}] finished: EM={row['em']:.2f}, F1={row['f1']:.2f}, "
                f"time={row['time_s']:.1f}s"
            )

        summary["runs"][mode] = row
        _save_json(os.path.join(args.output_root, "summary.json"), summary)

    baseline = summary["runs"]["none"]
    for mode, row in summary["runs"].items():
        row["delta_em"] = row["em"] - baseline["em"]
        row["delta_f1"] = row["f1"] - baseline["f1"]

    out_json = os.path.join(args.output_root, "summary.json")
    out_md = os.path.join(args.output_root, "ablation_table.md")
    _save_json(out_json, summary)
    _write_md(summary, out_md)
    print(f"\n[INFO] Saved summary: {out_json}")
    print(f"[INFO] Saved table:   {out_md}")


if __name__ == "__main__":
    main()
