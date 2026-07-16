#!/usr/bin/env python3
"""
Component ablation for StepKV (step_aware_h2o) on HotpotQA, 2WikiMultihopQA, and MuSiQue.

Default: 500 samples per dataset (seed=42), cache_ratio=0.5, 6 ablation runs + full baseline
imported from existing main experiments (results/*_qwen25_7b_v2/run2/stepaware_r50).

Ablation variants:
  full (imported, not re-run), no_repeat, no_novelty, no_success, no_cite, no_token, no_step

Usage:
    python run_stepkv_component_ablation.py
    python run_stepkv_component_ablation.py --skip_existing
    python run_stepkv_component_ablation.py --run_full_ablation   # force re-run full StepKV
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import run_all_wiki_experiments_v2 as base
import run_all_2wiki_experiments_v2 as runner_2wiki
import run_all_musique_experiments_v2 as runner_musique
from analyze_run_kv_metrics import detect_dataset_suffix, resolve_result_json
from models.model_paths import ensure_local_model_path


DATASET_NAMES = ("hotpotqa", "2wiki", "musique")

# Ablation dataset name -> results folder prefix (see run_qwen25_7b_*_experiments.sh)
DATASET_RESULTS_FOLDER_KEY = {
    "hotpotqa": "wiki",
    "2wiki": "2wiki",
    "musique": "musique",
}

ABLATION_SPECS: Dict[str, Dict[str, Any]] = {
    "full": {
        "label": "Full StepKV",
        "description": "All reward terms + token/step combined score",
        "override": {},
    },
    "no_repeat": {
        "label": "w/o repeat",
        "description": "Set repeat penalty to 0 in r_k",
        "override": {"step_repeat_penalty": 0.0},
    },
    "no_novelty": {
        "label": "w/o novelty",
        "description": "Set novelty term to 0 in r_k",
        "override": {"step_ablate_novelty": True},
    },
    "no_success": {
        "label": "w/o success",
        "description": "Ignore failed-observation penalty (succ_k always 1)",
        "override": {"step_ablate_success": True},
    },
    "no_cite": {
        "label": "w/o cite",
        "description": "Disable Phase-2 citation updates",
        "override": {"step_ablate_citation": True, "step_citation_weight": 0.0},
    },
    "no_token": {
        "label": "w/o token score",
        "description": "alpha=0, retain step score only in C_i",
        "override": {"step_aware_alpha": 0.0, "step_aware_beta": 0.8},
    },
    "no_step": {
        "label": "w/o step score",
        "description": "beta=0, retain H2O token score only in C_i",
        "override": {"step_aware_alpha": 0.8, "step_aware_beta": 0.0},
    },
}


def _save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _load_summary_metrics(result_json: str) -> Tuple[float, float, float]:
    with open(result_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    summary = data.get("summary", {}) if isinstance(data, dict) else {}
    return (
        float(summary.get("exact_match", 0.0)),
        float(summary.get("f1_score", 0.0)),
        float(summary.get("total_time_seconds", 0.0)),
    )


def _prepare_dataset(dataset: str, num_samples: int, seed: int):
    base.NUM_SAMPLES = int(num_samples)
    base.RANDOM_SEED = int(seed)
    if dataset == "hotpotqa":
        val_data = base.load_hotpotqa_data()
        metrics_dataset = "hotpotqa"
    elif dataset == "2wiki":
        val_data = runner_2wiki.load_2wiki_data(runner_2wiki.DEFAULT_2WIKI_LOCAL_PATH)
        metrics_dataset = "2wiki"
    elif dataset == "musique":
        val_data = runner_musique.load_musique_data(runner_musique.DEFAULT_MUSIQUE_LOCAL_PATH)
        metrics_dataset = "musique"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    selected = base.select_samples(val_data)
    from retrievers.WikiBM25Retriever import WikiBM25Retriever

    retriever = WikiBM25Retriever(index_dir=base.WIKI_INDEX_DIR, load_corpus=True)
    return selected, retriever, metrics_dataset


def _stepkv_subdir_for_ratio(cache_ratio: float) -> str:
    pct = int(round(float(cache_ratio) * 100))
    return f"stepaware_r{pct}"


def _ratio_tag(cache_ratio: float) -> str:
    return f"r{int(round(float(cache_ratio) * 100))}"


def _find_full_stepkv_source(
    dataset: str,
    results_root: str,
    model_results_suffix: str,
    run_tag: str,
    cache_ratio: float,
) -> str:
    folder_key = DATASET_RESULTS_FOLDER_KEY.get(dataset)
    if not folder_key:
        raise ValueError(f"Unknown dataset: {dataset}")
    run_dir = os.path.join(results_root, f"{folder_key}_{model_results_suffix}", run_tag)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Main run directory not found: {run_dir}")

    subdir = _stepkv_subdir_for_ratio(cache_ratio)
    dataset_suffix = detect_dataset_suffix(run_dir)
    if not dataset_suffix:
        raise FileNotFoundError(
            f"Cannot detect dataset suffix under {run_dir}. "
            f"Expected react_kv_*.json in {subdir}/ or sibling folders."
        )
    source = resolve_result_json(
        run_dir,
        subdir,
        "react_kv_step_aware_h2o",
        dataset_suffix,
        _ratio_tag(cache_ratio),
    )
    if not source:
        raise FileNotFoundError(
            f"Full StepKV result not found under {run_dir}/{subdir}/ "
            f"(react_kv_step_aware_h2o_{dataset_suffix}*.json)"
        )
    return source


def _import_full_baseline(source_json: str, out_dir: str, cache_ratio: float) -> Dict[str, Any]:
    """Copy existing full StepKV result into ablation output tree."""
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "result.json")
    shutil.copy2(source_json, out_json)

    out_ckpt = os.path.join(out_dir, "result_checkpoint.json")
    src_dir = os.path.dirname(source_json)
    ckpt_candidates = sorted(glob.glob(os.path.join(src_dir, "*checkpoint*.json")))
    if ckpt_candidates:
        shutil.copy2(ckpt_candidates[0], out_ckpt)

    em, f1, total_time = _load_summary_metrics(out_json)
    spec = ABLATION_SPECS["full"]
    return {
        "ablation": "full",
        "label": spec["label"],
        "description": spec["description"],
        "em": em,
        "f1": f1,
        "time_s": total_time,
        "output_json": out_json,
        "checkpoint_json": out_ckpt if os.path.isfile(out_ckpt) else "",
        "kv_override": _base_kv_override(cache_ratio),
        "source_json": os.path.abspath(source_json),
        "imported_from_main_run": True,
        "skipped_rerun": True,
    }


def _base_kv_override(cache_ratio: float) -> Dict[str, Any]:
    return {
        "cache_ratio": float(cache_ratio),
        "attn_mode": "piggyback",
        "observation_window": 0,
        "step_poolwise_prune": True,
    }


def _run_ablation(
    selected_samples,
    retriever,
    out_dir: str,
    ablation_name: str,
    cache_ratio: float,
    metrics_dataset: str,
) -> Dict[str, Any]:
    spec = ABLATION_SPECS[ablation_name]
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, "result.json")
    ckpt_json = os.path.join(out_dir, "result_checkpoint.json")

    kv_override = _base_kv_override(cache_ratio)
    kv_override.update(spec["override"])
    kv_override["step_ablation_name"] = ablation_name

    em, f1, total_time = base.run_react_kv_experiment(
        val_data=None,
        selected_samples=selected_samples,
        retriever=retriever,
        pruning_mode="step_aware_h2o",
        output_path=out_json,
        checkpoint_path=ckpt_json,
        kv_config_override=kv_override,
        metrics_dataset=metrics_dataset,
        metrics_method=f"stepkv_ablate_{ablation_name}",
    )
    return {
        "ablation": ablation_name,
        "label": spec["label"],
        "description": spec["description"],
        "em": float(em),
        "f1": float(f1),
        "time_s": float(total_time),
        "output_json": out_json,
        "checkpoint_json": ckpt_json,
        "kv_override": kv_override,
    }


def _attach_deltas(rows: List[Dict[str, Any]], full_row: Dict[str, Any]) -> None:
    for row in rows:
        row["delta_em"] = row["em"] - full_row["em"]
        row["delta_f1"] = row["f1"] - full_row["f1"]


def _write_summary_md(summary: Dict[str, Any], output_md: str) -> None:
    lines = [
        "# StepKV component ablation",
        "",
        f"- generated_at_utc: {summary['generated_at_utc']}",
        f"- seed: {summary['seed']}",
        f"- num_samples: {summary['num_samples']}",
        f"- cache_ratio: {summary['cache_ratio']}",
        f"- datasets: {', '.join(summary['datasets'].keys())}",
        "",
    ]
    for ds, ds_block in summary["datasets"].items():
        full = ds_block["baseline"]
        lines.extend([
            f"## {ds}",
            "",
            f"Full StepKV: EM={full['em']:.2f}, F1={full['f1']:.2f}",
            "",
            "| Ablation | Description | EM | F1 | ΔEM | ΔF1 | Time(s) |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        for row in ds_block["table_rows"]:
            lines.append(
                f"| {row['label']} | {row['description']} | "
                f"{row['em']:.2f} | {row['f1']:.2f} | "
                f"{row['delta_em']:+.2f} | {row['delta_f1']:+.2f} | {row['time_s']:.1f} |"
            )
        lines.append("")

    lines.extend([
        "## Combined (all datasets)",
        "",
        "| Dataset | Ablation | EM | F1 | ΔEM | ΔF1 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for ds, ds_block in summary["datasets"].items():
        for row in ds_block["table_rows"]:
            lines.append(
                f"| {ds} | {row['label']} | {row['em']:.2f} | {row['f1']:.2f} | "
                f"{row['delta_em']:+.2f} | {row['delta_f1']:+.2f} |"
            )
    os.makedirs(os.path.dirname(output_md) or ".", exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _dataset_display_name(ds: str) -> str:
    return {
        "hotpotqa": "HotpotQA",
        "2wiki": "2WikiMultihopQA",
        "musique": "MuSiQue",
    }.get(ds, ds)


def _write_summary_tex(summary: Dict[str, Any], output_tex: str) -> None:
    """Long-form table: one block per dataset with EM/F1/deltas."""
    n = summary["num_samples"]
    rho = summary["cache_ratio"]
    lines = [
        r"% Auto-generated by run_stepkv_component_ablation.py",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\caption{{Component ablation of StepKV ($N={n}$ per dataset, KV budget $\rho={rho:.2f}$). "
        r"$\Delta$EM and $\Delta$F1 are relative to full StepKV on the same dataset.}}",
        r"\label{tab:stepkv-ablation-detail}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Dataset & Variant & EM & F1 & $\Delta$EM & $\Delta$F1 \\",
        r"\midrule",
    ]
    for ds, ds_block in summary["datasets"].items():
        ds_tex = _dataset_display_name(ds).replace("_", r"\_")
        for row in ds_block["table_rows"]:
            label = row["label"].replace("_", r"\_")
            if row["ablation"] == "full":
                dem = r"--"
                df1 = r"--"
            else:
                dem = f"{row['delta_em']:+.2f}"
                df1 = f"{row['delta_f1']:+.2f}"
            lines.append(
                f"{ds_tex} & {label} & {row['em']:.2f} & {row['f1']:.2f} & "
                f"{dem} & {df1} \\\\"
            )
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")
    lines.extend([r"\end{tabular}", r"\end{table*}"])
    os.makedirs(os.path.dirname(output_tex) or ".", exist_ok=True)
    with open(output_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_summary_tex_compact(summary: Dict[str, Any], output_tex: str) -> None:
    """Paper-ready compact table: rows=variants, columns=dataset EM/F1."""
    datasets = list(summary["datasets"].keys())
    n = summary["num_samples"]
    rho = summary["cache_ratio"]
    ncol = 2 * len(datasets)
    col_spec = "l" + "cc" * len(datasets)

    lines = [
        r"% Auto-generated by run_stepkv_component_ablation.py (compact, for paper main text)",
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{5pt}",
        rf"\caption{{Component ablation of StepKV on three benchmarks ($N={n}$, $\rho={rho:.2f}$). "
        r"Each cell reports EM / F1 (\%). Best full-model results are on the \textit{Full StepKV} row.}}",
        r"\label{tab:stepkv-ablation}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
    ]

    header_parts = ["Variant"]
    cmidrules = []
    for i, ds in enumerate(datasets):
        name = _dataset_display_name(ds)
        start = 2 + 2 * i
        end = start + 1
        cmidrules.append(rf"\cmidrule(lr){{{start}-{end}}}")
        header_parts.extend([rf"\multicolumn{{2}}{{c}}{{{name}}}"])
    lines.append(" & ".join(header_parts) + r" \\")
    lines.extend(cmidrules)

    subheader = [" "]
    for _ in datasets:
        subheader.extend(["EM", "F1"])
    lines.append(" & ".join(subheader) + r" \\")
    lines.append(r"\midrule")

    ablation_order = summary.get("ablation_order") or list(ABLATION_SPECS.keys())
    for ab_name in ablation_order:
        label = ABLATION_SPECS[ab_name]["label"].replace("_", r"\_")
        if ab_name == "full":
            row_label = r"\textbf{Full StepKV}"
        else:
            row_label = label
        cells = [row_label]
        for ds in datasets:
            ds_block = summary["datasets"][ds]
            row = ds_block["ablations"][ab_name]
            cells.append(f"{row['em']:.2f}")
            cells.append(f"{row['f1']:.2f}")
        lines.append(" & ".join(cells) + r" \\")

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
        "",
        r"% Optional delta table (paste below if reviewers ask for drops):",
        r"% \begin{table}[t]",
        r"% \centering",
        r"% \small",
        r"% \caption{Ablation drops ($\Delta$EM) relative to full StepKV.}",
        r"% \begin{tabular}{lccc}",
        r"% \toprule",
        r"% Variant & HotpotQA & 2WikiMultihopQA & MuSiQue \\",
        r"% \midrule",
    ])
    for ab_name in ablation_order:
        if ab_name == "full":
            continue
        label = ABLATION_SPECS[ab_name]["label"].replace("_", r"\_")
        deltas = []
        for ds in datasets:
            row = summary["datasets"][ds]["ablations"][ab_name]
            full = summary["datasets"][ds]["ablations"]["full"]
            deltas.append(f"{row['em'] - full['em']:+.2f}")
        lines.append(r"% " + label + " & " + " & ".join(deltas) + r" \\")
    lines.extend([
        r"% \bottomrule",
        r"% \end{tabular}",
        r"% \end{table}",
    ])

    os.makedirs(os.path.dirname(output_tex) or ".", exist_ok=True)
    with open(output_tex, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run StepKV component ablation on HotpotQA / 2Wiki / MuSiQue."
    )
    parser.add_argument(
        "--output_root",
        default="results/stepkv_component_ablation",
        type=str,
    )
    parser.add_argument("--num_samples", default=500, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--cache_ratio", default=0.5, type=float)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DATASET_NAMES),
        choices=list(DATASET_NAMES),
    )
    parser.add_argument(
        "--ablations",
        nargs="+",
        default=list(ABLATION_SPECS.keys()),
        choices=list(ABLATION_SPECS.keys()),
    )
    parser.add_argument(
        "--model_label",
        default="",
        type=str,
        help="Optional display name stored in summary.json (e.g., Qwen2.5-7B).",
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
        help="If result json exists, load metrics instead of re-running (non-full ablations).",
    )
    parser.add_argument(
        "--run_full_ablation",
        action="store_true",
        help="Re-run full StepKV instead of importing from main experiment run2.",
    )
    parser.add_argument(
        "--refresh_full_import",
        action="store_true",
        help="Re-copy full StepKV from main run even if output already exists.",
    )
    parser.add_argument(
        "--results_root",
        default="results",
        type=str,
        help="Root directory containing wiki_qwen25_7b_v2/run2/ etc.",
    )
    parser.add_argument(
        "--run_tag",
        default="run2",
        type=str,
        help="Subfolder under each dataset results dir (default: run2, seed=42).",
    )
    parser.add_argument(
        "--model_results_suffix",
        default="qwen25_7b_v2",
        type=str,
        help="Results folder suffix, e.g. qwen25_7b_v2 or llama31_8b_v2.",
    )
    args = parser.parse_args()

    reuse_full_from_run = not args.run_full_ablation

    base.MODEL_PATH = ensure_local_model_path(
        args.model_path,
        model_family=args.model_family,
        allow_download=not args.no_download_model,
    )
    print(f"[INFO] Using model: {base.MODEL_PATH}")

    ordered_ablations = list(dict.fromkeys(args.ablations))
    if "full" not in ordered_ablations:
        ordered_ablations = ["full"] + ordered_ablations

    summary: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seed": int(args.seed),
        "num_samples": int(args.num_samples),
        "cache_ratio": float(args.cache_ratio),
        "model_path": base.MODEL_PATH,
        "model_label": args.model_label or base.MODEL_PATH,
        "ablation_order": ordered_ablations,
        "reuse_full_from_run": reuse_full_from_run,
        "full_baseline_run_tag": args.run_tag,
        "full_baseline_results_root": args.results_root,
        "full_baseline_model_suffix": args.model_results_suffix,
        "datasets": {},
    }

    for dataset in args.datasets:
        print(f"\n[INFO] ===== Dataset: {dataset} =====")
        selected, retriever, metrics_dataset = _prepare_dataset(
            dataset, args.num_samples, args.seed
        )
        ds_block: Dict[str, Any] = {"ablations": {}, "table_rows": []}

        for name in ordered_ablations:
            out_dir = os.path.join(args.output_root, dataset, name)
            out_json = os.path.join(out_dir, "result.json")
            spec = ABLATION_SPECS[name]

            if name == "full" and reuse_full_from_run:
                if (
                    args.refresh_full_import
                    or not args.skip_existing
                    or not os.path.isfile(out_json)
                ):
                    source_json = _find_full_stepkv_source(
                        dataset=dataset,
                        results_root=args.results_root,
                        model_results_suffix=args.model_results_suffix,
                        run_tag=args.run_tag,
                        cache_ratio=args.cache_ratio,
                    )
                    print(f"[INFO] [{dataset}] import full from {source_json}")
                    row = _import_full_baseline(source_json, out_dir, args.cache_ratio)
                elif os.path.isfile(out_json):
                    em, f1, t = _load_summary_metrics(out_json)
                    row = {
                        "ablation": name,
                        "label": spec["label"],
                        "description": spec["description"],
                        "em": em,
                        "f1": f1,
                        "time_s": t,
                        "output_json": out_json,
                        "checkpoint_json": os.path.join(out_dir, "result_checkpoint.json"),
                        "skipped_rerun": True,
                        "imported_from_main_run": True,
                    }
                    print(f"[INFO] [{dataset}] loaded existing full: EM={em:.2f}, F1={f1:.2f}")
                else:
                    raise FileNotFoundError(
                        f"[{dataset}] full baseline missing and reuse_full_from_run=True"
                    )
            elif args.skip_existing and os.path.isfile(out_json):
                em, f1, t = _load_summary_metrics(out_json)
                row = {
                    "ablation": name,
                    "label": spec["label"],
                    "description": spec["description"],
                    "em": em,
                    "f1": f1,
                    "time_s": t,
                    "output_json": out_json,
                    "checkpoint_json": os.path.join(out_dir, "result_checkpoint.json"),
                    "skipped_rerun": True,
                }
                print(f"[INFO] [{dataset}] loaded {name}: EM={em:.2f}, F1={f1:.2f}")
            else:
                print(f"[INFO] [{dataset}] running {name} ...")
                row = _run_ablation(
                    selected_samples=selected,
                    retriever=retriever,
                    out_dir=out_dir,
                    ablation_name=name,
                    cache_ratio=args.cache_ratio,
                    metrics_dataset=metrics_dataset,
                )
                row["skipped_rerun"] = False
                print(
                    f"[INFO] [{dataset}] finished {name}: "
                    f"EM={row['em']:.2f}, F1={row['f1']:.2f}, time={row['time_s']:.1f}s"
                )

            ds_block["ablations"][name] = row
            ds_block["table_rows"].append(row)
            summary["datasets"][dataset] = ds_block
            _save_json(os.path.join(args.output_root, "summary.json"), summary)

        full_row = ds_block["ablations"]["full"]
        _attach_deltas(ds_block["table_rows"], full_row)
        ds_block["baseline"] = {
            "ablation": "full",
            "em": full_row["em"],
            "f1": full_row["f1"],
        }
        summary["datasets"][dataset] = ds_block
        _save_json(os.path.join(args.output_root, "summary.json"), summary)

    out_json = os.path.join(args.output_root, "summary.json")
    out_md = os.path.join(args.output_root, "ablation_table.md")
    out_tex = os.path.join(args.output_root, "ablation_table_compact.tex")
    out_tex_detail = os.path.join(args.output_root, "ablation_table_detail.tex")
    _save_json(out_json, summary)
    _write_summary_md(summary, out_md)
    _write_summary_tex_compact(summary, out_tex)
    _write_summary_tex(summary, out_tex_detail)
    print(f"\n[INFO] Saved summary json:   {out_json}")
    print(f"[INFO] Saved summary md:     {out_md}")
    print(f"[INFO] Saved paper tex:      {out_tex}")
    print(f"[INFO] Saved detail tex:     {out_tex_detail}")


if __name__ == "__main__":
    main()
