"""Summarize completed main runs and exact-ID paired uncertainty estimates."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_rebuttal_schedule import atomic_json


def paired_stats(left, right):
    a = {r["id"]: r for r in left}
    b = {r["id"]: r for r in right}
    if a.keys() != b.keys():
        raise ValueError("Paired comparison requires exactly the same question IDs")
    ids = sorted(a)
    output = {"n": len(ids)}
    rng = np.random.default_rng(20261013)
    indices = rng.integers(0, len(ids), size=(10000, len(ids)))
    for metric in ("em", "f1"):
        delta = np.array([a[i][metric] - b[i][metric] for i in ids]) * 100
        ci = np.quantile(delta[indices].mean(axis=1), [.025, .975])
        output[metric] = {"difference_percentage_points": float(delta.mean()),
                          "paired_bootstrap_95ci": ci.tolist()}
    wins = sum(a[i]["em"] > b[i]["em"] for i in ids)
    losses = sum(a[i]["em"] < b[i]["em"] for i in ids)
    output["em"]["discordant_counts"] = [wins, losses]
    output["em"]["mcnemar_exact_p_unadjusted"] = float(binomtest(wins, wins + losses).pvalue) if wins + losses else 1.0
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    runs = {}
    for file in sorted(args.root.glob("seed*/r*/*/results.json")):
        status = json.loads((file.parent / "status.json").read_text())
        if status.get("status") != "complete":
            continue
        manifest = json.loads((file.parent / "manifest.json").read_text())["fingerprint"]
        result = json.loads(file.read_text())
        key = (manifest["seed"], manifest["config_override"]["cache_ratio"], manifest["method"])
        runs[key] = (result, manifest)
    comparisons = {}
    pairs = [("stepkv", method) for method in ["fullkv", "h2o_token", "h2o_step", "tova_token", "tova_step", "stepkv_no_step_score"]]
    pairs += [("h2o_step", "h2o_token"), ("tova_step", "tova_token")]
    for (seed, ratio, method), (result, manifest) in runs.items():
        for left, other in pairs:
            if method == left:
                baseline = runs.get((seed, .2 if other == "fullkv" else ratio, other))
                if baseline:
                    if baseline[1]["source_hashes"] != manifest["source_hashes"]:
                        raise ValueError("Source mismatch in paired comparison")
                    comparisons[f"seed{seed}/r{ratio}/{method}-minus-{other}"] = paired_stats(result["results"], baseline[0]["results"])
    lines = ["# Rebuttal controlled experiment status", "",
             "Only completed main runs are included. Smoke runs are excluded. "
             "Timing is diagnostic on a shared GPU. Sampling seeds select different question subsets; "
             "they are not stochastic decoding replications.", "",
             "| Seed | Ratio | Method | N | EM (%) | F1 (%) | Mean live peak KV (MiB) |",
             "|---|---|---|---:|---:|---:|---:|"]
    for (seed, ratio, method), (result, _) in sorted(runs.items()):
        summary = result["summary"]
        peak = np.mean([r["runtime_audit"]["peak_cache_bytes"] / 2**20 for r in result["results"]])
        lines.append(f"| {seed} | {ratio} | {method} | {len(result['results'])} | {summary['exact_match']:.2f} | {summary['f1_score']:.2f} | {peak:.1f} |")
    aggregates = {}
    for (seed, ratio, method), (result, _) in sorted(runs.items()):
        group = aggregates.setdefault(f"r{ratio}/{method}", {"seeds": [], "em": [], "f1": []})
        group["seeds"].append(seed)
        group["em"].append(result["summary"]["exact_match"])
        group["f1"].append(result["summary"]["f1_score"])
    lines += ["", "| Ratio / Method | Completed seeds | EM mean ± sample SD (%) | F1 mean ± sample SD (%) |",
              "|---|---|---:|---:|"]
    for name, group in aggregates.items():
        cells = []
        for metric in ("em", "f1"):
            values = group[metric]
            mean = float(np.mean(values))
            sd = float(np.std(values, ddof=1)) if len(values) > 1 else None
            group[f"{metric}_mean"] = mean
            group[f"{metric}_sample_sd"] = sd
            cells.append(f"{mean:.2f} ± {sd:.2f}" if sd is not None else f"{mean:.2f} (SD not applicable (single seed))")
        lines.append(f"| {name} | {group['seeds']} | {cells[0]} | {cells[1]} |")
    lines += ["", "Paired differences and 95% bootstrap CIs are in `paired_statistics.json`. "
              "Question-level paired bootstrap remains applicable for a single seed. "
              "McNemar p-values are explicitly unadjusted; no cross-seed pooling or significance claims are made."]
    atomic_json(args.root / "paired_statistics.json", comparisons)
    atomic_json(args.root / "seed_aggregates.json", aggregates)
    atomic_json(args.root / "analysis_metadata.json", {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "bootstrap_repetitions": 10000, "bootstrap_seed": 20261013,
        "ci": "percentile paired bootstrap, per sampling seed and ratio",
        "cross_seed_sd": "not applicable for a single seed; otherwise descriptive sample SD, ddof=1; overlapping question subsets are not independent decoding repeats",
        "mcnemar_p": "unadjusted, no significance claims",
    })
    (args.root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"Summarized {len(runs)} completed runs and {len(comparisons)} paired comparisons")


if __name__ == "__main__":
    main()
