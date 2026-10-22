"""Read-only validation and paired reporting for completed QA latency conditions."""
from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
import random
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.latency_runtime import read_json
from scripts.run_latency_experiment import condition_summary, validate_committed
from scripts.run_qa32_rebuttal import atomic_json

MIB = 1024 ** 2
COMPARABLE = (
    "schema_version", "model", "dataset", "seed", "sample_ids", "warmup_ids",
    "clean_repeats_per_question", "diagnostic_repeats_per_question", "max_steps",
    "max_new_tokens_per_block", "dtype", "hardware", "versions", "cpu_threads",
    "source_hashes", "assets", "comparison", "measurement_scope", "generation_config",
)


def mean(values):
    values = list(values)
    if not values:
        raise ValueError("Cannot summarize an empty measurement sequence")
    return statistics.mean(values)


def allocated(memory, key):
    devices = memory["allocator"]
    if not devices:
        raise RuntimeError("Completed GPU experiments must include allocator measurements")
    return sum(device[key] for device in devices.values())


def bootstrap_paired(values, repeats=2000, seed=233):
    if not values or repeats < 100:
        raise ValueError("Paired bootstrap requires measured pairs and at least 100 resamples")
    rng = random.Random(seed)
    draws = sorted(mean(rng.choices(values, k=len(values))) for _ in range(repeats))
    return {"mean": mean(values), "ci95": [draws[int(.025 * (repeats - 1))], draws[int(.975 * (repeats - 1))]],
            "questions": len(values), "resamples": repeats, "bootstrap_seed": seed,
            "unit": "paired question means, not greedy timing repeats"}


def validate_run(directory):
    status = read_json(directory / "status.json")
    if status["status"] != "complete":
        raise RuntimeError(f"Latency condition is not complete; excluded from comparisons: {directory}")
    protocol = read_json(directory / "manifest.json")["protocol"]
    measurements = read_json(directory / "measurements.json")
    rows = validate_committed(measurements, protocol)
    if len(rows) != len(protocol["sample_ids"]) or not rows:
        raise RuntimeError(f"Incomplete completed-condition IDs: {directory}")
    if measurements["summary"] != condition_summary(rows):
        raise RuntimeError(f"Summary differs from committed measurements: {directory}")
    return protocol, rows


def summarize_condition(protocol, rows):
    result = {
        "method": protocol["method"], "ratio": protocol["ratio"], "questions": len(rows),
        "budget_unit": "slots_per_query_head" if protocol["method"] == "lazyeviction" else "trajectory_logical_tokens",
        "byte_matched": False,
        **condition_summary(rows),
        "mean_episode_excluding_tool_seconds": mean(
            mean(repeat["episode_excluding_tool_seconds"] for repeat in row["clean_repeats"]) for row in rows),
        "mean_tool_seconds": mean(mean(repeat["tool_seconds"] for repeat in row["clean_repeats"]) for row in rows),
        "mean_generation_blocks": mean(row["generation_blocks"] for row in rows),
        "mean_final_kv_bytes": mean(row["diagnostics"]["final"]["kv"]["total_bytes"] for row in rows),
        "mean_final_trajectory_kv_bytes": mean(row["diagnostics"]["final"]["kv"]["trajectory_bytes"] for row in rows),
        "mean_final_slots_per_head": mean(row["diagnostics"]["final"]["kv"]["tokens_per_head"] for row in rows),
        "mean_final_allocated_above_model_bytes": mean(
            allocated(row["diagnostics"]["final"], "allocated_above_model_bytes") for row in rows),
        "mean_final_driver_above_model_bytes": mean(
            allocated(row["diagnostics"]["final"], "driver_above_model_bytes") for row in rows),
        "mean_clean_peak_allocated_above_model_bytes": mean(
            mean(repeat["memory"]["peak_allocated_above_model_bytes"] for repeat in row["clean_repeats"]) for row in rows),
        "max_clean_peak_allocated_above_model_bytes": max(
            repeat["memory"]["peak_allocated_above_model_bytes"] for row in rows for repeat in row["clean_repeats"]),
        "mean_sampled_peak_active_kv_bytes": mean(row["diagnostics"]["peak_active_kv"]["total_bytes"] for row in rows),
        "max_sampled_peak_active_kv_bytes": max(row["diagnostics"]["peak_active_kv"]["total_bytes"] for row in rows),
    }
    for endpoint, key in (("generation", "generation_boundaries"), ("interaction", "interaction_boundaries")):
        boundaries = ([step["memory"] for step in row["diagnostics"]["steps"]] if endpoint == "generation"
                      else [step["memory"] for step in row["interaction_boundaries"]]
                      for row in rows)
        result["mean_question_mean_" + key + "_kv_bytes"] = mean(
            mean(memory["kv"]["total_bytes"] for memory in question) for question in boundaries)
        boundaries = ([step["memory"] for step in row["diagnostics"]["steps"]] if endpoint == "generation"
                      else [step["memory"] for step in row["interaction_boundaries"]]
                      for row in rows)
        result["mean_question_mean_" + key + "_allocated_above_model_bytes"] = mean(
            mean(allocated(memory, "allocated_above_model_bytes") for memory in question) for question in boundaries)
    phases = sorted({phase for row in rows for phase in row["diagnostics"]["timing"]["exclusive_seconds"]})
    result["diagnostic_mean_exclusive_seconds"] = {
        phase: mean(row["diagnostics"]["timing"]["exclusive_seconds"].get(phase, 0.0) for row in rows)
        for phase in phases
    }
    result["diagnostic_mean_unattributed_seconds"] = mean(row["diagnostic_unattributed_seconds"] for row in rows)
    workload_phases = sorted({phase for row in rows for phase in row["diagnostics"]["workload"]})
    result["diagnostic_mean_workload"] = {
        phase: {key: mean(row["diagnostics"]["workload"].get(phase, {}).get(key, 0) for row in rows)
                for key in ("forward_calls", "input_positions")}
        for phase in workload_phases
    }
    result["workload_scope"] = (
        "Actual forwarded input positions, including later stop-discarded work; "
        "not emitted tokens or retained KV. Free rollouts are not matched-work kernel benchmarks."
    )
    expansions = {layer["actual_heads"] / layer["native_heads"]
                  for row in rows for layer in row["diagnostics"]["final"]["kv"]["layout"]}
    result["head_expansion_factors"] = sorted(expansions)
    result["uses_native_kv_head_count"] = expansions == {1.0}
    result["comparison_scope"] = (
        "Actual end-to-end free rollouts; same IDs do not imply same steps/generated work. "
        "Same slots are NOT same bytes for expanded per-head KV. No same-byte claim."
    )
    return result


def boundary_csv(protocol, rows):
    stream = io.StringIO()
    fields = ["method", "ratio", "id", "endpoint", "step", "tokens_per_head", "kv_bytes",
              "protected_bytes", "trajectory_bytes", "allocated_above_model_bytes",
              "reserved_above_model_bytes", "driver_above_model_bytes"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for row in rows:
        memories = [("generation", index + 1, step["memory"])
                    for index, step in enumerate(row["diagnostics"]["steps"])]
        memories += [(step["endpoint"], step["step"], step["memory"])
                     for step in row["interaction_boundaries"]]
        memories.append(("episode_final", row["generation_blocks"], row["diagnostics"]["final"]))
        for endpoint, step, memory in memories:
            kv = memory["kv"]
            writer.writerow({
                "method": protocol["method"], "ratio": protocol["ratio"], "id": row["id"],
                "endpoint": endpoint, "step": step, "tokens_per_head": kv["tokens_per_head"],
                "kv_bytes": kv["total_bytes"], "protected_bytes": kv["protected_bytes"],
                "trajectory_bytes": kv["trajectory_bytes"],
                **{key: allocated(memory, key) for key in (
                    "allocated_above_model_bytes", "reserved_above_model_bytes", "driver_above_model_bytes")},
            })
    return stream.getvalue()


def summarize_runs(directories, output, bootstrap_samples=2000):
    validated = [validate_run(Path(directory)) for directory in directories]
    if not validated:
        raise ValueError("At least one complete measurement directory is required")
    reference = validated[0][0]
    keys = [(protocol["method"], protocol["ratio"]) for protocol, _ in validated]
    if len(keys) != len(set(keys)):
        raise RuntimeError("Duplicate method/budget condition")
    for protocol, _ in validated:
        for key in COMPARABLE:
            if protocol[key] != reference[key]:
                raise RuntimeError(f"Incomparable latency protocols: {key}")
    full = [(protocol, rows) for protocol, rows in validated if protocol["method"] == "fullkv"]
    if len(full) != 1:
        raise RuntimeError("Exactly one complete, same-protocol FullKV reference is required")
    baseline_protocol, baseline = full[0]
    baseline_times = [mean(repeat["episode_seconds"] for repeat in row["clean_repeats"]) for row in baseline]
    baseline_summary = summarize_condition(baseline_protocol, baseline)
    results = []
    for protocol, rows in validated:
        result = summarize_condition(protocol, rows)
        times = [mean(repeat["episode_seconds"] for repeat in row["clean_repeats"]) for row in rows]
        result["paired_latency_difference_seconds_vs_fullkv"] = bootstrap_paired(
            [value - reference for value, reference in zip(times, baseline_times)], repeats=bootstrap_samples)
        for key in ("mean_episode_seconds", "mean_episode_excluding_tool_seconds", "mean_final_kv_bytes",
                    "mean_question_mean_generation_boundaries_kv_bytes",
                    "mean_clean_peak_allocated_above_model_bytes"):
            if baseline_summary[key] <= 0:
                raise RuntimeError(f"FullKV normalization denominator must be positive: {key}")
            result[key + "_over_fullkv"] = result[key] / baseline_summary[key]
        result["episode_speedup_over_fullkv"] = baseline_summary["mean_episode_seconds"] / result["mean_episode_seconds"]
        results.append(result)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": 1, "hardware": reference["hardware"], "model": reference["model"],
        "sample_ids": reference["sample_ids"], "conditions": results,
        "normalization": "X(method)/X(FullKV), same absolute metric and equal-question aggregation",
        "speedup": "T(FullKV)/T(method); free-rollout end-to-end, not fixed-work kernel speedup",
        "timing_warning": "Diagnostic phase timings are synchronized/instrumented; do not replace or add "
                          "them to clean headline latency. Exclusive phases can be summed only within diagnostics.",
        "memory_warning": "Actual active K/V payload, model-excluded allocator and model-excluded "
                          "driver process usage are different quantities. Reserved memory is not live overhead.",
        "lazy_warning": "No byte-matched comparison; protect prompt + W=175 and 7x heads can exceed nominal "
                        "native-GQA reference even before trajectory eviction. Compare actual bytes, not ratio labels.",
    }
    atomic_json(output / "SUMMARY.json", summary)
    lines = [
        "# QA latency / memory measurements", "",
        f"Model: {reference['model'][0]}; GPU: {reference['hardware']['name']}; "
        f"N={len(reference['sample_ids'])}; seed={reference['seed']}.",
        "", "Only complete, source-bound, trajectory/final-KV parity-checked conditions are shown.",
        "Slots are per head. LazyEviction is NOT a same-byte-budget condition.",
        "", "| Method | Nominal fraction | Episode s | / FullKV | Step-end KV MiB | Final KV MiB | "
        "Clean peak nonmodel allocated MiB | Final nonmodel driver MiB | Mean blocks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['method']} | {result['ratio']:.0%} | {result['mean_episode_seconds']:.4f} | "
            f"{result['mean_episode_seconds_over_fullkv']:.4f} | "
            f"{result['mean_question_mean_generation_boundaries_kv_bytes'] / MIB:.3f} | "
            f"{result['mean_final_kv_bytes'] / MIB:.3f} | "
            f"{result['mean_clean_peak_allocated_above_model_bytes'] / MIB:.3f} | "
            f"{result['mean_final_driver_above_model_bytes'] / MIB:.3f} | {result['mean_generation_blocks']:.2f} |")
    lines += ["", summary["timing_warning"], "", summary["memory_warning"], "", summary["lazy_warning"],
              "", "Per-question paired bootstrap intervals, step endpoints, protected/trajectory splits and "
              "diagnostic phases are in SUMMARY.json and condition-specific boundary CSV files."]
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    for protocol, rows in validated:
        name = f"{protocol['method']}_r{int(protocol['ratio'] * 100)}"
        (output / f"{name}_boundaries.csv").write_text(boundary_csv(protocol, rows), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    args = parser.parse_args(argv)
    summarize_runs(args.runs, args.output, args.bootstrap_samples)


if __name__ == "__main__":
    main()
