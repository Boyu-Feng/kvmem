"""Independent W3 measurements; never change the restored evaluator or paper scores."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_rebuttal_legacy import atomic_json, verify_original_sources

MODES = {"fullkv": "none", "stepkv": "step_aware_h2o", "h2o": "h2o", "tova": "tova"}


def gpu_state():
    physical = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    lines = subprocess.check_output([
        "nvidia-smi", f"--id={physical}", "--query-gpu=uuid,name,memory.total,driver_version",
        "--format=csv,noheader,nounits"], text=True).strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError("Exactly one visible GPU required")
    uuid, name, total, driver = [s.strip() for s in lines[0].split(",")]
    processes = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"], text=True)
    peers = []
    for line in processes.splitlines():
        fields = [s.strip() for s in line.split(",")]
        if len(fields) == 2 and fields[0] == uuid and int(fields[1]) != os.getpid():
            peers.append(int(fields[1]))
    return {"uuid": uuid, "name": name, "memory_total_MiB": int(total),
            "driver": driver, "other_compute_pids": peers}


def require_exclusive():
    state = gpu_state()
    if state["other_compute_pids"]:
        raise RuntimeError(f"GPU has other compute processes; timing invalid: {state}")
    return state


class ErrorWatch:
    """Preserve original stdout while rejecting swallowed inference exceptions."""
    def __init__(self, stream):
        self.stream, self.tail, self.errors = stream, "", []

    def write(self, value):
        self.stream.write(value)
        joined = self.tail + value
        if any(marker in joined for marker in (
            "[ERROR]", "Traceback (most recent call last)",
            "Scoring forward failed", "Prune failed for all tried signatures", "Pruning strategy failed",
            "Failed to apply prune results", "output_attentions=True returned None even with eager attention",
        )):
            self.errors.append(joined[-500:])
        self.tail = joined[-64:]
        return len(value)

    def flush(self):
        self.stream.flush()


def cache_digest(llm):
    import torch
    from scripts.legacy_efficiency_metrics import cache_tensor_pairs
    digest = hashlib.sha256()
    shapes = []
    for k, v in cache_tensor_pairs(llm.past_key_values):
        for tensor in (k, v):
            shapes.append(list(tensor.shape))
            digest.update(str((tuple(tensor.shape), str(tensor.dtype))).encode())
            digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return {"sha256": digest.hexdigest(), "shapes": shapes}


def semantic_record(result, llm):
    answer, trajectory, timings, debug = result
    return {"answer": answer, "trajectory": trajectory,
            "boundary_counter_lengths": [s.get("kv_cache_length") for s in timings],
            "prompt_token_count": debug.get("prompt_token_count", 0),
            "global_token_ids": debug.get("global_token_ids", []),
            "final_cache": cache_digest(llm)}


def semantic_hash(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def summarize(rows):
    if not rows:
        return {}
    per_question = [statistics.mean(r["clean_elapsed_seconds"]) for r in rows]
    result = {
        "questions": len(rows), "repeats_per_question": len(rows[0]["clean_elapsed_seconds"]),
        "mean_episode_seconds": statistics.mean(per_question),
        "median_episode_seconds": statistics.median(per_question),
        "std_across_question_means_seconds": statistics.stdev(per_question) if len(rows) > 1 else 0.0,
        "mean_num_steps": statistics.mean(r["num_steps"] for r in rows),
        "max_clean_peak_allocated_MiB": max(max(r["clean_peak_allocated_MiB"]) for r in rows),
        "mean_clean_peak_allocated_MiB": statistics.mean(statistics.mean(r["clean_peak_allocated_MiB"]) for r in rows),
        "all_output_and_final_KV_parity": all(r["parity"] for r in rows),
    }
    # Equal question weighting; longer rollouts do not get extra weight here.
    for quantity in ("tokens", "total_mib", "protected_tokens", "protected_mib", "trajectory_tokens", "trajectory_mib"):
        boundary = [statistics.mean(s["boundary"][quantity] for s in r["diagnostics"]["steps"]) for r in rows]
        peaks = [r["diagnostics"]["kv"]["peak_active"][quantity] for r in rows]
        result[f"mean_question_mean_boundary_{quantity}"] = statistics.mean(boundary)
        result[f"mean_question_peak_active_{quantity}"] = statistics.mean(peaks)
        result[f"max_peak_active_{quantity}"] = max(peaks)
    result["mean_clean_peak_above_parameter_MiB"] = statistics.mean(
        statistics.mean(r["clean_peak_allocated_MiB"]) - r["model_parameter_MiB"] for r in rows)
    result["cache_mean_definition"] = "Mean over generation-block endpoints within each question, then mean over questions."
    result["cache_peak_definition"] = "Forward-input/output and pre/post-pruning active K/V payload; all questions max and per-question peak mean both reported; transient duplicate storage is in allocator metric."
    components = sorted({key for r in rows for key in r["diagnostics"]["component_timings"]})
    result["diagnostic_mean_inclusive_seconds_do_not_sum"] = {
        key: statistics.mean(r["diagnostics"]["component_timings"].get(key, {}).get("inclusive_seconds", 0.0) for r in rows)
        for key in components}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=MODES, required=True)
    parser.add_argument("--ratio", type=float, choices=[.2, .5], default=.5)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--assets", type=Path, default=Path("/data/experiment/fengboyu/stepkv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.samples <= 500 or args.repeats < 1:
        parser.error("Require 1..500 samples and positive repeats")
    args.output.mkdir(parents=True, exist_ok=True)
    core_hashes = verify_original_sources(ROOT)
    hardware = require_exclusive()
    import torch
    import transformers
    from datasets import load_dataset
    from scripts.legacy_efficiency_metrics import attach_legacy_efficiency
    base = importlib.import_module("run_all_wiki_experiments_v2")
    assets = {k: json.loads((args.assets / f"{k}_ready.json").read_text()) for k in ("model", "data")}
    files = sorted(str(p) for p in (args.assets / "datasets/hotpotqa/distractor").glob("validation-*.parquet"))
    data = load_dataset("parquet", data_files={"validation": files}, split="validation",
                        cache_dir=str(args.assets / "datasets/arrow_cache"))
    if len(data) != 7405 or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected pinned HotpotQA data and exactly one visible GPU")
    base.MODEL_PATH = assets["model"]["path"]
    base.RANDOM_SEED, base.NUM_SAMPLES, base.MAX_STEPS, base.CHECKPOINT_INTERVAL = 233, 502, 7, 1
    all_selected = base.select_samples(data)
    warmups, selected = all_selected[500:502], all_selected[:args.samples]
    by_question = {sample["question"]: sample for _, sample in warmups + selected}
    if len(by_question) != len(warmups + selected):
        raise RuntimeError("Question mapping is not unique")
    warmup_ids = {s["id"] for _, s in warmups}
    protocol = {
        "version": "legacy_efficiency_v1", "method": args.method, "ratio": args.ratio,
        "display_budget": 1.0 if args.method == "fullkv" else args.ratio,
        "seed": 233, "model": "Qwen2.5-7B-Instruct", "dtype": "bfloat16", "max_steps": 7,
        "max_new_tokens_per_block": 256, "repeats": args.repeats,
        "sample_ids": [s["id"] for _, s in selected], "warmup_ids": [s["id"] for _, s in warmups],
        "assets": assets, "hardware": {k: v for k, v in hardware.items() if k != "other_compute_pids"},
        "generation_config": json.loads((Path(base.MODEL_PATH) / "generation_config.json").read_text()),
        "core_hashes": core_hashes,
        "measurement_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in (Path(__file__), ROOT / "scripts/legacy_efficiency_metrics.py")},
        "torch": torch.__version__, "transformers": transformers.__version__,
        "timing_scope": "Synchronized whole original episode; includes retrieval, parsing, step scores, all model/scoring/pruning/cache calls and original diagnostic prints; excludes model/index loading, warmups, metrics serialization and final-cache hash transfers.",
        "allocator_scope": "Reset immediately after the original per-episode llm.reset call. Absolute allocated peak includes model parameters and all process tensor allocations. Subtract-parameter metric is reported separately, not called KV memory.",
        "diagnostics": "Separate replay; fine-grained synchronized inclusive intervals must not be summed or substituted for clean latency.",
        "sampling": "First N of original seed-233 selection, not chosen by results; 501st/502nd warmup.",
        "cpu": {"count": os.cpu_count(), "omp_threads": os.environ.get("OMP_NUM_THREADS"),
                "model": next((s.split(":", 1)[1].strip() for s in Path("/proc/cpuinfo").read_text().splitlines() if s.startswith("model name")), None)},
        "historical_paper_provenance": "This is new measurement on restored source, not recovered Figure 4/5 raw data.",
    }
    manifest = args.output / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text())["protocol"] != protocol:
        raise RuntimeError("Measurement source/protocol changed; use a new output directory")
    if not manifest.exists():
        atomic_json(manifest, {"created": time.time(), "protocol": protocol})
    path = args.output / "measurements.json"
    previous = json.loads(path.read_text()) if path.exists() else {"rows": []}
    rows = {row["id"]: row for row in previous["rows"]}
    original_episode = base._run_react_kv_episode

    def save():
        ordered = [rows[s["id"]] for _, s in selected if s["id"] in rows]
        atomic_json(path, {"rows": ordered, "summary": summarize(ordered)})

    def once(question, llm, episode_args, episode_kwargs):
        require_exclusive()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        # Keep the original reset at its original call site. Reset the allocator
        # counter immediately afterwards so the previous episode's resident cache
        # cannot become this episode's reported peak. This does not alter KV state.
        original_reset = llm.reset
        missing = object()
        saved_reset = llm.__dict__.get("reset", missing)
        def reset_with_peak_boundary(*reset_args, **reset_kwargs):
            result = original_reset(*reset_args, **reset_kwargs)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            return result
        llm.reset = reset_with_peak_boundary
        watcher = ErrorWatch(sys.stdout)
        start = time.perf_counter()
        try:
            with contextlib.redirect_stdout(watcher):
                result = original_episode(question, llm, *episode_args, **episode_kwargs)
            torch.cuda.synchronize()
        finally:
            if saved_reset is missing:
                del llm.reset
            else:
                llm.reset = saved_reset
        elapsed = time.perf_counter() - start
        peak = torch.cuda.max_memory_allocated() / 2**20
        require_exclusive()
        if watcher.errors:
            raise RuntimeError(f"Original evaluator logged errors: {watcher.errors[:2]}")
        return result, elapsed, peak

    def measured_episode(question, llm, *episode_args, **episode_kwargs):
        sample = by_question[question]
        sid = sample["id"]
        atomic_json(args.output / "status.json", {"status": "running", "pid": os.getpid(),
                    "updated": time.time(), "sample_id": sid, "completed": len(rows), "target": args.samples})
        if sid in warmup_ids:
            return once(question, llm, episode_args, episode_kwargs)[0]
        elapsed, peaks, reference, clean_result = [], [], None, None
        for repeat in range(args.repeats):
            result, seconds, peak = once(question, llm, episode_args, episode_kwargs)
            record = semantic_record(result, llm)
            if reference is not None and record != reference:
                raise RuntimeError(f"Clean repeat output/KV mismatch at {sid}; measurements rejected")
            reference, clean_result = record, result
            elapsed.append(seconds)
            peaks.append(peak)
        monitor = attach_legacy_efficiency(llm, synchronize=True, synchronize_components=True)
        try:
            diagnostic, diag_seconds, diag_peak = once(question, llm, episode_args, episode_kwargs)
            stats = monitor.snapshot()
        finally:
            monitor.close()
        if stats["instrumentation_errors"]:
            raise RuntimeError(f"Invalid cache observations: {stats['instrumentation_errors']}")
        if not stats["steps"] or any(s["boundary"]["tokens"] is None for s in stats["steps"]):
            raise RuntimeError("Missing or nonuniform boundary cache observations")
        candidate = semantic_record(diagnostic, llm)
        if candidate != reference:
            atomic_json(args.output / f"parity_failure_{sid}.json", {"clean": reference, "diagnostic": candidate})
            raise RuntimeError(f"Instrumented output/KV mismatch at {sid}; measurements rejected")
        params = sum(p.numel() * p.element_size() for p in llm.model.parameters()) / 2**20
        rows[sid] = {"id": sid, "clean_elapsed_seconds": elapsed,
                     "clean_peak_allocated_MiB": peaks, "model_parameter_MiB": params,
                     "diagnostic_episode_seconds_not_main_latency": diag_seconds,
                     "diagnostic_peak_allocated_MiB": diag_peak, "diagnostics": stats,
                     "num_steps": len(clean_result[1]), "semantic_sha256": semantic_hash(reference),
                     "parity": True, "recorded": time.time()}
        save()
        return clean_result

    base._run_react_kv_episode = measured_episode
    try:
        atomic_json(args.output / "status.json", {"status": "loading_retriever", "pid": os.getpid(), "updated": time.time()})
        from retrievers.WikiBM25Retriever import WikiBM25Retriever
        retriever = WikiBM25Retriever(assets["data"]["wiki_index"])
        if len(retriever.titles) != 5233235 or retriever.corpus_texts is None:
            raise RuntimeError("Full Wikipedia retrieval assets failed validation")
        # Re-execute warmup on every process launch. Never reuse old timing caches.
        base._resolve_existing_kv_path = lambda path, label="": path
        checkpoint = args.output / f"internal_checkpoint_{os.getpid()}.json"
        todo = warmups + [(i, s) for i, s in selected if s["id"] not in rows]
        if len(todo) > len(warmups):
            base.run_react_kv_experiment(data, todo, retriever, MODES[args.method],
                str(args.output / f"internal_results_{os.getpid()}.json"), str(checkpoint),
                kv_config_override={"cache_ratio": args.ratio}, metrics_method=f"efficiency_{args.method}")
        if set(rows) != set(protocol["sample_ids"]):
            raise RuntimeError("Incomplete measurement IDs")
        save()
        atomic_json(args.output / "status.json", {"status": "complete", "pid": os.getpid(),
                    "updated": time.time(), "samples": len(rows)})
    except BaseException:
        atomic_json(args.output / "status.json", {"status": "failed", "pid": os.getpid(),
                    "updated": time.time(), "traceback": traceback.format_exc()})
        raise
    finally:
        base._run_react_kv_episode = original_episode


if __name__ == "__main__":
    main()
