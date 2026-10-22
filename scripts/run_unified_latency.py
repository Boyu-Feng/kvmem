"""Remeasure all requested methods with the original StepKV HF main decoder."""
from __future__ import annotations

import argparse
import contextlib
import importlib
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import latency_datasets, run_latency_experiment as legacy
from scripts.latency_runtime import (
    EpisodeTrace, MODEL, digest, file_hash, instrument_episode, observe_kv_gathers,
    patch_attribute, process_memory_reader, read_json, require_exclusive,
    single_gpu_loader, validate_diagnostics,
)
from scripts.qa32_runtime import QAErrorWatch
from scripts.run_qa32_rebuttal import atomic_json
from scripts.unified_latency_runtime import (
    DATASETS, DECODE_POLICY, METHODS, GenerationAudit, UnifiedMonitor,
    adapter_audit, drain_auxiliary, unified_method,
)

SOURCE_FILES = ("scripts/run_unified_latency.py", "scripts/unified_latency_runtime.py",
                "scripts/unified_sidequest_runtime.py", "scripts/unified_thinkkv_runtime.py",
                "scripts/run_unified_latency_queue.py", "scripts/thinkkv_selection.py",
                "scripts/sidequest_runtime.py", "models/RebuttalLLM.py")


def source_hashes():
    return {**latency_datasets.experiment_hashes(),
            **{name: file_hash(ROOT / name) for name in SOURCE_FILES}}


def prepare_plan(args):
    if args.method not in METHODS or args.dataset not in DATASETS:
        raise ValueError("Unknown unified method/dataset")
    if not 1 <= args.samples <= 20 or args.repeats != 2 or args.ratio not in (.2, .5):
        raise ValueError("Require1..20 questions, two clean repeats and20%/50% requested ratio")
    return {
        "schema_version": 1, "method": args.method, "dataset": args.dataset, "model": MODEL,
        "ratio": None if args.method == "sidequest_unified" else
                 1.0 if args.method == "fullkv" else args.ratio,
        "budget": "adaptive" if args.method == "sidequest_unified" else
                  "full" if args.method == "fullkv" else "trajectory_ratio",
        "seed": 233, "samples": args.samples, "sample_positions": list(range(args.samples)),
        "warmup_positions": [500, 501], "clean_repeats": 2, "diagnostic_repeats": 1,
        "clean_repeats_per_question": 2, "diagnostic_repeats_per_question": 1,
        "max_steps": 7, "max_new_tokens_per_block": 256, "dtype": "bfloat16",
        "quantization": False, "decode_policy": DECODE_POLICY,
        "generation_policy": "Original StepKV QwenLLMWithKVCache HF model.generate, greedy "
                             "EOS/256cap then posthoc stop-text and KV crop; no early-stop main loop",
        "generation_overrides": {"do_sample": False},
        "decoded_scope": "Main HF generation runs to EOS or its original token cap before "
                         "stop-text/KV cropping; generated then discarded tokens remain timed.",
        "requires_exclusive_gpu": True,
        "comparison": "Common original StepKV HF main generation and FullKV control. "
                      "Free rollouts still differ in actual work/quality. Lazy slots are not byte matched; "
                      "SideQuest is adaptive/untrained; ThinkKV is selection-only without CT/TBQ.",
        "official_sidequest_reproduction": False, "official_thinkkv_reproduction": False,
        "model_ready": str(args.model_ready.resolve()), "data_ready": str(args.data_ready.resolve()),
    }


def run_once(base, llm, retriever, method, sample):
    import torch
    from scripts.latency_metrics import model_storage_bytes
    require_exclusive()
    stdout, stderr = QAErrorWatch(legacy.QuietStream()), QAErrorWatch(legacy.QuietStream())
    reset_original, tool_original = llm.reset, base.execute_action
    resets, tool_seconds = 0, 0.0
    baseline = model_storage_bytes(llm.model)["cuda:0"]

    def reset(*args, **kwargs):
        nonlocal resets
        result = reset_original(*args, **kwargs)
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        resets += 1
        return result

    def tool(*args, **kwargs):
        nonlocal tool_seconds
        begin = time.perf_counter()
        try:
            return tool_original(*args, **kwargs)
        finally:
            tool_seconds += time.perf_counter() - begin

    with patch_attribute(llm, "reset", reset), patch_attribute(base, "execute_action", tool), \
            contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), GenerationAudit(llm) as gen:
        torch.cuda.synchronize(0)
        begin = time.perf_counter()
        result = base._run_react_kv_episode(
            sample["question"], llm, retriever, pruning_mode=method.mode, return_debug=True)
        drain_auxiliary(llm)
        torch.cuda.synchronize(0)
        seconds = time.perf_counter() - begin
        memory = legacy.cuda_allocator(baseline)
    if resets != 1 or stdout.errors or stderr.errors:
        raise RuntimeError(f"Invalid original QA pass: resets={resets}, errors={stdout.errors + stderr.errors}")
    if not 0 <= tool_seconds <= seconds or not gen.calls:
        raise RuntimeError("Invalid original-generation timing/audit")
    require_exclusive()
    semantics = legacy.semantic_state(result, llm)
    audit = adapter_audit(llm)
    record = {
        "episode_seconds": seconds, "tool_seconds": tool_seconds,
        "episode_excluding_tool_seconds": seconds - tool_seconds, "memory": memory,
        "em": base.exact_match(result[0], sample["answer"]),
        "f1": base.f1_score(result[0], sample["answer"]), "predicted_answer": result[0],
        "generation_blocks": len(result[2]), "main_generation_calls": gen.calls,
        "semantic_sha256": digest(semantics), "audit": audit,
        "timing_scope": "Original frozen QA episode/HF decoder, synchronized outer wall boundaries; "
                        "SideQuest includes draining unfinished auxiliary work before return.",
    }
    return result, record


def _mean(rows, getter):
    return statistics.mean(getter(row) for row in rows)


def summary(rows):
    if not rows:
        return {"questions": 0}
    result = {"questions": len(rows), "completed_questions": len(rows)}
    for name in ("episode_seconds", "episode_excluding_tool_seconds", "tool_seconds",
                 "em", "f1", "generation_blocks"):
        result["mean_" + name] = _mean(
            rows, lambda row: statistics.mean(r[name] for r in row["clean_repeats"]))
    result["mean_final_kv_bytes"] = _mean(rows, lambda r: r["diagnostics"]["final"]["kv"]["total_bytes"])
    result["mean_final_trajectory_kv_bytes"] = _mean(
        rows, lambda r: r["diagnostics"]["final"]["kv"]["trajectory_bytes"])
    result["mean_final_total_live_kv_bytes"] = _mean(
        rows, lambda r: r["diagnostics"]["final"]["all_live_kv"]["storage_bytes"])
    result["mean_final_tokens_per_head"] = _mean(
        rows, lambda r: r["diagnostics"]["final"]["kv"]["tokens_per_head"])
    result["mean_clean_peak_allocated_above_model_bytes"] = _mean(rows, lambda row:
        statistics.mean(r["memory"]["peak_allocated_above_model_bytes"] for r in row["clean_repeats"]))
    result["mean_step_kv_bytes"] = _mean(rows, lambda row: statistics.mean(
        b["memory"]["kv"]["total_bytes"] for b in row["interaction_boundaries"]))
    for kind in ("allocated", "reserved", "driver"):
        result[f"mean_final_{kind}_above_model_bytes"] = _mean(
            rows, lambda r: r["diagnostics"]["final"]["allocator"]["cuda:0"][f"{kind}_above_model_bytes"])
    phases = set().union(*(r["diagnostics"]["timing"]["exclusive_seconds"] for r in rows))
    result["diagnostic_mean_exclusive_seconds"] = {
        phase: _mean(rows, lambda row: row["diagnostics"]["timing"]["exclusive_seconds"].get(phase, 0))
        for phase in sorted(phases)}
    workload = set().union(*(r["diagnostics"]["workload"] for r in rows))
    result["diagnostic_mean_workload"] = {phase: {field: _mean(rows, lambda r:
        r["diagnostics"]["workload"].get(phase, {}).get(field, 0))
        for field in ("forward_calls", "input_positions")} for phase in sorted(workload)}
    result["async_semantic_divergence_questions"] = sum(r["async_semantic_divergence"] for r in rows)
    return result


def validate_rows(rows, method):
    for row in rows:
        if row["row_sha256"] != digest({k: v for k, v in row.items() if k != "row_sha256"}):
            raise RuntimeError("Corrupt unified latency measurement row")
        if len(row["clean_repeats"]) != 2:
            raise RuntimeError("Unified latency needs exactly two committed clean repeats")
        for repeat in [*row["clean_repeats"], row["diagnostic_wall"]]:
            value = repeat["episode_seconds"]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise RuntimeError("Invalid unified clean/diagnostic latency")
            if repeat["memory"]["peak_allocated_above_model_bytes"] < 0:
                raise RuntimeError("Invalid nonmodel memory measurement")
            tool = repeat["tool_seconds"]
            excluding = repeat["episode_excluding_tool_seconds"]
            if (isinstance(tool, bool) or not isinstance(tool, (int, float))
                    or not math.isfinite(tool) or not 0 <= tool <= value
                    or not isinstance(excluding, (int, float)) or isinstance(excluding, bool)
                    or not math.isfinite(excluding)
                    or not math.isclose(value - tool, excluding, rel_tol=1e-9, abs_tol=1e-9)):
                raise RuntimeError("Invalid unified tool/non-tool timing partition")
            if not repeat["main_generation_calls"] or any(
                    call["max_new_tokens"] not in (255, 256) for call in repeat["main_generation_calls"]):
                raise RuntimeError("Missing/corrupt original HF generation audit")
        equal = len({r["semantic_sha256"] for r in [
            *row["clean_repeats"], row["diagnostic_wall"]]}) == 1
        if row["parity"] is not equal or row["async_semantic_divergence"] is not (not equal):
            raise RuntimeError("Semantic hashes and parity/divergence flags disagree")
        if method != "sidequest_unified" and row["parity"] is not True:
            raise RuntimeError("Deterministic main generation/cache parity failed")
        validate_diagnostics(row["diagnostics"], row["interaction_boundaries"], row["generation_blocks"])


def validate_run(directory):
    protocol = read_json(directory / "manifest.json")["protocol"]
    saved = read_json(directory / "measurements.json")
    rows = saved["rows"]
    validate_rows(rows, protocol["method"])
    if (read_json(directory / "status.json")["status"] != "complete"
            or protocol["decode_policy"] != DECODE_POLICY or protocol["quantization"] is not False
            or saved["protocol_sha256"] != digest(protocol) or saved["rows_sha256"] != digest(rows)
            or len(rows) != protocol["samples"] or [r["id"] for r in rows] != protocol["sample_ids"]
            or saved["summary"] != summary(rows)):
        raise RuntimeError(f"Incomplete/incompatible unified latency run: {directory}")
    return protocol, rows


def export_points(directory, rows):
    from scripts.export_latency_points import flatten, save
    clean, phases, memory = [], [], []
    for row in rows:
        for number, repeat in enumerate(row["clean_repeats"], 1):
            clean.append({"id": row["id"], "repeat": number,
                          **flatten({k: v for k, v in repeat.items() if k not in ("audit",)})})
        for number, step in enumerate(row["diagnostics"]["steps"], 1):
            phases.append({"id": row["id"], "step": number, **flatten(step["timing"])})
            memory.append({"id": row["id"], "step": number, "site": "generation_return",
                           **flatten(step["memory"])})
        for b in row["interaction_boundaries"]:
            memory.append({"id": row["id"], "step": b["step"], "site": b["endpoint"],
                           **flatten(b["memory"])})
        phases.append({"id": row["id"], "site": "episode_total",
                       **flatten(row["diagnostics"]["timing"])})
        for event in row["diagnostics"]["auxiliary_events"]:
            phases.append({"id": row["id"], **event})
        memory.append({"id": row["id"], "site": "episode_final",
                       **flatten(row["diagnostics"]["final"])})
    save(directory / "clean_repeats.csv", clean)
    save(directory / "diagnostic_phases.csv", phases)
    save(directory / "memory_points.csv", memory)


def execute(args):
    import torch
    import transformers
    from datasets import load_dataset
    from retrievers.WikiBM25Retriever import WikiBM25Retriever
    from token_tracker import TokenTracker
    if torch.get_num_threads() != 4 or any(os.environ.get(n) != "4" for n in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")):
        raise RuntimeError("Original GPU7 latency requires4 CPU threads")
    hashes, hardware = source_hashes(), require_exclusive()
    model, index, files, assets = legacy.load_assets(args.model_ready, args.data_ready)
    base = importlib.import_module("run_all_wiki_experiments_v2")
    base.MAX_STEPS = 7
    if args.dataset == "hotpotqa":
        data = load_dataset("parquet", data_files={"validation": list(map(str, files))},
                            split="validation", cache_dir=str(args.output.parent / "arrow_cache"))
        selected, warmups = legacy.select_samples(base, data, args.samples)
    else:
        _, data, more = latency_datasets.normalized_assets(args.data_ready, args.dataset)
        assets = {**assets, **more}
        selected, warmups = latency_datasets.select_samples(base, data, args.samples, args.dataset)
    with unified_method(args.method) as method:
        config, scope = method.config(base, args.ratio), method.scope
    protocol = {**prepare_plan(args), "sample_ids": [r["id"] for _, r in selected],
                "warmup_ids": [r["id"] for _, r in warmups], "source_hashes": hashes, "assets": assets,
                "method_config": config, "method_scope": scope,
                "hardware": {k: v for k, v in hardware.items() if k != "processes"},
                "generation_config": read_json(model / "generation_config.json"),
                "cpu_threads": {n: os.environ[n] for n in (
                    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
                "versions": {"python": sys.version, "torch": torch.__version__,
                             "transformers": transformers.__version__, "cuda": torch.version.cuda}}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest, checkpoint = args.output / "manifest.json", args.output / "measurements.json"
    if manifest.exists() and read_json(manifest)["protocol"] != read_json_value(protocol):
        raise RuntimeError("Unified source/protocol changed; use a new output directory")
    atomic_json(manifest, {"protocol": protocol})
    rows = []
    if checkpoint.exists():
        saved = read_json(checkpoint)
        rows = saved["rows"]
        validate_rows(rows, args.method)
        if (saved["protocol_sha256"] != digest(protocol) or saved["rows_sha256"] != digest(rows)
                or [r["id"] for r in rows] != protocol["sample_ids"][:len(rows)]):
            raise RuntimeError("Invalid unified checkpoint prefix")

    def state(status, **extra):
        atomic_json(args.output / "status.json", {"status": status, "pid": os.getpid(),
                    "updated": time.time(), "completed_questions": len(rows),
                    "target_questions": args.samples, **extra})
    llm = None
    try:
        state("loading_model_and_retriever")
        with unified_method(args.method) as method, single_gpu_loader():
            llm = method.llm_class(str(model), kv_config=config, token_tracker=TokenTracker())
            retriever = WikiBM25Retriever(str(index))
            if len(retriever.titles) != 5233235 or retriever.corpus_texts is None:
                raise RuntimeError("Full Wikipedia retrieval assets missing")
            for _, sample in warmups:
                state("warming_up", sample_id=sample["id"])
                run_once(base, llm, retriever, method, sample)
            for _, sample in selected[len(rows):]:
                if source_hashes() != hashes:
                    raise RuntimeError("Unified experiment sources changed")
                state("measuring", sample_id=sample["id"])
                clean_results = [run_once(base, llm, retriever, method, sample) for _ in range(2)]
                clean = [r[1] for r in clean_results]
                with UnifiedMonitor(llm, process_memory_reader=process_memory_reader) as monitor:
                    instrument_episode(base, method, llm, monitor)
                    hooks = getattr(llm, "DIAGNOSTIC_HOOKS", ())
                    for name, phase in hooks:
                        monitor.wrap(llm, name, phase)
                    with observe_kv_gathers(llm, monitor), EpisodeTrace(base._run_react_kv_episode, monitor) as trace:
                        result, wall = run_once(base, llm, retriever, method, sample)
                    stats, boundaries = monitor.snapshot(), trace.boundaries
                equal = len({r["semantic_sha256"] for r in [*clean, wall]}) == 1
                if not equal and args.method != "sidequest_unified":
                    atomic_json(args.output / "parity_failure.json",
                                {"id": sample["id"], "hashes": [r["semantic_sha256"] for r in [*clean, wall]]})
                    raise RuntimeError(f"Unified clean/diagnostic parity failed: {sample['id']}")
                row = {"id": sample["id"], "clean_repeats": clean, "diagnostics": stats,
                       "interaction_boundaries": boundaries, "diagnostic_wall": wall,
                       "generation_blocks": len(result[2]), "trajectory": result[1],
                       "predicted_answer": result[0], "em": statistics.mean(r["em"] for r in clean),
                       "f1": statistics.mean(r["f1"] for r in clean), "parity": equal,
                       "async_semantic_divergence": not equal, "audit": wall["audit"]}
                row["row_sha256"] = digest(row)
                validate_rows([row], args.method)
                rows.append(row)
                atomic_json(checkpoint, {"schema_version": 1, "protocol_sha256": digest(protocol),
                                         "rows_sha256": digest(rows), "rows": rows, "summary": summary(rows)})
        state("complete")
        validate_run(args.output)
        export_points(args.output, rows)
    except BaseException:
        state("failed", traceback=traceback.format_exc())
        raise
    finally:
        if llm is not None and hasattr(llm, "close"):
            llm.close()


def read_json_value(value):
    import json
    from scripts.latency_runtime import canonical
    return json.loads(canonical(value))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--ratio", type=float, choices=(.2, .5), default=.2)
    for name in ("model-ready", "data-ready", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    plan = prepare_plan(args)
    if args.execute:
        execute(args)
    else:
        print(read_json_value(plan))


if __name__ == "__main__":
    main()
