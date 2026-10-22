"""Exclusive BF16 latency for ordinary-LLM ThinKV token selection, without TBQ."""
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
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.latency_runtime import (
    digest, file_hash, patch_attribute, process_memory_reader, read_json,
    require_exclusive, single_gpu_loader, validate_diagnostics,
)
from scripts.latency_metrics import DetailedLatencyMonitor, _tensor_group, model_storage_bytes
from scripts import thinkkv_selection as selection
from scripts.run_latency_experiment import QuietStream, cuda_allocator, load_assets, select_samples
from scripts.run_qa32_rebuttal import atomic_json
from scripts.run_rebuttal_efficiency import cache_digest

METHODS = ("fullkv_thinkkv_control", "thinkkv_selection_only")
DATASETS = ("hotpotqa", "2wiki", "musique")
SOURCE_FILES = ("scripts/thinkkv_selection.py", "scripts/run_thinkkv_latency.py",
                "scripts/run_thinkkv_latency_queue.py", "models/RebuttalLLM.py", "token_tracker.py")


def source_hashes():
    from scripts.latency_datasets import experiment_hashes
    return {**experiment_hashes(), **{name: file_hash(ROOT / name) for name in SOURCE_FILES}}


def episode(base, llm, question, retriever, boundary=None):
    prompt = base.REACT_KV_INITIAL_PROMPT.format(
        examples=base.REACT_EXAMPLES, question=question) + "Thought 1:"
    stops = ["\nObservation", "\nQuestion:"]
    response, prompt_cache, generated_cache = llm.generate_first(
        prompt, max_new_tokens=256, stop_strings=stops)
    del prompt_cache, generated_cache
    lookup = dict(page=None, lookup_keyword=None, lookup_list=None, lookup_cnt=0)
    trajectory = []
    answer, reason = "", "turn_limit"
    for step in range(1, 8):
        thought, action, arg = base.parse_action_original(f"Thought {step}:" + response, step)
        row = dict(step=step, response=response, thought=thought, action_type=action, action_arg=arg)
        trajectory.append(row)
        if action == "finish":
            answer, reason = arg or "", "finish"
        elif action not in ("search", "lookup"):
            reason = "unparsable_action"
        else:
            observation, lookup = base.execute_action(action, arg, retriever, lookup)
            observation = observation.replace("\\n", "")
            row["observation"] = observation
        if boundary is not None:
            boundary(step)
        if action == "finish" or action not in ("search", "lookup"):
            break
        if step < 7:
            llm.token_tracker.set_current_step(step + 1)
            response = llm.generate_incremental(
                f"\nObservation {step}: {observation}\nThought {step + 1}:",
                max_new_tokens=256, stop_strings=stops)
    return answer, reason, trajectory


class SelectionMonitor(DetailedLatencyMonitor):
    def _kv(self, cache):
        result = super()._kv(cache)
        result["token_scope"] = "Physical slots per native KV head; IDs can differ by head/layer"
        result["partition_basis"] = "Fully protected initial prompt, compressible trajectory"
        return result

    def _recurrence(self):
        ids = self.llm.selection.ids
        return {"selection_ids": _tensor_group([] if ids is None else [ids]),
                "quantization": False}


def semantic_hash(answer, trajectory, llm):
    ids = llm.selection.ids
    return digest({"answer": answer, "trajectory": trajectory,
                   "kv_sha256": cache_digest(llm),
                   "head_ids_sha256": cache_digest(SimpleNamespace(past_key_values=[(ids, ids)])),
                   "next_global_id": llm.token_tracker.next_global_id})


def pass_once(base, llm, retriever, sample, diagnostic):
    import torch
    from scripts.qa32_runtime import QAErrorWatch
    require_exclusive()
    stdout, stderr = QAErrorWatch(QuietStream()), QAErrorWatch(QuietStream())
    model_bytes = model_storage_bytes(llm.model)["cuda:0"]
    original_reset, original_tool = llm.reset, base.execute_action
    resets, tool_seconds = 0, 0.0
    boundaries = []
    monitor = None

    def reset():
        nonlocal resets
        original_reset()
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        resets += 1

    def tool(*args, **kwargs):
        nonlocal tool_seconds
        begin = time.perf_counter()
        try:
            return original_tool(*args, **kwargs)
        finally:
            tool_seconds += time.perf_counter() - begin

    def boundary(step):
        boundaries.append({"step": step, "endpoint": "action_and_tool_or_terminal_return",
                           "memory": monitor.sample_memory("interaction_step_end")})

    with contextlib.ExitStack() as scope:
        scope.enter_context(patch_attribute(llm, "reset", reset))
        scope.enter_context(patch_attribute(base, "execute_action", tool))
        if diagnostic:
            monitor = SelectionMonitor(llm, process_memory_reader=process_memory_reader)
            scope.enter_context(monitor)
            monitor.wrap(base, "execute_action", "tool")
            monitor.wrap(base, "parse_action_original", "agent_parse")
            monitor.wrap(llm, "_maybe_prune", "selection_and_cache_management")
            monitor.wrap(llm, "_rewind_to", "stop_crop")
            monitor.wrap(selection, "cluster_representatives", "token_clustering")
            monitor.wrap(selection.SelectionState, "materialize", "cache_gather")
        scope.enter_context(contextlib.redirect_stdout(stdout))
        scope.enter_context(contextlib.redirect_stderr(stderr))
        torch.cuda.synchronize(0)
        begin = time.perf_counter()
        answer, termination, trajectory = episode(
            base, llm, sample["question"], retriever, boundary if diagnostic else None)
        torch.cuda.synchronize(0)
        seconds = time.perf_counter() - begin
        memory = cuda_allocator(model_bytes)
        diagnostics = monitor.snapshot() if monitor is not None else None
    if resets != 1 or stdout.errors or stderr.errors:
        raise RuntimeError(f"Invalid ThinkKV pass: resets={resets}, errors={stdout.errors + stderr.errors}")
    if not 0 <= tool_seconds <= seconds:
        raise RuntimeError("Invalid tool/episode timing partition")
    audit = llm.audit()
    if audit["kv_dtype"] != "torch.bfloat16":
        raise RuntimeError("ThinkKV selection-only requires actual BF16 KV tensors")
    require_exclusive()
    result = {
        "episode_seconds": seconds, "tool_seconds": tool_seconds,
        "episode_excluding_tool_seconds": seconds - tool_seconds, "memory": memory,
        "predicted_answer": answer, "termination": termination, "trajectory": trajectory,
        "generation_blocks": len(trajectory), "em": base.exact_match(answer, sample["answer"]),
        "f1": base.f1_score(answer, sample["answer"]), "audit": audit,
        "semantic_sha256": semantic_hash(answer, trajectory, llm),
        "timing_scope": "Own matched greedy runtime; no diagnostic phase hooks in clean passes. "
                        "CUDA synchronized at outer boundaries only; token decisions still require host reads.",
    }
    if diagnostics is not None:
        validate_diagnostics(diagnostics, boundaries, len(trajectory))
        result.update(diagnostics=diagnostics, interaction_boundaries=boundaries)
    return result


def summary(rows):
    if not rows:
        return {"questions": 0}
    def mean_clean(field):
        return statistics.mean(statistics.mean(r[field] for r in row["clean_repeats"]) for row in rows)
    result = {f"mean_{name}": mean_clean(name) for name in (
        "episode_seconds", "episode_excluding_tool_seconds", "tool_seconds", "em", "f1", "generation_blocks")}
    result["questions"] = len(rows)
    result["mean_clean_peak_allocated_above_model_bytes"] = statistics.mean(
        statistics.mean(r["memory"]["peak_allocated_above_model_bytes"] for r in row["clean_repeats"])
        for row in rows)
    result["max_clean_peak_allocated_above_model_bytes"] = max(
        r["memory"]["peak_allocated_above_model_bytes"] for row in rows for r in row["clean_repeats"])
    for label, getter in (
        ("final_kv_bytes", lambda d: d["diagnostics"]["final"]["kv"]["total_bytes"]),
        ("final_tokens_per_head", lambda d: d["diagnostics"]["final"]["kv"]["tokens_per_head"]),
        ("step_kv_bytes", lambda d: statistics.mean(
            b["memory"]["kv"]["total_bytes"] for b in d["interaction_boundaries"])),
        ("actual_trajectory_ratio", lambda d: d["audit"]["actual_trajectory_ratio"]),
    ):
        values = [getter(row["diagnostic"]) for row in rows]
        result[f"mean_{label}"] = statistics.mean(v for v in values if v is not None) if any(
            v is not None for v in values) else None
    for phase in ("initial_prefill", "observation_prefill", "decode_forward", "token_clustering",
                  "selection_and_cache_management", "cache_gather", "stop_crop", "generation_other"):
        result[f"mean_diagnostic_{phase}_seconds"] = statistics.mean(
            row["diagnostic"]["diagnostics"]["timing"]["exclusive_seconds"].get(phase, 0) for row in rows)
    for kind in ("allocated", "reserved", "driver"):
        result[f"mean_final_{kind}_above_model_bytes"] = statistics.mean(
            row["diagnostic"]["diagnostics"]["final"]["allocator"]["cuda:0"][f"{kind}_above_model_bytes"]
            for row in rows)
    result["comparison_scope"] = "Own FullKV control, free rollouts; not official ThinKV/TBQ/CT throughput"
    return result


def export_points(directory, rows):
    from scripts.export_latency_points import flatten, save
    clean, phases, memory = [], [], []
    for row in rows:
        for number, repeat in enumerate(row["clean_repeats"], 1):
            clean.append({"id": row["id"], "repeat": number,
                          **flatten({k: v for k, v in repeat.items() if k not in ("trajectory", "audit")})})
        diagnostic = row["diagnostic"]
        for number, step in enumerate(diagnostic["diagnostics"]["steps"], 1):
            phases.append({"id": row["id"], "step": number, **flatten(step["timing"])})
            memory.append({"id": row["id"], "step": number, "site": "generation_return",
                           **flatten(step["memory"])})
        for boundary in diagnostic["interaction_boundaries"]:
            memory.append({"id": row["id"], "step": boundary["step"], "site": "interaction_step_end",
                           **flatten(boundary["memory"])})
        phases.append({"id": row["id"], "site": "episode_total",
                       **flatten(diagnostic["diagnostics"]["timing"])})
        memory.append({"id": row["id"], "site": "episode_final",
                       **flatten(diagnostic["diagnostics"]["final"])})
    save(directory / "clean_repeats.csv", clean)
    save(directory / "diagnostic_phases.csv", phases)
    save(directory / "memory_points.csv", memory)


def validate_run(directory):
    protocol = read_json(directory / "manifest.json")["protocol"]
    measurements = read_json(directory / "measurements.json")
    rows = measurements["rows"]
    validate_rows(rows)
    if (read_json(directory / "status.json")["status"] != "complete"
            or len(rows) != protocol["samples"]
            or [r["id"] for r in rows] != protocol["sample_ids"]
            or measurements["protocol_sha256"] != digest(protocol)
            or measurements["rows_sha256"] != digest(rows)
            or measurements["summary"] != summary(rows)):
        raise RuntimeError(f"Incomplete/corrupt ThinkKV latency run: {directory}")
    if protocol.get("dtype") != "bfloat16" or protocol.get("quantization") is not False:
        raise RuntimeError("ThinkKV latency protocol must disable quantization and require BF16")
    return protocol, rows


def validate_rows(rows):
    for row in rows:
        if len(row["clean_repeats"]) != 2 or row["all_passes_semantically_equal"] is not True:
            raise RuntimeError("ThinkKV clean/diagnostic parity failed")
        repeats = [*row["clean_repeats"], row["diagnostic"]]
        if len({r["semantic_sha256"] for r in repeats}) != 1:
            raise RuntimeError("ThinkKV persisted semantic hashes disagree")
        for repeat in repeats:
            seconds = repeat["episode_seconds"]
            if (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                    or not math.isfinite(seconds) or seconds <= 0):
                raise RuntimeError("Invalid clean/diagnostic ThinkKV latency")
            if repeat["audit"]["quantization"] is not False or repeat["audit"]["kv_dtype"] != "torch.bfloat16":
                raise RuntimeError("Quantized/non-BF16 ThinkKV result rejected")
            if repeat["memory"]["peak_allocated_above_model_bytes"] < 0:
                raise RuntimeError("Invalid above-model allocator measurement")
        diagnostic = row["diagnostic"]
        validate_diagnostics(diagnostic["diagnostics"], diagnostic["interaction_boundaries"],
                             diagnostic["generation_blocks"])


def execute(args):
    import torch
    import transformers
    from datasets import load_dataset
    from retrievers.WikiBM25Retriever import WikiBM25Retriever
    from scripts import latency_datasets
    if any(os.environ.get(name) != "4" for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")) or torch.get_num_threads() != 4:
        raise RuntimeError("Match previous GPU7 latency: OMP/MKL/OpenBLAS and torch require4 CPU threads")
    args.output.mkdir(parents=True, exist_ok=True)
    hashes, hardware = source_hashes(), require_exclusive()
    model, index, files, assets = load_assets(args.model_ready, args.data_ready)
    base = importlib.import_module("run_all_wiki_experiments_v2")
    if args.dataset == "hotpotqa":
        data = load_dataset("parquet", data_files={"validation": list(map(str, files))},
                            split="validation", cache_dir=str(args.output.parent / "arrow_cache"))
        selected, warmups = select_samples(base, data, args.samples)
    else:
        _, data, extra = latency_datasets.normalized_assets(args.data_ready, args.dataset)
        assets = {**assets, **extra}
        selected, warmups = latency_datasets.select_samples(base, data, args.samples, args.dataset)
    protocol = {
        "schema_version": 1, "method": args.method, "dataset": args.dataset,
        "ratio": args.ratio if args.method == "thinkkv_selection_only" else 1.0,
        "samples": args.samples, "seed": 233, "clean_repeats": 2, "diagnostic_repeats": 1,
        "sample_ids": [r["id"] for _, r in selected], "warmup_ids": [r["id"] for _, r in warmups],
        "max_steps": 7, "max_new_tokens_per_block": 256, "dtype": "bfloat16",
        "generation_config": read_json(model / "generation_config.json"),
        "generation_overrides": {"do_sample": False},
        "generation_policy": "Pinned repetition penalty as in previous latency: first block uses "
                             "prompt-plus-generated history; incremental first token is raw argmax, "
                             "subsequent tokens use new-block history. Own token-level loop/early "
                             "stop crop is paired with an own FullKV control, not claimed as the "
                             "unchanged legacy model.generate decoder.",
        "quantization": False, "thought_categories": 1, "segment_tokens": selection.SEGMENT_TOKENS,
        "retention_ladder": list(selection.RETENTION), "kmeans_max_iterations": selection.KMEANS_ITERATIONS,
        "clustering": "Per native KV head/layer, FP32 squared Euclidean Lloyd, evenly spaced seeds, "
                      "nearest cluster-member actual KV pairs; empty clusters filled without duplicates",
        "budget_policy": "Protected initial prompt plus floor(ratio * logical trajectory tokens); "
                         "oldest eligible128token segment anneals until budget/minimum4 floor; "
                         "unfinished segment and incomplete stop-string suffix protected",
        "official_reproduction": False, "ct_kernel": False,
        "scope": "Ordinary-LLM single-category TBE selection-only adaptation, NO TBQ/quantization. "
                 "Own matched RebuttalLLM greedy FullKV control. Software clustering/gather costs included. "
                 "Free rollouts, not fixed-work kernel benchmarks; synchronized diagnostics are separate.",
        "paper": "https://arxiv.org/html/2510.01290v2#A5.SS10",
        "source_hashes": hashes, "assets": assets,
        "hardware": {k: v for k, v in hardware.items() if k != "processes"},
        "versions": {"python": sys.version, "torch": torch.__version__,
                     "transformers": transformers.__version__, "cuda": torch.version.cuda},
        "cpu_threads": {k: os.environ.get(k) for k in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    }
    manifest = args.output / "manifest.json"
    if manifest.exists() and read_json(manifest)["protocol"] != protocol:
        raise RuntimeError("ThinkKV source/protocol changed; use a new output directory")
    atomic_json(manifest, {"protocol": protocol})
    checkpoint = args.output / "measurements.json"
    rows = []
    if checkpoint.exists():
        saved = read_json(checkpoint)
        rows = saved["rows"]
        if (saved["protocol_sha256"] != digest(protocol) or saved["rows_sha256"] != digest(rows)
                or [r["id"] for r in rows] != protocol["sample_ids"][:len(rows)]):
            raise RuntimeError("Invalid ThinkKV latency checkpoint")
        validate_rows(rows)
    def state(status, **extra):
        atomic_json(args.output / "status.json", {
            "status": status, "pid": os.getpid(), "completed_questions": len(rows),
            "target_questions": args.samples, "updated": time.time(), **extra})
    try:
        state("loading_model_and_retriever")
        with single_gpu_loader():
            llm = selection.ThinkKVSelectionLLM(
                str(model), enabled=args.method == "thinkkv_selection_only", ratio=args.ratio)
        retriever = WikiBM25Retriever(str(index))
        if len(retriever.titles) != 5233235 or retriever.corpus_texts is None:
            raise RuntimeError("Full Wikipedia retrieval corpus is missing")
        for _, sample in warmups:
            state("warming_up", sample_id=sample["id"])
            pass_once(base, llm, retriever, sample, False)
        for _, sample in selected[len(rows):]:
            if source_hashes() != hashes:
                raise RuntimeError("ThinkKV sources changed during measurement")
            state("measuring", sample_id=sample["id"])
            clean = [pass_once(base, llm, retriever, sample, False) for _ in range(2)]
            diagnostic = pass_once(base, llm, retriever, sample, True)
            equal = len({r["semantic_sha256"] for r in [*clean, diagnostic]}) == 1
            if not equal:
                raise RuntimeError(f"ThinkKV clean/diagnostic semantic parity failed: {sample['id']}")
            rows.append({"id": sample["id"], "clean_repeats": clean, "diagnostic": diagnostic,
                         "all_passes_semantically_equal": equal})
            validate_rows(rows[-1:])
            atomic_json(checkpoint, {"protocol_sha256": digest(protocol), "rows_sha256": digest(rows),
                                     "rows": rows, "summary": summary(rows)})
        state("complete")
        validate_run(args.output)
        export_points(args.output, rows)
    except BaseException:
        state("failed", traceback=traceback.format_exc())
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--ratio", type=float, choices=(.2, .5), default=.2)
    parser.add_argument("--model-ready", type=Path, required=True)
    parser.add_argument("--data-ready", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    if not 1 <= args.samples <= 20 or args.repeats != 2:
        parser.error("Use1..20 questions and exactly two clean repeats")
    if args.execute:
        execute(args)
    else:
        print(f"Planned {args.dataset}/{args.method}: {args.samples} questions, BF16 KV; "
              f"ratio={args.ratio if args.method == 'thinkkv_selection_only' else 1}; NO quantization")


if __name__ == "__main__":
    main()
