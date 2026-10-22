"""Prepare, but do not automatically execute, exclusive-GPU HotpotQA latency runs.

An actual experiment requires --execute and exactly one externally selected GPU.
Each method runs in a separate process/output directory. No queue is launched.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.latency_runtime import (
    EpisodeTrace, METHODS, MODEL, canonical, digest, experiment_hashes, file_hash,
    frozen_method, instrument_episode, native_equivalent_budget, observe_kv_gathers, patch_attribute,
    process_memory_reader, read_json, require_exclusive, single_gpu_loader,
    validate_diagnostics,
)
from scripts.qa32_runtime import QAErrorWatch
from scripts.run_qa32_rebuttal import atomic_json
from scripts.run_rebuttal_efficiency import cache_digest, semantic_hash, semantic_record

SCHEMA = 1


def semantic_state(result, llm):
    record = semantic_record(result, llm)
    if hasattr(llm, "_legacy_recurrence"):
        pairs = [(tensor, tensor) for _, state in sorted(llm._legacy_recurrence.items())
                 for tensor in (state.ids, state.last, state.mri)]
        record["lazy_recurrence"] = cache_digest(SimpleNamespace(past_key_values=pairs))
        record["lazy_clock"] = llm._legacy_clock
        record["lazy_next_query_id"] = llm._legacy_next_query_id
        record["lazy_evicted_slots"] = llm._legacy_evicted_slots
    if hasattr(llm, "_flowkv_frozen_ids"):
        record["flowkv_frozen_ids"] = list(llm._flowkv_frozen_ids)
    return record


class QuietStream(io.TextIOBase):
    def write(self, text):
        return len(text)

    def writable(self):
        return True


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=list(METHODS), required=True)
    parser.add_argument("--ratio", type=float, choices=[.2, .5], default=.2)
    parser.add_argument("--model-ready", type=Path, required=True)
    parser.add_argument("--data-ready", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--execute", action="store_true", help="Opt in to model loading/inference; otherwise print plan only")
    return parser


def prepare_plan(args):
    if not 1 <= args.samples <= 500 or args.repeats < 2:
        raise ValueError("Use 1..500 fixed questions and at least two timing repeats")
    return {
        "schema_version": SCHEMA, "status": "planned_not_executed", "method": args.method,
        "ratio": 1.0 if args.method == "fullkv" else args.ratio,
        "model": MODEL, "dataset": "hotpotqa", "seed": 233,
        "sample_positions": list(range(args.samples)), "warmup_positions": [500, 501],
        "clean_repeats_per_question": args.repeats, "diagnostic_repeats_per_question": 1,
        "max_steps": 7, "max_new_tokens_per_block": 256, "dtype": "bfloat16",
        "model_ready": str(args.model_ready.resolve()), "data_ready": str(args.data_ready.resolve()),
        "output": str(args.output.resolve()), "requires_exclusive_gpu": True,
        "methods_available": list(METHODS),
        "lazy_budget_warning": "Slots per query head, NOT the same bytes as native GQA; "
                               "Qwen2.5-7B stores 28 rather than 4 heads (7x), including prompt.",
        "comparison": "Free agent rollouts, not fixed-work kernel benchmarks; report steps and actual input positions.",
    }


def load_assets(model_ready, data_ready):
    model, data = read_json(model_ready), read_json(data_ready)
    reference = read_json(ROOT / "config/rebuttal_assets_20260928.json")
    if (model["repo"], model["revision"]) != MODEL:
        raise RuntimeError("Require the pinned Qwen2.5-7B-Instruct checkpoint")
    model_path = Path(model["path"]).resolve()
    config = read_json(model_path / "config.json")
    expected = {"model_type": "qwen2", "num_hidden_layers": 28,
                "num_attention_heads": 28, "num_key_value_heads": 4, "hidden_size": 3584}
    if any(config.get(key) != value for key, value in expected.items()) or config.get("quantization_config"):
        raise RuntimeError("Require ordinary, unquantized Qwen2.5-7B")
    for name in ("hotpotqa", "longrag"):
        if any(data[name][key] != reference["data"][name][key] for key in ("repo", "revision")):
            raise RuntimeError(f"Pinned data revision mismatch: {name}")
    files = sorted((Path(data["hotpotqa"]["path"]) / "distractor").glob("validation-*.parquet"))
    if not files:
        raise RuntimeError("Pinned HotpotQA validation parquet files are missing")
    index = Path(data["wiki_index"]).resolve()
    if read_json(index / "meta.json")["num_docs"] != 5233235:
        raise RuntimeError("Require the full 5,233,235-document Wikipedia index")
    index_hashes = {}
    for name in ("meta.json", "titles.json", "corpus_texts.jsonl", "params.index.json"):
        path = index / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing full-corpus retrieval asset: {path}")
        stat = path.stat()
        index_hashes[name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    build = index / "rebuttal_build_manifest.json"
    if build.exists():
        index_hashes["build_manifest_sha256"] = file_hash(build)
    weight_index = read_json(model_path / "model.safetensors.index.json")
    weights = {}
    for name in sorted(set(weight_index["weight_map"].values())):
        if Path(name).name != name:
            raise RuntimeError("Weight shard path escapes the model directory")
        path = model_path / name
        stat = path.stat()
        weights[name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    hashes = {name: file_hash(model_path / name) for name in (
        "config.json", "generation_config.json", "tokenizer_config.json",
        "tokenizer.json", "model.safetensors.index.json",
    )}
    provenance = {
        "model_ready_sha256": file_hash(model_ready), "data_ready_sha256": file_hash(data_ready),
        "model": model, "data": data, "model_config_hashes": hashes, "weight_shard_inventory": weights,
        "parquet_sha256": {str(path.resolve()): file_hash(path) for path in files},
        "retrieval_index_inventory": index_hashes,
        "index_provenance_scope": "Pinned corpus revision/build manifest plus size/mtime inventory; "
                                  "not a fresh hash of every multi-GB index payload.",
    }
    return model_path, index, files, provenance


def select_samples(base, data, count):
    if len(data) != 7405:
        raise RuntimeError("Pinned HotpotQA validation split must contain 7405 rows")
    base.RANDOM_SEED, base.NUM_SAMPLES = 233, 502
    ordered = base.select_samples(data)
    selected, warmups = ordered[:count], ordered[500:502]
    ids = [sample["id"] for _, sample in selected + warmups]
    if len(ids) != count + 2 or len(ids) != len(set(ids)):
        raise RuntimeError("Main/warmup sample selection is incomplete or overlaps")
    return selected, warmups


def cuda_allocator(model_bytes):
    import torch

    result = {
        "allocated_bytes": torch.cuda.memory_allocated(0),
        "reserved_bytes": torch.cuda.memory_reserved(0),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
        "model_storage_bytes": model_bytes,
    }
    for prefix in ("allocated", "reserved", "peak_allocated", "peak_reserved"):
        value = result[prefix + "_bytes"] - model_bytes
        if value < 0:
            raise RuntimeError("Allocator usage is below the deduplicated model storage baseline")
        result[prefix + "_above_model_bytes"] = value
    return result


def run_once(base, llm, retriever, method, question):
    import torch
    from scripts.latency_metrics import model_storage_bytes

    require_exclusive()
    stdout, stderr = QAErrorWatch(QuietStream()), QAErrorWatch(QuietStream())
    original_reset, original_tool = llm.reset, base.execute_action
    tool_seconds, tool_calls = 0.0, 0
    reset_calls = 0
    baseline = model_storage_bytes(llm.model).get("cuda:0")
    if baseline is None:
        raise RuntimeError("No single-GPU model storage baseline")

    def reset(*args, **kwargs):
        nonlocal reset_calls, baseline
        result = original_reset(*args, **kwargs)
        baseline = model_storage_bytes(llm.model)["cuda:0"]
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        reset_calls += 1
        return result

    def tool(*args, **kwargs):
        nonlocal tool_seconds, tool_calls
        start = time.perf_counter()
        try:
            return original_tool(*args, **kwargs)
        finally:
            tool_seconds += time.perf_counter() - start
            tool_calls += 1

    torch.cuda.synchronize(0)
    with patch_attribute(llm, "reset", reset), patch_attribute(base, "execute_action", tool):
        start = time.perf_counter()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            result = base._run_react_kv_episode(
                question, llm, retriever, pruning_mode=method.mode, return_debug=True)
        torch.cuda.synchronize(0)
        elapsed = time.perf_counter() - start
    if reset_calls != 1:
        raise RuntimeError(f"Expected one original reset boundary, observed {reset_calls}")
    if stdout.errors or stderr.errors:
        raise RuntimeError(f"Frozen evaluator suppressed an error: {(stdout.errors + stderr.errors)[:4]}")
    if not 0 <= tool_seconds <= elapsed:
        raise RuntimeError("Invalid disjoint tool/episode timing")
    current_model_bytes = model_storage_bytes(llm.model)["cuda:0"]
    memory = cuda_allocator(baseline)
    memory["final_model_storage_bytes"] = current_model_bytes
    memory["model_storage_changed_since_reset"] = current_model_bytes != baseline
    require_exclusive()
    return result, {
        "episode_seconds": elapsed, "tool_seconds": tool_seconds,
        "episode_excluding_tool_seconds": elapsed - tool_seconds, "tool_calls": tool_calls,
        "memory": memory,
        "timing_scope": "Whole episode synchronized only at outer boundaries; coarse CPU tool spans. "
                        "No fine model/phase hooks. Logging is swallowed-error checked but not written.",
        "peak_memory_scope": "Peaks reset after original episode reset; above-model peaks subtract "
                             "that reset-time parameter+buffer storage, not claimed to be KV bytes.",
    }


def condition_summary(rows):
    import statistics

    if not rows:
        return {"completed_questions": 0}
    seconds = [statistics.mean(repeat["episode_seconds"] for repeat in row["clean_repeats"]) for row in rows]
    return {
        "completed_questions": len(rows), "mean_episode_seconds": statistics.mean(seconds),
        "median_episode_seconds": statistics.median(seconds),
        "mean_em": statistics.mean(row["em"] for row in rows),
        "mean_f1": statistics.mean(row["f1"] for row in rows),
        "aggregation": "Mean repeats within question, then equal question weighting. "
                       "Repeats are timing noise estimates, not independent samples/seeds.",
    }


def validate_committed(data, protocol):
    if data.get("protocol_sha256") != digest(protocol):
        raise RuntimeError("Measurement checkpoint belongs to a different protocol")
    rows = data["rows"]
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)) or ids != protocol["sample_ids"][:len(ids)]:
        raise RuntimeError("Measurement checkpoint IDs must be an ordered unique prefix")
    for row in rows:
        if row["row_sha256"] != digest({key: value for key, value in row.items() if key != "row_sha256"}):
            raise RuntimeError(f"Corrupt committed measurement: {row['id']}")
        if len(row["clean_repeats"]) != protocol["clean_repeats_per_question"] or row["parity"] is not True:
            raise RuntimeError("Unverified or incomplete measurement cannot be committed")
        validate_diagnostics(row["diagnostics"], row["interaction_boundaries"], row["generation_blocks"])
    return rows


def execute(args, plan):
    import torch
    import transformers
    from datasets import load_dataset
    from scripts.latency_metrics import DetailedLatencyMonitor
    from retrievers.WikiBM25Retriever import WikiBM25Retriever

    hardware = require_exclusive()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one externally selected CUDA device is required")
    hashes = experiment_hashes()
    model_path, index, files, assets = load_assets(args.model_ready, args.data_ready)
    base = importlib.import_module("run_all_wiki_experiments_v2")
    base.MAX_STEPS = 7
    data = load_dataset("parquet", data_files={"validation": list(map(str, files))},
                        split="validation", cache_dir=str(args.output.parent / "latency_arrow_cache"))
    selected, warmups = select_samples(base, data, args.samples)
    protocol = {
        **{key: value for key, value in plan.items() if key not in ("status", "output")},
        "sample_ids": [sample["id"] for _, sample in selected],
        "warmup_ids": [sample["id"] for _, sample in warmups],
        "hardware": {key: value for key, value in hardware.items() if key != "processes"},
        "versions": {"python": sys.version, "torch": torch.__version__,
                     "transformers": transformers.__version__, "cuda": torch.version.cuda},
        "cpu_threads": {name: os.environ.get(name) for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        "source_hashes": hashes, "assets": assets,
        "generation_config": read_json(model_path / "generation_config.json"),
        "measurement_scope": "Free-rollout clean headline + parity-checked synchronized diagnostics; "
                             "interaction endpoints after tool return, generation endpoints after original pruning, "
                             "final memory after episode-frame release. No source repair or extra inference.",
        "exclusivity_scope": "All selected-GPU processes checked before/after every pass and at "
                             "diagnostic memory endpoints; very short processes between samples cannot be ruled out.",
    }
    with frozen_method(args.method) as method:
        protocol["method_config"] = method.config(base, args.ratio)
        protocol["method_scope"] = method.scope
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".latency.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest_path, checkpoint_path = args.output / "manifest.json", args.output / "measurements.json"
        if manifest_path.exists():
            if read_json(manifest_path)["protocol"] != json.loads(canonical(protocol)):
                raise RuntimeError("Protocol/source/hardware changed; use a new output directory")
        else:
            if any(path.name != ".latency.lock" for path in args.output.iterdir()):
                raise RuntimeError("Nonempty output has no experiment manifest")
            atomic_json(manifest_path, {"protocol": protocol, "created": time.time()})
        rows = validate_committed(read_json(checkpoint_path), protocol) if checkpoint_path.exists() else []

        def state(status, **extra):
            atomic_json(args.output / "status.json", {
                "status": status, "pid": os.getpid(), "updated": time.time(),
                "completed_questions": len(rows), "target_questions": len(selected), **extra,
            })

        def save():
            atomic_json(checkpoint_path, {"schema_version": SCHEMA, "protocol_sha256": digest(protocol),
                                         "rows": rows, "summary": condition_summary(rows)})

        try:
            state("loading_model_and_retriever")
            with frozen_method(args.method) as method, single_gpu_loader():
                config = method.config(base, args.ratio)
                from token_tracker import TokenTracker
                llm = method.llm_class(str(model_path), kv_config=config, token_tracker=TokenTracker())
                retriever = WikiBM25Retriever(str(index))
                if len(retriever.titles) != 5233235 or retriever.corpus_texts is None:
                    raise RuntimeError("Full Wikipedia retriever is missing corpus text")
                for _, sample in warmups:
                    state("warming_up", sample_id=sample["id"])
                    run_once(base, llm, retriever, method, sample["question"])
                for _, sample in selected[len(rows):]:
                    if experiment_hashes() != hashes:
                        raise RuntimeError("Frozen experiment source changed during run")
                    state("measuring", sample_id=sample["id"])
                    repeats, reference, result = [], None, None
                    for _ in range(args.repeats):
                        result, measurement = run_once(base, llm, retriever, method, sample["question"])
                        candidate = semantic_state(result, llm)
                        if reference is not None and candidate != reference:
                            raise RuntimeError(f"Clean repeat trajectory/KV parity failed: {sample['id']}")
                        reference = candidate
                        repeats.append(measurement)
                    with DetailedLatencyMonitor(llm, process_memory_reader=process_memory_reader) as monitor:
                        instrument_episode(base, method, llm, monitor)
                        with observe_kv_gathers(llm, monitor), EpisodeTrace(base._run_react_kv_episode, monitor) as trace:
                            diagnostic, diagnostic_wall = run_once(
                                base, llm, retriever, method, sample["question"])
                        stats = monitor.snapshot()
                        stats["parent_attribution"] = {
                            "inline_active_kv_gathers": ["Tensor.gather", "Tensor.index_select"],
                            "gather_scope": "Input storage must alias currently live active K/V; "
                                            "generation/logits gathers excluded. Metadata gather_state is separate.",
                            "utility_scope": "AST-identified reward/novelty/repeat and citation-history blocks",
                            "limitations": "Fused C/CUDA operators not exposed by these Python calls remain "
                                           "inside enclosing forward/selection time; no estimate is subtracted.",
                        }
                        interactions = trace.boundaries
                    if semantic_state(diagnostic, llm) != reference:
                        atomic_json(args.output / "parity_failure.json", {
                            "id": sample["id"], "reference": reference,
                            "diagnostic": semantic_state(diagnostic, llm),
                        })
                        raise RuntimeError(f"Diagnostic trajectory/KV parity failed: {sample['id']}")
                    blocks = len(result[2])
                    validate_diagnostics(stats, interactions, blocks)
                    measured = sum(stats["timing"]["exclusive_seconds"].values())
                    residual = diagnostic_wall["episode_seconds"] - measured
                    if residual < -1e-6:
                        raise RuntimeError("Exclusive diagnostic phase intervals double-count episode time")
                    audit = llm.legacy_audit() if hasattr(llm, "legacy_audit") else None
                    clock = (audit["processed_nonprompt_queries"] if args.method == "lazyeviction"
                             else llm.token_tracker.next_global_id - stats["final"]["kv"]["protected_tokens"])
                    row = {
                        "id": sample["id"], "clean_repeats": repeats, "diagnostics": stats,
                        "interaction_boundaries": interactions, "generation_blocks": blocks,
                        "trajectory_entries": len(result[1]), "predicted_answer": result[0],
                        "em": base.exact_match(result[0], sample["answer"]),
                        "f1": base.f1_score(result[0], sample["answer"]),
                        "trajectory": result[1], "semantic_sha256": semantic_hash(reference), "parity": True,
                        "model_runtime": {
                            "class": type(llm.model).__name__,
                            "attention_implementation": llm.model.config._attn_implementation,
                            "generation_config": llm.model.generation_config.to_dict(),
                        },
                        "diagnostic_wall": diagnostic_wall,
                        "diagnostic_unattributed_seconds": max(0.0, residual),
                        "diagnostic_unattributed_scope": "Python agent work, tracing/sampling/synchronization "
                                                         "and other instrumentation; not called algorithm overhead.",
                        "legacy_adapter_audit": audit, "method_scope": method.scope,
                        "nominal_native_byte_reference": native_equivalent_budget(
                            stats["final"], 1.0 if args.method == "fullkv" else args.ratio, clock),
                    }
                    row["row_sha256"] = digest(row)
                    rows.append(row)
                    save()
                validate_committed(read_json(checkpoint_path), protocol)
                if len(rows) != len(selected):
                    raise RuntimeError("Incomplete measurement task set")
                state("complete")
        except BaseException:
            state("failed", traceback=traceback.format_exc())
            raise


def main(argv=None):
    args = build_parser().parse_args(argv)
    plan = prepare_plan(args)
    if not args.execute:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0
    execute(args, plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
