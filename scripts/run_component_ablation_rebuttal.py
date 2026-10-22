#!/usr/bin/env python3
"""Isolated, resumable Qwen7B/Llama8B component ablations using the original evaluator.

Without --execute, emit a CPU-only plan (no assets/weights loaded or output written).
Main uses seed-specific positions 1..500; smoke uses positions 501/502. Every
variant, including full StepKV, is freshly evaluated; no historical result import.
Shared-GPU elapsed times are diagnostic, NOT latency measurements. Checkpoints
are authoritative and bound to sources, assets, configuration and selected rows.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VARIANTS = ("full", "no_repeat", "no_novelty", "no_success", "no_cite", "no_token", "no_step")
DATASETS = {"hotpotqa": 7405, "2wiki": 12576, "musique": 2417}
MODE = "step_aware_h2o"
SCHEMA = 1
MODELS = {
    "Qwen/Qwen2.5-7B-Instruct": {
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "config": {"model_type": "qwen2", "num_hidden_layers": 28,
                   "num_attention_heads": 28, "num_key_value_heads": 4, "hidden_size": 3584},
    },
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {
        "revision": "0e9e39f249a16976918f6564b8830bc894c89659",
        "config": {"model_type": "llama", "num_hidden_layers": 32,
                   "num_attention_heads": 32, "num_key_value_heads": 8, "hidden_size": 4096},
    },
}


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hashes():
    """CPU-only binding of frozen evaluator/core, asset helpers and this worker."""
    manifest = read_json(ROOT / "rebuttal/adapters/manifest.json")
    expected = manifest["methods"]["h2o_step"]["original_core_hashes"]
    for name, checksum in expected.items():
        if file_hash(ROOT / name) != checksum:
            raise RuntimeError(f"Frozen source mismatch: {name}")
    names = set(expected) | {
        "scripts/run_component_ablation_rebuttal.py", "run_stepkv_component_ablation.py",
        "run_all_2wiki_experiments_v2.py", "run_all_musique_experiments_v2.py",
        "analyze_run_kv_metrics.py", "scripts/run_latency_experiment.py",
        "scripts/latency_datasets.py", "scripts/latency_runtime.py",
        "scripts/qa32_runtime.py", "scripts/run_qa32_rebuttal.py",
        "scripts/run_rebuttal_efficiency.py", "scripts/legacy_efficiency_metrics.py",
        "rebuttal/adapters/manifest.json", "config/rebuttal_assets_20260928.json",
    }
    names.update(str(path.relative_to(ROOT)) for path in (ROOT / "retrievers").glob("*.py"))
    names.update(str(path.relative_to(ROOT)) for directory in ("models", "kv_cache")
                 for path in (ROOT / directory).glob("*.py"))
    names.add("token_tracker.py")
    return {name: file_hash(ROOT / name) for name in sorted(names)}


@contextlib.contextmanager
def patch_attribute(obj, name, value):
    own, previous = name in vars(obj), vars(obj).get(name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if own:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)


def variant_override(variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown component variant: {variant}")
    original = importlib.import_module("run_stepkv_component_ablation")
    return {**original._base_kv_override(.5), **original.ABLATION_SPECS[variant]["override"],
            "step_ablation_name": variant}


def component_config(base, variant):
    """Collect original shared Hotpot defaults via its no-samples/no-model path."""
    def collect(results, output, mode, config, *args, **kwargs):
        if results or output or mode != MODE:
            raise RuntimeError("Unexpected original configuration collection")
        return dict(config)

    with patch_attribute(base, "_finalize_react_kv_output", collect):
        config = base.run_react_kv_experiment(
            [], [], None, MODE, "", "", kv_config_override=variant_override(variant))
    if config["pruning_mode"] != MODE or config["cache_ratio"] != .5:
        raise RuntimeError("Component defaults no longer describe StepKV")
    return config


def protocol(args):
    if args.dataset not in DATASETS or args.seed not in (233, 3407) or args.variant not in VARIANTS:
        raise ValueError("Unsupported component condition")
    if args.purpose not in ("main", "smoke"):
        raise ValueError("Unsupported purpose")
    start, count = (0, 500) if args.purpose == "main" else (500, 2)
    return {"dataset": args.dataset, "seed": args.seed, "variant": args.variant,
            "purpose": args.purpose, "sample_start": start, "samples": count,
            "max_steps": 7, "ratio": .5, "pruning_mode": MODE,
            "dtype": "bfloat16", "greedy": True, "memory_limit_gib": 44,
            "device": "cuda:0", "shared_gpu_allowed": True,
            "full_control_policy": "fresh seed-aligned full StepKV, never imported"}


def select_samples(base, rows, *, seed, purpose, expected_rows=None):
    if seed not in (233, 3407) or purpose not in ("main", "smoke"):
        raise ValueError("Invalid selection protocol")
    if expected_rows is not None and len(rows) != expected_rows:
        raise RuntimeError("Unexpected dataset row count")
    ids = []
    for row in rows:
        for key in ("id", "question", "answer"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise RuntimeError(f"Invalid dataset {key}")
        ids.append(row["id"])
    if len(ids) != len(set(ids)) or len(rows) < 502:
        raise RuntimeError("Duplicate IDs or insufficient dataset rows")
    with patch_attribute(base, "RANDOM_SEED", seed), patch_attribute(base, "NUM_SAMPLES", 502):
        ordered = base.select_samples(rows)
    if len(ordered) != 502 or len({row["id"] for _, row in ordered}) != 502:
        raise RuntimeError("Original selection returned incomplete/duplicate IDs")
    return ordered[:500] if purpose == "main" else ordered[500:502]


def load_assets(args):
    from scripts.latency_datasets import normalized_assets

    ready, data = read_json(args.model_ready), read_json(args.data_ready)
    pinned = MODELS.get(ready.get("repo"))
    if pinned is None or ready.get("revision") != pinned["revision"]:
        raise RuntimeError("Require the pinned Qwen7B or Llama3.1-8B model revision")
    model = Path(ready["path"]).resolve()
    config = read_json(model / "config.json")
    if any(config.get(key) != value for key, value in pinned["config"].items()) or config.get("quantization_config"):
        raise RuntimeError("Unexpected or quantized model topology")
    index = read_json(model / "model.safetensors.index.json")
    inventory = {}
    for name in sorted(set(index["weight_map"].values())):
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise RuntimeError("Invalid model weight shard path")
        stat = (model / name).stat()
        inventory[name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if (not inventory or any(row["bytes"] <= 0 for row in inventory.values())
            or sum(row["bytes"] for row in inventory.values()) < index["metadata"]["total_size"]):
        raise RuntimeError("Incomplete model shard inventory")
    hashes = {name: file_hash(model / name) for name in (
        "config.json", "generation_config.json", "tokenizer_config.json",
        "tokenizer.json", "model.safetensors.index.json",
    )}
    for name, expected in ready.get("config_hashes", {}).items():
        if Path(name).name != name or file_hash(model / name) != expected:
            raise RuntimeError("Model metadata differs from its prepared asset record")
    reference = read_json(ROOT / "config/rebuttal_assets_20260928.json")
    for name in ("hotpotqa", "longrag"):
        if any(data[name][key] != reference["data"][name][key] for key in ("repo", "revision")):
            raise RuntimeError(f"Pinned data revision mismatch: {name}")
    files = sorted((Path(data["hotpotqa"]["path"]) / "distractor").glob("validation-*.parquet"))
    if not files:
        raise RuntimeError("Missing pinned HotpotQA validation parquet")
    wiki = Path(data["wiki_index"]).resolve()
    if read_json(wiki / "meta.json")["num_docs"] != 5233235:
        raise RuntimeError("Require the full Wikipedia retrieval index")
    wiki_inventory = {}
    for name in ("meta.json", "titles.json", "corpus_texts.jsonl", "params.index.json"):
        stat = (wiki / name).stat()
        if stat.st_size <= 0:
            raise RuntimeError("Empty full-Wikipedia retrieval asset")
        wiki_inventory[name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    provenance = {
        "model": ready, "data": data, "model_repo": ready["repo"], "model_revision": ready["revision"],
        "model_ready_sha256": file_hash(args.model_ready), "data_ready_sha256": file_hash(args.data_ready),
        "model_config_hashes": hashes, "weight_shard_inventory": inventory,
        "parquet_sha256": {str(path.resolve()): file_hash(path) for path in files},
        "retrieval_index_inventory": wiki_inventory,
        "weight_integrity_scope": "Pinned revision, metadata hashes and shard size/mtime inventory; "
                                  "not a fresh hash of every weight shard",
    }
    if args.dataset == "hotpotqa":
        from datasets import load_dataset
        rows = list(load_dataset("parquet", data_files={"validation": [str(p) for p in files]},
                                 split="validation"))
    else:
        _, rows, additional = normalized_assets(args.data_ready, args.dataset)
        provenance = {**provenance, **additional}
    return model, wiki, rows, provenance, read_json(model / "generation_config.json")


def prepare_plan(args):
    return {"schema_version": SCHEMA, "status": "planned_not_executed", **protocol(args),
            "source_hashes": source_hashes(), "kv_override": variant_override(args.variant),
            "model_ready": str(args.model_ready.resolve()), "data_ready": str(args.data_ready.resolve()),
            "output": str(args.output.resolve()), "timing_scope": "diagnostic shared-GPU accuracy rollout"}


def summarize(rows):
    n = len(rows)
    return {"total_samples": n, "exact_match": 100 * sum(row["em"] for row in rows) / n if n else 0,
            "f1_score": 100 * sum(row["f1"] for row in rows) / n if n else 0,
            "total_diagnostic_seconds": sum(row["diagnostic_elapsed_seconds"] for row in rows)}


def validate_fingerprint(fingerprint):
    spec = protocol(argparse.Namespace(**fingerprint["protocol"]))
    if fingerprint["protocol"] != spec or fingerprint["source_hashes"] != source_hashes():
        raise RuntimeError("Source/protocol fingerprint mismatch")
    base = importlib.import_module("run_all_wiki_experiments_v2")
    if fingerprint["kv_config"] != component_config(base, spec["variant"]):
        raise RuntimeError("Original component configuration mismatch")
    ids, indices = fingerprint["sample_ids"], fingerprint["sample_indices"]
    selected = fingerprint["selected_samples"]
    if (len(ids) != spec["samples"] or len(set(ids)) != len(ids)
            or len(set(indices)) != len(ids) or len(selected) != len(ids)
            or ids != [row["id"] for _, row in selected]
            or indices != [index for index, _ in selected]
            or digest(selected) != fingerprint["selected_content_sha256"]):
        raise RuntimeError("Sample fingerprint mismatch")
    for index, sample in selected:
        if type(index) is not int or not 0 <= index < DATASETS[spec["dataset"]]:
            raise RuntimeError("Invalid original sample index")
        if any(not isinstance(sample.get(key), str) or not sample[key].strip()
               for key in ("id", "question", "answer")):
            raise RuntimeError("Invalid sample fingerprint content")


def validate_rows(rows, fingerprint):
    if not isinstance(rows, list) or len(rows) > len(fingerprint["sample_ids"]):
        raise RuntimeError("Corrupt checkpoint rows")
    for position, row in enumerate(rows):
        index, sample = fingerprint["selected_samples"][position]
        if (row.get("status") != "success" or row.get("id") != sample["id"]
                or row.get("index") != index or row.get("question") != sample["question"]
                or row.get("gold_answer") != sample["answer"]
                or not isinstance(row.get("predicted_answer"), str)):
            raise RuntimeError("Checkpoint sample/status mismatch")
        for key in ("em", "f1", "diagnostic_elapsed_seconds"):
            value = row.get(key)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise RuntimeError("Invalid checkpoint metrics")
        if row["em"] not in (0, 1) or row["f1"] > 1:
            raise RuntimeError("Invalid checkpoint scores")
        base = importlib.import_module("run_all_wiki_experiments_v2")
        if (row["em"] != base.exact_match(row["predicted_answer"], sample["answer"])
                or row["f1"] != base.f1_score(row["predicted_answer"], sample["answer"])):
            raise RuntimeError("Checkpoint scores differ from original scorer")
        audit = row.get("pruning_audit")
        if not isinstance(audit, list):
            raise RuntimeError("Missing actual tensor pruning audit")
        for event in audit:
            before, after = event["before"], event["after"]
            if (not before or len(before) != len(after)
                    or any(type(n) is not int or n < 0 for n in before + after)
                    or len(set(before)) != 1 or len(set(after)) != 1
                    or any(a > b for a, b in zip(after, before))):
                raise RuntimeError("Invalid actual tensor eviction audit")
        if row.get("has_actual_eviction") is not any(
                event["after"][0] < event["before"][0] for event in audit):
            raise RuntimeError("Actual eviction flag mismatch")


class ResultStore:
    """Atomic checkpoint-first progress; derived results may be rebuilt on resume."""
    def __init__(self, output, fingerprint):
        from scripts.run_qa32_rebuttal import atomic_json
        self.output, self.fingerprint = Path(output), fingerprint
        self.atomic_json = atomic_json
        validate_fingerprint(fingerprint)
        self.hash = digest(fingerprint)
        self.rows = []
        checkpoint, manifest = self.output / "checkpoint.json", self.output / "manifest.json"
        if manifest.exists():
            if read_json(manifest) != {"schema_version": SCHEMA, "fingerprint": fingerprint,
                                       "fingerprint_sha256": self.hash}:
                raise RuntimeError("Existing manifest fingerprint mismatch")
        elif checkpoint.exists() or (self.output / "results.json").exists():
            raise RuntimeError("Missing manifest for existing output")
        if checkpoint.exists():
            saved = read_json(checkpoint)
            if (saved.get("schema_version") != SCHEMA or saved.get("fingerprint_sha256") != self.hash
                    or saved.get("rows_sha256") != digest(saved.get("rows"))):
                raise RuntimeError("Corrupt/mismatched checkpoint")
            validate_rows(saved["rows"], fingerprint)
            self.rows = saved["rows"]
        elif (self.output / "results.json").exists():
            raise RuntimeError("Missing authoritative checkpoint")
        self.atomic_json(manifest, {"schema_version": SCHEMA, "fingerprint": fingerprint,
                                   "fingerprint_sha256": self.hash})

    def commit(self, row):
        rows = self.rows + [row]
        validate_rows(rows, self.fingerprint)
        self.atomic_json(self.output / "checkpoint.json", {
            "schema_version": SCHEMA, "fingerprint_sha256": self.hash,
            "rows": rows, "rows_sha256": digest(rows)})
        self.rows = rows
        self.atomic_json(self.output / "status.json", {
            "status": "running", "completed_samples": len(rows), "last_successful_id": row["id"]})

    def finalize(self):
        if len(self.rows) != len(self.fingerprint["sample_ids"]):
            raise RuntimeError("Cannot finalize an incomplete run")
        result = {"schema_version": SCHEMA, "fingerprint_sha256": self.hash,
                  "rows": self.rows, "summary": summarize(self.rows)}
        self.atomic_json(self.output / "results.json", result)
        self.atomic_json(self.output / "status.json", {
            "status": "complete", "completed_samples": len(self.rows), "fingerprint_sha256": self.hash})
        return result


def validate_completed_run(output):
    """Read-only CPU validation; never load model weights or invoke CUDA."""
    try:
        output = Path(output)
        manifest = read_json(output / "manifest.json")
        fingerprint = manifest["fingerprint"]
        validate_fingerprint(fingerprint)
        checksum = digest(fingerprint)
        checkpoint = read_json(output / "checkpoint.json")
        result, status = read_json(output / "results.json"), read_json(output / "status.json")
        rows = checkpoint["rows"]
        validate_rows(rows, fingerprint)
        if (manifest != {"schema_version": SCHEMA, "fingerprint": fingerprint,
                         "fingerprint_sha256": checksum}
                or checkpoint.get("schema_version") != SCHEMA
                or checkpoint.get("fingerprint_sha256") != checksum
                or checkpoint.get("rows_sha256") != digest(rows)
                or len(rows) != fingerprint["protocol"]["samples"]
                or result != {"schema_version": SCHEMA, "fingerprint_sha256": checksum,
                              "rows": rows, "summary": summarize(rows)}
                or status.get("status") != "complete"
                or status.get("completed_samples") != len(rows)
                or status.get("fingerprint_sha256") != checksum):
            raise RuntimeError("Incomplete/corrupt completed run")
        return {"completed_samples": len(rows), "sample_ids": fingerprint["sample_ids"],
                "has_actual_eviction": any(row["has_actual_eviction"] for row in rows),
                "summary": result["summary"]}
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        raise RuntimeError(f"Invalid completed run: {error}") from error


@contextlib.contextmanager
def pruning_observer(llm):
    """Observe selector input/output tensor lengths only; no hooks/timers/sync."""
    from scripts.legacy_efficiency_metrics import cache_tensor_pairs
    events, errors = [], []
    strategy = llm.kv_manager.pruning_strategy
    original = strategy.prune

    def lengths(cache):
        pairs = cache_tensor_pairs(cache)
        if not pairs or any(k.shape[-2] != v.shape[-2] for k, v in pairs):
            raise RuntimeError("Missing/mismatched real KV tensors")
        return [int(k.shape[-2]) for k, _ in pairs]

    def prune(past_key_values, *args, **kwargs):
        try:
            before = lengths(past_key_values)
            result = original(past_key_values, *args, **kwargs)
            events.append({"before": before, "after": lengths(result[0])})
            return result
        except BaseException as error:
            errors.append(f"{type(error).__name__}: {error}")
            raise

    with patch_attribute(strategy, "prune", prune):
        yield events, errors


def run_episode(base, sample, llm, retriever):
    from scripts.qa32_runtime import QAErrorWatch
    stdout, stderr = QAErrorWatch(sys.stdout), QAErrorWatch(sys.stderr)
    start = time.perf_counter()
    with pruning_observer(llm) as (events, errors), \
            contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        answer, trajectory, timings, debug = base._run_react_kv_episode(
            sample["question"], llm, retriever, pruning_mode=MODE, return_debug=True)
    if stdout.errors or stderr.errors or errors:
        raise RuntimeError(f"Original episode suppressed errors: {stdout.errors + stderr.errors + errors}")
    if not isinstance(answer, str):
        raise RuntimeError("Original evaluator returned invalid answer")
    return {"status": "success", "predicted_answer": answer,
            "em": base.exact_match(answer, sample["answer"]), "f1": base.f1_score(answer, sample["answer"]),
            "num_steps": len(trajectory), "trajectory": trajectory,
            "legacy_step_timings_bookkeeping_only": timings, "legacy_debug": debug,
            "diagnostic_elapsed_seconds": time.perf_counter() - start,
            "pruning_audit": events,
            "has_actual_eviction": any(e["after"][0] < e["before"][0] for e in events)}


def configure_cuda():
    import torch
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.strip() or "," in visible or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Inherit exactly one visible GPU; use logical cuda:0")
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 44 * 1024 ** 3 / total), 0)


def run(args):
    from scripts.latency_runtime import single_gpu_loader
    import torch
    import transformers

    if torch.__version__.split("+")[0] != "2.10.0" or transformers.__version__ != "5.0.0":
        raise RuntimeError("Require isolated torch 2.10.0 / transformers 5.0.0 runtime")
    spec, sources = protocol(args), source_hashes()
    model, wiki, rows, assets, generation = load_assets(args)
    base = importlib.import_module("run_all_wiki_experiments_v2")
    selected = select_samples(base, rows, seed=args.seed, purpose=args.purpose,
                              expected_rows=DATASETS[args.dataset])
    selected_content = [[index, {key: row[key] for key in ("id", "question", "answer")}]
                        for index, row in selected]
    config = component_config(base, args.variant)
    fingerprint = {"protocol": spec, "source_hashes": sources, "kv_config": config,
                   "sample_ids": [row["id"] for _, row in selected],
                   "sample_indices": [index for index, _ in selected],
                   "selected_samples": selected_content, "selected_content_sha256": digest(selected_content),
                   "assets": assets, "generation_config": generation,
                   "versions": {name: importlib.metadata.version(name) for name in
                                ("torch", "transformers", "accelerate", "datasets", "bm25s", "PyStemmer")}}
    store = ResultStore(args.output, fingerprint)
    if len(store.rows) == len(selected):
        return store.finalize()
    configure_cuda()
    from scripts.run_qa32_rebuttal import atomic_json
    def state(stage):
        atomic_json(args.output / "status.json", {
            "status": stage, "pid": os.getpid(), "updated": time.time(),
            "completed_samples": len(store.rows), "samples": spec["samples"],
        })
    from retrievers.WikiBM25Retriever import WikiBM25Retriever
    from models.QwenLLMWithKVCache import QwenLLMWithKVCache
    from token_tracker import TokenTracker
    state("loading_retriever")
    retriever = WikiBM25Retriever(str(wiki), load_corpus=True)
    if len(retriever.titles) != 5233235 or retriever.corpus_texts is None or len(retriever.corpus_texts) != 5233235:
        raise RuntimeError("Full Wikipedia retrieval assets failed validation")
    state("loading_model")
    with single_gpu_loader():
        llm = QwenLLMWithKVCache(str(model), config, token_tracker=TokenTracker())
    if llm.kv_config != config:
        raise RuntimeError("Loaded component configuration changed")
    with patch_attribute(base, "MAX_STEPS", 7), patch_attribute(base, "MODEL_PATH", str(model)):
        for index, sample in selected[len(store.rows):]:
            if source_hashes() != sources:
                raise RuntimeError("Sources changed before the next episode")
            state("running")
            row = run_episode(base, sample, llm, retriever)
            row.update(id=sample["id"], index=index, question=sample["question"], gold_answer=sample["answer"])
            store.commit(row)
    if source_hashes() != sources:
        raise RuntimeError("Sources changed during execution")
    return store.finalize()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--execute", action="store_true")
    result.add_argument("--dataset", choices=DATASETS, required=True)
    result.add_argument("--seed", type=int, choices=(233, 3407), required=True)
    result.add_argument("--variant", choices=VARIANTS, required=True)
    result.add_argument("--purpose", choices=("smoke", "main"), required=True)
    result.add_argument("--model-ready", type=Path, required=True)
    result.add_argument("--data-ready", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.execute:
        print(canonical(prepare_plan(args)))
        return
    from scripts.run_qa32_rebuttal import atomic_json
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".component.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Component output is owned by another worker") from None
        try:
            run(args)
            validate_completed_run(args.output)
        except BaseException as error:
            atomic_json(args.output / "status.json", {
                "status": "blocked", "error_type": type(error).__name__, "reason": str(error),
                "traceback": traceback.format_exc(),
                "failure_policy": "Failed questions are never committed/scored; identical fingerprints only"})
            raise


if __name__ == "__main__":
    main()
