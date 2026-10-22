"""Standalone, opt-in accuracy worker; never chooses GPUs or launches a queue."""
from __future__ import annotations

import argparse
import ast
import contextlib
import importlib
import importlib.metadata
import io
import json
import os
import random
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_qa32_rebuttal import (
    DATASETS, ResultStore, atomic_json, digest, file_hash, output_lock, read_json,
    select_legacy_samples, summarize, _read_question_audit, _validate_result_rows,
)

MODEL = ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28")
NEW_FILES = (
    "rebuttal/adapters/intentkv_rule/query_memory.py",
    "rebuttal/adapters/intentkv_rule/adapter.py", "scripts/run_intentkv_rule.py",
)
DEPENDENCIES = (
    "scripts/latency_runtime.py", "scripts/run_latency_experiment.py",
    "scripts/latency_datasets.py", "scripts/run_qa32_rebuttal.py",
    "scripts/qa32_runtime.py", "scripts/legacy_efficiency_metrics.py",
    "scripts/run_rebuttal_efficiency.py", "rebuttal/adapters/manifest.json",
    "config/rebuttal_assets_20260928.json",
    "run_all_2wiki_experiments_v2.py", "run_all_musique_experiments_v2.py",
)


def source_hashes():
    expected = read_json(ROOT / "rebuttal/adapters/manifest.json")["methods"]["h2o_step"]["original_core_hashes"]
    for name, checksum in expected.items():
        if file_hash(ROOT / name) != checksum:
            raise RuntimeError(f"Frozen QA source checksum mismatch: {name}")
    return {**expected,
            **{name: file_hash(ROOT / name) for name in (*NEW_FILES, *DEPENDENCIES)}}


def condition_path(root, dataset, ratio, purpose="main"):
    return Path(root) / f"{purpose}_{dataset}_ratio{int(ratio * 100)}_seed233"


def model_placement(model):
    from scripts.latency_runtime import validate_loaded_model

    validate_loaded_model(model)
    return {
        "device_map": {"": str(next(model.parameters()).device)},
        "layer_devices": [str(next(layer.parameters()).device) for layer in model.model.layers],
        "placement_source": "validated_parameter_and_buffer_devices",
    }


def validate_completed_run(output):
    """Read-only progress/gating contract; never imports model weights or CUDA."""
    output = Path(output)
    manifest = read_json(output / "manifest.json")
    fingerprint = manifest["fingerprint"]
    checksum = digest(fingerprint)
    checkpoint = read_json(output / "checkpoint.json")
    final = read_json(output / "results.json")
    status = read_json(output / "status.json")
    rows = checkpoint["results"]
    if (manifest.get("fingerprint_sha256") != checksum
            or checkpoint.get("fingerprint_sha256") != checksum
            or checkpoint.get("rows_sha256") != digest(rows)
            or final.get("fingerprint_sha256") != checksum
            or final.get("results") != rows or final.get("summary") != summarize(rows)
            or status.get("status") != "complete"
            or status.get("completed_samples") != fingerprint["samples"]
            or len(rows) != fingerprint["samples"]
            or [row["id"] for row in rows] != fingerprint["sample_ids"]
            or len(set(fingerprint["sample_ids"])) != fingerprint["samples"]
            or manifest.get("actual_placement_sha256") != digest(manifest["actual_placement"])):
        raise RuntimeError(f"Invalid/incomplete IntentKV rule condition: {output}")
    selected = [(row["index"], {"id": row["id"], "question": row["question"],
                               "answer": row["gold_answer"]}) for row in rows]
    _validate_result_rows(rows, selected)
    for row in rows:
        audit = _read_question_audit(output, checksum, row)
        if audit["intentkv_rule"]["method"] != "intentkv_rule":
            raise RuntimeError("Missing IntentKV rule per-question audit")
    return fingerprint


def require_all_50(root, current=None):
    completed = {}
    for dataset in DATASETS:
        fingerprint = validate_completed_run(condition_path(root, dataset, .5))
        if any(fingerprint.get(key) != value for key, value in (
            ("runtime", "intentkv_rule_frozen_qa_v1"), ("dataset", dataset),
            ("method", "intentkv_rule"), ("ratio", .5), ("purpose", "main"),
            ("samples", 500), ("sample_start", 0), ("seed", 233), ("max_steps", 7),
            ("model", MODEL[0]), ("model_revision", MODEL[1]))):
            raise RuntimeError(f"Wrong 50% prerequisite protocol: {dataset}")
        if fingerprint["source_hashes"] != source_hashes():
            raise RuntimeError(f"50% prerequisite source changed: {dataset}")
        completed[dataset] = fingerprint
    if current is not None:
        previous = completed[current["dataset"]]
        for key in ("sample_ids", "sample_indices", "selected_content_sha256", "assets",
                    "generation_config", "source_hashes", "versions", "method_scope"):
            if previous[key] != current[key]:
                raise RuntimeError(f"20% condition does not match its completed 50% run: {key}")
    return completed


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--dataset", choices=list(DATASETS))
    value.add_argument("--ratio", type=float, choices=[.5, .2], default=.5)
    value.add_argument("--seed", type=int, choices=[233], default=233)
    value.add_argument("--purpose", choices=["main", "smoke"], default="main")
    value.add_argument("--model-ready", type=Path)
    value.add_argument("--data-ready", type=Path)
    value.add_argument("--output-root", type=Path)
    value.add_argument("--samples", type=int)
    value.add_argument("--sample-start", type=int)
    value.add_argument("--max-steps", type=int, default=7)
    value.add_argument("--validate-output", type=Path,
                       help="CPU-only completed-artifact validation; no worker arguments required")
    value.add_argument("--prepare-protocol", type=Path,
                       help="CPU-only write of exact source/asset/sample protocol inputs")
    value.add_argument("--gpu", type=int, choices=[0, 1],
                       help="Queue-managed worker only: verify its exact registered admission lease")
    value.add_argument("--queue-contract", type=Path,
                       help="Registered queue plan; must equal OUTPUT_ROOT/plan.json")
    value.add_argument("--execute", action="store_true")
    return value


def validate_protocol(args):
    expected = (500, 0) if args.purpose == "main" else (2, 500)
    if args.samples is None:
        args.samples = expected[0]
    if args.sample_start is None:
        args.sample_start = expected[1]
    if (args.samples, args.sample_start) != expected or args.max_steps != 7:
        raise ValueError(f"{args.purpose} requires samples={expected[0]}, sample-start={expected[1]}, max-steps=7")
    if args.seed != 233 or args.ratio not in (.5, .2) or args.purpose not in ("main", "smoke"):
        raise ValueError("Require seed233 and the fixed .5/.2 main/smoke protocol")


def plan_inputs(args):
    """CPU-only exact source/assets/sample pins; imports neither torch nor model code."""
    validate_protocol(args)
    hashes = source_hashes()
    model_ready, data_ready = read_json(args.model_ready), read_json(args.data_ready)
    if (model_ready.get("repo"), model_ready.get("revision")) != MODEL:
        raise RuntimeError("Require pinned Qwen2.5-7B-Instruct ready metadata")
    model = Path(model_ready["path"]).resolve()
    config = read_json(model / "config.json")
    expected_config = {"model_type": "qwen2", "num_hidden_layers": 28,
                       "num_attention_heads": 28, "num_key_value_heads": 4, "hidden_size": 3584}
    if any(config.get(k) != v for k, v in expected_config.items()) or config.get("quantization_config"):
        raise RuntimeError("Require unquantized ordinary Qwen2.5-7B")
    spec = read_json(ROOT / "config/rebuttal_assets_20260928.json")
    for name in ("hotpotqa", "longrag"):
        if any(data_ready[name][key] != spec["data"][name][key] for key in ("repo", "revision")):
            raise RuntimeError(f"Pinned dataset revision mismatch: {name}")
    payloads, metadata = {}, {}
    if args.dataset == "hotpotqa":
        files = sorted((Path(data_ready["hotpotqa"]["path"]) / "distractor").glob("validation-*.parquet"))
        if not files:
            raise RuntimeError("Missing HotpotQA validation parquet")
        import pyarrow.parquet as parquet
        rows = []
        for path in files:
            payloads[str(path.resolve())] = file_hash(path)
            rows.extend(parquet.read_table(path, columns=["id", "question", "answer"]).to_pylist())
        metadata = data_ready["hotpotqa"]
    else:
        directory = Path(args.data_ready).resolve().parent / "datasets" / args.dataset
        ready = read_json(directory / "ready.json")
        pinned = spec["additional_datasets"][args.dataset]
        for key in ("dataset", "repository", "revision", "split", "rows", "sha256",
                    "raw_sha256", "normalizer", "normalizer_sha256"):
            if ready.get(key) != pinned[key]:
                raise RuntimeError(f"Pinned normalized dataset mismatch: {args.dataset}/{key}")
        path = Path(ready["path"]).resolve()
        if path != (directory / "normalized_dev.json").resolve() or file_hash(path) != pinned["sha256"]:
            raise RuntimeError("Invalid normalized dataset path/checksum")
        normalizer = ROOT / (ready["normalizer"].split(".", 1)[0] + ".py")
        if file_hash(normalizer) != pinned["normalizer_sha256"]:
            raise RuntimeError("Frozen normalizer checksum mismatch")
        rows = read_json(path)
        payloads[str(path)] = pinned["sha256"]
        metadata = {"ready": ready, "ready_sha256": file_hash(directory / "ready.json"),
                    "normalizer_sha256": pinned["normalizer_sha256"]}
    # Execute exactly the frozen selector, without importing its model dependencies.
    selector_path = ROOT / "run_all_wiki_experiments_v2.py"
    tree = ast.parse(selector_path.read_text())
    function = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "select_samples")
    globals_ = {"random": random, "RANDOM_SEED": 233,
                "NUM_SAMPLES": args.sample_start + args.samples}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(selector_path), "exec"), globals_)
    selector = SimpleSelector(globals_["select_samples"])
    with contextlib.redirect_stdout(io.StringIO()):
        selected = select_legacy_samples(selector, rows, samples=args.samples,
                                         sample_start=args.sample_start, expected_rows=DATASETS[args.dataset])
    weight_index = read_json(model / "model.safetensors.index.json")
    weights = {}
    for name in sorted(set(weight_index["weight_map"].values())):
        if Path(name).name != name:
            raise RuntimeError("Weight shard path escapes model directory")
        stat = (model / name).stat()
        weights[name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    wiki = Path(data_ready["wiki_index"]).resolve()
    if read_json(wiki / "meta.json")["num_docs"] != 5233235:
        raise RuntimeError("Require full-Wikipedia index")
    wiki_inventory = {}
    for name in ("meta.json", "titles.json", "corpus_texts.jsonl", "params.index.json"):
        stat = (wiki / name).stat()
        wiki_inventory[name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    build = wiki / "rebuttal_build_manifest.json"
    if build.exists():
        wiki_inventory["build_manifest_sha256"] = file_hash(build)
    return {
        "runtime": "intentkv_rule_frozen_qa_v1", "method": "intentkv_rule",
        "model": MODEL[0], "model_revision": MODEL[1], "dataset": args.dataset,
        "purpose": args.purpose, "ratio": args.ratio, "seed": 233,
        "samples": args.samples, "sample_start": args.sample_start, "max_steps": 7,
        "source_hashes": hashes, "model_ready": model_ready, "data_ready": data_ready,
        "model_ready_sha256": file_hash(args.model_ready), "data_ready_sha256": file_hash(args.data_ready),
        "model_config_hashes": {name: file_hash(model / name) for name in
                                ("config.json", "generation_config.json", "tokenizer_config.json",
                                 "tokenizer.json", "model.safetensors.index.json")},
        "weight_shard_inventory": weights, "retrieval_index_inventory": wiki_inventory,
        "dataset_metadata": metadata, "dataset_payload_sha256": payloads,
        "generation_config": read_json(model / "generation_config.json"),
        "sample_ids": [sample["id"] for _, sample in selected],
        "sample_indices": [index for index, _ in selected],
        "selected_content_sha256": digest([(index, {key: sample[key] for key in ("id", "question", "answer")})
                                           for index, sample in selected]),
    }


class SimpleSelector:
    def __init__(self, function):
        self.select_samples = function

    def __setattr__(self, name, value):
        if name in ("RANDOM_SEED", "NUM_SAMPLES"):
            self.select_samples.__globals__[name] = value
        else:
            object.__setattr__(self, name, value)


def cpu_function(relative, name, namespace):
    """Compile a verified pure helper without importing its torch-facing module."""
    path = ROOT / relative
    tree = ast.parse(path.read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = dict(namespace)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def build_protocol(args):
    """Full inference fingerprint, prepared entirely on CPU without torch imports."""
    pins = plan_inputs(args)
    namespace = {"Path": Path, "ROOT": ROOT, "MODEL": MODEL, "read_json": read_json,
                 "file_hash": file_hash, "DATASETS": DATASETS,
                 "ASSET_SPEC": ROOT / "config/rebuttal_assets_20260928.json",
                 "ADDITIONAL_DATASETS": ("2wiki", "musique"), "runtime": SimpleNamespace(ROOT=ROOT)}
    _, _, _, assets = cpu_function(
        "scripts/run_latency_experiment.py", "load_assets", namespace)(args.model_ready, args.data_ready)
    if args.dataset != "hotpotqa":
        _, _, extra = cpu_function(
            "scripts/latency_datasets.py", "normalized_assets", namespace)(args.data_ready, args.dataset)
        assets = {**assets, "parquet_sha256": {}, **extra}
    adapter_tree = ast.parse((ROOT / NEW_FILES[1]).read_text())
    constants = {}
    for node in adapter_tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("METHOD", "OVERRIDES", "SCOPE"):
                if isinstance(node.value, ast.Dict):
                    constants[node.targets[0].id] = {
                        ast.literal_eval(key): constants[value.id] if isinstance(value, ast.Name)
                        else ast.literal_eval(value) for key, value in zip(node.value.keys, node.value.values)}
                else:
                    constants[node.targets[0].id] = ast.literal_eval(node.value)
    core_tree = ast.parse((ROOT / "run_all_wiki_experiments_v2.py").read_text())
    experiment = next(node for node in core_tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "run_react_kv_experiment")
    assignment = next(node for node in experiment.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == "kv_config" for target in node.targets))
    config = eval(compile(ast.Expression(assignment.value), "<frozen-kv-defaults>", "eval"),
                  {"pruning_mode": "tova"})
    config = {**config, **constants["OVERRIDES"], "cache_ratio": args.ratio}
    return {
        "runtime": "intentkv_rule_frozen_qa_v1", "dataset": args.dataset, "method": constants["METHOD"],
        "ratio": args.ratio, "purpose": args.purpose, "samples": args.samples, "sample_start": args.sample_start,
        "seed": 233, "max_steps": 7, "model": MODEL[0], "model_revision": MODEL[1],
        "sample_ids": pins["sample_ids"], "sample_indices": pins["sample_indices"],
        "selected_content_sha256": pins["selected_content_sha256"],
        "source_hashes": pins["source_hashes"], "assets": assets,
        "generation_config": pins["generation_config"],
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers")},
        "kv_config": config, "method_scope": constants["SCOPE"],
        "protocol_inputs": pins, "protocol_inputs_sha256": digest(pins),
        "budget": "unchanged cumulative logical non-prompt ratio; full prompt plus forced observation",
    }


def run(args):
    validate_protocol(args)
    if args.ratio == .2 and args.purpose == "main":
        require_all_50(args.output_root)
    fingerprint = build_protocol(args)
    pins = fingerprint["protocol_inputs"]
    admitted = getattr(args, "_admitted_protocol", None)
    if admitted is not None and pins != admitted:
        raise RuntimeError("Source/asset/sample protocol changed after queue admission")
    hashes = fingerprint["source_hashes"]
    import torch
    import transformers
    from datasets import load_dataset
    from rebuttal.adapters.intentkv_rule.adapter import IntentKVRuleLLM, METHOD, OVERRIDES, SCOPE
    from scripts.latency_datasets import normalized_assets
    from scripts.latency_runtime import single_gpu_loader
    from scripts.run_latency_experiment import load_assets
    from scripts.qa32_runtime import QAErrorWatch, allocator_snapshot, attach_monitor, compact_metrics, cache_layout
    from token_tracker import TokenTracker

    if torch.__version__.split("+")[0] != "2.10.0" or transformers.__version__ != "5.0.0":
        raise RuntimeError("Require existing torch 2.10.0 / transformers 5.0.0 environment")
    model, wiki, files, provenance = load_assets(args.model_ready, args.data_ready)
    if args.dataset == "hotpotqa":
        data = load_dataset("parquet", data_files={"validation": [str(p) for p in files]},
                            split="validation",
                            cache_dir=str(args.data_ready.resolve().parent / "datasets/arrow_cache"))
    else:
        _, data, extra = normalized_assets(args.data_ready, args.dataset)
        provenance = {**provenance, "parquet_sha256": {}, **extra}
    base = importlib.import_module("run_all_wiki_experiments_v2")
    base.MODEL_PATH, base.MAX_STEPS = str(model), 7
    count, start = (500, 0) if args.purpose == "main" else (2, 500)
    selected = select_legacy_samples(
        base, data, samples=count, sample_start=start, expected_rows=DATASETS[args.dataset])
    selected_content = digest([(index, {key: sample[key] for key in ("id", "question", "answer")})
                               for index, sample in selected])
    if ([sample["id"] for _, sample in selected] != pins["sample_ids"]
            or [index for index, _ in selected] != pins["sample_indices"]
            or selected_content != pins["selected_content_sha256"]):
        raise RuntimeError("CPU provenance selection disagrees with restored evaluator")
    from scripts.qa32_runtime import legacy_config
    config = {**legacy_config(base, "tova", args.ratio), **OVERRIDES, "cache_ratio": args.ratio}
    generation = read_json(model / "generation_config.json")
    if (fingerprint["assets"] != provenance or fingerprint["kv_config"] != config
            or fingerprint["generation_config"] != generation or fingerprint["method"] != METHOD
            or fingerprint["method_scope"] != SCOPE
            or fingerprint["versions"] != {"torch": torch.__version__.split("+")[0],
                                           "transformers": transformers.__version__}):
        raise RuntimeError("CPU-prepared protocol disagrees with runtime helper/defaults")
    if args.ratio == .2 and args.purpose == "main":
        require_all_50(args.output_root, fingerprint)
    output = condition_path(args.output_root, args.dataset, args.ratio, args.purpose)
    with output_lock(output):
        store = ResultStore(output, fingerprint, selected)

        def status(state, **extra):
            atomic_json(output / "status.json", {
                "status": state, "pid": os.getpid(), "updated": time.time(),
                "completed_samples": len(store.rows), "target_samples": count, **extra})

        if len(store.rows) == count:
            store.finalize()
            status("complete")
            validate_completed_run(output)
            return
        try:
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError("Require exactly one GPU selected externally; worker never schedules GPUs")
            props = torch.cuda.get_device_properties(0)
            store.record_hardware(
                {"0": {"name": props.name, "total_memory": int(props.total_memory),
                       "torch_build": torch.__version__}},
                torch.version.cuda)
            status("loading_retriever")
            from retrievers.WikiBM25Retriever import WikiBM25Retriever
            retriever = WikiBM25Retriever(str(wiki))
            if len(retriever.titles) != 5233235 or retriever.corpus_texts is None:
                raise RuntimeError("Require full-Wikipedia retriever")
            status("loading_model")
            with single_gpu_loader():
                llm = IntentKVRuleLLM(str(model), config, token_tracker=TokenTracker())
            store.set_placement(model_placement(llm.model))
            with attach_monitor(llm) as monitor:
                monitor._patch(llm, "_compress_completed_block",
                               monitor._method_wrapper("intentkv_rule.completed_block",
                                                       "pruning_calls", prune=True))
                for index, sample in selected[len(store.rows):]:
                    if source_hashes() != hashes:
                        raise RuntimeError("Source changed during IntentKV worker execution")
                    status("running", sample_id=sample["id"])
                    before = allocator_snapshot((0,), reset=True)
                    stdout, stderr = QAErrorWatch(sys.stdout), QAErrorWatch(sys.stderr)
                    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                        answer, trajectory, timings, debug = base._run_react_kv_episode(
                            sample["question"], llm, retriever, pruning_mode=METHOD, return_debug=True)
                    if stdout.errors or stderr.errors:
                        raise RuntimeError(f"Suppressed legacy error: {(stdout.errors + stderr.errors)[:4]}")
                    snapshot = monitor.snapshot()
                    snapshot["active_cache_layout"] = cache_layout(llm)
                    snapshot["qa32_allocator"] = {"before_episode": before,
                                                  "after_episode": allocator_snapshot((0,))}
                    metrics = compact_metrics(snapshot)
                    row = {"index": index, "id": sample["id"], "question": sample["question"],
                           "gold_answer": sample["answer"], "predicted_answer": answer,
                           "em": base.exact_match(answer, sample["answer"]),
                           "f1": base.f1_score(answer, sample["answer"]), "kv_metrics": metrics}
                    store.commit(row, {"diagnostics": snapshot, "trajectory": trajectory,
                                       "legacy_step_timings_bookkeeping_only": timings,
                                       "intentkv_rule": llm.legacy_audit(), "debug": debug})
                    status("running")
            store.finalize()
            status("complete")
            validate_completed_run(output)
        except BaseException:
            status("failed", traceback=traceback.format_exc())
            raise


def main(argv=None):
    command_parser = parser()
    args = command_parser.parse_args(argv)
    if args.validate_output is not None:
        if (args.execute or args.dataset or args.model_ready or args.data_ready or args.output_root
                or args.gpu is not None or args.samples is not None or args.sample_start is not None
                or args.prepare_protocol is not None):
            command_parser.error("--validate-output is a standalone CPU-only command")
        try:
            fingerprint = validate_completed_run(args.validate_output)
            print(json.dumps({"status": "validated_complete",
                              "completed_samples": fingerprint["samples"],
                              "sample_ids": fingerprint["sample_ids"],
                              "fingerprint_sha256": digest(fingerprint)}, indent=2))
            return 0
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
            print(json.dumps({"status": "invalid_or_incomplete", "error": str(error)}), file=sys.stderr)
            return 1
    for name in ("dataset", "model_ready", "data_ready", "output_root"):
        if getattr(args, name) is None:
            command_parser.error(f"--{name.replace('_', '-')} is required for a worker plan/run")
    validate_protocol(args)
    if args.prepare_protocol is not None:
        if args.execute:
            command_parser.error("--prepare-protocol cannot execute a worker")
        pins = plan_inputs(args)
        atomic_json(args.prepare_protocol, pins)
        print(json.dumps({"status": "prepared_cpu_protocol",
                          "protocol_inputs_sha256": digest(pins)}, indent=2))
        return 0
    if not args.execute:
        print(json.dumps({
            "status": "planned_not_executed", "method": "intentkv_rule",
            "dataset": args.dataset, "ratio": args.ratio, "seed": 233,
            "samples": 500 if args.purpose == "main" else 2,
            "output": str(condition_path(args.output_root, args.dataset, args.ratio, args.purpose)),
            "gate": "ALL three completed 500-question 50% main conditions before any 20% main",
            "requires": "externally selected single GPU and explicit --execute",
        }, indent=2))
        return 0
    if args.gpu is None:
        command_parser.error("--execute requires --gpu and a registered queue lease")
    if (args.queue_contract is not None
            and args.queue_contract.resolve() != (args.output_root / "plan.json").resolve()):
        command_parser.error("--queue-contract must equal OUTPUT_ROOT/plan.json")
    if args.ratio == .2:
        require_all_50(args.output_root)
    from scripts.intentkv_rule_qa_queue import verify_registered
    admitted = plan_inputs(args)
    verify_registered(args, protocol=admitted)
    args._admitted_protocol = admitted
    source_hashes()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
