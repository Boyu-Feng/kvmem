"""Isolated Qwen2.5-32B legacy QA comparison (no asset preparation or GPU queueing).

Main: --dataset hotpotqa --method stepkv --ratio .2 --assets ASSETS --output OUT
Smoke: the same command with --purpose smoke (defaults to two selected positions
501/502, never the main first 500). Explicit sample/start values must match.

The frozen HotpotQA select_samples, QA episode, exact-match/F1, and per-method
defaults are reused for all datasets. 2Wiki/MuSiQue use pinned normalized dev
rows, NOT their historical dataset-specific CLI configuration. FullKV accepts
both ratio labels but never prunes. Generation configuration is never modified.
Methods h2o/tova retain ORIGINAL generation-time pruning. Explicit h2o_step and
tova_step controls load the frozen timing-only adapters: the first block stays
original; later pruning follows completed generation and stop cropping. The
comparison is one FullKV plus .2/.5 for the other five methods: 11 conditions
per dataset, 33 main conditions. Scheduling remains external.

Outputs: manifest.json (fingerprint.sample_ids and actual placement), atomic
checkpoint.json, results.json, status.json, kv_audit.json (per-ID audit index),
and questions/<position>_<ID-hash>/kv_audit.json (full tensor diagnostics).
The checkpoint is authoritative. An audit is durable before its row is committed;
interrupted uncommitted audits are ignored and derived indexes can be rebuilt.
Corrupt/mismatched checkpoints fail closed, without silently dropping samples.
Queue controllers can call validate_completed_run(output): read-only validation
of complete status, committed IDs, scores, actual tensor metrics, and audit hashes.
It returns completed_samples, sample_ids, has_actual_eviction, and summary, or
raises RuntimeError. It never loads model weights or touches CUDA.

Weight provenance uses the pinned ready revision, hashed config/tokenizer/index,
and shard size/mtime inventory; it does not claim to rehash 64 GB of weights per
condition. Dataset payloads ARE hashed. Timers are diagnostic, not latency claims.
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

MODEL_REPO = "Qwen/Qwen2.5-32B-Instruct"
MODEL_REVISION = "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"
DATASETS = {"hotpotqa": 7405, "2wiki": 12576, "musique": 2417}
METHODS = ("fullkv", "stepkv", "h2o", "tova", "h2o_step", "tova_step")
ASSET_SPEC = ROOT / "config/rebuttal_assets_20260928.json"
SCHEMA = 1


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def json_value(value):
    """Match persisted JSON types, including numeric GPU keys becoming strings."""
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".pending")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def output_lock(output):
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".qa32.lock").open("a+", encoding="utf-8") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Output directory is already owned by another runner") from None
        yield


def validate_protocol(args):
    expected = (500, 0) if args.purpose == "main" else (2, 500)
    if args.samples is None:
        args.samples = expected[0]
    if args.sample_start is None:
        args.sample_start = expected[1]
    if (args.samples, args.sample_start) != expected:
        raise ValueError(f"{args.purpose} requires --samples {expected[0]} --sample-start {expected[1]}")
    if args.seed != 233 or args.max_steps != 7 or args.ratio not in (.2, .5):
        raise ValueError("Require seed 233, max-steps 7, and ratio .2 or .5")
    if args.memory_limit_gib != 44:
        raise ValueError("All conditions require the same 44 GiB/device allocator cap")
    args.memory_limit_gib = 44


def select_legacy_samples(base, data, *, samples, sample_start, seed=233, expected_rows=None):
    if expected_rows is not None and len(data) != expected_rows:
        raise RuntimeError(f"Unexpected dataset size: {len(data)} (expected {expected_rows})")
    ids = set()
    for row in data:
        sid = row.get("id")
        if not isinstance(sid, str) or not sid.strip() or sid in ids:
            raise RuntimeError(f"Missing, invalid, or duplicate dataset ID: {sid!r}")
        ids.add(sid)
        for key in ("question", "answer"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise RuntimeError(f"Invalid {key} for dataset ID {sid!r}")
    if samples <= 0 or sample_start < 0 or seed != 233:
        raise ValueError("Invalid selection protocol")
    base.RANDOM_SEED, base.NUM_SAMPLES = seed, sample_start + samples
    selected = base.select_samples(data)[sample_start:]
    if len(selected) != samples or len({row["id"] for _, row in selected}) != samples:
        raise RuntimeError("Original selection returned an incomplete/duplicate sample set")
    return selected


def load_assets(assets, dataset):
    spec = read_json(ASSET_SPEC)
    ready = {kind: read_json(assets / f"{kind}_ready.json") for kind in ("model", "data")}
    model_ready = ready["model"]
    if (model_ready.get("repo"), model_ready.get("revision")) != (MODEL_REPO, MODEL_REVISION):
        raise RuntimeError("Require the pinned Qwen2.5-32B-Instruct model revision")
    model = Path(model_ready["path"]).resolve()
    if not model.is_dir():
        raise RuntimeError(f"Missing local model directory: {model}")
    config = read_json(model / "config.json")
    expected = {"model_type": "qwen2", "hidden_size": 5120, "num_hidden_layers": 64,
                "num_attention_heads": 40, "num_key_value_heads": 8}
    if any(config.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Model configuration is not Qwen2.5-32B")
    if config.get("quantization_config"):
        raise RuntimeError("Quantized model assets are not the BF16 comparison")
    index = read_json(model / "model.safetensors.index.json")
    shards = {}
    for name in sorted(set(index["weight_map"].values())):
        path = model / name
        if Path(name).name != name or not name.endswith(".safetensors") or not path.is_file():
            raise RuntimeError(f"Invalid/missing model shard: {name!r}")
        stat = path.stat()
        if stat.st_size <= 0:
            raise RuntimeError(f"Empty model shard: {name}")
        shards[name] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if not shards or sum(row["bytes"] for row in shards.values()) < index["metadata"]["total_size"]:
        raise RuntimeError("Truncated/incomplete model shard inventory")
    model_files = sorted({*model.glob("*.json"), *model.glob("*.txt"), *model.glob("*.model")})
    model_hashes = {path.name: file_hash(path) for path in model_files}
    for name in ("generation_config.json", "tokenizer_config.json", "tokenizer.json"):
        if name not in model_hashes:
            raise RuntimeError(f"Missing required local model/tokenizer file: {name}")
    for name in ("hotpotqa", "longrag"):
        actual, pinned = ready["data"].get(name, {}), spec["data"][name]
        if any(actual.get(key) != pinned[key] for key in ("repo", "revision")):
            raise RuntimeError(f"Unpinned retrieval/evaluation data revision: {name}")
    wiki = Path(ready["data"]["wiki_index"]).resolve()
    if not (wiki / "titles.json").is_file() or not (wiki / "corpus_texts.jsonl").is_file():
        raise RuntimeError("Full-Wikipedia index is not ready")
    wiki_inventory = {path.name: {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                      for path in sorted(wiki.iterdir()) if path.is_file()}
    wiki_metadata_hashes = {name: file_hash(wiki / name) for name in
                           ("meta.json", "params.index.json", "rebuttal_build_manifest.json")
                           if (wiki / name).is_file()}
    build_hash = ready["data"].get("index_build_manifest_sha256")
    if build_hash is not None:
        if wiki_metadata_hashes.get("rebuttal_build_manifest.json") != build_hash:
            raise RuntimeError("Full-Wikipedia build manifest checksum mismatch")
        build = read_json(wiki / "rebuttal_build_manifest.json")
        if (build.get("corpus") != ready["data"]["longrag"]
                or build.get("expected_docs") != 5233235
                or build.get("builder_sha256") != file_hash(ROOT / "build_wiki_index.py")):
            raise RuntimeError("Full-Wikipedia builder/corpus provenance mismatch")
    hashes = {f"{kind}_ready.json": file_hash(assets / f"{kind}_ready.json") for kind in ready}
    if dataset == "hotpotqa":
        files = sorted((assets / "datasets/hotpotqa/distractor").glob("validation-*.parquet"))
        if not files:
            raise RuntimeError("Pinned HotpotQA validation parquet is missing")
        hashes.update({str(path.relative_to(assets)): file_hash(path) for path in files})
        from datasets import load_dataset
        data = load_dataset("parquet", data_files={"validation": [str(path) for path in files]},
                            split="validation", cache_dir=str(assets / "datasets/arrow_cache"))
        dataset_ready = ready["data"]["hotpotqa"]
    else:
        directory = assets / "datasets" / dataset
        dataset_ready = read_json(directory / "ready.json")
        pinned = spec["additional_datasets"][dataset]
        for key in ("dataset", "repository", "revision", "split", "rows", "sha256",
                    "raw_sha256", "normalizer", "normalizer_sha256"):
            if dataset_ready.get(key) != pinned[key]:
                raise RuntimeError(f"{dataset}: pinned data/normalizer metadata mismatch ({key})")
        path = Path(dataset_ready["path"]).resolve()
        if path != (directory / "normalized_dev.json").resolve():
            raise RuntimeError("Normalized dataset path is outside the selected assets dataset")
        payload_hash = file_hash(path)
        if payload_hash != pinned["sha256"]:
            raise RuntimeError(f"{dataset}: normalized dataset payload checksum mismatch")
        hashes[f"datasets/{dataset}/normalized_dev.json"] = payload_hash
        hashes[f"datasets/{dataset}/ready.json"] = file_hash(directory / "ready.json")
        data = read_json(path)
    provenance = {
        "ready": ready, "evaluation_dataset": dataset_ready, "payload_sha256": hashes,
        "pinned_data_spec_sha256": file_hash(ASSET_SPEC),
        "model_revision": MODEL_REVISION, "model_metadata_sha256": model_hashes,
        "model_shards": shards,
        "weight_integrity_scope": "pinned ready revision + config/tokenizer/index SHA256 + shard sizes/mtime; "
                                  "weight payload SHA256 is not recomputed",
        "wiki_index_inventory": wiki_inventory, "wiki_index_metadata_sha256": wiki_metadata_hashes,
        "wiki_integrity_scope": "pinned LongRAG revision + index inventory/metadata; full index not rehashed",
    }
    return model, wiki, data, provenance, read_json(model / "generation_config.json")


def source_fingerprint():
    from run_rebuttal_legacy import CORE_FILES, SNAPSHOT, verify_original_sources
    core = verify_original_sources(ROOT)
    extra = (
        "run_rebuttal_legacy.py", "run_rebuttal_multidataset.py",
        "scripts/run_qa32_rebuttal.py", "scripts/qa32_runtime.py",
        "scripts/legacy_efficiency_metrics.py", "scripts/run_rebuttal_efficiency.py",
        "kv_cache/anchor_utils.py", "kv_cache/position_remapper.py", "kv_cache/ours.py",
        "models/__init__.py", "kv_cache/__init__.py", "retrievers/__init__.py",
    )
    return {
        "original_core_hashes": core,
        "source_hashes": {name: file_hash(ROOT / name) for name in (*CORE_FILES, *extra)},
        "reference_archive_sha256": file_hash(SNAPSHOT),
    }


def summarize(rows):
    from scripts.qa32_runtime import QUANTITIES
    result = {
        "total_samples": len(rows),
        "exact_match": 100 * sum(row["em"] for row in rows) / len(rows) if rows else 0,
        "f1_score": 100 * sum(row["f1"] for row in rows) / len(rows) if rows else 0,
        "kv_measurement": "actual resident K/V tensor payload across all layers, not legacy counters",
        "cache_mean_definition": "equal question weighting; mean of generation-block boundaries within each question",
        "preprune_definition": "observed callback-entry peak; null means no prune callback (e.g. FullKV/first step)",
        "timing_scope": "diagnostic shared-GPU host time only; two-GPU sharding is not single-GPU latency",
    }
    for quantity in QUANTITIES:
        for metric in ("boundary_means", "peak_active", "peak_trajectory", "peak_preprune"):
            values = [row["kv_metrics"][metric][quantity] for row in rows
                      if row["kv_metrics"][metric] is not None]
            result[f"mean_question_{metric}_{quantity}"] = sum(values) / len(values) if values else None
            if metric != "boundary_means":
                result[f"max_question_{metric}_{quantity}"] = max(values) if values else None
    devices = sorted({device for row in rows
                      for device in row["kv_metrics"]["allocator"].get("after_episode", {})})
    result["max_question_cuda_peak_allocated_bytes_per_device"] = {
        device: max(row["kv_metrics"]["allocator"]["after_episode"][device]["peak_allocated_bytes"]
                    for row in rows) for device in devices}
    return result


def _validate_result_rows(rows, selected):
    if not isinstance(rows, list) or len(rows) > len(selected):
        raise RuntimeError("Invalid checkpoint row count/type")
    expected_ids = [sample["id"] for _, sample in selected[:len(rows)]]
    if [row.get("id") for row in rows] != expected_ids:
        raise RuntimeError("Checkpoint IDs must be a unique ordered prefix of selected IDs")
    for row, (index, sample) in zip(rows, selected):
        if any(row.get(key) != value for key, value in (
            ("index", index), ("question", sample["question"]), ("gold_answer", sample["answer"]))):
            raise RuntimeError("Checkpoint sample content differs from selected data")
        if (not isinstance(row.get("predicted_answer"), str) or row.get("em") not in (True, False)
                or not isinstance(row.get("f1"), (int, float))
                or not math.isfinite(row["f1"]) or not 0 <= row["f1"] <= 1
                or not isinstance(row.get("kv_metrics"), dict)):
            raise RuntimeError("Invalid checkpoint answer/score/metric fields")


def _read_question_audit(output, fingerprint_hash, row):
    reference = row["kv_audit"]
    path = (output / reference["path"]).resolve()
    if not path.is_relative_to(output.resolve() / "questions"):
        raise RuntimeError("Audit reference is outside this run's questions directory")
    if file_hash(path) != reference["sha256"]:
        raise RuntimeError(f"Missing/corrupt per-question KV audit for {row['id']}")
    audit = read_json(path)
    if (audit.get("schema_version") != SCHEMA or audit.get("id") != row["id"]
            or audit.get("fingerprint_sha256") != fingerprint_hash
            or audit.get("metrics") != row["kv_metrics"]):
        raise RuntimeError("Per-question audit does not match its committed result")
    return audit


def _validate_tensor_audit(audit, placement):
    from scripts.qa32_runtime import QUANTITIES, compact_metrics

    snapshot = audit["diagnostics"]
    if compact_metrics(snapshot) != audit["metrics"]:
        raise RuntimeError("Compact cache metrics disagree with the native tensor audit")
    layer_devices = placement["layer_devices"]
    layout = snapshot["active_cache_layout"]
    if not layout or len(layout) != len(layer_devices):
        raise RuntimeError("Actual KV layer count disagrees with model placement")
    per_token_bytes = 0
    for index, layer in enumerate(layout):
        if layer["layer"] != index:
            raise RuntimeError("Actual KV layer IDs are not ordered and unique")
        for kind in ("key", "value"):
            shape = layer[f"{kind}_shape"]
            if (len(shape) != 4 or any(type(size) is not int or size <= 0 for size in shape)
                    or layer[f"{kind}_device"] != layer_devices[index]
                    or layer[f"{kind}_dtype"] != "torch.bfloat16"
                    or layer[f"{kind}_bytes"] != math.prod(shape) * 2):
                raise RuntimeError("Invalid per-layer BF16 K/V tensor payload/device")
            if shape[-2] != snapshot["kv"]["current"]["tokens"]:
                raise RuntimeError("Final tensor shape disagrees with measured resident tokens")
            per_token_bytes += layer[f"{kind}_bytes"] // shape[-2]

    def sample_checked(sample):
        if sample is None:
            return
        if any(type(sample[key]) is not int or sample[key] < 0 for key in QUANTITIES):
            raise RuntimeError("Invalid real KV token/byte counts")
        tokens, protected = sample["tokens"], sample["protected_tokens"]
        expected_layers = [tokens] * len(layout)
        # A zero-trajectory peak can originate from the empty pre-prefill cache.
        if sample["layer_tokens"] != expected_layers and not (tokens == 0 and sample["layer_tokens"] == []):
            raise RuntimeError("Nonuniform/missing per-layer resident token counts")
        if (protected > tokens or sample["trajectory_tokens"] != tokens - protected
                or sample["total_bytes"] != per_token_bytes * tokens
                or sample["protected_bytes"] != per_token_bytes * protected
                or sample["trajectory_bytes"] != per_token_bytes * (tokens - protected)):
            raise RuntimeError("Real KV byte totals or protected/trajectory partition disagree")

    for name in ("current", "peak_active", "peak_trajectory", "peak_preprune", "peak_post_step", "peak_probe"):
        sample_checked(snapshot["kv"][name])
    for step in snapshot["steps"]:
        for name in ("boundary", "peak_active", "peak_preprune"):
            sample_checked(step[name])
    if len(snapshot["steps"]) != len(audit["legacy_step_timings_bookkeeping_only"]):
        raise RuntimeError("Audit generation blocks disagree with legacy episode boundaries")
    events = snapshot["pruning_events"]
    if snapshot["counts"]["pruning_calls"] != len(events):
        raise RuntimeError("Pruning callback count disagrees with recorded audit events")
    evicted = False
    for event in events:
        if not event["succeeded"] or event["before"] is None or event["after"] is None:
            raise RuntimeError("Failed/unmeasured pruning callback in completed run")
        sample_checked(event["before"])
        sample_checked(event["after"])
        removed = event["before"]["tokens"] - event["after"]["tokens"]
        if removed < 0 or event["actual_tokens_removed"] != removed:
            raise RuntimeError("Actual eviction count disagrees with before/after tensor lengths")
        evicted |= removed > 0
    for boundary in ("before_episode", "after_episode"):
        allocators = snapshot["qa32_allocator"][boundary]
        if set(allocators) != {"0", "1"}:
            raise RuntimeError("Missing per-device CUDA allocation measurements")
        for allocator in allocators.values():
            values = [allocator[key] for key in ("allocated_bytes", "reserved_bytes",
                                                 "peak_allocated_bytes", "peak_reserved_bytes")]
            if (any(type(value) is not int or value < 0 for value in values)
                    or values[0] == 0 or values[1] < values[0] or values[2] < values[0]
                    or values[3] < max(values[1], values[2])):
                raise RuntimeError("Invalid per-device CUDA allocation measurements")
    return evicted


def _validate_step_control_audit(audit, row, fingerprint):
    control = audit["legacy_adapter_audit"]
    protocol = fingerprint["method_protocol"]
    metrics = row["kv_metrics"]
    if (control["method"] != fingerprint["method"] or control["base_mode"] != protocol["base_mode"]
            or control["scope"] != protocol["scope"]
            or control["recent_window"] != fingerprint["kv_config"]["observation_window"]
            or control["resident_tokens"] != metrics["final"]["tokens"]
            or control["cache_bytes"] != metrics["final"]["total_bytes"]):
        raise RuntimeError("Frozen step-control scope or final resident cache disagrees with metrics")
    boundaries = control["step_boundaries"]
    if len(boundaries) != metrics["generation_blocks"]:
        raise RuntimeError("Frozen step-control boundary count disagrees with native monitor")
    for index, (boundary, measured) in enumerate(zip(boundaries, metrics["step_boundaries"])):
        phase = "initial_completed_block" if index == 0 else "incremental_completed_block"
        if (boundary["phase"] != phase or boundary["block"] != index + 1
                or boundary["resident_tokens"] != measured["tokens"]
                or boundary["cache_bytes"] != measured["total_bytes"]):
            raise RuntimeError("Frozen step-control boundary disagrees with measured K/V")
    native_evictions = [(event["before"]["tokens"], event["after"]["tokens"],
                         event["before"]["total_bytes"], event["after"]["total_bytes"])
                        for event in audit["diagnostics"]["pruning_events"]
                        if event["actual_tokens_removed"] > 0]
    adapter_evictions = [(event["before_tokens"], event["after_tokens"],
                          event["before_bytes"], event["after_bytes"])
                         for event in control["cache_measurements"] if event["pruned"]]
    if adapter_evictions != native_evictions:
        raise RuntimeError("Frozen step-control eviction audit disagrees with native tensor monitor")


def validate_completed_run(output: Path) -> dict:
    """Read-only controller contract; invalid/incomplete outputs raise RuntimeError.

    Eviction means a successful pruning callback shortened the measured resident
    K/V tensors, never an old bookkeeping counter or stop-string crop. False is
    valid for FullKV or smoke questions that finish before any pruning.
    """
    try:
        return _validate_completed_run(Path(output))
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, IndexError, AttributeError) as error:
        raise RuntimeError(f"Invalid/incomplete QA32 run at {output}: {error}") from error


def _validate_completed_run(output):
    manifest = read_json(output / "manifest.json")
    fingerprint = manifest["fingerprint"]
    fingerprint_hash = digest(fingerprint)
    if manifest.get("schema_version") != SCHEMA or manifest.get("fingerprint_sha256") != fingerprint_hash:
        raise RuntimeError("Manifest fingerprint checksum mismatch")
    ids = fingerprint["sample_ids"]
    if (not isinstance(ids, list) or not ids or any(not isinstance(sid, str) or not sid for sid in ids)
            or len(set(ids)) != len(ids) or len(ids) != fingerprint["samples"]):
        raise RuntimeError("Manifest sample IDs are incomplete or duplicated")
    placement = manifest["actual_placement"]
    if (manifest.get("actual_placement_sha256") != digest(placement)
            or set(placement["hf_device_map"].values()) != {0, 1}
            or set(placement["layer_devices"]) != {"cuda:0", "cuda:1"}):
        raise RuntimeError("Missing/corrupt actual two-device placement")
    status = read_json(output / "status.json")
    if status.get("status") != "complete" or status.get("samples") != len(ids):
        raise RuntimeError("Run status is not complete for the selected sample count")
    checkpoint = read_json(output / "checkpoint.json")
    final = read_json(output / "results.json")
    audit_index = read_json(output / "kv_audit.json")
    for value in (checkpoint, final, audit_index):
        if value.get("schema_version") != SCHEMA or value.get("fingerprint_sha256") != fingerprint_hash:
            raise RuntimeError("Result/checkpoint/audit-index fingerprint mismatch")
    rows = checkpoint["results"]
    if (not isinstance(rows, list) or checkpoint["rows_sha256"] != digest(rows)
            or [row["id"] for row in rows] != ids or final["results"] != rows):
        raise RuntimeError("Full results disagree with the committed ordered sample IDs")
    selected = [(row["index"], {"id": row["id"], "question": row["question"], "answer": row["gold_answer"]})
                for row in rows]
    if ([index for index, _ in selected] != fingerprint["sample_indices"]
            or digest(selected) != fingerprint["selected_content_sha256"]):
        raise RuntimeError("Committed sample contents differ from the manifest selection")
    _validate_result_rows(rows, selected)
    summary = summarize(rows)
    if (final["summary"] != summary or audit_index["summary"] != summary
            or status.get("exact_match") != summary["exact_match"]
            or status.get("f1_score") != summary["f1_score"]
            or audit_index["sample_ids"] != ids
            or audit_index["questions"] != {row["id"]: row["kv_audit"] for row in rows}):
        raise RuntimeError("Summary/status/audit index disagree with committed results")
    base = importlib.import_module("run_all_wiki_experiments_v2")
    has_actual_eviction = False
    for row in rows:
        if (row["em"] != base.exact_match(row["predicted_answer"], row["gold_answer"])
                or row["f1"] != base.f1_score(row["predicted_answer"], row["gold_answer"])):
            raise RuntimeError("Committed accuracy disagrees with the original QA scorer")
        audit = _read_question_audit(output, fingerprint_hash, row)
        has_actual_eviction |= _validate_tensor_audit(audit, placement)
        if fingerprint.get("method") in ("h2o_step", "tova_step"):
            _validate_step_control_audit(audit, row, fingerprint)
    if fingerprint.get("method") == "fullkv" and has_actual_eviction:
        raise RuntimeError("FullKV output contains actual pruning")
    return {"completed_samples": len(rows), "sample_ids": ids,
            "has_actual_eviction": has_actual_eviction, "summary": summary}


class ResultStore:
    """One atomic row commit references an already durable, hash-checked audit."""
    def __init__(self, output, fingerprint, selected):
        fingerprint = json_value(fingerprint)
        self.output, self.fingerprint, self.selected = Path(output), fingerprint, selected
        self.fingerprint_hash = digest(fingerprint)
        self.rows = []
        manifest = self.output / "manifest.json"
        if not manifest.exists():
            if any((self.output / name).exists() for name in
                   ("checkpoint.json", "results.json", "kv_audit.json", "questions")):
                raise RuntimeError("Existing QA output lacks a manifest; use a fresh directory")
            atomic_json(manifest, {"schema_version": SCHEMA, "created": time.time(),
                                   "fingerprint": fingerprint, "fingerprint_sha256": self.fingerprint_hash})
        self.manifest = read_json(manifest)
        if (self.manifest.get("schema_version") != SCHEMA
                or json_value(self.manifest.get("fingerprint")) != fingerprint
                or self.manifest.get("fingerprint_sha256") != self.fingerprint_hash):
            raise RuntimeError("Changed protocol/source/model/data/sample IDs; use a fresh output directory")
        if ("actual_placement" in self.manifest
                and self.manifest.get("actual_placement_sha256") != digest(self.manifest["actual_placement"])):
            raise RuntimeError("Recorded placement checksum mismatch")
        checkpoint = self.output / "checkpoint.json"
        if checkpoint.exists():
            state = read_json(checkpoint)
            if (state.get("schema_version") != SCHEMA
                    or state.get("fingerprint_sha256") != self.fingerprint_hash
                    or state.get("rows_sha256") != digest(state.get("results"))):
                raise RuntimeError("Checkpoint fingerprint or payload checksum mismatch")
            self.rows = state["results"]
            self._validate_rows(self.rows)
            if self.rows and "actual_placement" not in self.manifest:
                raise RuntimeError("Checkpoint has results but no verified model placement")
            for row in self.rows:
                self._read_audit(row)
        results = self.output / "results.json"
        if results.exists():
            final = read_json(results)
            if (len(self.rows) != len(selected) or final.get("results") != self.rows
                    or final.get("fingerprint_sha256") != self.fingerprint_hash
                    or final.get("summary") != summarize(self.rows)):
                raise RuntimeError("Final results disagree with the complete checkpoint")
        if self.rows:
            self.write_audit_index()

    def _validate_rows(self, rows):
        _validate_result_rows(rows, self.selected)

    def _read_audit(self, row):
        return _read_question_audit(self.output, self.fingerprint_hash, row)

    def set_placement(self, placement):
        placement = json_value(placement)
        if ("actual_placement" in self.manifest
                and json_value(self.manifest["actual_placement"]) != placement):
            raise RuntimeError("Loaded placement changed on resume")
        self.manifest["actual_placement"] = placement
        self.manifest["actual_placement_sha256"] = digest(placement)
        atomic_json(self.output / "manifest.json", self.manifest)

    def record_hardware(self, hardware, cuda_version):
        self.manifest.setdefault("hardware_attempts", []).append({
            "pid": os.getpid(), "at": time.time(), "devices": hardware,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_version": cuda_version,
        })
        atomic_json(self.output / "manifest.json", self.manifest)

    def commit(self, row, audit):
        position = len(self.rows)
        if position >= len(self.selected):
            raise RuntimeError("Cannot commit beyond the selected sample set")
        self._validate_rows(self.rows + [row])
        if "actual_placement" not in self.manifest:
            raise RuntimeError("Cannot commit results without verified model placement")
        name = f"{position:04d}_{hashlib.sha256(row['id'].encode()).hexdigest()[:16]}"
        relative = Path("questions") / name / "kv_audit.json"
        path = self.output / relative
        audit = {**audit, "schema_version": SCHEMA, "id": row["id"],
                 "fingerprint_sha256": self.fingerprint_hash, "metrics": row["kv_metrics"]}
        atomic_json(path, audit)
        row = json.loads(canonical({**row, "kv_audit": {"path": str(relative), "sha256": file_hash(path)}}))
        rows = self.rows + [row]
        self._validate_rows(rows)
        atomic_json(self.output / "checkpoint.json", {
            "schema_version": SCHEMA, "fingerprint_sha256": self.fingerprint_hash,
            "rows_sha256": digest(rows), "results": rows})
        self.rows = rows
        self.write_audit_index()

    def write_audit_index(self):
        atomic_json(self.output / "kv_audit.json", {
            "schema_version": SCHEMA, "fingerprint_sha256": self.fingerprint_hash,
            "sample_ids": [row["id"] for row in self.rows],
            "questions": {row["id"]: row["kv_audit"] for row in self.rows},
            "summary": summarize(self.rows)})

    def finalize(self):
        self._validate_rows(self.rows)
        if len(self.rows) != len(self.selected):
            raise RuntimeError("Cannot finalize an incomplete sample set")
        value = {"schema_version": SCHEMA, "fingerprint_sha256": self.fingerprint_hash,
                 "results": self.rows, "summary": summarize(self.rows)}
        atomic_json(self.output / "results.json", value)
        self.write_audit_index()
        return value


def run(args):
    from scripts.qa32_runtime import (
        ADAPTATIONS, PLACEMENT_POLICY, BalancedModelFactory, LayerDeviceAdapter,
        attach_monitor, configure_cuda, legacy_config, llm_class_for_method, method_protocol, observed_episode,
    )
    import torch
    import transformers

    validate_protocol(args)
    sources = source_fingerprint()
    if torch.__version__.split("+")[0] != "2.10.0" or transformers.__version__ != "5.0.0":
        raise RuntimeError("Require the isolated torch 2.10.0 / transformers 5.0.0 runtime")
    model, wiki, data, assets, generation = load_assets(args.assets, args.dataset)
    base = importlib.import_module("run_all_wiki_experiments_v2")
    model_module = importlib.import_module("models.QwenLLMWithKVCache")
    base.MODEL_PATH, base.MAX_STEPS = str(model), args.max_steps
    selected = select_legacy_samples(base, data, samples=args.samples, sample_start=args.sample_start,
                                     seed=args.seed, expected_rows=DATASETS[args.dataset])
    config = legacy_config(base, args.method, args.ratio)
    protocol = method_protocol(args.method)
    sources["source_hashes"].update(protocol["adapter_source_hashes"])
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "accelerate", "datasets", "bm25s", "PyStemmer")}
    fingerprint = {
        "runtime": "qa32_frozen_legacy_v1", "dataset": args.dataset, "method": args.method,
        "ratio": args.ratio, "purpose": args.purpose, "samples": args.samples,
        "sample_start": args.sample_start, "seed": args.seed, "max_steps": args.max_steps,
        "sample_ids": [sample["id"] for _, sample in selected],
        "sample_indices": [index for index, _ in selected],
        "selected_content_sha256": digest([(index, {key: sample[key] for key in ("id", "question", "answer")})
                                            for index, sample in selected]),
        "kv_config": config, "assets": assets, **sources, "versions": versions,
        "model": MODEL_REPO, "model_revision": MODEL_REVISION, "generation_config": generation,
        "placement_policy": PLACEMENT_POLICY, "memory_limit_gib": args.memory_limit_gib,
        "adaptations": ADAPTATIONS, "method_protocol": protocol,
        "dataset_config_policy": "current HotpotQA legacy defaults for all three datasets; "
                                 "normalized pinned 2Wiki/MuSiQue dev rows",
        "fullkv_ratio_policy": "ratio is a condition label only; FullKV never prunes",
        "provenance": "verified local frozen evaluator; not a claim of exact historical paper-run provenance",
    }
    store = ResultStore(args.output, fingerprint, selected)
    if len(store.rows) == len(selected):
        return store.finalize()
    hardware = configure_cuda(args.memory_limit_gib)
    store.record_hardware(hardware, torch.version.cuda)
    status = args.output / "status.json"
    atomic_json(status, {"status": "loading_retriever", "pid": os.getpid(), "at": time.time(),
                         "completed_samples": len(store.rows), "devices": hardware})
    from retrievers.WikiBM25Retriever import WikiBM25Retriever
    retriever = WikiBM25Retriever(str(wiki))
    if len(retriever.titles) != 5233235 or retriever.corpus_texts is None or len(retriever.corpus_texts) != 5233235:
        raise RuntimeError("Full-Wikipedia retrieval assets failed validation")
    atomic_json(status, {"status": "loading_model", "pid": os.getpid(), "at": time.time(),
                         "completed_samples": len(store.rows), "devices": hardware})
    from token_tracker import TokenTracker
    tracker = TokenTracker() if args.method != "fullkv" else None
    with BalancedModelFactory(model_module, model, on_loaded=store.set_placement):
        llm = llm_class_for_method(args.method)(str(model), config, token_tracker=tracker)
    if llm.kv_config != config:
        raise RuntimeError("Loaded legacy KV defaults changed")
    with LayerDeviceAdapter(llm) as adapter, attach_monitor(llm) as monitor:
        for index, sample in selected[len(store.rows):]:
            atomic_json(status, {"status": "running", "pid": os.getpid(), "at": time.time(),
                                 "id": sample["id"], "completed_samples": len(store.rows),
                                 "devices": hardware})
            start = time.perf_counter()
            row, audit = observed_episode(base, sample, llm, retriever, args.method, monitor, adapter)
            row.update(id=sample["id"], index=index, question=sample["question"],
                       gold_answer=sample["answer"], diagnostic_elapsed_seconds=time.perf_counter() - start)
            store.commit(row, audit)
    return store.finalize()


def parser():
    result = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    result.add_argument("--dataset", choices=DATASETS, required=True)
    result.add_argument("--method", choices=METHODS, required=True)
    result.add_argument("--ratio", type=float, choices=(.2, .5), default=.2)
    result.add_argument("--samples", type=int)
    result.add_argument("--sample-start", type=int)
    result.add_argument("--seed", type=int, default=233)
    result.add_argument("--max-steps", type=int, default=7)
    result.add_argument("--assets", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--purpose", choices=("smoke", "main"), default="main")
    result.add_argument("--memory-limit-gib", type=float, default=44)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    args.assets, args.output = args.assets.resolve(), args.output.resolve()
    # Local files only, including any accidental downstream HF lookup.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    with output_lock(args.output):
        try:
            atomic_json(args.output / "status.json", {
                "status": "validating", "pid": os.getpid(), "at": time.time()})
            result = run(args)
            atomic_json(args.output / "status.json", {
                "status": "complete", "pid": os.getpid(), "finished": time.time(),
                "samples": result["summary"]["total_samples"],
                "exact_match": result["summary"]["exact_match"], "f1_score": result["summary"]["f1_score"]})
        except BaseException as error:
            atomic_json(args.output / "status.json", {
                "status": "blocked", "pid": os.getpid(), "at": time.time(),
                "error_type": type(error).__name__, "reason": str(error),
                "traceback": traceback.format_exc(),
                "failure_policy": "no failed question is committed or converted into a wrong answer; "
                                  "fix the blocker and resume only under an identical fingerprint"})
            raise


if __name__ == "__main__":
    main()
