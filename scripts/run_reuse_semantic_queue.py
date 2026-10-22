#!/usr/bin/env python3
"""Plan/resume the approved 260-rollout HotpotQA citation-only accuracy pilot."""
from __future__ import annotations

import argparse
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts import reuse_semantic_resources as resources
from scripts import run_component_ablation_rebuttal as original
from scripts import run_reuse_semantic_pilot as pilot
from scripts.run_qa32_rebuttal import atomic_json

BASE = resources.BASE
OUTPUT = BASE / "results/reuse_semantic_hotpotqa50_r20_seed233_20261003_v11"
CONDITIONS = pilot.CONDITIONS
SMOKE_CONDITIONS = (
    "lexical_paraphrase", "semantic_paraphrase",
    "lexical_original", "no_citation", "semantic_original",
)
MAIN_CONDITIONS = SMOKE_CONDITIONS
DISPLAY_CONDITIONS = CONDITIONS
EXECUTION_ORDER_VERSION = "auxiliary_first_smoke_and_main_v2"
SMOKE_ORDER_POLICY = (
    "Pre-registered operational validation order, independent of F1; auxiliary "
    "conditions first. All five two-question actual-tensor-pruning smokes must "
    "pass before any main worker admission or registration."
)
MAIN_ORDER_POLICY = (
    "Pre-registered operational validation order, independent of F1; auxiliary "
    "main conditions first to expose unseen source boundaries early. All 250 "
    "main rollouts are fresh under one full provenance; no prior-run imports."
)
WORKER = ROOT / "scripts/run_reuse_semantic_pilot.py"
ENCODER_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
MAX_ATTEMPTS = 3
RETRY_STRATEGY = "numeric_diff_guided_v1"
TIMING = "All timing is diagnostic shared-GPU accuracy overhead, NOT latency."


def jobs():
    return [{"name": f"{purpose}/{condition}", "condition": condition, "purpose": purpose,
             "samples": 2 if purpose == "smoke" else 50,
             "variant": "no_cite" if condition == "no_citation" else "full",
             "dataset": "hotpotqa", "seed": 233, "ratio": .2,
             "sample_start": 500 if purpose == "smoke" else 0}
            for purpose in ("smoke", "main")
            for condition in (SMOKE_CONDITIONS if purpose == "smoke" else MAIN_CONDITIONS)]


def api_gate():
    if (tuple(CONDITIONS) != ("lexical_original", "no_citation", "semantic_original",
                             "lexical_paraphrase", "semantic_paraphrase")
            or any(not callable(getattr(pilot, name, None)) for name in
                   ("parser", "main", "condition_scope", "episode_tree",
                    "ast_proof", "validate_completed_run"))
            or not callable(getattr(original, "source_hashes", None))):
        raise RuntimeError("Pilot/frozen worker API drift")


def worker_sources():
    api_gate()
    with pilot.condition_scope("lexical_original"):
        return original.source_hashes()


def source_hashes():
    return {**worker_sources(), **{name: original.file_hash(ROOT / name) for name in (
        "scripts/run_deadline_gpu_load_guard.py", "scripts/run_parallel_qa32_queue.py",
        "scripts/run_qa32_rebuttal_queue.py", "scripts/reuse_semantic_assets.py",
        "scripts/reuse_semantic_resources.py", "scripts/run_reuse_semantic_queue.py",
        "scripts/run_rebuttal_efficiency_queue.py", "scripts/run_parallel_llama_ablation.py",
        "scripts/run_shared_llama_ablation.py", "scripts/run_component_ablation_queue.py",
    )}}


def runtime_versions():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "accelerate", "datasets", "bm25s", "PyStemmer")}


def runtime_protocol_pin(runtime=None, *, qwen_model_path=None):
    """CPU-only static contract; no encoder or auxiliary model construction."""
    if runtime is None:
        from scripts import reuse_semantic_runtime as runtime
    if not isinstance(getattr(runtime, "PROTOCOL", None), dict):
        raise RuntimeError("Runtime protocol fingerprint/default API mismatch")
    protocol = json.loads(original.canonical(runtime.PROTOCOL))
    if (type(protocol.get("max_paraphrase_attempts")) is not int
            or protocol["max_paraphrase_attempts"] != MAX_ATTEMPTS
            or protocol.get("retry_strategy") != RETRY_STRATEGY):
        raise RuntimeError("Runtime retry protocol must declare max_paraphrase_attempts=3, numeric_diff_guided_v1")
    provider = getattr(runtime, "LocalQwenParaphraser", None)
    if not callable(provider):
        raise RuntimeError("Runtime auxiliary provider/default API mismatch")
    parameter = inspect.signature(provider.__init__).parameters.get("max_attempts") if provider else None
    if parameter is not None and (type(parameter.default) is not int or parameter.default != MAX_ATTEMPTS):
        raise RuntimeError("LocalQwenParaphraser constructor max_attempts default must be 3")
    sha = original.digest(protocol)
    scorer = runtime.RuntimeCitationScorer
    if (any(not callable(getattr(scorer, name, None)) for name in
            ("fingerprint", "protocol_fingerprint", "source_fingerprint"))
            or scorer.protocol_fingerprint() != sha):
        raise RuntimeError("Runtime protocol fingerprint/default API mismatch")
    source_sha = original.file_hash(ROOT / "scripts/reuse_semantic_runtime.py")
    if scorer.source_fingerprint() != source_sha:
        raise RuntimeError("Loaded runtime source differs from the frozen file")
    model_path = Path(qwen_model_path).resolve() if qwen_model_path is not None else None
    generation = scorer("lexical_paraphrase", qwen_model_path=model_path,
                        device="cpu").fingerprint().get("paraphrase_generation", {})
    cache_binding = ["model_sha", "source_sha", "text_sha", "protocol_sha"]
    if (type(generation.get("max_attempts")) is not int
            or generation["max_attempts"] != MAX_ATTEMPTS
            or generation.get("retry_strategy") != RETRY_STRATEGY
            or generation.get("protocol_sha") != sha
            or generation.get("source_sha") != source_sha
            or generation.get("cache_key_fields") != cache_binding):
        raise RuntimeError("Runtime static default generation/cache fingerprint mismatch")
    if model_path is not None:
        constraint = generation.get("prose_decoding_constraint")
        if (not isinstance(constraint, dict)
                or constraint.get("policy_id") != protocol.get("prose_decoding_constraint", {}).get("id")
                or type(constraint.get("ban_count")) is not int or constraint["ban_count"] <= 0
                or not isinstance(constraint.get("ban_sha"), str) or len(constraint["ban_sha"]) != 64
                or not isinstance(constraint.get("tokenizer_fingerprint"), str)
                or len(constraint["tokenizer_fingerprint"]) != 64):
            raise RuntimeError("Pinned Qwen tokenizer must supply the actual prose suppression mask identity")
    return {"protocol": protocol, "protocol_sha256": sha, "source_sha256": source_sha,
            "max_attempts": MAX_ATTEMPTS, "retry_strategy": RETRY_STRATEGY,
            "qwen_model_path": str(model_path) if model_path is not None else None,
            "cache_binding": cache_binding,
            "default_generation_fingerprint": json.loads(original.canonical(generation))}


def resource_protocol():
    return {"gpu": resources.GPU, "uuid": resources.UUID, "min_free_gib": 64,
            "max_known_foreign_contexts": 2, "lock": str(resources.LOCK),
            "maintenance": str(resources.maintenance.maintenance_path(resources.GPU)),
            "maintenance_policy": resources.POLICY, "shared_accuracy": True}


def ready_pins(paths):
    missing = [str(path) for path in paths if not Path(path).is_file()]
    if missing:
        raise resources.Waiting("Missing ready artifacts: " + ", ".join(missing))
    return {str(Path(path).resolve()): original.file_hash(path) for path in paths}


def encoder_pin(path):
    ready = original.read_json(path)
    required = {"version": 1, "repo": "BAAI/bge-small-en-v1.5", "revision": ENCODER_REVISION,
                "pooling": "cls",
                "normalize": True, "max_tokens": 512, "dtype": "float32", "device": "cpu"}
    if (any(ready.get(k) != v for k, v in required.items())
            or not isinstance(ready.get("revision"), str)
            or len(ready["revision"]) != 40
            or any(c not in "0123456789abcdef" for c in ready["revision"])):
        raise RuntimeError("Expected immutable pinned BGE-small CLS/L2/512 CPU FP32")
    model = Path(ready["path"]).resolve()
    expected = BASE / "semantic_reuse_assets/bge-small-en-v1.5"
    if model != expected.resolve():
        raise RuntimeError("Encoder path differs from the private prepared asset namespace")
    for name, checksum in ready["config_hashes"].items():
        if Path(name).name != name or original.file_hash(model / name) != checksum:
            raise RuntimeError("Encoder configuration hash mismatch")
    if not ready["config_hashes"] or set(ready["weight_inventory"]) != {"model.safetensors"}:
        raise RuntimeError("Missing encoder config/weight inventory")
    for name, record in ready["weight_inventory"].items():
        if ((model / name).stat().st_size != record["bytes"]
                or original.file_hash(model / name) != record["sha256"]):
            raise RuntimeError("Encoder weight hash/inventory mismatch")
    return {"ready": ready, "sha256": original.digest(ready),
            "weight_integrity_scope": "fresh SHA256 of encoder safetensors and metadata"}


def make_plan(args):
    paths = (args.model_ready, args.data_ready, args.encoder_ready)
    pins = ready_pins(paths)
    encoder = encoder_pin(args.encoder_ready)
    with pilot.condition_scope("lexical_original"):
        model, wiki, rows, provenance, generation = original.load_assets(args)
        if provenance["model_repo"] != "Qwen/Qwen2.5-7B-Instruct":
            raise RuntimeError("Main and auxiliary model must be pinned local Qwen7B")
        import run_all_wiki_experiments_v2 as base
        selected = {purpose: original.select_samples(
            base, rows, seed=233, purpose=purpose,
            expected_rows=original.DATASETS["hotpotqa"]) for purpose in ("main", "smoke")}
    runtime_pin = runtime_protocol_pin(qwen_model_path=model)
    samples = {purpose: {"ids": [row["id"] for _, row in entries],
                         "indices": [index for index, _ in entries],
                         "content_sha256": original.digest(
                             [[index, {key: row[key] for key in ("id", "question", "answer")}]
                              for index, row in entries])}
               for purpose, entries in selected.items()}
    if len(samples["main"]["ids"]) != 50 or len(samples["smoke"]["ids"]) != 2:
        raise RuntimeError("Worker selection API drift")
    if set(samples["main"]["ids"]) & set(samples["smoke"]["ids"]):
        raise RuntimeError("Smoke/main overlap")
    payload = {
        "version": 1, "output": str(args.output.resolve()), "jobs": jobs(),
        "worker_api": "condition_scope + original.source_hashes/validate_rows + pilot.validate_completed_run",
        "sources": source_hashes(), "worker_sources": worker_sources(),
        "ready_pins": pins, "model_ready": str(args.model_ready.resolve()),
        "data_ready": str(args.data_ready.resolve()), "encoder_ready": str(args.encoder_ready.resolve()),
        "encoder_fingerprint": encoder, "assets_provenance": provenance,
        "runtime_versions": runtime_versions(), "python": sys.executable,
        "runtime_protocol": runtime_pin,
        "raw_hf_generation_config": generation, "samples": samples,
        "paraphrase_cache": str((args.output / "paraphrases.json").resolve()),
        "protocol": {"dataset": "hotpotqa", "seed": 233, "ratio": .2, "max_steps": 7,
                     "main_decoder_tokens": 256, "posthoc_decoder": "frozen unchanged",
                     "main_input": "raw original, never auxiliary rewritten",
                     "tau": .5, "tau_status": "fixed pilot, not validated optimal",
                     "main_rollouts": 250, "smoke_rollouts": 10,
                     "budget": "original non-fixed budget; no alternate/fallback budget",
                     "max_attempts": MAX_ATTEMPTS, "retry_strategy": RETRY_STRATEGY,
                     "retry_failure_policy": "all attempts audited; exhaustion fails closed; no fallback",
                     "auxiliary_cost_policy": "sum every attempt; all timing diagnostic, never latency",
                     "smoke_condition_order": list(SMOKE_CONDITIONS),
                     "main_condition_order": list(MAIN_CONDITIONS),
                     "display_condition_order": list(DISPLAY_CONDITIONS),
                     "execution_order_version": EXECUTION_ORDER_VERSION,
                     "smoke_order_policy": SMOKE_ORDER_POLICY,
                     "main_order_policy": MAIN_ORDER_POLICY,
                     "cpu_threads": 4, "memory_limit_gib": 44},
        "resource": resource_protocol(),
        "timing_policy": TIMING,
        "statistics": {"reference": "lexical_original", "bootstrap_unit": "question",
                       "paired": True, "resamples": 2000, "seed": 233, "confidence": .95},
    }
    return {**payload, "plan_sha256": original.digest(payload)}


def assert_plan(plan, approval_path=None):
    payload = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if original.digest(payload) != plan["plan_sha256"] or plan["jobs"] != jobs():
        raise RuntimeError("Queue plan/protocol hash mismatch")
    if (plan["output"] != str(OUTPUT.resolve())
            or plan["resource"] != resource_protocol()):
        raise RuntimeError("Unexpected pilot output/resource scope")
    if (plan["protocol"].get("smoke_condition_order") != list(SMOKE_CONDITIONS)
            or plan["protocol"].get("main_condition_order") != list(MAIN_CONDITIONS)
            or plan["protocol"].get("display_condition_order") != list(DISPLAY_CONDITIONS)
            or plan["protocol"].get("execution_order_version") != EXECUTION_ORDER_VERSION
            or plan["protocol"].get("smoke_order_policy") != SMOKE_ORDER_POLICY
            or plan["protocol"].get("main_order_policy") != MAIN_ORDER_POLICY):
        raise RuntimeError("Pre-registered operational smoke/main order drift")
    if source_hashes() != plan["sources"] or worker_sources() != plan["worker_sources"]:
        raise RuntimeError("Source/API drift; no resume or startup allowed")
    if runtime_versions() != plan["runtime_versions"] or sys.executable != plan["python"]:
        raise RuntimeError("Interpreter/runtime dependency drift")
    if ready_pins([Path(path) for path in plan["ready_pins"]]) != plan["ready_pins"]:
        raise RuntimeError("Ready file drift")
    if encoder_pin(Path(plan["encoder_ready"])) != plan["encoder_fingerprint"]:
        raise RuntimeError("Frozen encoder drift")
    args = argparse.Namespace(model_ready=Path(plan["model_ready"]),
                              data_ready=Path(plan["data_ready"]), dataset="hotpotqa")
    with pilot.condition_scope("lexical_original"):
        model, _, _, provenance, generation = original.load_assets(args)
    if provenance != plan["assets_provenance"] or generation != plan["raw_hf_generation_config"]:
        raise RuntimeError("Model/data/index/raw generation config drift")
    if (runtime_protocol_pin(qwen_model_path=model) != plan["runtime_protocol"]
            or plan["protocol"].get("max_attempts") != MAX_ATTEMPTS
            or plan["protocol"].get("retry_strategy") != RETRY_STRATEGY
            or plan["paraphrase_cache"] != str((OUTPUT / "paraphrases.json").resolve())):
        raise RuntimeError("Runtime retry protocol/cache binding drift")
    if approval_path is not None:
        if not Path(approval_path).is_file():
            raise resources.Waiting("Missing explicit parent launch approval")
        approval = resources.read(approval_path)
        if (approval.get("version") != 1 or approval.get("launch_approved") is not True
                or approval.get("plan_sha256") != plan["plan_sha256"]
                or approval.get("output") != plan["output"]
                or approval.get("gpu_uuid") != resources.UUID):
            raise RuntimeError("Parent approval does not bind this exact plan/output/GPU")


def command(plan, job):
    return [sys.executable, "-u", str(WORKER), "--condition", job["condition"],
            "--encoder-ready", plan["encoder_ready"], "--paraphrase-cache", plan["paraphrase_cache"],
            "--model-ready", plan["model_ready"], "--data-ready", plan["data_ready"],
            "--output", str(Path(plan["output"]) / job["name"]), "--dataset", "hotpotqa",
            "--seed", "233", "--variant", job["variant"], "--purpose", job["purpose"], "--execute"]


def worker_environment(output):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=resources.UUID, OMP_NUM_THREADS="4",
               MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", HF_HUB_OFFLINE="1",
               HF_DATASETS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    caches = output / "runtime_cache"
    for key in ("TMPDIR", "HF_HOME", "HF_HUB_CACHE", "TORCH_HOME", "XDG_CACHE_HOME",
                "TRITON_CACHE_DIR", "TORCH_EXTENSIONS_DIR", "TORCHINDUCTOR_CACHE_DIR"):
        path = caches / key.lower()
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    return env


def validate_runtime_fingerprint(row, plan, condition):
    if condition in ("lexical_original", "no_citation"):
        return
    fingerprint = row["reuse_scoring_audit"]["runtime_fingerprint"]
    generation = fingerprint.get("paraphrase_generation", {})
    if (fingerprint.get("protocol_sha") != plan["runtime_protocol"]["protocol_sha256"]
            or fingerprint.get("source_sha") != plan["runtime_protocol"]["source_sha256"]
            or generation.get("protocol_sha") != plan["runtime_protocol"]["protocol_sha256"]
            or generation.get("max_attempts") != MAX_ATTEMPTS
            or generation.get("retry_strategy") != RETRY_STRATEGY
            or generation.get("cache_key_fields") != plan["runtime_protocol"]["cache_binding"]
            or fingerprint.get("paraphrase_cache_path") != plan["paraphrase_cache"]):
        raise RuntimeError("Committed runtime fingerprint differs from approved retry/cache defaults")
    expected_generation = plan["runtime_protocol"]["default_generation_fingerprint"]
    actual_generation = dict(generation)
    if "device" in expected_generation:
        if actual_generation.get("device") != "cuda:0":
            raise RuntimeError("Committed auxiliary device differs from logical cuda:0")
        actual_generation["device"] = expected_generation["device"]
    if actual_generation != expected_generation:
        raise RuntimeError("Committed full generation/transport fingerprint differs from approved defaults")


def validate_job(plan, job, complete=False):
    directory = Path(plan["output"]) / job["name"]
    if not (directory / "manifest.json").exists():
        if any((directory / name).exists() for name in ("checkpoint.json", "results.json")):
            raise RuntimeError("Orphan checkpoint/results without manifest")
        return None
    manifest = original.read_json(directory / "manifest.json")
    fingerprint = manifest["fingerprint"]
    spec = fingerprint["protocol"]
    for key in ("dataset", "seed", "variant", "purpose", "samples", "ratio", "sample_start"):
        if spec[key] != job[key]:
            raise RuntimeError(f"Wrong pilot job protocol: {key}")
    if (spec["pilot_condition"] != job["condition"]
            or fingerprint["source_hashes"] != plan["worker_sources"]
            or fingerprint["sample_ids"] != plan["samples"][job["purpose"]]["ids"]
            or fingerprint["sample_indices"] != plan["samples"][job["purpose"]]["indices"]
            or fingerprint["selected_content_sha256"] != plan["samples"][job["purpose"]]["content_sha256"]
            or fingerprint["assets"] != plan["assets_provenance"]
            or fingerprint["generation_config"] != plan["raw_hf_generation_config"]
            or fingerprint["versions"] != plan["runtime_versions"]):
        raise RuntimeError("Checkpoint source/selection/asset pins mismatch")
    with pilot.condition_scope(job["condition"], Path(plan["encoder_ready"]),
                               Path(plan["paraphrase_cache"])):
        original.validate_fingerprint(fingerprint)
        if manifest["fingerprint_sha256"] != original.digest(fingerprint):
            raise RuntimeError("Manifest hash mismatch")
        if (directory / "checkpoint.json").exists():
            checkpoint = original.read_json(directory / "checkpoint.json")
            if (checkpoint["fingerprint_sha256"] != manifest["fingerprint_sha256"]
                    or checkpoint["rows_sha256"] != original.digest(checkpoint["rows"])):
                raise RuntimeError("Checkpoint prefix hash mismatch")
            original.validate_rows(checkpoint["rows"], fingerprint)
            for row in checkpoint["rows"]:
                validate_runtime_fingerprint(row, plan, job["condition"])
    if complete or (directory / "status.json").exists() and original.read_json(
            directory / "status.json").get("status") == "complete":
        result = pilot.validate_completed_run(directory)
        if job["purpose"] == "smoke" and not result["has_actual_eviction"]:
            raise RuntimeError("Smoke lacks actual selector/tensor pruning")
        return result
    return None


def paired_bootstrap(rows, reference, key, resamples=2000):
    if len(rows) != len(reference) or not rows:
        raise RuntimeError("Paired bootstrap requires identical nonempty ordered questions")
    if [r["id"] for r in rows] != [r["id"] for r in reference]:
        raise RuntimeError("Paired question identity mismatch")
    differences = [100 * (row[key] - ref[key]) for row, ref in zip(rows, reference)]
    rng = random.Random(233)
    distribution = sorted(sum(differences[rng.randrange(len(rows))] for _ in rows)
                          / len(rows) for _ in range(resamples))
    def percentile(p):
        index = (len(distribution) - 1) * p
        lo, hi = math.floor(index), math.ceil(index)
        return distribution[lo] + (distribution[hi] - distribution[lo]) * (index - lo)
    return {"difference_percentage_points": sum(differences) / len(rows),
            "paired_question_bootstrap_95ci": [percentile(.025), percentile(.975)],
            "resamples": resamples, "seed": 233}


def audit_summary(rows):
    totals = {"citation_rounds": 0, "compared_pairs": 0, "nonzero_increments": 0,
              "paraphrase_changed": 0, "paraphrase_unchanged": 0, "paraphrase_truncated": 0,
              "paraphrase_changed_vs_full_original": 0,
              "paraphrase_cache_hits": 0, "paraphrase_cache_misses": 0,
              "paraphrase_generation_attempts": 0, "paraphrase_retry_attempts": 0,
              "paraphrase_attempt_audits_missing": 0,
              "invalid_raw_outputs": 0, "encoder_calls": 0, "encoder_truncated": 0,
              "citation_diagnostic_seconds_including_auxiliary": 0,
              "encoder_diagnostic_seconds": 0, "paraphrase_diagnostic_seconds": 0}
    cosines, memory = [], []
    for row in rows:
        audit = row["reuse_scoring_audit"]
        runtime = audit["runtime"]
        rounds = audit["citation_rounds"]
        totals["citation_rounds"] += len(rounds)
        increments = [value for round_ in rounds for value in round_["increments"].values()]
        totals["nonzero_increments"] += sum(value > 0 for value in increments)
        totals["compared_pairs"] += runtime.get("pairs", len(increments))
        totals["citation_diagnostic_seconds_including_auxiliary"] += runtime.get(
            "citation_seconds_including_auxiliary", 0)
        for event in runtime.get("events", []):
            cosines.extend(pair["raw_cosine"] for pair in event.get("pairs", [])
                           if pair.get("raw_cosine") is not None)
        encoder = runtime.get("encoder") or {}
        paraphraser = runtime.get("paraphraser") or {}
        events = encoder.get("events", [])
        totals["encoder_calls"] += len(events)
        totals["encoder_truncated"] += sum(bool(event.get("truncated")) for event in events)
        totals["encoder_diagnostic_seconds"] += encoder.get("forward_seconds", 0)
        totals["paraphrase_diagnostic_seconds"] += paraphraser.get("forward_seconds", 0)
        totals["paraphrase_cache_hits"] += paraphraser.get("hits", 0)
        totals["paraphrase_cache_misses"] += paraphraser.get("misses", 0)
        for event in paraphraser.get("events", []):
            record = event.get("entry", event)
            if "unchanged" in record:
                totals["paraphrase_unchanged" if record["unchanged"] else "paraphrase_changed"] += 1
            if "paraphrase" in record and "original" in record:
                totals["paraphrase_changed_vs_full_original"] += record["paraphrase"] != record["original"]
            totals["paraphrase_truncated"] += bool(record.get("truncated"))
            if not event.get("cache_hit", False):
                attempts = record.get("attempts", event.get("attempts"))
                if isinstance(attempts, list):
                    totals["paraphrase_generation_attempts"] += len(attempts)
                    totals["paraphrase_retry_attempts"] += max(0, len(attempts) - 1)
                    totals["invalid_raw_outputs"] += sum(
                        bool(attempt.get("error") or attempt.get("rejection_reason")
                             or attempt.get("rejection_type")) for attempt in attempts)
                else:
                    totals["paraphrase_attempt_audits_missing"] += 1
                    totals["invalid_raw_outputs"] += bool(event.get("error"))
        memory.append({"id": row["id"], "encoder": encoder, "paraphraser": paraphraser,
                       "worker_resource_metrics": {
                           key: value for key, value in row.items()
                           if any(term in key for term in ("memory", "peak", "rss"))}})
    return {**totals, "raw_cosines": cosines,
            "cosine_count": len(cosines), "cosine_mean": sum(cosines) / len(cosines) if cosines else None,
            "memory_audits": memory,
            "memory_scope": "CPU encoder tensor/cache audits and CUDA auxiliary tensors/KV, "
                            "not whole-host RSS or CUDA allocator peaks unless explicitly recorded; "
                            "no cross-process/GPU attribution. Admission free VRAM is node-level.",
            "invalid_scope": "Successful committed rows only; failures are not scored, "
                             "attempt errors counted from entry/event attempts for cache misses; "
                             "otherwise outer errors with missing-attempt audits flagged. Cached "
                             "attempt histories are retained but not counted as new generation. "
                             "Failed raw outputs remain in logs/cache, never substituted.",
            "auxiliary_cost_scope": "Provider forward_seconds sums every actual attempt, "
                                    "not just accepted output; full provider/attempt audits retained. "
                                    "All shared-GPU timing remains diagnostic, never latency.",
            "timing_policy": TIMING}


def summarize(plan, completed):
    records, rows_by_condition = [], {}
    main_jobs = {job["condition"]: job for job in jobs() if job["purpose"] == "main"}
    for condition in DISPLAY_CONDITIONS:
        job = main_jobs[condition]
        if job["name"] not in completed:
            continue
        record = validate_job(plan, job, complete=True)
        rows = original.read_json(Path(plan["output"]) / job["name"] / "results.json")["rows"]
        rows_by_condition[job["condition"]] = rows
        records.append({"condition": job["condition"], **record, "audit": audit_summary(rows)})
    baseline = rows_by_condition.get("lexical_original")
    paired_status = "available" if baseline else "unavailable_baseline_incomplete"
    for record in records:
        record["paired_comparison_status"] = paired_status
        record["paired_vs_lexical_original"] = None
        if baseline:
            record["paired_vs_lexical_original"] = {
                key: paired_bootstrap(rows_by_condition[record["condition"]], baseline, key)
                for key in ("em", "f1")}
    result = {"plan_sha256": plan["plan_sha256"], "completed_main_conditions": len(records),
              "expected_main_conditions": 5, "results": records, "timing_policy": TIMING,
              "execution_order_version": EXECUTION_ORDER_VERSION,
              "main_execution_order": list(MAIN_CONDITIONS),
              "display_condition_order": list(DISPLAY_CONDITIONS),
              "paired_comparison_status": paired_status,
              "paired_baseline_completed_samples": len(baseline) if baseline else None,
              "accuracy_scope": "Original exact_match/F1 on successful raw main-agent answers; "
                                "no test-F1 tuning, no auxiliary equivalence guarantee",
              "resource_observations": {
                  j["name"]: original.read_json(Path(plan["output"]) / j["name"] / "resource_audit.json")
                  for j in jobs()
                  if (Path(plan["output"]) / j["name"] / "resource_audit.json").is_file()},
              "smoke_completed": [j["name"] for j in jobs()
                                  if j["purpose"] == "smoke" and j["name"] in completed]}
    atomic_json(Path(plan["output"]) / "SUMMARY.json", result)
    return result


def execute(plan, approval_path):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Controller must be launched CUDA_VISIBLE_DEVICES=''")
    assert_plan(plan, approval_path)
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=True)
    with resources.locked() as lock_fd:
        completed = set()
        for job in jobs():
            if validate_job(plan, job):
                completed.add(job["name"])
        summarize(plan, completed)
        pending = [job for job in jobs() if job["name"] not in completed]
        if not pending:
            return
        if pending[0]["purpose"] == "main" and any(
                f"smoke/{condition}" not in completed for condition in CONDITIONS):
            raise RuntimeError("All five independent two-question smokes must pass first")
        state_path = output / "resource_state.json"
        cpu_env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1")
        state = {"version": 1, "status": "preparing", "owner": resources.inspect_api.inspect(os.getpid()),
                 "child": None, "registered": False, "maintenance_owner": None,
                 "maintenance_claimed": False, "plan": str(output / "queue_plan.json"),
                 "approval": str(approval_path.resolve()), "output": str(output)}
        resources.write(state_path, state)
        (output / "logs").mkdir(parents=True, exist_ok=True)
        with (output / "logs/recovery.log").open("a") as recovery_log:
            recovery = subprocess.Popen([
                sys.executable, "-u", str(resources.SELF), "--mode", "recovery",
                "--state", str(state_path), "--lock-fd", str(lock_fd)],
                cwd=ROOT, env=cpu_env, pass_fds=(lock_fd,), start_new_session=True,
                stdout=recovery_log, stderr=subprocess.STDOUT)
            try:
                state["recovery"] = resources.inspect_api.inspect(recovery.pid)
                state["maintenance_owner"] = state["recovery"]["identity"]
                resources.write(state_path, state)
                foreign, admission = resources.claim_and_capture(state)
                state.update(foreign_contexts=foreign, admission=admission, maintenance_claimed=True)
                resources.write(state_path, state)
                for job in pending:
                    if job["purpose"] == "main" and any(
                            f"smoke/{condition}" not in completed for condition in CONDITIONS):
                        raise RuntimeError("All five independent two-question smokes must pass first")
                    assert_plan(plan, approval_path)
                    resources.verify_maintenance(state)
                    admission = resources.admit(foreign)
                    state.update(status="registering", child=None, registered=False,
                                 command=command(plan, job), admission=admission, job=job["name"])
                    resources.write(state_path, state)
                    log_path = output / "logs" / (job["name"].replace("/", "__") + ".log")
                    with log_path.open("a") as log:
                        child = subprocess.Popen([
                            sys.executable, "-u", str(resources.SELF), "--mode", "bootstrap",
                            "--state", str(state_path)], cwd=ROOT, env=cpu_env,
                            stdout=log, stderr=subprocess.STDOUT)
                        boot = resources.inspect_api.inspect(child.pid)
                        state["child"] = {"bootstrap": boot,
                                          "execution": {**boot, "argv": state["command"]}}
                        state["registered"] = True
                        resources.write(state_path, state)
                        observations = []
                        next_sample = 0
                        while child.poll() is None:
                            if recovery.poll() is not None:
                                raise RuntimeError("CPU own-only recovery died during worker")
                            disk = resources.read(state_path)
                            if disk.get("status") == "failed":
                                raise RuntimeError("CPU recovery rejected pins/maintenance identity")
                            if time.monotonic() >= next_sample:
                                observations.append(resources.own_memory(state))
                                next_sample = time.monotonic() + 20
                            time.sleep(.5)
                        if child.returncode != 0:
                            raise RuntimeError(f"Pilot worker failed ({child.returncode}); see {log_path}")
                    assert_plan(plan, approval_path)
                    validate_job(plan, job, complete=True)
                    atomic_json(output / job["name"] / "resource_audit.json", {
                        "plan_sha256": plan["plan_sha256"], "job": job["name"],
                        "startup": resources.read(state_path).get("startup_admission"),
                        "foreign_contexts": foreign, "observations": observations,
                        "scope": "Own process VmRSS/VmHWM from /proc and sampled driver CUDA "
                                 "used memory only, not allocator peak or whole-host attribution. "
                                 "44GiB torch allocator cap is enforced by the frozen worker.",
                        "timing_policy": TIMING})
                    completed.add(job["name"])
                    summarize(plan, completed)
                resources.write(state_path, {**state, "status": "finished"})
            except BaseException:
                resources.stop_owned(state)
                resources.write(state_path, {**state, "status": "failed"})
                raise
            finally:
                if recovery.poll() is not None:
                    resources.stop_owned(state)
                    if not resources.cleanup(state):
                        raise RuntimeError("Recovery died; live registered child still holds maintenance1")
                try:
                    recovery.wait(timeout=40)
                except subprocess.TimeoutExpired as error:
                    raise RuntimeError("CPU recovery retains resource gates pending safe child exit") from error


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--output", type=Path, default=OUTPUT)
    result.add_argument("--model-ready", type=Path, default=BASE / "assets/7b_ready.json")
    result.add_argument("--data-ready", type=Path, default=BASE / "qa32_assets/data_ready.json")
    result.add_argument("--encoder-ready", type=Path, default=BASE / "semantic_reuse_assets/encoder_ready.json")
    result.add_argument("--approval", type=Path)
    result.add_argument("--write-plan", action="store_true")
    result.add_argument("--execute", action="store_true")
    result.add_argument("--summarize", action="store_true")
    result.set_defaults(dataset="hotpotqa")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.execute and not args.approval:
            raise resources.Waiting("Explicit parent --approval is required; no automatic launch")
        if args.output.resolve() != OUTPUT.resolve():
            raise RuntimeError("Only the approved new pilot result namespace is supported")
        plan_path = args.output / "queue_plan.json"
        if args.execute or args.summarize:
            if not plan_path.is_file():
                raise resources.Waiting("Missing persistent queue plan; create/review it first")
            plan = original.read_json(plan_path)
            assert_plan(plan, args.approval if args.execute else None)
            if args.execute:
                execute(plan, args.approval)
            else:
                completed = {j["name"] for j in jobs() if validate_job(plan, j)}
                print(original.canonical(summarize(plan, completed)))
        else:
            plan = make_plan(args)
            if args.write_plan:
                if plan_path.exists() and original.read_json(plan_path) != plan:
                    raise RuntimeError("Existing persistent queue plan differs")
                atomic_json(plan_path, plan)
            print(original.canonical(plan))
    except resources.Waiting as error:
        print(original.canonical({"status": "waiting_not_launched", "reason": str(error)}))
    except (OSError, KeyError, ValueError, RuntimeError, TypeError) as error:
        print(original.canonical({"status": "failed_closed", "error": str(error)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
