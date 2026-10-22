"""CPU-only tests: no model, CUDA, real child launch, or process signals."""
import contextlib
import json
import os
from pathlib import Path
import shutil
import sys
import types

import pytest

from scripts import reuse_semantic_resources as resource
from scripts import run_reuse_semantic_queue as queue


@pytest.fixture
def scratch(request):
    directory = Path(__file__).resolve().parent / ".reuse_queue_test_files" / (
        request.node.name.replace("/", "_") + f"_{os.getpid()}")
    directory.mkdir(parents=True, exist_ok=False)
    try:
        yield directory
    finally:
        shutil.rmtree(directory)
        if not any(directory.parent.iterdir()):
            directory.parent.rmdir()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def info(pid=123, argv=None):
    return {"identity": {"pid": pid, "start_ticks": 77}, "uid": os.getuid(),
            "cwd": str(queue.ROOT), "argv": argv or ["/usr/bin/python3", "-u", str(resource.LLAMA)]}


@pytest.fixture
def plan(scratch, monkeypatch):
    output = scratch / "results/pilot"
    monkeypatch.setattr(queue, "BASE", scratch)
    monkeypatch.setattr(resource, "BASE", scratch)
    monkeypatch.setattr(queue, "OUTPUT", output)
    encoder = scratch / "semantic_reuse_assets/bge-small-en-v1.5"
    encoder.mkdir(parents=True)
    (encoder / "config.json").write_text("{}")
    (encoder / "model.safetensors").write_bytes(b"fake")
    ready = {"version": 1, "repo": "BAAI/bge-small-en-v1.5", "revision": queue.ENCODER_REVISION,
             "path": str(encoder), "pooling": "cls", "normalize": True, "max_tokens": 512,
             "dtype": "float32", "device": "cpu",
             "config_hashes": {"config.json": queue.original.file_hash(encoder / "config.json")},
             "weight_inventory": {"model.safetensors": {
                 "bytes": 4, "sha256": queue.original.file_hash(encoder / "model.safetensors")}}}
    args = types.SimpleNamespace(output=output, model_ready=scratch / "model_ready.json",
                                 data_ready=scratch / "data_ready.json",
                                 encoder_ready=scratch / "semantic_reuse_assets/encoder_ready.json",
                                 dataset="hotpotqa")
    save(args.model_ready, {})
    save(args.data_ready, {})
    save(args.encoder_ready, ready)
    monkeypatch.setattr(queue.pilot, "condition_scope", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(queue, "source_hashes", lambda: {"source": "sha"})
    monkeypatch.setattr(queue, "worker_sources", lambda: {"worker": "sha"})
    monkeypatch.setattr(queue, "runtime_versions", lambda: {"torch": "2.10.0"})
    protocol = {"max_paraphrase_attempts": 3, "retry_strategy": "numeric_diff_guided_v1"}
    runtime_pin = {"protocol": protocol, "protocol_sha256": queue.original.digest(protocol),
                   "source_sha256": "runtime-source", "max_attempts": 3,
                   "retry_strategy": "numeric_diff_guided_v1",
                   "cache_binding": ["model_sha", "source_sha", "text_sha", "protocol_sha"]}
    runtime_pin["default_generation_fingerprint"] = {
        "protocol_sha": runtime_pin["protocol_sha256"], "source_sha": runtime_pin["source_sha256"],
        "max_attempts": 3, "retry_strategy": "numeric_diff_guided_v1",
        "cache_key_fields": runtime_pin["cache_binding"], "device": "cpu",
        "format": "plain_utf8_eos_v1"}
    def pinned_runtime(**kwargs):
        result = json.loads(json.dumps(runtime_pin))
        model = kwargs.get("qwen_model_path")
        result["qwen_model_path"] = str(Path(model).resolve()) if model is not None else None
        return result
    monkeypatch.setattr(queue, "runtime_protocol_pin", pinned_runtime)
    provenance = {"model_repo": "Qwen/Qwen2.5-7B-Instruct", "fake": "pinned"}
    monkeypatch.setattr(queue.original, "load_assets",
                        lambda args: (scratch, scratch, [], provenance, {"raw": True}))
    monkeypatch.setitem(sys.modules, "run_all_wiki_experiments_v2", types.SimpleNamespace())
    def selected(base, rows, **kwargs):
        start, count = (0, 50) if kwargs["purpose"] == "main" else (500, 2)
        return [(i, {"id": str(i), "question": f"q{i}", "answer": f"a{i}", "context": "discard"})
                for i in range(start, start + count)]
    monkeypatch.setattr(queue.original, "select_samples", selected)
    result = queue.make_plan(args)
    result["_test_args"] = args
    return result


def clean(plan):
    result = dict(plan)
    result.pop("_test_args", None)
    return result


def approve(plan, path):
    value = {"version": 1, "launch_approved": True, "plan_sha256": plan["plan_sha256"],
             "output": plan["output"], "gpu_uuid": resource.UUID}
    save(path, value)
    return value


def test_exact_260_rollout_protocol():
    jobs = queue.jobs()
    assert len(jobs) == 10
    assert sum(j["samples"] for j in jobs if j["purpose"] == "main") == 250
    assert sum(j["samples"] for j in jobs if j["purpose"] == "smoke") == 10
    assert {j["dataset"] for j in jobs} == {"hotpotqa"}
    assert {j["ratio"] for j in jobs} == {.2}
    assert all(j["variant"] == ("no_cite" if j["condition"] == "no_citation" else "full")
               for j in jobs)
    assert all(j["sample_start"] == 500 for j in jobs[:5])
    assert queue.parser().parse_args([]).dataset == "hotpotqa"


def test_auxiliary_conditions_first_in_both_execution_orders():
    jobs = queue.jobs()
    assert [j["condition"] for j in jobs[:5]] == [
        "lexical_paraphrase", "semantic_paraphrase",
        "lexical_original", "no_citation", "semantic_original"]
    assert [j["condition"] for j in jobs[5:]] == list(queue.MAIN_CONDITIONS) == list(queue.SMOKE_CONDITIONS)
    assert list(queue.DISPLAY_CONDITIONS) == list(queue.CONDITIONS)
    assert all(j["purpose"] == "smoke" and j["samples"] == 2 for j in jobs[:5])
    assert all(j["purpose"] == "main" and j["samples"] == 50 for j in jobs[5:])


def test_smoke_order_is_pre_registered_operational_policy_not_f1_selection(plan):
    assert plan["protocol"]["smoke_condition_order"] == list(queue.SMOKE_CONDITIONS)
    assert plan["protocol"]["main_condition_order"] == list(queue.MAIN_CONDITIONS)
    assert plan["protocol"]["display_condition_order"] == list(queue.CONDITIONS)
    assert plan["protocol"]["execution_order_version"] == "auxiliary_first_smoke_and_main_v2"
    assert "independent of F1" in plan["protocol"]["smoke_order_policy"]
    assert "independent of F1" in plan["protocol"]["main_order_policy"]
    modified = json.loads(json.dumps(clean(plan)))
    modified["protocol"]["smoke_condition_order"].reverse()
    modified["plan_sha256"] = queue.original.digest(
        {key: value for key, value in modified.items() if key != "plan_sha256"})
    with pytest.raises(RuntimeError, match="operational smoke/main order"):
        queue.assert_plan(modified)


@pytest.mark.parametrize("field", [
    "main_condition_order", "display_condition_order", "execution_order_version"])
def test_main_and_display_order_are_bound_by_full_plan_provenance(plan, field):
    modified = json.loads(json.dumps(clean(plan)))
    value = modified["protocol"][field]
    modified["protocol"][field] = list(reversed(value)) if isinstance(value, list) else "old-order"
    modified["plan_sha256"] = queue.original.digest(
        {key: value for key, value in modified.items() if key != "plan_sha256"})
    with pytest.raises(RuntimeError, match="operational smoke/main order"):
        queue.assert_plan(modified)


@pytest.mark.parametrize("complete", [False, True])
def test_summary_display_is_canonical_independent_of_auxiliary_first_execution(
        plan, monkeypatch, complete):
    selected = list(queue.CONDITIONS) if complete else ["lexical_paraphrase", "semantic_paraphrase"]
    for condition in selected:
        save(Path(plan["output"]) / "main" / condition / "results.json",
             {"rows": [{"id": "q", "em": 1, "f1": 1}]})
    monkeypatch.setattr(queue, "validate_job",
                        lambda *a, **k: {"completed_samples": 50, "sample_ids": ["q"], "summary": {}})
    monkeypatch.setattr(queue, "audit_summary", lambda rows: {"full_attempt_audit_retained": True})
    monkeypatch.setattr(queue, "paired_bootstrap",
                        lambda *a: {"paired": True} if complete else pytest.fail("CI without baseline"))
    completed = {f"main/{condition}" for condition in selected}
    result = queue.summarize(plan, completed)
    expected = [condition for condition in queue.CONDITIONS if condition in selected]
    assert [record["condition"] for record in result["results"]] == expected
    assert result["display_condition_order"] == list(queue.CONDITIONS)
    assert result["main_execution_order"] == list(queue.MAIN_CONDITIONS)
    assert result["execution_order_version"] == queue.EXECUTION_ORDER_VERSION
    assert result["completed_main_conditions"] == len(selected)
    if not complete:
        assert all(record["paired_vs_lexical_original"] is None for record in result["results"])
        assert result["paired_comparison_status"] == "unavailable_baseline_incomplete"


def test_real_partial_summary_means_then_paired_ci_after_fifty_baseline_rows(scratch):
    """Synthetic CPU rows use real stores, validators, scorers, summary and bootstrap."""
    import run_all_wiki_experiments_v2 as base
    from scripts import reuse_semantic_runtime as runtime

    output = scratch / "summary_cpu_fixture"
    cache = output / "paraphrases.json"
    selected = [[i, {"id": f"unit-{i}", "question": f"unit question {i}",
                     "answer": "alpha beta"}] for i in range(50)]
    sources = queue.worker_sources()
    versions = queue.runtime_versions()
    selection = {"ids": [sample["id"] for _, sample in selected],
                 "indices": list(range(50)), "content_sha256": queue.original.digest(selected)}
    plan = {"output": str(output), "plan_sha256": "synthetic-cpu-summary-fixture",
            "worker_sources": sources, "samples": {"main": selection},
            "assets_provenance": {}, "raw_hf_generation_config": {}, "runtime_versions": versions,
            "runtime_protocol": queue.runtime_protocol_pin(), "paraphrase_cache": str(cache.resolve()),
            "encoder_ready": str(output / "unused_lexical_encoder_ready.json")}
    attempts = [
        {"accepted": False, "rejection_type": "NumericMismatch",
         "rejection_reason": "unit numeric mismatch", "raw_response": "unit rejected", "seconds": 1},
        {"accepted": True, "raw_response": "unit accepted", "seconds": 2},
    ]
    entry = {"original": "unit original", "paraphrase": "unit rewrite",
             "unchanged": False, "attempts": attempts}

    def create(condition, count, finalize):
        with queue.pilot.condition_scope(condition, None, cache):
            args = types.SimpleNamespace(dataset="hotpotqa", seed=233, variant="full", purpose="main")
            spec = queue.original.protocol(args)
            fp = {"protocol": spec, "source_hashes": sources,
                  "kv_config": queue.original.component_config(base, "full"),
                  "sample_ids": selection["ids"], "sample_indices": selection["indices"],
                  "selected_samples": selected, "selected_content_sha256": selection["content_sha256"],
                  "assets": {}, "generation_config": {}, "versions": versions}
            store = queue.original.ResultStore(output / "main" / condition, fp)
            runtime_fp = (runtime.RuntimeCitationScorer(
                condition, paraphrase_cache=cache, device="cuda:0").fingerprint()
                if "paraphrase" in condition else {"mode": condition, "citation_loop": "frozen original"})
            for i in range(len(store.rows), count):
                _, sample = selected[i]
                answer = ("alpha beta" if i < 25 else "alpha") if "paraphrase" in condition else (
                    "alpha beta" if i < 20 else "wrong")
                provider = ({"forward_seconds": 3, "hits": 1, "misses": 1,
                             "events": [{"cache_hit": False, "entry": entry},
                                        {"cache_hit": True, "entry": entry}]}
                            if "paraphrase" in condition and i == 0 else None)
                row = {"status": "success", "id": sample["id"], "index": i,
                       "question": sample["question"], "gold_answer": sample["answer"],
                       "predicted_answer": answer, "em": base.exact_match(answer, sample["answer"]),
                       "f1": base.f1_score(answer, sample["answer"]), "diagnostic_elapsed_seconds": 1,
                       "pruning_audit": [], "has_actual_eviction": False,
                       "reuse_scoring_audit": {
                           "pilot_condition": condition, "ast_proof": spec["ast_proof"],
                           "mainagent_source_text_policy": "unchanged", "citation_rounds": [],
                           "runtime_fingerprint": runtime_fp,
                           "paraphrase_retry_protocol": queue.pilot.PARAPHRASE_RETRY_PROTOCOL
                           if "paraphrase" in condition else None,
                           "runtime": {"paraphraser": provider}}}
                store.commit(row)
            if finalize:
                store.finalize()
            return store.rows

    para_rows = create("lexical_paraphrase", 50, True)
    complete = {"main/lexical_paraphrase"}
    before = queue.summarize(plan, complete)
    assert before["completed_main_conditions"] == 1
    assert before["paired_comparison_status"] == "unavailable_baseline_incomplete"
    assert before["paired_baseline_completed_samples"] is None
    para = before["results"][0]
    assert para["summary"]["exact_match"] == 50
    assert para["summary"]["f1_score"] == pytest.approx(100 * 5 / 6)
    assert para["paired_vs_lexical_original"] is None
    assert para["audit"]["paraphrase_generation_attempts"] == 2
    assert para["audit"]["paraphrase_retry_attempts"] == para["audit"]["invalid_raw_outputs"] == 1
    assert para["audit"]["paraphrase_diagnostic_seconds"] == 3

    create("lexical_original", 49, False)
    baseline_job = next(job for job in queue.jobs() if job["name"] == "main/lexical_original")
    assert queue.validate_job(plan, baseline_job) is None
    partial = queue.summarize(plan, complete)
    assert partial["paired_comparison_status"] == "unavailable_baseline_incomplete"
    assert partial["results"][0]["summary"] == para["summary"]
    assert partial["results"][0]["paired_vs_lexical_original"] is None

    baseline_rows = create("lexical_original", 50, True)
    after = queue.summarize(plan, complete | {"main/lexical_original"})
    assert after["paired_comparison_status"] == "available"
    assert after["paired_baseline_completed_samples"] == 50
    assert [record["condition"] for record in after["results"]] == [
        "lexical_original", "lexical_paraphrase"]
    assert after["results"][0]["summary"]["exact_match"] == 40
    assert after["results"][0]["summary"]["f1_score"] == 40
    scored_para = after["results"][1]
    assert scored_para["summary"] == para["summary"]
    for key in ("em", "f1"):
        expected = queue.paired_bootstrap(para_rows, baseline_rows, key)
        assert scored_para["paired_vs_lexical_original"][key] == expected
        assert expected["resamples"] == 2000
    assert scored_para["audit"] == para["audit"]


def test_only_fixed_v11_namespace_is_the_default_and_explicit_output():
    expected = resource.BASE / "results/reuse_semantic_hotpotqa50_r20_seed233_20261003_v11"
    assert queue.OUTPUT == expected
    assert queue.parser().parse_args([]).output == expected
    assert queue.parser().parse_args(["--write-plan", "--output", str(expected)]).output == expected


@pytest.mark.parametrize("namespace", [
    "reuse_semantic_hotpotqa50_r20_seed233_20261003_v1",
    "reuse_semantic_hotpotqa50_r20_seed233_20261003_v2",
    "reuse_semantic_hotpotqa50_r20_seed233_20261003_v3",
    "reuse_semantic_hotpotqa50_r20_seed233_20261003_v4",
    "reuse_semantic_hotpotqa50_r20_seed233_20261003_v5",
    "reuse_semantic_hotpotqa50_r20_seed233_20261003_v6",
    "reuse_semantic_hotpotqa50_r20_seed233_20261003_v7",
    "reuse_semantic_hotpotqa50_r20_seed233_20261003_v8",
    "reuse_semantic_hotpotqa50_r20_seed233_20261003_v9",
    "reuse_semantic_hotpotqa50_r20_seed233_20261003_v10", "arbitrary_output"])
def test_prior_attempts_and_arbitrary_roots_rejected_without_writes(monkeypatch, capsys, namespace):
    path = resource.BASE / "results" / namespace
    monkeypatch.setattr(queue, "make_plan", lambda args: pytest.fail("Rejected root was read/written"))
    monkeypatch.setattr(queue, "atomic_json", lambda *a: pytest.fail("Wrote rejected root"))
    assert queue.main(["--write-plan", "--output", str(path)]) == 1
    assert json.loads(capsys.readouterr().err)["status"] == "failed_closed"


@pytest.mark.parametrize("version", ["v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"])
def test_plan_validation_rejects_prior_attempt_even_with_a_valid_plan_hash(plan, version):
    rejected = clean(plan)
    rejected["output"] = str(resource.BASE / f"results/reuse_semantic_hotpotqa50_r20_seed233_20261003_{version}")
    rejected["plan_sha256"] = queue.original.digest(
        {key: value for key, value in rejected.items() if key != "plan_sha256"})
    with pytest.raises(RuntimeError, match="output/resource scope"):
        queue.assert_plan(rejected)


@pytest.mark.parametrize("mode", ["bootstrap", "recovery"])
@pytest.mark.parametrize("namespace", [
    "reuse_semantic_hotpotqa50_r20_seed233_20261003_v1",
    "reuse_semantic_hotpotqa50_r20_seed233_20261003_v2",
    "reuse_semantic_hotpotqa50_r20_seed233_20261003_v3",
    "reuse_semantic_hotpotqa50_r20_seed233_20261003_v4",
    "reuse_semantic_hotpotqa50_r20_seed233_20261003_v5",
    "reuse_semantic_hotpotqa50_r20_seed233_20261003_v6",
    "reuse_semantic_hotpotqa50_r20_seed233_20261003_v7",
    "reuse_semantic_hotpotqa50_r20_seed233_20261003_v8",
    "reuse_semantic_hotpotqa50_r20_seed233_20261003_v9",
    "reuse_semantic_hotpotqa50_r20_seed233_20261003_v10", "arbitrary_output"])
def test_resource_entrypoint_rejects_prior_attempts_and_arbitrary_state_roots(
        monkeypatch, capsys, mode, namespace):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(resource, "read", lambda *a: pytest.fail("Read rejected root"))
    monkeypatch.setattr(resource, "write", lambda *a: pytest.fail("Wrote rejected root"))
    path = resource.BASE / "results" / namespace / "resource_state.json"
    with pytest.raises(SystemExit) as error:
        resource.main(["--mode", mode, "--state", str(path)])
    assert error.value.code == 2
    assert "fixed v11" in capsys.readouterr().err


def test_resource_entrypoint_accepts_exact_queue_output_state(monkeypatch, scratch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(queue, "OUTPUT", scratch / "approved_v11")
    calls = []
    monkeypatch.setattr(resource, "bootstrap", lambda path: calls.append(path))
    state = queue.OUTPUT / "resource_state.json"
    resource.main(["--mode", "bootstrap", "--state", str(state)])
    assert calls == [state]


def test_plan_selection_content_matches_frozen_worker(plan):
    result = clean(plan)
    queue.assert_plan(result)
    assert result["samples"]["main"]["ids"] == list(map(str, range(50)))
    assert result["samples"]["smoke"]["ids"] == ["500", "501"]
    content = [[i, {"id": str(i), "question": f"q{i}", "answer": f"a{i}"}] for i in range(50)]
    assert result["samples"]["main"]["content_sha256"] == queue.original.digest(content)
    assert result["raw_hf_generation_config"] == {"raw": True}
    assert result["protocol"]["max_attempts"] == 3
    assert result["protocol"]["retry_strategy"] == "numeric_diff_guided_v1"
    assert result["runtime_protocol"]["protocol"]["max_paraphrase_attempts"] == 3
    assert result["runtime_protocol"]["qwen_model_path"] == str(plan["_test_args"].output.parents[1])


def test_production_tokenizer_mask_cannot_be_unbound(scratch):
    with pytest.raises(RuntimeError, match="actual prose suppression mask identity"):
        queue.runtime_protocol_pin(qwen_model_path=scratch / "missing_tokenizer")


def test_actual_tokenizer_mask_pin_stable_before_after_provider_load_and_reset(scratch, monkeypatch):
    from scripts import reuse_semantic_runtime as runtime
    from transformers import AutoTokenizer
    import torch

    ready_path = resource.BASE / "assets/7b_ready.json"
    if not ready_path.is_file():
        pytest.skip("Requires the existing pinned local Qwen tokenizer")
    ready = queue.original.read_json(ready_path)
    assert ready["repo"] == "Qwen/Qwen2.5-7B-Instruct"
    model_path = Path(ready["path"]).resolve()
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    pin = queue.runtime_protocol_pin(qwen_model_path=model_path)
    expected = pin["default_generation_fingerprint"]["prose_decoding_constraint"]
    assert expected["ban_count"] > 0
    actual = runtime.build_prose_digit_constraint(tokenizer)
    assert expected["ban_count"] == len(actual["suppress_tokens"])
    assert expected["ban_sha"] == actual["ban_sha"]
    assert expected["tokenizer_fingerprint"] == actual["tokenizer_fingerprint"]

    scorer = runtime.RuntimeCitationScorer(
        "lexical_paraphrase", qwen_model_path=model_path,
        paraphrase_cache=scratch / "cpu_mask_cache.json", device="cpu")
    before = scorer.fingerprint()
    assert before["paraphrase_generation"] == pin["default_generation_fingerprint"]
    real_provider = runtime.LocalQwenParaphraser
    providers = []
    def cpu_provider(model_path, cache_path, device):
        assert str(device) == "cpu"
        provider = real_provider(
            model_path, cache_path, device="cpu", tokenizer=tokenizer,
            model=torch.nn.Linear(1, 1), model_fingerprint="cpu-fixture-no-Qwen-weight-load")
        providers.append(provider)
        return provider
    monkeypatch.setattr(runtime, "LocalQwenParaphraser", cpu_provider)
    try:
        provider = scorer._paraphraser()
        assert provider.audit()["prose_decoding_constraint"] == expected
        del scorer._fingerprint_json
        assert scorer.fingerprint() == before
        scorer.reset_episode()
        del scorer._fingerprint_json
        assert scorer.fingerprint() == before
        assert scorer.source_fingerprint() == pin["source_sha256"]
        assert provider.audit()["auxiliary_generate_calls"] == 0
        row = {"reuse_scoring_audit": {"runtime_fingerprint": json.loads(json.dumps(before))}}
        row["reuse_scoring_audit"]["runtime_fingerprint"]["paraphrase_generation"]["device"] = "cuda:0"
        plan = {"runtime_protocol": pin, "paraphrase_cache": str((scratch / "cpu_mask_cache.json").resolve())}
        queue.validate_runtime_fingerprint(row, plan, "lexical_paraphrase")
        row["reuse_scoring_audit"]["runtime_fingerprint"]["paraphrase_generation"][
            "prose_decoding_constraint"]["ban_sha"] = "0" * 64
        with pytest.raises(RuntimeError, match="full generation/transport"):
            queue.validate_runtime_fingerprint(row, plan, "lexical_paraphrase")
    finally:
        for provider in providers:
            provider._forward_hook.remove()


def retry_runtime():
    protocol = {"max_paraphrase_attempts": 3, "retry_strategy": "numeric_diff_guided_v1"}
    class Scorer:
        def __init__(self, condition, *, device, qwen_model_path=None):
            assert condition == "lexical_paraphrase" and device == "cpu"
        @staticmethod
        def protocol_fingerprint():
            return queue.original.digest(protocol)
        @staticmethod
        def source_fingerprint():
            return queue.original.file_hash(queue.ROOT / "scripts/reuse_semantic_runtime.py")
        def fingerprint(self):
            return {"paraphrase_generation": {
                "max_attempts": protocol["max_paraphrase_attempts"],
                "retry_strategy": protocol["retry_strategy"],
                "protocol_sha": self.protocol_fingerprint(), "source_sha": self.source_fingerprint(),
                "cache_key_fields": ["model_sha", "source_sha", "text_sha", "protocol_sha"]}}
    class Provider:
        def __init__(self, *, max_attempts=3):
            pytest.fail("Static retry validation constructed an auxiliary provider")
    return types.SimpleNamespace(PROTOCOL=protocol, RuntimeCitationScorer=Scorer,
                                 LocalQwenParaphraser=Provider)


def test_static_runtime_retry_protocol_is_bound_without_provider_construction():
    runtime = retry_runtime()
    pin = queue.runtime_protocol_pin(runtime)
    assert pin["max_attempts"] == 3
    assert pin["retry_strategy"] == "numeric_diff_guided_v1"
    assert pin["protocol_sha256"] == queue.original.digest(runtime.PROTOCOL)
    assert pin["cache_binding"] == ["model_sha", "source_sha", "text_sha", "protocol_sha"]


def test_full_default_generation_transport_fingerprint_is_pinned():
    runtime = retry_runtime()
    original = runtime.RuntimeCitationScorer.fingerprint
    def full(self):
        result = original(self)
        result["paraphrase_generation"].update(
            format="plain_utf8_eos_v1", device="cpu",
            transport_policy={"payload": "literal UTF-8 scalar", "hf_eos_required": True})
        return result
    runtime.RuntimeCitationScorer.fingerprint = full
    pin = queue.runtime_protocol_pin(runtime)
    assert pin["default_generation_fingerprint"] == full(runtime.RuntimeCitationScorer(
        "lexical_paraphrase", device="cpu"))["paraphrase_generation"]
    assert pin["default_generation_fingerprint"]["format"] == "plain_utf8_eos_v1"


@pytest.mark.parametrize("field,value", [
    ("max_paraphrase_attempts", 1), ("max_paraphrase_attempts", 4), ("max_paraphrase_attempts", True),
    ("retry_strategy", "fallback_original"), ("retry_strategy", None)])
def test_unapproved_runtime_retry_defaults_fail_closed(field, value):
    runtime = retry_runtime()
    runtime.PROTOCOL[field] = value
    with pytest.raises(RuntimeError, match="Runtime retry protocol"):
        queue.runtime_protocol_pin(runtime)


@pytest.mark.parametrize("method", ["protocol_fingerprint", "source_fingerprint", "fingerprint"])
def test_runtime_static_fingerprint_api_and_source_cannot_drift(method):
    runtime = retry_runtime()
    setattr(runtime.RuntimeCitationScorer, method,
            None if method == "fingerprint" else staticmethod(lambda: "wrong"))
    with pytest.raises(RuntimeError, match="fingerprint/default API|Loaded runtime source"):
        queue.runtime_protocol_pin(runtime)


@pytest.mark.parametrize("default", [1, 4])
def test_runtime_constructor_default_must_match_declared_three_attempts(default):
    runtime = retry_runtime()
    class Provider:
        def __init__(self, *, max_attempts=default):
            pytest.fail("Constructed provider")
    runtime.LocalQwenParaphraser = Provider
    with pytest.raises(RuntimeError, match="constructor max_attempts default"):
        queue.runtime_protocol_pin(runtime)


def test_native_fixed_protocol_default_needs_no_constructor_override():
    runtime = retry_runtime()
    class Provider:
        def __init__(self, model_path, cache_path, *, timeout_seconds=180):
            pytest.fail("Constructed provider")
    runtime.LocalQwenParaphraser = Provider
    assert queue.runtime_protocol_pin(runtime)["max_attempts"] == 3


@pytest.mark.parametrize("field,value", [
    ("max_attempts", 1), ("retry_strategy", "unapproved"),
    ("cache_key_fields", ["model_sha", "text_sha"])])
def test_static_scorer_defaults_must_match_fixed_runtime_protocol(field, value):
    runtime = retry_runtime()
    original = runtime.RuntimeCitationScorer.fingerprint
    def changed(self):
        result = original(self)
        result["paraphrase_generation"][field] = value
        return result
    runtime.RuntimeCitationScorer.fingerprint = changed
    with pytest.raises(RuntimeError, match="static default generation/cache"):
        queue.runtime_protocol_pin(runtime)


def test_real_retry_runtime_static_contract_without_models_or_cuda(monkeypatch):
    from scripts import reuse_semantic_runtime as runtime
    class ForbiddenProvider:
        def __init__(self, *args, **kwargs):
            pytest.fail("CPU static validation constructed a model provider")
    monkeypatch.setattr(runtime, "LocalQwenParaphraser", ForbiddenProvider)
    monkeypatch.setattr(runtime, "EmbeddingEncoder", ForbiddenProvider)
    pin = queue.runtime_protocol_pin()
    assert pin["max_attempts"] == pin["protocol"]["max_paraphrase_attempts"] == 3
    assert pin["retry_strategy"] == "numeric_diff_guided_v1"
    assert pin["protocol_sha256"] == runtime.RuntimeCitationScorer.protocol_fingerprint()
    assert pin["default_generation_fingerprint"]["max_attempts"] == 3


def test_live_runtime_protocol_drift_blocks_a_hashed_plan(plan, monkeypatch):
    pin = json.loads(json.dumps(plan["runtime_protocol"]))
    pin["protocol_sha256"] = "changed"
    monkeypatch.setattr(queue, "runtime_protocol_pin", lambda **kwargs: pin)
    with pytest.raises(RuntimeError, match="retry protocol/cache"):
        queue.assert_plan(clean(plan))


def test_previous_attempt_cache_rejected_even_with_rehashed_plan(plan):
    changed = clean(plan)
    changed["paraphrase_cache"] = str(resource.BASE / (
        "results/reuse_semantic_hotpotqa50_r20_seed233_20261003_v2/paraphrases.json"))
    changed["plan_sha256"] = queue.original.digest(
        {key: value for key, value in changed.items() if key != "plan_sha256"})
    with pytest.raises(RuntimeError, match="retry protocol/cache"):
        queue.assert_plan(changed)


def scorer_row(plan):
    pin = plan["runtime_protocol"]
    generation = json.loads(json.dumps(pin["default_generation_fingerprint"]))
    generation["device"] = "cuda:0"
    return {"reuse_scoring_audit": {"runtime_fingerprint": {
        "protocol_sha": pin["protocol_sha256"], "source_sha": pin["source_sha256"],
        "paraphrase_generation": generation, "paraphrase_cache_path": plan["paraphrase_cache"]}}}


def test_committed_retry_defaults_and_cache_fingerprint_match_plan(plan):
    row = scorer_row(plan)
    for condition in ("semantic_original", "lexical_paraphrase", "semantic_paraphrase"):
        queue.validate_runtime_fingerprint(row, plan, condition)


@pytest.mark.parametrize("field,value", [
    ("format", "strict single paraphrase JSON"), ("device", "cuda:1"),
    ("new_unpinned_transport_policy", "lenient")])
def test_committed_transport_and_device_cannot_drift_from_full_default_pin(plan, field, value):
    row = scorer_row(plan)
    row["reuse_scoring_audit"]["runtime_fingerprint"]["paraphrase_generation"][field] = value
    with pytest.raises(RuntimeError, match="full generation/transport|auxiliary device"):
        queue.validate_runtime_fingerprint(row, plan, "lexical_paraphrase")


@pytest.mark.parametrize("field,value", [
    ("max_attempts", 1), ("retry_strategy", "unapproved"),
    ("protocol_sha", "v2-protocol"), ("cache_key_fields", ["model_sha", "text_sha"])])
def test_committed_unapproved_retry_or_cache_defaults_are_rejected(plan, field, value):
    row = scorer_row(plan)
    row["reuse_scoring_audit"]["runtime_fingerprint"]["paraphrase_generation"][field] = value
    with pytest.raises(RuntimeError, match="Committed runtime fingerprint"):
        queue.validate_runtime_fingerprint(row, plan, "lexical_paraphrase")


def test_ready_missing_is_wait_not_successful_run(scratch):
    with pytest.raises(resource.Waiting, match="Missing ready"):
        queue.ready_pins([scratch / "missing.json"])


@pytest.mark.parametrize("field,value", [
    ("device", "cuda"), ("pooling", "mean"), ("normalize", False),
    ("max_tokens", 256), ("dtype", "float16"), ("revision", "main"),
    ("revision", "a" * 40)])
def test_encoder_fail_closed(plan, field, value):
    path = plan["_test_args"].encoder_ready
    ready = queue.original.read_json(path)
    ready[field] = value
    save(path, ready)
    with pytest.raises(RuntimeError, match="immutable"):
        queue.encoder_pin(path)


def test_encoder_weight_drift(plan):
    path = plan["_test_args"].encoder_ready
    model = Path(queue.original.read_json(path)["path"])
    (model / "model.safetensors").write_bytes(b"evil")
    with pytest.raises(RuntimeError, match="weight"):
        queue.encoder_pin(path)


def test_source_drift_blocks_resume(plan, monkeypatch):
    monkeypatch.setattr(queue, "source_hashes", lambda: {"source": "changed"})
    with pytest.raises(RuntimeError, match="Source/API drift"):
        queue.assert_plan(clean(plan))


def test_generation_drift_blocks_resume(plan, monkeypatch):
    monkeypatch.setattr(queue.original, "load_assets",
                        lambda args: (None, None, [], plan["assets_provenance"], {"raw": False}))
    with pytest.raises(RuntimeError, match="generation config drift"):
        queue.assert_plan(clean(plan))


def test_parent_approval_required_and_bound(plan, scratch):
    path = scratch / "approval.json"
    with pytest.raises(resource.Waiting, match="parent launch"):
        queue.assert_plan(clean(plan), path)
    approval = approve(plan, path)
    queue.assert_plan(clean(plan), path)
    approval["plan_sha256"] = "other"
    save(path, approval)
    with pytest.raises(RuntimeError, match="approval"):
        queue.assert_plan(clean(plan), path)


def test_unapproved_execute_never_launches(monkeypatch, capsys):
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *a, **k: pytest.fail("launched"))
    assert queue.main(["--execute"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "waiting_not_launched"


def test_cpu_controller_required(plan, scratch, monkeypatch):
    approval = scratch / "approval.json"
    approve(plan, approval)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    with pytest.raises(RuntimeError, match="Controller"):
        queue.execute(clean(plan), approval)


def test_cli_pins_no_citation_and_auxiliary_paths(plan):
    cmd = queue.command(plan, next(j for j in queue.jobs() if j["name"] == "smoke/no_citation"))
    assert resource.flag(cmd, "--variant") == "no_cite"
    assert resource.flag(cmd, "--condition") == "no_citation"
    assert resource.flag(cmd, "--encoder-ready") == plan["encoder_ready"]
    assert resource.flag(cmd, "--paraphrase-cache") == plan["paraphrase_cache"]
    assert "--execute" in cmd


def test_all_commands_match_frozen_public_worker_parser(plan):
    for job in queue.jobs():
        cmd = queue.command(plan, job)
        args = queue.pilot.parser().parse_args(cmd[3:])
        queue.pilot.validate_args(args)
        assert args.condition == job["condition"]
        assert args.variant == job["variant"]
        assert args.dataset == "hotpotqa" and args.seed == 233


def test_private_offline_inputs_do_not_claim_condition_completion(plan):
    output = Path(plan["output"])
    save(output / "offline_probe_inputs.json", {"private_offline_probe": ["approved"]})
    save(output / "offline_probe_scores.json", {"pairs": 662, "scope": "offline only"})
    (output / "offline_probe.log").write_text("approved CPU-only offline probe\n")
    rebuilt = queue.make_plan(plan["_test_args"])
    assert rebuilt == clean(plan)
    assert all(queue.validate_job(plan, job) is None for job in queue.jobs())
    summary = queue.summarize(plan, set())
    assert summary["completed_main_conditions"] == 0
    assert summary["smoke_completed"] == []
    assert (output / "offline_probe_inputs.json").is_file()
    assert (output / "offline_probe_scores.json").is_file()
    assert (output / "offline_probe.log").is_file()


def test_write_plan_is_persistent_without_worker_launch(plan, monkeypatch, capsys):
    output = Path(plan["output"])
    save(output / "offline_probe_inputs.json", {"private": True})
    monkeypatch.setattr(queue, "make_plan", lambda args: clean(plan))
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *a, **k: pytest.fail("launched"))
    assert queue.main(["--write-plan"]) == 0
    assert queue.original.read_json(output / "queue_plan.json") == clean(plan)
    assert json.loads(capsys.readouterr().out)["plan_sha256"] == plan["plan_sha256"]
    assert not (output / "resource_state.json").exists()


def devices(pids=(), free=64 * 1024, uuid=resource.UUID):
    return {1: {"uuid": uuid, "name": "NVIDIA H200", "free_mib": free, "pids": list(pids)}}


def test_shared_admission_two_known_only(monkeypatch):
    monkeypatch.setattr(resource, "exact", lambda r: True)
    monkeypatch.setattr(resource, "known_llama", lambda *a: True)
    approval = [{"info": info(pid), "source_sha256": "frozen"} for pid in (1, 2)]
    admitted = resource.admit(approval, devices((1, 2)), blockers=[])
    assert admitted["foreign_contexts"] == 2
    assert admitted["allocator_limit_gib"] == 44
    with pytest.raises(resource.Waiting, match="Unknown"):
        resource.admit(approval, devices((1, 9)), blockers=[])


@pytest.mark.parametrize("gpu", [2, 4, 7])
def test_forbidden_gpu_never_admitted(gpu):
    with pytest.raises(RuntimeError, match="GPU1"):
        resource.admit([], {gpu: devices()[1]}, blockers=[])


def test_low_memory_and_uuid_mismatch():
    with pytest.raises(resource.Waiting, match="64GiB"):
        resource.admit([], devices(free=64 * 1024 - 1), blockers=[])
    with pytest.raises(RuntimeError, match="UUID"):
        resource.admit([], devices(uuid="other"), blockers=[])


def test_third_llama_context_and_unknown_assignment():
    with pytest.raises(RuntimeError, match="two"):
        resource.admit([None] * 3, devices(), blockers=[])
    with pytest.raises(resource.Waiting, match="unknown"):
        resource.admit([], devices(), blockers=[{"reason": "unknown GPU1 assignment"}])


def test_known_llama_checks_source_uid_cwd_and_selector(monkeypatch, scratch):
    record = info(argv=["/usr/bin/python3", "-u", str(resource.LLAMA),
                       "--output", str(scratch / "job")])
    monkeypatch.setattr(resource, "exact", lambda r: True)
    monkeypatch.setattr(resource.inspect_api, "digest", lambda p: "frozen")
    monkeypatch.setattr(resource.inspect_api, "gpu_selector", lambda pid: "7")
    assert resource.known_llama(record, "frozen") is False
    assert resource.known_llama(record, "changed") is False
    record["cwd"] = str(scratch)
    assert resource.known_llama(record, "frozen") is False


def test_pending_gpu1_but_qa_schedulers_do_not_block(monkeypatch, scratch):
    directory = scratch / "123"
    directory.mkdir()
    record = info(argv=["/usr/bin/python3", "-u",
                        str(queue.ROOT / "scripts/run_parallel_qa32_queue.py"), "--pairs", "3,4"])
    (directory / "cmdline").write_bytes(b"\0".join(os.fsencode(v) for v in record["argv"]) + b"\0")
    monkeypatch.setattr(resource.inspect_api, "identity", lambda pid: record["identity"])
    monkeypatch.setattr(resource.inspect_api, "inspect", lambda pid: record)
    monkeypatch.setattr(resource.inspect_api, "gpu_selector", lambda pid: "")
    assert resource.assignment_blockers(set(), scratch) == []
    monkeypatch.setattr(resource.inspect_api, "gpu_selector", lambda pid: resource.UUID)
    assert resource.assignment_blockers(set(), scratch)[0]["reason"].startswith("unknown")


def scan_fixture(monkeypatch, scratch, rows):
    for pid, row in rows.items():
        directory = scratch / str(pid)
        directory.mkdir()
        (directory / "cmdline").write_bytes(
            b"\0".join(os.fsencode(v) for v in row["argv"]) + b"\0")
    def identity(pid):
        row = rows[int(pid)]
        return None if row.get("dead") else row.get("info", info(int(pid), row["argv"]))["identity"]
    def selector(pid):
        row = rows[int(pid)]
        if row.get("unreadable_selector"):
            raise PermissionError("protected environ")
        return row.get("selector")
    def inspect(pid):
        row = rows[int(pid)]
        if row.get("inspect_error"):
            if row.get("dies_on_inspect"):
                row["dead"] = True
            raise row["inspect_error"]
        if row.get("no_inspect"):
            pytest.fail("Unrelated process cwd was inspected")
        return row.get("info", info(int(pid), row["argv"]))
    monkeypatch.setattr(resource.inspect_api, "identity", identity)
    monkeypatch.setattr(resource.inspect_api, "gpu_selector", selector)
    monkeypatch.setattr(resource.inspect_api, "inspect", inspect)


def test_reported_zombie_and_unreadable_cpu_services_are_not_gpu1_candidates(monkeypatch, scratch):
    rows = {
        262335: {"argv": [], "dead": True, "no_inspect": True, "unreadable_selector": True},
        469756: {"argv": ["sshd: bruli@notty"], "no_inspect": True, "unreadable_selector": True},
        964996: {"argv": ["/usr/lib/systemd/systemd", "--user"],
                 "no_inspect": True, "unreadable_selector": True},
        964997: {"argv": ["(sd-pam)"], "no_inspect": True, "unreadable_selector": True},
    }
    scan_fixture(monkeypatch, scratch, rows)
    assert resource.assignment_blockers({}, scratch) == []


def test_unrelated_python_selector_excludes_gpu1_before_cwd_inspection(monkeypatch, scratch):
    rows = {123: {"argv": ["/usr/bin/python3", "-u", "other_qa.py"],
                  "selector": "3,4", "no_inspect": True},
            124: {"argv": ["/usr/bin/python3", "-c", "CPU code"],
                  "selector": "", "no_inspect": True}}
    scan_fixture(monkeypatch, scratch, rows)
    assert resource.assignment_blockers({}, scratch) == []


@pytest.mark.parametrize("entry", ["/usr/bin/python3", "/usr/local/bin/cuda_worker"])
def test_unknown_explicit_gpu1_pending_assignment_blocks(monkeypatch, scratch, entry):
    scan_fixture(monkeypatch, scratch, {123: {"argv": [entry, "worker"], "selector": resource.UUID}})
    assert resource.assignment_blockers({}, scratch) == [
        {"pid": 123, "reason": "unknown/pending GPU1 assignment"}]


def test_unreadable_live_python_assignment_remains_fail_closed(monkeypatch, scratch):
    scan_fixture(monkeypatch, scratch, {
        123: {"argv": ["/usr/bin/python3", "unknown.py"], "unreadable_selector": True,
              "no_inspect": True}})
    assert resource.assignment_blockers({}, scratch)[0]["reason"] == "unreadable live GPU-capable assignment"


def test_unreadable_cwd_for_actual_gpu1_candidate_blocks(monkeypatch, scratch):
    scan_fixture(monkeypatch, scratch, {
        123: {"argv": ["/usr/bin/python3", "unknown.py"], "selector": "1",
              "inspect_error": PermissionError("protected cwd")}})
    assert resource.assignment_blockers({}, scratch)[0]["reason"] == "unreadable/changing live GPU1 candidate"


@pytest.mark.parametrize("dies", [False, True])
def test_gone_race_is_skipped_only_when_identity_dead(monkeypatch, scratch, dies):
    scan_fixture(monkeypatch, scratch, {
        123: {"argv": ["/usr/bin/python3", "worker.py"], "selector": "1",
              "inspect_error": FileNotFoundError("cwd vanished"), "dies_on_inspect": dies}})
    assert bool(resource.assignment_blockers({}, scratch)) is not dies


def test_unexpected_inspection_error_is_fatal_not_a_wait(monkeypatch, scratch):
    scan_fixture(monkeypatch, scratch, {
        123: {"argv": ["/usr/bin/python3", "worker.py"], "selector": "1",
              "inspect_error": RuntimeError("unexpected admission implementation bug")}})
    with pytest.raises(RuntimeError, match="unexpected admission"):
        resource.assignment_blockers({}, scratch)


@pytest.mark.parametrize("waiting", [False, True])
def test_entrypoint_distinguishes_admission_wait_from_fatal_error(monkeypatch, scratch, capsys, waiting):
    monkeypatch.setattr(queue, "OUTPUT", scratch)
    save(scratch / "queue_plan.json", {})
    monkeypatch.setattr(queue, "assert_plan", lambda *a: None)
    def execute(*args):
        if waiting:
            raise resource.Waiting("legitimate resource conflict")
        raise RuntimeError("unexpected admission implementation bug")
    monkeypatch.setattr(queue, "execute", execute)
    code = queue.main(["--execute", "--approval", str(scratch / "approval.json")])
    captured = capsys.readouterr()
    if waiting:
        assert code == 0
        assert json.loads(captured.out)["status"] == "waiting_not_launched"
    else:
        assert code == 1
        assert json.loads(captured.err)["status"] == "failed_closed"


def test_scanner_compares_full_captured_identity_not_pid_alone(monkeypatch, scratch):
    argv = ["/usr/bin/python3", "-u", str(resource.LLAMA)]
    captured = info(123, argv)
    current = {**captured, "identity": {"pid": 123, "start_ticks": 88}}
    scan_fixture(monkeypatch, scratch, {123: {"argv": argv, "selector": "1", "info": current}})
    assert resource.assignment_blockers({123: captured}, scratch)[0]["reason"].startswith("unknown")
    assert resource.assignment_blockers({123: current}, scratch) == []


def test_own_only_recovery_never_signals_pid_reuse(monkeypatch):
    child = {"bootstrap": info(), "execution": info(argv=["python", "worker"])}
    monkeypatch.setattr(resource, "exact", lambda r: False)
    monkeypatch.setattr(resource.inspect_api, "signal_exact", lambda *a: pytest.fail("foreign signal"))
    resource.stop_owned({"child": child})


def test_recovery_stops_only_registered_child_if_owner_dead(monkeypatch, scratch):
    path = scratch / "state.json"
    state = {"owner": info(10), "status": "executing",
             "child": {"bootstrap": info(20), "execution": info(20, ["python", "worker"])}}
    save(path, state)
    stopped = []
    monkeypatch.setattr(resource, "exact", lambda r: False)
    monkeypatch.setattr(resource, "stop_owned", lambda s: stopped.append(s["child"]))
    monkeypatch.setattr(resource, "cleanup", lambda s: True)
    resource.recovery(path)
    assert stopped == [state["child"]]
    assert resource.read(path)["status"] == "owner_dead_released"


def test_bootstrap_needs_cpu_and_registration(scratch, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4")
    with pytest.raises(RuntimeError, match="CUDA-empty"):
        resource.bootstrap(scratch / "unused")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    path = scratch / "state.json"
    save(path, {"owner": info(10)})
    monkeypatch.setattr(resource, "exact", lambda r: False)
    monkeypatch.setattr(resource.os, "execvpe", lambda *a: pytest.fail("CUDA worker launched"))
    with pytest.raises(RuntimeError, match="Controller died"):
        resource.bootstrap(path)


def test_registered_bootstrap_rechecks_admission_before_exec(scratch, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    path = scratch / "state.json"
    plan_path, approval = scratch / "plan.json", scratch / "approval.json"
    save(plan_path, {"pinned": True})
    save(approval, {"approved": True})
    child = info(os.getpid(), ["python", "bootstrap"])
    state = {"owner": info(10), "registered": True, "recovery": info(11),
             "child": {"bootstrap": child, "execution": info(os.getpid(), ["python", "worker"])},
             "command": ["python", "worker"], "plan": str(plan_path),
             "approval": str(approval), "output": str(scratch),
             "maintenance_owner": info(11)["identity"], "foreign_contexts": []}
    save(path, state)
    monkeypatch.setattr(resource, "exact", lambda r: True)
    events = []
    monkeypatch.setattr(queue, "assert_plan", lambda *a: events.append("pins"))
    monkeypatch.setattr(resource, "verify_maintenance", lambda s: events.append("maintenance"))
    monkeypatch.setattr(resource.maintenance, "register_worker",
                        lambda *a: events.append("standard_registered") or child["identity"])
    monkeypatch.setattr(resource, "admit", lambda a: events.append("admission") or {"gpu": 1})
    monkeypatch.setattr(queue, "worker_environment", lambda p: {"CUDA_VISIBLE_DEVICES": resource.UUID})
    class ExecCalled(Exception):
        pass
    def execute(executable, command, env):
        events.append("exec")
        assert command == state["command"]
        assert env["CUDA_VISIBLE_DEVICES"] == resource.UUID
        raise ExecCalled
    monkeypatch.setattr(resource.os, "execvpe", execute)
    with pytest.raises(ExecCalled):
        resource.bootstrap(path)
    assert events == ["pins", "maintenance", "standard_registered", "admission", "exec"]
    assert resource.read(path)["status"] == "executing"


def test_bootstrap_foreign_registration_fails_closed(scratch, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    path = scratch / "state.json"
    save(path, {"owner": info(10), "registered": True, "recovery": info(11),
                "child": {"bootstrap": info(os.getpid() + 1)}})
    monkeypatch.setattr(resource, "exact", lambda r: True)
    monkeypatch.setattr(resource.os, "execvpe", lambda *a: pytest.fail("unregistered exec"))
    with pytest.raises(RuntimeError, match="Registration"):
        resource.bootstrap(path)


def test_own_lock_serial_in_addition_to_standard_maintenance(scratch):
    path = scratch / "reuse_semantic_gpu1.lock"
    with resource.locked(path):
        with pytest.raises(resource.Waiting, match="Another pilot"):
            with resource.locked(path):
                pytest.fail("double acquired")
    assert resource.LOCK.name == "reuse_semantic_gpu1.lock"
    assert "maintenance" not in str(resource.LOCK)


@pytest.fixture
def standard_gate(scratch, monkeypatch):
    monkeypatch.setattr(resource.maintenance, "maintenance_path",
                        lambda gpu: scratch / f"maintenance_gpu{gpu}.json")
    return scratch / "maintenance_gpu1.json"


def test_standard_claim_precedes_snapshot_and_allows_existing_llama(standard_gate, scratch, monkeypatch):
    state = {"owner": info(10), "recovery": info(11), "maintenance_owner": info(11)["identity"],
             "output": str(scratch), "child": None}
    monkeypatch.setattr(resource, "exact", lambda r: True)
    monkeypatch.setattr(resource, "known_llama", lambda *a: True)
    monkeypatch.setattr(resource, "assignment_blockers", lambda p: [])
    monkeypatch.setattr(resource.inspect_api, "inspect", lambda pid: info(pid))
    monkeypatch.setattr(resource.inspect_api, "digest", lambda path: "frozen")
    def snapshot():
        record = resource.maintenance.read_maintenance(standard_gate)
        assert record["owner"] == state["maintenance_owner"]
        assert record["output"] == str(scratch)
        return devices((20, 21))
    monkeypatch.setattr(resource, "snapshot", snapshot)
    peers, admitted = resource.claim_and_capture(state)
    assert [p["info"]["identity"]["pid"] for p in peers] == [20, 21]
    assert admitted["gpu"] == 1 and admitted["foreign_contexts"] == 2
    assert standard_gate.exists()
    assert resource.cleanup(state)
    assert not standard_gate.exists()


def test_standard_registration_blocks_release_until_child_exits(standard_gate, scratch, monkeypatch):
    owner, child = info(11)["identity"], info(20)["identity"]
    resource.maintenance.claim_maintenance(1, scratch, owner)
    live = {11, 20}
    monkeypatch.setattr(resource.maintenance, "identity_alive",
                        lambda token: bool(token) and token["pid"] in live)
    monkeypatch.setattr(resource.maintenance, "process_identity", lambda pid: child)
    assert resource.maintenance.register_worker(1, owner, ["python", "pilot"]) == child
    assert not resource.maintenance.release_maintenance(1, owner)
    assert standard_gate.exists()
    live.remove(20)
    assert resource.maintenance.release_maintenance(1, owner)
    assert not standard_gate.exists()


def test_cleanup_does_not_release_live_own_child_or_foreign_gate(standard_gate, scratch, monkeypatch):
    owner = info(11)["identity"]
    state = {"maintenance_owner": owner, "output": str(scratch),
             "child": {"bootstrap": info(20), "execution": info(20, ["python", "pilot"])}}
    resource.maintenance.claim_maintenance(1, scratch, owner)
    monkeypatch.setattr(resource.inspect_api, "alive", lambda token: True)
    assert resource.cleanup(state) is False
    assert standard_gate.exists()
    monkeypatch.setattr(resource.inspect_api, "alive", lambda token: False)
    assert resource.cleanup(state) is True
    foreign = info(999)["identity"]
    resource.maintenance.claim_maintenance(1, "foreign", foreign)
    assert resource.cleanup(state) is True
    assert resource.maintenance.read_maintenance(standard_gate)["owner"] == foreign


def test_existing_foreign_context_may_exit_but_new_one_is_rejected(monkeypatch):
    monkeypatch.setattr(resource, "exact", lambda record: record["identity"]["pid"] == 20)
    monkeypatch.setattr(resource, "known_llama", lambda *a: True)
    peers = [{"info": info(pid), "source_sha256": "frozen"} for pid in (20, 21)]
    assert resource.admit(peers, devices((20,)), blockers=[])["foreign_contexts"] == 1
    with pytest.raises(resource.Waiting, match="Unknown"):
        resource.admit(peers, devices((20, 22)), blockers=[])


def test_foreign_pid_reuse_is_not_the_captured_context(monkeypatch):
    peers = [{"info": info(20), "source_sha256": "frozen"}]
    monkeypatch.setattr(resource, "exact", lambda record: False)
    with pytest.raises(resource.Waiting, match="Unknown"):
        resource.admit(peers, devices((20,)), blockers=[])


def test_recovery_releases_only_after_own_child_exit(monkeypatch, scratch):
    path = scratch / "state.json"
    state = {"owner": info(10), "status": "executing",
             "child": {"bootstrap": info(20), "execution": info(20, ["python", "pilot"])}}
    save(path, state)
    monkeypatch.setattr(resource, "exact", lambda r: False)
    events = []
    release = iter((False, True))
    monkeypatch.setattr(resource, "stop_owned", lambda s: events.append("own_stop"))
    monkeypatch.setattr(resource, "cleanup", lambda s: events.append("cleanup") or next(release))
    monkeypatch.setattr(resource.time, "sleep", lambda n: events.append("retaining_both_gates"))
    resource.recovery(path)
    assert events == ["own_stop", "cleanup", "retaining_both_gates", "own_stop", "cleanup"]
    assert resource.read(path)["status"] == "owner_dead_released"


def test_known_llama_positive_and_strict_scope(standard_gate, scratch, monkeypatch):
    scope = scratch / "llama_scope"
    ready_path = scratch / "llama_ready.json"
    monkeypatch.setattr(resource, "LLAMA_SCOPE", scope)
    monkeypatch.setattr(resource, "LLAMA_READY", ready_path)
    job = {"name": "llama31_8b/seed233/hotpotqa/main/full", "model": "llama31_8b",
           "dataset": "hotpotqa", "seed": 233, "variant": "full", "purpose": "main",
           "samples": 500, "ratio": .5, "max_steps": 7}
    directory = scope / job["name"]
    data = str(resource.BASE / "qa32_assets/data_ready.json")
    save(scope / "queue_plan.json",
         {"model_readys": {"llama31_8b": str(ready_path)}, "data_ready": data, "jobs": [job]})
    repo = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    save(ready_path, {"repo": repo, "revision": queue.original.MODELS[repo]["revision"]})
    fp = {"source_hashes": {"frozen": "sha"}, "protocol": job,
          "assets": {"model_ready_sha256": "frozen"}}
    save(directory / "manifest.json",
         {"fingerprint": fp, "fingerprint_sha256": queue.original.digest(fp)})
    save(directory / "status.json", {"status": "running", "pid": 20})
    record = info(20, ["/usr/bin/python3", "-u", str(resource.LLAMA), "--execute",
                      "--model-ready", str(ready_path), "--data-ready", data, "--output", str(directory),
                      "--dataset", "hotpotqa", "--seed", "233", "--variant", "full",
                      "--purpose", "main"])
    monkeypatch.setattr(resource, "exact", lambda r: True)
    monkeypatch.setattr(resource.inspect_api, "gpu_selector", lambda pid: resource.UUID)
    monkeypatch.setattr(resource.inspect_api, "digest", lambda path: "frozen")
    monkeypatch.setattr(queue.original, "source_hashes", lambda: {"frozen": "sha"})
    assert resource.known_llama(record, "frozen")
    record["uid"] = os.getuid() + 1
    assert not resource.known_llama(record, "frozen")
    record["uid"] = os.getuid()
    record["argv"][-1] = "smoke"
    assert not resource.known_llama(record, "frozen")


def test_frozen_llama_gate_serializes_popen_and_checks_maintenance():
    source = (queue.ROOT / "scripts/run_parallel_llama_ablation.py").read_text()
    gate = source.index("with maintenance_launch_lock(gpu):")
    check = source.index("if maintenance_blockers(maintenance_path(gpu)):", gate)
    popen = source.index("process = subprocess.Popen(", check)
    assert gate < check < popen
    assert "run_rebuttal_efficiency_queue" in source


def test_queue_uses_one_recovery_and_keeps_gate_for_all_ten_jobs(plan, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(queue, "assert_plan", lambda *a: None)
    monkeypatch.setattr(resource, "locked", lambda: contextlib.nullcontext(123))
    monkeypatch.setattr(queue, "summarize", lambda *a: None)
    monkeypatch.setattr(queue, "validate_job", lambda p, j, complete=False: {} if complete else None)
    events, launches, worker_jobs = [], [], []
    monkeypatch.setattr(resource, "claim_and_capture",
                        lambda s: events.append("claim") or ([], {"gpu": 1}))
    monkeypatch.setattr(resource, "verify_maintenance", lambda s: None)
    monkeypatch.setattr(resource, "admit", lambda p: {"gpu": 1})
    monkeypatch.setattr(resource, "cleanup", lambda s: True)
    monkeypatch.setattr(resource.inspect_api, "inspect", lambda pid: info(pid))
    class Process:
        returncode = 0
        def __init__(self, command):
            self.pid = 100 + len(launches)
            self.recovery = resource.flag(command, "--mode") == "recovery"
        def poll(self):
            return None if self.recovery else 0
        def wait(self, timeout):
            events.append("release")
            return 0
    def spawn(command, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        launches.append(command)
        events.append(resource.flag(command, "--mode"))
        if resource.flag(command, "--mode") == "bootstrap":
            state = resource.read(Path(plan["output"]) / "resource_state.json")
            worker_jobs.append(state["job"])
            assert not state["registered"]
        return Process(command)
    monkeypatch.setattr(queue.subprocess, "Popen", spawn)
    queue.execute(clean(plan), Path(plan["output"]) / "approval.json")
    assert events.count("recovery") == 1 and events.count("bootstrap") == 10
    assert events.index("claim") < events.index("bootstrap") < events.index("release")
    assert events[-1] == "release"
    assert worker_jobs == [j["name"] for j in queue.jobs()]
    assert all(name.startswith("smoke/") for name in worker_jobs[:5])
    assert all(name.startswith("main/") for name in worker_jobs[5:])


def test_failed_first_auxiliary_smoke_never_admits_or_registers_main(plan, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(queue, "assert_plan", lambda *a: None)
    monkeypatch.setattr(resource, "locked", lambda: contextlib.nullcontext(123))
    monkeypatch.setattr(queue, "summarize", lambda *a: None)
    monkeypatch.setattr(queue, "validate_job", lambda *a, **k: None)
    monkeypatch.setattr(resource, "claim_and_capture", lambda s: ([], {"gpu": 1}))
    monkeypatch.setattr(resource, "verify_maintenance", lambda s: None)
    admitted, registered = [], []
    monkeypatch.setattr(resource, "admit", lambda peers: admitted.append("smoke") or {"gpu": 1})
    monkeypatch.setattr(resource, "stop_owned", lambda s: None)
    monkeypatch.setattr(resource, "cleanup", lambda s: True)
    monkeypatch.setattr(resource.inspect_api, "inspect", lambda pid: info(pid))
    class Process:
        def __init__(self, recovery):
            self.pid = 101 if recovery else 102
            self.recovery = recovery
            self.returncode = None if recovery else 1
        def poll(self):
            return self.returncode
        def wait(self, timeout):
            return 0
    def spawn(command, **kwargs):
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        is_recovery = resource.flag(command, "--mode") == "recovery"
        if not is_recovery:
            state = resource.read(Path(plan["output"]) / "resource_state.json")
            registered.append(state["job"])
        return Process(is_recovery)
    monkeypatch.setattr(queue.subprocess, "Popen", spawn)
    with pytest.raises(RuntimeError, match="worker failed"):
        queue.execute(clean(plan), Path(plan["output"]) / "approval.json")
    assert admitted == ["smoke"]
    assert registered == ["smoke/lexical_paraphrase"]
    assert resource.read(Path(plan["output"]) / "resource_state.json")["status"] == "failed"

def fingerprint(plan, job):
    selection = plan["samples"][job["purpose"]]
    return {"protocol": {**job, "pilot_condition": job["condition"]},
            "source_hashes": plan["worker_sources"], "sample_ids": selection["ids"],
            "sample_indices": selection["indices"], "selected_content_sha256": selection["content_sha256"],
            "assets": plan["assets_provenance"], "generation_config": plan["raw_hf_generation_config"],
            "versions": plan["runtime_versions"]}


def test_hash_prefix_resume_and_completion_validation(plan, monkeypatch):
    job = next(j for j in queue.jobs() if j["name"] == "smoke/lexical_original")
    directory = Path(plan["output"]) / job["name"]
    fp = fingerprint(plan, job)
    sha = queue.original.digest(fp)
    save(directory / "manifest.json", {"fingerprint": fp, "fingerprint_sha256": sha})
    save(directory / "checkpoint.json",
         {"fingerprint_sha256": sha, "rows": [{"id": "500"}],
          "rows_sha256": queue.original.digest([{"id": "500"}])})
    calls = []
    monkeypatch.setattr(queue.original, "validate_fingerprint", lambda fp: None)
    monkeypatch.setattr(queue.original, "validate_rows", lambda rows, fp: calls.append(rows))
    assert queue.validate_job(plan, job) is None
    assert calls == [[{"id": "500"}]]
    checkpoint = queue.original.read_json(directory / "checkpoint.json")
    checkpoint["rows"][0]["id"] = "other"
    save(directory / "checkpoint.json", checkpoint)
    with pytest.raises(RuntimeError, match="prefix hash"):
        queue.validate_job(plan, job)


def test_smoke_actual_pruning_required(plan, monkeypatch):
    job = next(j for j in queue.jobs() if j["name"] == "smoke/lexical_original")
    fp = fingerprint(plan, job)
    directory = Path(plan["output"]) / job["name"]
    save(directory / "manifest.json",
         {"fingerprint": fp, "fingerprint_sha256": queue.original.digest(fp)})
    monkeypatch.setattr(queue.original, "validate_fingerprint", lambda fp: None)
    monkeypatch.setattr(queue.pilot, "validate_completed_run", lambda p: {"has_actual_eviction": False})
    with pytest.raises(RuntimeError, match="pruning"):
        queue.validate_job(plan, job, complete=True)


def test_all_five_smokes_gate_main(plan, monkeypatch, scratch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(queue, "assert_plan", lambda *a: None)
    monkeypatch.setattr(resource, "locked", lambda: contextlib.nullcontext(123))
    monkeypatch.setattr(queue, "summarize", lambda *a: None)
    monkeypatch.setattr(resource, "admit", lambda *a: pytest.fail("main admitted before smoke"))
    original_jobs = queue.jobs()
    monkeypatch.setattr(queue, "jobs", lambda: original_jobs[5:])
    monkeypatch.setattr(queue, "validate_job", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="smokes"):
        queue.execute(clean(plan), scratch / "approval")


def test_paired_bootstrap_is_question_paired_and_reproducible():
    reference = [{"id": str(i), "em": 0, "f1": .25} for i in range(50)]
    rows = [{"id": str(i), "em": 1, "f1": .5} for i in range(50)]
    assert queue.paired_bootstrap(rows, reference, "em") == {
        "difference_percentage_points": 100, "paired_question_bootstrap_95ci": [100, 100],
        "resamples": 2000, "seed": 233}
    assert queue.paired_bootstrap(rows, reference, "f1")["difference_percentage_points"] == 25
    with pytest.raises(RuntimeError, match="identity"):
        queue.paired_bootstrap(rows[::-1], reference, "em")


def test_audit_cosines_changes_invalid_outputs_and_scope():
    rows = [{"id": "q", "reuse_scoring_audit": {
        "citation_rounds": [{"increments": {"1": .8}}],
        "runtime": {"pairs": 2, "citation_seconds_including_auxiliary": 3,
                    "events": [{"pairs": [{"raw_cosine": .8}, {"raw_cosine": .2}]}],
                    "encoder": {"events": [{"truncated": True}], "forward_seconds": 1},
                    "paraphraser": {"hits": 1, "misses": 1, "forward_seconds": 2, "events": [
                        {"entry": {"unchanged": False, "original": "raw", "paraphrase": "rewrite",
                                   "truncated": True}}, {"error": "invalid JSON"}]}}}}]
    result = queue.audit_summary(rows)
    assert result["raw_cosines"] == [.8, .2]
    assert result["cosine_mean"] == .5
    assert result["paraphrase_changed"] == result["paraphrase_changed_vs_full_original"] == 1
    assert result["invalid_raw_outputs"] == result["encoder_truncated"] == 1
    assert "NOT latency" in result["timing_policy"]
    assert "not whole-host RSS" in result["memory_scope"]


def test_retry_audit_counts_all_generated_attempts_without_recounting_cache_hits():
    entry = {"unchanged": False, "original": "raw", "paraphrase": "rewrite",
             "attempts": [{"rejection_type": "NumericMismatch", "rejection_reason": "numbers changed",
                           "accepted": False, "raw_response": "bad", "seconds": 1},
                          {"accepted": True, "raw_response": "accepted", "seconds": 2}]}
    rows = [{"id": "q", "reuse_scoring_audit": {"citation_rounds": [], "runtime": {
        "paraphraser": {"forward_seconds": 3, "hits": 1, "misses": 1,
                       "events": [{"cache_hit": False, "entry": entry},
                                  {"cache_hit": True, "entry": entry}]}}}}]
    result = queue.audit_summary(rows)
    assert result["paraphrase_generation_attempts"] == 2
    assert result["paraphrase_retry_attempts"] == 1
    assert result["invalid_raw_outputs"] == 1
    assert result["paraphrase_attempt_audits_missing"] == 0
    assert result["paraphrase_diagnostic_seconds"] == 3
    assert result["memory_audits"][0]["paraphraser"]["events"][0]["entry"]["attempts"] == entry["attempts"]


def test_api_drift_is_error(monkeypatch):
    monkeypatch.setattr(queue.pilot, "validate_completed_run", None)
    with pytest.raises(RuntimeError, match="API drift"):
        queue.api_gate()


def test_real_frozen_worker_sources_include_new_modules_without_cuda():
    sources = queue.source_hashes()
    for name in ("scripts/run_reuse_semantic_pilot.py", "scripts/reuse_semantic_runtime.py",
                 "scripts/reuse_semantic_resources.py", "scripts/run_reuse_semantic_queue.py",
                 "scripts/run_deadline_gpu_load_guard.py", "scripts/run_parallel_qa32_queue.py",
                 "scripts/reuse_semantic_assets.py", "scripts/reuse_semantic_probe.py"):
        assert sources[name] == queue.original.file_hash(queue.ROOT / name)
    assert all(len(value) == 64 for value in sources.values())
