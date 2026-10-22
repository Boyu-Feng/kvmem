"""CPU-only protocol, frozen-AST and runtime-provider regression coverage."""
import ast
import contextlib
import copy
import hashlib
import importlib
import json
import math
from argparse import Namespace
from pathlib import Path

import pytest

from scripts import run_reuse_semantic_pilot as pilot
from test_legacy_efficiency_metrics import (
    assert_legacy_state_equal, make_llm, model_factory,
)


def fixture_episode(question, llm, retriever, pruning_mode="step_aware_h2o", return_debug=True):
    def _compute_step_external_score(reward_val, citation_val):
        reward = max(-1.0, min(2.0, float(reward_val)))
        score = .85 * reward + .15 * math.log1p(max(0., citation_val))
        return float(max(0.0, min(8.0, score)))

    step_meta, step_scores = {}, {}
    use_step_scores = True
    ablate_citation = llm.kv_config.get("step_ablate_citation", False)
    llm.generate_first(question, max_new_tokens=4)
    for prev_step, new_text in enumerate(("Apple alpha\nThought 2:", "Apple beta\nThought 3:"), 1):
        prev_obs_text, prev_thought_text = "Apple alpha", "beta evidence"
        prev_anchor_text = "\n".join((prev_obs_text, prev_thought_text))
        prev_anchor_terms = extract_anchor_terms(prev_anchor_text)
        step_meta.setdefault(prev_step, {})
        step_meta[prev_step]["anchors"] = prev_anchor_terms
        step_meta[prev_step]["reward"] = 1.5
        step_meta[prev_step].setdefault("citation", 0.)
        step_scores[prev_step] = _compute_step_external_score(
            step_meta[prev_step]["reward"], step_meta[prev_step]["citation"])
        if pruning_mode == "step_aware_h2o" and use_step_scores and step_meta and not ablate_citation:
            ref_terms = extract_anchor_terms(new_text)
            if ref_terms:
                for sid, meta in step_meta.items():
                    anchors = meta.get("anchors", set())
                    if not anchors:
                        continue
                    overlap = len(ref_terms.intersection(anchors))
                    if overlap <= 0:
                        continue
                    base = float(min(len(ref_terms), len(anchors)))
                    overlap_ratio = float(overlap) / max(1.0, base)
                    citation_inc = min(1.0, overlap_ratio * 1.5)
                    meta["citation"] = float(meta.get("citation", 0.0)) + float(citation_inc)
                    step_scores[sid] = _compute_step_external_score(
                        float(meta.get("reward", 0.0)),
                        float(meta.get("citation", 0.0)),
                    )
            if llm.kv_manager is not None and hasattr(llm.kv_manager, "update_step_scores"):
                llm.kv_manager.update_step_scores(step_scores)
            _print_step_scores_summary()
        kv_stop_strings = ["\nObservation", "\nQuestion:"]
        response = llm.generate_incremental(
            new_text, max_new_tokens=256, stop_strings=kv_stop_strings)
    return response, [], [], {"step_meta": step_meta, "step_scores": step_scores}


def extract_anchor_terms(text):
    return set(text.lower().split())


def _print_step_scores_summary():
    pass


class FakeScorer:
    instances = []

    def __init__(self, condition, **kwargs):
        self.condition, self.kwargs = condition, kwargs
        self.registered, self.currents, self.resets = [], [], 0
        self.closed = False
        self.instances.append(self)

    def reset_episode(self):
        self.resets += 1
        self.registered.clear()
        self.currents.clear()

    def register_step(self, sid, original_text, original_anchor_set):
        self.registered.append((sid, original_text, set(original_anchor_set)))

    def increments(self, current_original_text, step_meta):
        self.currents.append(current_original_text)
        assert all("citation_text" in meta for meta in step_meta.values())
        return {sid: .25 for sid in step_meta}

    def fingerprint(self):
        result = {"condition": self.condition, "auxiliary_generation_config": {"greedy": True},
                  "encoder_protocol": "fake-only"}
        if "paraphrase" in self.condition:
            result["paraphrase_generation"] = {
                "max_attempts": 3, "retry_strategy": "numeric_diff_guided_v1",
                "max_new_tokens": 512, "max_source_tokens": 512,
                "do_sample": False, "num_beams": 1, "dtype": "bfloat16",
            }
        return result

    def audit(self):
        return {"calls": len(self.currents), "seconds": 0., "truncations": 0,
                "semantic_raw_scores": [.25], "rewritten_texts": ["fake scorer-only"]}

    def close(self):
        self.closed = True


def args(condition="lexical_original", purpose="main"):
    return Namespace(dataset="hotpotqa", seed=233,
                     variant="no_cite" if condition == "no_citation" else "full",
                     purpose=purpose, condition=condition, encoder_ready=Path("encoder-ready.json"),
                     paraphrase_cache=Path("scorer-cache"), model_ready=Path("qwen-ready.json"),
                     data_ready=Path("data-ready.json"), output=Path("pilot-unused-output"),
                     execute=False)


@pytest.fixture
def base():
    return importlib.import_module("run_all_wiki_experiments_v2")


@pytest.mark.parametrize("condition", pilot.CONDITIONS)
def test_five_conditions_only_change_budget_and_original_citation_mode(base, condition):
    spec = args(condition)
    expected = pilot.worker.component_config(base, spec.variant)
    expected["cache_ratio"] = .2
    original_selector = importlib.import_module("kv_cache.pruning_strategy").PruningStrategy._prune_step_aware_h2o
    with pilot.condition_scope(condition, spec.encoder_ready, spec.paraphrase_cache):
        assert pilot.worker.component_config(base, spec.variant) == expected
        protocol = pilot.worker.protocol(spec)
        assert protocol["samples"] == 50 and protocol["budget_percent"] == 20
        assert protocol["max_steps"] == 7 and protocol["greedy"]
        assert protocol["memory_limit_gib"] == 44 and protocol["ratio"] == .2
        assert protocol["mainagent_source_text_policy"] == "unchanged"
        assert protocol["max_attempts"] == 3
        assert protocol["retry_strategy"] == "numeric_diff_guided_v1"
        if "paraphrase" in condition:
            auxiliary = protocol["auxiliary_generation_config"]
            assert auxiliary["max_attempts"] == 3
            assert auxiliary["retry_strategy"] == "numeric_diff_guided_v1"
            assert auxiliary["max_new_tokens"] == auxiliary["max_source_tokens"] == 512
            assert auxiliary["do_sample"] is False and auxiliary["num_beams"] == 1
        else:
            assert protocol["auxiliary_generation_config"] is None
        assert "accuracy-only" in protocol["timing_scope"]
        assert importlib.import_module("kv_cache.pruning_strategy").PruningStrategy._prune_step_aware_h2o is original_selector
    assert not FakeScorer.instances


def test_exact_seed233_first50_and_independent_smoke(base):
    rows = [{"id": f"q{i}", "question": f"question{i}", "answer": "a"} for i in range(800)]
    original_main = pilot.worker.select_samples(base, rows, seed=233, purpose="main")
    original_smoke = pilot.worker.select_samples(base, rows, seed=233, purpose="smoke")
    for condition in pilot.CONDITIONS:
        with pilot.condition_scope(condition, Path("encoder-ready.json"), Path("cache")):
            main = pilot.worker.select_samples(base, rows, seed=233, purpose="main")
            smoke = pilot.worker.select_samples(base, rows, seed=233, purpose="smoke")
            assert main == original_main[:50]
            assert smoke == original_smoke and len(smoke) == 2
            assert not {row["id"] for _, row in main} & {row["id"] for _, row in smoke}


@pytest.mark.parametrize("condition", ("lexical_original", "no_citation"))
def test_original_ast_byte_equivalent_after_removing_observational_hooks(base, condition):
    class StripHooks(ast.NodeTransformer):
        def visit_Expr(self, node):
            if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
                if node.value.func.id.startswith("_reuse_"):
                    return None
            return self.generic_visit(node)

        def visit_Assign(self, node):
            if any(ast.unparse(t) == "step_meta[prev_step]['citation_text']" for t in node.targets):
                return None
            return self.generic_visit(node)

    original = ast.parse(pilot.source_text(base._run_react_kv_episode))
    instrumented = pilot.episode_tree(base._run_react_kv_episode, condition)
    restored = StripHooks().visit(instrumented)
    assert ast.dump(restored) == ast.dump(original)
    assert pilot.ast_proof(base._run_react_kv_episode, condition)["citation_loop_policy"] == "original"


@pytest.mark.parametrize("before,after", [
    ("overlap_ratio * 1.5", "overlap_ratio * 2.0"),
    ("min(1.0, overlap_ratio", "min(0.5, overlap_ratio"),
    ('meta.get("anchors", set())', 'meta.get("anchors", [])'),
    ("max_new_tokens=256", "max_new_tokens=128"),
    ("min(2.0, float(reward_val))", "min(4.0, float(reward_val))"),
    ("min(8.0, score)", "min(9.0, score)"),
    ('"\\n".join((prev_obs_text, prev_thought_text))',
     '" ".join((prev_obs_text, prev_thought_text))'),
])
def test_source_expression_drift_fails_closed(before, after):
    text = pilot.source_text(fixture_episode)
    assert before in text
    with pytest.raises(RuntimeError, match="Frozen"):
        pilot.episode_tree(text.replace(before, after), "semantic_original")


def test_duplicate_ast_shape_fails_closed():
    text = pilot.source_text(fixture_episode)
    line = '        step_meta[prev_step]["anchors"] = prev_anchor_terms\n'
    with pytest.raises(RuntimeError, match="found 2"):
        pilot.episode_tree(text.replace(line, line * 2), "lexical_original")


@pytest.mark.parametrize("condition", pilot.CONDITIONS)
def test_plan_stage_never_constructs_runtime_models_or_cuda(monkeypatch, capsys, condition):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA queried"))
    monkeypatch.setattr(pilot.worker, "load_assets", lambda *a: pytest.fail("Assets loaded"))
    original_import = pilot.importlib.import_module

    def safe_import(name, *a, **kw):
        if name == "scripts.reuse_semantic_runtime":
            pytest.fail("Runtime imported during plan")
        return original_import(name, *a, **kw)

    monkeypatch.setattr(pilot.importlib, "import_module", safe_import)
    spec = args(condition)
    command = ["--condition", condition, "--dataset", "hotpotqa", "--seed", "233",
               "--variant", spec.variant, "--purpose", "main", "--model-ready", str(spec.model_ready),
               "--data-ready", str(spec.data_ready), "--output", str(spec.output),
               "--encoder-ready", str(spec.encoder_ready), "--paraphrase-cache", str(spec.paraphrase_cache)]
    pilot.main(command)
    plan = json.loads(capsys.readouterr().out)
    assert plan["status"] == "planned_not_executed"
    assert plan["samples"] == 50 and plan["kv_override"]["cache_ratio"] == .2
    assert not spec.output.exists()


@pytest.mark.parametrize("field,value", [
    ("dataset", "2wiki"), ("seed", 3407), ("variant", "no_step"),
    ("encoder_ready", None), ("paraphrase_cache", None),
])
def test_registration_rejects_unapproved_protocol(field, value):
    spec = args("semantic_paraphrase")
    setattr(spec, field, value)
    with pytest.raises(ValueError):
        pilot.validate_args(spec)


def setup_episode(monkeypatch, base, llm):
    monkeypatch.setattr(base, "_run_react_kv_episode", fixture_episode)
    monkeypatch.setattr(pilot.worker, "pruning_observer",
                        lambda llm: contextlib.nullcontext(([], [])))
    original = llm.generate_incremental
    inputs = []

    def small_generation(text, max_new_tokens=256, stop_strings=None):
        inputs.append((text, max_new_tokens, stop_strings))
        return original(text, max_new_tokens=6, stop_strings=stop_strings)

    monkeypatch.setattr(llm, "generate_incremental", small_generation)
    return inputs


def test_lexical_original_preserves_original_method_tokens_kv_and_increment_bytes(
        monkeypatch, base, model_factory):
    reference, instrumented = make_llm("step_aware_h2o", .2), make_llm("step_aware_h2o", .2)
    reference.kv_config.update(step_reward_weight=.85, step_citation_weight=.15)
    instrumented.kv_config.update(step_reward_weight=.85, step_citation_weight=.15)
    setup_episode(monkeypatch, base, reference)
    expected_inputs = reference.generate_incremental
    plain = fixture_episode("question", reference, None)
    actual_inputs = setup_episode(monkeypatch, base, instrumented)
    with pilot.condition_scope("lexical_original", execute=True):
        row = pilot.worker.run_episode(base, {"question": "question", "answer": ""}, instrumented, None)
        audit = row["reuse_scoring_audit"]
        assert row["predicted_answer"].encode() == plain[0].encode()
        actual_debug = copy.deepcopy(row["legacy_debug"])
        for meta in actual_debug["step_meta"].values():
            del meta["citation_text"]
        assert actual_debug == plain[3]
        # Original loop's cap/denominator and float increments, not a substitute scorer.
        assert audit["citation_rounds"][0]["increments"] == {1: .75}
        assert audit["citation_rounds"][1]["increments"] == {1: .75, 2: .75}
        assert [r["utf8_sha256"] for r in audit["main_incremental_inputs"]] == [
            hashlib.sha256(text.encode()).hexdigest() for text, _, _ in actual_inputs]
        assert all(cap == 256 for _, cap, _ in actual_inputs)
    assert_legacy_state_equal(reference, instrumented)
    assert reference.get_global_token_id_log() == instrumented.get_global_token_id_log()
    assert expected_inputs is reference.generate_incremental


@pytest.mark.parametrize("condition", (
    "semantic_original", "lexical_paraphrase", "semantic_paraphrase",
))
def test_parent_provider_receives_exact_original_text_never_transforms_main_route(
        monkeypatch, base, model_factory, condition):
    FakeScorer.instances.clear()
    llm = make_llm("step_aware_h2o", .2)
    inputs = setup_episode(monkeypatch, base, llm)
    with pilot.condition_scope(condition, Path("encoder-ready.json"), Path("para-cache"),
                               execute=True, qwen_model_path=Path("pinned-qwen"),
                               runtime_factory=FakeScorer):
        assert not FakeScorer.instances
        for _ in range(2):
            row = pilot.worker.run_episode(base, {"question": "question", "answer": ""}, llm, None)
            scorer = FakeScorer.instances[-1]
            assert scorer.currents == [text for text, _, _ in inputs[-2:]]
            assert all(text == "Apple alpha\nbeta evidence" for _, text, _ in scorer.registered)
            assert row["legacy_debug"]["step_meta"][1]["citation"] == .5
            assert row["legacy_debug"]["step_meta"][2]["citation"] == .25
            assert row["reuse_scoring_audit"]["runtime"]["rewritten_texts"] == ["fake scorer-only"]
            assert "fake scorer-only" not in [text for text, _, _ in inputs]
            assert len(row["reuse_scoring_audit"]["main_incremental_inputs"]) == 2
        assert scorer.resets == 2
        assert scorer.kwargs["device"] == "cuda:0"
    assert scorer.closed and len(FakeScorer.instances) == 1
    FakeScorer.instances.clear()


def test_no_citation_uses_disabled_original_branch_and_never_calls_provider(
        monkeypatch, base, model_factory):
    llm = make_llm("step_aware_h2o", .2)
    llm.kv_config["step_ablate_citation"] = True
    setup_episode(monkeypatch, base, llm)
    with pilot.condition_scope("no_citation", execute=True,
                               runtime_factory=lambda *a, **k: pytest.fail("Provider called")):
        row = pilot.worker.run_episode(base, {"question": "question", "answer": ""}, llm, None)
        assert row["reuse_scoring_audit"]["citation_rounds"] == []
        assert row["reuse_scoring_audit"]["runtime"]["runtime_citation_calls"] == 0
        assert all(meta["citation"] == 0 for meta in row["legacy_debug"]["step_meta"].values())


def test_source_hashes_pin_only_original_and_named_pilot_helpers(base):
    original = pilot.worker.source_hashes()
    with pilot.condition_scope("lexical_original"):
        hashes = pilot.worker.source_hashes()
    assert all(hashes[key] == value for key, value in original.items())
    extra = set(hashes) - set(original)
    assert "scripts/run_reuse_semantic_pilot.py" in extra
    assert all(Path(name).name.startswith(("reuse_semantic_", "run_reuse_semantic_")) for name in extra)


def test_mutable_paraphrase_cache_not_hashed_in_protocol(monkeypatch, base):
    def reject_cache(path):
        assert Path(path) != Path("mutable-para-cache.json")
        return "encoder-ready-hash"

    monkeypatch.setattr(pilot.worker, "file_hash", reject_cache)
    with pilot.condition_scope("semantic_paraphrase", Path("nonexistent-encoder-ready.json"),
                               Path("mutable-para-cache.json")):
        spec = pilot.worker.protocol(args("semantic_paraphrase"))
        assert spec["paraphrase_cache"].endswith("mutable-para-cache.json")
        assert "paraphrase_cache_sha256" not in spec


def test_missing_provider_is_fail_closed_not_fake_fallback(monkeypatch, base, model_factory):
    llm = make_llm("step_aware_h2o", .2)
    setup_episode(monkeypatch, base, llm)
    original_import = pilot.importlib.import_module

    def missing(name, *a, **kw):
        if name == "scripts.reuse_semantic_runtime":
            raise ModuleNotFoundError(name)
        return original_import(name, *a, **kw)

    monkeypatch.setattr(pilot.importlib, "import_module", missing)
    with pilot.condition_scope("semantic_original", Path("encoder-ready.json"),
                               execute=True, qwen_model_path=Path("qwen")):
        with pytest.raises(ModuleNotFoundError):
            pilot.worker.run_episode(base, {"question": "q", "answer": "a"}, llm, None)


@pytest.mark.parametrize("condition", ("lexical_paraphrase", "semantic_paraphrase"))
def test_all_retry_attempt_audits_and_costs_preserved_without_casting_or_defaults(
        monkeypatch, base, model_factory, condition):
    class RetryingProvider(FakeScorer):
        def fingerprint(self):
            return super().fingerprint()

        def audit(self):
            return {
                "paraphraser": {
                    "max_attempts": 3, "retry_strategy": "numeric_diff_guided_v1",
                    "auxiliary_generate_calls": 3, "auxiliary_forward_calls": 41,
                    "forward_seconds": 2.375, "failed_attempts": 2,
                    "events": [{
                        "original": "There are 2 trees.",
                        "attempts": [
                            {"attempt": 1, "raw_response": '{"paraphrase":"There are 3 trees."}',
                             "numeric_diff": {"missing": {"2": 1}, "extra": {"3": 1}},
                             "seconds": .5, "auxiliary_forward_calls": 13, "accepted": False},
                            {"attempt": 2, "raw_response": '{"paraphrase":"There are 4 trees."}',
                             "numeric_diff": {"missing": {"2": 1}, "extra": {"4": 1}},
                             "seconds": .625, "auxiliary_forward_calls": 13, "accepted": False},
                            {"attempt": 3, "raw_response": '{"paraphrase":"2 trees exist."}',
                             "numeric_diff": {"missing": {}, "extra": {}},
                             "seconds": 1.25, "auxiliary_forward_calls": 15, "accepted": True},
                        ],
                    }],
                },
                "encoder": {"forward_seconds": .125, "truncations": 0},
            }

    FakeScorer.instances.clear()
    llm = make_llm("step_aware_h2o", .2)
    setup_episode(monkeypatch, base, llm)
    with pilot.condition_scope(condition, Path("encoder-ready.json"), Path("para-cache"),
                               execute=True, qwen_model_path=Path("qwen"),
                               runtime_factory=RetryingProvider):
        row = pilot.worker.run_episode(base, {"question": "question", "answer": ""}, llm, None)
        provider = FakeScorer.instances[-1]
        audit = row["reuse_scoring_audit"]
        assert audit["runtime"] == provider.audit()
        assert audit["runtime_fingerprint"] == provider.fingerprint()
        assert audit["paraphrase_retry_protocol"] == pilot.PARAPHRASE_RETRY_PROTOCOL
        # The original constructor contract remains intact: the provider owns its
        # approved three-attempt default, not a worker-side retry implementation.
        assert set(provider.kwargs) == {
            "encoder_ready", "qwen_model_path", "paraphrase_cache", "device",
        }
        attempts = audit["runtime"]["paraphraser"]["events"][0]["attempts"]
        assert [attempt["accepted"] for attempt in attempts] == [False, False, True]
        assert sum(attempt["seconds"] for attempt in attempts) == 2.375
        assert sum(attempt["auxiliary_forward_calls"] for attempt in attempts) == 41
        original_hash = pilot.worker.digest(audit)
        attempts[0]["raw_response"] = "altered"
        assert pilot.worker.digest(audit) != original_hash
        assert provider.audit()["paraphraser"]["events"][0]["attempts"][0]["raw_response"] != "altered"
    FakeScorer.instances.clear()


def test_exhausted_provider_retries_fail_without_main_text_fallback(
        monkeypatch, base, model_factory):
    class ExhaustedProvider(FakeScorer):
        def increments(self, current_original_text, step_meta):
            raise ValueError("Numeric validation failed after 3 attempts")

    FakeScorer.instances.clear()
    llm = make_llm("step_aware_h2o", .2)
    inputs = setup_episode(monkeypatch, base, llm)
    with pilot.condition_scope("lexical_paraphrase", paraphrase_cache=Path("para-cache"),
                               execute=True, qwen_model_path=Path("qwen"),
                               runtime_factory=ExhaustedProvider):
        with pytest.raises(ValueError, match="after 3 attempts"):
            pilot.worker.run_episode(base, {"question": "question", "answer": ""}, llm, None)
        assert inputs == []
    assert FakeScorer.instances[-1].closed
    FakeScorer.instances.clear()


@pytest.mark.parametrize("key,value", [
    ("max_attempts", 1), ("max_attempts", "3"), ("max_attempts", None),
    ("max_new_tokens", 256), ("do_sample", True), ("retry_strategy", None),
    ("retry_strategy", ""),
])
def test_provider_fingerprint_missing_or_changed_defaults_fail_closed(key, value):
    provider = FakeScorer("lexical_paraphrase")
    fingerprint = provider.fingerprint()
    if value is None:
        del fingerprint["paraphrase_generation"][key]
    else:
        fingerprint["paraphrase_generation"][key] = value
    with pytest.raises(RuntimeError, match="fingerprint"):
        pilot.validate_runtime_fingerprint(fingerprint, "lexical_paraphrase")
    FakeScorer.instances.clear()


def test_suppressed_original_errors_remain_fatal(monkeypatch, base, model_factory):
    llm = make_llm("step_aware_h2o", .2)
    setup_episode(monkeypatch, base, llm)
    monkeypatch.setattr(llm, "generate_incremental",
                        lambda *a, **k: print("[ERROR] generation failed") or "")
    with pilot.condition_scope("lexical_original", execute=True):
        with pytest.raises(RuntimeError, match="suppressed errors"):
            pilot.worker.run_episode(base, {"question": "q", "answer": "a"}, llm, None)


def test_persisted_checkpoint_hash_includes_post_episode_audit(monkeypatch, base):
    """Exercise the actual store with in-memory atomic I/O, no test files."""
    qa = importlib.import_module("scripts.run_qa32_rebuttal")
    files = {}
    original_exists = Path.exists
    monkeypatch.setattr(Path, "exists",
                        lambda path: str(path) in files or original_exists(path))
    monkeypatch.setattr(qa, "atomic_json",
                        lambda path, value: files.__setitem__(str(path), copy.deepcopy(value)))
    original_read = pilot.worker.read_json
    monkeypatch.setattr(pilot.worker, "read_json",
                        lambda path: copy.deepcopy(files[str(path)])
                        if str(path) in files else original_read(path))
    rows = [{"id": f"q{i}", "question": f"question{i}", "answer": "a"} for i in range(800)]
    with pilot.condition_scope("lexical_original"):
        selected = pilot.worker.select_samples(base, rows, seed=233, purpose="smoke")
        content = [[index, row] for index, row in selected]
        proof = pilot.worker.protocol(args(purpose="smoke"))["ast_proof"]
        fingerprint = {
            "protocol": pilot.worker.protocol(args(purpose="smoke")),
            "source_hashes": pilot.worker.source_hashes(),
            "kv_config": pilot.worker.component_config(base, "full"),
            "sample_ids": [row["id"] for _, row in selected],
            "sample_indices": [index for index, _ in selected],
            "selected_samples": content, "selected_content_sha256": pilot.worker.digest(content),
        }
        store = pilot.worker.ResultStore(Path("pilot-in-memory-store"), fingerprint)
        index, sample = selected[0]
        row = {
            "status": "success", "id": sample["id"], "index": index,
            "question": sample["question"], "gold_answer": "a", "predicted_answer": "a",
            "em": 1, "f1": 1., "diagnostic_elapsed_seconds": .1,
            "pruning_audit": [], "has_actual_eviction": False,
            "reuse_scoring_audit": {"pilot_condition": "lexical_original", "ast_proof": proof,
                                   "mainagent_source_text_policy": "unchanged",
                                   "runtime_fingerprint": {"mode": "lexical_original"}},
        }
        store.commit(row)
        checkpoint = files["pilot-in-memory-store/checkpoint.json"]
        assert checkpoint["rows_sha256"] == pilot.worker.digest([row])
        checkpoint["rows"][0]["reuse_scoring_audit"]["runtime_fingerprint"]["mode"] = "tampered"
        with pytest.raises(RuntimeError, match="Corrupt"):
            pilot.worker.ResultStore(Path("pilot-in-memory-store"), fingerprint)
