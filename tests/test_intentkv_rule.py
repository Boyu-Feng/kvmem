"""Focused offline CPU checks; no GPU, model download, or system scratch files."""
import copy
import importlib
import math
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
import uuid

import pytest
import torch
from transformers import AutoTokenizer, BatchEncoding, Qwen2Config, Qwen2ForCausalLM

from rebuttal.adapters.intentkv_rule.query_memory import QueryMemory, kept_positions
from rebuttal.adapters.intentkv_rule.adapter import IntentKVRuleLLM, OVERRIDES
from models.QwenLLMWithKVCache import QwenLLMWithKVCache
from scripts.legacy_efficiency_metrics import cache_tensor_pairs
from scripts import run_intentkv_rule as worker
from scripts.run_qa32_rebuttal import ResultStore, atomic_json, read_json
from token_tracker import TokenTracker


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for name in ("synchronize", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, lambda *a, **k: pytest.fail("Touched CUDA"))
    torch.set_num_threads(2)


def test_equation_two_normalizes_each_head_after_every_recurrence():
    memory = QueryMemory()
    first = torch.tensor([[3., 4.], [0., 2.]])
    second = torch.tensor([[2., -1.], [4., 0.]])
    memory.update([first])
    expected = first / first.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(memory.rows[0], expected)
    memory.update([second])
    expected = math.exp(-.5) * expected + second
    expected /= expected.norm(dim=-1, keepdim=True)
    torch.testing.assert_close(memory.rows[0], expected)
    assert memory.updates == 2


def test_equation_three_literal_candidate_softmax_gqa_and_all_layers():
    memory = QueryMemory()
    memory.update([torch.tensor([[1., 2.], [-3., 1.], [2., 1.], [1., -1.]]),
                   torch.tensor([[2., 1.], [1., 3.], [1., -1.], [-1., 1.]])])
    torch.manual_seed(233)
    keys = [torch.randn(1, 2, 5, 2), torch.randn(1, 2, 5, 2)]
    keys[0][:, :, 0] = 10000  # Forced prefix must NOT enter the softmax denominator.
    candidates = [1, 3, 4]
    expected = torch.zeros(3)
    for layer in range(2):
        for head in range(4):
            logits = torch.tensor([
                float(memory.rows[layer][head] @ keys[layer][0, head // 2, j]) / math.sqrt(2)
                for j in candidates])
            expected += torch.exp(logits) / torch.exp(logits).sum()
    torch.testing.assert_close(memory.scores(keys, candidates), expected)
    assert math.isclose(float(expected.sum()), 8, abs_tol=1e-5)


def test_forced_overflow_stable_ties_empty_and_live_only_candidates():
    memory = QueryMemory()
    memory.update([torch.ones(4, 2)])
    keys = [torch.zeros(1, 2, 6, 2)]
    assert kept_positions(memory, keys, [0, 4], 1) == [0, 4]
    assert kept_positions(memory, keys, [0, 4], 4) == [0, 1, 2, 4]
    assert kept_positions(memory, keys, [0], 99) == list(range(6))
    assert memory.scores(keys, []).numel() == 0
    # Previously dropped slots simply are not inputs to the next scorer.
    compact = [keys[0][:, :, [0, 2, 4]]]
    assert kept_positions(memory, compact, [0], 2) == [0, 1]


@pytest.mark.parametrize("encoding", [torch.ones(2), torch.tensor([[float("nan")]])])
def test_invalid_encodings_fail_closed(encoding):
    with pytest.raises(ValueError):
        QueryMemory().update([encoding])


class TinyTokenizer:
    eos_token_id = 63

    def apply_chat_template(self, messages, **kwargs):
        return "SYS|" + messages[-1]["content"] + "|ASSIST"

    def __call__(self, text, return_tensors=None, return_offsets_mapping=False, **kwargs):
        ids = [ord(c) - 256 if 256 <= ord(c) < 320 else ord(c) % 61 + 1 for c in text]
        if return_tensors:
            tensor = torch.tensor([ids])
            return BatchEncoding({"input_ids": tensor, "attention_mask": torch.ones_like(tensor)})
        if return_offsets_mapping:
            return {"input_ids": ids, "offset_mapping": [(i, i + 1) for i in range(len(text))]}
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids, **kwargs):
        return "".join(chr(256 + int(i)) for i in ids)


@pytest.fixture
def factory(monkeypatch):
    torch.manual_seed(233)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=1024, eos_token_id=None, pad_token_id=0)
    config._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(config).eval()
    model.generation_config.repetition_penalty = 1.05
    module = importlib.import_module("models.QwenLLMWithKVCache")
    monkeypatch.setattr(module, "AutoModelForCausalLM",
                        SimpleNamespace(from_pretrained=lambda *a, **k: copy.deepcopy(model)))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: TinyTokenizer())

    def create(adapted=True, ratio=.5):
        cls = IntentKVRuleLLM if adapted else QwenLLMWithKVCache
        config = {**OVERRIDES, "cache_ratio": ratio}
        if not adapted:
            llm = cls("tiny-offline", dict(config, pruning_mode="tova"),
                      token_tracker=TokenTracker())
            llm.kv_config = config
            llm._do_pruning = lambda *a, **k: None
            return llm
        return cls("tiny-offline", config, token_tracker=TokenTracker())
    return create


def assert_same_tensors(left, right):
    for a, b in zip(cache_tensor_pairs(left.past_key_values), cache_tensor_pairs(right.past_key_values)):
        for x, y in zip(a, b):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert left.current_cache_len == right.current_cache_len
    assert left.token_tracker.__dict__ == right.token_tracker.__dict__
    assert left._all_token_ids == right._all_token_ids
    assert left._global_token_id_log == right._global_token_id_log


def test_placement_reads_real_tensors_without_hf_device_map(factory, monkeypatch):
    from scripts import latency_runtime

    model = factory().model
    assert not hasattr(model, "hf_device_map")
    validated = []
    monkeypatch.setattr(latency_runtime, "validate_loaded_model", validated.append)
    placement = worker.model_placement(model)
    assert validated == [model]
    assert placement == {
        "device_map": {"": "cpu"},
        "layer_devices": ["cpu"] * len(model.model.layers),
        "placement_source": "validated_parameter_and_buffer_devices",
    }


def test_placement_still_rejects_cpu_offload(factory):
    with pytest.raises(RuntimeError, match="Single-GPU model required"):
        worker.model_placement(factory().model)


def test_real_qwen_no_eviction_preserves_original_decoder_forward_count_and_cache(factory):
    original, adapted = factory(False), factory()
    adapted._compute_target_budget = lambda: adapted.current_cache_len
    counts = [0, 0]
    hooks = []
    for i, llm in enumerate((original, adapted)):
        def count(*args, i=i):
            counts[i] += 1
        hooks.append(llm.model.register_forward_hook(count))
    generation = copy.deepcopy(adapted.model.generation_config.to_dict())
    try:
        assert original.generate_first("question", 8)[0] == adapted.generate_first("question", 8)[0]
        assert_same_tensors(original, adapted)
        for text in ("first observation", "second observation"):
            assert original.generate_incremental(text, 8) == adapted.generate_incremental(text, 8)
            assert_same_tensors(original, adapted)
        assert counts[0] == counts[1]
        assert adapted.query_memory.updates == 3
        assert adapted.model.generation_config.to_dict() == generation
        assert adapted.current_cache_len > cache_tensor_pairs(adapted.past_key_values)[0][0].shape[-2]
    finally:
        for hook in hooks:
            hook.remove()


@pytest.mark.parametrize("ratio", [.5, .2])
def test_multiple_blocks_prompt_query_protection_budget_and_no_resurrection(factory, ratio):
    llm = factory(ratio=ratio)
    llm.generate_first("question", 16)
    prompt = llm.kv_manager.protected_prefix_len
    dead = set()
    for _ in range(4):
        llm.generate_incremental("tool", 16)
        event = llm.intent_events[-1]
        assert not dead.intersection(event["kept_inherited_ids"])
        dead.update(event["removed_inherited_ids"])
        assert event["kept_positions"] == sorted(event["kept_positions"])
        assert set(event["forced_positions"]).issubset(event["kept_positions"])
        assert llm.token_tracker.global_id_mapper[:prompt] == list(range(prompt))
        assert event["resident_after"] == min(
            event["resident_before"], max(event["target"], len(event["forced_positions"])))
        assert event["resident_after"] == cache_tensor_pairs(llm.past_key_values)[0][0].shape[-2]
    assert any(event.get("pruned") for event in llm.intent_events)
    llm.generate_first("new question", 4)
    assert llm.query_memory.updates == 1
    assert len(llm.intent_events) == 1


def test_capture_uses_actual_post_rope_q_and_removes_hooks(factory):
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
    llm = factory()
    expected, hooks = {}, []
    for i, layer in enumerate(llm.model.model.layers):
        def capture(attention, args, kwargs, i=i):
            if i in expected:
                return
            hidden = kwargs["hidden_states"]
            q = torch.nn.functional.linear(
                hidden, attention.q_proj.weight, attention.q_proj.bias
            ).view(1, hidden.shape[1], 4, 8).transpose(1, 2)
            q, _ = apply_rotary_pos_emb(q, q, *kwargs["position_embeddings"])
            expected[i] = q[0, :, 4:12].float().mean(dim=1)
        hooks.append(layer.self_attn.register_forward_pre_hook(capture, with_kwargs=True))
    try:
        llm.generate_first("question", 4)
    finally:
        for hook in hooks:
            hook.remove()
    for i, row in enumerate(llm.query_memory.rows):
        torch.testing.assert_close(row, torch.nn.functional.normalize(expected[i], dim=-1))
    for layer in llm.model.model.layers:
        assert not layer.self_attn._forward_pre_hooks
        assert not layer.self_attn.q_proj._forward_hooks


def test_monitor_records_actual_rule_compaction_and_stop_crop_parity(factory):
    from scripts.qa32_runtime import attach_monitor, compact_metrics
    llm = factory()
    with attach_monitor(llm) as monitor:
        monitor._patch(llm, "_compress_completed_block",
                       monitor._method_wrapper("intentkv_rule.completed_block", "pruning_calls", prune=True))
        llm.generate_first("question", 12)
        llm.generate_incremental("tool", 12)
        snapshot = monitor.snapshot()
        metrics = compact_metrics(snapshot)
        events = [event for event in snapshot["pruning_events"] if event["actual_tokens_removed"] > 0]
        assert len(events) == 1
        assert events[0]["before"]["tokens"] == llm.intent_events[-1]["resident_before"]
        assert events[0]["after"]["tokens"] == llm.intent_events[-1]["resident_after"]
        assert metrics["final"]["tokens"] == llm.intent_events[-1]["resident_after"]
    original, adapted = factory(False), factory()
    adapted._compute_target_budget = lambda: adapted.current_cache_len
    text = original.generate_first("question", 8)[0]
    stop = [text[3:4]]
    assert original.generate_first("question", 8, stop)[0] == adapted.generate_first("question", 8, stop)[0]
    assert_same_tensors(original, adapted)


def test_no_gpu_dry_run_and_seed_guard(capsys):
    common = ["--dataset", "musique", "--ratio", ".2", "--model-ready", "unused",
              "--data-ready", "unused", "--output-root", "unused"]
    assert worker.main(common) == 0
    assert "planned_not_executed" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        worker.parser().parse_args(common + ["--seed", "42"])


def test_queue_lease_verification_precedes_execution(monkeypatch):
    order = []
    monkeypatch.setattr(worker, "plan_inputs", lambda args: {"prepared": "cpu"})
    def registered(args, protocol=None):
        assert protocol == {"prepared": "cpu"}
        order.append(("registered", args.gpu))
    monkeypatch.setitem(sys.modules, "scripts.intentkv_rule_qa_queue",
                        SimpleNamespace(verify_registered=registered))
    monkeypatch.setattr(worker, "run", lambda args: order.append(("run", args.gpu)))
    worker.main(["--dataset", "hotpotqa", "--model-ready", "unused", "--data-ready", "unused",
                 "--output-root", "unused", "--gpu", "1", "--execute"])
    assert order == [("registered", 1), ("run", 1)]


def test_twenty_percent_smoke_also_requires_global_barrier_before_admission(monkeypatch):
    order = []
    monkeypatch.setattr(worker, "require_all_50", lambda root: order.append("barrier"))
    monkeypatch.setattr(worker, "plan_inputs", lambda args: {"prepared": "cpu"})
    monkeypatch.setitem(sys.modules, "scripts.intentkv_rule_qa_queue",
                        SimpleNamespace(verify_registered=lambda args, protocol: order.append("admission")))
    monkeypatch.setattr(worker, "run", lambda args: order.append("run"))
    worker.main(["--dataset", "hotpotqa", "--ratio", ".2", "--purpose", "smoke",
                 "--model-ready", "unused", "--data-ready", "unused",
                 "--output-root", "unused", "--gpu", "0", "--execute"])
    assert order == ["barrier", "admission", "run"]


@pytest.mark.parametrize("extra", [
    ["--samples", "499"], ["--sample-start", "1"], ["--max-steps", "8"],
    ["--purpose", "smoke", "--samples", "500"],
])
def test_explicit_protocol_flags_cannot_weaken_matching(extra):
    args = worker.parser().parse_args(extra)
    with pytest.raises(ValueError):
        worker.validate_protocol(args)


def test_cpu_full_protocol_matches_original_defaults(monkeypatch):
    import run_all_wiki_experiments_v2 as base
    from scripts.qa32_runtime import legacy_config
    pins = {"source_hashes": worker.source_hashes(), "sample_ids": ["s1", "s2"],
            "sample_indices": [12, 18], "selected_content_sha256": "fixture", "generation_config": {}}
    monkeypatch.setattr(worker, "plan_inputs", lambda args: pins)
    assets = {"model_ready_sha256": "fixture"}
    monkeypatch.setattr(worker, "cpu_function", lambda *a: lambda *b: (None, None, None, assets))
    args = worker.parser().parse_args([
        "--dataset", "hotpotqa", "--purpose", "smoke", "--samples", "2", "--sample-start", "500",
        "--model-ready", "unused", "--data-ready", "unused"])
    fingerprint = worker.build_protocol(args)
    assert fingerprint["kv_config"] == {**legacy_config(base, "tova", .5), **OVERRIDES, "cache_ratio": .5}
    assert fingerprint["assets"] == assets
    assert fingerprint["protocol_inputs"] == pins
    assert fingerprint["versions"]["torch"] == torch.__version__.split("+")[0]


def test_global_gate_requires_all_three_500_question_50_conditions(monkeypatch):
    hashes = {"source": "hash"}
    monkeypatch.setattr(worker, "source_hashes", lambda: hashes)
    seen = []

    def complete(path):
        dataset = next(name for name in worker.DATASETS if name in str(path))
        seen.append(dataset)
        return {"runtime": "intentkv_rule_frozen_qa_v1", "dataset": dataset,
                "method": "intentkv_rule", "ratio": .5, "purpose": "main", "samples": 500,
                "sample_start": 0, "seed": 233, "max_steps": 7,
                "model": worker.MODEL[0], "model_revision": worker.MODEL[1],
                "source_hashes": hashes}
    monkeypatch.setattr(worker, "validate_completed_run", complete)
    assert set(worker.require_all_50(Path("unused"))) == set(worker.DATASETS)
    assert seen == list(worker.DATASETS)

    def incomplete(path):
        value = complete(path)
        if value["dataset"] == "musique":
            value["samples"] = 499
        return value
    monkeypatch.setattr(worker, "validate_completed_run", incomplete)
    with pytest.raises(RuntimeError, match="prerequisite"):
        worker.require_all_50(Path("unused"))


@pytest.fixture
def local_output():
    path = Path(".intentkv-test-" + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        shutil.rmtree(path)


def test_durable_checkpoint_audit_resume_and_corruption_fail_closed(local_output):
    sample = {"id": "stable-id", "question": "Q?", "answer": "A"}
    fingerprint = {"samples": 1, "sample_ids": ["stable-id"]}
    store = ResultStore(local_output, fingerprint, [(0, sample)])
    store.set_placement({"layer_devices": ["cpu"]})
    from scripts.qa32_runtime import QUANTITIES
    quantities = {key: 0 for key in QUANTITIES}
    metrics = {key: quantities for key in ("boundary_means", "peak_active", "peak_trajectory", "peak_preprune")}
    metrics["allocator"] = {"after_episode": {}}
    row = {"index": 0, "id": "stable-id", "question": "Q?", "gold_answer": "A",
           "predicted_answer": "A", "em": True, "f1": 1., "kv_metrics": metrics}
    store.commit(row, {"intentkv_rule": {"method": "intentkv_rule"}})
    store.finalize()
    atomic_json(local_output / "status.json", {"status": "complete", "completed_samples": 1})
    assert worker.validate_completed_run(local_output) == fingerprint
    assert worker.main(["--validate-output", str(local_output)]) == 0
    resumed = ResultStore(local_output, fingerprint, [(0, sample)])
    assert len(resumed.rows) == 1
    reference = resumed.rows[0]["kv_audit"]["path"]
    (local_output / reference).write_text("{}")
    with pytest.raises(RuntimeError, match="corrupt"):
        worker.validate_completed_run(local_output)
    assert worker.main(["--validate-output", str(local_output)]) == 1


@pytest.mark.parametrize("ratio", [.5, .2])
def test_cpu_prepare_protocol_cli_writes_bound_inputs(local_output, monkeypatch, ratio):
    fingerprint = {"runtime": "fixture", "sample_ids": ["s1", "s2"]}
    monkeypatch.setattr(worker, "plan_inputs", lambda args: fingerprint)
    monkeypatch.setattr(worker, "require_all_50", lambda root: pytest.fail("CPU preparation demanded completion"))
    target = local_output / "prepared.json"
    assert worker.main([
        "--dataset", "hotpotqa", "--purpose", "smoke", "--gpu", "0",
        "--ratio", str(ratio),
        "--model-ready", "unused", "--data-ready", "unused", "--output-root", str(local_output),
        "--prepare-protocol", str(target)]) == 0
    assert read_json(target) == fingerprint


def test_admission_protocol_race_fails_before_execution_dependencies(monkeypatch):
    args = worker.parser().parse_args(["--dataset", "hotpotqa"])
    args._admitted_protocol = {"generation": "before"}
    monkeypatch.setattr(worker, "build_protocol",
                        lambda args: {"protocol_inputs": {"generation": "changed"}})
    with pytest.raises(RuntimeError, match="changed after queue admission"):
        worker.run(args)


def test_incomplete_global_barrier_precedes_assets_and_cuda(local_output, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("Reached CUDA before global barrier"))
    args = worker.parser().parse_args([
        "--dataset", "hotpotqa", "--ratio", ".2", "--model-ready", "unused",
        "--data-ready", "unused", "--output-root", str(local_output), "--execute"])
    with pytest.raises(FileNotFoundError):
        worker.run(args)
    assert list(local_output.iterdir()) == []


def test_frozen_core_hashes_still_match():
    assert worker.source_hashes()
