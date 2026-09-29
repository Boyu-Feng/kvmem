"""CPU regression tests for the observational legacy-efficiency sidecar."""
import copy
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
from transformers import Qwen2Config, Qwen2ForCausalLM

from models.QwenLLMWithKVCache import QwenLLMWithKVCache
from scripts.legacy_efficiency_metrics import (
    attach_legacy_efficiency,
    install_legacy_efficiency,
)
from token_tracker import TokenTracker


BYTES_PER_TOKEN = 3 * 2 * 8 * 2 * 4  # layers, KV heads, head dim, K/V, float32


class TinyTokenizer:
    eos_token_id = 63

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        return "prompt"

    def __call__(self, text, return_tensors=None, **kwargs):
        self.calls.append(text)
        ids = [ord(c) - 256 if 256 <= ord(c) < 320 else ord(c) % 61 + 1
               for c in text]
        if return_tensors:
            tensor = torch.tensor([ids])
            return BatchEncoding(dict(input_ids=tensor,
                                      attention_mask=torch.ones_like(tensor)))
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids, **kwargs):
        return "".join(chr(256 + int(i)) for i in ids)


@pytest.fixture
def model_factory(monkeypatch):
    torch.set_num_threads(1)
    torch.manual_seed(233)
    config = Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, eos_token_id=None, pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    template = Qwen2ForCausalLM(config).eval()
    template.generation_config.repetition_penalty = 1.05
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained",
                        lambda *a, **k: copy.deepcopy(template))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained",
                        lambda *a, **k: TinyTokenizer())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def forbidden_sync(*args, **kwargs):
        pytest.fail("A CPU-only model must never synchronize CUDA")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden_sync)
    return template


def make_llm(mode="none", ratio=.5):
    config = dict(pruning_mode=mode, cache_ratio=ratio,
                  observation_window=0, protect_prompt=True,
                  prune_every_n=1, attn_mode="scoring_forward",
                  prompt_prefill_keep_ratio=1.0,
                  step_poolwise_prune=True, step_aware_min_keep=1)
    return QwenLLMWithKVCache("tiny-test", config, TokenTracker())


def cache_pairs(llm):
    return [(layer.keys, layer.values) for layer in llm.past_key_values.layers]


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for left, right in zip(a, b):
            assert_nested_equal(left, right)
    else:
        assert a == b


def assert_legacy_state_equal(a, b):
    assert_nested_equal(cache_pairs(a), cache_pairs(b))
    assert a.current_cache_len == b.current_cache_len
    assert a.token_tracker.__dict__ == b.token_tracker.__dict__
    assert a._all_token_ids == b._all_token_ids
    assert a._global_token_id_log == b._global_token_id_log
    assert a.model.config._attn_implementation == b.model.config._attn_implementation
    if a.kv_manager is not None:
        for name in ("current_cache_len", "protected_prefix_len", "step_count"):
            assert getattr(a.kv_manager, name) == getattr(b.kv_manager, name)


def assert_sample(sample, tokens, protected=6):
    assert sample["tokens"] == tokens
    assert sample["layer_tokens"] == [tokens] * 3
    assert sample["total_bytes"] == tokens * BYTES_PER_TOKEN
    assert sample["total_mib"] == pytest.approx(tokens * BYTES_PER_TOKEN / 2**20)
    assert sample["protected_tokens"] == min(tokens, protected)
    assert sample["trajectory_tokens"] == max(0, tokens - protected)
    assert sample["protected_bytes"] == min(tokens, protected) * BYTES_PER_TOKEN
    assert sample["trajectory_bytes"] == max(0, tokens - protected) * BYTES_PER_TOKEN


@pytest.mark.parametrize("mode", ["none", "h2o", "tova", "step_aware_h2o"])
def test_monitor_preserves_legacy_outputs_kv_and_tracker(model_factory, mode):
    plain, observed = make_llm(mode), make_llm(mode)
    monitor = attach_legacy_efficiency(observed, synchronize=True)
    try:
        first_a = plain.generate_first("question", max_new_tokens=8)
        first_b = observed.generate_first("question", max_new_tokens=8)
        assert_nested_equal(first_a, first_b)
        assert_legacy_state_equal(plain, observed)
        for step, text in enumerate(("observation", "second observation"), start=2):
            plain.token_tracker.set_current_step(step)
            observed.token_tracker.set_current_step(step)
            assert plain.generate_incremental(text, max_new_tokens=6) == (
                observed.generate_incremental(text, max_new_tokens=6))
            assert_legacy_state_equal(plain, observed)
        snapshot = monitor.snapshot()
        assert snapshot["counts"]["active_model_forwards"] > 0
        if mode != "none":
            assert observed.token_tracker.total_discarded > 0
            assert snapshot["counts"]["pruning_calls"] > 0
            assert snapshot["pruning_events"]
        if mode == "step_aware_h2o":
            assert snapshot["counts"]["attention_score_calls"] > 0
    finally:
        monitor.close()


def test_fullkv_infers_prompt_and_counts_actual_kv_not_bookkeeping(model_factory):
    llm = make_llm()
    assert llm.kv_manager is None
    monitor = attach_legacy_efficiency(llm, synchronize=True)
    try:
        llm.generate_first("question", max_new_tokens=5)
        snapshot = monitor.snapshot()
        assert snapshot["prompt_length"] == 6
        assert snapshot["prompt_length_source"]
        assert llm.tokenizer.calls == ["prompt"]  # observation adds no tokenization
        assert llm.current_cache_len == 11  # the last generated token is not materialized
        assert_sample(snapshot["kv"]["current"], 10)
        assert_sample(snapshot["kv"]["peak_active"], 10)
        assert_sample(snapshot["kv"]["peak_post_step"], 10)
        assert snapshot["counts"]["active_model_forwards"] == 5
        assert snapshot["counts"]["eos_probe_forwards"] == 0
        assert snapshot["counts"]["active_input_positions"] == 10
        assert snapshot["compute_workload"]["initial_prefill"] == {
            "forward_calls": 1, "input_positions": 6,
            "sequence_length_sum": 6, "batch_size_sum": 1,
            "max_batch_size": 1, "max_sequence_length": 6,
        }
        assert snapshot["compute_workload"]["decode"] == {
            "forward_calls": 4, "input_positions": 4,
            "sequence_length_sum": 4, "batch_size_sum": 4,
            "max_batch_size": 1, "max_sequence_length": 1,
        }
        assert len(snapshot["steps"]) == 1
        assert snapshot["steps"][0]["method"] == "generate_first"
        assert snapshot["steps"][0]["elapsed_s"] >= 0
    finally:
        monitor.close()


def test_probe_kv_is_separate_from_active_peak_and_does_not_change_cache(model_factory):
    llm = make_llm()
    monitor = attach_legacy_efficiency(llm, synchronize=True)
    try:
        llm.generate_first("question", max_new_tokens=5)
        before = copy.deepcopy(cache_pairs(llm))
        tracker = copy.deepcopy(llm.token_tracker.__dict__)
        active_before = monitor.snapshot()["kv"]["peak_active"]["total_bytes"]
        attentions = llm._get_attention_for_scoring()
        assert attentions[-1].shape[-1] == 11
        assert_nested_equal(cache_pairs(llm), before)
        assert llm.token_tracker.__dict__ == tracker
        snapshot = monitor.snapshot()
        assert_sample(snapshot["kv"]["current"], 10)
        assert_sample(snapshot["kv"]["peak_active"], 10)
        assert snapshot["kv"]["peak_active"]["total_bytes"] == active_before
        assert_sample(snapshot["kv"]["peak_probe"], 11)
        assert snapshot["kv"]["peak_active_plus_probe_bytes"] == 21 * BYTES_PER_TOKEN
        assert snapshot["counts"]["model_forwards"] == 6
        assert snapshot["counts"]["active_model_forwards"] == 5
        assert snapshot["counts"]["eos_probe_forwards"] == 1
        assert snapshot["counts"]["eos_probe_calls"] == 1
        assert snapshot["counts"]["active_input_positions"] == 10
        assert snapshot["counts"]["probe_input_positions"] == 1
        assert snapshot["compute_workload"]["eos_probe"] == {
            "forward_calls": 1, "input_positions": 1,
            "sequence_length_sum": 1, "batch_size_sum": 1,
            "max_batch_size": 1, "max_sequence_length": 1,
        }
    finally:
        monitor.close()


def initialize_prunable_cache(llm):
    llm.reset()
    with torch.no_grad():
        output = llm.model(input_ids=torch.tensor([list(range(1, 21))]),
                           use_cache=True, return_dict=True)
    llm.past_key_values = output.past_key_values
    llm.current_cache_len = 20
    llm.kv_manager.register_initial_cache(6)
    llm.kv_manager.current_cache_len = 20
    llm.token_tracker.append_new_tokens(14)
    llm.token_tracker.set_current_step(2)
    llm._all_token_ids = list(range(1, 21))
    llm._global_token_id_log = list(range(1, 21))


def test_pruning_event_measures_resident_before_after_and_actual_removal(model_factory):
    plain, observed = make_llm("tova", .25), make_llm("tova", .25)
    initialize_prunable_cache(plain)
    initialize_prunable_cache(observed)
    monitor = attach_legacy_efficiency(observed, synchronize=True)
    try:
        plain._do_pruning()
        observed._do_pruning()
        assert_legacy_state_equal(plain, observed)
        snapshot = monitor.snapshot()
        assert snapshot["counts"]["pruning_calls"] == 1
        assert snapshot["counts"]["selector_calls"] > 0
        assert len(snapshot["pruning_events"]) == 1
        event = snapshot["pruning_events"][0]
        assert_sample(event["before"], 20)
        assert_sample(event["after"], 9)
        assert event["actual_tokens_removed"] == 11
        assert_sample(snapshot["kv"]["peak_preprune"], 20)
        assert_sample(snapshot["kv"]["current"], 9)
    finally:
        monitor.close()


def test_stop_crop_is_observed_without_changing_legacy_result(model_factory):
    preview = make_llm().generate_first("question", max_new_tokens=8)[0]
    stop = preview[3:5]
    assert len(stop) == 2
    plain, observed = make_llm(), make_llm()
    monitor = attach_legacy_efficiency(observed, synchronize=True)
    try:
        expected = plain.generate_first("question", max_new_tokens=8, stop_strings=[stop])
        actual = observed.generate_first("question", max_new_tokens=8, stop_strings=[stop])
        assert_nested_equal(expected, actual)
        assert_legacy_state_equal(plain, observed)
        snapshot = monitor.snapshot()
        assert snapshot["counts"]["crop_calls"] == 1
        resident = int(observed.past_key_values.layers[0].keys.shape[2])
        assert_sample(snapshot["kv"]["current"], resident)
        assert snapshot["kv"]["peak_active"]["tokens"] > resident
    finally:
        monitor.close()


def test_reset_clears_measurements_without_mutating_resident_cache(model_factory):
    llm = make_llm("tova", .25)
    initialize_prunable_cache(llm)
    monitor = attach_legacy_efficiency(llm, synchronize=True)
    try:
        llm._do_pruning()
        before_cache = copy.deepcopy(cache_pairs(llm))
        before_tracker = copy.deepcopy(llm.token_tracker.__dict__)
        before_tokens = list(llm._all_token_ids)
        before_log = list(llm._global_token_id_log)
        monitor.reset()
        snapshot = monitor.snapshot()
        assert_nested_equal(cache_pairs(llm), before_cache)
        assert llm.token_tracker.__dict__ == before_tracker
        assert llm._all_token_ids == before_tokens
        assert llm._global_token_id_log == before_log
        assert llm.current_cache_len == 9
        assert not any(snapshot["counts"].values())
        assert snapshot["pruning_events"] == []
        assert snapshot["steps"] == []
        assert_sample(snapshot["kv"]["current"], 9)
        assert_sample(snapshot["kv"]["peak_active"], 9)
        assert snapshot["kv"]["peak_probe"] is None
    finally:
        monitor.close()


def test_new_episode_does_not_inherit_the_previous_cache_peak(model_factory):
    llm = make_llm()
    llm.generate_first("unmonitored clean pass", max_new_tokens=12)
    assert int(llm.past_key_values.layers[0].keys.shape[2]) == 17
    monitor = attach_legacy_efficiency(llm, synchronize=True)
    try:
        llm.generate_first("first monitored question", max_new_tokens=5)
        assert_sample(monitor.snapshot()["kv"]["peak_active"], 10)
        llm.generate_first("second question", max_new_tokens=3)
        snapshot = monitor.snapshot()
        assert snapshot["prompt_length"] == 6
        assert_sample(snapshot["kv"]["peak_active"], 8)
        assert_sample(snapshot["kv"]["peak_post_step"], 8)
        assert snapshot["counts"]["model_forwards"] == 3
        assert len(snapshot["steps"]) == 1
    finally:
        monitor.close()


def instance_surface(llm):
    """Only wrappers/hooks should change; do not compare model runtime state."""
    objects = [llm, llm.model]
    if llm.kv_manager is not None:
        objects.extend([llm.kv_manager, llm.kv_manager.pruning_strategy,
                        llm.kv_manager.pruning_strategy.h2o_scorer])
    return [(obj, dict(vars(obj))) for obj in objects]


def assert_surface_restored(surface):
    for obj, old in surface:
        assert set(vars(obj)) == set(old)
        for name, value in old.items():
            assert vars(obj)[name] is value, name


def test_close_restores_instance_overrides_and_model_hooks(model_factory):
    llm = make_llm("tova")
    original_prune = llm._do_pruning

    def custom_prune(*args, **kwargs):
        return original_prune(*args, **kwargs)

    llm._do_pruning = custom_prune
    external_hook = llm.model.register_forward_hook(lambda *args: None)
    before_hooks = dict(llm.model._forward_hooks)
    before_pre_hooks = dict(llm.model._forward_pre_hooks)
    surface = instance_surface(llm)
    monitor = attach_legacy_efficiency(llm, synchronize=True)
    monitor.close()
    monitor.close()  # cleanup is safe twice
    assert_surface_restored(surface)
    assert llm._do_pruning is custom_prune
    assert llm.model._forward_hooks == before_hooks
    assert llm.model._forward_pre_hooks == before_pre_hooks
    external_hook.remove()


def test_install_collects_instances_and_restores_constructor_and_each_monitor(model_factory):
    original_init = QwenLLMWithKVCache.__init__
    handle = install_legacy_efficiency(QwenLLMWithKVCache, synchronize=True)
    try:
        first, second = make_llm(), make_llm("tova")
        assert len(handle.monitors) == 2
        first.generate_first("question", max_new_tokens=3)
        assert handle.monitors[0].snapshot()["counts"]["model_forwards"] == 3
    finally:
        handle.close()
    assert QwenLLMWithKVCache.__init__ is original_init
    for llm in (first, second):
        assert not llm.model._forward_hooks
        assert not llm.model._forward_pre_hooks
        assert "generate_first" not in vars(llm)
        assert "_do_pruning" not in vars(llm)
    make_llm()
    assert len(handle.monitors) == 2
