import copy
import json
from types import SimpleNamespace

import pytest
import torch
from transformers import BatchEncoding, Qwen2Config, Qwen2ForCausalLM

from models.QwenLLMWithKVCache import QwenLLMWithKVCache
from scripts.thinkkv_selection import cluster_representatives
from scripts.unified_thinkkv_runtime import UnifiedThinkKVLLM
from token_tracker import TokenTracker


class Tokenizer:
    eos_token_id = 1
    chars = {2: "X", 3: "Y", 4: "\n", 5: "!", 6: "P", 7: "O", 8: "b", 9: "s",
             10: "XY"}

    def apply_chat_template(self, messages, **kwargs):
        return "PPP"

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.chars.get(int(token), "") for token in ids)

    def __call__(self, text, return_tensors=None, **kwargs):
        inverse = {value: token for token, value in self.chars.items()}
        ids = []
        while text:
            piece = "XY" if text.startswith("XY") else text[0]
            ids.append(inverse[piece])
            text = text[len(piece):]
        if return_tensors:
            return BatchEncoding({"input_ids": torch.tensor([ids]),
                                  "attention_mask": torch.ones(1, len(ids), dtype=torch.long)})
        return SimpleNamespace(input_ids=ids)


@pytest.fixture
def factory(monkeypatch):
    torch.manual_seed(233)
    config = Qwen2Config(vocab_size=16, hidden_size=56, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=28,
                         num_key_value_heads=4, max_position_embeddings=4096)
    config._attn_implementation = "sdpa"
    template = Qwen2ForCausalLM(config).to(dtype=torch.bfloat16).eval()
    template.generation_config.eos_token_id = None
    template.generation_config.pad_token_id = 0
    template.generation_config.repetition_penalty = 1.0

    def initialize(self, path, kv_config=None, token_tracker=None):
        self.model = copy.deepcopy(template)
        self.device = torch.device("cpu")
        self.tokenizer = Tokenizer()
        self.kv_config = kv_config or {}
        self.token_tracker = token_tracker
        self.kv_manager = None
        self.pruning_enabled = False
        self.attn_mode = "scoring_forward"
        self.step_kv = None
        self.needs_attn_scoring = False
        self.needs_new_step_kv = False
        QwenLLMWithKVCache.reset(self)

    monkeypatch.setattr(QwenLLMWithKVCache, "__init__", initialize)

    def create(enabled=True, ratio=.2, original=False):
        cls = QwenLLMWithKVCache if original else UnifiedThinkKVLLM
        kwargs = {} if original else {"enabled": enabled, "ratio": ratio}
        return cls("unused", token_tracker=TokenTracker(), **kwargs)
    return create


def script(llm, tokens):
    queue = None if tokens is None else iter(tokens)
    calls = []
    logits = []
    pairs = []

    def hook(model, args, kwargs, outputs):
        if queue is not None:
            token = next(queue, 1)
            outputs.logits.fill_(-1000)
            outputs.logits[..., token] = 1000
        calls.append({
            "positions": None if kwargs.get("position_ids") is None
            else kwargs["position_ids"].clone(),
            "cache_position": None if kwargs.get("cache_position") is None
            else kwargs["cache_position"].clone(),
            "mask": None if kwargs.get("attention_mask") is None
            else kwargs["attention_mask"].clone(),
        })
        logits.append(outputs.logits.clone())
        pairs.append([(layer.keys.clone(), layer.values.clone())
                      for layer in outputs.past_key_values.layers])
        return outputs

    llm.model.register_forward_hook(hook, with_kwargs=True)
    return calls, logits, pairs


def assert_caches_equal(first, second):
    for one, two in zip(first.past_key_values.layers, second.past_key_values.layers):
        torch.testing.assert_close(one.keys, two.keys, rtol=0, atol=0)
        torch.testing.assert_close(one.values, two.values, rtol=0, atol=0)


def test_disabled_exact_original_hf_multi_observation_and_posthoc_crop(factory):
    control, observed = factory(original=True), factory(enabled=False)
    expected = script(control, [2] * 5 + [4, 5] + [3] * 400)
    actual = script(observed, [2] * 5 + [4, 5] + [3] * 400)
    for llm in (control, observed):
        llm.generate_first("P", 20, ["\n!"])
    assert_caches_equal(control, observed)
    for observation in ("Obs", "XX", "O"):
        assert control.generate_incremental(observation, 12, ["\n!"]) == \
            observed.generate_incremental(observation, 12, ["\n!"])
        assert_caches_equal(control, observed)
        assert control.current_cache_len == observed.current_cache_len
        assert control._all_token_ids == observed._all_token_ids
        assert control.token_tracker.__dict__ == observed.token_tracker.__dict__
    for one, two in zip(expected[1], actual[1]):
        torch.testing.assert_close(one, two, rtol=0, atol=0)
    for one, two in zip(expected[0], actual[0]):
        for key in one:
            if one[key] is None:
                assert two[key] is None
            else:
                torch.testing.assert_close(one[key], two[key], rtol=0, atol=0)


def test_disabled_real_qwen_greedy_logits_cache_and_inherited_generation_config(factory):
    control, observed = factory(original=True), factory(enabled=False)
    # Use actual Qwen logits here, not the deterministic scheduling test hook.
    for llm in (control, observed):
        llm.model.generation_config.repetition_penalty = 1.17
    expected = script(control, None)
    actual = script(observed, None)
    config = copy.deepcopy(control.model.generation_config.to_dict())
    assert control.generate_first("P", 9)[0] == observed.generate_first("P", 9)[0]
    assert_caches_equal(control, observed)
    for text in ("Obs", "XY", "O"):
        assert control.generate_incremental(text, 7) == observed.generate_incremental(text, 7)
        assert_caches_equal(control, observed)
    for one, two in zip(expected[1], actual[1]):
        torch.testing.assert_close(one, two, rtol=0, atol=0)
    assert control.model.generation_config.to_dict() == config
    assert observed.audit()["generation_config"] == config


def test_disabled_preserves_optional_tracker_none(factory):
    observed = UnifiedThinkKVLLM("unused", enabled=False)
    assert observed.token_tracker is None
    script(observed, [2] * 20)
    observed.generate_first("P", 6)
    observed.generate_incremental("Obs", 4)
    assert observed.token_tracker is None
    json.dumps(observed.audit(), allow_nan=False)


@pytest.mark.parametrize("ratio", [.2, .5])
def test_parent_legacy_config_supplies_ratio_without_explicit_argument(factory, ratio):
    tracker = TokenTracker()
    config = {"pruning_mode": "none", "cache_ratio": ratio,
              "attn_mode": "scoring_forward", "protect_prompt": True}
    llm = UnifiedThinkKVLLM("unused", kv_config=config, token_tracker=tracker)
    assert llm.ratio == ratio and llm.token_tracker is tracker
    assert config["pruning_mode"] == "none" and config["cache_ratio"] == ratio
    script(llm, [2] * 200)
    llm.generate_first("P", 140)
    event = next(event for event in llm.events if event["annealing"])
    assert event["target_tokens"] == 3 + int(ratio * 128)
    assert llm.audit()["requested_ratio"] == ratio
    stats = llm.get_stats()
    assert stats["current_cache_len"] == llm.current_cache_len
    assert all(stats[name] >= 0 for name in
               ("prefill_time", "decode_time", "scoring_time", "pruning_time"))
    assert stats["scoring_time"] == 0


def test_constructor_explicit_ratio_wins_and_missing_ratio_defaults_to_point_two(factory):
    assert UnifiedThinkKVLLM("unused", kv_config={"cache_ratio": .5}, ratio=.2).ratio == .2
    assert UnifiedThinkKVLLM("unused").ratio == .2


@pytest.mark.parametrize("enabled", [False, True])
def test_bpe_boundary_posthoc_crop_matches_original_retokenized_length(factory, enabled):
    control, observed = factory(original=True), factory(enabled=enabled)
    # Two sampled tokens decode to "XY", whose tokenizer representation is ONE
    # token. Original QA intentionally crops by retokenized length, not a token
    # boundary search; the adapter must retain that same convention.
    for llm in (control, observed):
        script(llm, [2, 3, 4, 5] + [2] * 40)
    expected = control.generate_first("P", 12, ["\n!"])
    actual = observed.generate_first("P", 12, ["\n!"])
    assert expected[0] == actual[0] == "XY"
    assert control.current_cache_len == observed.current_cache_len == 4
    assert_caches_equal(control, observed)
    assert control.generate_incremental("Obs", 8, ["\n!"]) == \
        observed.generate_incremental("Obs", 8, ["\n!"])
    assert_caches_equal(control, observed)
    if enabled:
        assert observed.selection.ids.shape[-1] == observed.current_cache_len
        assert observed.get_global_token_id_log() == observed._logical_tokens


def test_enabled_postcrop_inside_block_after_old_per_head_pruning(factory):
    llm = factory()
    script(llm, [2] * 256 + [2, 3, 4, 5] + [3] * 40)
    llm.generate_first("P", 256)
    old_ids = llm.selection.ids.clone()
    result = llm.generate_incremental("Obs", 16, ["\n!"])
    assert result == "XY"
    assert llm.selection.next_id == 3 + 256 + 3 + 1
    assert bool((llm.selection.ids < llm.selection.next_id).all())
    assert llm.selection.ids.shape[-1] == llm.current_cache_len
    assert not llm.pending_tokens
    # Old independently-selected IDs survive without padding or deduplication.
    retained = llm.selection.ids[..., :7]
    torch.testing.assert_close(retained, old_ids[..., :7], rtol=0, atol=0)
    for layer in llm.past_key_values.layers:
        assert layer.keys.shape[1:3] == (4, llm.current_cache_len)
    assert llm.token_tracker.global_id_mapper == llm.selection.ids[0, 0].tolist()


@pytest.mark.parametrize("ratio", [.2, .5])
def test_bpe_stop_crop_at128_boundary_keeps_exact_actual_pairs_for_each_native_head(factory, ratio):
    control = factory(original=True)
    llm = UnifiedThinkKVLLM("unused", kv_config={"pruning_mode": "none", "cache_ratio": ratio})
    # Generated X/Y are separate tokens at logical offsets 127/128. Decode then
    # re-tokenize merges them, moving the original posthoc crop to offset 128.
    sequence = [2] * 128 + [3, 4, 5] + [2] * 300
    expected_calls, _, _ = script(control, sequence)
    actual_calls, _, _ = script(llm, sequence)
    expected = control.generate_first("P", 256, ["\n!"])
    observed = llm.generate_first("P", 256, ["\n!"])
    assert expected[0] == observed[0] == "X" * 128 + "Y"
    assert len(expected_calls) == len(actual_calls) == 256
    assert control.current_cache_len == llm.selection.next_id == 131
    assert llm.selection.ids.shape[:2] == (2, 4)
    assert llm.selection.counts == [16 if ratio == .2 else 64]
    assert not llm.pending_tokens
    assert bool((llm.selection.ids[..., 1:] > llm.selection.ids[..., :-1]).all())
    assert bool((llm.selection.ids < 131).all())
    for index, (source, actual) in enumerate(zip(control.past_key_values.layers,
                                                llm.past_key_values.layers)):
        slots = llm.selection.ids[index].unsqueeze(0).unsqueeze(-1).expand_as(actual.keys)
        torch.testing.assert_close(actual.keys, source.keys.gather(2, slots), rtol=0, atol=0)
        torch.testing.assert_close(actual.values, source.values.gather(2, slots), rtol=0, atol=0)
        assert actual.keys.dtype == actual.values.dtype == torch.bfloat16
        assert actual.keys.shape[1:3] == (4, llm.current_cache_len)
    audit = llm.audit()
    assert audit["stop_crop_floor"] == "entire current stop-bearing block (arbitrary BPE-safe)"
    assert audit["events"][-2]["stop_crop_lower_bound"] == 3
    assert audit["scoring_eos_probe"] is None and not audit["attention_saliency"]


def test_diagnostic_hooks_cover_selection_gather_rewind_without_attention_scoring(factory):
    from scripts.unified_latency_runtime import UnifiedMonitor
    llm = factory()
    script(llm, [2] * 128 + [4, 5] + [2] * 200)
    hooks = llm.DIAGNOSTIC_HOOKS
    assert all(callable(getattr(llm, name)) for name, _ in hooks)
    assert {phase for _, phase in hooks} == {
        "selection_and_cache_management", "cache_gather", "stop_crop"}
    with UnifiedMonitor(llm) as monitor:
        for name, phase in hooks:
            monitor.wrap(llm, name, phase)
        llm.generate_first("P", 256, ["\n!"])
        stats = monitor.snapshot()
    calls = stats["timing"]["calls"]
    assert calls["selection_and_cache_management"] > 0
    assert calls["cache_gather"] > 0
    assert calls["stop_crop"] > 0
    assert calls["scoring_forward"] == calls["token_saliency"] == 0


@pytest.mark.parametrize("enabled", [False, True])
def test_eos_and_immediate_eos_incremental_use_original_hf_semantics(factory, enabled):
    llm = factory(enabled=enabled)
    llm.model.generation_config.eos_token_id = 1
    calls, _, _ = script(llm, [2, 1, 1])
    assert llm.generate_first("P", 256)[0] == "X"
    assert len(calls) == 2
    before = len(calls)
    assert llm.generate_incremental("Obs", 256) == ""
    assert len(calls) == before + 1  # Observation prefill, no decode forward.


def test_native_head_bf16_actual_pair_selection_and_token_forward_budget(factory):
    llm = factory()
    calls, _, pairs = script(llm, [2] * 400)
    llm.generate_first("P", 256)
    state = llm.selection
    audit = llm.audit()
    assert len(calls) == 256  # HF cap, not a replacement decoder.
    events = [event for event in audit["events"] if event["annealing"]]
    assert events and events[0]["forward"] == 128
    assert events[0]["annealing"][0] == {"segment": 0, "from": 128, "to": 64}
    assert state.ids.shape[:2] == (2, 4)  # not expanded 28 query heads
    assert state.counts == [4, 127]
    assert len(llm.pending_tokens) == 1
    assert llm.current_cache_len == state.ids.shape[-1] == 134
    # Each selected pair is exactly an actual pre-pruning pair, never a centroid.
    event = events[0]
    original = pairs[event["forward"]]
    # That event anneals the oldest segment all the way to four representatives.
    kept = state.ids[..., :7]
    for index, layer in enumerate(llm.past_key_values.layers):
        slots = kept[index].unsqueeze(0).unsqueeze(-1).expand(1, 4, 7, 2)
        torch.testing.assert_close(layer.keys[..., :7, :],
                                   original[index][0].gather(2, slots), rtol=0, atol=0)
        torch.testing.assert_close(layer.values[..., :7, :],
                                   original[index][1].gather(2, slots), rtol=0, atol=0)
        assert layer.keys.dtype == layer.values.dtype == torch.bfloat16
    assert audit["actual_trajectory_ratio"] == pytest.approx(131 / 256)
    assert audit["events"][-1]["budget_excess_tokens"] > 0
    assert audit["events"][-1]["floor_tokens"] == 134
    json.dumps(audit, allow_nan=False)
    assert llm.live_caches() == [llm.past_key_values]


def test_late_stop_is_posthoc_full_forward_count_and_exact_crop_all_ids(factory):
    llm = factory()
    calls, _, _ = script(llm, [2] * 255 + [4, 5] + [3] * 100)
    text, _, _ = llm.generate_first("P", 280, ["\n!"])
    assert len(calls) == 280
    assert text == "X" * 255
    assert llm.selection.next_id == 258
    assert llm.selection.ids.shape == (2, 4, 134)
    assert llm.selection.counts == [4, 127]
    assert not llm.pending_tokens
    assert llm.events[-2]["stop_crop_lower_bound"] == 3
    assert llm.events[-2]["floor_tokens"] == 282
    assert llm.events[-1]["trigger"] == "posthoc_stop_floor_release"
    assert llm.events[-1]["floor_tokens"] == llm.audit()["floor_tokens"] == 134
    prior_events = len(llm.events)
    # A subsequent stop-bearing block can still prune older completed segments.
    llm.generate_incremental("Obs", 12, ["\n!"])
    assert any(event["annealing"] and event["trigger"] == "after_forward"
               for event in llm.events[prior_events:])
    assert all(int(layer.keys.shape[-2]) == llm.current_cache_len
               for layer in llm.past_key_values.layers)


def test_compaction_hf_masks_and_positions_keep_full_causal_prefix_multiobs(factory):
    llm = factory()
    script(llm, [2] * 600)
    llm.generate_first("P", 256)
    logical_end = llm.selection.next_id
    llm.generate_incremental("Obs" * 45, 20)
    llm.generate_incremental("YY", 10)
    for event in llm.forward_events:
        if event["attention_mask_length"] is not None:
            assert event["attention_mask_length"] == event["physical_start"] + event["input_tokens"]
        if event["positions"] is not None:
            assert event["positions"] == [list(range(
                event["physical_start"], event["physical_start"] + event["input_tokens"]))]
    observation = llm.forward_events[256]
    assert observation["logical_start"] == logical_end
    assert observation["input_tokens"] == 135
    assert observation["physical_start"] < observation["logical_start"]
    ids = llm.selection.ids
    assert bool((ids[..., 1:] > ids[..., :-1]).all())
    assert all(layer.keys.shape[-2] == ids.shape[-1]
               for layer in llm.past_key_values.layers)
    assert llm.token_tracker.next_global_id == llm.selection.next_id
    assert llm.token_tracker.global_id_mapper == ids[0, 0].tolist()
    assert llm.pending_tokens[0]["global_id"] == llm.selection.next_id - 1
    assert [token["global_id"] for token in llm.uncached_sampled_tokens] == [
        258, logical_end + 135 + 19, llm.selection.next_id - 1]
    for token in llm.uncached_sampled_tokens:
        assert llm.get_global_token_id_log()[token["global_id"]] == token["token_id"]
        assert not bool((ids == token["global_id"]).any())


def test_reset_restart_and_hooks_restored(factory):
    llm = factory()
    original_prepare = llm.model.prepare_inputs_for_generation
    script(llm, [2] * 400)
    llm.generate_first("P", 140)
    assert llm.model.prepare_inputs_for_generation == original_prepare
    llm.reset()
    assert llm.live_caches() == []
    assert llm.selection.ids is None and not llm.events and not llm.pending_tokens
    assert llm.token_tracker.next_global_id == 0
    llm.generate_first("P", 12)
    assert llm.selection.next_id == 15
    assert len(llm.forward_events) == 12


def test_degenerate_kmeans_is_deterministic_distinct_native_head_selection():
    keys = torch.ones(8, 128, 2, dtype=torch.bfloat16)
    for keep in (64, 32, 16, 8, 4):
        selected = cluster_representatives(keys, keep)
        assert all(len(set(row.tolist())) == keep for row in selected)
        torch.testing.assert_close(selected, cluster_representatives(keys, keep), rtol=0, atol=0)


@pytest.mark.parametrize("ratio", [0, -1, 1.1, float("nan")])
def test_invalid_ratio_rejected_without_loading(ratio):
    with pytest.raises(ValueError, match="ratio"):
        UnifiedThinkKVLLM("unused", ratio=ratio)
