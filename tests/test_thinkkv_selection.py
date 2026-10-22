import copy
from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM

from scripts.thinkkv_selection import (
    SelectionState, ThinkKVSelectionLLM, cluster_representatives,
)


def cache_with_tokens(count):
    torch.manual_seed(233)
    cache = DynamicCache()
    for layer in range(2):
        keys = torch.randn(1, 2, count, 8, dtype=torch.bfloat16)
        values = torch.arange(count, dtype=torch.bfloat16).reshape(1, 1, count, 1).expand_as(keys).clone()
        cache.update(keys, values, layer)
    return cache


def initialized(count=258):
    cache = cache_with_tokens(count)
    state = SelectionState()
    state.append(cache, 0, count)
    state.register_prompt(2)
    return cache, state


def test_identical_keys_still_select_distinct_representative_pairs():
    keys = torch.ones(4, 128, 8, dtype=torch.bfloat16)
    indices = cluster_representatives(keys, 64)
    assert indices.shape == (4, 64)
    assert all(len(set(row.tolist())) == 64 for row in indices)
    torch.testing.assert_close(indices, cluster_representatives(keys, 64), rtol=0, atol=0)


@pytest.mark.parametrize("ratio,expected", [(.2, [4, 32]), (.5, [4, 64])])
def test_annealing_budget_native_heads_and_actual_bf16_pairs(ratio, expected):
    cache, state = initialized()
    originals = [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]
    indices, lengths, event = state.plan(cache, ratio, 258)
    assert lengths == expected
    assert event["after_tokens"] <= event["target_tokens"]
    assert event["budget_excess_tokens"] == 0
    assert event["annealing"][0] == {"segment": 0, "from": 128, "to": 64}
    state.materialize(cache, indices, lengths)
    assert state.ids.shape == (2, 2, 2 + sum(expected))
    for i, layer in enumerate(cache.layers):
        assert layer.keys.dtype == layer.values.dtype == torch.bfloat16
        assert layer.keys.shape[1] == 2
        slots = indices[i].unsqueeze(0).unsqueeze(-1).expand_as(layer.keys)
        torch.testing.assert_close(layer.keys, originals[i][0].gather(2, slots), rtol=0, atol=0)
        torch.testing.assert_close(layer.values, originals[i][1].gather(2, slots), rtol=0, atol=0)
        torch.testing.assert_close(layer.keys[:, :, :2], originals[i][0][:, :, :2], rtol=0, atol=0)
    assert bool((state.ids[..., 1:] > state.ids[..., :-1]).all())


def test_completed_compressed_segments_never_reinflate_on_registration_or_append():
    cache, state = initialized()
    indices, lengths, _ = state.plan(cache, .2, 258)
    state.materialize(cache, indices, lengths)
    state.register_prompt(2)
    assert state.counts == [4, 32]
    for layer in cache.layers:
        layer.keys = torch.cat((layer.keys, layer.keys[:, :, -1:]), 2)
        layer.values = torch.cat((layer.values, layer.values[:, :, -1:]), 2)
    state.append(cache, 258, 1)
    state.register_prompt(2)
    assert state.counts == [4, 32, 1]
    indices, lengths, event = state.plan(cache, .2, 259)
    assert event["after_tokens"] <= event["target_tokens"]
    assert lengths[-1] == 1


def test_active_or_pending_stop_segment_floor_is_explicit_not_a_fake_ratio():
    cache, state = initialized(130)
    indices, lengths, event = state.plan(cache, .2, 129)
    assert lengths == [128]
    assert event["floor_tokens"] == event["after_tokens"] == 130
    assert event["budget_excess_tokens"] == 103
    assert event["annealing"] == []
    assert indices.shape[-1] == 130


def test_invalid_keys_ratios_and_logical_append_fail_explicitly():
    with pytest.raises(ValueError, match="nonfinite"):
        cluster_representatives(torch.full((1, 128, 4), float("nan")), 4)
    cache, state = initialized()
    with pytest.raises(ValueError):
        state.plan(cache, float("nan"), 258)
    with pytest.raises(RuntimeError, match="contiguously"):
        state.append(cache, 259, 1)


class Tokenizer:
    letters = {2: "X", 3: "\n", **dict(zip(range(4, 15), "Observation"))}

    def decode(self, ids, skip_special_tokens=True):
        return "".join(self.letters.get(int(i), "?") for i in ids)

    def __call__(self, text, **kwargs):
        reverse = {value: key for key, value in self.letters.items()}
        return SimpleNamespace(input_ids=[reverse[c] for c in text])


def runtime(monkeypatch, enabled=True, ratio=.2):
    from models.QwenLLMWithKVCache import QwenLLMWithKVCache
    from token_tracker import TokenTracker
    torch.manual_seed(233)
    config = Qwen2Config(vocab_size=40, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=4096)
    config._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(config).eval()
    model.generation_config.eos_token_id = 1
    model.generation_config.repetition_penalty = 1.0

    def fake_init(self, path, kv_config=None, token_tracker=None):
        self.model = copy.deepcopy(model)
        self.device = torch.device("cpu")
        self.tokenizer = Tokenizer()
        self.kv_config = kv_config
        self.kv_manager = None
        self.token_tracker = token_tracker or TokenTracker()
    monkeypatch.setattr(QwenLLMWithKVCache, "__init__", fake_init)
    return ThinkKVSelectionLLM("unused", enabled=enabled, ratio=ratio)


def test_disabled_selection_matches_unmodified_control_logits_and_kv(monkeypatch):
    from models.RebuttalLLM import RebuttalLLM
    observed = runtime(monkeypatch, enabled=False)
    control = RebuttalLLM("unused", {"pruning_mode": "none", "pruning_schedule": "token",
                                    "attn_mode": "scoring_forward"})
    with torch.inference_mode():
        for tokens in ([2, 4, 5], [2], [6, 7, 8], [2]):
            expected = control._append(tokens)
            actual = observed._append(tokens)
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
            for (k, v), (other_k, other_v) in zip(control._layers(), observed._layers()):
                torch.testing.assert_close(k, other_k, rtol=0, atol=0)
                torch.testing.assert_close(v, other_v, rtol=0, atol=0)


def test_tool_prefill_after_pruning_uses_logical_not_compressed_rope_positions(monkeypatch):
    llm = runtime(monkeypatch)
    with torch.inference_mode():
        llm._append([2, 4, 5, 6], "prefill_time")
        llm.prompt_length = 4
        llm._append([2] * 256, "prefill_time")
        llm._maybe_prune("observation_prefill")
        assert llm.current_cache_len < llm.token_tracker.next_global_id
        before = llm.current_cache_len
        llm._append([7, 8, 9], "prefill_time")
        assert llm.last_forward_positions == [260, 261, 262]
        assert llm.current_cache_len == before + 3
        llm._maybe_prune("observation_prefill")
        assert llm.selection.counts[-1] == 3
        assert llm.selection.ids.shape[-1] == llm.current_cache_len


def test_stop_marker_crossing128_boundary_rewinds_all_heads_without_stale_ids(monkeypatch):
    llm = runtime(monkeypatch)
    with torch.inference_mode():
        llm._append([2, 4, 5, 6], "prefill_time")
        llm.prompt_length = 4
        tokens = [2] * 255 + list(range(3, 15))
        remaining = iter(tokens[1:])
        original = llm._append

        def logits(token):
            scores = torch.full((1, 40), -1000.)
            scores[0, token] = 1000
            return scores

        def scripted(ids, timing_key="decode_time"):
            original(ids, timing_key)
            return logits(next(remaining, 1))
        monkeypatch.setattr(llm, "_append", scripted)
        result = llm._generate_block(logits(tokens[0]), 280, ["\nObservation"])
        assert result == "X" * 255
        assert llm.token_tracker.next_global_id == 259
        assert llm.selection.next_id == 259
        assert llm.selection.counts == [4, 127]
        assert bool((llm.selection.ids < 259).all())
        assert llm.current_cache_len == 4 + 4 + 127


def test_monitor_clustering_gather_and_forward_times_are_disjoint(monkeypatch):
    from scripts.run_thinkkv_latency import SelectionMonitor
    from scripts import thinkkv_selection
    llm = runtime(monkeypatch)
    with torch.inference_mode(), SelectionMonitor(llm) as monitor:
        monitor.wrap(llm, "_maybe_prune", "selection_and_cache_management")
        monitor.wrap(thinkkv_selection, "cluster_representatives", "token_clustering")
        monitor.wrap(thinkkv_selection.SelectionState, "materialize", "cache_gather")
        llm._append([2, 4, 5, 6], "prefill_time")
        llm.prompt_length = 4
        llm._append([2] * 256, "prefill_time")
        llm._maybe_prune("observation_prefill")
        stats = monitor.snapshot()
    exclusive = stats["timing"]["exclusive_seconds"]
    assert exclusive["token_clustering"] > 0
    assert exclusive["cache_gather"] > 0
    assert exclusive["scoring_forward"] == 0
    assert stats["final"]["kv"]["head_expansion"]["uniform_factor"] == 1
    assert sum(exclusive.values()) == pytest.approx(stats["timing"]["root_seconds"])


def test_pinned_repetition_penalty_uses_original_first_and_incremental_histories(monkeypatch):
    from transformers import RepetitionPenaltyLogitsProcessor
    llm = runtime(monkeypatch)
    llm.model.generation_config.repetition_penalty = 1.05
    with torch.inference_mode():
        llm._append([2, 4, 5, 6], "prefill_time")
        llm.prompt_length = 4
        llm._generation_start = 4
        llm._initial_decode = True
        scores = torch.linspace(-2, 2, 40).unsqueeze(0)
        expected = RepetitionPenaltyLogitsProcessor(1.05)(torch.tensor([[2, 4, 5, 6]]), scores.clone())
        torch.testing.assert_close(llm._repetition_logits(scores), expected, rtol=0, atol=0)
        llm._initial_decode = False
        torch.testing.assert_close(llm._repetition_logits(scores), scores, rtol=0, atol=0)
        llm._append([7, 8])
        expected = RepetitionPenaltyLogitsProcessor(1.05)(torch.tensor([[7, 8]]), scores.clone())
        torch.testing.assert_close(llm._repetition_logits(scores), expected, rtol=0, atol=0)


def test_clean_runtime_adds_no_forward_phase_cuda_synchronization(monkeypatch):
    llm = runtime(monkeypatch)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: pytest.fail("Unexpected clean phase sync"))
    llm._sync()
