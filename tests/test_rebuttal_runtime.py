"""Small real-transformer tests for cache integrity; no downloaded model needed."""
from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from kv_cache.kv_cache_manager import KVCacheManager
from models.RebuttalLLM import RebuttalLLM
from token_tracker import TokenTracker


def tiny_runtime(mode="none", schedule="step"):
    torch.manual_seed(9)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=256,
                         eos_token_id=63)
    config._attn_implementation = "sdpa"
    llm = RebuttalLLM.__new__(RebuttalLLM)
    llm.model = Qwen2ForCausalLM(config).eval()
    llm.device = torch.device("cpu")
    llm.tokenizer = SimpleNamespace(eos_token_id=63,
        decode=lambda ids, **kwargs: "".join(chr(65 + i) for i in ids))
    llm.kv_config = dict(pruning_mode=mode, cache_ratio=0.2,
                         observation_window=0, protect_prompt=True,
                         step_poolwise_prune=True, step_aware_alpha=.8,
                         step_aware_beta=.8, attn_mode="scoring_forward")
    llm.token_tracker = TokenTracker()
    llm.kv_manager = KVCacheManager(llm.kv_config, llm.token_tracker) if mode != "none" else None
    llm.schedule = schedule
    llm.reset()
    return llm


def test_incremental_logits_match_full_sequence():
    llm = tiny_runtime()
    llm._append([1, 2, 3])
    actual = llm._append([4, 5])
    with torch.inference_mode():
        expected = llm.model(torch.tensor([[1, 2, 3, 4, 5]]), use_cache=False).logits[:, -1]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert llm.current_cache_len == llm.token_tracker.next_global_id == 5


def test_attention_probe_does_not_mutate_live_cache():
    llm = tiny_runtime("h2o")
    llm._append([1, 2, 3, 4])
    before = [(k.clone(), v.clone()) for k, v in llm._layers()]
    attn = llm._get_attention_for_scoring()
    assert attn[-1].shape[-1] == 5
    assert llm.current_cache_len == llm.token_tracker.next_global_id == 4
    for (old_k, old_v), (new_k, new_v) in zip(before, llm._layers()):
        torch.testing.assert_close(old_k, new_k)
        torch.testing.assert_close(old_v, new_v)
    assert llm.model.config._attn_implementation == "sdpa"


@pytest.mark.parametrize("mode", ["h2o", "tova", "step_aware_h2o"])
def test_pruning_budget_and_absolute_positions(mode):
    llm = tiny_runtime(mode)
    llm._append([1, 2, 3])
    llm.prompt_length = 3
    llm.kv_manager.register_initial_cache(3)
    llm.token_tracker.set_current_step(2)
    llm._append(list(range(4, 24)))
    llm._maybe_prune("test_boundary")
    assert llm.current_cache_len == 7
    kept = set(llm.token_tracker.global_id_mapper)
    assert {0, 1, 2} <= kept
    assert llm.token_tracker.next_global_id == 23
    llm._append([24, 25, 26])
    assert llm.last_forward_positions == [23, 24, 25]
    assert set(llm.token_tracker.global_id_mapper) == kept | {23, 24, 25}
    assert llm.peak_cache_tokens == 23


@pytest.mark.parametrize("schedule,expected", [("step", ["completed_decode_block"]),
                                              ("token", ["decode_token"] * 3)])
def test_generation_schedules_and_materializes_last_token(schedule, expected):
    llm = tiny_runtime(schedule=schedule)
    logits = llm._append([1, 2])
    llm.model.generation_config.eos_token_id = None
    calls = []
    llm._maybe_prune = calls.append
    llm._generate_block(logits, 3, None)
    assert calls == expected
    assert llm.current_cache_len == llm.token_tracker.next_global_id == 5


def test_stop_marker_rewinds_previous_marker_tokens():
    llm = tiny_runtime()
    logits = llm._append([1, 2]).clone()
    # Force tokens B,C,D; stop marker CD includes one already cached token.
    original = llm._append
    logits.fill_(-100)
    logits[0, 1] = 100
    next_tokens = iter([2, 3])
    def append(ids):
        out = original(ids).clone()
        out.fill_(-100)
        out[0, next(next_tokens)] = 100
        return out
    llm._append = append
    assert llm._generate_block(logits, 8, ["CD"]) == "B"
    assert llm.token_tracker.global_id_mapper == [0, 1, 2]
    assert llm.current_cache_len == llm.token_tracker.next_global_id == 3


def test_stop_marker_preserves_action_bracket_in_shared_bpe_token():
    llm = tiny_runtime()
    class BracketTokenizer:
        eos_token_id = 63
        def decode(self, ids, **kwargs):
            pieces = {1: "B", 2: "]\n", 3: "Observation", 4: "]"}
            return "".join(pieces[i] for i in ids)
        def __call__(self, text, **kwargs):
            assert text == "]"
            return SimpleNamespace(input_ids=[4])
    llm.tokenizer = BracketTokenizer()
    logits = llm._append([1, 1]).clone()
    logits.fill_(-100)
    logits[0, 1] = 100
    original = llm._append
    next_tokens = iter([2, 3, 63])
    def append(ids):
        out = original(ids).clone()
        out.fill_(-100)
        out[0, next(next_tokens)] = 100
        return out
    llm._append = append
    assert llm._generate_block(logits, 8, ["\nObservation"]) == "B]"
    assert llm._global_token_id_log == [1, 1, 1, 4]
    assert llm.current_cache_len == llm.token_tracker.next_global_id == 4
