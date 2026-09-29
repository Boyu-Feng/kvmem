"""CPU-only real-Qwen2 and official-selector equivalence checks."""
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from models.LazyEvictionLLM import (LazyEvictionLLM, RecurrenceState,
    update_recurrence, retention_indices, gather_state, official_importance)


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny_runtime(ratio=0.2, window=3):
    torch.manual_seed(9)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, eos_token_id=63)
    config._attn_implementation = "sdpa"
    llm = LazyEvictionLLM.__new__(LazyEvictionLLM)
    llm.model = Qwen2ForCausalLM(config).eval()
    llm.device = torch.device("cpu")
    llm.tokenizer = SimpleNamespace(eos_token_id=63,
        decode=lambda ids, **kwargs: "".join(chr(65 + i) for i in ids))
    llm.kv_config = dict(cache_ratio=ratio, protect_prompt=True,
        lazy_window=window, lazy_alpha=.15)
    llm._configure()
    llm.reset()
    return llm


def prefill(llm, ids):
    llm.prompt_length = len(ids)
    llm._prefilling_prompt = True
    try:
        return llm._append(ids, "prefill_time")
    finally:
        llm._prefilling_prompt = False


def load_official(monkeypatch):
    path = Path(__file__).resolve().parents[1] / "vendor/lazyeviction_reference"
    module = ModuleType("model.temp_cacheobs")
    exec(compile((path / "temp_cacheobs.py").read_text(), str(path / "temp_cacheobs.py"), "exec"), module.__dict__)
    monkeypatch.setitem(sys.modules, "model", ModuleType("model"))
    monkeypatch.setitem(sys.modules, "model.temp_cacheobs", module)
    selector = ModuleType("official_lazy_kv")
    exec(compile((path / "kv_utils.py").read_text(), str(path / "kv_utils.py"), "exec"), selector.__dict__)
    return selector.Window_LAZYKVCluster, module.TempCache


def test_selector_matches_official_every_query_and_eviction(monkeypatch):
    Cluster, TempCache = load_official(monkeypatch)
    TempCache.reset()
    TempCache.alpha = .12
    window, capacity = 3, 7
    selector = Cluster(obs_size=window, max_kv_capacity=capacity)
    gen = torch.Generator().manual_seed(123)
    keys = torch.randn(1, 4, 5, 8, generator=gen)
    values = torch.randn(1, 4, 5, 8, generator=gen)
    ids = torch.arange(5).view(1, 1, -1).expand(1, 4, -1)
    state = RecurrenceState(ids, torch.zeros_like(ids), torch.zeros_like(ids))
    varied_heads = False
    for clock in range(1, 19):
        new_keys = torch.randn(1, 4, 1, 8, generator=gen)
        new_values = torch.randn(1, 4, 1, 8, generator=gen)
        query = torch.randn(1, 4, 1, 8, generator=gen)
        keys = torch.cat((keys, new_keys), dim=2)
        values = torch.cat((values, new_values), dim=2)
        expected_k, expected_v, weights = selector.update_kv_in_decoding(
            keys, query, values, None, 1, window, 0, obs_count=clock)
        new_ids = torch.full((1, 4, 1), clock + 4)
        state = update_recurrence(state, weights, clock, new_ids, TempCache.alpha)
        indices = retention_indices(state, clock, capacity, window)
        if indices is not None:
            state = gather_state(state, indices)
            keys = keys.gather(2, indices[..., None].expand(-1, -1, -1, 8))
            values = values.gather(2, indices[..., None].expand(-1, -1, -1, 8))
            varied_heads |= bool((state.ids[:, 0] != state.ids[:, 1]).any())
        torch.testing.assert_close(keys, expected_k, atol=0, rtol=0)
        torch.testing.assert_close(values, expected_v, atol=0, rtol=0)
        torch.testing.assert_close(state.last.float(), TempCache.layer_last_recurrent[0], atol=0, rtol=0)
        torch.testing.assert_close(state.mri.float(), TempCache.layer_max_period[0], atol=0, rtol=0)
    assert varied_heads


def test_incremental_logits_match_native_full_sequence_without_eviction():
    llm = tiny_runtime(ratio=1)
    native = Qwen2ForCausalLM(llm.model.config).eval()
    native.load_state_dict(llm.model.state_dict())
    prefill(llm, [1, 2, 3])
    actual = llm._append([4, 5, 6, 7])
    with torch.inference_mode():
        expected = native(torch.tensor([[1, 2, 3, 4, 5, 6, 7]]), use_cache=False).logits[:, -1]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert llm.current_cache_len == llm.token_tracker.next_global_id == 7
    assert llm.recurrence[0].ids.shape == (1, 4, 7)
    assert not hasattr(llm.token_tracker, "global_id_mapper")


@pytest.mark.parametrize("ratio", [.2, .5])
def test_multiturn_budget_prompt_protection_positions_and_real_attention(ratio):
    llm = tiny_runtime(ratio=ratio)
    prefill(llm, [1, 2, 3])
    llm.token_tracker.set_current_step(2)
    llm._append(list(range(4, 16)), "prefill_time")
    assert llm.last_forward_positions == [14]
    assert llm.current_cache_len == 3 + max(3, int(12 * ratio))
    previous_clock = llm.get_stats()["logical_query_clock"]
    llm.token_tracker.set_current_step(3)
    llm._append([16, 17, 18], "prefill_time")
    assert llm.get_stats()["logical_query_clock"] == previous_clock + 3
    assert llm.last_forward_positions == [17]
    assert [x["clock"] for x in llm.cache_measurements] == [3, 6, 9, 12, 15]
    for state in llm.recurrence.values():
        torch.testing.assert_close(state.ids[..., :3], torch.arange(3).expand(1, 4, -1))
        torch.testing.assert_close(state.ids[..., -3:], torch.arange(15, 18).expand(1, 4, -1))
        assert state.last.max() <= 15
        assert state.last.max() > 3
    assert llm.peak_cache_tokens > llm.cache_measurements[1]["after_tokens"]
    assert llm.get_stats()["head_expansion_factor"] == 2
    assert llm.get_stats()["cache_bytes"] == sum(k.numel() * 4 + v.numel() * 4 for k, v in llm._layers())


def test_rewind_restores_eviction_and_recurrence_exactly():
    llm = tiny_runtime()
    prefill(llm, [1, 2, 3])
    llm._append([4, 5, 6, 7, 8])
    snapshot = llm._snapshot()
    llm._pending_snapshots = {5: snapshot}
    before = llm.current_cache_len
    llm._append([9])  # clock 6 triggers an actual eviction
    assert llm.current_cache_len < before + 1
    llm._rewind_to(snapshot["logical_end"])
    assert llm.current_cache_len == before
    assert llm.get_stats()["logical_query_clock"] == 5
    assert len(llm.cache_measurements) == 1
    for layer, (keys, values) in enumerate(llm._layers()):
        torch.testing.assert_close(keys, snapshot["kv"][layer][0], atol=0, rtol=0)
        torch.testing.assert_close(values, snapshot["kv"][layer][1], atol=0, rtol=0)
        for field in ("ids", "last", "mri"):
            torch.testing.assert_close(getattr(llm.recurrence[layer], field),
                                       getattr(snapshot["recurrence"][layer], field), atol=0, rtol=0)
    llm._pending_snapshots = {}
    llm._append([10])
    assert llm.last_forward_positions == [8]
    assert llm.peak_stop_snapshot_extra_bytes > 0


def forced_generation(llm, initial, subsequent):
    logits = initial.clone().fill_(-100)
    logits[0, subsequent[0]] = 100
    original = llm._append
    future = iter(subsequent[1:])
    def append(ids, timing_key="decode_time"):
        out = original(ids, timing_key).clone().fill_(-100)
        out[0, next(future)] = 100
        return out
    llm._append = append
    return logits


def test_stop_marker_with_window_eviction_restores_all_per_head_state():
    llm = tiny_runtime(window=2)
    logits = prefill(llm, [1, 2])
    logits = forced_generation(llm, logits, [1, 2, 3, 4, 5])
    # B,C are content. D is appended at t=3, E at t=4 evicts, F completes DEF.
    assert llm._generate_block(logits, 9, ["DEF"]) == "BC"
    assert llm._global_token_id_log == [1, 2, 1, 2]
    assert llm.current_cache_len == 4
    assert llm.get_stats()["logical_query_clock"] == 2
    assert llm.window_checks == 1
    for state in llm.recurrence.values():
        assert int(state.last.max()) <= 2
        torch.testing.assert_close(state.ids, torch.arange(4).expand(1, 4, -1))


def test_stop_marker_shared_bpe_token_preserves_action_bracket():
    llm = tiny_runtime(window=2)
    class BracketTokenizer:
        eos_token_id = 63
        def decode(self, ids, **kwargs):
            return "".join({1: "B", 2: "]\n", 3: "Observation", 4: "]"}[i] for i in ids)
        def __call__(self, text, **kwargs):
            assert text == "]"
            return SimpleNamespace(input_ids=[4])
    llm.tokenizer = BracketTokenizer()
    logits = prefill(llm, [1, 1])
    logits = forced_generation(llm, logits, [1, 2, 3, 63])
    assert llm._generate_block(logits, 8, ["\nObservation"]) == "B]"
    assert llm._global_token_id_log == [1, 1, 1, 4]
    assert llm.current_cache_len == llm.token_tracker.next_global_id == 4
    assert llm.last_forward_positions == [3]
    assert llm.rewind_count == 1


def test_reset_clears_per_episode_state():
    llm = tiny_runtime()
    prefill(llm, [1, 2])
    llm._append([3, 4, 5, 6, 7, 8])
    llm.reset()
    prefill(llm, [9, 10])
    llm._append([11])
    assert llm.token_tracker.next_global_id == 3
    assert llm.executed_query_count == 1
    assert not llm.cache_measurements
    for state in llm.recurrence.values():
        assert state.ids.max() == 2
        assert state.mri.max() <= 1


def test_public_generation_api_and_snapshot_bytes():
    llm = tiny_runtime(window=3)
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages[-1]["content"] == "question"
            return "formatted prompt"
        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=[1, 2, 3] if text == "formatted prompt" else [4, 5, 6, 7])
        def decode(self, ids, **kwargs):
            return "".join(chr(65 + i) for i in ids)
    llm.tokenizer = Tokenizer()
    llm.model.generation_config.eos_token_id = None
    response, prompt, generated = llm.generate_first("question", max_new_tokens=2)
    assert len(response) == 2 and len(prompt) == len(generated) == 2
    assert prompt[0][0].shape[2] == 3
    assert generated[0][0].shape[2] == 2
    snapshot_bytes = sum(t.numel() * t.element_size() for pair in prompt + generated for t in pair)
    assert llm.get_stats()["returned_cache_snapshot_bytes"] == snapshot_bytes
    llm.token_tracker.set_current_step(2)
    llm.generate_incremental("observation", max_new_tokens=2)
    assert llm.token_tracker.next_global_id == 11
    assert llm.get_stats()["logical_query_clock"] == 8
    assert llm.get_stats()["total_prune_count"] == 1
    assert not llm.get_step_pruned(1)
    assert llm.get_step_pruned(2)
    assert llm.get_stats()["peak_cache_and_tracking_bytes"] >= snapshot_bytes + llm.get_stats()["cache_bytes"]


def test_long_clock_uses_exact_integer_recurrence_not_bfloat16():
    ids = torch.tensor([[[0, 1]]])
    state = RecurrenceState(ids, torch.tensor([[[8192, 8192]]]), torch.zeros_like(ids))
    out = update_recurrence(state, torch.tensor([[[[.1, .9, 0.]]]], dtype=torch.bfloat16),
                            8193, torch.tensor([[[2]]]), .5)
    assert out.last.tolist() == [[[8192, 8193, 8192]]]
    assert out.mri.tolist() == [[[0, 1, 0]]]
    assert torch.isfinite(official_importance(out, 8193)).all()
