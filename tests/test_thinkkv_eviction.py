from types import SimpleNamespace

import torch
from transformers import DynamicCache
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

from scripts.thinkkv_eviction_runtime import ThinKVEvictionOnlyLLM
from token_tracker import TokenTracker


def test_segment_eviction_keeps_native_kv_and_prompt_exact():
    torch.manual_seed(3)
    config = Qwen2Config(num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=4, hidden_size=512)
    cache = DynamicCache(config=config)
    original = []
    for index in range(2):
        keys = torch.randn(1, 4, 16 + 256, 128, dtype=torch.bfloat16)
        values = torch.randn_like(keys)
        cache.update(keys, values, index)
        original.append((keys, values))
    llm = object.__new__(ThinKVEvictionOnlyLLM)
    llm.model = SimpleNamespace(config=config)
    llm.device = torch.device("cpu")
    llm.kv_manager = None
    llm.token_tracker = TokenTracker()
    llm.token_tracker.set_initial_cache_length(16)
    llm.token_tracker.append_new_tokens(256)
    llm._global_token_id_log = list(range(272))
    llm._all_token_ids = list(range(272))
    llm.prompt_length = 16
    llm.current_cache_len = 272
    llm.ratio = .2
    llm.past_key_values = cache
    llm.peak_cache_tokens = 0
    llm.peak_cache_bytes = 0
    llm.eviction_seconds = 0.
    llm.eviction_events = []

    llm._enforce_budget()
    mapper = llm.token_tracker.global_id_mapper
    assert mapper[:16] == list(range(16))
    assert mapper == sorted(set(mapper))
    assert len([gid for gid in mapper if 16 <= gid < 144]) == 4
    assert [gid for gid in mapper if gid >= 144] == list(range(144, 272))
    assert llm.current_cache_len == len(mapper)
    for layer, (keys, values) in zip(llm.past_key_values.layers, original):
        assert layer.keys.dtype == keys.dtype
        assert layer.values.dtype == values.dtype
        assert torch.equal(layer.keys, keys[:, :, mapper, :])
        assert torch.equal(layer.values, values[:, :, mapper, :])
    audit = llm.thin_audit()
    assert audit["quantization"] == "none"
    assert audit["resident_bytes"] == 2 * 2 * 4 * len(mapper) * 128 * 2
    assert audit["evictions"] and audit["eviction_seconds"] >= 0
