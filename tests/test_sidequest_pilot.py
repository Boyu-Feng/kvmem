"""Safety/correctness checks for cursor parsing and cache branches, without downloads."""
import pytest
import torch
from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM

from scripts.sidequest_runtime import evict_cache, fork_cache, parse_deletions


def config():
    return Qwen2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
                       num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1)


def make_cache():
    cfg = config()
    cache = DynamicCache(config=cfg)
    for i in range(2):
        keys = torch.arange(48, dtype=torch.float32).reshape(1, 1, 6, 8) + i
        cache.update(keys, keys + 100, i)
    return cfg, cache


def test_parse_commands():
    assert parse_deletions('```json\n{"del_cursors": [0, 2]}\n```', {0, 1, 2}) == [0, 2]
    assert parse_deletions('{"del_cursors": []}', {0}) == []


@pytest.mark.parametrize('text', ['{}', '{"del_cursors":[9]}', '{"del_cursors":[true]}',
                                  '{"del_cursors":[1,1]}', '{"del_cursors":"1"}', 'no JSON'])
def test_reject_invalid_commands(text):
    with pytest.raises(ValueError):
        parse_deletions(text, {0, 1})


def test_shared_prefix_is_not_mutated_by_branch_append():
    cfg, cache = make_cache()
    original = cache.layers[0].keys.clone()
    branch = fork_cache(cache, cfg)
    assert branch.layers[0].keys.data_ptr() == cache.layers[0].keys.data_ptr()
    branch.update(torch.zeros(1, 1, 1, 8), torch.zeros(1, 1, 1, 8), 0)
    assert branch.layers[0].keys.shape[2] == 7
    assert torch.equal(cache.layers[0].keys, original)


def test_deletion_preserves_prompt_other_cursors_and_branch():
    cfg, cache = make_cache()
    branch = fork_cache(cache, cfg)
    pruned, mapper = evict_cache(cache, list(range(6)), [(2, 4)], 2, cfg)
    assert mapper == [0, 1, 4, 5]
    for old, new in zip(cache.layers, pruned.layers):
        assert torch.equal(new.keys, old.keys[:, :, mapper])
        assert torch.equal(new.values, old.values[:, :, mapper])
    assert branch.get_seq_length() == 6
    with pytest.raises(ValueError):
        evict_cache(cache, list(range(6)), [(1, 3)], 2, cfg)


@torch.inference_mode()
def test_real_model_can_continue_after_noncontiguous_eviction():
    torch.manual_seed(233)
    cfg = config()
    model = Qwen2ForCausalLM(cfg).eval()
    out = model(input_ids=torch.tensor([[1, 2, 3, 4, 5, 6]]), use_cache=True)
    branch = fork_cache(out.past_key_values, cfg)
    pruned, mapper = evict_cache(out.past_key_values, list(range(6)), [(2, 4)], 2, cfg)
    next_out = model(input_ids=torch.tensor([[7]]), past_key_values=pruned,
                     position_ids=torch.tensor([[6]]), attention_mask=torch.ones(1, 5, dtype=torch.long),
                     use_cache=True)
    assert next_out.past_key_values.get_seq_length() == 5
    assert branch.get_seq_length() == 6
    assert torch.isfinite(next_out.logits).all()
