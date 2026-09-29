"""CPU checks for timing-only changes against the original Qwen implementation."""
import copy
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
from transformers import Qwen2Config, Qwen2ForCausalLM

from models.LegacyBaseline import (LegacyBaselineLLM, LEGACY_CONFIG_OVERRIDE,
                                   LEGACY_METHOD, BASE_MODE, RECENT_WINDOW)
from models.QwenLLMWithKVCache import QwenLLMWithKVCache
from token_tracker import TokenTracker

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path('/home/fengboyu/kvmem/results/rebuttal_20260927/source_snapshot/before_changes.tar')
torch.set_num_threads(1)


class TinyTokenizer:
    eos_token_id = 63
    def apply_chat_template(self, messages, **kwargs):
        return 'prompt'
    def __call__(self, text, return_tensors=None, **kwargs):
        ids = [ord(c) - 256 if 256 <= ord(c) < 320 else ord(c) % 61 + 1 for c in text]
        if return_tensors:
            tensor = torch.tensor([ids])
            return BatchEncoding(dict(input_ids=tensor, attention_mask=torch.ones_like(tensor)))
        return SimpleNamespace(input_ids=ids)
    def decode(self, ids, **kwargs):
        return ''.join(chr(256 + int(i)) for i in ids)


@pytest.fixture
def tiny_model(monkeypatch):
    torch.manual_seed(233)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512, eos_token_id=None, pad_token_id=0)
    config._attn_implementation = 'sdpa'
    template = Qwen2ForCausalLM(config).eval()
    template.generation_config.repetition_penalty = 1.05
    monkeypatch.setattr(AutoModelForCausalLM, 'from_pretrained', lambda *a, **k: copy.deepcopy(template))
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *a, **k: TinyTokenizer())


def make(cls=LegacyBaselineLLM, ratio=.2):
    return cls('tiny-test', {**LEGACY_CONFIG_OVERRIDE, 'cache_ratio': ratio}, TokenTracker())


def set_ratio(llm, ratio):
    llm.kv_config['cache_ratio'] = ratio
    llm.kv_manager.cache_ratio = ratio
    llm.kv_manager.target_cache_ratio = ratio
    llm.kv_manager.keep_ratio = ratio


def assert_equal_cache(left, right):
    assert left.current_cache_len == right.current_cache_len
    assert left.token_tracker.global_id_mapper == right.token_tracker.global_id_mapper
    assert left.token_tracker.next_global_id == right.token_tracker.next_global_id
    for a, b in zip(left.past_key_values.layers, right.past_key_values.layers):
        torch.testing.assert_close(a.keys, b.keys, rtol=0, atol=0)
        torch.testing.assert_close(a.values, b.values, rtol=0, atol=0)


def test_core_sources_and_decoder_probe_crop_are_unchanged():
    core = ('run_all_wiki_experiments_v2.py', 'models/QwenLLMWithKVCache.py',
            'models/QwenLLM.py', 'models/model_paths.py', 'kv_cache/kv_cache_manager.py',
            'kv_cache/pruning_strategy.py', 'kv_cache/h2o_scorer.py', 'token_tracker.py',
            'retrievers/WikiBM25Retriever.py')
    with tarfile.open(SNAPSHOT) as archive:
        for name in core:
            assert (ROOT / name).read_bytes() == archive.extractfile(name).read()
    for name in ('_decode', '_decode_token_by_token_with_pruning',
                 '_get_attention_for_scoring', 'truncate_cache'):
        assert getattr(LegacyBaselineLLM, name) is getattr(QwenLLMWithKVCache, name)
    assert LEGACY_CONFIG_OVERRIDE['num_score_layers'] == 3
    assert RECENT_WINDOW == (32 if BASE_MODE == 'h2o' else 0)


def test_no_eviction_matches_original_outputs_kv_positions_and_penalty(tiny_model, monkeypatch):
    old, step = make(QwenLLMWithKVCache, 1), make(ratio=1)
    first_old = old.generate_first('question', max_new_tokens=12)
    first_step = step.generate_first('question', max_new_tokens=12)
    assert first_old[0] == first_step[0]
    assert_equal_cache(old, step)
    # Preserve the original first generate() last-token bookkeeping discrepancy.
    assert step.current_cache_len == step.past_key_values.layers[0].keys.shape[2] + 1
    assert step.model.generation_config.repetition_penalty == 1.05
    assert step.cache_measurements == []
    for llm in (old, step):
        monkeypatch.setattr(llm.model, 'generate', lambda **k: pytest.fail('incremental must use original manual decoder'))
    for observation in ('first observation', 'another observation'):
        assert old.generate_incremental(observation, max_new_tokens=16) == step.generate_incremental(observation, max_new_tokens=16)
        assert_equal_cache(old, step)
    assert step.cache_measurements == []


@pytest.mark.parametrize('ratio', [.2, .5])
def test_first_round_keeps_original_pruning_and_budget_floor(tiny_model, ratio):
    old, step = make(QwenLLMWithKVCache, ratio), make(ratio=ratio)
    a = old.generate_first('question', max_new_tokens=80)
    b = step.generate_first('question', max_new_tokens=80)
    assert a[0] == b[0]
    assert_equal_cache(old, step)
    assert old.kv_manager.pruning_history == step.kv_manager.pruning_history
    assert step.cache_measurements and any(e['pruned'] for e in step.cache_measurements)
    assert {e['phase'] for e in step.cache_measurements} == {'initial_completed_block'}
    assert all(e['single_token_mode'] for e in step.cache_measurements)
    floor = step.kv_manager.protected_prefix_len + RECENT_WINDOW + 1
    assert step.current_cache_len <= max(step._compute_h2o_budget(), floor)
    assert step.token_tracker.global_id_mapper[:6] == list(range(6))


@pytest.mark.parametrize('ratio', [.2, .5])
def test_later_scoring_only_after_stop_crop_and_original_selector_parity(tiny_model, monkeypatch, ratio):
    # First generate identically, then let the old decoder run unpruned as a
    # reference. Its final cache is passed to the original boundary operations.
    old, step, preview = make(QwenLLMWithKVCache, 1), make(ratio=1), make(QwenLLMWithKVCache, 1)
    for llm in (old, step, preview):
        llm.generate_first('question', max_new_tokens=12)
    observation = 'observation evidence ' * 4
    text = preview.generate_incremental(observation, max_new_tokens=40)
    assert len(text) > 4
    # Any occurrence before the end gives actual inherited retokenization/crop.
    stop = text[3:5]
    assert text.find(stop) < len(text) - 2
    set_ratio(step, ratio)
    probe_states = []
    original_probe = step._get_attention_for_scoring
    def probe():
        assert not step._step_defer_pruning
        before = [(l.keys.clone(), l.values.clone()) for l in step.past_key_values.layers]
        probe_states.append((step.current_cache_len, step.past_key_values.layers[0].keys.shape[2]))
        result = original_probe()
        for prior, layer in zip(before, step.past_key_values.layers):
            torch.testing.assert_close(prior[0], layer.keys, rtol=0, atol=0)
            torch.testing.assert_close(prior[1], layer.values, rtol=0, atol=0)
        return result
    monkeypatch.setattr(step, '_get_attention_for_scoring', probe)
    for llm in (old, step):
        monkeypatch.setattr(llm.model, 'generate', lambda **k: pytest.fail('manual decode must remain active'))
    expected_response = old.generate_incremental(observation, max_new_tokens=40, stop_strings=[stop])
    actual_response = step.generate_incremental(observation, max_new_tokens=40, stop_strings=[stop])
    assert expected_response == actual_response
    assert len(actual_response) < len(text)
    assert probe_states and probe_states[0] == (old.current_cache_len, old.past_key_values.layers[0].keys.shape[2])
    # Apply exactly the original prefill batch/top-up recipe to the same final
    # cache, proving the adapter changes timing rather than score/selection.
    set_ratio(old, ratio)
    target = old._compute_h2o_budget()
    attentions = old._resolve_scoring_attentions()
    old._do_pruning(single_token_mode=False, attentions_override=attentions)
    for _ in range(max(0, old.current_cache_len - target + 8)):
        if old.current_cache_len <= target:
            break
        before = old.current_cache_len
        old._do_pruning(single_token_mode=True, attentions_override=attentions)
        if old.current_cache_len >= before:
            break
    assert_equal_cache(old, step)
    assert step.cache_measurements[0]['single_token_mode'] is False
    assert {e['phase'] for e in step.cache_measurements} == {'incremental_completed_block'}
    assert step._step_suppressed_calls > 0 and step._step_suppressed_probes == 1


def test_empty_decode_still_prunes_completed_observation(tiny_model, monkeypatch):
    step = make(ratio=1)
    step.generate_first('question', max_new_tokens=12)
    set_ratio(step, .2)
    # This represents the inherited decoder's immediate-EOS return.
    monkeypatch.setattr(step, '_decode', lambda *a, **k: ('', 0))
    assert step.generate_incremental('observation ' * 8, max_new_tokens=16) == ''
    assert step.cache_measurements and any(e['pruned'] for e in step.cache_measurements)
    assert all(e['phase'] == 'incremental_completed_block' for e in step.cache_measurements)


def test_exception_does_not_prune_partial_block_and_clears_gate(tiny_model, monkeypatch):
    step = make(ratio=1)
    step.generate_first('question', max_new_tokens=12)
    set_ratio(step, .2)
    def failed_decode(*a, **k):
        raise RuntimeError('synthetic decode failure')
    monkeypatch.setattr(step, '_decode', failed_decode)
    with pytest.raises(RuntimeError, match='synthetic decode failure'):
        step.generate_incremental('observation ' * 8, max_new_tokens=16)
    assert not step._step_defer_pruning
    assert step.cache_measurements == []


def test_audit_and_reset(tiny_model):
    step = make()
    step.generate_first('question', max_new_tokens=48)
    audit = step.legacy_audit()
    assert audit['method'] == LEGACY_METHOD
    assert audit['recent_window'] == RECENT_WINDOW
    assert audit['step_boundaries'][0]['phase'] == 'initial_completed_block'
    assert audit['cache_measurements']
    assert 'boundary' in audit['memory_scope']
    step.reset()
    assert step.cache_measurements == [] and step._step_boundaries == []
    assert not step._step_defer_pruning
