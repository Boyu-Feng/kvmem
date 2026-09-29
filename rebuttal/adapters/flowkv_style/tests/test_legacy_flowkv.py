"""Legacy engine parity and Flow-specific isolation tests; CPU only."""
import copy
import hashlib
from pathlib import Path
import tarfile
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
from transformers import Qwen2Config, Qwen2ForCausalLM

from models.LegacyBaseline import LegacyBaselineLLM, LEGACY_CONFIG_OVERRIDE
from models.QwenLLMWithKVCache import QwenLLMWithKVCache
from token_tracker import TokenTracker

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = Path('/home/fengboyu/kvmem/results/rebuttal_20260927/source_snapshot/before_changes.tar')


class TinyTokenizer:
    eos_token_id = 63
    def apply_chat_template(self, messages, **kwargs):
        return "prompt"
    def __call__(self, text, return_tensors=None, **kwargs):
        ids = [ord(c) - 256 if 256 <= ord(c) < 320 else ord(c) % 61 + 1 for c in text]
        if return_tensors:
            tensor = torch.tensor([ids])
            return BatchEncoding(dict(input_ids=tensor, attention_mask=torch.ones_like(tensor)))
        return SimpleNamespace(input_ids=ids)
    def decode(self, ids, **kwargs):
        return "".join(chr(256 + int(i)) for i in ids)


@pytest.fixture
def model_factory(monkeypatch):
    torch.manual_seed(233)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, eos_token_id=None, pad_token_id=0)
    config._attn_implementation = "sdpa"
    template = Qwen2ForCausalLM(config).eval()
    template.generation_config.repetition_penalty = 1.05
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *a, **k: copy.deepcopy(template))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: TinyTokenizer())
    return template


def flow(ratio=.2):
    return LegacyBaselineLLM("tiny-test", {**LEGACY_CONFIG_OVERRIDE, "cache_ratio": ratio})


def append_legacy(llm, ids):
    with torch.no_grad():
        out = llm.model(input_ids=torch.tensor([ids]), past_key_values=llm.past_key_values,
                        use_cache=True, return_dict=True)
    llm.past_key_values = out.past_key_values
    llm.current_cache_len += len(ids)
    llm.kv_manager.current_cache_len = llm.current_cache_len
    llm.token_tracker.append_new_tokens(len(ids))
    llm._all_token_ids.extend(ids)
    llm._extend_global_token_log(ids)


def initialize_prompt(llm):
    append_legacy(llm, [1, 2, 3])
    llm.kv_manager.register_initial_cache(3)


def assert_caches_equal(left, right):
    for a, b in zip(left.past_key_values.layers, right.past_key_values.layers):
        torch.testing.assert_close(a.keys, b.keys, rtol=0, atol=0)
        torch.testing.assert_close(a.values, b.values, rtol=0, atol=0)


def test_legacy_sources_are_byte_identical_to_before_changes_snapshot():
    names = ['models/QwenLLMWithKVCache.py', 'run_all_wiki_experiments_v2.py',
             'kv_cache/pruning_strategy.py', 'kv_cache/kv_cache_manager.py',
             'kv_cache/h2o_scorer.py', 'token_tracker.py']
    with tarfile.open(SNAPSHOT) as tar:
        for name in names:
            before = tar.extractfile(name).read()
            assert hashlib.sha256(before).digest() == hashlib.sha256((ROOT / name).read_bytes()).digest()
    for name in ['_decode', '_decode_token_by_token_with_pruning',
                 '_get_attention_for_scoring', 'truncate_cache']:
        assert getattr(LegacyBaselineLLM, name) is getattr(QwenLLMWithKVCache, name)


def test_no_eviction_preserves_legacy_generate_stop_penalty_and_cache(model_factory):
    # Discover a deterministic stop substring using only the tiny CPU model.
    old = QwenLLMWithKVCache('tiny-test', {'pruning_mode': 'none'}, TokenTracker())
    old.reset()
    preview, _, _ = old.generate_first('question', max_new_tokens=12)
    stop = preview[3:5]
    assert len(stop) == 2
    baseline = flow(1.0)
    a = old.generate_first('question', max_new_tokens=12, stop_strings=[stop])
    b = baseline.generate_first('question', max_new_tokens=12, stop_strings=[stop])
    assert a[0] == b[0]
    assert old.current_cache_len == baseline.current_cache_len
    assert_caches_equal(old, baseline)
    assert baseline.cache_measurements == [] and baseline._flowkv_frozen_ids == ()
    assert baseline.model.generation_config.repetition_penalty == 1.05
    # Leave the second block untruncated: its last selected token remains
    # unmaterialized just as in the old model.generate path.
    a = old.generate_incremental('observation', max_new_tokens=12)
    b = baseline.generate_incremental('observation', max_new_tokens=12)
    assert a == b
    assert old.current_cache_len == baseline.current_cache_len
    assert_caches_equal(old, baseline)
    assert not baseline.cache_measurements[-1]['pruned']
    assert baseline.current_cache_len == baseline._resident_state()[0] + 1


@pytest.mark.parametrize('ratio', [.2, .5])
def test_multiple_blocks_preserve_frozen_history_and_budget(model_factory, ratio):
    llm = flow(ratio)
    initialize_prompt(llm)
    for step, count in enumerate([20, 9, 13], start=2):
        frozen_ids = tuple(llm.token_tracker.global_id_mapper)
        prior = [(k.clone(), v.clone()) for k, v in llm._resident_layers()]
        llm.token_tracker.set_current_step(step)
        append_legacy(llm, [4] * count)
        llm._freeze_completed_block()
        assert llm.current_cache_len == 3 + int(ratio * (llm.token_tracker.next_global_id - 3))
        assert tuple(llm.token_tracker.global_id_mapper[:len(frozen_ids)]) == frozen_ids
        for (ka, va), (kb, vb) in zip(prior, llm._resident_layers()):
            torch.testing.assert_close(kb[:, :, :len(frozen_ids)], ka, rtol=0, atol=0)
            torch.testing.assert_close(vb[:, :, :len(frozen_ids)], va, rtol=0, atol=0)
        assert llm.cache_measurements[-1]['frozen_global_ids_before'] == list(frozen_ids)


def test_selector_matches_repository_tova_and_ignores_step_utility(model_factory):
    llm = flow()
    initialize_prompt(llm)
    append_legacy(llm, [4 + i for i in range(20)])
    attentions = llm._get_attention_for_scoring()
    tracker = TokenTracker()
    tracker.append_new_tokens(23)
    selector = llm.kv_manager.pruning_strategy
    selector.token_tracker = tracker
    selector.prune(llm.past_key_values, attentions, prune_start=3, prune_end=23,
                   keep_ratio=.2, observation_window=0)
    expected_ids = tracker.global_id_mapper
    selector.token_tracker = llm.token_tracker
    llm.kv_manager.step_scores = {1: 1e12, 2: -1e12}
    llm.kv_manager.step_aware_alpha = 0.0
    llm.kv_manager.step_aware_beta = 1.0
    llm._freeze_completed_block()
    assert llm.token_tracker.global_id_mapper == expected_ids


def test_zero_quota_evicts_only_new_tokens(model_factory):
    llm = flow()
    initialize_prompt(llm)
    append_legacy(llm, [4] * 20)
    llm._freeze_completed_block()
    frozen = llm._flowkv_frozen_ids
    append_legacy(llm, [5])
    llm._get_attention_for_scoring = lambda: pytest.fail('zero quota needs no probe')
    llm._freeze_completed_block()
    assert llm._flowkv_frozen_ids == frozen
    assert llm.cache_measurements[-1]['new_tokens_kept'] == 0


def test_nonresident_count_is_audited_without_materializing_missing_token(model_factory):
    llm = flow()
    initialize_prompt(llm)
    append_legacy(llm, [4] * 20)
    llm.current_cache_len += 1
    llm.kv_manager.current_cache_len += 1
    llm.token_tracker.append_new_tokens(1)
    llm._all_token_ids.append(5)
    old_tokens = list(llm._all_token_ids)
    llm._freeze_completed_block()
    event = llm.cache_measurements[-1]
    assert event['before_tokens'] == 23
    assert event['bookkeeping_tokens_before'] == 24
    assert event['counted_but_nonresident_before'] == 1
    assert 23 not in llm.token_tracker.global_id_mapper
    assert llm.token_tracker.next_global_id == 24
    assert llm._all_token_ids == old_tokens  # do not upgrade legacy token logging


def test_inherited_probe_does_not_change_live_cache_or_use_absolute_positions(model_factory):
    llm = flow()
    initialize_prompt(llm)
    append_legacy(llm, [4] * 20)
    llm._freeze_completed_block()
    prior = [(k.clone(), v.clone()) for k, v in llm._resident_layers()]
    calls = []
    handle = llm.model.register_forward_pre_hook(lambda module, args, kwargs: calls.append(kwargs), with_kwargs=True)
    llm._get_attention_for_scoring()
    handle.remove()
    assert len(calls) == 1 and 'position_ids' not in calls[0]
    for (ka, va), (kb, vb) in zip(prior, llm._resident_layers()):
        torch.testing.assert_close(ka, kb, rtol=0, atol=0)
        torch.testing.assert_close(va, vb, rtol=0, atol=0)


def test_legacy_audit_and_episode_reset(model_factory):
    llm = flow()
    initialize_prompt(llm)
    append_legacy(llm, [4] * 20)
    llm._freeze_completed_block()
    audit = llm.legacy_audit()
    assert audit['method'] == 'flowkv_style'
    assert len(audit['cache_measurements']) == 1
    assert audit['resident_tokens'] == 7
    assert 'boundary' in audit['memory_scope']
    llm.reset()
    assert not llm.cache_measurements and not llm._flowkv_frozen_ids
