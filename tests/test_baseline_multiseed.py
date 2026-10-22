import copy
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoTokenizer, BatchEncoding, LlamaConfig, LlamaForCausalLM

from scripts.baseline_multiseed import protocol as p
from scripts.baseline_multiseed.adapters import LlamaIntentKVRuleLLM, method_spec, OVERRIDES
from scripts.baseline_multiseed.queue import prerequisites, blocked_jobs
from scripts.baseline_multiseed.worker import Store, NeuralTrace, check_rows, audit_read
from scripts.legacy_efficiency_metrics import cache_tensor_pairs
from models.QwenLLMWithKVCache import QwenLLMWithKVCache
from token_tracker import TokenTracker


class TinyTokenizer:
    eos_token_id = 63

    def apply_chat_template(self, messages, **kwargs):
        return 'SYS|' + messages[-1]['content'] + '|ASSIST'

    def __call__(self, text, return_tensors=None, return_offsets_mapping=False, **kwargs):
        ids = [ord(c) - 256 if 256 <= ord(c) < 320 else ord(c) % 61 + 1 for c in text]
        if return_tensors:
            tensor = torch.tensor([ids])
            return BatchEncoding({'input_ids': tensor, 'attention_mask': torch.ones_like(tensor)})
        if return_offsets_mapping:
            return {'input_ids': ids, 'offset_mapping': [(i, i + 1) for i in range(len(text))]}
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids, **kwargs):
        return ''.join(chr(256 + int(i)) for i in ids)


@pytest.fixture
def tiny(monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(233)
    config = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                        max_position_embeddings=2048, eos_token_id=None, pad_token_id=0)
    config._attn_implementation = 'sdpa'
    model = LlamaForCausalLM(config).eval()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    module = importlib.import_module('models.QwenLLMWithKVCache')
    monkeypatch.setattr(module, 'AutoModelForCausalLM',
                        SimpleNamespace(from_pretrained=lambda *a, **k: copy.deepcopy(model)))
    monkeypatch.setattr(AutoTokenizer, 'from_pretrained', lambda *a, **k: TinyTokenizer())
    return model


def test_requested_matrix_has_135_unique_groups_and_six_reusable_intent_conditions():
    jobs = p.requested()
    assert len(jobs) == len({j['name'] for j in jobs}) == 135
    assert sum(j['model'] == 'llama31_8b' for j in jobs) == 117
    reuse = [j for j in jobs if j['model'] == 'qwen25_7b' and j['seed'] == 233]
    assert len(reuse) == 6
    assert all(j['samples'] == 500 and j['max_steps'] == 7 for j in jobs)


def test_intent_20_requires_all_three_50_mains_same_model_and_seed():
    job = next(j for j in p.requested() if j['method'] == 'intentkv_rule' and j['ratio'] == .2)
    deps = prerequisites(job)
    assert len(deps) == 4 and any(d.endswith('/smoke') for d in deps)
    for dataset in p.DATASETS:
        assert p.name({**job, 'dataset': dataset, 'ratio': .5}) in deps


def test_failed_smoke_transitively_blocks_20_without_blocking_other_seeds():
    mains = p.requested()
    jobs = mains + [{**j, 'purpose': 'smoke', 'name': p.name({**j, 'purpose': 'smoke'})} for j in mains]
    failed = next(j for j in jobs if j['model'] == 'llama31_8b' and j['method'] == 'intentkv_rule'
                  and j['seed'] == 233 and j['ratio'] == .5 and j['purpose'] == 'smoke')
    blocked = blocked_jobs(jobs, {failed['name']: 'failed'})
    target = next(j for j in jobs if j['model'] == 'llama31_8b' and j['method'] == 'intentkv_rule'
                  and j['seed'] == 233 and j['ratio'] == .2 and j['purpose'] == 'main')
    assert target['name'] in blocked
    assert not any('/seed42/' in value for value in blocked)


def test_original_selector_is_shared_and_smoke_is_disjoint():
    import run_all_wiki_experiments_v2 as base
    rows = [{'id': str(i), 'question': f'Question {i}', 'answer': str(i)} for i in range(700)]
    for seed in p.SEEDS:
        main, smoke = p.select(base, rows, seed, 'main'), p.select(base, rows, seed, 'smoke')
        assert len(main) == 500 and len(smoke) == 2
        assert not {r['id'] for _, r in main} & {r['id'] for _, r in smoke}
        assert p.select(base, rows, seed, 'main') == main


def test_llama_intent_no_eviction_preserves_real_decoder_and_tensors(tiny):
    config = {**OVERRIDES, 'cache_ratio': .5}
    original = QwenLLMWithKVCache('tiny', dict(config, pruning_mode='tova'), TokenTracker())
    original.kv_config = config
    original._do_pruning = lambda *a, **k: None
    adapted = LlamaIntentKVRuleLLM('tiny', config, TokenTracker())
    adapted._compute_target_budget = lambda: adapted.current_cache_len
    with NeuralTrace(original.model) as a, NeuralTrace(adapted.model) as b:
        assert original.generate_first('question', 8)[0] == adapted.generate_first('question', 8)[0]
        assert original.generate_incremental('observation', 8) == adapted.generate_incremental('observation', 8)
    assert a.rows == b.rows
    for left, right in zip(cache_tensor_pairs(original.past_key_values), cache_tensor_pairs(adapted.past_key_values)):
        for x, y in zip(left, right):
            torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert adapted.query_memory.updates == 2
    assert len(adapted.query_memory.rows) == 2


@pytest.mark.parametrize('ratio', [.2, .5])
def test_llama_intent_really_prunes_and_forces_prompt_and_current_query(tiny, ratio):
    llm = LlamaIntentKVRuleLLM('tiny', {**OVERRIDES, 'cache_ratio': ratio}, TokenTracker())
    llm.generate_first('question', 40)
    for _ in range(2):
        llm.generate_incremental('observation' * 3, 40)
    events = [e for e in llm.intent_events if e.get('pruned')]
    assert events
    for event in events:
        assert set(event['forced_positions']) <= set(event['kept_positions'])
        assert event['resident_after'] < event['resident_before']
    assert llm.query_memory.updates == 3


def test_llama_lazy_uses_native_model_and_really_evicts(tiny):
    with method_spec('lazyeviction', 'llama31_8b') as spec:
        config = {**spec.overrides, 'cache_ratio': .2, 'lazy_window': 8}
        llm = spec.llm_class('tiny', config)
        assert llm.model.config.model_type == 'llama'
        llm.generate_first('question', 20)
        llm.generate_incremental('observation' * 4, 20)
        assert llm._legacy_evicted_slots > 0
        assert all(k.shape[1] == 4 for k, _ in cache_tensor_pairs(llm.past_key_values))
        assert all(state.ids.shape[-1] == llm.past_key_values.get_seq_length()
                   for state in llm._legacy_recurrence.values())


def test_llama_lazy_initial_generation_matches_original_without_eviction(tiny):
    original = QwenLLMWithKVCache('tiny', {'pruning_mode': 'none'}, TokenTracker())
    with method_spec('lazyeviction', 'llama31_8b') as spec:
        llm = spec.llm_class('tiny', {**spec.overrides, 'cache_ratio': .5})
        with NeuralTrace(original.model) as a, NeuralTrace(llm.model) as b:
            assert original.generate_first('question', 20)[0] == llm.generate_first('question', 20)[0]
        assert a.rows == b.rows
        assert llm._legacy_evicted_slots == 0


def test_durable_store_refuses_changed_content_or_corrupt_audit(tmp_path):
    samples = [[7, {'id': 'x', 'question': 'q', 'answer': 'gold'}]]
    fp = {'samples': 1, 'sample_ids': ['x']}
    store = Store(tmp_path, fp, samples)
    store.placement({'model': 'tiny'})
    row = {'index': 7, 'id': 'x', 'question': 'q', 'gold_answer': 'gold',
           'predicted_answer': 'gold', 'em': True, 'f1': 1., 'seconds': 1.}
    store.commit(row, {'actual': True})
    assert Store(tmp_path, fp, samples).rows == store.rows
    with pytest.raises(RuntimeError, match='content-bound'):
        Store(tmp_path, fp, [[7, {'id': 'x', 'question': 'changed', 'answer': 'gold'}]])
    path = tmp_path / store.rows[0]['audit']['path']
    path.write_text('{}')
    with pytest.raises(RuntimeError, match='corrupt'):
        Store(tmp_path, fp, samples)


def test_store_cannot_finalize_partial_or_change_model(tmp_path):
    store = Store(tmp_path, {'samples': 1}, [[1, {'id': 'x', 'question': 'q', 'answer': 'a'}]])
    with pytest.raises(RuntimeError, match='incomplete'):
        store.finalize()
    store.placement({'model': 'a'})
    with pytest.raises(RuntimeError, match='model'):
        store.placement({'model': 'b'})
