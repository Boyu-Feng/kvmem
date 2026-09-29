"""Legacy engine parity and LazyEviction hook integrity, entirely on CPU."""
from pathlib import Path
import tarfile

import pytest
import torch
from transformers import BatchEncoding, Qwen2Config, Qwen2ForCausalLM

from models.QwenLLMWithKVCache import QwenLLMWithKVCache
from models.LegacyBaseline import LegacyBaselineLLM, LEGACY_CONFIG_OVERRIDE, _select_at_boundary
from models.LazyEvictionLLM import RecurrenceState, retention_indices


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class Tokenizer:
    eos_token_id = 63
    def apply_chat_template(self, messages, **kwargs):
        return "formatted prompt"
    def __call__(self, text, return_tensors=None, **kwargs):
        if text == "formatted prompt":
            ids = [1, 2, 3]
        elif text == "observation":
            ids = [4, 5, 6, 7]
        elif text == "long observation":
            ids = list(range(4, 16))
        else:
            ids = [ord(c) - 65 for c in text]
        if return_tensors:
            return BatchEncoding(dict(input_ids=torch.tensor([ids]), attention_mask=torch.ones((1, len(ids)), dtype=torch.long)))
        return BatchEncoding(dict(input_ids=ids))
    def decode(self, ids, **kwargs):
        return "".join(chr(65 + i) for i in ids)


def runtime(cls=LegacyBaselineLLM, ratio=1, window=3):
    torch.manual_seed(9)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, eos_token_id=63, pad_token_id=0)
    config._attn_implementation = "sdpa"
    llm = cls.__new__(cls)
    llm.model = Qwen2ForCausalLM(config).eval()
    llm.model.generation_config.eos_token_id = None
    llm.model.generation_config.repetition_penalty = 1.05
    llm.device = torch.device("cpu")
    llm.tokenizer = Tokenizer()
    llm.kv_config = {**LEGACY_CONFIG_OVERRIDE, "cache_ratio": ratio, "lazy_window": window}
    llm.token_tracker = None
    llm.kv_manager = None
    llm.pruning_enabled = False
    llm.attn_mode = "scoring_forward"
    if isinstance(llm, LegacyBaselineLLM):
        llm._install_legacy_hooks()
    llm.reset()
    return llm


def capture_forward(llm):
    calls, logits = [], []
    def pre(module, args, kwargs):
        calls.append({name: kwargs[name].clone() if torch.is_tensor(kwargs.get(name)) else kwargs.get(name)
                      for name in ("input_ids", "position_ids", "cache_position", "attention_mask")})
    def post(module, args, kwargs, outputs):
        logits.append(outputs.logits.detach().clone())
    llm.model.register_forward_pre_hook(pre, with_kwargs=True)
    llm.model.register_forward_hook(post, with_kwargs=True)
    return calls, logits


def assert_capture_equal(left, right):
    assert len(left[0]) == len(right[0])
    for first, second in zip(left[0], right[0]):
        for name in first:
            if torch.is_tensor(first[name]):
                torch.testing.assert_close(first[name], second[name], atol=0, rtol=0)
            else:
                assert first[name] == second[name]
    for first, second in zip(left[1], right[1]):
        torch.testing.assert_close(first, second, atol=1e-6, rtol=1e-5)


def test_original_core_files_match_pre_change_snapshot():
    root = Path(__file__).resolve().parents[1]
    with tarfile.open('/home/fengboyu/kvmem/results/rebuttal_20260927/source_snapshot/before_changes.tar') as archive:
        for name in ('models/QwenLLMWithKVCache.py', 'run_all_wiki_experiments_v2.py',
                     'kv_cache/pruning_strategy.py'):
            assert (root / name).read_bytes() == archive.extractfile(name).read()


def test_no_eviction_preserves_generate_outputs_positions_processors_and_missing_kv():
    original, adapted = runtime(QwenLLMWithKVCache), runtime()
    old_capture, new_capture = capture_forward(original), capture_forward(adapted)
    old_first = original.generate_first('question', max_new_tokens=5)
    new_first = adapted.generate_first('question', max_new_tokens=5)
    assert old_first[0] == new_first[0]
    assert original.current_cache_len == adapted.current_cache_len
    assert adapted.current_cache_len - adapted._legacy_actual_length() == 1
    for _ in range(2):
        expected = original.generate_incremental('observation', max_new_tokens=5)
        actual = adapted.generate_incremental('observation', max_new_tokens=5)
        assert actual == expected
        assert original.current_cache_len == adapted.current_cache_len
        assert original.past_key_values.get_seq_length() == adapted._legacy_actual_length()
    assert adapted.model.generation_config.repetition_penalty == 1.05
    assert adapted.legacy_audit()['total_prune_count'] == 0
    assert_capture_equal(old_capture, new_capture)


def test_legacy_stop_crops_same_physical_cache_and_matching_per_head_metadata():
    original, adapted = runtime(QwenLLMWithKVCache), runtime()
    def force_h(module, args, outputs):
        outputs.logits.fill_(-100)
        outputs.logits[..., 7] = 100
        return outputs
    for llm in (original, adapted):
        llm.model.register_forward_hook(force_h)
    assert original.generate_first('question', max_new_tokens=7, stop_strings=['HHH'])[0] == ''
    assert adapted.generate_first('question', max_new_tokens=7, stop_strings=['HHH'])[0] == ''
    for _ in range(2):
        assert original.generate_incremental('observation', max_new_tokens=6, stop_strings=['HH']) == ''
        assert adapted.generate_incremental('observation', max_new_tokens=6, stop_strings=['HH']) == ''
        assert original.current_cache_len == adapted.current_cache_len
        assert original.past_key_values.get_seq_length() == adapted._legacy_actual_length()
        for state in adapted._legacy_recurrence.values():
            assert state.ids.shape[-1] == adapted._legacy_actual_length()
            assert state.last.max() <= adapted._legacy_clock
    assert len(adapted.legacy_audit()['stop_crops']) == 3
    assert adapted._legacy_clock > adapted._legacy_actual_length() - 3


@pytest.mark.parametrize('ratio', [.2, .5])
def test_initial_turn_is_unpruned_but_observation_and_decode_windows_evict(ratio):
    adapted = runtime(ratio=ratio)
    adapted.generate_first('question', max_new_tokens=8)
    initial_audit = adapted.legacy_audit()
    assert initial_audit['total_prune_count'] == 0
    assert initial_audit['actual_cache_slots_per_head'] == 10
    assert all(e['initial_generation_eviction_disabled'] for e in initial_audit['cache_measurements'])
    adapted.generate_incremental('long observation', max_new_tokens=5)
    audit = adapted.legacy_audit()
    assert audit['total_prune_count'] >= 1
    prunes = [e for e in audit['cache_measurements'] if e['pruned']]
    assert any(e['deferred_prefill_window'] for e in prunes)
    for event in prunes:
        assert event['after_tokens'] == event['effective_target_tokens']
        assert event['after_tokens'] >= 3 + min(3, event['query_clock'])
    for state in adapted._legacy_recurrence.values():
        torch.testing.assert_close(state.ids[..., :3], torch.arange(3).expand(1, 4, -1))
    assert audit['head_expansion_factor'] == 2
    assert audit['legacy_reported_minus_actual'] == 2
    assert audit['cache_bytes'] == sum(x.numel() * x.element_size() for pair in adapted._legacy_pairs() for x in pair)
    assert audit['returned_cache_snapshot_bytes'] > 0


def test_selector_matches_official_port_at_true_window_boundary():
    ids = torch.arange(18).expand(1, 4, -1)
    gen = torch.Generator().manual_seed(1)
    state = RecurrenceState(ids, torch.randint(0, 9, (1, 4, 18), generator=gen),
                            torch.randint(0, 5, (1, 4, 18), generator=gen))
    torch.testing.assert_close(_select_at_boundary(state, 9, 11, 3, 3),
                              retention_indices(state, 9, 11, 3, 3), atol=0, rtol=0)


def test_reset_does_not_leak_query_clocks_or_evictions():
    adapted = runtime(ratio=.2)
    adapted.generate_first('question', max_new_tokens=8)
    adapted.generate_incremental('long observation', max_new_tokens=5)
    assert adapted.legacy_audit()['total_prune_count']
    adapted.generate_first('question', max_new_tokens=3)
    audit = adapted.legacy_audit()
    assert audit['total_prune_count'] == 0
    assert audit['processed_nonprompt_queries'] == 2
    assert audit['actual_cache_slots_per_head'] == 5
