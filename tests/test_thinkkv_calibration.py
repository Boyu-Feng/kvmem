import copy

import numpy as np
import pytest

from scripts.thinkkv_calibration import calibrate, collector_class, kde_thresholds


def test_three_modes_recover_valleys_and_degenerate_data_is_rejected():
    rng = np.random.default_rng(233)
    data = np.concatenate([rng.normal(m, .01, 200) for m in [.15, .5, .85]])
    fit = kde_thresholds(data)
    assert fit['status'] == 'passed'
    assert .25 < fit['thresholds'][0] < .4
    assert .6 < fit['thresholds'][1] < .75
    assert kde_thresholds([.5] * 100)['status'] == 'insufficient_variation'


def test_failed_layer_intersection_cannot_fall_back_to_quantiles():
    rng = np.random.default_rng(233)
    three = np.concatenate([rng.normal(m, .01, 200) for m in [.15, .5, .85]]).tolist()
    rows = [dict(id='a', sparsity={'0': three, '1': [.5] * 100}),
            dict(id='b', sparsity={'0': [.5] * 100, '1': three})]
    report = calibrate(rows, required_layers=1)
    assert report['eligible_layers'] == []
    assert report['thresholds'] is None
    assert report['fallback'] is None
    assert report['status'] == 'not_applicable_under_strict_calibration'


def test_duplicate_prompts_rejected():
    with pytest.raises(ValueError, match='unique'):
        calibrate([dict(id='a'), dict(id='a')])


def test_observer_preserves_logits_and_cache_across_prefill_decode_and_crop(monkeypatch):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from models.QwenLLMWithKVCache import QwenLLMWithKVCache
    torch.manual_seed(233)
    config = Qwen2Config(vocab_size=40, hidden_size=32, intermediate_size=48,
                         num_hidden_layers=2, num_attention_heads=4,
                         num_key_value_heads=2, max_position_embeddings=128)
    config._attn_implementation = 'sdpa'
    reference = Qwen2ForCausalLM(config).eval()
    observed_model = copy.deepcopy(reference)

    def fake_init(self, *args, **kwargs):
        self.model = observed_model
        self.kv_manager = None
        self.token_tracker = None

    monkeypatch.setattr(QwenLLMWithKVCache, '__init__', fake_init)
    observer = collector_class()('unused')
    caches = [None, None]
    with torch.inference_mode():
        for ids in [[1, 2, 3], [4], [5], [6, 7], [8]]:
            outputs = []
            for i, model in enumerate([reference, observed_model]):
                out = model(input_ids=torch.tensor([ids]), past_key_values=caches[i], use_cache=True)
                caches[i] = out.past_key_values
                outputs.append(out.logits)
            torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
            for ref, obs in zip(caches[0].layers, caches[1].layers):
                torch.testing.assert_close(ref.keys, obs.keys, rtol=0, atol=0)
                torch.testing.assert_close(ref.values, obs.values, rtol=0, atol=0)
        assert all(len(v) == 3 for v in observer.export_sparsity().values())
        observer.past_key_values = caches[1]
        observer.current_cache_len = 8
        observer.truncate_cache(4)
        assert caches[1].get_seq_length() == 4
        assert all(len(v) == 1 for v in observer.export_sparsity().values())
        assert all(0 <= v[0] <= 1 for v in observer.export_sparsity().values())
