"""CPU qualification of the new shared actor and its source closure."""
import copy
import importlib
import json
import os
import shutil
import subprocess
import sys

import pytest
import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from models.QwenLLMWithKVCache import QwenLLMWithKVCache
from scripts import webbench_v1 as runner
from scripts.unified_sidequest_runtime import UnifiedSideQuestLLM
from scripts.unified_thinkkv_runtime import UnifiedThinkKVLLM
from token_tracker import TokenTracker


def test_source_snapshot_contains_frozen_adapter_dependencies_and_root_imports():
    sources = runner.sources()
    manifest = runner.read(runner.ROOT / 'rebuttal/adapters/manifest.json')
    required = {'run_rebuttal_legacy.py'}
    for name in ('h2o_step', 'tova_step', 'flowkv_style', 'lazyeviction'):
        method = manifest['methods'][name]
        required.update(method['original_core_hashes'])
        for path, record in method['source_files'].items():
            required.add(f'rebuttal/adapters/{name}/{path}' if record['from'] == 'adapter' else path)
    assert required <= sources.keys(), sorted(required - sources.keys())


def test_full_method_metadata_loads_from_isolated_source_snapshot(tmp_path):
    for relative in runner.sources():
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(runner.ROOT / relative, destination)
    code = (
        "import json; from scripts import webbench_v1 as r; "
        "jobs=[j for j in r.jobs() if j['purpose']=='smoke']; "
        "print(json.dumps([r.metadata(j) for j in jobs]))"
    )
    completed = subprocess.run([sys.executable, '-B', '-c', code], cwd=tmp_path,
                               env={**os.environ, 'PYTHONPATH': str(tmp_path),
                                    'CUDA_VISIBLE_DEVICES': ''},
                               text=True, capture_output=True, check=True, timeout=120)
    metadata = json.loads(completed.stdout.splitlines()[-1])
    assert len(metadata) == 64
    assert all(item['config'] and item['class'] and item['scope'] for item in metadata)


def test_matrix_has_exact_user_conditions_and_fresh_baselines():
    jobs = runner.jobs()
    main = [job for job in jobs if job['purpose'] == 'main']
    assert len({job['name'] for job in jobs}) == len(jobs)
    assert len(main) == 192
    assert len([job for job in jobs if job['purpose'] == 'smoke']) == 64
    for benchmark in runner.BENCHMARKS:
        for model in runner.READY:
            for seed in runner.SEEDS:
                subset = [job for job in main if (job['benchmark'], job['model'], job['seed'])
                          == (benchmark, model, seed)]
                assert len(subset) == 16
                assert sum(job['method'] == 'fullkv' for job in subset) == 1
                assert sum(job['method'] == 'sidequest_untrained' for job in subset) == 1
                assert {job['ratio'] for job in subset if job['method'] == 'stepkv'} == {.5, .2}


@pytest.fixture
def tiny_llama(monkeypatch):
    torch.set_num_threads(2)
    torch.manual_seed(233)
    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {'<eos>': 0, '<pad>': 1}
    vocab.update({char: index + 2 for index, char in enumerate(alphabet)})
    core = Tokenizer(models.BPE(vocab, []))
    core.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)
    core.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=core, eos_token='<eos>', pad_token='<pad>')
    tokenizer.chat_template = "{{ messages[0]['content'] }} {{ messages[1]['content'] }}"
    config = LlamaConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=48,
                         num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                         eos_token_id=0, pad_token_id=1, max_position_embeddings=4096)
    config._attn_implementation = 'sdpa'
    model = LlamaForCausalLM(config).to(dtype=torch.bfloat16).eval()
    module = importlib.import_module('models.QwenLLMWithKVCache')
    monkeypatch.setattr(module.AutoTokenizer, 'from_pretrained', lambda *a, **k: copy.deepcopy(tokenizer))
    monkeypatch.setattr(module.AutoModelForCausalLM, 'from_pretrained', lambda *a, **k: copy.deepcopy(model))
    return model


def force(llm, main_text, auxiliary_text='{"del_cursors": [0]}'):
    sequences = {key: llm.tokenizer(text, add_special_tokens=False).input_ids + [0]
                 for key, text in (('main', main_text), ('auxiliary', auxiliary_text))}
    state = {'main': 0, 'auxiliary': 0}

    def forward(model, args, kwargs, output):
        auxiliary = hasattr(llm, '_thread') and getattr(llm._thread, 'auxiliary', False)
        branch = 'auxiliary' if auxiliary else 'main'
        sequence = sequences[branch]
        token = sequence[min(state[branch], len(sequence) - 1)]
        output.logits.fill_(-100)
        output.logits[:, -1, token] = 100
        state[branch] += 1
        return output

    llm.model.register_forward_hook(forward, with_kwargs=True)
    return state


@pytest.mark.parametrize('cls', [UnifiedThinkKVLLM, UnifiedSideQuestLLM])
def test_disabled_adapters_keep_actual_llama_greedy_decoder_and_cache(tiny_llama, cls):
    control = QwenLLMWithKVCache('tiny', token_tracker=TokenTracker())
    llm = cls('tiny', token_tracker=TokenTracker(), enabled=False)
    try:
        assert control.generate_first('Question: public', 8)[0] == llm.generate_first('Question: public', 8)[0]
        text = '\nObservation 1: public evidence\nThought 2:'
        assert control.generate_incremental(text, 8) == llm.generate_incremental(text, 8)
        for a, b in zip(control.past_key_values.layers, llm.past_key_values.layers):
            assert torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values)
        assert llm.model.config.model_type == 'llama'
    finally:
        if hasattr(llm, 'close'):
            llm.close()


@pytest.mark.parametrize('method', ['h2o_step', 'tova_step'])
def test_frozen_baseline_accepts_shared_runner_metadata(tiny_llama, method):
    job = {'method': method, 'model': 'llama31_8b', 'ratio': .2}
    configuration = runner.metadata(job)['config']
    with runner.method_spec(method, job['model']) as specification:
        llm = specification.llm_class('tiny', configuration, TokenTracker())
    assert llm.kv_config['attn_mode'] == 'scoring_forward'
    assert llm.model.config.model_type == 'llama'


def test_thinkkv_real_llama_native_bf16_eviction(tiny_llama):
    llm = UnifiedThinkKVLLM('tiny', {'pruning_mode': 'none', 'cache_ratio': .2}, TokenTracker())
    force(llm, 'x' * 300)
    llm.generate_first('Question: public', 140)
    assert any(event['annealing'] for event in llm.events)
    assert llm.current_cache_len < llm.selection.next_id
    assert all(layer.keys.dtype == torch.bfloat16 and layer.keys.shape[1] == 2
               for layer in llm.past_key_values.layers)


def test_sidequest_actual_llama_auxiliary_and_next_turn_eviction(tiny_llama):
    llm = UnifiedSideQuestLLM('tiny', {'sidequest_interval': 1}, TokenTracker())
    state = force(llm, 'Action 1: Search[public]\nObservation hallucinated')
    try:
        llm.generate_first('Question: public\nThought 1:', 70, ['\nObservation'])
        state['main'] = 0
        llm.generate_incremental('\nObservation 1: public evidence\nThought 2:', 70, ['\nObservation'])
        llm.collect_pending(apply=True, wait=True)
        assert state['auxiliary'] > 0
        llm.interval = 100
        state['main'] = 0
        llm.generate_incremental('\nObservation 2: next evidence\nThought 3:', 70, ['\nObservation'])
        event = llm.audit()['aux_events'][0]
        assert event['applied']
        assert event['after_tokens'] < event['before_tokens']
        assert llm.model.config.model_type == 'llama'
    finally:
        llm.close()
