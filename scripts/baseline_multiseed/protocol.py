"""Freeze shared task selections and inventory completed runs before admission."""
from __future__ import annotations

import importlib.metadata
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

from scripts import run_component_ablation_rebuttal as assets
from scripts import run_qa32_rebuttal as io

ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/mnt/checkpoints/bruli/fengboyu/kvmem_rebuttal_20260929')
DEFAULT_OUTPUT = BASE / 'results/qa_baselines_two_models_three_seeds_20261013_v1'
OLD_INTENT = BASE / 'results/intentkv_rule_qa_7b_seed233_20261005_v5'
MODEL_READY = {'qwen25_7b': BASE / 'assets/7b_ready.json',
               'llama31_8b': BASE / 'ablation_assets/llama31_8b_shared_ready.json'}
DATA_READY = BASE / 'qa32_assets/data_ready.json'
SEEDS = (233, 42, 3407)
DATASETS = ('hotpotqa', '2wiki', 'musique')
BASELINES = ('flowkv_style', 'lazyeviction', 'sidequest_untrained',
             'thinkkv_eviction_only', 'h2o_step', 'tova_step')
VERSIONS = ('torch', 'transformers', 'tokenizers', 'datasets', 'numpy', 'safetensors')


def name(job):
    budget = 'adaptive' if job['ratio'] is None else f"r{int(100 * job['ratio'])}"
    return f"{job['model']}/seed{job['seed']}/{job['dataset']}/{job['method']}/{budget}/{job['purpose']}"


def requested():
    jobs = []
    for model in ('llama31_8b', 'qwen25_7b'):
        methods = (*BASELINES, 'intentkv_rule') if model == 'llama31_8b' else ('intentkv_rule',)
        for seed, method, dataset in itertools.product(SEEDS, methods, DATASETS):
            for ratio in ((None,) if method == 'sidequest_untrained' else (.5, .2)):
                job = dict(model=model, seed=seed, method=method, dataset=dataset,
                           ratio=ratio, purpose='main', samples=500, sample_start=0, max_steps=7)
                job['name'] = name(job)
                jobs.append(job)
    return jobs


def select(base, rows, seed, purpose):
    if seed not in SEEDS or purpose not in ('main', 'smoke'):
        raise ValueError('Unregistered selection')
    if len({r['id'] for r in rows}) != len(rows):
        raise RuntimeError('Duplicate dataset IDs')
    with assets.patch_attribute(base, 'RANDOM_SEED', seed), assets.patch_attribute(base, 'NUM_SAMPLES', 502):
        selected = base.select_samples(rows)
    if len(selected) != 502 or len({r['id'] for _, r in selected}) != 502:
        raise RuntimeError('Incomplete common task selection')
    return selected[:500] if purpose == 'main' else selected[500:502]


def source_hashes():
    from scripts.latency_runtime import experiment_hashes
    from scripts.run_intentkv_rule import source_hashes as intent_hashes
    values = {**assets.source_hashes(), **experiment_hashes(), **intent_hashes()}
    names = ('scripts/sidequest_runtime.py', 'scripts/run_sidequest_baseline.py',
             'scripts/thinkkv_eviction_runtime.py', 'scripts/run_thinkkv_eviction.py',
             'scripts/qa32_recent_tensor_audit.py')
    values.update({p: io.file_hash(ROOT / p) for p in names})
    values.update({str(p.relative_to(ROOT)): io.file_hash(p)
                   for p in Path(__file__).parent.glob('*.py')})
    return values


def verify_sources(plan):
    for p, checksum in plan['source_hashes'].items():
        if io.file_hash(ROOT / p) != checksum:
            raise RuntimeError(f'Frozen source changed: {p}')
    versions = {p: importlib.metadata.version(p) for p in VERSIONS}
    if versions != plan['versions']:
        raise RuntimeError('Frozen runtime changed')


def reference_rows(dataset, seed):
    directory = 'rebuttal_20261001_053953_UTC' if seed == 233 else 'multiseed_20261007_065533_UTC'
    base = ROOT / 'reports' / directory
    summary = io.read_json(base / 'summary.json')
    groups = summary.get('main_conditions', summary.get('groups'))
    row, = [g for g in groups if (g['dataset'], g['seed'], g['method'], g['budget'])
            == (dataset, seed, 'flowkv_style', .5)]
    path = base / row['per_question']
    return io.read_json(path), {'path': str(path), 'sha256': io.file_hash(path)}


def selected(plan, job):
    return plan['tasks'][job['dataset']][str(job['seed'])][job['purpose']]


def method_metadata(base, method, model, ratio):
    if method in ('sidequest_untrained', 'thinkkv_eviction_only'):
        return {'kv_config': {'pruning_mode': 'none', 'attn_mode': 'scoring_forward'},
                'mode': 'none', 'decoder': 'historical_RebuttalLLM_independent_greedy_logical_RoPE',
                'scope': ('Historical SideQuestPilot untrained adaptive decoder, interval=4, aux_max_tokens=128; '
                          'cursor labels in observations; asynchronous decisions; no fixed-ratio claim.'
                          if method == 'sidequest_untrained' else
                          'Historical ThinKVEvictionOnlyLLM; native-dtype KV, 128-token segments, '
                          'oldest completed segment halving, min4, post-RoPE K-means medoids; no quantization.')}
    from scripts.baseline_multiseed.adapters import method_spec
    with method_spec(method, model) as spec:
        return {'kv_config': spec.config(base, ratio), 'mode': spec.mode, 'scope': spec.scope,
                'decoder': 'historical_QwenLLMWithKVCache_HF_generate_posthoc_crop'}


def validate_existing_intent(plan, job):
    from scripts.run_intentkv_rule import validate_completed_run, source_hashes as old_sources
    import run_all_wiki_experiments_v2 as base
    path = OLD_INTENT / f"main_{job['dataset']}_ratio{int(100 * job['ratio'])}_seed233"
    fp = validate_completed_run(path)
    rows = io.read_json(path / 'results.json')['results']
    samples = selected(plan, job)
    if (fp['seed'] != job['seed'] or fp['model'] != plan['models'][job['model']]['repo']
            or fp['model_revision'] != plan['models'][job['model']]['revision']
            or fp['ratio'] != job['ratio'] or fp['max_steps'] != 7 or fp['samples'] != 500
            or fp['source_hashes'] != old_sources()):
        raise RuntimeError('Existing IntentKV model/protocol/source mismatch')
    for row, (index, sample) in zip(rows, samples, strict=True):
        if (row['index'], row['id'], row['question'], row['gold_answer']) != (
                index, sample['id'], sample['question'], sample['answer']):
            raise RuntimeError('Existing IntentKV task content/order differs')
        if (row['em'] != base.exact_match(row['predicted_answer'], row['gold_answer'])
                or row['f1'] != base.f1_score(row['predicted_answer'], row['gold_answer'])):
            raise RuntimeError('Existing IntentKV original score mismatch')
    return {'path': str(path), 'reason': 'complete_500_validated_including_durable_audits_and_original_scorer',
            'file_hashes': {p: io.file_hash(path / p) for p in ('manifest.json', 'checkpoint.json', 'results.json')},
            'fingerprint_sha256': io.digest(fp)}


def prepare(output, devices):
    import run_all_wiki_experiments_v2 as base
    from transformers import GenerationConfig
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new output directory for a new frozen plan')
    if not devices or len(set(devices)) != len(devices) or any(d not in range(8) for d in devices):
        raise ValueError('Invalid GPU pool')
    plan = {'schema': 1, 'protocol': 'historical_baseline_model_seed_extension_v1',
            'output': str(output), 'devices': devices, 'models': {}, 'tasks': {}, 'datasets': {},
            'methods': {}, 'existing': {}, 'requested': requested(), 'jobs': [],
            'versions': {p: importlib.metadata.version(p) for p in VERSIONS},
            'data_ready': str(DATA_READY), 'source_hashes': source_hashes(),
            'notes': ['User requested missing-only expansion; completed runs retained without rerunning.',
                      'Same ordered tasks/content per dataset and seed for both models and all methods.',
                      'Historical SideQuest/ThinKV independent decoders remain separate protocol strata; no matched-runtime ranking.',
                      'Nominal token ratios do not imply equal KV bytes; accuracy times are diagnostic only.',
                      'IntentKV is the agent QueryMemory rule adaptation, without a learned residual head.']}
    data_rows = {}
    for model_key, ready in MODEL_READY.items():
        args = SimpleNamespace(model_ready=ready, data_ready=DATA_READY, dataset='hotpotqa')
        model, wiki, rows, provenance, generation = assets.load_assets(args)
        record = io.read_json(ready)
        weight_hashes = {}
        for shard in provenance['weight_shard_inventory']:
            checksum = io.file_hash(model / shard)
            expected = record.get('weight_inventory', {}).get(shard, {}).get('sha256')
            if expected and checksum != expected:
                raise RuntimeError('Model weight differs from readiness hashes')
            weight_hashes[shard] = checksum
        plan['models'][model_key] = {'repo': record['repo'], 'revision': record['revision'],
            'path': str(model), 'model_ready': str(ready), 'provenance': provenance,
            'config': io.read_json(model / 'config.json'), 'generation_file': generation,
            'loaded_generation_config': GenerationConfig.from_pretrained(model, local_files_only=True).to_dict(),
            'weight_sha256': weight_hashes}
        data_rows['hotpotqa'] = rows
        if not plan['datasets']:
            for dataset in DATASETS:
                args.dataset = dataset
                _, _, rows, provenance, _ = assets.load_assets(args)
                data_rows[dataset] = rows
                plan['datasets'][dataset] = {'wiki_path': str(wiki), 'provenance': provenance}
    for dataset, rows in data_rows.items():
        if len(rows) != assets.DATASETS[dataset]:
            raise RuntimeError('Wrong complete dataset length')
        plan['tasks'][dataset] = {}
        for seed in SEEDS:
            main, smoke = select(base, rows, seed, 'main'), select(base, rows, seed, 'smoke')
            published, binding = reference_rows(dataset, seed)
            if [(i, r['id']) for i, r in main] != [(r['index'], r['id']) for r in published]:
                raise RuntimeError('Tasks differ from published three-seed baseline selection')
            plan['tasks'][dataset][str(seed)] = {'main': main, 'smoke': smoke,
                'main_content_sha256': io.digest(main), 'smoke_content_sha256': io.digest(smoke),
                'published_selection_binding': binding}
    for job in plan['requested']:
        method_key = f"{job['model']}/{job['method']}/{job['ratio']}"
        if method_key not in plan['methods']:
            plan['methods'][method_key] = method_metadata(base, job['method'], job['model'], job['ratio'])
        if job['model'] == 'qwen25_7b' and job['seed'] == 233:
            plan['existing'][job['name']] = validate_existing_intent(plan, job)
        else:
            smoke = {**job, 'purpose': 'smoke', 'samples': 2, 'sample_start': 500}
            smoke['name'] = name(smoke)
            plan['jobs'].extend([smoke, job])
    verify_sources(plan)
    plan['plan_sha256'] = io.digest(plan)
    output.mkdir(parents=True)
    io.atomic_json(output / 'plan.json', plan)
    io.atomic_json(output / 'preflight.json', {'status': 'passed', 'plan_sha256': plan['plan_sha256'],
        'requested_main': len(plan['requested']), 'skipped_main': len(plan['existing']),
        'new_main': sum(j['purpose'] == 'main' for j in plan['jobs']), 'new_smoke': sum(j['purpose'] == 'smoke' for j in plan['jobs']),
        'task_selections_match_all_nine_published_dataset_seed_pairs': True})
    return plan


def load_plan(output):
    plan = io.read_json(Path(output) / 'plan.json')
    checksum = plan.pop('plan_sha256')
    if io.digest(plan) != checksum:
        raise RuntimeError('Frozen plan checksum mismatch')
    plan['plan_sha256'] = checksum
    verify_sources(plan)
    return plan


def fingerprint(plan, job):
    model = plan['models'][job['model']]
    samples = selected(plan, job)
    return {'protocol': plan['protocol'], 'plan_sha256': plan['plan_sha256'], **job,
            'model_repo': model['repo'], 'model_revision': model['revision'],
            'model_weight_sha256': model['weight_sha256'], 'versions': plan['versions'],
            'method': job['method'], 'method_protocol': plan['methods'][f"{job['model']}/{job['method']}/{job['ratio']}"],
            'selected_content_sha256': io.digest(samples), 'sample_ids': [r['id'] for _, r in samples],
            'sample_indices': [i for i, _ in samples], 'dtype': 'bfloat16', 'max_generated_tokens_per_block': 256}
