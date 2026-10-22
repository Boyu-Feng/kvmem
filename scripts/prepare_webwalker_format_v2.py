"""Derive a fresh WebWalker-only run while preserving the old frozen run."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import time

from scripts import webbench_v1 as p
from scripts.webbench_action_format_v2 import VERSION


def prepare(old_path, output):
    old_path, output = Path(old_path).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('Use a new output; prior experiments cannot be overwritten')
    old = p.read(old_path)
    unsigned = {k: v for k, v in old.items() if k != 'sha256'}
    if p.digest(unsigned) != old['sha256'] or p.versions() != old['versions']:
        raise RuntimeError('Parent plan or actor environment changed')
    parent = Path(old['output'])
    if old_path.parent != parent:
        raise RuntimeError('Parent plan moved from its frozen output')
    for name, expected in old['source_hashes'].items():
        if p.file_hash(parent / 'source_snapshot' / name) != expected:
            raise RuntimeError(f'Parent source changed: {name}')
    # A new benchmark protocol applies to every method and both base models.
    plan = copy.deepcopy(unsigned)
    plan.update(schema='webbench_webwalker_action_format_v2', output=str(output), created=time.time())
    plan['jobs'] = [j for j in plan['jobs'] if j['benchmark'] == 'webwalkerqa']
    plan['methods'] = {j['name']: plan['methods'][j['name']] for j in plan['jobs']}
    if len(plan['jobs']) != 128 or sum(j['purpose'] == 'smoke' for j in plan['jobs']) != 32:
        raise RuntimeError('Expected complete common WebWalker matrix')
    plan['action_format'] = {'webwalkerqa': {
        'version': VERSION,
        'change': 'Deterministic single-line Action N: Visit <absolute http(s) URL> alias; '
                  'strict forms unchanged; no guessed targets or added actor calls.',
        'scope': 'Shared by both base models and every WebWalker method and retention.',
        'audit': 'Raw model text and original/normalized action metadata are both preserved.',
        'smoke_gate': 'Every smoke first action must parse; each condition must record at '
                      'least one successful website visit across its two held-out tasks.',
    }}
    diagnostic = parent / 'diagnostics/ww_llama_smoke_format_20261021.json'
    plan['lineage'] = {
        'parent_plan': str(old_path), 'parent_plan_sha256': old['sha256'],
        'parent_plan_file_sha256': p.file_hash(old_path),
        'user_instruction': 'WebWalker / Llama 的 32 次预检均因动作格式错误提前结束修改了接着启动',
        'diagnostic': str(diagnostic), 'diagnostic_sha256': p.file_hash(diagnostic),
        'supersedes': 'All WebWalker jobs in parent run; no old predictions or smoke '
                      'qualifications are imported. BrowseComp continues under parent plan.',
        'unchanged': ['exact task content and order', 'models and tokenizer', 'prompt',
                      'decoding and context accounting', 'budgets and method configurations',
                      'browser visibility constraints and cached pages', 'official scoring'],
        'reporting': 'Compare methods within this WebWalker protocol only; do not rank '
                     'old WebWalker smoke or parent incomplete rows against this run.',
    }
    for key in ('tasks', 'models', 'assets', 'budget', 'versions', 'runtime_manifests', 'prompts'):
        if plan[key] != old[key]:
            raise RuntimeError(f'Unexpected scientific change: {key}')
    replacements = ('scripts/webbench_v1.py', 'scripts/webbench_action_format_v2.py',
                    'scripts/prepare_webwalker_format_v2.py')
    for name in replacements:
        plan['source_hashes'][name] = p.file_hash(p.ROOT / name)
    plan['sha256'] = p.digest(plan)
    output.mkdir(parents=True, exist_ok=False)
    for name, expected in plan['source_hashes'].items():
        source = p.ROOT / name if name in replacements else parent / 'source_snapshot' / name
        target = output / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        if p.file_hash(target) != expected:
            raise RuntimeError(f'Source changed during freeze: {name}')
    p.write(output / 'plan.json', plan)
    p.write(output / 'shared_evaluation_check.json', {
        'status': 'passed_configuration_check', 'plan_sha256': plan['sha256'],
        'parent_plan_sha256': old['sha256'], 'created': time.time(),
        'instructions': '/home/bruli/fengboyu/cl/ttcl/universal_experience/SHARED_EVALUATION_20261012.md',
        'task_order_and_content_unchanged': p.digest(plan['tasks']) == p.digest(old['tasks']),
        'fresh_fullkv_per_model_and_seed': all(sum(j['method'] == 'fullkv'
            and j['model'] == model and j['seed'] == seed and j['purpose'] == 'main'
            for j in plan['jobs']) == 1 for model in plan['models'] for seed in p.SEEDS),
        'same_actor_interface_for_all_methods': plan['action_format'],
        'formal_conditions': 96, 'formal_episodes': 9600, 'smoke_conditions': 32,
        'gpu_episodes': 0,
        'loaded_model_and_neural_inputs': 'Must be checked in each new worker and smoke audit.',
    })
    p.write(output / 'preflight.json', {'status': 'prepared', 'plan_sha256': plan['sha256'],
            'formal_conditions': 96, 'formal_episodes': 9600, 'source_count': len(plan['source_hashes']),
            'gpu_episodes': 0})
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = prepare(args.old_plan, args.output)
    print(json.dumps({'plan': str(args.output / 'plan.json'), 'sha256': plan['sha256']}))


if __name__ == '__main__':
    main()
