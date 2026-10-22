"""Preserve running BrowseComp while replacing every WebWalker condition."""
import copy
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import webbench_seed_queue_v4 as scheduler


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def read(path):
    return json.loads(Path(path).read_text())


@pytest.mark.parametrize('version', ['v2', 'v3'])
def test_real_dual_snapshot_actor_imports_bind_new_helper_without_changing_original(tmp_path, version):
    """Exercise the actual dynamic parser import with an older helper-free package."""
    root = Path(scheduler.__file__).resolve().parents[1]
    old, repaired = tmp_path / 'old_snapshot', tmp_path / 'repaired_snapshot'
    for snapshot in (old, repaired):
        (snapshot / 'scripts').mkdir(parents=True)
        shutil.copy2(root / 'scripts/webbench_v1.py', snapshot / 'scripts/webbench_v1.py')
    source = f'scripts/webbench_action_format_{version}.py'
    helper = repaired / source
    shutil.copy2(root / source, helper)
    expected = hashlib.sha256(helper.read_bytes()).hexdigest()
    program = r'''
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('isolated_scheduler_v4', sys.argv[1])
scheduler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduler)
old_path, repaired_path = map(Path, sys.argv[2:4])
old = scheduler.load_actor_module(old_path)
repaired = scheduler.load_actor_module(repaired_path, repaired=True)
version = sys.argv[5]
source = f'scripts/webbench_action_format_{version}.py'
name = f'scripts.webbench_action_format_{version}'
plan = {'source_hashes': {source: sys.argv[4]},
        'action_format': {'webwalkerqa': {'version': f'webwalker_action_format_{version}'}}}
job = {'benchmark': 'webwalkerqa'}
bare = 'Thought 1: Follow the visible link.\nAction 1: Visit https://example.org/page'
strict = 'Thought 1: Follow the visible link.\nAction 1: Visit[https://example.org/page]'
assert old.parse_action(bare, 1)[1:] == (None, None)
assert old.parse_action(strict, 1)[1:] == ('visit', 'https://example.org/page')
try:
    repaired.action_interface(plan, job)
except ModuleNotFoundError as exc:
    assert exc.name == name
else:
    raise AssertionError('Test did not reproduce the separate-snapshot import boundary')

helper = scheduler.load_action_format_helper(repaired_path, plan)
parser, metadata = repaired.action_interface(plan, job)
assert parser(bare, 1)[1:] == ('visit', 'https://example.org/page')
assert metadata(bare, 1)['normalized'] is True
if version == 'v3':
    search = 'Thought 12: Find details.\nAction 12: Search new feature'
    finish = 'Thought 13: Submit the answer.\nAction 13: Finish released in 2024'
    assert parser(search, 12)[1:] == ('search', 'new feature')
    assert parser(finish, 13)[1:] == ('finish', 'released in 2024')
    assert metadata(search, 12)['normalized'] is True
    assert metadata(finish, 13)['normalized'] is True
    assert 'scripts.webbench_action_format_v2' not in sys.modules
assert parser(strict, 1) == old.parse_action(strict, 1)
assert old.parse_action(bare, 1)[1:] == (None, None)
original_parser, original_metadata = old.action_interface({}, job)
assert original_parser(bare, 1)[1:] == (None, None) and original_metadata is None
bcp_parser, bcp_metadata = repaired.action_interface(plan, {'benchmark': 'browsecomp_plus'})
assert bcp_parser(bare, 1)[1:] == (None, None) and bcp_metadata is None
assert Path(old.__file__).resolve() == old_path / 'scripts/webbench_v1.py'
assert Path(repaired.__file__).resolve() == repaired_path / 'scripts/webbench_v1.py'
assert Path(helper.__file__).resolve() == repaired_path / source
assert helper._webbench_frozen_sha256 == sys.argv[4]
assert list(sys.modules['scripts'].__path__) == [str(old_path / 'scripts')]
assert scheduler.load_action_format_helper(repaired_path, plan) is helper

sys.modules[name] = SimpleNamespace(__file__=str(old_path / source))
try:
    scheduler.load_action_format_helper(repaired_path, plan)
except RuntimeError as exc:
    assert 'another source' in str(exc)
else:
    raise AssertionError('A previously imported helper from another snapshot was trusted')
sys.modules[name] = helper
(repaired_path / source).write_text('# changed after freeze\n')
try:
    scheduler.load_action_format_helper(repaired_path, plan)
except RuntimeError as exc:
    assert 'not bound' in str(exc)
else:
    raise AssertionError('An unbound helper change was accepted')
print(json.dumps({'action_interface': 'verified', 'old_package_path': 'preserved', 'helper_hash': sys.argv[4]}))
'''
    result = subprocess.run([sys.executable, '-B', '-c', program, scheduler.__file__, str(old),
                             str(repaired), expected, version], cwd=tmp_path, capture_output=True, text=True,
                            env={key: value for key, value in os.environ.items() if key != 'PYTHONPATH'})
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout)['helper_hash'] == expected


@pytest.mark.parametrize('version', [None, 'webwalker_action_format_v4', '../webbench_v1'])
def test_helper_loader_rejects_unknown_version_without_importing(tmp_path, version):
    plan = {'action_format': {'webwalkerqa': {'version': version}}, 'source_hashes': {}}
    with pytest.raises(RuntimeError, match='Unknown frozen WebWalker action-format version'):
        scheduler.load_action_format_helper(tmp_path, plan)


@pytest.fixture
def plans(tmp_path):
    jobs = [dict(name=f'{benchmark}/{purpose}/{seed}/condition{index}', benchmark=benchmark,
                 purpose=purpose, seed=seed, method=f'method{index}', model='frozen', ratio=.2)
            for benchmark in scheduler.BENCHMARKS
            for purpose, seeds in [('smoke', (233,)), ('main', scheduler.SEEDS)]
            for seed in seeds for index in range(32)]
    old = dict(jobs=jobs, tasks={'webwalkerqa': {'frozen': 'same ordered questions'}},
               models={'frozen': 'same model'}, versions={'python': '3.12'}, budget={'steps': 40},
               prompts={'webwalkerqa': 'same prompt'}, methods={j['name']: {'same': 'configuration'} for j in jobs},
               sha256='original-plan', output=str(tmp_path / 'original'), devices=[4],
               max_workers=8, slots_per_device=2)
    new = copy.deepcopy(old)
    new.update(jobs=[j for j in jobs if j['benchmark'] == 'webwalkerqa'],
               sha256='repaired-plan', output=str(tmp_path / 'repaired'))
    return old, new


def test_route_partition_excludes_all_original_webwalker_conditions(plans):
    old, new = plans
    conditions = scheduler.routed_jobs(old, new)
    assert len(conditions) == 256
    assert sum(item['route'] == 'browsecomp_plus' for item in conditions.values()) == 128
    assert sum(item['route'] == 'webwalkerqa' for item in conditions.values()) == 128
    assert all(item['job'] in new['jobs'] for item in conditions.values() if item['route'] == 'webwalkerqa')


@pytest.mark.parametrize('key', ['budget', 'models', 'prompts', 'tasks'])
def test_parser_only_repair_cannot_change_other_frozen_contracts(plans, key):
    old, new = plans
    if key in ('tasks', 'prompts'):
        new[key]['webwalkerqa'] = 'changed'
    else:
        new[key] = {'changed': True}
    with pytest.raises(ValueError):
        scheduler.routed_jobs(old, new)


def test_webwalker_smoke_has_priority_while_browsecomp_seed233_remains_eligible(plans):
    conditions = scheduler.routed_jobs(*plans)
    complete = {name for name, item in conditions.items()
                if item['route'] == 'browsecomp_plus' and item['job']['purpose'] == 'smoke'}
    seed, qualified, eligible = scheduler.eligibility(conditions, complete, set())
    pending = [job for job in eligible if job['name'] not in complete]
    assert seed == 233 and qualified == {'browsecomp_plus': True, 'webwalkerqa': False}
    assert all(job['benchmark'] == 'webwalkerqa' and job['purpose'] == 'smoke' for job in pending[:32])
    assert len([j for j in pending if j['benchmark'] == 'browsecomp_plus' and j['purpose'] == 'main']) == 32
    assert not any(j['benchmark'] == 'webwalkerqa' and j['purpose'] == 'main' for j in pending)


def test_seed_barrier_spans_both_plans_and_both_actor_and_score_results(plans):
    conditions = scheduler.routed_jobs(*plans)
    all_smoke = {name for name, item in conditions.items() if item['job']['purpose'] == 'smoke'}
    seed233 = {name for name, item in conditions.items()
               if item['job']['purpose'] == 'main' and item['job']['seed'] == 233}
    last = next(name for name in seed233 if name.startswith('webwalkerqa'))
    assert scheduler.eligibility(conditions, all_smoke | seed233, seed233 - {last})[0] == 233
    assert scheduler.eligibility(conditions, all_smoke | (seed233 - {last}), seed233)[0] == 233
    assert scheduler.eligibility(conditions, all_smoke | seed233, seed233)[0] == 42


def test_repaired_smoke_prioritizes_llama_fullkv_then_remaining_llama(plans):
    for job in plans[0]['jobs']:
        if job['benchmark'] == 'webwalkerqa':
            index = int(job['name'].rsplit('condition', 1)[1])
            job['model'] = 'qwen25_7b' if index < 16 else 'llama31_8b'
            job['method'] = 'fullkv' if index in (0, 16) else f'method{index}'
    conditions = scheduler.routed_jobs(*plans)
    _, _, eligible = scheduler.eligibility(conditions, set(), set())
    smoke = eligible[:32]
    assert (smoke[0]['model'], smoke[0]['method']) == ('llama31_8b', 'fullkv')
    assert all(job['model'] == 'llama31_8b' for job in smoke[:16])
    assert all(job['model'] == 'qwen25_7b' for job in smoke[16:])


def make_routes(plans, actor):
    return {benchmark: dict(actor=actor, plan=plan, path=Path(plan['output']) / 'plan.json',
                            output=Path(plan['output']), snapshot=Path(plan['output']) / 'source_snapshot',
                            scoring={'queue_contract': 'contract', 'queue_contract_sha256': 'contract-hash',
                                     'assets': benchmark + '-assets', 'qrels': 'same-qrels'})
            for benchmark, plan in zip(scheduler.BENCHMARKS, plans, strict=True)}


@pytest.mark.parametrize('location', ['previous_status', 'copied_launch'])
def test_superseded_webwalker_worker_or_launch_cannot_be_adopted(plans, location):
    old, repaired = plans
    job = next(j for j in repaired['jobs'] if j['purpose'] == 'smoke')
    record = dict(identity={'pid': 111, 'start_ticks': '1'}, gpu=4, gpu_uuid='gpu4', job=job,
                  route='webwalkerqa', worker_plan_sha256='superseded-r03-plan',
                  worker_output='superseded-r03-output')
    if location == 'previous_status':
        path = Path(old['output']) / 'queue_status.json'
        value = {'plan_sha256': old['sha256'], 'active': {job['name']: record}}
        message = 'superseded plan'
    else:
        path = Path(repaired['output']) / 'jobs' / job['name'] / 'launch.json'
        value = record
        message = 'another frozen plan'
    write(path, value)
    actor = SimpleNamespace(read=read, write=write, lock=lambda path: nullcontext(),
                            identity=lambda pid: {'pid': pid, 'start_ticks': '1'})
    routes = make_routes(plans, actor)
    with pytest.raises(RuntimeError, match=message):
        scheduler.queue(routes, scheduler.routed_jobs(*plans), SimpleNamespace(),
                        {'sha256': 'contract', 'scheduler_snapshot': 'frozen-v4'})
    assert read(path) == value


def test_finished_superseded_worker_is_ignored_and_its_launch_is_preserved(plans, monkeypatch):
    old, repaired = plans
    job = next(j for j in repaired['jobs'] if j['purpose'] == 'smoke')
    record = dict(identity={'pid': 111, 'start_ticks': '1'}, gpu=4, gpu_uuid='gpu4', job=job,
                  route='webwalkerqa', worker_plan_sha256='superseded-r03-plan',
                  worker_output=str(Path(old['output']).parent / 'r03'))
    old_launch = Path(record['worker_output']) / 'jobs' / job['name'] / 'launch.json'
    write(old_launch, record)
    write(Path(old['output']) / 'queue_status.json',
          {'plan_sha256': old['sha256'], 'active': {job['name']: record}})
    actor = SimpleNamespace(read=read, write=write, lock=lambda path: nullcontext(),
                            identity=lambda pid: {'pid': pid, 'start_ticks': 'new-generation'},
                            file_hash=lambda path: 'contract-hash',
                            gpu_inventory=lambda: {4: {'free_gib': 0}}, can_admit=lambda *args: False)

    def stop(seconds):
        raise StopIteration

    monkeypatch.setattr(scheduler.time, 'sleep', stop)
    with pytest.raises(StopIteration):
        scheduler.queue(make_routes(plans, actor), scheduler.routed_jobs(*plans), SimpleNamespace(),
                        {'sha256': 'contract', 'scheduler_snapshot': 'frozen-v4'})
    state = read(Path(old['output']) / 'queue_status.json')
    assert not state['active'] and not state['failures']
    assert read(old_launch) == record
    assert not (Path(repaired['output']) / 'jobs' / job['name'] / 'launch.json').exists()


def test_scores_validate_against_each_actual_plan_and_output_once(plans):
    actor = SimpleNamespace(read=read, write=write)
    routes = make_routes(plans, actor)
    conditions = scheduler.routed_jobs(*plans)
    chosen = [next(item['job'] for item in conditions.values()
                   if item['route'] == benchmark and item['job']['purpose'] == 'main')
              for benchmark in scheduler.BENCHMARKS]
    for job in chosen:
        write(routes[job['benchmark']]['output'] / 'jobs' / job['name'] / 'scoring/scores.json', {})
    calls = []

    def validate(plan, job, directory, assets, qrels):
        route = routes[job['benchmark']]
        assert plan is route['plan'] and directory == route['output'] / 'jobs' / job['name']
        assert assets == job['benchmark'] + '-assets'
        calls.append(job['name'])
        return {'status': 'incomplete_environment_failures'}

    scorer = SimpleNamespace(validate_scores=validate, TERMINAL_SCORES={'complete', 'incomplete_environment_failures'})
    complete, scored, failures = {j['name'] for j in chosen}, {}, {}
    scheduler.scan_scores(routes, conditions, scorer, complete, scored, failures)
    scheduler.scan_scores(routes, conditions, scorer, complete, scored, failures)
    assert len(calls) == 2 and set(scored) == complete and not failures


def test_coordinator_adopts_browsecomp_and_launches_repaired_smoke_from_its_snapshot(plans, monkeypatch):
    conditions = scheduler.routed_jobs(*plans)
    old, new = plans
    for job in old['jobs']:
        if job['purpose'] == 'smoke':
            write(Path(old['output']) / 'jobs' / job['name'] / 'status.json', {'status': 'complete'})
    running = next(j for j in old['jobs'] if j['benchmark'] == 'browsecomp_plus' and j['purpose'] == 'main')
    old_record = dict(identity={'pid': 111, 'start_ticks': '1'}, gpu=4, gpu_uuid='gpu4', job=running)
    launch_path = Path(old['output']) / 'jobs' / running['name'] / 'launch.json'
    write(launch_path, old_record)
    write(launch_path.parent / 'status.json', {'status': 'running'})
    write(Path(old['output']) / 'queue_status.json',
          {'plan_sha256': old['sha256'], 'active': {running['name']: old_record}})
    calls, validations = [], []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(pid=222, poll=lambda: None)

    def identity(pid):
        return {'pid': pid, 'start_ticks': '1'} if pid in (111, 222, os.getpid()) else None

    def validate(plan, job, directory):
        validations.append((plan['sha256'], job['name'], directory))

    actor = SimpleNamespace(read=read, write=write, lock=lambda path: nullcontext(), identity=identity,
                            file_hash=lambda path: 'contract-hash', validate_completed=validate,
                            gpu_inventory=lambda: {4: {'uuid': 'gpu4', 'free_gib': 140, 'process_gib': {}}},
                            can_admit=lambda *args: True, PYTHON='frozen-python')
    routes = make_routes(plans, actor)
    scorer = SimpleNamespace(TERMINAL_SCORES={'complete'})
    monkeypatch.setattr(scheduler.subprocess, 'Popen', popen)

    def stop(seconds):
        raise StopIteration

    monkeypatch.setattr(scheduler.time, 'sleep', stop)
    contract = {'sha256': 'route-contract', 'scheduler_snapshot': 'frozen-v4'}
    with pytest.raises(StopIteration):
        scheduler.queue(routes, conditions, scorer, contract)
    assert len(calls) == 1
    command, kwargs = calls[0]
    assert command[command.index('--plan') + 1] == str(routes['webwalkerqa']['path'])
    launched = command[command.index('--job') + 1]
    assert launched.startswith('webwalkerqa/smoke/')
    assert kwargs['cwd'] == routes['webwalkerqa']['snapshot']
    assert kwargs['env']['PYTHONPATH'] == str(routes['webwalkerqa']['snapshot'])
    assert read(launch_path) == old_record
    state = read(Path(old['output']) / 'queue_status.json')
    assert state['active'][running['name']]['identity']['pid'] == 111
    assert state['active'][launched]['worker_plan_sha256'] == new['sha256']
    assert state['benchmark_qualified'] == {'browsecomp_plus': True, 'webwalkerqa': False}
    assert len(validations) == 32 and all(name.startswith('browsecomp_plus') for _, name, _ in validations)
    reservations = read(Path(old['output']) / 'actor_reservations.json')
    assert reservations['plan_sha256'] == old['sha256']
    assert set(reservations['active']) == {running['name'], launched}


def test_repaired_smoke_uses_the_stronger_private_actor_validator(plans, monkeypatch):
    old, new = plans
    for plan in plans:
        for job in plan['jobs']:
            if job['purpose'] == 'smoke':
                write(Path(plan['output']) / 'jobs' / job['name'] / 'status.json', {'status': 'complete'})

    def stronger_validator(plan, job, directory):
        assert plan is new and job['benchmark'] == 'webwalkerqa'
        raise RuntimeError('repaired smoke has no successful visible-link visit')

    def stop(seconds):
        raise StopIteration

    actor = SimpleNamespace(read=read, write=write, lock=lambda path: nullcontext(),
                            identity=lambda pid: {'pid': pid, 'start_ticks': '1'},
                            file_hash=lambda path: 'contract-hash', validate_completed=lambda *args: None,
                            gpu_inventory=lambda: {4: {'free_gib': 0}}, can_admit=lambda *args: False)
    routes = make_routes(plans, actor)
    routes['webwalkerqa']['actor'] = SimpleNamespace(read=read, write=write,
                                                    validate_completed=stronger_validator)
    monkeypatch.setattr(scheduler.time, 'sleep', stop)
    with pytest.raises(StopIteration):
        scheduler.queue(routes, scheduler.routed_jobs(*plans), SimpleNamespace(),
                        {'sha256': 'contract', 'scheduler_snapshot': 'frozen-v4'})
    state = read(Path(old['output']) / 'queue_status.json')
    assert state['benchmark_qualified'] == {'browsecomp_plus': True, 'webwalkerqa': False}
    assert len(state['failures']) == 32
    assert all('no successful visible-link visit' in error for error in state['failures'].values())
