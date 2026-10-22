"""Seed advancement requires both bound actor output and official score output."""
from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest

from scripts import webbench_seed_queue_v2 as scheduler


@pytest.fixture
def jobs():
    return [dict(name=f'{purpose}/{seed}/{condition}', purpose=purpose, seed=seed)
            for purpose, seeds in [('smoke', (233,)), ('main', scheduler.SEED_ORDER)]
            for seed in seeds for condition in range(64)]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def read(path):
    return json.loads(path.read_text())


@pytest.mark.parametrize('seed,following', [(233, 'seed42'), (42, 'seed3407')])
def test_all_actors_and_all_scores_are_required_before_next_seed(jobs, seed, following):
    groups = scheduler.stages(jobs)
    index = next(i for i, (name, _) in enumerate(groups) if name == f'seed{seed}')
    complete = {j['name'] for _, group in groups[:index + 1] for j in group}
    scored = {j['name'] for _, group in groups[1:index + 1] for j in group}
    final = groups[index][1][-1]['name']
    assert scheduler.current_stage(jobs, complete, scored - {final})[0] == f'seed{seed}'
    assert scheduler.current_stage(jobs, complete - {final}, scored)[0] == f'seed{seed}'
    assert scheduler.current_stage(jobs, complete, scored)[0] == following


def test_smoke_needs_no_scores_but_all_actors(jobs):
    smoke = [j['name'] for j in jobs if j['purpose'] == 'smoke']
    assert scheduler.current_stage(jobs, set(smoke[:-1]), set())[0] == 'smoke'
    assert scheduler.current_stage(jobs, set(smoke), set())[0] == 'seed233'


@pytest.mark.parametrize('status', ['complete', 'incomplete_environment_failures'])
def test_official_validation_is_cached_and_recorded_missing_outcome_can_advance(tmp_path, jobs, status):
    smoke_and_first = [j for j in jobs if j['purpose'] == 'smoke' or j['seed'] == 233]
    plan = {'output': str(tmp_path), 'sha256': 'frozen', 'jobs': jobs}
    complete = {j['name'] for j in smoke_and_first}
    expected = {j['name'] for j in smoke_and_first if j['purpose'] == 'main'}
    for name in expected:
        write(tmp_path / 'jobs' / name / 'scoring/scores.json', {})
    calls = []

    def validate(plan, job, directory, assets, qrels):
        calls.append(job['name'])
        assert assets == 'original-assets' and qrels == 'original-qrels'
        return {'status': status, 'accuracy': None if status != 'complete' else .5}

    scorer = SimpleNamespace(validate_scores=validate,
                             TERMINAL_SCORES={'complete', 'incomplete_environment_failures'})
    actor = SimpleNamespace(read=read, write=write)
    binding = {'assets': 'original-assets', 'qrels': 'original-qrels'}
    scored, failures = {}, {}
    scheduler.score_progress(actor, scorer, plan, binding, complete, scored, failures)
    scheduler.score_progress(actor, scorer, plan, binding, complete, scored, failures)
    assert len(calls) == 64 and set(scored) == expected and not failures
    assert scheduler.current_stage(jobs, complete, set(scored))[0] == 'seed42'


def test_failed_score_validation_is_preserved_without_automatic_revalidation(tmp_path, jobs):
    job = next(j for j in jobs if j['purpose'] == 'main')
    plan = {'output': str(tmp_path), 'sha256': 'frozen', 'jobs': jobs}
    write(tmp_path / 'jobs' / job['name'] / 'scoring/scores.json', {})
    calls = []

    def validate(*args):
        calls.append(1)
        raise RuntimeError('score belongs to another condition')

    actor = SimpleNamespace(read=read, write=write)
    scorer = SimpleNamespace(validate_scores=validate, TERMINAL_SCORES={'complete'})
    binding = {'assets': 'original-assets', 'qrels': 'original-qrels'}
    scored, failures = {}, {}
    scheduler.score_progress(actor, scorer, plan, binding, {job['name']}, scored, failures)
    # Even a new controller process must preserve the failed validation record.
    scheduler.score_progress(actor, scorer, plan, binding, {job['name']}, {}, {})
    assert calls == [1]
    assert not scored and failures[job['name']] == 'score belongs to another condition'


def test_official_scoring_failure_overrides_previously_cached_success(tmp_path, jobs):
    job = next(j for j in jobs if j['purpose'] == 'main')
    plan = {'output': str(tmp_path), 'sha256': 'frozen', 'jobs': jobs}
    write(tmp_path / 'jobs' / job['name'] / 'scoring/queue_failure.json',
          {'plan_sha256': 'frozen', 'job': job, 'status': 'scoring_failed', 'error': 'recorded failure'})
    scored, failures = {job['name']: {'status': 'complete'}}, {}
    scheduler.score_progress(SimpleNamespace(read=read, write=write), None, plan, {},
                             {job['name']}, scored, failures)
    assert not scored and failures[job['name']] == 'recorded failure'


@pytest.mark.parametrize('failure', [False, True])
def test_controller_waits_for_missing_score_or_blocks_on_failed_score_without_next_seed_gpu(
        tmp_path, jobs, monkeypatch, failure):
    for job in jobs:
        if job['purpose'] == 'smoke' or job['seed'] == 233:
            write(tmp_path / 'jobs' / job['name'] / 'status.json', {'status': 'complete'})
            if job['purpose'] == 'main' and not job['name'].endswith('/63'):
                write(tmp_path / 'jobs' / job['name'] / 'scoring/scores.json', {})
    final = next(j for j in jobs if j['purpose'] == 'main' and j['seed'] == 233
                 and j['name'].endswith('/63'))
    if failure:
        write(tmp_path / 'jobs' / final['name'] / 'scoring/queue_failure.json',
              {'plan_sha256': 'frozen', 'job': final, 'status': 'scoring_failed', 'error': 'failed judge'})

    def forbidden_admission():
        pytest.fail('Seed 42 reached GPU admission before every seed 233 score validated')

    def stop_after_state(seconds):
        raise StopIteration

    monkeypatch.setattr(scheduler.time, 'sleep', stop_after_state)
    actor = SimpleNamespace(read=read, write=write, lock=lambda path: nullcontext(),
                            file_hash=lambda path: 'score-contract-hash',
                            identity=lambda pid: {'pid': pid, 'start_ticks': '1'},
                            validate_completed=lambda *args: None, gpu_inventory=forbidden_admission)
    scorer = SimpleNamespace(validate_scores=lambda *args: {'status': 'complete'},
                             TERMINAL_SCORES={'complete', 'incomplete_environment_failures'})
    plan = {'output': str(tmp_path), 'jobs': jobs, 'sha256': 'frozen',
            'devices': [4], 'max_workers': 8, 'slots_per_device': 2}
    contract = {'sha256': 'contract', 'scheduler_snapshot': 'v2-frozen',
                'scoring': {'queue_contract': 'original-contract',
                            'queue_contract_sha256': 'score-contract-hash',
                            'assets': 'original-assets', 'qrels': 'original-qrels'}}
    if failure:
        scheduler.queue(actor, scorer, plan, tmp_path / 'plan.json', contract)
    else:
        with pytest.raises(StopIteration):
            scheduler.queue(actor, scorer, plan, tmp_path / 'plan.json', contract)
    state = read(tmp_path / 'queue_status.json')
    assert state['current_stage'] == 'seed233'
    assert state['status'] == ('seed_failed' if failure else 'waiting_for_scoring')
    assert state['pending_scoring'] == (0 if failure else 1)
    assert state['scored_conditions'] == 63
    assert bool(state['scoring_failures']) is failure
