"""A later seed must not start because earlier work merely failed or was launched."""
from contextlib import nullcontext
import json
from types import SimpleNamespace

import pytest

from scripts.webbench_seed_queue_v1 import SEED_ORDER, current_stage, queue, stages


@pytest.fixture
def jobs():
    return [dict(name=f'{purpose}/{seed}/{condition}', purpose=purpose, seed=seed)
            for purpose, seeds in [('smoke', (233,)), ('main', SEED_ORDER)]
            for seed in seeds for condition in range(64)]


def test_all_smoke_conditions_must_complete_before_first_seed(jobs):
    smoke = [j['name'] for j in jobs if j['purpose'] == 'smoke']
    assert current_stage(jobs, set(smoke[:-1]))[0] == 'smoke'
    name, group = current_stage(jobs, set(smoke))
    assert name == 'seed233'
    assert len(group) == 64 and {j['seed'] for j in group} == {233}


@pytest.mark.parametrize('seed,following', [(233, 'seed42'), (42, 'seed3407')])
def test_next_seed_requires_every_condition_complete(jobs, seed, following):
    groups = stages(jobs)
    index = next(i for i, (name, _) in enumerate(groups) if name == f'seed{seed}')
    complete = {j['name'] for _, group in groups[:index] for j in group}
    current = groups[index][1]
    complete.update(j['name'] for j in current[:-1])
    # The remaining job may be pending, active, or failed; none counts as complete.
    assert current_stage(jobs, complete)[0] == f'seed{seed}'
    complete.add(current[-1]['name'])
    assert current_stage(jobs, complete)[0] == following


def test_later_results_do_not_unlock_an_earlier_failed_seed(jobs):
    complete = {j['name'] for j in jobs}
    failed = next(j for j in jobs if j['purpose'] == 'main' and j['seed'] == 233)
    complete.remove(failed['name'])
    assert current_stage(jobs, complete)[0] == 'seed233'


def test_complete_matrix_and_unknown_seed_are_checked(jobs):
    assert current_stage(jobs, {j['name'] for j in jobs}) == ('complete', [])
    jobs.append(dict(name='extra', purpose='main', seed=999))
    with pytest.raises(ValueError, match='exactly one'):
        stages(jobs)


@pytest.mark.parametrize('failed_stage,expected', [('smoke', 'qualification_failed'),
                                                 ('seed233', 'seed_failed')])
def test_queue_stops_on_recorded_failure_without_admitting_later_work(tmp_path, jobs,
                                                                    failed_stage, expected):
    def read(path):
        return json.loads(path.read_text())

    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def forbidden_admission():
        pytest.fail('A later stage reached GPU admission despite an earlier failure')

    for name, group in stages(jobs):
        for index, job in enumerate(group):
            state = ('failed' if name == failed_stage and index == 63 else
                     'complete_with_environment_failures' if name == 'seed233' and index == 0 else
                     'complete')
            write(tmp_path / 'jobs' / job['name'] / 'status.json', {'status': state})
        if name == failed_stage:
            break
    validated = []
    actor = SimpleNamespace(read=read, write=write, lock=lambda path: nullcontext(),
                            identity=lambda pid: {'pid': pid, 'start_ticks': '1'},
                            validate_completed=lambda plan, job, path: validated.append(job['name']),
                            gpu_inventory=forbidden_admission)
    plan = {'output': str(tmp_path), 'jobs': jobs, 'sha256': 'frozen',
            'devices': [4], 'max_workers': 8, 'slots_per_device': 2}
    contract = {'sha256': 'contract', 'scheduler_snapshot': 'frozen-scheduler'}
    queue(actor, plan, tmp_path / 'plan.json', contract)
    state = read(tmp_path / 'queue_status.json')
    assert state['status'] == expected
    assert state['current_stage'] == failed_stage
    assert state['later_stages_blocked_by_failure']
    assert len(state['failures']) == 1
    assert len(validated) == (63 if failed_stage == 'smoke' else 127)
