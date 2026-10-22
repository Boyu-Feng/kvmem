"""CPU checks for frozen result bindings, resource admission, and actor budgets."""
from types import SimpleNamespace

import pytest

from scripts import webbench_v1 as runner


@pytest.fixture
def completed(tmp_path):
    job = next(job for job in runner.jobs()
               if job['benchmark'] == 'webwalkerqa' and job['method'] == 'fullkv'
               and job['model'] == 'qwen25_7b' and job['purpose'] == 'smoke')
    tasks = []
    for index in range(2):
        task = {'id': str(index), 'question': f'Question {index}',
                'answer': f'Answer {index}', 'root_url': 'https://example.org/'}
        task['task_sha256'] = runner.digest(task)
        tasks.append(task)
    plan = {'sha256': 'frozen-plan',
            'tasks': {job['benchmark']: {str(job['seed']): {'smoke': tasks}}}}
    rows = []
    for index, task in enumerate(tasks):
        audit_name = f'audits/{index:04d}.json'
        runner.write(tmp_path / audit_name,
                     {'plan_sha256': plan['sha256'], 'job': job,
                      'task_sha256': task['task_sha256'], 'actor_calls': [{'input_ids': [1]}]})
        rows.append({'id': task['id'], 'question': task['question'],
                     'gold_answer': task['answer'], 'task_sha256': task['task_sha256'],
                     'predicted_answer': 'model answer', 'status': 'completed',
                     'audit': audit_name, 'audit_sha256': runner.file_hash(tmp_path / audit_name)})
    predictions = {'plan_sha256': plan['sha256'], 'job': job, 'results': rows}
    runner.write(tmp_path / 'predictions.json', predictions)
    assert runner.validate_completed(plan, job, tmp_path) == predictions
    return plan, job, tmp_path, predictions


@pytest.mark.parametrize('field,value', [('method', 'stepkv'), ('ratio', .2)])
def test_completed_results_cannot_be_reused_across_conditions(completed, field, value):
    plan, job, directory, predictions = completed
    predictions['job'] = {**job, field: value}
    runner.write(directory / 'predictions.json', predictions)
    with pytest.raises(RuntimeError, match='mismatched result'):
        runner.validate_completed(plan, job, directory)


@pytest.mark.parametrize('binding', ['job', 'plan_sha256', 'task_sha256'])
def test_valid_audit_hash_does_not_allow_cross_condition_or_task_reuse(completed, binding):
    plan, job, directory, predictions = completed
    row = predictions['results'][0]
    audit_path = directory / row['audit']
    audit = runner.read(audit_path)
    audit[binding] = {**job, 'method': 'stepkv'} if binding == 'job' else 'different-binding'
    runner.write(audit_path, audit)
    row['audit_sha256'] = runner.file_hash(audit_path)
    runner.write(directory / 'predictions.json', predictions)
    with pytest.raises(RuntimeError, match='different task/condition'):
        runner.validate_completed(plan, job, directory)


def test_same_task_id_and_hash_cannot_hide_changed_gold(completed):
    plan, job, directory, predictions = completed
    predictions['results'][0]['gold_answer'] = 'new unreviewed answer'
    runner.write(directory / 'predictions.json', predictions)
    with pytest.raises(RuntimeError, match='Task order/content changed'):
        runner.validate_completed(plan, job, directory)


def test_one_unavailable_smoke_root_cannot_qualify_a_method(completed):
    plan, job, directory, predictions = completed
    predictions['results'][1].update(status='environment_failure', inference_started=False,
                                    predicted_answer=None, score=None)
    runner.write(directory / 'predictions.json', predictions)
    with pytest.raises(RuntimeError, match='Unavailable smoke roots'):
        runner.validate_completed(plan, job, directory)


@pytest.mark.parametrize('free,used,worker_gpu,admit', [
    (100, None, 4, False),  # Launched worker has not appeared in nvidia-smi yet.
    (132, None, 4, True),   # Two full 50 GiB reservations plus 32 GiB headroom.
    (111, 20, 4, False),
    (112, 20, 4, True),    # Existing allocation and its remaining 30 GiB are both counted.
    (82, None, 5, True),   # A worker on another device uses none of this device's reserve.
])
def test_loading_worker_memory_is_reserved(free, used, worker_gpu, admit):
    plan = {'worker_allocator_gib': 48, 'driver_overhead_gib': 2,
            'external_headroom_gib': 32}
    active = {'loading': {'identity': {'pid': 123, 'start_ticks': '1'}, 'gpu': worker_gpu}}
    inventory = {'free_gib': free, 'process_gib': {} if used is None else {123: used}}
    assert runner.can_admit(plan, active, 4, inventory) is admit


class TinyTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return 'render:' + messages[-1]['content']

    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=list(text.encode()))


class TinyActor:
    def __init__(self):
        self.tokenizer = TinyTokenizer()
        self.calls = []

    def generate_first(self, text, *args, **kwargs):
        self.calls.append(('first', text))
        return 'Action 1: Search[public]', None

    def generate_incremental(self, text, *args, **kwargs):
        self.calls.append(('incremental', text))
        return 'Action 2: Finish[answer]'


def test_context_cap_stops_before_inference_and_restores_original_methods():
    actor = TinyActor()
    initial_ids = list(b'render:q')
    allowance = len(initial_ids) + 256
    boundary = runner.EpisodeBoundary(actor, {'logical_context_tokens': allowance})
    with pytest.raises(runner.ContextBudgetReached):
        with boundary:
            actor.generate_first('q', 256)
            actor.generate_incremental('x', 256)
    assert actor.calls == [('first', 'q')]
    assert boundary.calls[0]['input_ids'] == initial_ids
    assert boundary.logical_tokens == allowance
    assert boundary.exhausted and boundary.exceptions == []
    assert 'generate_first' not in actor.__dict__
    assert 'generate_incremental' not in actor.__dict__
    assert actor.generate_incremental('after boundary') == 'Action 2: Finish[answer]'


def test_generation_exception_is_recorded_and_method_restored():
    actor = TinyActor()

    def fail(*args, **kwargs):
        raise RuntimeError('actual model failure')

    actor.generate_incremental = fail
    boundary = runner.EpisodeBoundary(actor, {'logical_context_tokens': 1000})
    with pytest.raises(RuntimeError, match='actual model failure'):
        with boundary:
            actor.generate_incremental('evidence')
    assert boundary.exceptions == [{'type': 'RuntimeError', 'error': 'actual model failure'}]
    assert actor.generate_incremental is fail
    assert 'generate_first' not in actor.__dict__
    assert not boundary.exhausted
