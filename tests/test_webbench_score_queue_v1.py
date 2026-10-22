import os
from types import SimpleNamespace

import pytest

from scripts import webbench_score_queue_v1 as q


def job(seed=233, purpose='main', **changes):
    result = dict(name=f'test/seed{seed}/{purpose}', benchmark='browsecomp_plus',
        model='test', method='fullkv', ratio=1., seed=seed, purpose=purpose)
    result.update(changes)
    return result


def plan(tmp_path):
    return dict(output=str(tmp_path), sha256='frozen-plan', devices=[0, 1, 2],
                jobs=[job(seed) for seed in q.p.SEEDS] + [job(purpose='smoke')])


def test_admission_reads_both_actor_records_and_rejects_pid_reuse(tmp_path, monkeypatch):
    value = plan(tmp_path)
    old = {'pid': 10, 'start_ticks': '100'}
    first = {'pid': 11, 'start_ticks': '110'}
    second = {'pid': 12, 'start_ticks': '120'}
    monkeypatch.setattr(q.p, 'identity', lambda pid: {10: dict(old, start_ticks='200'),
                                                   11: first, 12: second}.get(pid))
    q.p.write(tmp_path / 'actor_reservations.json', dict(plan_sha256=value['sha256'], active={
        'old': dict(identity=old, gpu=2), 'first': dict(identity=first, gpu=0)}))
    q.p.write(tmp_path / 'queue_status.json', dict(plan_sha256=value['sha256'], active={
        'second': dict(identity=second, gpu=1)}))
    active = q.actor_records(value)
    assert {r['identity']['pid'] for r in active} == {11, 12}
    inventory = {0: {'free_gib': 140}, 1: {'free_gib': 130}, 2: {'free_gib': 101}}
    assert q.available_gpu(value, inventory, active) == 2
    inventory[2]['free_gib'] = 99.9
    assert q.available_gpu(value, inventory, active) is None


def test_foreign_live_actor_reservation_is_not_ignored(tmp_path, monkeypatch):
    who = {'pid': 10, 'start_ticks': '100'}
    monkeypatch.setattr(q.p, 'identity', lambda _: who)
    q.p.write(tmp_path / 'actor_reservations.json', dict(plan_sha256='another-plan',
        active={'other': dict(identity=who, gpu=1)}))
    with pytest.raises(RuntimeError, match='another frozen plan'):
        q.actor_records(plan(tmp_path))


def test_three_seeds_required_and_smoke_is_never_aggregated(tmp_path):
    value = plan(tmp_path)
    outcomes = {job(seed)['name']: dict(status='complete', accuracy=acc)
                for seed, acc in zip(q.p.SEEDS, [.3, .4, .5])}
    groups = q.aggregate(value, outcomes)['groups']
    assert len(groups) == 1 and len(groups[0]['seeds']) == 3
    assert groups[0]['mean_accuracy'] == pytest.approx(.4)
    assert groups[0]['sample_sd'] == pytest.approx(.1)
    outcomes[job(42)['name']] = dict(status='incomplete_environment_failures',
                                    accuracy=None, scored_subset_accuracy=.9)
    result = q.aggregate(value, outcomes)['groups'][0]
    assert result['mean_accuracy'] is None and result['sample_sd'] is None
    assert result['status'] == 'missing'
    del outcomes[job(42)['name']]
    assert q.aggregate(value, outcomes)['groups'][0]['mean_accuracy'] is None


def test_release_retains_live_child_then_clears_dead_child(tmp_path, monkeypatch):
    who = {'pid': 42, 'start_ticks': '100'}
    owner = {'pid': os.getpid(), 'start_ticks': '200'}
    identities = {42: who, os.getpid(): owner}
    monkeypatch.setattr(q.p, 'identity', identities.get)
    path = tmp_path / 'scorer_reservation.json'
    q.p.write(path, dict(status='running', owner=None, child=who, gpu=0))
    q.release_reservation(plan(tmp_path))
    assert q.p.read(path)['status'] == 'running'
    assert q.reservation_busy(tmp_path)
    del identities[42]
    q.release_reservation(plan(tmp_path))
    assert q.p.read(path)['status'] == 'released'
    assert q.p.read(path)['owner'] == owner
    assert not q.reservation_busy(tmp_path)


def test_recovers_reserved_child_and_finishes_durable_launch_gate(tmp_path, monkeypatch):
    value = plan(tmp_path)
    who = {'pid': 42, 'start_ticks': '100'}
    owner = {'pid': os.getpid(), 'start_ticks': '200'}
    monkeypatch.setattr(q.p, 'identity', {42: who, os.getpid(): owner}.get)
    record = dict(identity=who, job=job(), gpu=1, plan_sha256=value['sha256'], token='token')
    q.p.write(tmp_path / 'scorer_reservation.json', dict(plan_sha256=value['sha256'],
        status='running', owner={'pid': 999, 'start_ticks': '0'}, child=who, gpu=1, launch_record=record))
    recovered = q.recover(value)
    assert recovered == {job()['name']: record}
    assert q.p.read(tmp_path / 'jobs' / job()['name'] / 'scoring/queue_launch.json') == record
    assert q.p.read(tmp_path / 'scorer_reservation.json')['owner'] == owner


def test_missing_judge_qualification_blocks_without_api_call(tmp_path):
    probe, reason = q.qualification(tmp_path, 'http://localhost:8313/v1')
    assert probe is None and 'No qualified' in reason


def test_score_validation_rejects_tampering_and_partial_zero_filling(tmp_path, monkeypatch):
    if not (q.s.DEFAULT_ASSETS / 'sources/manifest.json').exists():
        pytest.skip('Public scorer sources are not prepared')
    official = q.s.official(q.s.DEFAULT_ASSETS)
    monkeypatch.setattr(q.s, 'official', lambda _: official)
    monkeypatch.setattr(q.p, 'validate_completed', lambda *args: None)
    value = plan(tmp_path)
    value['source_hashes'] = {'scripts/webbench_scoring_v1.py': q.s.file_sha(q.s.__file__)}
    assets = tmp_path / 'assets'
    q.s.atomic(assets / 'model_ready.json', dict(repo=q.s.MODEL_REPO, revision=q.s.MODEL_REVISION))
    directory = tmp_path / 'jobs' / job()['name']
    q.s.atomic(directory / 'predictions.json', {'results': [dict(id='test', question='Question',
        gold_answer='Answer', predicted_answer='', termination='step_budget')]})
    qrels = tmp_path / 'qrels.txt'
    qrels.write_text('')
    args = SimpleNamespace(assets=assets, output=directory / 'scoring',
        predictions=directory / 'predictions.json', benchmark='browsecomp_plus',
        qrels=qrels, endpoint='http://localhost:8313/v1', gpu_memory_utilization=.65,
        max_model_len=16384)
    q.s.score(args)
    assert q.validate_scores(value, job(), directory, assets, qrels)['accuracy'] == 0
    scores_path = directory / 'scoring/scores.json'
    scores = q.s.read(scores_path)
    scores['accuracy'] = 1
    q.s.atomic(scores_path, scores)
    with pytest.raises(RuntimeError, match='summary'):
        q.validate_scores(value, job(), directory, assets, qrels)
    scores['accuracy'] = 0
    scores['results'][0]['input_sha256'] = 'old-history-id-with-different-content'
    q.s.atomic(scores_path, scores)
    with pytest.raises(RuntimeError, match='content/order binding'):
        q.validate_scores(value, job(), directory, assets, qrels)
