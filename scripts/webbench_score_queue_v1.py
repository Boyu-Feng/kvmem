"""Durable main-condition scoring queue for the frozen WebBench comparison.

Run this controller with the actor Python and from the frozen source snapshot.
The separate pinned judge Python owns all scoring children. Failed conditions
are preserved, never automatically retried, and never included as zero scores.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid

from scripts import webbench_v1 as p
from scripts import webbench_scoring_v1 as s

JUDGE_PYTHON = p.BASE / 'runtime/judge_env/bin/python'
TERMINAL_SCORES = {'complete', 'incomplete_environment_failures'}
ACTIVE_RESERVATIONS = {'admitting', 'running'}


@contextlib.contextmanager
def resource_lock(output):
    """Block while another controller publishes its resource admission."""
    path = Path(output) / 'resource_admission.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def live(who):
    return bool(who and p.identity(who['pid']) == who)


def main_jobs(plan):
    return [job for job in plan['jobs'] if job['purpose'] == 'main']


def actor_records(plan):
    """Both durable records cover actor launch and queue-publication windows."""
    active = {}
    for name in ('actor_reservations.json', 'queue_status.json'):
        path = Path(plan['output']) / name
        if not path.exists():
            continue
        value = p.read(path)
        records = {k: v for k, v in value.get('active', {}).items() if live(v.get('identity'))}
        if records and value.get('plan_sha256') != plan['sha256']:
            raise RuntimeError('Live actor reservations belong to another frozen plan')
        for key, record in records.items():
            # Keep both identities if interrupted publication left two records.
            active[(key, record['identity']['pid'], record['identity']['start_ticks'])] = record
    return list(active.values())


def available_gpu(plan, inventory, actors):
    occupied = {r['gpu'] for r in actors}
    candidates = [gpu for gpu in plan['devices'] if gpu not in occupied
                  and inventory.get(gpu, {}).get('free_gib', 0) >= 100]
    return max(candidates, key=lambda gpu: inventory[gpu]['free_gib']) if candidates else None


def reservation_busy(output):
    path = Path(output) / 'scorer_reservation.json'
    value = p.read(path) if path.exists() else {}
    return value.get('status') in ACTIVE_RESERVATIONS and (
        live(value.get('child')) or live(value.get('owner')))


def qualification(assets, endpoint):
    path = Path(assets) / 'webwalker_probe.json'
    if not path.exists():
        return None, 'No qualified and authorized WebWalker judge probe is available'
    try:
        probe, _ = s.validate_webwalker_probe(assets, endpoint)
        return probe, None
    except Exception as exc:
        return None, str(exc)


def validate_scores(plan, job, directory, assets, qrels):
    """Reject plausible-looking scores without complete immutable input bindings."""
    p.validate_completed(plan, job, directory)
    path, rows = s.load_predictions(directory / 'predictions.json')
    output = directory / 'scoring'
    contract = s.read(output / 'contract.json')
    scores = s.read(output / 'scores.json')
    _, _, source = s.official(assets)
    expected = dict(benchmark=job['benchmark'], version=s.VERSION,
        predictions_path=str(path), predictions_sha256=s.file_sha(path),
        rows_sha256=s.digest(rows), ids=[r['id'] for r in rows],
        scorer_sha256=plan['source_hashes']['scripts/webbench_scoring_v1.py'],
        sources_sha256=s.digest(source), batch_size=1)
    runtime_path = Path(plan['output']) / 'score_queue_contract.json'
    if runtime_path.exists():
        frozen_runtime = p.read(runtime_path)['judge_runtime_manifest']
        if s.file_sha(frozen_runtime['path']) != frozen_runtime['sha256']:
            raise RuntimeError('Pinned judge runtime manifest changed')
        expected.update(runtime_manifest=frozen_runtime,
                        runtime=s.read(frozen_runtime['path'])['runtime'])
    if job['benchmark'] == 'browsecomp_plus':
        expected.update(sampling=s.BCP_SAMPLING, judge_model=s.read(Path(assets) / 'model_ready.json'),
            qrels_sha256=s.file_sha(qrels), gpu_memory_utilization=.65,
            max_model_len=16384, engine_seed=0, tensor_parallel_size=1, judge_attempt_budget=1)
    else:
        probe, error = qualification(assets, contract['endpoint'])
        if error:
            raise RuntimeError(error)
        policy = s.webwalker_policy(assets)
        expected.update(sampling=policy['sampling'], webwalker_policy=policy,
                        expected_api_model=probe['returned_model'],
                        judge_model=None, qrels_sha256=None, judge_attempt_budget=10)
    if any(contract.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Score contract differs from the frozen prediction/scorer contract')
    if scores.get('contract_sha256') != s.digest(contract) or len(scores.get('results', [])) != len(rows):
        raise RuntimeError('Score result count/contract differs')
    for index, (row, result) in enumerate(zip(rows, scores['results'], strict=True)):
        if (s.read(output / 'items' / f'{index:04d}.json') != result
                or result.get('contract_sha256') != s.digest(contract)
                or result.get('input_sha256') != s.digest(row)
                or result.get('source_row_sha256') != row['source_row_sha256']
                or result.get('id') != row['id']):
            raise RuntimeError('Scored item content/order binding differs')
        if row['environment_failed']:
            if result.get('score') is not None or result.get('status') != 'environment_failure':
                raise RuntimeError('Environment failures must remain missing')
        elif result.get('score') not in (0, 1):
            raise RuntimeError('Missing/nonbinary score for an inferred question')
    scored = [r for r in scores['results'] if r['score'] is not None]
    missing, correct = len(rows) - len(scored), sum(r['score'] for r in scored)
    expected_summary = dict(benchmark=job['benchmark'], total=len(rows), requested=len(rows),
        inferred=sum(r['inference_started'] for r in rows), scored=len(scored),
        environment_failed=missing, correct=correct,
        accuracy=None if missing else correct / len(rows),
        scored_subset_accuracy=correct / len(scored) if scored else None,
        status='incomplete_environment_failures' if missing else 'complete')
    if any(scores.get(k) != v for k, v in expected_summary.items()):
        raise RuntimeError('Score summary does not match bound per-question results')
    return expected_summary


def aggregate(plan, outcomes):
    groups = {}
    for job in main_jobs(plan):
        key = (job['benchmark'], job['model'], job['method'], job['ratio'])
        group = groups.setdefault(key, dict(benchmark=key[0], model=key[1], method=key[2],
            ratio=key[3], expected_seeds=sorted(p.SEEDS), seeds={}))
        result = outcomes.get(job['name'], {'status': 'pending_actor'})
        group['seeds'][str(job['seed'])] = dict(result, job=job['name'])
    for group in groups.values():
        complete = all(group['seeds'].get(str(seed), {}).get('status') == 'complete'
                       for seed in p.SEEDS)
        values = [group['seeds'][str(seed)]['accuracy'] for seed in p.SEEDS] if complete else []
        group.update(status='complete' if complete else 'missing',
                     mean_accuracy=statistics.mean(values) if values else None,
                     sample_sd=statistics.stdev(values) if len(values) > 1 else None)
    qualification_path = Path(plan['output']) / 'webwalker_qualification.json'
    probe = p.read(qualification_path) if qualification_path.exists() else None
    return dict(plan_sha256=plan['sha256'], updated_unix=time.time(),
        judges=dict(browsecomp_plus=dict(model=s.MODEL_REPO, revision=s.MODEL_REVISION),
            webwalkerqa=dict(status='qualified', actual_model=probe['returned_model'],
                             policy=probe['policy']) if probe else dict(status='unqualified')),
        aggregation='Unweighted across-seed mean and sample SD only when all three seeds are complete',
        incomplete_conditions_are_not_zero=True, groups=list(groups.values()))


def write_reports(plan, outcomes):
    summary = aggregate(plan, outcomes)
    output = Path(plan['output'])
    p.write(output / 'SUMMARY.json', summary)
    lines = ['# WebBench shared comparison', '', f"Frozen plan: `{plan['sha256']}`", '',
        f'BrowseComp-Plus judge: {s.MODEL_REPO}, revision {s.MODEL_REVISION}.',
        'WebWalkerQA judge: ' + (summary['judges']['webwalkerqa']['actual_model'] + ' (' +
            summary['judges']['webwalkerqa']['policy']['label'] + ').'
            if summary['judges']['webwalkerqa']['status'] == 'qualified' else 'not qualified.'), '',
        'Accuracy is reported only for complete conditions. Environment and infrastructure failures remain missing.',
        'Across-seed mean and sample SD require all three seeds. No partial-result rankings are produced.', '',
        '| Benchmark | Model | Method | Retained | Seed 233 | Seed 42 | Seed 3407 | Mean ± sample SD |',
        '|---|---|---|---|---|---|---|---|']
    for group in summary['groups']:
        cells = []
        for seed in p.SEEDS:
            item = group['seeds'].get(str(seed), {})
            cells.append(f"{100 * item['accuracy']:.2f}%" if item.get('status') == 'complete'
                         else item.get('status', 'missing'))
        mean = (f"{100 * group['mean_accuracy']:.2f}% ± {100 * group['sample_sd']:.2f}%"
                if group['status'] == 'complete' else 'missing')
        ratio = 'adaptive' if group['ratio'] is None else f"{100 * group['ratio']:g}%"
        lines.append('| ' + ' | '.join([group['benchmark'], group['model'], group['method'], ratio, *cells, mean]) + ' |')
    temporary = output / f'RESULTS.md.{os.getpid()}.tmp'
    temporary.write_text('\n'.join(lines) + '\n')
    os.replace(temporary, output / 'RESULTS.md')


def failure(plan, job, message, **details):
    directory = Path(plan['output']) / 'jobs' / job['name'] / 'scoring'
    record = dict(status='scoring_failed', plan_sha256=plan['sha256'], job=job,
                  error=str(message), updated_unix=time.time(), **details)
    path = directory / 'queue_failure.json'
    if not path.exists():
        p.write(path, record)
    return p.read(path)


def launch_guard(record_path, token):
    """No GPU is touched until the parent durably publishes child admission."""
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        path = Path(record_path)
        if path.exists():
            record = p.read(path)
            if record.get('token') == token and record['identity'] == p.identity(os.getpid()):
                os.execvpe(record['command'][0], record['command'], os.environ)
        time.sleep(.1)
    raise RuntimeError('Scorer launch was not durably admitted; no judge was invoked')


def launch(plan, job, args, qrels, gpu=None, inventory=None):
    output = Path(plan['output'])
    directory = output / 'jobs' / job['name']
    score_output = directory / 'scoring'
    score_output.mkdir(parents=True, exist_ok=True)
    snapshot = output / 'source_snapshot'
    command = [str(args.judge_python), '-B', '-u', '-m', 'scripts.webbench_scoring_v1', 'score',
        '--benchmark', job['benchmark'], '--predictions', str(directory / 'predictions.json'),
        '--output', str(score_output), '--assets', str(args.assets), '--endpoint', args.endpoint,
        '--runtime-manifest', str(args.runtime_manifest),
        '--gpu-memory-utilization', '.65', '--max-model-len', '16384']
    if job['benchmark'] == 'browsecomp_plus':
        command.extend(['--qrels', str(qrels)])
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '' if gpu is None else str(gpu),
        'PYTHONPATH': str(snapshot), 'PYTHONUNBUFFERED': '1', 'OMP_NUM_THREADS': '2',
        'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false'}
    token = uuid.uuid4().hex
    launch_path = score_output / 'queue_launch.json'
    guard_command = [str(args.judge_python), '-B', '-u', '-m', 'scripts.webbench_score_queue_v1',
                     'launch', '--record', str(launch_path), '--token', token]
    with (score_output / 'worker.log').open('a') as log:
        child = subprocess.Popen(guard_command, cwd=snapshot, env=env, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    record = dict(identity=p.identity(child.pid), gpu=gpu, started=time.time(), job=job,
        command=command, token=token, plan_sha256=plan['sha256'],
        gpu_uuid=inventory[gpu]['uuid'] if gpu is not None else None)
    if record['identity'] is None:
        raise RuntimeError('Scorer launch guard exited before admission')
    if gpu is not None:
        p.write(output / 'scorer_reservation.json', dict(plan_sha256=plan['sha256'],
            owner=p.identity(os.getpid()), child=record['identity'], gpu=gpu,
            status='running', updated=time.time(), launch_record=record))
    p.write(launch_path, record)
    return record, child


def recover(plan):
    output = Path(plan['output'])
    active = {}
    for job in main_jobs(plan):
        path = output / 'jobs' / job['name'] / 'scoring/queue_launch.json'
        if path.exists():
            record = p.read(path)
            if record.get('plan_sha256') != plan['sha256'] or record.get('job') != job:
                raise RuntimeError('Scorer launch belongs to another plan/condition')
            if live(record['identity']):
                active[job['name']] = record
    with resource_lock(output):
        path = output / 'scorer_reservation.json'
        reserved = p.read(path) if path.exists() else {}
        if reserved.get('status') in ACTIVE_RESERVATIONS and live(reserved.get('child')):
            if reserved.get('plan_sha256') != plan['sha256']:
                raise RuntimeError('Live scorer reservation belongs to another plan')
            record = reserved['launch_record']
            if record['identity'] != reserved['child'] or record['job'] not in main_jobs(plan):
                raise RuntimeError('Invalid recoverable child reservation')
            active[record['job']['name']] = record
            reserved.update(owner=p.identity(os.getpid()), updated=time.time())
            p.write(path, reserved)
            # Complete the durable admission gate after a parent publication crash.
            p.write(output / 'jobs' / record['job']['name'] / 'scoring/queue_launch.json', record)
        elif reserved:
            reserved.update(owner=p.identity(os.getpid()), child=None, status='released', updated=time.time())
            p.write(path, reserved)
    return active


def release_reservation(plan):
    output = Path(plan['output'])
    with resource_lock(output):
        path = output / 'scorer_reservation.json'
        reserved = p.read(path) if path.exists() else {}
        if reserved and not live(reserved.get('child')):
            reserved.update(owner=p.identity(os.getpid()), child=None, status='released', updated=time.time())
            p.write(path, reserved)


def queue(args):
    plan = p.verify_plan(Path(args.plan).resolve())
    output = Path(plan['output'])
    jobs = main_jobs(plan)
    qrels = p.read(plan['assets']['browsecomp_plus']['manifest'])['qrel_evidence']['path']
    s.offline_check(args.assets, require_model=True)
    runtime_manifest = p.read(args.runtime_manifest)
    if runtime_manifest.get('cpu_import_verified') is not True:
        raise RuntimeError('Judge runtime CPU import qualification is incomplete')
    contract = dict(plan_sha256=plan['sha256'], assets=str(Path(args.assets).resolve()),
        judge_python=str(Path(args.judge_python).absolute()), endpoint=args.endpoint,
        gpu_memory_utilization=.65, max_model_len=16384, minimum_free_gib=100,
        model_ready_sha256=s.file_sha(Path(args.assets) / 'model_ready.json'),
        judge_runtime_manifest=dict(path=str(Path(args.runtime_manifest).resolve()),
                                    sha256=s.file_sha(args.runtime_manifest)),
        sources_manifest_sha256=s.file_sha(Path(args.assets) / 'sources/manifest.json'),
        scorer_queue_sha256=p.file_hash(Path(__file__)),
        failures_are_not_automatically_retried=True)
    with p.lock(output / 'scorer_controller.lock'):
        contract_path = output / 'score_queue_contract.json'
        if contract_path.exists() and p.read(contract_path) != contract:
            raise RuntimeError('Scorer queue resume contract changed')
        p.write(contract_path, contract)
        active, children, completed = recover(plan), {}, {}
        while True:
            outcomes, ready = {}, []
            for job in jobs:
                name = job['name']
                directory = output / 'jobs' / name
                scoring = directory / 'scoring'
                if name in completed:
                    outcomes[name] = completed[name]
                    continue
                if (scoring / 'queue_failure.json').exists():
                    outcomes[name] = p.read(scoring / 'queue_failure.json')
                    continue
                actor = p.read(directory / 'status.json') if (directory / 'status.json').exists() else {}
                if actor.get('status') not in ('complete', 'complete_with_environment_failures'):
                    outcomes[name] = dict(status='actor_failed' if actor.get('status') == 'failed'
                        or (directory / 'failure.json').exists() else 'pending_actor')
                    continue
                try:
                    if (scoring / 'scores.json').exists():
                        outcomes[name] = completed[name] = validate_scores(plan, job, directory, args.assets, qrels)
                    elif name in active:
                        outcomes[name] = dict(status='scoring', gpu=active[name]['gpu'])
                    elif (scoring / 'queue_launch.json').exists() or (scoring / 'failures').exists():
                        outcomes[name] = failure(plan, job, 'Prior scoring attempt ended without valid terminal scores; automatic retry prohibited')
                    else:
                        p.validate_completed(plan, job, directory)
                        s.load_predictions(directory / 'predictions.json')
                        outcomes[name] = dict(status='ready_for_scoring')
                        ready.append(job)
                except Exception as exc:
                    outcomes[name] = failure(plan, job, f'Final input/score validation failed: {exc}')
            for name, record in list(active.items()):
                if name in children:
                    children[name].poll()
                if not live(record['identity']):
                    if name not in completed:
                        # The child may publish scores between this loop's scan and exit.
                        directory = output / 'jobs' / name
                        try:
                            outcomes[name] = completed[name] = validate_scores(
                                plan, record['job'], directory, args.assets, qrels)
                        except Exception as exc:
                            outcomes[name] = failure(plan, record['job'],
                                f'Scorer exited without validated terminal scores: {exc}',
                                child_identity=record['identity'],
                                exit_code=children[name].returncode if name in children else None)
                    del active[name]
            release_reservation(plan)
            for benchmark in ('browsecomp_plus', 'webwalkerqa'):
                if any(record['job']['benchmark'] == benchmark for record in active.values()):
                    continue
                pending = [job for job in ready if job['benchmark'] == benchmark]
                if not pending:
                    continue
                job = pending[0]
                if benchmark == 'webwalkerqa':
                    probe, blocked = qualification(args.assets, args.endpoint)
                    if blocked:
                        for item in pending:
                            outcomes[item['name']] = dict(status='blocked_judge', reason=blocked)
                        continue
                    probe_path = output / 'webwalker_qualification.json'
                    if probe_path.exists() and p.read(probe_path) != probe:
                        raise RuntimeError('Qualified WebWalker judge changed within this comparison')
                    p.write(probe_path, probe)
                try:
                    if benchmark == 'browsecomp_plus':
                        with resource_lock(output):
                            if reservation_busy(output):
                                outcomes[job['name']] = dict(status='waiting_for_reserved_scorer')
                                continue
                            inventory = p.gpu_inventory()
                            gpu = available_gpu(plan, inventory, actor_records(plan))
                            if gpu is None:
                                outcomes[job['name']] = dict(status='waiting_for_gpu')
                                continue
                            p.write(output / 'scorer_reservation.json', dict(plan_sha256=plan['sha256'],
                                owner=p.identity(os.getpid()), child=None, gpu=gpu,
                                status='admitting', updated=time.time()))
                            record, child = launch(plan, job, args, qrels, gpu, inventory)
                    else:
                        record, child = launch(plan, job, args, qrels)
                    active[job['name']], children[job['name']] = record, child
                    outcomes[job['name']] = dict(status='scoring', gpu=record['gpu'])
                except Exception as exc:
                    outcomes[job['name']] = failure(plan, job, f'Scorer admission/launch failed: {exc}')
            terminal = TERMINAL_SCORES | {'scoring_failed', 'actor_failed'}
            done = all(outcomes[job['name']]['status'] in terminal for job in jobs)
            actor_queue = p.read(output / 'queue_status.json') if (output / 'queue_status.json').exists() else {}
            qualified_failed = actor_queue.get('status') == 'qualification_failed' and not active
            status = ('complete' if done and all(v['status'] == 'complete' for v in outcomes.values())
                else 'incomplete' if done else 'blocked_actor_qualification' if qualified_failed
                else 'scoring' if active else 'blocked_judge' if any(v['status'] == 'blocked_judge' for v in outcomes.values())
                else 'waiting')
            p.write(output / 'scoring_status.json', dict(plan_sha256=plan['sha256'], updated_unix=time.time(),
                controller=p.identity(os.getpid()), status=status, active=active, conditions=outcomes,
                completed_conditions=len(completed), total_conditions=len(jobs)))
            write_reports(plan, outcomes)
            if not active and (done or qualified_failed):
                return
            time.sleep(args.poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('queue')
    run.add_argument('--plan', required=True)
    run.add_argument('--judge-python', type=Path, default=JUDGE_PYTHON)
    run.add_argument('--assets', type=Path, default=s.DEFAULT_ASSETS)
    run.add_argument('--runtime-manifest', type=Path, default=JUDGE_PYTHON.parent.parent / 'runtime_ready.json')
    run.add_argument('--endpoint', default='http://127.0.0.1:8313/v1')
    run.add_argument('--poll-seconds', type=float, default=10)
    guard = sub.add_parser('launch')
    guard.add_argument('--record', required=True)
    guard.add_argument('--token', required=True)
    args = parser.parse_args()
    if args.command == 'launch':
        launch_guard(args.record, args.token)
    else:
        if not 0 < args.poll_seconds <= 60:
            parser.error('Poll interval must be between 0 and 60 seconds')
        queue(args)


if __name__ == '__main__':
    main()
