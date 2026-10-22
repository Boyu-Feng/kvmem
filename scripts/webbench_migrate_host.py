"""Host-only resource adapter for the frozen WebBench experiment.

One actor or local judge per selected GPU.  Frozen plans, source files,
scientific budgets, BF16 precision, seed gates and score contracts stay intact;
effective allocation and CPU offload are separately bound to a host contract.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def private_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    sys.modules[name] = value
    spec.loader.exec_module(value)
    return value


m = private_module('_webbench_migration_helpers', HERE / 'webbench_migrate_resume.py')


def validate_profile(profile):
    m.require(profile.get('schema') == 'webbench_migration_host_resources_v1', 'Unsupported host resource schema')
    devices = profile.get('devices', [])
    m.require(devices and len(set(devices)) == len(devices)
              and all(type(g) is int and g >= 0 for g in devices), 'Select distinct physical GPU indices')
    m.require(profile.get('workers_per_gpu') == 1, 'Host migration permits only one actor or local judge per GPU')
    m.require(profile.get('tensor_parallel_size') == 1 and profile.get('dtype') == 'bfloat16',
              'Host migration preserves TP1 and BF16')
    m.require(profile.get('endpoint') == 'http://127.0.0.1:8313/v1', 'Frozen API endpoint must remain unchanged')
    for key in ('actor_allocator_gib', 'gpu_budget_gib', 'headroom_gib', 'judge_cpu_offload_gb',
                'minimum_available_host_ram_gib'):
        value = profile.get(key)
        m.require(type(value) in (int, float) and math.isfinite(value), f'Invalid host resource value: {key}')
    m.require(0 < profile['actor_allocator_gib'] <= profile['gpu_budget_gib'] <= 48,
              'Actor allocator must be positive and no greater than GPU budget or original 48 GiB')
    m.require(profile['headroom_gib'] >= 2, 'Reserve at least 2 GiB for non-allocator GPU overhead')
    m.require(profile['judge_cpu_offload_gb'] >= 0 and profile['minimum_available_host_ram_gib'] > 0,
              'Invalid CPU offload or host RAM admission')
    return profile


def load_contract(path):
    value = m.read(path)
    body = dict(value)
    checksum = body.pop('sha256')
    m.require(m.digest(body) == checksum, 'Host resource contract hash differs')
    validate_profile(value)
    m.require(value['route_plan_sha256'] == m.EXPECTED_PLANS, 'Host contract refers to another frozen comparison')
    m.require(value['target_host']['boot_id'] == m.host_identity()['boot_id'],
              'Host resource contract belongs to another boot; explicit recovery is required')
    for name, expected in value['source_hashes'].items():
        m.require(m.file_hash(name) == expected, f'Host adapter source changed: {name}')
    return value


def freeze_contract(profile, path):
    validate_profile(profile)
    value = dict(profile, created_at_utc=m.utcnow(), target_host=m.host_identity(),
                 route_plan_sha256=m.EXPECTED_PLANS,
                 source_hashes={str(p): m.file_hash(p) for p in [Path(__file__).resolve(),
                     HERE / 'webbench_migrate_resume.py', HERE / 'webbench_migrate_judge.py']},
                 original_resources={'devices': [4, 5, 6, 7], 'actor_allocator_gib': 48,
                                     'slots_per_gpu': 2, 'judge_gpu_memory_utilization': .65,
                                     'judge_tensor_parallel_size': 1},
                 qualification='A100 with 30 GiB free has not been execution-qualified on the source H200. '
                     'OOM remains a recorded infrastructure failure; budgets/precision are never reduced.')
    value['sha256'] = m.digest(value)
    m.require(not Path(path).exists(), 'Host resource contract already exists')
    m.atomic_json(path, value)
    return value


def available_host_ram_gib():
    fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(fields['MemAvailable'].split()[0]) / (1024 * 1024)


def select_gpu(profile, inventory, active):
    occupied = {value['gpu'] for value in active.values() if value['gpu'] is not None}
    needed = profile['gpu_budget_gib'] + profile['headroom_gib']
    candidates = [gpu for gpu in profile['devices'] if gpu not in occupied and gpu in inventory
                  and inventory[gpu]['free_gib'] >= needed]
    return max(candidates, key=lambda gpu: inventory[gpu]['free_gib']) if candidates else None


def actor(args):
    contract = load_contract(args.host_contract)
    m.require(args.gpu in contract['devices'], 'GPU is outside the host contract')
    plan_path = Path(args.plan).resolve()
    snapshot = plan_path.parent / 'source_snapshot'
    sys.path.insert(0, str(snapshot))
    import importlib
    p = importlib.import_module('scripts.webbench_v1')
    m.require(Path(p.__file__).resolve() == snapshot / 'scripts/webbench_v1.py', 'Actor loaded outside frozen snapshot')
    plan = p.verify_plan(plan_path)
    job, = [job for job in plan['jobs'] if job['name'] == args.job]
    directory = Path(plan['output']) / 'jobs' / args.job
    import torch
    m.require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(args.gpu)
              and torch.cuda.device_count() == 1, 'Actor physical GPU assignment differs')
    total = torch.cuda.get_device_properties(0).total_memory
    original = torch.cuda.set_per_process_memory_fraction
    expected = plan['worker_allocator_gib'] * 1024**3 / total
    effective = contract['actor_allocator_gib'] * 1024**3 / total
    path = directory / 'host_execution.json'
    m.require(not path.exists(), f'Existing host actor attempt requires separate recovery: {path}')
    record = dict(schema='webbench_host_actor_execution_v1', state='started',
                  plan_sha256=plan['sha256'], job=job, host_contract_sha256=contract['sha256'],
                  host_contract=str(Path(args.host_contract).resolve()), host=m.host_identity(),
                  physical_gpu=args.gpu, physical_total_bytes=total,
                  original_allocator_gib=plan['worker_allocator_gib'],
                  effective_allocator_gib=contract['actor_allocator_gib'],
                  source_sha256=m.file_hash(__file__), started_at_utc=m.utcnow())
    m.atomic_json(path, record)
    calls = []
    def cap(fraction, device=None):
        m.require(abs(fraction - expected) < 1e-9, 'Frozen actor requested an unexpected allocator fraction')
        calls.append(dict(requested=fraction, effective=effective))
        return original(effective, device)
    torch.cuda.set_per_process_memory_fraction = cap
    try:
        p.worker(plan_path, args.job, args.gpu)
        record.update(state='completed', completed_at_utc=m.utcnow(), allocator_calls=calls,
                      predictions_sha256=m.file_hash(directory / 'predictions.json'))
    except BaseException as exc:
        record.update(state='failed', completed_at_utc=m.utcnow(), allocator_calls=calls,
                      error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        torch.cuda.set_per_process_memory_fraction = original
        m.atomic_json(path, record)


def validate_actor_host_record(plan, job, directory, contract, required=False):
    path = directory / 'host_execution.json'
    if not path.exists():
        m.require(not required, f'Host actor execution record missing: {path}')
        return
    record = m.read(path)
    m.require(record['state'] == 'completed' and record['plan_sha256'] == plan['sha256']
              and record['job'] == job and record['host_contract_sha256'] == contract['sha256']
              and record['effective_allocator_gib'] == contract['actor_allocator_gib']
              and record['source_sha256'] == m.file_hash(__file__)
              and record['predictions_sha256'] == m.file_hash(directory / 'predictions.json'),
              f'Host actor execution binding differs: {path}')


def waiting_for_host_record(record_path, launch, identity):
    """Terminal artifacts can precede the wrapper's durable completion record."""
    if launch is None or identity(launch['identity']['pid']) != launch['identity']:
        return False
    return not record_path.exists() or m.read(record_path).get('state') not in {'completed', 'failed'}


def gate(args):
    command = args.child_command
    if command and command[0] == '--':
        command = command[1:]
    for _ in range(200):
        if Path(args.record).exists():
            record = m.read(args.record)
            if record.get('token') == args.token:
                m.require(record['identity'] == m.process_identity(os.getpid()) and record['command'] == command,
                          'Host child admission identity differs')
                os.execvpe(command[0], command, os.environ)
        time.sleep(.1)
    raise RuntimeError('Host worker was not durably admitted; no actor/judge call was started')


def queue(args):
    contract = load_contract(args.host_contract)
    scheduler = private_module('_host_frozen_scheduler_v4', m.SCHEDULER)
    routes, conditions, scorer = scheduler.load_routes(m.OWNER / 'plan.json', m.WEBWALKER / 'plan.json')
    judge = private_module('_host_judge_wrapper', HERE / 'webbench_migrate_judge.py')
    for gpu in contract['devices']:
        judge.validate_probe_record(m.OWNER / 'plan.json', Path(args.host_contract), gpu)
    p = routes['browsecomp_plus']['actor']
    folder = Path(args.host_contract).resolve().parent
    active, children = {}, {}
    validated, scored, failures, scoring_failures = set(), {}, {}, {}
    with contextlib.ExitStack() as stack:
        for output, name in [(m.OWNER, 'controller.lock'), (m.WEBWALKER, 'controller.lock'),
                             (m.OWNER, 'scorer_controller.lock'),
                             (m.WEBWALKER, 'scorer_controller.lock')]:
            stack.enter_context(m.exclusive_lock(output / name))
        previous_path = folder / 'host_queue_status.json'
        if previous_path.exists():
            previous = m.read(previous_path)
            m.require(previous['host_contract_sha256'] == contract['sha256'], 'Previous host queue contract differs')
            m.require(previous['host']['boot_id'] == m.host_identity()['boot_id'], 'Host queue cannot adopt another boot')
        for name, item in conditions.items():
            route, job = routes[item['route']], item['job']
            directory = route['output'] / 'jobs' / name
            for role, path in [('actor', directory / 'launch.json'), ('score', directory / 'scoring/queue_launch.json')]:
                if path.exists():
                    record = m.read(path)
                    m.require(record.get('host_contract_sha256') == contract['sha256'],
                              f'Source-machine launch was not archived by migration: {path}')
                    m.require(record['job'] == job, f'Host launch job differs: {path}')
                    active[f'{role}:{name}'] = record
        def publish_reservations():
            p.write(m.OWNER / 'actor_reservations.json', dict(plan_sha256=m.EXPECTED_PLANS['browsecomp_plus'],
                    host_contract_sha256=contract['sha256'], active={r['job']['name']: r for r in active.values()
                                                                  if r['role'] == 'actor'}, updated=time.time()))
        def launch(job, role, gpu, inventory):
            route = routes[job['benchmark']]
            directory = route['output'] / 'jobs' / job['name']
            target = directory if role == 'actor' else directory / 'scoring'
            target.mkdir(parents=True, exist_ok=True)
            record_path = target / ('launch.json' if role == 'actor' else 'queue_launch.json')
            m.require(not record_path.exists(), f'Previous launch cannot be silently retried: {record_path}')
            if role == 'actor':
                command = [str(m.ACTOR_PYTHON), '-B', '-u', str(Path(__file__).resolve()), 'actor',
                           '--plan', str(route['path']), '--job', job['name'], '--gpu', str(gpu),
                           '--host-contract', str(Path(args.host_contract).resolve())]
            elif job['benchmark'] == 'browsecomp_plus':
                command = [str(m.JUDGE_PYTHON), '-B', '-u', str(HERE / 'webbench_migrate_judge.py'), 'score',
                           '--plan', str(route['path']), '--job', job['name'], '--gpu', str(gpu),
                           '--host-contract', str(Path(args.host_contract).resolve())]
            else:
                command = [str(m.JUDGE_PYTHON), '-B', '-u', '-m', 'scripts.webbench_scoring_v1', 'score',
                           '--benchmark', 'webwalkerqa', '--predictions', str(directory / 'predictions.json'),
                           '--output', str(target), '--assets', route['scoring']['assets'],
                           '--endpoint', contract['endpoint'], '--runtime-manifest',
                           str(m.BASE / 'runtime/judge_env/runtime_ready.json'),
                           '--gpu-memory-utilization', '.65', '--max-model-len', '16384']
            token = uuid.uuid4().hex
            gate_command = [str(m.ACTOR_PYTHON), '-B', '-u', str(Path(__file__).resolve()), 'gate',
                            '--record', str(record_path), '--token', token, '--', *command]
            env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '' if gpu is None else str(gpu),
                   'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
                   'PYTHONPATH': str(route['snapshot']), 'PYTHONDONTWRITEBYTECODE': '1',
                   'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1'}
            with (target / ('worker.log' if role == 'actor' else 'host_worker.log')).open('a') as log:
                child = subprocess.Popen(gate_command, cwd=route['snapshot'], env=env,
                                         stdout=log, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
            identity = p.identity(child.pid)
            m.require(identity is not None, 'Host launch guard exited before durable admission')
            record = dict(identity=identity, gpu=gpu, gpu_uuid=inventory[gpu]['uuid'] if gpu is not None else None,
                          started=time.time(), job=job, command=command, token=token, role=role,
                          route=job['benchmark'], worker_plan_sha256=route['plan']['sha256'],
                          plan_sha256=route['plan']['sha256'], host_contract_sha256=contract['sha256'],
                          host=m.host_identity())
            p.write(record_path, record)
            key = f"{role}:{job['name']}"
            active[key], children[key] = record, child
            publish_reservations()
        def collect(name, complete):
            route, job = routes[conditions[name]['route']], conditions[name]['job']
            directory = route['output'] / 'jobs' / name
            status = p.read(directory / 'status.json') if (directory / 'status.json').exists() else {}
            actor_launch = active.get('actor:' + name)
            if status.get('status') == 'failed' or (directory / 'failure.json').exists():
                failures[name] = status.get('error', 'recorded actor failure')
            elif status.get('status') in m.TERMINAL_ACTOR and not waiting_for_host_record(
                    directory / 'host_execution.json', actor_launch, p.identity):
                try:
                    if name not in validated:
                        route['actor'].validate_completed(route['plan'], job, directory)
                        validate_actor_host_record(route['plan'], job, directory, contract,
                                                   required=actor_launch is not None)
                        validated.add(name)
                    complete.add(name)
                except Exception as exc:
                    failures[name] = str(exc)
            scoring = directory / 'scoring'
            if job['purpose'] != 'main':
                return
            permanent = m.OWNER / 'scheduler/seed_order_v4/score_validation_failures' / name / 'failure.json'
            if permanent.exists():
                marker = p.read(permanent)
                m.require(marker.get('plan_sha256') == route['plan']['sha256'] and marker.get('job') == job,
                          f'Frozen scheduler score failure belongs to another route: {permanent}')
                scoring_failures[name] = marker.get('error', 'Recorded score validation failure; no automatic retry')
                scored.pop(name, None)
            elif (scoring / 'queue_failure.json').exists() or (scoring / 'failures').exists():
                scoring_failures[name] = 'Recorded scoring failure; no automatic retry'
                scored.pop(name, None)
            elif name in complete and name not in scored and (scoring / 'scores.json').exists():
                score_launch = active.get('score:' + name)
                if job['benchmark'] == 'browsecomp_plus' and waiting_for_host_record(
                        scoring / 'host_execution.json', score_launch, p.identity):
                    return
                try:
                    value = scorer.validate_scores(route['plan'], job, directory,
                                                   route['scoring']['assets'], route['scoring']['qrels'])
                    m.require(value.get('status') in scorer.TERMINAL_SCORES,
                              'Frozen score validator did not return a terminal recorded outcome')
                    m.require(score_launch is None or job['benchmark'] != 'browsecomp_plus'
                              or (scoring / 'host_execution.json').exists(),
                              f'Host judge execution record missing: {scoring}')
                    if (scoring / 'host_execution.json').exists() or (score_launch is not None
                            and job['benchmark'] == 'browsecomp_plus'):
                        judge.validate_host_record(route['path'], name, Path(args.host_contract))
                    scored[name] = value
                except Exception as exc:
                    scoring_failures[name] = str(exc)
                    p.write(permanent, dict(status='scoring_failed', plan_sha256=route['plan']['sha256'],
                                           job=job, error=str(exc), time=time.time(),
                                           host_contract_sha256=contract['sha256']))
        while True:
            complete = set()
            for name in conditions:
                collect(name, complete)
            for key, record in list(active.items()):
                if key in children:
                    children[key].poll()
                if p.identity(record['identity']['pid']) == record['identity']:
                    continue
                name, role = record['job']['name'], record['role']
                route = routes[record['job']['benchmark']]
                directory = route['output'] / 'jobs' / name
                # The child may commit its final result between the scan and
                # this liveness check. Re-read after observing its exit.
                collect(name, complete)
                if role == 'actor' and name not in complete and name not in failures:
                    failures[name] = 'Host actor exited without terminal result; no automatic retry'
                    p.write(directory / 'failure.json', dict(type='HostWorkerExited', error=failures[name],
                            identity=record['identity'], host_contract_sha256=contract['sha256'], time=time.time()))
                elif role == 'score' and name not in scored and name not in scoring_failures:
                    scoring_failures[name] = 'Host scorer exited without validated terminal scores; no automatic retry'
                    p.write(directory / 'scoring/queue_failure.json', dict(status='scoring_failed',
                            plan_sha256=route['plan']['sha256'], job=record['job'], error=scoring_failures[name],
                            identity=record['identity'], host_contract_sha256=contract['sha256'], time=time.time()))
                del active[key]
            seed, qualified, eligible = scheduler.eligibility(conditions, complete, set(scored))
            pending = [job for job in eligible if job['name'] not in complete | set(failures)
                       and 'actor:' + job['name'] not in active]
            ready_scores = [item['job'] for name, item in conditions.items()
                            if item['job']['purpose'] == 'main' and name in complete
                            and name not in scored and name not in scoring_failures
                            and 'score:' + name not in active]
            with p.lock(m.OWNER / 'resource_admission.lock'):
                inventory = p.gpu_inventory()
                for benchmark in ('webwalkerqa', 'browsecomp_plus'):
                    if any(r['role'] == 'score' and r['job']['benchmark'] == benchmark for r in active.values()):
                        continue
                    ready = [job for job in ready_scores if job['benchmark'] == benchmark]
                    if not ready:
                        continue
                    gpu = None if benchmark == 'webwalkerqa' else select_gpu(contract, inventory, active)
                    if benchmark == 'browsecomp_plus' and (gpu is None or available_host_ram_gib()
                                                           < contract['minimum_available_host_ram_gib']):
                        continue
                    launch(ready[0], 'score', gpu, inventory)
                while pending and (gpu := select_gpu(contract, inventory, active)) is not None:
                    launch(pending.pop(0), 'actor', gpu, inventory)
            relevant = [item['job'] for item in conditions.values()
                        if item['job']['purpose'] == 'smoke' or item['job']['seed'] == seed]
            blocked = any(job['name'] in failures or job['name'] in scoring_failures for job in relevant)
            state = dict(schema='webbench_host_queue_v1', updated=time.time(), host=m.host_identity(),
                         controller=p.identity(os.getpid()), host_contract_sha256=contract['sha256'],
                         host_contract=str(Path(args.host_contract).resolve()),
                         plan_sha256=m.EXPECTED_PLANS['browsecomp_plus'], route_plans=m.EXPECTED_PLANS,
                         seed_order=list(scheduler.SEEDS), current_stage='complete' if seed is None else f'seed{seed}',
                         status='complete' if seed is None and not active else
                                'seed_failed' if blocked and not active and not pending else
                                'running' if active else 'waiting_for_gpu_or_scoring',
                         benchmark_qualified=qualified, smoke_complete=all(qualified.values()),
                         complete_conditions=len(complete), formal_complete_conditions=sum(
                             item['job']['purpose'] == 'main' and name in complete for name, item in conditions.items()),
                         scored_conditions=len(scored), active=active, failures=failures,
                         scoring_failures=scoring_failures, later_stages_blocked_by_failure=blocked,
                         stages=[dict(seed=s, total=64,
                             actor_complete=sum(item['job']['purpose'] == 'main' and item['job']['seed'] == s
                                                and name in complete for name, item in conditions.items()),
                             scores_complete=sum(item['job']['seed'] == s and name in scored
                                                 for name, item in conditions.items())) for s in scheduler.SEEDS])
            p.write(previous_path, state)
            p.write(m.OWNER / 'queue_status.json', state)
            p.write(m.WEBWALKER / 'queue_status.json', {**state, 'plan_sha256': m.EXPECTED_PLANS['webwalkerqa']})
            publish_reservations()
            for benchmark, route in routes.items():
                outcomes = {name: scored.get(name, dict(status='scoring_failed' if name in scoring_failures
                             else 'actor_failed' if name in failures else 'scoring' if 'score:' + name in active
                             else 'ready_for_scoring' if name in complete else 'pending_actor'))
                            for name, item in conditions.items() if item['route'] == benchmark
                            and item['job']['purpose'] == 'main'}
                p.write(route['output'] / 'scoring_status.json', dict(
                    plan_sha256=route['plan']['sha256'], updated_unix=time.time(),
                    controller=p.identity(os.getpid()), host_contract_sha256=contract['sha256'],
                    status=state['status'], active={r['job']['name']: r for r in active.values()
                        if r['role'] == 'score' and r['job']['benchmark'] == benchmark}, conditions=outcomes,
                    completed_conditions=sum(name in scored for name in outcomes), total_conditions=len(outcomes)))
                scorer.write_reports(route['plan'], outcomes)
                p.write(route['output'] / 'HOST_EXECUTION.json', dict(host_contract=str(args.host_contract),
                        host_contract_sha256=contract['sha256'], original_plan_unchanged=True,
                        effective_resources=contract, numerical_equivalence_on_A100_not_claimed=True))
            if not active and (seed is None or (blocked and not pending)):
                return
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    work = sub.add_parser('actor')
    work.add_argument('--plan', type=Path, required=True)
    work.add_argument('--job', required=True)
    work.add_argument('--gpu', type=int, required=True)
    work.add_argument('--host-contract', type=Path, required=True)
    control = sub.add_parser('queue')
    control.add_argument('--host-contract', type=Path, required=True)
    guarded = sub.add_parser('gate')
    guarded.add_argument('--record', type=Path, required=True)
    guarded.add_argument('--token', required=True)
    guarded.add_argument('child_command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    {'actor': actor, 'queue': queue, 'gate': gate}[args.command](args)


if __name__ == '__main__':
    main()
