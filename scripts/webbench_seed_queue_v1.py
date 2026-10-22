"""Schedule an existing frozen web benchmark plan one complete seed at a time.

This controller never changes or imports the working-tree actor. Workers keep
using the existing source_snapshot and plan, including their original hashes.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SEED_ORDER = (233, 42, 3407)
USER_INSTRUCTION = '先跑一个seed全部的 然后再接着跑其他seed'


def stages(jobs):
    """Partition every frozen job without changing its task/method identity."""
    groups = [('smoke', [j for j in jobs if j['purpose'] == 'smoke'])]
    groups.extend((f'seed{seed}', [j for j in jobs
                                 if j['purpose'] == 'main' and j['seed'] == seed])
                  for seed in SEED_ORDER)
    names = [j['name'] for _, group in groups for j in group]
    if len(names) != len(jobs) or len(set(names)) != len(names):
        raise ValueError('Every unique frozen job must belong to exactly one scheduling stage')
    if any(not group for _, group in groups):
        raise ValueError('Missing smoke qualification or requested seed stage')
    return groups


def current_stage(jobs, complete):
    """A failed or unfinished condition never unlocks the following seed."""
    for name, group in stages(jobs):
        if not {job['name'] for job in group} <= complete:
            return name, group
    return 'complete', []


def load_actor(plan_path):
    preliminary = json.loads(plan_path.read_text())
    output = Path(preliminary['output']).resolve()
    if plan_path.parent != output:
        raise RuntimeError('Plan must remain in its original output directory')
    snapshot = output / 'source_snapshot'
    sys.path.insert(0, str(snapshot))
    actor = importlib.import_module('scripts.webbench_v1')
    if Path(actor.__file__).resolve() != snapshot / 'scripts/webbench_v1.py':
        raise RuntimeError('Controller must import the actual frozen actor snapshot')
    plan = actor.verify_plan(plan_path)
    if tuple(actor.SEEDS) != SEED_ORDER:
        raise RuntimeError('Frozen seeds differ from the requested scheduling order')
    groups = stages(plan['jobs'])
    if len(groups) != 4 or any(len(group) != 64 for _, group in groups):
        raise RuntimeError('Expected all 64 qualification conditions and 64 conditions per seed')
    return actor, plan


def freeze_controller(actor, plan, plan_path):
    folder = Path(plan['output']) / 'scheduler/seed_order_v1'
    source = Path(__file__).resolve()
    snapshot = folder / 'webbench_seed_queue_v1.py'
    contract_path = folder / 'contract.json'
    checksum = actor.file_hash(source)
    with actor.lock(folder / 'contract.lock'):
        if contract_path.exists():
            contract = actor.read(contract_path)
            expected = contract.pop('sha256')
            if actor.digest(contract) != expected:
                raise RuntimeError('Scheduling contract changed')
            contract['sha256'] = expected
            if (contract['plan_sha256'] != plan['sha256']
                    or contract['plan_file_sha256'] != actor.file_hash(plan_path)
                    or contract['scheduler_source_sha256'] != checksum
                    or actor.file_hash(snapshot) != checksum
                    or contract['user_instruction'] != USER_INSTRUCTION
                    or contract['seed_order'] != list(SEED_ORDER)):
                raise RuntimeError('Existing scheduling contract differs; do not overwrite it')
            return contract
        if snapshot.exists():
            if actor.file_hash(snapshot) != checksum:
                raise RuntimeError('Uncommitted scheduler snapshot differs')
        else:
            temporary = snapshot.with_name(snapshot.name + f'.{os.getpid()}.tmp')
            shutil.copy2(source, temporary)
            if actor.file_hash(temporary) != checksum:
                raise RuntimeError('Scheduler source changed while freezing')
            os.replace(temporary, snapshot)
        contract = {
            'schema': 'webbench_seed_order_v1', 'created': time.time(),
            'plan': str(plan_path), 'plan_sha256': plan['sha256'],
            'plan_file_sha256': actor.file_hash(plan_path),
            'scheduler_snapshot': str(snapshot), 'scheduler_source_sha256': checksum,
            'actor_source_sha256': plan['source_hashes']['scripts/webbench_v1.py'],
            'user_instruction': USER_INSTRUCTION, 'seed_order': list(SEED_ORDER),
            'stages': [{'name': name, 'conditions': len(group)}
                       for name, group in stages(plan['jobs'])],
            'gate': 'All 64 smoke conditions must qualify; all 64 main conditions of a seed '
                    'must complete before any next-seed worker starts. Audited completion '
                    'with recorded environment failures counts as complete. Infrastructure '
                    'failure blocks later stages; no automatic question retries.',
            'workers': 'Original frozen scripts.webbench_v1 worker with unchanged plan and sources',
        }
        contract['sha256'] = actor.digest(contract)
        actor.write(contract_path, contract)
        return contract


def queue(actor, plan, plan_path, contract):
    p = actor
    output = Path(plan['output'])
    snapshot = output / 'source_snapshot'
    known = {job['name']: job for job in plan['jobs']}
    with p.lock(output / 'controller.lock'):
        active, children, launches = {}, {}, {}
        previous = p.read(output / 'queue_status.json') if (output / 'queue_status.json').exists() else {}
        if previous and previous.get('plan_sha256') != plan['sha256']:
            raise RuntimeError('Previous controller state belongs to another plan')
        for name, record in previous.get('active', {}).items():
            if name not in known or record.get('job') != known[name]:
                raise RuntimeError('Previous worker identity differs from the frozen job')
            if p.identity(record['identity']['pid']) == record['identity']:
                active[name] = record
        for job in plan['jobs']:
            path = output / 'jobs' / job['name'] / 'launch.json'
            if path.exists():
                record = p.read(path)
                if record.get('job') != job:
                    raise RuntimeError('Prior launch belongs to another condition')
                launches[job['name']] = record
                if p.identity(record['identity']['pid']) == record['identity']:
                    active[job['name']] = record
        validated = set()
        while True:
            complete, failed = set(), {}
            for job in plan['jobs']:
                directory = output / 'jobs' / job['name']
                status = p.read(directory / 'status.json') if (directory / 'status.json').exists() else {}
                if status.get('status') == 'failed' or (directory / 'failure.json').exists():
                    failed[job['name']] = status.get('error', 'recorded failure')
                elif status.get('status') in ('complete', 'complete_with_environment_failures'):
                    try:
                        if job['name'] not in validated:
                            p.validate_completed(plan, job, directory)
                            validated.add(job['name'])
                        complete.add(job['name'])
                    except Exception as exc:
                        failed[job['name']] = f'Final audit failed: {exc}'
            for name, record in {**launches, **active}.items():
                if name in children:
                    children[name].poll()
                if p.identity(record['identity']['pid']) != record['identity']:
                    if name not in complete and name not in failed:
                        directory = output / 'jobs' / name
                        p.write(directory / 'failure.json', {
                            'type': 'WorkerExited', 'time': time.time(), 'identity': record['identity'],
                            'error': 'Worker exited without terminal status; no automatic question rerun'})
                        failed[name] = 'Worker exited without terminal status'
                    active.pop(name, None)
            stage, stage_jobs = current_stage(plan['jobs'], complete)
            pending = [j for j in stage_jobs
                       if j['name'] not in complete | set(failed) | set(active)]
            for gpu in plan['devices']:
                if len(active) >= plan['max_workers'] or not pending:
                    break
                if sum(r['gpu'] == gpu for r in active.values()) >= plan['slots_per_device']:
                    continue
                try:
                    with p.lock(output / 'resource_admission.lock'):
                        inventory = p.gpu_inventory()
                        reservation_path = output / 'scorer_reservation.json'
                        reservation = p.read(reservation_path) if reservation_path.exists() else {}
                        reserved = any(who and p.identity(who['pid']) == who
                                       for who in (reservation.get('owner'), reservation.get('child')))
                        reserved = (reserved and reservation.get('status') in ('admitting', 'running')
                                    and reservation.get('plan_sha256') == plan['sha256'])
                        if ((reserved and reservation.get('gpu') == gpu)
                                or not p.can_admit(plan, active, gpu, inventory[gpu])):
                            continue
                        job = pending.pop(0)
                        directory = output / 'jobs' / job['name']
                        directory.mkdir(parents=True, exist_ok=True)
                        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '2',
                               'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
                               'TOKENIZERS_PARALLELISM': 'false', 'PYTHONPATH': str(snapshot),
                               'PYTHONUNBUFFERED': '1'}
                        command = [str(p.PYTHON), '-B', '-u', '-m', 'scripts.webbench_v1', 'worker',
                                   '--plan', str(plan_path), '--job', job['name'], '--gpu', str(gpu)]
                        with (directory / 'worker.log').open('a') as log:
                            child = subprocess.Popen(command, cwd=snapshot, env=env,
                                                     stdout=log, stderr=subprocess.STDOUT,
                                                     start_new_session=True)
                        record = {'identity': p.identity(child.pid), 'gpu': gpu,
                                  'gpu_uuid': inventory[gpu]['uuid'], 'started': time.time(), 'job': job}
                        p.write(directory / 'launch.json', {**record, 'command': command})
                        active[job['name']] = record
                        launches[job['name']] = record
                        children[job['name']] = child
                        p.write(output / 'actor_reservations.json', {
                            'plan_sha256': plan['sha256'], 'active': active, 'updated': time.time()})
                except BlockingIOError:
                    continue
            blocked = any(job['name'] in failed for job in stage_jobs)
            terminal_failure = not active and not pending and blocked
            state = {
                'updated': time.time(), 'controller': p.identity(os.getpid()),
                'plan_sha256': plan['sha256'], 'scheduler_contract_sha256': contract['sha256'],
                'scheduler': contract['scheduler_snapshot'], 'seed_order': list(SEED_ORDER),
                'current_stage': stage,
                'status': ('complete' if stage == 'complete' and not active else
                           'qualification_failed' if terminal_failure and stage == 'smoke' else
                           'seed_failed' if terminal_failure else 'running' if active else 'waiting_for_gpu'),
                'complete_conditions': len(complete), 'total_conditions': len(plan['jobs']),
                'smoke_complete': {j['name'] for j in plan['jobs'] if j['purpose'] == 'smoke'} <= complete,
                'active': active, 'failures': failed, 'later_stages_blocked_by_failure': blocked,
                'formal_complete_conditions': sum(j['purpose'] == 'main' and j['name'] in complete
                                                  for j in plan['jobs']),
                'stages': [{'name': name, 'complete': sum(j['name'] in complete for j in group),
                            'failed': sum(j['name'] in failed for j in group), 'total': len(group)}
                           for name, group in stages(plan['jobs'])],
            }
            p.write(output / 'queue_status.json', state)
            if not active and (stage == 'complete' or terminal_failure):
                return
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'queue'))
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan_path = args.plan.resolve()
    actor, plan = load_actor(plan_path)
    contract = freeze_controller(actor, plan, plan_path)
    if args.command == 'prepare':
        print(json.dumps(contract, ensure_ascii=False, indent=2))
    else:
        queue(actor, plan, plan_path, contract)


if __name__ == '__main__':
    main()
