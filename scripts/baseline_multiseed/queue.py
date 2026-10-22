"""Durable missing-only controller; never stops other GPU workloads."""
from __future__ import annotations

import argparse
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from scripts.baseline_multiseed import protocol as p


def identity(pid):
    try:
        stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'start_ticks': int(stat[19])} if stat[0] != 'Z' else None
    except (FileNotFoundError, ProcessLookupError):
        return None


def check_gpu(gpu, allowed_pid=None):
    result = subprocess.check_output(['nvidia-smi', f'--id={gpu}',
        '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'], text=True, timeout=20)
    uuid, free = [x.strip() for x in result.strip().split(',')]
    rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                   '--format=csv,noheader,nounits'], text=True, timeout=20)
    others = []
    for row in rows.splitlines():
        parts = [x.strip() for x in row.split(',')]
        if parts[0] == uuid and int(parts[1]) != allowed_pid:
            others.append(int(parts[1]))
    if others or int(free) < 52 * 1024:
        raise RuntimeError(f'GPU{gpu} unavailable: other PIDs={others}, free MiB={free}')
    return {'gpu': gpu, 'uuid': uuid, 'free_mib': int(free)}


def prerequisites(job):
    names = []
    if job['purpose'] == 'main':
        names.append(p.name({**job, 'purpose': 'smoke'}))
    if job['method'] == 'intentkv_rule' and job['ratio'] == .2:
        names.extend(p.name({**job, 'dataset': d, 'ratio': .5, 'purpose': 'main'}) for d in p.DATASETS)
    return names


def blocked_jobs(jobs, failed):
    blocked = set(failed)
    while True:
        expanded = blocked | {j['name'] for j in jobs if any(d in blocked for d in prerequisites(j))}
        if expanded == blocked:
            return blocked - set(failed)
        blocked = expanded


def validated(plan, job):
    directory = Path(plan['output']) / job['name']
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false'}
    log_path = directory / 'validation.log'
    with log_path.open('w') as log:
        result = subprocess.run([sys.executable, '-B', '-m', 'scripts.baseline_multiseed.worker',
            '--output', str(directory), '--validate'], cwd=p.ROOT, env=env,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Completed-output validation failed: {log_path}')


def controller(output):
    output = Path(output).resolve()
    plan = p.load_plan(output)
    lock = (output / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = output / 'queue_status.json'
    if state_path.exists():
        old = p.io.read_json(state_path)
        for row in old.get('active', {}).values():
            if identity(row['identity']['pid']) == row['identity']:
                raise RuntimeError('Previous worker is still alive; refusing a duplicate launch')
    complete, failed, active = set(plan['existing']), {}, {}
    for name, record in plan['existing'].items():
        for filename, checksum in record['file_hashes'].items():
            if p.io.file_hash(Path(record['path']) / filename) != checksum:
                raise RuntimeError(f'Previously complete source changed: {name}')
    for job in plan['jobs']:
        path = output / job['name'] / 'status.json'
        if path.exists():
            status = p.io.read_json(path)
            if status['status'] == 'complete':
                validated(plan, job)
                complete.add(job['name'])
            elif status['status'] == 'failed':
                failed[job['name']] = 'Preserved prior worker failure; no implicit rerun'

    def state(status, **extra):
        main = [j for j in plan['requested'] if j['name'] in complete]
        p.io.atomic_json(state_path, {'status': status, 'updated': time.time(),
            'controller': identity(os.getpid()), 'plan_sha256': plan['plan_sha256'],
            'requested_main': len(plan['requested']), 'complete_main_including_existing': len(main),
            'reused_main': len(plan['existing']), 'new_main_complete': len(main) - len(plan['existing']),
            'new_main_planned': sum(j['purpose'] == 'main' for j in plan['jobs']),
            'complete_new_conditions': len(complete) - len(plan['existing']),
            'failed': failed,
            'active': {str(g): {'identity': row['identity'], 'job': row['job']['name']}
                       for g, row in active.items()}, **extra})

    # Exercise every method/model once early, before lengthy formal conditions.
    first_smokes = set()
    covered = set()
    for j in plan['jobs']:
        key = (j['model'], j['method'])
        if key not in covered and j['purpose'] == 'smoke':
            covered.add(key)
            first_smokes.add(j['name'])
    jobs = sorted(plan['jobs'], key=lambda j: j['name'] not in first_smokes)
    try:
        state('running')
        while True:
            p.verify_sources(plan)
            for gpu, row in list(active.items()):
                code = row['process'].poll()
                if code is None:
                    continue
                row['log'].close()
                if code == 0:
                    try:
                        validated(plan, row['job'])
                        complete.add(row['job']['name'])
                    except Exception as exc:
                        failed[row['job']['name']] = str(exc)
                else:
                    failed[row['job']['name']] = f'Worker exit {code}; see preserved worker.log/status.json'
                row['gpu_lock'].close()
                del active[gpu]
            for gpu in plan['devices']:
                if gpu in active:
                    continue
                occupied = {r['job']['name'] for r in active.values()}
                candidates = [j for j in jobs if j['name'] not in complete | failed.keys() | occupied
                              and all(d in complete for d in prerequisites(j))]
                if not candidates:
                    continue
                gpu_lock = (p.BASE / 'runtime_maintenance' / f'baseline_expansion_gpu{gpu}.lock').open('a')
                try:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    device = check_gpu(gpu)
                except (RuntimeError, BlockingIOError):
                    gpu_lock.close()
                    continue
                job = candidates[0]
                directory = output / job['name']
                directory.mkdir(parents=True, exist_ok=True)
                env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'PYTHONUNBUFFERED': '1',
                       'TOKENIZERS_PARALLELISM': 'false', 'OMP_NUM_THREADS': '4',
                       'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
                       'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1'}
                log = (directory / 'worker.log').open('a')
                child = subprocess.Popen([sys.executable, '-B', '-u', '-m', 'scripts.baseline_multiseed.worker',
                    '--output', str(output), '--job', job['name'], '--gpu', str(gpu)],
                    cwd=p.ROOT, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                active[gpu] = {'process': child, 'identity': identity(child.pid), 'job': job,
                               'gpu_lock': gpu_lock, 'log': log}
                p.io.atomic_json(directory / 'launch.json', {'command': child.args, 'identity': identity(child.pid),
                    'device': device, 'plan_sha256': plan['plan_sha256'], 'at': time.time()})
                state('running')
            pending = [j for j in jobs if j['name'] not in complete | failed.keys()]
            blocked = blocked_jobs(jobs, failed)
            if not active and (not pending or all(j['name'] in blocked for j in pending)):
                state('complete' if not failed and not pending else 'failed',
                      blocked_jobs=[j['name'] for j in pending])
                results = []
                for job in plan['requested']:
                    if job['name'] not in complete:
                        results.append({**job, 'status': 'failed' if job['name'] in failed else 'blocked'})
                        continue
                    directory = Path(plan['existing'][job['name']]['path']) if job['name'] in plan['existing'] else output / job['name']
                    results.append({**job, 'status': 'complete', 'source': str(directory),
                        'reused': job['name'] in plan['existing'],
                        'summary': p.io.read_json(directory / 'results.json')['summary']})
                p.io.atomic_json(output / 'SUMMARY.json', {'plan_sha256': plan['plan_sha256'],
                    'complete_main': sum(j['name'] in complete for j in plan['requested']),
                    'requested_main': len(plan['requested']), 'results': results})
                return
            state('running' if active else 'waiting_for_gpu', pending_conditions=len(pending))
            time.sleep(10)
    except BaseException:
        state('controller_failed', traceback=traceback.format_exc(),
              worker_policy='Existing children retain their output locks; never kill unknown processes or duplicate live workers.')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=p.DEFAULT_OUTPUT)
    parser.add_argument('--devices', default='4,5,6')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if args.prepare and not args.execute:
        plan = p.prepare(args.output, [int(g) for g in args.devices.split(',')])
        print(p.io.canonical({'plan_sha256': plan['plan_sha256'], 'new_conditions': len(plan['jobs']),
                              'skipped_main': len(plan['existing'])}))
    elif args.execute and not args.prepare:
        controller(args.output)
    else:
        parser.error('Choose --prepare or --execute')


if __name__ == '__main__':
    main()
