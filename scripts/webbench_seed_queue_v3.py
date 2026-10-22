"""Route preserved BrowseComp and repaired WebWalker through separate frozen plans.

The original run owns GPU admission and the controller lock. Each worker and
score validator uses its benchmark's actual frozen plan and output directory.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

SEEDS = (233, 42, 3407)
BENCHMARKS = ('browsecomp_plus', 'webwalkerqa')
USER_ORDER = '先跑一个seed全部的 然后再接着跑其他seed'


def load_actor_module(snapshot, repaired=False):
    """Keep original imports pinned while loading the repaired actor privately."""
    path = snapshot / 'scripts/webbench_v1.py'
    if repaired:
        spec = importlib.util.spec_from_file_location('_routed_webwalker_actor_v3', path)
        actor = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = actor
        spec.loader.exec_module(actor)
    else:
        sys.path.insert(0, str(snapshot))
        actor = importlib.import_module('scripts.webbench_v1')
    if Path(actor.__file__).resolve() != path:
        raise RuntimeError('Actor verifier did not load from its actual frozen snapshot')
    return actor


def load_action_format_helper(snapshot, plan):
    """Bind the new-only dynamic dependency without changing scripts.__path__."""
    source = 'scripts/webbench_action_format_v2.py'
    name = 'scripts.webbench_action_format_v2'
    path = snapshot / source
    expected = plan['source_hashes'].get(source)
    if (not expected or path.resolve() != path
            or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
        raise RuntimeError('Repaired action-format helper is not bound to its frozen plan')
    helper = sys.modules.get(name)
    if helper is not None:
        if (Path(helper.__file__).resolve() != path
                or getattr(helper, '_webbench_frozen_sha256', None) != expected):
            raise RuntimeError('Action-format helper was already loaded from another source')
    else:
        spec = importlib.util.spec_from_file_location(name, path)
        helper = importlib.util.module_from_spec(spec)
        sys.modules[name] = helper
        try:
            spec.loader.exec_module(helper)
            if (Path(helper.__file__).resolve() != path
                    or hashlib.sha256(path.read_bytes()).hexdigest() != expected):
                raise RuntimeError('Action-format helper changed during import')
            helper._webbench_frozen_sha256 = expected
        except BaseException:
            sys.modules.pop(name, None)
            raise
    if helper.VERSION != plan['action_format']['webwalkerqa']['version']:
        raise RuntimeError('Action-format helper version differs from its frozen plan')
    return helper


def routed_jobs(old_plan, webwalker_plan):
    selected = {}
    for benchmark, plan in zip(BENCHMARKS, (old_plan, webwalker_plan), strict=True):
        jobs = [job for job in plan['jobs'] if job['benchmark'] == benchmark]
        if benchmark == 'webwalkerqa' and len(jobs) != len(plan['jobs']):
            raise ValueError('The repaired plan must contain only WebWalker jobs')
        for purpose, seed, count in [('smoke', 233, 32), *[('main', s, 32) for s in SEEDS]]:
            if sum(j['purpose'] == purpose and j['seed'] == seed for j in jobs) != count:
                raise ValueError('Expected 32 smoke conditions and 32 main conditions per seed per benchmark')
        if len(jobs) != 128:
            raise ValueError('Unexpected benchmark condition count')
        for job in jobs:
            if job['name'] in selected:
                raise ValueError('Duplicate routed condition identity')
            selected[job['name']] = {'route': benchmark, 'job': job}
    old_ww = {j['name']: j for j in old_plan['jobs'] if j['benchmark'] == 'webwalkerqa'}
    if {j['name']: j for j in webwalker_plan['jobs']} != old_ww:
        raise ValueError('Repair must preserve WebWalker tasks, seeds, methods and ratios')
    if old_plan['tasks']['webwalkerqa'] != webwalker_plan['tasks']['webwalkerqa']:
        raise ValueError('Repair changed frozen WebWalker task content or order')
    for key in ('models', 'versions', 'budget'):
        if old_plan[key] != webwalker_plan[key]:
            raise ValueError(f'Repair unexpectedly changed shared {key}')
    if old_plan['prompts']['webwalkerqa'] != webwalker_plan['prompts']['webwalkerqa']:
        raise ValueError('This authorized repair preserves the existing WebWalker prompt')
    if any(old_plan['methods'][name] != webwalker_plan['methods'][name] for name in old_ww):
        raise ValueError('Repair unexpectedly changed a WebWalker method configuration')
    return selected


def eligibility(conditions, complete, scored):
    """New WebWalker qualification has priority without stopping BrowseComp."""
    jobs = [record['job'] for record in conditions.values()]
    qualified = {benchmark: {j['name'] for j in jobs
                            if j['benchmark'] == benchmark and j['purpose'] == 'smoke'} <= complete
                 for benchmark in BENCHMARKS}
    seed = next((s for s in SEEDS if not
                 {j['name'] for j in jobs if j['purpose'] == 'main' and j['seed'] == s}
                 <= complete & scored), None)
    eligible = sorted((j for j in jobs if j['benchmark'] == 'webwalkerqa' and j['purpose'] == 'smoke'),
                      key=lambda j: (0 if j['model'] == 'llama31_8b' else 1,
                                     0 if j['method'] == 'fullkv' else 1))
    eligible.extend(j for j in jobs if j['benchmark'] == 'browsecomp_plus' and j['purpose'] == 'smoke')
    if seed is not None:
        # Alternate benchmark conditions after qualification so a repaired
        # WebWalker condition does not wait behind the whole BrowseComp seed.
        groups = [[j for j in jobs if j['benchmark'] == benchmark and j['purpose'] == 'main'
                   and j['seed'] == seed and qualified[benchmark]] for benchmark in BENCHMARKS]
        for index in range(max(map(len, groups))):
            for group in groups:
                if index < len(group):
                    eligible.append(group[index])
    return seed, qualified, eligible


def load_routes(old_path, ww_path):
    if old_path == ww_path:
        raise RuntimeError('A repaired benchmark requires a separate frozen plan')
    paths = dict(zip(BENCHMARKS, (old_path, ww_path), strict=True))
    routes = {}
    for benchmark, path in paths.items():
        preliminary = json.loads(path.read_text())
        output = Path(preliminary['output']).resolve()
        if output != path.parent:
            raise RuntimeError('Plan left its original output directory')
        snapshot = output / 'source_snapshot'
        actor = load_actor_module(snapshot, repaired=benchmark == 'webwalkerqa')
        plan = actor.verify_plan(path)
        if benchmark == 'webwalkerqa':
            load_action_format_helper(snapshot, plan)
        if tuple(actor.SEEDS) != SEEDS:
            raise RuntimeError('Frozen seed order changed')
        routes[benchmark] = dict(actor=actor, plan=plan, path=path, output=output, snapshot=snapshot)
    old, repaired = (routes[name]['plan'] for name in BENCHMARKS)
    conditions = routed_jobs(old, repaired)
    for job in old['jobs']:
        if job['benchmark'] == 'webwalkerqa' and job['purpose'] == 'main':
            if (routes['browsecomp_plus']['output'] / 'jobs' / job['name'] / 'launch.json').exists():
                raise RuntimeError('Old WebWalker main launch exists; preserve it and review the transition explicitly')
    scorer = importlib.import_module('scripts.webbench_score_queue_v1')
    if Path(scorer.__file__).resolve() != routes['browsecomp_plus']['snapshot'] / 'scripts/webbench_score_queue_v1.py':
        raise RuntimeError('Score validator must come from the original frozen scoring source')
    for source in ('scripts/webbench_scoring_v1.py', 'scripts/webbench_score_queue_v1.py'):
        if old['source_hashes'][source] != repaired['source_hashes'][source]:
            raise RuntimeError('The repair must preserve the official scoring implementation')
    for benchmark, route in routes.items():
        p, plan = route['actor'], route['plan']
        path = route['output'] / 'score_queue_contract.json'
        contract = p.read(path)
        if (contract['plan_sha256'] != plan['sha256']
                or contract['scorer_queue_sha256'] != plan['source_hashes']['scripts/webbench_score_queue_v1.py']):
            raise RuntimeError('Score queue contract belongs to a different route')
        assets = Path(contract['assets'])
        for dependency, checksum in (
            (assets / 'model_ready.json', contract['model_ready_sha256']),
            (assets / 'sources/manifest.json', contract['sources_manifest_sha256']),
            (Path(contract['judge_runtime_manifest']['path']), contract['judge_runtime_manifest']['sha256']),
        ):
            if p.file_hash(dependency) != checksum:
                raise RuntimeError('Frozen scoring dependency changed')
        qrels = p.read(old['assets']['browsecomp_plus']['manifest'])['qrel_evidence']['path']
        route['scoring'] = dict(queue_contract=str(path), queue_contract_sha256=p.file_hash(path),
                                assets=str(assets), qrels=qrels, qrels_sha256=p.file_hash(qrels))
    return routes, conditions, scorer


def freeze(routes, conditions):
    original = routes['browsecomp_plus']
    p, output = original['actor'], original['output']
    folder = output / 'scheduler/seed_order_v3'
    source = Path(__file__).resolve()
    snapshot = folder / 'webbench_seed_queue_v3.py'
    binding = {benchmark: dict(plan=str(route['path']), plan_sha256=route['plan']['sha256'],
                              plan_file_sha256=p.file_hash(route['path']), output=str(route['output']),
                              source_snapshot=str(route['snapshot']), scoring=route['scoring'])
               for benchmark, route in routes.items()}
    expected = dict(schema='webbench_benchmark_routes_v3', routes=binding,
                    scheduler_snapshot=str(snapshot), scheduler_source_sha256=p.file_hash(source),
                    user_order=USER_ORDER, seed_order=list(SEEDS),
                    conditions={name: record['route'] for name, record in conditions.items()},
                    excluded_original_conditions=[j['name'] for j in original['plan']['jobs']
                                                  if j['benchmark'] == 'webwalkerqa'],
                    supersession='Original WebWalker smoke is retained as historical qualification only; '
                                 'all repaired WebWalker methods/models rerun smoke and main under the new plan.',
                    gate='Per benchmark 32 smoke conditions must qualify. Existing BrowseComp workers '
                         'continue during repaired WebWalker qualification. Both benchmarks must finish '
                         'all 64 actor conditions and validated official scores for a seed before the next.',
                    admission='Repaired WebWalker smoke has priority: Llama FullKV first, remaining Llama, '
                              'then Qwen; condition task order is unchanged. Then alternate benchmark admissions. '
                              'One original-run controller/resource lock and global actor reservations.',
                    reservation_compatibility='Top-level plan_sha256 remains the original resource-owner plan '
                                              'for its frozen scorer; every actor record identifies its actual '
                                              'worker_plan_sha256, route and output.',
                    failures='Preserved without automatic question or scoring retries; later seeds stay blocked.')
    path = folder / 'contract.json'
    with p.lock(folder / 'contract.lock'):
        if path.exists():
            saved = p.read(path)
            checksum = saved.pop('sha256')
            if p.digest(saved) != checksum or any(saved.get(k) != v for k, v in expected.items()):
                raise RuntimeError('Existing route contract differs; never overwrite it')
            if p.file_hash(snapshot) != expected['scheduler_source_sha256']:
                raise RuntimeError('Frozen scheduler snapshot changed')
            return {**saved, 'sha256': checksum}
        if snapshot.exists() and p.file_hash(snapshot) != expected['scheduler_source_sha256']:
            raise RuntimeError('Uncommitted scheduler snapshot differs')
        if not snapshot.exists():
            temporary = snapshot.with_name(snapshot.name + f'.{os.getpid()}.tmp')
            shutil.copy2(source, temporary)
            if p.file_hash(temporary) != expected['scheduler_source_sha256']:
                raise RuntimeError('Scheduler source changed during freeze')
            os.replace(temporary, snapshot)
        contract = {**expected, 'created': time.time()}
        contract['sha256'] = p.digest(contract)
        p.write(path, contract)
        return contract


def scan_scores(routes, conditions, scorer, complete, scored, failures):
    owner = routes['browsecomp_plus']
    for name, item in conditions.items():
        job = item['job']
        if job['purpose'] != 'main':
            continue
        route = routes[item['route']]
        p, plan = route['actor'], route['plan']
        directory = route['output'] / 'jobs' / name
        local = owner['output'] / 'scheduler/seed_order_v3/score_validation_failures' / name / 'failure.json'
        markers = [directory / 'scoring/queue_failure.json', local]
        marker = next((path for path in markers if path.exists()), None)
        if marker:
            record = p.read(marker)
            if record.get('plan_sha256') != plan['sha256'] or record.get('job') != job:
                raise RuntimeError('Score failure belongs to another route or condition')
            failures[name] = record.get('error', 'recorded scoring failure')
            scored.pop(name, None)
        elif name not in scored and name not in failures and name in complete and (directory / 'scoring/scores.json').exists():
            try:
                summary = scorer.validate_scores(plan, job, directory, route['scoring']['assets'], route['scoring']['qrels'])
                if summary.get('status') not in scorer.TERMINAL_SCORES:
                    raise RuntimeError('Score validator did not return a terminal recorded outcome')
                scored[name] = summary
            except Exception as exc:
                failures[name] = str(exc)
                p.write(local, dict(status='scoring_failed', plan_sha256=plan['sha256'],
                                    job=job, error=str(exc), time=time.time()))


def routed_record(record, route, job):
    if record.get('job') != job:
        raise RuntimeError('Launch identity differs from the routed frozen job')
    if record.get('worker_plan_sha256', route['plan']['sha256']) != route['plan']['sha256']:
        raise RuntimeError('Worker came from another frozen plan')
    return {**record, 'route': job['benchmark'], 'worker_plan_sha256': route['plan']['sha256'],
            'worker_output': str(route['output'])}


def queue(routes, conditions, scorer, contract):
    owner = routes['browsecomp_plus']
    p, plan, output = owner['actor'], owner['plan'], owner['output']
    with p.lock(output / 'controller.lock'):
        active, launches, children = {}, {}, {}
        previous_path = output / 'queue_status.json'
        previous = p.read(previous_path) if previous_path.exists() else {}
        if previous and previous.get('plan_sha256') != plan['sha256']:
            raise RuntimeError('Previous controller belongs to another resource owner')
        for name, record in previous.get('active', {}).items():
            if not p.identity(record['identity']['pid']) == record['identity']:
                continue
            if name not in conditions:
                raise RuntimeError('Live worker is excluded from the authorized routes')
            item = conditions[name]
            route = routes[item['route']]
            if item['route'] == 'webwalkerqa' and record.get('worker_plan_sha256') != route['plan']['sha256']:
                raise RuntimeError('Live original WebWalker worker requires explicit transition review')
            active[name] = routed_record(record, route, item['job'])
        for name, item in conditions.items():
            route = routes[item['route']]
            path = route['output'] / 'jobs' / name / 'launch.json'
            if path.exists():
                record = routed_record(p.read(path), route, item['job'])
                launches[name] = record
                if p.identity(record['identity']['pid']) == record['identity']:
                    active[name] = record
        validated, scored, scoring_failures = set(), {}, {}
        while True:
            for route in routes.values():
                binding = route['scoring']
                if p.file_hash(binding['queue_contract']) != binding['queue_contract_sha256']:
                    raise RuntimeError('A routed scoring queue contract changed')
            complete, failed = set(), {}
            for name, item in conditions.items():
                route = routes[item['route']]
                directory = route['output'] / 'jobs' / name
                status = p.read(directory / 'status.json') if (directory / 'status.json').exists() else {}
                if status.get('status') == 'failed' or (directory / 'failure.json').exists():
                    failed[name] = status.get('error', 'recorded actor failure')
                elif status.get('status') in ('complete', 'complete_with_environment_failures'):
                    try:
                        if name not in validated:
                            route['actor'].validate_completed(route['plan'], item['job'], directory)
                            validated.add(name)
                        complete.add(name)
                    except Exception as exc:
                        failed[name] = f'Final actor audit failed: {exc}'
            for name, record in {**launches, **active}.items():
                if name in children:
                    children[name].poll()
                if p.identity(record['identity']['pid']) != record['identity']:
                    if name not in complete and name not in failed:
                        route = routes[conditions[name]['route']]
                        p.write(route['output'] / 'jobs' / name / 'failure.json', dict(
                            type='WorkerExited', time=time.time(), identity=record['identity'],
                            error='Worker exited without terminal status; no automatic question rerun'))
                        failed[name] = 'Worker exited without terminal status'
                    active.pop(name, None)
            scan_scores(routes, conditions, scorer, complete, scored, scoring_failures)
            seed, qualified, eligible = eligibility(conditions, complete, set(scored))
            pending = [job for job in eligible if job['name'] not in complete | set(failed) | set(active)]
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
                        reserved = (reservation.get('status') in ('admitting', 'running')
                                    and any(who and p.identity(who['pid']) == who
                                            for who in (reservation.get('owner'), reservation.get('child'))))
                        if reserved and reservation.get('plan_sha256') != plan['sha256']:
                            raise RuntimeError('Live scorer reservation has a different resource owner')
                        if ((reserved and reservation.get('gpu') == gpu)
                                or not p.can_admit(plan, active, gpu, inventory[gpu])):
                            continue
                        job = pending.pop(0)
                        route = routes[job['benchmark']]
                        directory = route['output'] / 'jobs' / job['name']
                        directory.mkdir(parents=True, exist_ok=True)
                        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '2',
                               'MKL_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2',
                               'TOKENIZERS_PARALLELISM': 'false', 'PYTHONPATH': str(route['snapshot']),
                               'PYTHONUNBUFFERED': '1'}
                        command = [str(route['actor'].PYTHON), '-B', '-u', '-m', 'scripts.webbench_v1',
                                   'worker', '--plan', str(route['path']), '--job', job['name'], '--gpu', str(gpu)]
                        with (directory / 'worker.log').open('a') as log:
                            child = subprocess.Popen(command, cwd=route['snapshot'], env=env,
                                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                        record = routed_record(dict(identity=p.identity(child.pid), gpu=gpu,
                                                     gpu_uuid=inventory[gpu]['uuid'], started=time.time(), job=job), route, job)
                        if record['identity'] is None:
                            raise RuntimeError('Worker exited before durable launch admission')
                        p.write(directory / 'launch.json', {**record, 'command': command})
                        active[job['name']] = launches[job['name']] = record
                        children[job['name']] = child
                        p.write(output / 'actor_reservations.json', dict(
                            plan_sha256=plan['sha256'], scheduler_contract_sha256=contract['sha256'],
                            route_plans={b: r['plan']['sha256'] for b, r in routes.items()},
                            active=active, updated=time.time()))
                except BlockingIOError:
                    continue
            relevant = [item['job'] for item in conditions.values()
                        if item['job']['purpose'] == 'smoke' or item['job']['seed'] == seed]
            blocked = any(j['name'] in failed or j['name'] in scoring_failures for j in relevant)
            terminal_failure = not active and not pending and blocked
            pending_scoring = sum(j['purpose'] == 'main' and j['name'] in complete
                                  and j['name'] not in scored and j['name'] not in scoring_failures for j in relevant)
            state = dict(updated=time.time(), controller=p.identity(os.getpid()), plan_sha256=plan['sha256'],
                         scheduler_contract_sha256=contract['sha256'], scheduler=contract['scheduler_snapshot'],
                         route_plans={b: r['plan']['sha256'] for b, r in routes.items()}, seed_order=list(SEEDS),
                         current_stage='complete' if seed is None else f'seed{seed}',
                         status=('complete' if seed is None and not active else
                                 'qualification_failed' if terminal_failure and not all(qualified.values()) else
                                 'seed_failed' if terminal_failure else 'running' if active else
                                 'waiting_for_scoring' if pending_scoring and not pending else 'waiting_for_gpu'),
                         complete_conditions=len(complete), total_conditions=len(conditions),
                         smoke_complete=all(qualified.values()), benchmark_qualified=qualified,
                         active=active, failures=failed, scoring_failures=scoring_failures,
                         pending_scoring=pending_scoring, scored_conditions=len(scored),
                         later_stages_blocked_by_failure=blocked,
                         formal_complete_conditions=sum(item['job']['purpose'] == 'main' and name in complete
                                                        for name, item in conditions.items()),
                         stages=[dict(seed=s, actor_complete=sum(j['purpose'] == 'main' and j['seed'] == s and
                                      j['name'] in complete for j in (item['job'] for item in conditions.values())),
                                      scores_complete=sum(j['purpose'] == 'main' and j['seed'] == s and
                                      j['name'] in scored for j in (item['job'] for item in conditions.values())), total=64)
                                 for s in SEEDS])
            p.write(output / 'queue_status.json', state)
            # This status view lets the independent API scorer recognize a
            # failed repaired qualification without altering any worker output.
            ww = routes['webwalkerqa']
            p.write(ww['output'] / 'queue_status.json', {**state, 'plan_sha256': ww['plan']['sha256'],
                    'coordinator_output': str(output),
                    'active': {name: value for name, value in active.items() if value['route'] == 'webwalkerqa'}})
            if not active and (seed is None or terminal_failure):
                return
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'queue'))
    parser.add_argument('--old-plan', type=Path, required=True)
    parser.add_argument('--webwalker-plan', type=Path, required=True)
    args = parser.parse_args()
    routes, conditions, scorer = load_routes(args.old_plan.resolve(), args.webwalker_plan.resolve())
    contract = freeze(routes, conditions)
    if args.command == 'prepare':
        print(json.dumps(contract, ensure_ascii=False, indent=2))
    else:
        queue(routes, conditions, scorer, contract)


if __name__ == '__main__':
    main()
