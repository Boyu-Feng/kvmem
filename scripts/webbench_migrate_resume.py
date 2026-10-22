"""Verify, then explicitly resume a transferred frozen WebBench run.

The default operation is read-only.  Original absolute paths, frozen source,
plans, checkpoints, browser snapshots and judge calls are never rewritten.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time
from urllib.parse import urlsplit

sys.dont_write_bytecode = True
BASE = Path('/mnt/checkpoints/bruli/fengboyu/kvmem_webbench_20261021')
OLD = Path('/mnt/checkpoints/bruli/fengboyu/kvmem_rebuttal_20260929')
OWNER = BASE / 'runs/comparison_20261021_r01'
WEBWALKER = BASE / 'runs/comparison_20261021_webwalker_r04'
ACTOR_PYTHON = OLD / 'venv/bin/python'
JUDGE_PYTHON = BASE / 'runtime/judge_env/bin/python'
BROWSER_PYTHON = BASE / 'webwalker_runtime/venv/bin/python'
SCHEDULER = OWNER / 'scheduler/seed_order_v4/webbench_seed_queue_v4.py'
ROUTES = {'browsecomp_plus': OWNER, 'webwalkerqa': WEBWALKER}
EXPECTED_PLANS = {
    'browsecomp_plus': 'a79227d003e92b3736ad0fd64b041e4bf14c15243d47b44d252fc35ecde769ec',
    'webwalkerqa': '1a0afdde2dd1a2069175937ad5602b95541506592d3646914c914f0d3b525e36',
}
TERMINAL_ACTOR = {'complete', 'complete_with_environment_failures'}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(',', ':'))


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def host_identity():
    return {'hostname': socket.gethostname(),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'machine': platform.machine(), 'python': platform.python_version(),
            'platform': platform.platform()}


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return None if fields[0] == 'Z' else {'pid': int(pid), 'start_ticks': fields[19]}
    except (FileNotFoundError, ProcessLookupError):
        return None


@contextlib.contextmanager
def exclusive_lock(path):
    """Only used by --start; verification never creates a lock file."""
    with Path(path).open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def host_adapter():
    path = Path(__file__).resolve().with_name('webbench_migrate_host.py')
    spec = importlib.util.spec_from_file_location('_migration_host_adapter', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Verifier:
    def __init__(self, skip_weights=False, host_profile=None):
        self.skip_weights = skip_weights
        self.host_profile = host_profile
        self.checked = {}
        self.errors = []
        self.warnings = []
        self.plans = {}
        self.jobs = []
        self.gpus = []

    def check(self, name, function):
        try:
            return function()
        except Exception as exc:
            self.errors.append(f'{name}: {type(exc).__name__}: {exc}')
            return None

    def hashed(self, path, expected):
        path = Path(path)
        require(path.is_file(), f'Missing required file: {path}')
        if self.checked.get(str(path)) != expected:
            require(file_hash(path) == expected, f'SHA256 mismatch: {path}')
            self.checked[str(path)] = expected

    def inventory(self, root, inventory):
        for relative, entry in inventory.items():
            path = Path(root) / relative
            if self.skip_weights and path.suffix == '.safetensors':
                continue
            expected = entry['sha256'] if isinstance(entry, dict) else entry
            if isinstance(entry, dict) and 'bytes' in entry:
                require(path.is_file(), f'Missing required file: {path}')
                require(path.stat().st_size == entry['bytes'], f'File size differs: {path}')
            self.hashed(path, expected)

    def plan(self, benchmark, output):
        # Symlinks alter resolve() identities in the frozen routing contract.
        require(output.resolve() == output, f'Use real paths or bind mounts, not symlinks: {output}')
        plan = read(output / 'plan.json')
        saved = dict(plan)
        checksum = saved.pop('sha256')
        require(digest(saved) == checksum == EXPECTED_PLANS[benchmark], 'Frozen plan identity differs')
        require(plan['output'] == str(output), 'Frozen output path differs')
        require(plan['devices'] == [4, 5, 6, 7], 'Unexpected frozen GPU selectors')
        self.inventory(output / 'source_snapshot', plan['source_hashes'])
        for asset in plan['assets'].values():
            self.hashed(asset['manifest'], asset['sha256'])
        for asset in plan.get('runtime_manifests', []):
            self.hashed(asset['path'], asset['sha256'])
        for model in plan['models'].values():
            self.inventory(model['path'], model['hashes'])
        self.plans[benchmark] = plan
        return plan

    def assets(self):
        browse = read(BASE / 'assets/browsecomp/manifest.json')
        self.inventory('/', browse['files'])
        self.inventory('/', browse['source_hashes'])
        for source in browse['sources'].values():
            self.inventory(source['path'], source['files'])
        self.hashed(browse['qrel_evidence']['path'], browse['qrel_evidence']['sha256'])
        runtime = browse['retrieval']
        self.inventory(runtime['python_dependencies'], runtime['runtime_files'])
        self.inventory(runtime['java']['java_home'], runtime['java_files'])
        actual = {str(p.relative_to(runtime['python_dependencies']))
                  for p in Path(runtime['python_dependencies']).rglob('*')
                  if p.is_file() and '.cache' not in p.parts and '__pycache__' not in p.parts}
        require(actual == set(runtime['runtime_files']), 'Retrieval installation inventory differs')
        webwalker = read(BASE / 'assets/webwalkerqa/manifest.json')
        for asset in webwalker['assets'].values():
            self.hashed(asset['path'], asset['sha256'])
        self.hashed(webwalker['tasks']['path'], webwalker['tasks']['sha256'])
        browser = read(BASE / 'assets/webwalkerqa/provenance/crawler_runtime.json')
        self.inventory('/', browser['files'])
        sources = read(BASE / 'assets/judge/sources/manifest.json')
        self.inventory('/', sources['bindings'])
        judge = read(BASE / 'assets/judge/model_ready.json')
        self.inventory(judge['path'], judge['inventory'])

    def page_cache(self):
        root = BASE / 'assets/webwalkerqa/pages'
        renderer = read(root / 'renderer.json')
        require(renderer['runtime_manifest_sha256'] ==
                file_hash(BASE / 'assets/webwalkerqa/provenance/crawler_runtime.json'),
                'Browser cache renderer refers to another runtime')
        count = 0
        for path in root.glob('[0-9a-f][0-9a-f]/*.json'):
            record = read(path)
            value = record['payload']
            require(digest(value) == record['sha256'], f'Page record hash differs: {path}')
            require(value['renderer_sha256'] == digest(renderer), f'Page renderer differs: {path}')
            require(hashlib.sha256(value['url'].encode()).hexdigest() == path.stem,
                    f'Page URL identity differs: {path}')
            body = root / value['html']['path']
            require(body.resolve().is_relative_to(root), f'Invalid page object path: {path}')
            self.hashed(body, value['html']['sha256'])
            count += 1
        require(count > 0, 'Shared browser cache is empty; never regenerate prior observations')
        return count

    def scheduler(self):
        failures = list((OWNER / 'scheduler/seed_order_v4/score_validation_failures').rglob('failure.json'))
        require(not failures, 'Recorded frozen scheduler score validation failure requires separate recovery: '
                + ', '.join(str(path) for path in failures))
        contract_path = OWNER / 'scheduler/seed_order_v4/contract.json'
        contract = read(contract_path)
        body = dict(contract)
        checksum = body.pop('sha256')
        require(digest(body) == checksum, 'Frozen scheduler contract hash differs')
        self.hashed(SCHEDULER, contract['scheduler_source_sha256'])
        require(contract['scheduler_snapshot'] == str(SCHEDULER), 'Scheduler path differs')
        for benchmark, output in ROUTES.items():
            binding = contract['routes'][benchmark]
            require(binding['plan'] == str(output / 'plan.json') and
                    binding['output'] == str(output) and
                    binding['source_snapshot'] == str(output / 'source_snapshot'),
                    f'Frozen route path differs: {benchmark}')
            self.hashed(output / 'plan.json', binding['plan_file_sha256'])
            scoring = binding['scoring']
            self.hashed(scoring['queue_contract'], scoring['queue_contract_sha256'])
            self.hashed(scoring['qrels'], scoring['qrels_sha256'])
            queue = read(output / 'score_queue_contract.json')
            require(queue['endpoint'] == 'http://127.0.0.1:8313/v1', 'Unexpected frozen judge endpoint')
            require(queue['judge_python'] == str(JUDGE_PYTHON), 'Judge interpreter path differs')
            assets = Path(queue['assets'])
            self.hashed(assets / 'model_ready.json', queue['model_ready_sha256'])
            self.hashed(assets / 'sources/manifest.json', queue['sources_manifest_sha256'])
            self.hashed(queue['judge_runtime_manifest']['path'], queue['judge_runtime_manifest']['sha256'])

    def job(self, plan, job):
        directory = Path(plan['output']) / 'jobs' / job['name']
        if not directory.exists():
            return None
        checkpoint = directory / 'checkpoint.json'
        predictions = directory / 'predictions.json'
        status = read(directory / 'status.json') if (directory / 'status.json').exists() else {}
        require(not (directory / 'failure.json').exists() and status.get('status') != 'failed',
                f'Recorded actor failure requires separate recovery: {directory}')
        saved = read(checkpoint) if checkpoint.exists() else None
        rows = saved['results'] if saved else []
        if saved:
            require(saved['plan_sha256'] == plan['sha256'] and saved['job'] == job,
                    f'Checkpoint binding differs: {directory}')
        tasks = plan['tasks'][job['benchmark']][str(job['seed'])][job['purpose']]
        require(len(rows) <= len(tasks), f'Checkpoint too long: {directory}')
        if predictions.exists():
            final = read(predictions)
            require(saved == final and len(rows) == len(tasks),
                    f'Final/checkpoint disagreement: {directory}')
        elif status.get('status') in TERMINAL_ACTOR:
            raise RuntimeError(f'Terminal status without predictions: {directory}')
        expected_audits = set()
        for index, (row, task) in enumerate(zip(rows, tasks)):
            require(all(row.get(k) == task.get(t) for k, t in
                        [('id', 'id'), ('task_sha256', 'task_sha256'),
                         ('question', 'question'), ('gold_answer', 'answer')]),
                    f'Committed task binding differs: {directory}/{index}')
            relative = f'audits/{index:04d}.json'
            require(row['audit'] == relative, f'Unexpected audit path: {directory}/{index}')
            self.hashed(directory / relative, row['audit_sha256'])
            audit = read(directory / relative)
            require(audit['plan_sha256'] == plan['sha256'] and audit['job'] == job
                    and audit['task_sha256'] == task['task_sha256'],
                    f'Audit binding differs: {directory}/{index}')
            expected_audits.add(f'{index:04d}.json')
        require({p.name for p in (directory / 'audits').glob('*.json')} == expected_audits,
                f'Orphan/missing audit: {directory}; restore a consistent snapshot')
        pending = directory / 'attempts' / f'{len(rows):04d}.json'
        expected_attempts = {f'{i:04d}.json' for i in range(len(rows))}
        if pending.exists():
            require(len(rows) < len(tasks), f'Attempt beyond terminal task: {directory}')
            expected_attempts.add(pending.name)
        require({p.name for p in (directory / 'attempts').glob('*.json')} == expected_attempts,
                f'Unexpected attempt set: {directory}')
        for name in expected_attempts:
            attempt = read(directory / 'attempts' / name)
            index = int(Path(name).stem)
            require(attempt['plan_sha256'] == plan['sha256'] and attempt['job'] == job
                    and attempt['task_sha256'] == tasks[index]['task_sha256'],
                    f'Attempt binding differs: {directory}/{name}')
        record = {'job': job['name'], 'directory': str(directory),
                  'plan_sha256': plan['sha256'], 'committed_samples': len(rows),
                  'complete': predictions.exists(),
                  'pending_attempt': str(pending) if pending.exists() else None,
                  'checkpoint_sha256': file_hash(checkpoint) if checkpoint.exists() else None}
        self.jobs.append(record)
        return record

    def runtime(self, python, manifest, mode):
        require(python.is_file(), f'Missing interpreter: {python}; restore Python 3.12.3 and the bundled venv')
        code = ('import importlib.metadata as m,json,sys; '
                'print(json.dumps({"python":sys.version.split()[0],'
                '"packages":{d.metadata["Name"].lower().replace("_","-"):d.version '
                'for d in m.distributions()}}))')
        output = subprocess.run([str(python), '-B', '-c', code], capture_output=True, text=True,
                                timeout=60)
        require(output.returncode == 0,
                f'{mode} interpreter failed: {output.stderr.strip()[-2000:]}')
        actual = json.loads(output.stdout)
        require(actual['python'] == '3.12.3', f'{mode} Python must be 3.12.3, got {actual["python"]}')
        if mode == 'judge':
            expected = manifest['distributions']
            require(actual['packages'] == expected,
                    'Judge installed distribution set differs: ' + canonical({
                        k: {'expected': expected.get(k), 'actual': actual['packages'].get(k)}
                        for k in sorted(set(expected) | set(actual['packages']))
                        if expected.get(k) != actual['packages'].get(k)}))
        else:
            expected = manifest['versions'] if mode == 'actor' else manifest['packages']
            for name, version in expected.items():
                require(actual['packages'].get(name.lower().replace('_', '-')) == version,
                        f'{mode} requires {name}=={version}')
        imports = {'actor': 'import torch,transformers,tokenizers,numpy,safetensors,accelerate',
                   'judge': 'import torch,vllm,transformers,tokenizers,numpy,safetensors',
                   'browser': 'import crawl4ai,playwright,patchright,lxml'}
        imported = subprocess.run([str(python), '-B', '-c', imports[mode]],
                                  capture_output=True, text=True, timeout=180,
                                  env={**os.environ, 'CUDA_VISIBLE_DEVICES': '',
                                       'PYTHONDONTWRITEBYTECODE': '1'})
        require(imported.returncode == 0,
                f'{mode} native/package import failed: {imported.stderr.strip()[-3000:]}')
        if mode == 'browser':
            libraries = BASE / 'webwalker_runtime/system_packages/usr/lib/x86_64-linux-gnu'
            for name in manifest['files']:
                if Path(name).name not in {'chrome', 'chrome-headless-shell'}:
                    continue
                launched = subprocess.run([name, '--version'], capture_output=True, text=True, timeout=30,
                    env={**os.environ, 'LD_LIBRARY_PATH': str(libraries) + ':' + os.environ.get('LD_LIBRARY_PATH', '')})
                require(launched.returncode == 0 and manifest['browser_version'] in launched.stdout,
                        f'Browser native binary/system libraries failed: {name}: '
                        f'{launched.stderr.strip()[-3000:]} {launched.stdout.strip()}')

    def hardware(self):
        require(sys.platform == 'linux' and platform.machine() == 'x86_64',
                'Bundled native dependencies require Linux x86_64')
        completed = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.free,driver_version',
                                    '--format=csv,noheader,nounits'], capture_output=True, text=True,
                                   timeout=30, check=True)
        for line in completed.stdout.splitlines():
            index, uuid, name, total, free, driver = [part.strip() for part in line.split(',')]
            self.gpus.append(dict(index=int(index), uuid=uuid, name=name, total_mib=int(total),
                                  free_mib=int(free), driver=driver))
        by_index = {row['index']: row for row in self.gpus}
        if self.host_profile is not None:
            adapter = host_adapter()
            profile = adapter.validate_profile(self.host_profile)
            required = profile['gpu_budget_gib'] + profile['headroom_gib']
            for index in profile['devices']:
                require(index in by_index, f'Host profile GPU {index} is not present')
                require(by_index[index]['free_mib'] >= required * 1024,
                        f'GPU {index} needs {required:g} GiB free for budget plus headroom')
            available = adapter.available_host_ram_gib()
            require(available >= profile['minimum_available_host_ram_gib'],
                    f'Host has {available:.1f} GiB available RAM; profile requires '
                    f'{profile["minimum_available_host_ram_gib"]:g} GiB for judge CPU offload')
            self.warnings.append('Host resource overrides are recorded separately from frozen nominal '
                                 'resources; A100 execution and numerical equivalence are not prequalified. '
                                 '--start runs a synthetic local-judge capacity probe before scheduling.')
            return
        for index in [4, 5, 6, 7]:
            require(index in by_index, f'Frozen scheduler requires physical GPU index {index}; no remapping is implemented')
            require(by_index[index]['total_mib'] >= 100 * 1024,
                    f'GPU {index} has less than 100 GiB required by the frozen local-judge admission policy')
            if by_index[index]['name'] != 'NVIDIA H200':
                self.warnings.append(f'GPU {index} differs from original NVIDIA H200; record hardware change; numerical identity is not guaranteed')

    def endpoint(self):
        require_endpoint(read(OWNER / 'score_queue_contract.json')['endpoint'])

    def run(self):
        for benchmark, output in ROUTES.items():
            self.check(f'{benchmark} plan/source/models', lambda b=benchmark, o=output: self.plan(b, o))
        self.check('benchmark/retrieval/browser/judge assets', self.assets)
        pages = self.check('immutable browser cache', self.page_cache)
        self.check('frozen scheduler and scoring contracts', self.scheduler)
        for benchmark, plan in self.plans.items():
            for job in plan['jobs']:
                if job['benchmark'] == benchmark:
                    self.check(job['name'], lambda p=plan, j=job: self.job(p, j))
        for mode, python, manifest_path in [
            ('actor', ACTOR_PYTHON, OWNER / 'plan.json'),
            ('judge', JUDGE_PYTHON, BASE / 'runtime/judge_env/runtime_ready.json'),
            ('browser', BROWSER_PYTHON, BASE / 'assets/webwalkerqa/provenance/crawler_runtime.json')]:
            self.check(f'{mode} environment', lambda m=mode, p=python, f=manifest_path: self.runtime(p, read(f), m))
        self.check('GPU/host prerequisites', self.hardware)
        self.check('local GPT-4.1 proxy', self.endpoint)
        # The actual frozen verifier also checks repaired action parsing and score bindings.
        frozen = self.check('frozen CPU/score verification', self.frozen)
        return {'schema': 'webbench_migration_verification_v1', 'created_at_utc': utcnow(),
                'status': 'verified' if not self.errors else 'blocked',
                'skip_model_weights': self.skip_weights, 'host': host_identity(),
                'host_profile': self.host_profile,
                'gpus': self.gpus, 'checked_hash_files': len(self.checked), 'cached_pages': pages,
                'route_plan_sha256': {b: p['sha256'] for b, p in self.plans.items()},
                'jobs': self.jobs, 'frozen_checks': frozen,
                'errors': self.errors, 'warnings': self.warnings}

    def frozen(self):
        require(ACTOR_PYTHON.is_file(), f'Missing actor interpreter: {ACTOR_PYTHON}')
        completed = subprocess.run([str(ACTOR_PYTHON), '-B', str(Path(__file__).resolve()),
                                    '--internal-frozen-check'], cwd=OWNER / 'source_snapshot',
                                   env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1',
                                        'CUDA_VISIBLE_DEVICES': ''},
                                   capture_output=True, text=True, timeout=1800)
        require(completed.returncode == 0, completed.stderr.strip() or completed.stdout.strip())
        return json.loads(completed.stdout)


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def require_endpoint(url):
    endpoint = urlsplit(url)
    try:
        with socket.create_connection((endpoint.hostname, endpoint.port or 80), timeout=5):
            pass
    except OSError as exc:
        raise RuntimeError(f'Required local GPT-4.1 endpoint unavailable at {url}: {exc}') from exc


def frozen_check():
    """Runs with the actor interpreter, and never calls scheduler.freeze/queue."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('_migration_frozen_scheduler', SCHEDULER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    routes, conditions, scorer = module.load_routes(OWNER / 'plan.json', WEBWALKER / 'plan.json')
    counts = {'completed_actor_conditions': 0, 'completed_score_conditions': 0,
              'partial_scoring_conditions': 0, 'preserved_judge_calls': 0}
    for item in conditions.values():
        route, job = routes[item['route']], item['job']
        directory = route['output'] / 'jobs' / job['name']
        if (directory / 'predictions.json').exists():
            route['actor'].validate_completed(route['plan'], job, directory)
            counts['completed_actor_conditions'] += 1
        scoring = directory / 'scoring'
        permanent = OWNER / 'scheduler/seed_order_v4/score_validation_failures' / job['name'] / 'failure.json'
        require(not permanent.exists(), f'Recorded frozen scheduler score validation failure: {permanent}')
        require(not (scoring / 'queue_failure.json').exists(), f'Recorded score failure: {scoring}')
        require(not (scoring / 'failures').exists(), f'Recorded score failure history requires explicit review: {scoring}')
        if (scoring / 'scores.json').exists():
            scorer.validate_scores(route['plan'], job, directory,
                                   route['scoring']['assets'], route['scoring']['qrels'])
            counts['completed_score_conditions'] += 1
        elif (scoring / 'contract.json').exists():
            partial_contract(scorer.s, route, job, scoring)
            scoring_prefix(scorer.s, directory)
            counts['partial_scoring_conditions'] += 1
        counts['preserved_judge_calls'] += len(list((scoring / 'calls').glob('*.json')))
    print(canonical(counts))


def partial_contract(scoring_module, route, job, scoring):
    """Apply the frozen queue's static contract checks before resuming a scorer."""
    contract = read(scoring / 'contract.json')
    queue = read(route['output'] / 'score_queue_contract.json')
    assets = Path(route['scoring']['assets'])
    _, _, source = scoring_module.official(assets)
    expected = dict(benchmark=job['benchmark'], version=scoring_module.VERSION,
                    scorer_sha256=route['plan']['source_hashes']['scripts/webbench_scoring_v1.py'],
                    sources_sha256=digest(source), batch_size=1,
                    runtime_manifest=queue['judge_runtime_manifest'],
                    runtime=read(queue['judge_runtime_manifest']['path'])['runtime'])
    if job['benchmark'] == 'browsecomp_plus':
        expected.update(sampling=scoring_module.BCP_SAMPLING, judge_model=read(assets / 'model_ready.json'),
                        qrels_sha256=file_hash(route['scoring']['qrels']), gpu_memory_utilization=.65,
                        max_model_len=16384, engine_seed=0, tensor_parallel_size=1, judge_attempt_budget=1,
                        endpoint=None, webwalker_policy=None, expected_api_model=None)
    else:
        probe, policy = scoring_module.validate_webwalker_probe(assets, queue['endpoint'])
        expected.update(sampling=policy['sampling'], webwalker_policy=policy,
                        expected_api_model=probe['returned_model'], endpoint=queue['endpoint'],
                        judge_model=None, qrels_sha256=None, judge_attempt_budget=10,
                        engine_seed=None, tensor_parallel_size=None,
                        gpu_memory_utilization=None, max_model_len=None)
    require(all(contract.get(k) == v for k, v in expected.items()),
            f'Partial scoring contract differs from frozen source/runtime/policy: {scoring}')


def scoring_prefix(scoring_module, directory):
    """Validate partial score records and retain all requested/failed call counts."""
    scoring = directory / 'scoring'
    contract = read(scoring / 'contract.json')
    input_path, rows = scoring_module.load_predictions(directory / 'predictions.json')
    require(contract['predictions_path'] == str(input_path)
            and contract['predictions_sha256'] == file_hash(input_path)
            and contract['rows_sha256'] == digest(rows)
            and contract['ids'] == [row['id'] for row in rows], f'Partial scoring input changed: {scoring}')
    checksum = digest(contract)
    expected_budget = 1 if contract['benchmark'] == 'browsecomp_plus' else 10
    require(contract['judge_attempt_budget'] == expected_budget, f'Scoring attempt budget changed: {scoring}')
    known_items, known_calls = set(), set()
    for index, row in enumerate(rows):
        path = scoring / 'items' / f'{index:04d}.json'
        if path.exists():
            item = read(path)
            require(item['input_sha256'] == digest(row) and item['contract_sha256'] == checksum
                    and item['id'] == row['id'] and item['source_row_sha256'] == row['source_row_sha256'],
                    f'Partial score item binding changed: {path}')
            known_items.add(path.name)
        calls = list((scoring / 'calls').glob(f'{index:04d}_*.json'))
        returned = False
        for call in calls:
            value = read(call)
            require(value['input_sha256'] == digest(row) and value['contract_sha256'] == checksum
                    and value['id'] == row['id'] and value['status'] in {'requested', 'returned', 'failed'},
                    f'Prior judge call binding changed: {call}')
            returned |= value['status'] == 'returned'
            known_calls.add(call.name)
        require(len(calls) <= expected_budget, f'Prior judge call count exceeds budget: {scoring}/{index}')
        require(path.exists() or returned or len(calls) < expected_budget,
                f'Unfinished judge call exhausted its declared budget: {scoring}/{index}; never erase calls')
    require(known_items == {p.name for p in (scoring / 'items').glob('*.json')},
            f'Unknown partial score items: {scoring}')
    require(known_calls == {p.name for p in (scoring / 'calls').glob('*.json')},
            f'Unknown prior judge calls: {scoring}')


def validate_handoff(path):
    require(path is not None, '--start requires --handoff from the finalized export')
    value = read(path)
    require(value.get('schema') == 'webbench_migration_handoff_v1', 'Unsupported handoff schema')
    require(value.get('source_writers_stopped') is True,
            'Export is not finalized: source actors, score controllers and score children must stop writing before handoff')
    require(value.get('route_plan_sha256') == EXPECTED_PLANS, 'Handoff frozen plans differ')
    require(bool(value.get('captured_at_utc')), 'Handoff capture time missing')
    source = value.get('source_host', {})
    require(bool(source.get('hostname')) and bool(source.get('boot_id')), 'Handoff source host/boot identity missing')
    target = host_identity()
    require(source['boot_id'] != target['boot_id'],
            'This is still the source boot; use the existing local recovery procedure instead')
    return value


def ensure_no_local_workers():
    """Do not infer ownership from source-machine PID/start_ticks records."""
    matches = []
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv = path.read_bytes().split(b'\0')
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
        if int(path.parent.name) == os.getpid():
            continue
        joined = b' '.join(argv).decode(errors='replace')
        if any(str(root) in joined for root in ROUTES.values()) and any(
                word in joined for word in ('webbench_seed_queue', 'webbench_v1', 'webbench_migrate_host',
                                            'webbench_migrate_judge', 'webbench_score_queue', 'webbench_scoring_v1')):
            matches.append(path.parent.name)
    require(not matches, 'Target already has WebBench processes; refusing duplicate launch. PIDs: ' + ', '.join(matches))


def resume(report, handoff_path):
    handoff = validate_handoff(handoff_path)
    require(not report['errors'] and not report['skip_model_weights'],
            '--start requires successful full verification, including model weights')
    ensure_no_local_workers()
    endpoint = urlsplit(read(OWNER / 'score_queue_contract.json')['endpoint'])
    require_endpoint(endpoint.geturl())  # No judge request; actual returned models stay checked by the frozen scorer.
    folder = OWNER / 'migration' / ('host_' + host_identity()['boot_id'])
    require(not folder.exists(), f'Target migration already started or partially prepared: {folder}; inspect its manifest, never rerun blindly')
    with contextlib.ExitStack() as stack:
        for output in ROUTES.values():
            for name in ['controller.lock', 'scorer_controller.lock', 'resource_admission.lock']:
                stack.enter_context(exclusive_lock(output / name))
        for record in report['jobs']:
            directory = Path(record['directory'])
            stack.enter_context(exclusive_lock(directory / 'worker.lock'))
            if (directory / 'scoring').exists():
                stack.enter_context(exclusive_lock(directory / 'scoring/scoring.lock'))
        ensure_no_local_workers()
        preserved, movable = {}, []
        for output in ROUTES.values():
            # All these identities belong to another host. Keeping them in the
            # active namespace permits accidental PID reuse or WorkerExited.
            for name in ['queue_status.json', 'actor_reservations.json',
                         'scoring_status.json', 'scorer_reservation.json']:
                path = output / name
                if path.exists():
                    movable.append(path)
        for record in report['jobs']:
            directory = Path(record['directory'])
            if record['checkpoint_sha256']:
                require(file_hash(directory / 'checkpoint.json') == record['checkpoint_sha256'],
                        f'Checkpoint changed after verification: {directory}')
            for path in [directory / 'launch.json', directory / 'scoring/queue_launch.json']:
                if path.exists():
                    movable.append(path)
            if record['pending_attempt']:
                movable.append(Path(record['pending_attempt']))
            # Copy all raw evidence for interrupted conditions and partial scorers.
            if not record['complete'] or ((directory / 'scoring/contract.json').exists()
                                          and not (directory / 'scoring/scores.json').exists()):
                for path in directory.rglob('*'):
                    if path.is_file() and 'recovery' not in path.relative_to(directory).parts:
                        preserved[str(path)] = file_hash(path)
        for path in movable:
            preserved[str(path)] = file_hash(path)
        manifest = {'schema': 'webbench_host_migration_v1', 'state': 'validated',
                    'created_at_utc': utcnow(), 'handoff': handoff, 'handoff_sha256': file_hash(handoff_path),
                    'target_host': host_identity(), 'target_gpus': report['gpus'],
                    'recovery_script_sha256': file_hash(__file__), 'verification': report,
                    'preserved_files': preserved, 'move_after_archive': [str(p) for p in movable],
                    'policy': 'One explicit migration recovery of interrupted episodes; preserve all completed answers, '
                              'judge returns and requested/failed judge call counts. No frozen protocol changes.'}
        atomic_json(folder / 'manifest.json', manifest)
        for name, checksum in preserved.items():
            source = Path(name)
            target = folder / 'before' / source.relative_to(BASE)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            require(file_hash(target) == checksum, f'Archive copy differs: {source}')
        manifest['state'] = 'archived_before_mutation'
        atomic_json(folder / 'manifest.json', manifest)
        for source in movable:
            require(file_hash(source) == preserved[str(source)], f'Source changed before move: {source}')
            target = folder / 'moved_originals' / source.relative_to(BASE)
            target.parent.mkdir(parents=True, exist_ok=True)
            os.rename(source, target)
        for record in report['jobs']:
            if record['checkpoint_sha256']:
                require(file_hash(Path(record['directory']) / 'checkpoint.json') == record['checkpoint_sha256'],
                        'Migration must never alter a checkpoint')
        manifest['state'] = 'ready_to_resume'
        atomic_json(folder / 'manifest.json', manifest)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
           'OPENBLAS_NUM_THREADS': '2', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1'}
    commands = [
        ('browsecomp_scorer', OWNER, [str(ACTOR_PYTHON), '-B', '-u', '-m', 'scripts.webbench_score_queue_v1',
                                     'queue', '--plan', str(OWNER / 'plan.json'), '--endpoint', endpoint.geturl()]),
        ('webwalker_scorer', WEBWALKER, [str(ACTOR_PYTHON), '-B', '-u', '-m', 'scripts.webbench_score_queue_v1',
                                       'queue', '--plan', str(WEBWALKER / 'plan.json'), '--endpoint', endpoint.geturl()]),
        ('actor_coordinator', OWNER, [str(ACTOR_PYTHON), '-B', '-u', str(SCHEDULER), 'queue',
                                     '--old-plan', str(OWNER / 'plan.json'),
                                     '--webwalker-plan', str(WEBWALKER / 'plan.json')]),
    ]
    if report.get('host_profile') is not None:
        adapter = host_adapter()
        contract_path = folder / 'host_resources.json'
        contract = adapter.freeze_contract(report['host_profile'], contract_path)
        manifest.update(host_contract=str(contract_path), host_contract_sha256=contract['sha256'],
                        state='probing_judge_capacity')
        atomic_json(folder / 'manifest.json', manifest)
        for gpu in contract['devices']:
            command = [str(JUDGE_PYTHON), '-B', '-u', str(Path(__file__).resolve().with_name('webbench_migrate_judge.py')),
                       'probe', '--plan', str(OWNER / 'plan.json'), '--gpu', str(gpu),
                       '--host-contract', str(contract_path)]
            with (folder / f'judge_probe_gpu{gpu}.log').open('a') as log:
                child = subprocess.Popen(command, cwd=OWNER / 'source_snapshot',
                        env={**env, 'CUDA_VISIBLE_DEVICES': str(gpu),
                             'PYTHONPATH': str(OWNER / 'source_snapshot')},
                        stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                manifest.setdefault('judge_probes', {})[str(gpu)] = dict(
                    identity=process_identity(child.pid), command=command)
                atomic_json(folder / 'manifest.json', manifest)
                code = child.wait()
            if code:
                manifest.update(state='judge_probe_failed', failed_probe_gpu=gpu, probe_exit_code=code)
                atomic_json(folder / 'manifest.json', manifest)
                raise RuntimeError(f'Judge capacity probe failed on GPU {gpu}; no benchmark worker started. '
                                   f'Inspect {folder}/judge_probe_gpu{gpu}.log; no automatic retry.')
        commands = [('host_coordinator', OWNER, [str(ACTOR_PYTHON), '-B', '-u',
                    str(Path(__file__).resolve().with_name('webbench_migrate_host.py')), 'queue',
                    '--host-contract', str(contract_path)])]
    children = []
    for label, output, command in commands:
        with (folder / (label + '.log')).open('a') as log:
            child = subprocess.Popen(command, cwd=output / 'source_snapshot',
                                     env={**env, 'PYTHONPATH': str(output / 'source_snapshot')},
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
        identity = process_identity(child.pid)
        require(identity is not None, f'{label} exited before launch record; inspect {folder}')
        children.append((label, child))
        manifest.setdefault('controllers', {})[label] = dict(identity=identity, command=command,
                                                              cwd=str(output / 'source_snapshot'))
        manifest['state'] = 'starting_controllers'
        atomic_json(folder / 'manifest.json', manifest)
    time.sleep(3)
    for label, child in children:
        require(child.poll() is None, f'{label} exited with {child.returncode}; inspect {folder}/{label}.log')
    manifest['state'] = 'controllers_started'
    atomic_json(folder / 'manifest.json', manifest)
    return {'migration_manifest': str(folder / 'manifest.json'), 'controllers': manifest['controllers'],
            'note': 'Process startup verified; inspect queue_status and checkpoints for ongoing progress.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Read-only verification (the default)')
    parser.add_argument('--skip-model-weights', action='store_true',
                        help='Validate the transfer before downloading weights; incompatible with --start')
    parser.add_argument('--start', action='store_true', help='Archive source-host process records and interrupted attempts, then start the frozen controllers')
    parser.add_argument('--handoff', type=Path, help='Finalized source handoff JSON; mandatory for --start')
    parser.add_argument('--report', type=Path, help='Explicitly save the verification report (default: stdout only)')
    parser.add_argument('--host-profile', type=Path,
                        help='Explicit host-only resource profile, required for A100 / physical GPU 0–3')
    parser.add_argument('--devices', help='Comma-separated physical GPU indices overriding only the host profile')
    parser.add_argument('--internal-frozen-check', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.internal_frozen_check:
        frozen_check()
        return 0
    if args.start and (args.skip_model_weights or args.verify_only):
        parser.error('--start cannot be combined with --skip-model-weights or --verify-only')
    if args.start:
        validate_handoff(args.handoff)
    if args.devices and not args.host_profile:
        parser.error('--devices requires --host-profile; frozen plan devices are never rewritten')
    profile = read(args.host_profile) if args.host_profile else None
    if profile is not None:
        if args.devices:
            profile['devices'] = [int(value) for value in args.devices.split(',')]
        host_adapter().validate_profile(profile)
    print('Verifying original paths, immutable state, assets and isolated runtimes; large files are hashed.', file=sys.stderr)
    report = Verifier(args.skip_model_weights, profile).run()
    if args.report:
        atomic_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report['errors']:
        return 1
    if args.start:
        print(json.dumps(resume(report, args.handoff), ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'Migration stopped: {type(exc).__name__}: {exc}', file=sys.stderr)
        raise SystemExit(1)
