"""Durable official answer scoring for the new WebBench comparison.

Actors never import this module or receive judge answers. BrowseComp uses the
vendored grader verbatim; WebWalker uses the public cot_qa evaluator's exact
template/parser from a pinned LangChain wheel. The upstream WebWalker project
did not pin LangChain, so this is an explicit new runtime, not a historical
environment claim. API/engine failures leave a condition incomplete.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import fcntl
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import re
import string
import sys
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ASSETS = Path('/mnt/checkpoints/bruli/fengboyu/kvmem_webbench_20261021/assets/judge')
VERSION = 'webbench_official_answer_scoring_v1'
BCP_SOURCE = ROOT / 'BrowseComp-Plus-main/scripts_evaluation/evaluate_run.py'
BCP_SHA = '1a21233937c377ab6323c98ff9af67742756a57fbacab4ebf9bc30852eae530a'
MODEL_REPO = 'Qwen/Qwen3-32B'
MODEL_REVISION = '9216db5781bf21249d130ec9da846c4624c16137'
LC_URL = 'https://files.pythonhosted.org/packages/f6/d5/4861816a95b2f6993f1360cfb605aacb015506ee2090433a71de9cca8477/langchain-0.3.27-py3-none-any.whl'
LC_SHA = '7b20c4f338826acb148d885b20a73a16e410ede9ee4f19bb02011852d5f98798'
WW_COMMIT = 'f72f75d8c3eb842f2bbbab096a12206ff66e270f'
WW_URL = f'https://raw.githubusercontent.com/Alibaba-NLP/DeepResearch/{WW_COMMIT}/WebAgent/WebWalker/src/evaluate.py'
WW_SHA = 'dfc3bc39be4b0b042842b2153533eb6edc00940646003cc8a6dc1cc2436fc4f8'
BCP_SAMPLING = dict(temperature=0.7, top_p=0.8, top_k=20, max_tokens=4096)
WW_SAMPLING = dict(model='gpt-4', seed=42, temperature=0)
NORMAL_TERMINATIONS = {'finish', 'step_budget', 'context_budget', 'invalid_action'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read(path):
    return json.loads(Path(path).read_text())


def immutable_bytes(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f'Existing source differs: {path}')
    else:
        with path.open('xb') as stream:
            stream.write(content)


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return response.read()


def extract_namespace(path, names):
    """Execute only exact named literals/functions, avoiding upstream imports.

    The complete upstream file is independently hash-bound. This permits CPU
    preflight without importing vLLM or installing LangChain into the actors.
    """
    tree = ast.parse(Path(path).read_text())
    nodes = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
    found = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            nodes.append(node)
            found.add(node.name)
        elif isinstance(node, ast.Assign):
            assigned = {target.id for target in node.targets if isinstance(target, ast.Name)}
            if assigned & names:
                nodes.append(node)
                found.update(assigned & names)
    if found != names:
        raise RuntimeError(f'Upstream symbols changed: {names - found}')
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    namespace = {'re': re, 'string': string, 'defaultdict': defaultdict, 'Path': Path}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace


def official(assets):
    assets = Path(assets)
    source_manifest = read(assets / 'sources/manifest.json')
    for path, checksum in source_manifest['bindings'].items():
        if file_sha(path) != checksum:
            raise RuntimeError(f'Changed scorer source: {path}')
    bcp = extract_namespace(assets / 'sources/browsecomp_evaluate_run.py',
        {'GRADER_TEMPLATE', 'create_judge_prompt', 'parse_judge_response',
         'extract_citations_from_response', 'compute_citation_metrics', 'load_qrel_data'})
    lc = extract_namespace(assets / 'sources/eval_prompt.py', {'cot_template'})
    lc.update(extract_namespace(assets / 'sources/eval_chain.py',
                                {'_get_score', '_parse_string_eval_output'}))
    return bcp, lc, source_manifest


def prepare_sources(assets):
    source = Path(assets) / 'sources'
    if file_sha(BCP_SOURCE) != BCP_SHA:
        raise RuntimeError('Vendored BrowseComp scorer changed')
    immutable_bytes(source / 'browsecomp_evaluate_run.py', BCP_SOURCE.read_bytes())
    wheel = fetch(LC_URL)
    if hashlib.sha256(wheel).hexdigest() != LC_SHA:
        raise RuntimeError('LangChain wheel hash mismatch')
    archive = zipfile.ZipFile(io.BytesIO(wheel))
    entries = ['langchain/evaluation/qa/eval_prompt.py',
               'langchain/evaluation/qa/eval_chain.py', 'langchain/evaluation/loading.py',
               'langchain-0.3.27.dist-info/licenses/LICENSE']
    for name in entries:
        immutable_bytes(source / Path(name).name, archive.read(name))
    webwalker_source = fetch(WW_URL)
    if hashlib.sha256(webwalker_source).hexdigest() != WW_SHA:
        raise RuntimeError('Pinned WebWalker scorer hash mismatch')
    immutable_bytes(source / 'webwalker_evaluate.py', webwalker_source)
    manifest = dict(version=VERSION, langchain_version='0.3.27', historical_langchain_version_known=False,
        langchain_wheel_url=LC_URL, langchain_wheel_sha256=LC_SHA,
        webwalker_url=WW_URL, browsecomp_vendored_path=str(BCP_SOURCE),
        bindings={str(source / name): file_sha(source / name) for name in
                  ['browsecomp_evaluate_run.py', 'eval_prompt.py', 'eval_chain.py',
                   'loading.py', 'LICENSE', 'webwalker_evaluate.py']},
        browsecomp_sampling=BCP_SAMPLING, webwalker_sampling=WW_SAMPLING,
        model_repo=MODEL_REPO, model_revision=MODEL_REVISION)
    existing = source / 'manifest.json'
    if existing.exists() and read(existing) != manifest:
        raise RuntimeError('Prepared source contract changed')
    atomic(existing, manifest)
    official(assets)
    return manifest


def download_model(assets):
    from huggingface_hub import snapshot_download
    assets = Path(assets)
    model = assets / 'Qwen3-32B'
    atomic(assets / 'download_status.json', dict(status='downloading', model=MODEL_REPO,
           revision=MODEL_REVISION, pid=os.getpid(), updated_unix=time.time()))
    snapshot_download(MODEL_REPO, revision=MODEL_REVISION, local_dir=model,
        allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.model', 'LICENSE*', 'README.md'],
        max_workers=8)
    index = read(model / 'model.safetensors.index.json')
    shards = sorted(set(index['weight_map'].values()))
    required = ['config.json', 'tokenizer.json', 'tokenizer_config.json',
                'generation_config.json', 'model.safetensors.index.json', *shards]
    inventory = {name: dict(bytes=(model / name).stat().st_size,
                            sha256=file_sha(model / name)) for name in required}
    ready = dict(repo=MODEL_REPO, revision=MODEL_REVISION, path=str(model.resolve()),
                 inventory=inventory, weight_bytes=sum(inventory[s]['bytes'] for s in shards))
    atomic(assets / 'model_ready.json', ready)
    atomic(assets / 'download_status.json', dict(status='complete', updated_unix=time.time(),
                                                model_ready_sha256=file_sha(assets / 'model_ready.json')))
    return ready


class JudgeFailure(RuntimeError):
    def __init__(self, message, evidence=None):
        super().__init__(message)
        self.evidence = evidence


def webwalker_policy(assets):
    """A later explicit user authorization can qualify a distinctly named judge.

    Creating this manifest is an operator action after approval, never an
    automatic fallback. Every qualification and score binds its exact content.
    """
    path = Path(assets) / 'webwalker_authorization.json'
    if not path.exists():
        return dict(label='public_default_gpt4', sampling=WW_SAMPLING,
            accepted_models=['gpt-4', 'gpt-4-0314', 'gpt-4-0613'], authorization=None)
    authorization = read(path)
    if (authorization.get('schema') != 'webwalker_judge_authorization_v1'
            or authorization.get('benchmark') != 'webwalkerqa'
            or authorization.get('authorized') is not True
            or authorization.get('authorized_by') != 'user'
            or not isinstance(authorization.get('user_instruction'), str)
            or not authorization['user_instruction'].strip()
            or not authorization.get('approved_at_utc')
            or authorization.get('requested_model') != 'gpt-4.1'
            or authorization.get('expected_api_model') != 'gpt-4.1-2025-04-14'):
        raise RuntimeError('Invalid explicit WebWalker judge authorization manifest')
    return dict(label='user_authorized_gpt41_changed_judge',
        sampling=dict(WW_SAMPLING, model=authorization['requested_model']),
        accepted_models=[authorization['expected_api_model']], authorization=authorization)


def validate_webwalker_probe(assets, endpoint):
    probe = read(Path(assets) / 'webwalker_probe.json')
    policy = webwalker_policy(assets)
    _, _, sources = official(assets)
    if (probe.get('benchmark_score') is not False or probe.get('synthetic') is not True
            or probe.get('parsed', {}).get('score') != 1
            or probe.get('returned_model') not in policy['accepted_models']
            or probe.get('endpoint') != endpoint
            or probe.get('source_manifest_sha256') != digest(sources)
            or probe.get('policy_sha256') != digest(policy)
            or probe.get('sampling') != policy['sampling']):
        raise RuntimeError('Invalid/mismatched synthetic WebWalker judge qualification')
    return probe, policy


def webwalker_request(prompt, endpoint='http://127.0.0.1:8313/v1', timeout=180, policy=None):
    policy = policy or dict(sampling=WW_SAMPLING, accepted_models=['gpt-4', 'gpt-4-0314', 'gpt-4-0613'])
    payload = dict(policy['sampling'], messages=[dict(role='user', content=prompt)])
    request = urllib.request.Request(endpoint.rstrip('/') + '/chat/completions',
        data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json',
        'Authorization': 'Bearer not-needed'}, method='POST')
    with urllib.request.urlopen(request, timeout=timeout) as response:
        value = json.load(response)
    model = value.get('model')
    # A proxy alias cannot silently change the approved judge identity.
    if model not in policy['accepted_models']:
        raise JudgeFailure(f'Unexpected actual GPT-4 judge identity: {model!r}', value)
    choice, = value['choices']
    if choice.get('finish_reason') != 'stop':
        raise JudgeFailure(f'Judge response did not finish: {choice.get("finish_reason")}', value)
    text = choice['message']['content']
    if not isinstance(text, str):
        raise JudgeFailure('Judge returned no text', value)
    return text, value


def probe_webwalker(assets, endpoint):
    _, lc, source = official(assets)
    policy = webwalker_policy(assets)
    prompt = lc['cot_template'].format(query='What is 2 + 2?', context='4', result='4')
    started = time.time()
    try:
        text, response = webwalker_request(prompt, endpoint, 180, policy)
    except BaseException as exc:
        atomic(Path(assets) / 'probes' / f'failed_{time.time_ns()}.json',
            dict(benchmark_score=False, synthetic=True, requested_model=policy['sampling']['model'],
                 endpoint=endpoint, prompt=prompt, started_unix=started,
                 completed_unix=time.time(), error=str(exc), evidence=getattr(exc, 'evidence', None)))
        raise
    parsed = lc['_parse_string_eval_output'](text)
    if parsed['score'] != 1:
        raise RuntimeError('Synthetic official cot_qa probe failed')
    record = dict(benchmark_score=False, synthetic=True, requested_model=policy['sampling']['model'],
        returned_model=response['model'], endpoint=endpoint, sampling=policy['sampling'],
        policy_sha256=digest(policy), policy=policy,
        source_manifest_sha256=digest(source), prompt=prompt, response=response,
        parsed=parsed, started_unix=started, completed_unix=time.time())
    atomic(Path(assets) / 'webwalker_probe.json', record)
    return record


def runtime_versions():
    result = dict(python=sys.version.split()[0])
    for package in ['vllm', 'torch', 'transformers', 'tokenizers', 'numpy', 'safetensors']:
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def validate_runtime_manifest(path):
    if not path:
        return None
    manifest = read(path)
    installed = {dist.metadata['Name'].lower().replace('_', '-'): dist.version
                 for dist in importlib.metadata.distributions()}
    if (manifest.get('cpu_import_verified') is not True
            or manifest.get('runtime') != runtime_versions()
            or manifest.get('distributions') != installed):
        raise RuntimeError('Judge runtime differs from its pinned, import-qualified manifest')
    return dict(path=str(Path(path).resolve()), sha256=file_sha(path))


def offline_check(assets, require_model=False):
    assets = Path(assets)
    _, _, sources = official(assets)
    model_path = assets / 'model_ready.json'
    if require_model and not model_path.exists():
        raise RuntimeError('Pinned Qwen3-32B model preparation is incomplete')
    if model_path.exists():
        model = read(model_path)
        if model['repo'] != MODEL_REPO or model['revision'] != MODEL_REVISION:
            raise RuntimeError('Judge model provenance differs')
        for name, info in model['inventory'].items():
            p = Path(model['path']) / name
            if not p.is_file() or p.stat().st_size != info['bytes']:
                raise RuntimeError('Judge model inventory incomplete: ' + name)
    probe_path = assets / 'webwalker_probe.json'
    probe = read(probe_path) if probe_path.exists() else None
    return dict(status='sources_verified', offline=True, gpu_used=False,
        source_manifest_sha256=digest(sources), browsecomp_model_ready=model_path.exists(),
        webwalker_probe_recorded=probe_path.exists(),
        webwalker_exact_gpt4_ready=bool(probe and probe.get('returned_model') in
                                       {'gpt-4', 'gpt-4-0314', 'gpt-4-0613'}),
        runtime=runtime_versions(),
        caveat='Full model file hashes and actual loaded dtype are verified at scoring startup.')


def load_predictions(path):
    path = Path(path)
    if path.is_dir():
        candidates = [path / n for n in ['predictions.json', 'predictions.jsonl'] if (path / n).is_file()]
        if len(candidates) != 1:
            raise ValueError('Prediction directory must contain one predictions.json or predictions.jsonl')
        path = candidates[0]
    value = ([json.loads(line) for line in path.read_text().splitlines() if line.strip()]
             if path.suffix == '.jsonl' else read(path))
    rows = value.get('results', value.get('predictions')) if isinstance(value, dict) else value
    if not isinstance(rows, list) or not rows:
        raise ValueError('Expected a nonempty list of prediction rows')
    normalized = []
    for row in rows:
        def field(*keys):
            for key in keys:
                if key in row:
                    return row[key]
            raise ValueError('Missing prediction field: ' + '/'.join(keys))
        key = str(field('id', 'query_id'))
        question = field('question', 'query')
        answer = field('gold_answer', 'answer', 'reference', 'correct_answer')
        prediction = field('predicted_answer', 'prediction', 'pred', 'response')
        termination = row.get('termination', 'finish')
        environment_failed = row.get('status') == 'environment_failure'
        if environment_failed:
            if row.get('inference_started') is not False or prediction is not None:
                raise ValueError('Environment failure must precede inference and have no prediction')
        elif row.get('infrastructure_error') or row.get('error') or termination not in NORMAL_TERMINATIONS:
            raise ValueError(f'Infrastructure/unknown termination cannot be scored: {key}')
        if (not key or key == 'None' or not all(isinstance(v, str) for v in (question, answer))
                or not environment_failed and not isinstance(prediction, str)):
            raise ValueError(f'Invalid strings in prediction: {key}')
        normalized.append(dict(id=key, question=question, gold_answer=answer,
            predicted_answer=prediction, termination=termination,
            environment_failed=environment_failed,
            inference_started=False if environment_failed else row.get('inference_started', True),
            retrieved_docids=row.get('retrieved_docids', []),
            tool_call_counts=row.get('tool_call_counts', {}), task_sha256=row.get('task_sha256'),
            source_row_sha256=digest(row)))
    if len({row['id'] for row in normalized}) != len(normalized):
        raise ValueError('Duplicate prediction IDs')
    return path.resolve(), normalized


def score_row(row, benchmark, bcp, lc, *, judge=None, endpoint=None, qrels=None):
    identity = dict(id=row['id'], input_sha256=digest(row), source_row_sha256=row['source_row_sha256'])
    if row.get('environment_failed'):
        return dict(identity, status='environment_failure', score=None, judge_called=False,
                    inference_started=False, missing=True)
    if benchmark == 'browsecomp_plus':
        positives = (qrels or {}).get(row['id'], [])
        retrieved = set(map(str, row['retrieved_docids']))
        recall = len(retrieved & set(positives)) / len(positives) if positives else None
        if not row['predicted_answer'] or row['termination'] != 'finish':
            return dict(identity, score=0, scientific_failure=True, judge_called=False,
                judge_result=dict(parse_error=True, error='Response incomplete or cannot be parsed'),
                retrieval_recall=recall)
        prompt = bcp['create_judge_prompt'](row['question'], row['predicted_answer'], row['gold_answer'])
        text, response = judge(prompt)
        parsed = bcp['parse_judge_response'](text)
        if parsed['parse_error'] or parsed['correct'] is None:
            raise RuntimeError('BrowseComp judge parse error; score remains missing')
        cited = bcp['extract_citations_from_response'](row['predicted_answer'])
        return dict(identity, score=int(parsed['correct']), judge_called=True, judge_prompt=prompt,
            judge_response=response, judge_result=parsed, retrieval_recall=recall,
            citations=dict(cited_docids=cited, metrics=bcp['compute_citation_metrics'](cited, positives)))
    prompt = lc['cot_template'].format(query=row['question'], context=row['gold_answer'],
                                       result=row['predicted_answer'])
    text, response = (judge(prompt) if judge else webwalker_request(prompt, endpoint))
    parsed = lc['_parse_string_eval_output'](text)
    if parsed['score'] not in (0, 1):
        raise RuntimeError('WebWalker judge parse error; score remains missing')
    return dict(identity, score=parsed['score'], judge_called=True, judge_prompt=prompt,
                judge_response=response, judge_result=parsed)


def score(args):
    assets, output = Path(args.assets), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'scoring.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        bcp, lc, sources = official(assets)
        input_path, rows = load_predictions(args.predictions)
        source_hash = file_sha(Path(__file__))
        model = read(assets / 'model_ready.json') if args.benchmark == 'browsecomp_plus' else None
        probe, policy = validate_webwalker_probe(assets, args.endpoint) if not model else (None, None)
        contract = dict(version=VERSION, benchmark=args.benchmark, scorer_sha256=source_hash,
            sources_sha256=digest(sources), predictions_path=str(input_path), predictions_sha256=file_sha(input_path),
            rows_sha256=digest(rows), ids=[r['id'] for r in rows],
            sampling=BCP_SAMPLING if model else policy['sampling'], webwalker_policy=policy,
            judge_model=model, runtime=runtime_versions(),
            runtime_manifest=validate_runtime_manifest(getattr(args, 'runtime_manifest', None)),
            endpoint=args.endpoint if not model else None,
            expected_api_model=probe['returned_model'] if probe else None,
            judge_attempt_budget=1 if model else 10,
            batch_size=1, engine_seed=0 if model else None,
            tensor_parallel_size=1 if model else None,
            gpu_memory_utilization=args.gpu_memory_utilization if model else None,
            max_model_len=args.max_model_len if model else None,
            qrels_sha256=file_sha(args.qrels) if args.qrels else None)
        path = output / 'contract.json'
        if path.exists() and read(path) != contract:
            raise RuntimeError('Scoring resume contract differs; preserve old run and use a new output')
        atomic(path, contract)
        records = []
        for index, row in enumerate(rows):
            saved = output / 'items' / f'{index:04d}.json'
            if saved.exists():
                value = read(saved)
                if value.get('input_sha256') != digest(row) or value.get('contract_sha256') != digest(contract):
                    raise RuntimeError('Scored item binding changed')
                records.append(value)
            else:
                records.append(None)
        def status(stage, **extra):
            atomic(output / 'status.json', dict(status=stage, total=len(rows),
                scored=sum(r is not None and r.get('score') is not None for r in records),
                environment_failed=sum(r is not None and r.get('status') == 'environment_failure' for r in records),
                recorded=sum(r is not None for r in records),
                updated_unix=time.time(), pid=os.getpid(), **extra))
        status('preparing_judge')
        try:
            judge = None
            if model and any(r is None and row['predicted_answer'] and row['termination'] == 'finish'
                             for row, r in zip(rows, records)):
                if not os.environ.get('CUDA_VISIBLE_DEVICES') or ',' in os.environ['CUDA_VISIBLE_DEVICES']:
                    raise RuntimeError('Select one capacity-admitted physical GPU for the judge')
                for name, info in model['inventory'].items():
                    p = Path(model['path']) / name
                    if not p.is_file() or p.stat().st_size != info['bytes'] or file_sha(p) != info['sha256']:
                        raise RuntimeError('Judge weight/tokenizer identity changed: ' + name)
                import torch
                from vllm import LLM, SamplingParams
                llm = LLM(model=model['path'], tokenizer=model['path'], dtype='bfloat16',
                    tensor_parallel_size=1, seed=0, gpu_memory_utilization=args.gpu_memory_utilization,
                    max_model_len=args.max_model_len, enforce_eager=True, max_num_seqs=1)
                provenance = dict(repo=MODEL_REPO, revision=MODEL_REVISION,
                    config=llm.llm_engine.model_config.hf_config.to_dict(),
                    runtime=runtime_versions(), dtype=str(llm.llm_engine.model_config.dtype),
                    device=torch.cuda.get_device_name(0), visible_device=os.environ['CUDA_VISIBLE_DEVICES'])
                if provenance['dtype'] != 'torch.bfloat16':
                    raise RuntimeError('Judge dtype differs')
                atomic(output / 'actual_judge.json', provenance)
                sampling = SamplingParams(**BCP_SAMPLING)
                def judge(prompt):
                    batch = llm.chat([[dict(role='user', content=prompt)]], sampling,
                        chat_template_kwargs={'enable_thinking': False}, use_tqdm=False)
                    result, = batch
                    completion, = result.outputs
                    if completion.finish_reason not in ('stop', 'length'):
                        raise RuntimeError('Judge generation did not finish')
                    return completion.text, dict(text=completion.text, token_ids=completion.token_ids,
                        prompt_token_ids=result.prompt_token_ids, finish_reason=completion.finish_reason,
                        stop_reason=completion.stop_reason, model=MODEL_REPO, revision=MODEL_REVISION)
            qrels = bcp['load_qrel_data'](Path(args.qrels)) if args.qrels else {}
            for index, row in enumerate(rows):
                if records[index] is not None:
                    continue
                status('scoring', current_id=row['id'])
                def recorded_judge(prompt):
                    previous = sorted((output / 'calls').glob(f'{index:04d}_*.json'))
                    for prior in previous:
                        saved = read(prior)
                        if (saved['input_sha256'] != digest(row)
                                or saved['contract_sha256'] != digest(contract) or saved['prompt'] != prompt):
                            raise RuntimeError('Prior judge call input/contract changed')
                        if saved['status'] == 'returned':
                            response = saved['response']
                            if probe and response['model'] != probe['returned_model']:
                                raise JudgeFailure('Actual API judge changed from the qualified identity', response)
                            # A returned judgement is immutable, including a parse failure.
                            # Restarting must never sample a more convenient judgement.
                            return saved['text'], response
                    if len(previous) >= contract['judge_attempt_budget']:
                        raise RuntimeError('Declared judge call attempt budget exhausted')
                    request_record = dict(id=row['id'], input_sha256=digest(row),
                        contract_sha256=digest(contract), prompt=prompt, started_unix=time.time())
                    request_path = output / 'calls' / f'{index:04d}_{time.time_ns()}.json'
                    atomic(request_path, dict(request_record, status='requested'))
                    try:
                        text, response = judge(prompt) if judge else webwalker_request(prompt, args.endpoint, 180, policy)
                        atomic(request_path, dict(request_record, status='returned', text=text,
                                                  response=response, completed_unix=time.time()))
                        if probe and response['model'] != probe['returned_model']:
                            raise JudgeFailure('Actual API judge changed from the qualified identity', response)
                        return text, response
                    except BaseException as exc:
                        atomic(request_path, dict(request_record, status='failed', error=str(exc),
                            evidence=getattr(exc, 'evidence', None), completed_unix=time.time()))
                        raise
                item = score_row(row, args.benchmark, bcp, lc, judge=recorded_judge,
                                 endpoint=args.endpoint, qrels=qrels)
                item.update(contract_sha256=digest(contract), completed_unix=time.time())
                atomic(output / 'items' / f'{index:04d}.json', item)
                records[index] = item
            if file_sha(input_path) != contract['predictions_sha256'] or file_sha(Path(__file__)) != source_hash:
                raise RuntimeError('Scoring input/source changed during execution')
            scored = [r for r in records if r['score'] is not None]
            missing = len(records) - len(scored)
            stage = 'incomplete_environment_failures' if missing else 'complete'
            correct = sum(r['score'] for r in scored)
            summary = dict(status=stage, benchmark=args.benchmark, total=len(rows), requested=len(rows),
                inferred=sum(row['inference_started'] for row in rows), scored=len(scored),
                environment_failed=missing, correct=correct,
                accuracy=None if missing else correct / len(rows),
                scored_subset_accuracy=correct / len(scored) if scored else None,
                contract_sha256=digest(contract), results=records,
                infrastructure_failures_are_not_zero=True)
            atomic(output / 'scores.json', summary)
            status(stage, accuracy=summary['accuracy'])
            return summary
        except BaseException as exc:
            failure = dict(error_type=type(exc).__name__, error=str(exc), updated_unix=time.time(),
                           contract_sha256=digest(contract),
                           scored=sum(r is not None and r.get('score') is not None for r in records))
            atomic(output / 'failures' / f'{time.time_ns()}.json', failure)
            status('infrastructure_failure', error_type=type(exc).__name__, error=str(exc))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--assets', default=str(DEFAULT_ASSETS))
    prepare.add_argument('--download-model', action='store_true')
    prepare.add_argument('--probe-webwalker', action='store_true')
    prepare.add_argument('--endpoint', default='http://127.0.0.1:8313/v1')
    check = sub.add_parser('check')
    check.add_argument('--assets', default=str(DEFAULT_ASSETS))
    check.add_argument('--require-model', action='store_true')
    run = sub.add_parser('score')
    run.add_argument('--benchmark', choices=['browsecomp_plus', 'webwalkerqa'], required=True)
    run.add_argument('--predictions', required=True)
    run.add_argument('--output', required=True)
    run.add_argument('--assets', default=str(DEFAULT_ASSETS))
    run.add_argument('--endpoint', default='http://127.0.0.1:8313/v1')
    run.add_argument('--qrels')
    run.add_argument('--runtime-manifest')
    run.add_argument('--gpu-memory-utilization', type=float, default=0.6)
    run.add_argument('--max-model-len', type=int, default=16384)
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare_sources(args.assets)
        if args.probe_webwalker:
            probe_webwalker(args.assets, args.endpoint)
        if args.download_model:
            download_model(args.assets)
        print(json.dumps(dict(status='prepared', assets=args.assets)))
    elif args.command == 'check':
        print(json.dumps(offline_check(args.assets, args.require_model)))
    else:
        if not 0 < args.gpu_memory_utilization < 1 or args.max_model_len < 8192:
            parser.error('Judge capacity must be bounded and context >=8192')
        result = score(args)
        print(json.dumps({k: v for k, v in result.items() if k != 'results'}))


if __name__ == '__main__':
    main()
