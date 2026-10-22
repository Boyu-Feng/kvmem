"""One immutable condition, with durable per-question commits and original scores."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from types import SimpleNamespace

from scripts.baseline_multiseed import protocol as p

io = p.io


def summary(rows):
    return {'samples': len(rows), 'em_percent': 100 * sum(r['em'] for r in rows) / len(rows) if rows else None,
            'f1_percent': 100 * sum(r['f1'] for r in rows) / len(rows) if rows else None,
            'diagnostic_seconds': sum(r['seconds'] for r in rows)}


def check_rows(rows, samples):
    if len(rows) > len(samples):
        raise RuntimeError('Too many committed rows')
    for row, (index, sample) in zip(rows, samples):
        if (row['index'], row['id'], row['question'], row['gold_answer']) != (
                index, sample['id'], sample['question'], sample['answer']):
            raise RuntimeError('Checkpoint is not an exact content-bound ordered prefix')
        if (row['em'] not in (False, True) or not math.isfinite(row['f1']) or not 0 <= row['f1'] <= 1
                or not isinstance(row['predicted_answer'], str)):
            raise RuntimeError('Invalid saved score or prediction')


def audit_read(output, row, checksum):
    path = (Path(output) / row['audit']['path']).resolve()
    if not path.is_relative_to(Path(output).resolve() / 'questions'):
        raise RuntimeError('Audit path escapes condition')
    if io.file_hash(path) != row['audit']['sha256']:
        raise RuntimeError('Missing or corrupt committed audit')
    value = io.read_json(path)
    if value['fingerprint_sha256'] != checksum or value['id'] != row['id']:
        raise RuntimeError('Audit identity mismatch')
    return value


class Store:
    def __init__(self, output, fp, samples):
        self.output, self.fp, self.samples = Path(output), fp, samples
        self.checksum, self.rows = io.digest(fp), []
        path = self.output / 'manifest.json'
        if not path.exists():
            if any((self.output / n).exists() for n in ('checkpoint.json', 'results.json', 'questions')):
                raise RuntimeError('Existing data without manifest')
            io.atomic_json(path, {'fingerprint': fp, 'fingerprint_sha256': self.checksum})
        self.manifest = io.read_json(path)
        if self.manifest['fingerprint'] != fp or self.manifest['fingerprint_sha256'] != self.checksum:
            raise RuntimeError('Changed condition fingerprint')
        cp = self.output / 'checkpoint.json'
        if cp.exists():
            data = io.read_json(cp)
            if data['fingerprint_sha256'] != self.checksum or data['rows_sha256'] != io.digest(data['results']):
                raise RuntimeError('Corrupt checkpoint')
            self.rows = data['results']
            check_rows(self.rows, samples)
            for row in self.rows:
                audit_read(self.output, row, self.checksum)

    def placement(self, actual):
        if 'actual_model' in self.manifest and self.manifest['actual_model'] != actual:
            raise RuntimeError('Actual loaded model or generation changed on resume')
        self.manifest['actual_model'] = actual
        self.manifest['actual_model_sha256'] = io.digest(actual)
        io.atomic_json(self.output / 'manifest.json', self.manifest)

    def commit(self, row, audit):
        check_rows(self.rows + [row], self.samples)
        if 'actual_model' not in self.manifest:
            raise RuntimeError('Missing actual model provenance')
        relative = f"questions/{len(self.rows):04d}_{hashlib.sha256(row['id'].encode()).hexdigest()[:16]}.json"
        io.atomic_json(self.output / relative, {**audit, 'id': row['id'], 'fingerprint_sha256': self.checksum})
        row = {**row, 'audit': {'path': relative, 'sha256': io.file_hash(self.output / relative)}}
        rows = self.rows + [row]
        io.atomic_json(self.output / 'checkpoint.json', {'fingerprint_sha256': self.checksum,
                       'rows_sha256': io.digest(rows), 'results': rows})
        self.rows = rows

    def finalize(self):
        if len(self.rows) != len(self.samples):
            raise RuntimeError('Cannot finalize incomplete condition')
        io.atomic_json(self.output / 'results.json', {'fingerprint_sha256': self.checksum,
                       'results': self.rows, 'summary': summary(self.rows)})


def validate(output, fp=None, samples=None):
    import run_all_wiki_experiments_v2 as base
    output = Path(output)
    manifest = io.read_json(output / 'manifest.json')
    saved_fp = manifest['fingerprint']
    checksum = io.digest(saved_fp)
    cp, final, status = [io.read_json(output / name) for name in ('checkpoint.json', 'results.json', 'status.json')]
    rows = cp['results']
    if (fp is not None and saved_fp != fp or manifest['fingerprint_sha256'] != checksum
            or cp['fingerprint_sha256'] != checksum or cp['rows_sha256'] != io.digest(rows)
            or final['fingerprint_sha256'] != checksum or final['results'] != rows
            or final['summary'] != summary(rows) or status['status'] != 'complete'
            or len(rows) != saved_fp['samples'] or [r['id'] for r in rows] != saved_fp['sample_ids']
            or len({r['id'] for r in rows}) != len(rows)
            or manifest['actual_model_sha256'] != io.digest(manifest['actual_model'])):
        raise RuntimeError(f'Incomplete/corrupt condition: {output}')
    if samples is not None:
        check_rows(rows, samples)
    for row in rows:
        audit_read(output, row, checksum)
        if (row['em'] != base.exact_match(row['predicted_answer'], row['gold_answer'])
                or row['f1'] != base.f1_score(row['predicted_answer'], row['gold_answer'])):
            raise RuntimeError('Saved score differs from original scorer')
    return {'samples': len(rows), 'summary': summary(rows), 'fingerprint_sha256': checksum}


class NeuralTrace:
    """Observe actual neural calls; no extra forwards or generation changes."""
    def __init__(self, model):
        self.model, self.rows, self.pending = model, [], {}
        self.main_thread = threading.get_ident()

    def __enter__(self):
        def before(module, args, kwargs):
            ids = kwargs.get('input_ids', args[0] if args else None)
            if ids is None:
                raise RuntimeError('Missing actual input token IDs')
            pos = kwargs.get('position_ids')
            self.pending[threading.get_ident()] = {'input_ids': ids.detach().cpu().tolist(),
                'position_ids': pos.detach().cpu().tolist() if pos is not None else None,
                'branch': 'main' if threading.get_ident() == self.main_thread else 'auxiliary'}

        def after(module, args, kwargs, result):
            row = self.pending.pop(threading.get_ident())
            row['last_logit_argmax'] = result.logits[:, -1].argmax(-1).detach().cpu().tolist()
            self.rows.append(row)
        self.handles = [self.model.register_forward_pre_hook(before, with_kwargs=True),
                        self.model.register_forward_hook(after, with_kwargs=True)]
        return self

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()


def actual_model(llm, model):
    from scripts.latency_runtime import validate_loaded_model
    validate_loaded_model(llm.model)
    expected = p.assets.MODELS[model['repo']]['config']
    if any(getattr(llm.model.config, k) != v for k, v in expected.items()):
        raise RuntimeError('Wrong actual model topology')
    generation = llm.model.generation_config.to_dict()
    if generation != model['loaded_generation_config']:
        raise RuntimeError('Loaded generation configuration mismatch')
    return {'repo': model['repo'], 'revision': model['revision'], 'topology': expected,
            'parameter_dtype': 'bfloat16', 'device': 'cuda:0',
            'layer_devices': [str(next(layer.parameters()).device) for layer in llm.model.model.layers],
            'attention_backend': llm.model.config._attn_implementation,
            'generation_config': generation, 'weight_sha256': model['weight_sha256']}


def actual_kv(llm):
    from scripts.qa32_runtime import cache_layout
    layout = cache_layout(llm)
    if any(row[k + '_dtype'] != 'torch.bfloat16' for row in layout for k in ('key', 'value')):
        raise RuntimeError('KV is not native BF16')
    lengths = [row['key_shape'][2] for row in layout]
    if not lengths or len(set(lengths)) != 1:
        raise RuntimeError('Missing or inconsistent actual KV layers')
    return {'resident_slots_per_head': lengths[0], 'bytes': sum(row['key_bytes'] + row['value_bytes'] for row in layout),
            'layout': layout, 'native_kv_heads': llm.model.config.num_key_value_heads,
            'query_heads': llm.model.config.num_attention_heads}


@contextlib.contextmanager
def make_llm(job, model, metadata):
    from scripts.latency_runtime import single_gpu_loader
    from scripts.baseline_multiseed.adapters import method_spec
    from token_tracker import TokenTracker
    with contextlib.ExitStack() as stack:
        stack.enter_context(single_gpu_loader())
        if job['method'] == 'sidequest_untrained':
            from scripts.sidequest_runtime import SideQuestPilot
            llm = SideQuestPilot(model['path'], interval=4, aux_max_tokens=128)
        elif job['method'] == 'thinkkv_eviction_only':
            from scripts.thinkkv_eviction_runtime import ThinKVEvictionOnlyLLM
            llm = ThinKVEvictionOnlyLLM(model['path'], job['ratio'])
        else:
            spec = stack.enter_context(method_spec(job['method'], job['model']))
            llm = spec.llm_class(model['path'], metadata['kv_config'], token_tracker=TokenTracker())
        try:
            yield llm
        finally:
            if hasattr(llm, 'close'):
                llm.close()


def run(output, job_name, gpu):
    from scripts.baseline_multiseed.queue import check_gpu
    plan = p.load_plan(output)
    job, = [j for j in plan['jobs'] if j['name'] == job_name]
    if gpu not in plan['devices'] or os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu):
        raise RuntimeError('Worker GPU does not match frozen admitted pool')
    fp, samples = p.fingerprint(plan, job), p.selected(plan, job)
    directory = Path(output) / job_name
    if job['purpose'] == 'main':
        smoke = next(j for j in plan['jobs'] if j['name'] == job_name.rsplit('/', 1)[0] + '/smoke')
        validate(Path(output) / smoke['name'], p.fingerprint(plan, smoke), p.selected(plan, smoke))
    from scripts.baseline_multiseed.queue import prerequisites
    for dependency in prerequisites(job):
        if dependency in plan['existing']:
            for name, checksum in plan['existing'][dependency]['file_hashes'].items():
                if io.file_hash(Path(plan['existing'][dependency]['path']) / name) != checksum:
                    raise RuntimeError('Skipped prerequisite changed')
        else:
            dep = next(j for j in plan['jobs'] if j['name'] == dependency)
            validate(Path(output) / dependency, p.fingerprint(plan, dep), p.selected(plan, dep))
    import torch
    import run_all_wiki_experiments_v2 as base
    from retrievers.WikiBM25Retriever import WikiBM25Retriever
    from scripts.qa32_runtime import QAErrorWatch
    model = plan['models'][job['model']]
    with io.output_lock(directory):
        store = Store(directory, fp, samples)
        def status(value, **extra):
            io.atomic_json(directory / 'status.json', {'status': value, 'pid': os.getpid(),
                'gpu': gpu, 'updated': time.time(), 'completed_samples': len(store.rows),
                'expected_samples': job['samples'], **extra})
        try:
            if len(store.rows) == job['samples']:
                store.finalize()
                status('complete')
                validate(directory, fp, samples)
                return
            check_gpu(gpu, allowed_pid=os.getpid())
            if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
                raise RuntimeError('Require one externally selected CUDA GPU')
            torch.cuda.set_per_process_memory_fraction(48 * 1024**3 / torch.cuda.get_device_properties(0).total_memory)
            status('checking_assets')
            args = SimpleNamespace(model_ready=Path(model['model_ready']), data_ready=Path(plan['data_ready']), dataset=job['dataset'])
            model_path, wiki, data, provenance, _ = p.assets.load_assets(args)
            if str(model_path) != model['path'] or any(provenance.get(k) != v for k, v in model['provenance'].items()):
                raise RuntimeError('Model/data asset provenance changed')
            actual_selected = p.select(base, data, job['seed'], job['purpose'])
            if io.digest(actual_selected) != io.digest(samples):
                raise RuntimeError('Frozen task input content changed')
            metadata = p.method_metadata(base, job['method'], job['model'], job['ratio'])
            if metadata != fp['method_protocol']:
                raise RuntimeError('Method configuration changed')
            status('loading_retriever')
            retriever = WikiBM25Retriever(str(wiki))
            if len(retriever.titles) != 5233235 or retriever.corpus_texts is None:
                raise RuntimeError('Incomplete shared Wikipedia retriever')
            base.MAX_STEPS, base.MODEL_PATH = 7, model['path']
            status('loading_model')
            with make_llm(job, model, metadata) as llm:
                store.placement(actual_model(llm, model))
                for index, sample in samples[len(store.rows):]:
                    p.verify_sources(plan)
                    status('running', current_id=sample['id'])
                    torch.cuda.reset_peak_memory_stats()
                    begin = time.monotonic()
                    watch_out, watch_err = QAErrorWatch(sys.stdout), QAErrorWatch(sys.stderr)
                    with contextlib.redirect_stdout(watch_out), contextlib.redirect_stderr(watch_err), NeuralTrace(llm.model) as trace:
                        if job['method'] == 'sidequest_untrained':
                            from scripts.run_sidequest_baseline import episode
                            answer, termination, trajectory = episode(llm, sample['question'], retriever, 7)
                            audit = llm.pilot_audit()
                            detail = {'termination': termination}
                        elif job['method'] == 'thinkkv_eviction_only':
                            from scripts.run_thinkkv_eviction import episode
                            answer, termination, trajectory = episode(llm, sample['question'], retriever, 7)
                            audit = llm.thin_audit()
                            detail = {'termination': termination}
                        else:
                            answer, trajectory, timings, debug = base._run_react_kv_episode(
                                sample['question'], llm, retriever, pruning_mode=metadata['mode'], return_debug=True)
                            audit = llm.legacy_audit()
                            detail = {'legacy_step_timings': timings, 'debug': debug}
                    if watch_out.errors or watch_err.errors or not trace.rows:
                        raise RuntimeError(f'Suppressed inference error or no neural calls: {watch_out.errors + watch_err.errors}')
                    kv = actual_kv(llm)
                    row = {'index': index, 'id': sample['id'], 'question': sample['question'],
                           'gold_answer': sample['answer'], 'predicted_answer': answer,
                           'em': base.exact_match(answer, sample['answer']),
                           'f1': base.f1_score(answer, sample['answer']), 'seconds': time.monotonic() - begin,
                           'actual_kv': kv, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()}
                    store.commit(row, {'trajectory': trajectory, 'adapter': audit, 'actual_kv': kv,
                                       'neural_calls': trace.rows, **detail})
                    status('running')
                    print(f"Committed {len(store.rows)}/{job['samples']}: {sample['id']}", flush=True)
                store.finalize()
                status('complete')
                validate(directory, fp, samples)
        except BaseException:
            status('failed', traceback=traceback.format_exc())
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--job')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    if args.validate:
        print(io.canonical(validate(args.output)))
    elif args.job and args.gpu is not None:
        run(args.output, args.job, args.gpu)
    else:
        parser.error('Worker requires --job and --gpu')


if __name__ == '__main__':
    main()
