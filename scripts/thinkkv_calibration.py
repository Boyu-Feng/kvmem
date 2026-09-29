"""ThinkKV paper Algorithm 1 preflight on held-out QA prompts.

This collector leaves the original generator and its attention outputs unchanged.
It records real decode queries, never synthetic EOS probes or keyword labels.
The strict paper gate requires three KDE modes in the same layers for every
calibration prompt. No quantile/keyword fallback is permitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde

from run_rebuttal_legacy import atomic_json, verify_original_sources


def kde_thresholds(values, grid_size=2049):
    """Scott-bandwidth Gaussian KDE; exactly three interior modes required."""
    values = np.asarray(values, dtype=float)
    if len(values) < 32 or not np.isfinite(values).all() or np.std(values) < 1e-8:
        return {"status": "insufficient_variation", "n": len(values), "modes": []}
    kde = gaussian_kde(values, bw_method="scott")
    # Extend past [0,1] to detect boundary modes without dropping them.
    bandwidth = float(np.sqrt(kde.covariance[0, 0]))
    grid = np.linspace(min(0., values.min()) - 3 * bandwidth,
                       max(1., values.max()) + 3 * bandwidth, grid_size)
    density = kde(grid)
    peaks, _ = find_peaks(density)
    result = {"n": len(values), "bandwidth": bandwidth,
              "modes": [float(grid[i]) for i in peaks], "status": "mode_count_mismatch"}
    if len(peaks) == 3:
        result.update(status="passed", thresholds=[
            float(grid[a + np.argmin(density[a:b + 1])])
            for a, b in zip(peaks[:-1], peaks[1:])])
    return result


def calibrate(rows, required_layers=4):
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError("Calibration requires nonempty unique prompt IDs")
    per_prompt = {}
    common = None
    for row in rows:
        fits = {layer: kde_thresholds(values) for layer, values in row['sparsity'].items()}
        eligible = {layer for layer, fit in fits.items() if fit['status'] == 'passed'}
        common = eligible if common is None else common & eligible
        per_prompt[row['id']] = fits
    eligible = sorted(common, key=int)
    # Paper specifies four selected layers but not a tie-break. Fixed layer
    # index avoids selecting for downstream accuracy.
    selected = eligible[:required_layers]
    passed = len(selected) == required_layers
    thresholds = np.mean([per_prompt[r['id']][layer]['thresholds']
                          for r in rows for layer in selected], axis=0).tolist() if passed else None
    return dict(status='passed' if passed else 'not_applicable_under_strict_calibration',
                required_layers=required_layers, eligible_layers=eligible,
                selected_layers=selected if passed else [], thresholds=thresholds,
                prompts=len(rows), per_prompt=per_prompt,
                bandwidth='Scott; paper does not specify KDE bandwidth',
                selection='intersection of exactly-three-mode layers over all prompts; lowest four indices',
                fallback=None,
                reason=None if passed else 'Fewer than four layers show exactly three KDE modes on every calibration prompt')


def collector_class():
    import torch
    from models.QwenLLMWithKVCache import QwenLLMWithKVCache
    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

    class ThinkKVCollector(QwenLLMWithKVCache):
        def __init__(self, model_path):
            super().__init__(model_path, {'pruning_mode': 'none', 'raise_on_runtime_error': True})
            if self.model.config.model_type != 'qwen2':
                raise ValueError('Collector currently audited for Qwen2 only')
            for block in self.model.model.layers:
                attention = block.self_attn
                original = attention.forward

                def observed(attention, hidden_states, position_embeddings, attention_mask,
                             past_key_values=None, cache_position=None, _original=original, **kwargs):
                    result = _original(hidden_states, position_embeddings, attention_mask,
                                       past_key_values=past_key_values,
                                       cache_position=cache_position, **kwargs)
                    if hidden_states.shape[1] == 1 and past_key_values is not None:
                        shape = (*hidden_states.shape[:-1], -1, attention.head_dim)
                        query = attention.q_proj(hidden_states).view(shape).transpose(1, 2)
                        key_dummy = attention.k_proj(hidden_states).view(shape).transpose(1, 2)
                        query, _ = apply_rotary_pos_emb(query, key_dummy, *position_embeddings)
                        keys = past_key_values.layers[attention.layer_idx].keys
                        groups = attention.num_key_value_groups
                        query = query.reshape(query.shape[0], keys.shape[1], groups, 1, attention.head_dim)
                        logits = torch.einsum('bhgqd,bhkd->bhgqk', query.float(), keys.float()) * attention.scaling
                        # Paper C.2: max logits within GQA group, softmax,
                        # then average attention across KV groups.
                        weights = logits.amax(dim=2).softmax(dim=-1).mean(dim=1)
                        sparsity = (weights < .01 * weights.amax(dim=-1, keepdim=True)).float().mean().item()
                        self.decode_sparsity.setdefault(str(attention.layer_idx), []).append(
                            [int(keys.shape[2] - 1), sparsity])
                    return result

                attention.forward = MethodType(observed, attention)
            self.reset()

        def reset(self):
            super().reset()
            self.decode_sparsity = {}

        def truncate_cache(self, keep_token_count):
            super().truncate_cache(keep_token_count)
            # Discard scores of hallucinated continuation removed by the
            # original stop/crop logic, retaining earlier valid decode queries.
            for layer, records in self.decode_sparsity.items():
                self.decode_sparsity[layer] = [r for r in records if r[0] < keep_token_count]

        def export_sparsity(self):
            return {layer: [r[1] for r in records] for layer, records in self.decode_sparsity.items()}

    return ThinkKVCollector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--sample-start', type=int, default=522)
    parser.add_argument('--analyze-only', action='store_true')
    args = parser.parse_args()
    if args.samples < 1 or args.sample_start < 502:
        parser.error('Use positive samples and a calibration split outside main/smoke')
    args.output.mkdir(parents=True, exist_ok=True)
    state = args.output / 'status.json'
    cp = args.output / 'checkpoint.json'
    try:
        rows = json.loads(cp.read_text()) if cp.exists() else []
        if not args.analyze_only:
            import torch
            import transformers
            from datasets import load_dataset
            import run_all_wiki_experiments_v2 as base
            from retrievers.WikiBM25Retriever import WikiBM25Retriever
            assets = Path('/data/experiment/fengboyu/stepkv')
            metadata = {k: json.loads((assets / f'{k}_ready.json').read_text()) for k in ['model', 'data']}
            files = sorted(str(p) for p in (assets / 'datasets/hotpotqa/distractor').glob('validation-*.parquet'))
            data = load_dataset('parquet', data_files={'validation': files}, split='validation',
                                cache_dir=str(assets / 'datasets/arrow_cache'))
            base.RANDOM_SEED = 233
            base.NUM_SAMPLES = args.sample_start + args.samples
            base.MAX_STEPS = 7
            selected = base.select_samples(data)[args.sample_start:]
            manifest = dict(protocol='thinkkv_qa_calibration_preflight_v1', paper='https://arxiv.org/html/2510.01290v2',
                original_source_hashes=verify_original_sources(ROOT),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                samples=args.samples, sample_start=args.sample_start, ids=[r['id'] for _, r in selected],
                seed=233, assets=metadata, torch=torch.__version__, transformers=transformers.__version__,
                scope='Compatibility check using held-out QA; paper uses 100 s1K prompts and reasoning models',
                runtime='Original FullKV generator and attention output; read-only real-decode-query observer',
                sparsity='Paper C.2 GQA logit max pooling, softmax, mean across groups, fraction below 1% row max',
                no_weight_training=True)
            mp = args.output / 'manifest.json'
            if mp.exists() and json.loads(mp.read_text()) != manifest:
                raise RuntimeError('Changed calibration protocol; use a fresh output directory')
            atomic_json(mp, manifest)
            if not set(r['id'] for r in rows) <= set(manifest['ids']):
                raise RuntimeError('Calibration checkpoint ID mismatch')
            atomic_json(state, dict(status='loading', pid=os.getpid(), updated=time.time()))
            retriever = WikiBM25Retriever(metadata['data']['wiki_index'])
            if len(retriever.titles) != 5233235 or retriever.corpus_texts is None:
                raise RuntimeError('Incomplete retrieval corpus')
            llm = collector_class()(metadata['model']['path'])
            done = {r['id'] for r in rows}
            for index, sample in selected:
                if sample['id'] in done:
                    continue
                atomic_json(state, dict(status='running', pid=os.getpid(), updated=time.time(), saved=len(rows), total=args.samples))
                start = time.monotonic()
                answer, trajectory, timings, debug = base._run_react_kv_episode(
                    sample['question'], llm, retriever, pruning_mode='none', return_debug=True)
                rows.append(dict(id=sample['id'], index=index, prediction=answer,
                                 steps=len(trajectory), seconds=time.monotonic()-start,
                                 sparsity=llm.export_sparsity()))
                atomic_json(cp, rows)
                print(f"Calibration {len(rows)}/{args.samples}; valid decode queries={len(next(iter(rows[-1]['sparsity'].values()), []))}", flush=True)
            del llm
            torch.cuda.empty_cache()
        if len(rows) != args.samples:
            raise RuntimeError('Incomplete calibration sample count')
        report = calibrate(rows)
        atomic_json(args.output / 'calibration.json', report)
        atomic_json(state, dict(status='complete', gate_status=report['status'], pid=os.getpid(), updated=time.time(), samples=len(rows)))
        print(json.dumps({k: v for k, v in report.items() if k != 'per_prompt'}, indent=2), flush=True)
    except BaseException:
        atomic_json(state, dict(status='failed', pid=os.getpid(), updated=time.time(), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()
