"""Single adaptive-budget SideQuest-inspired arm; never label as an official reproduction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_rebuttal_legacy import atomic_json, verify_original_sources


def episode(llm, question, retriever, max_steps):
    import run_all_wiki_experiments_v2 as base
    prompt = base.REACT_KV_INITIAL_PROMPT.format(examples=base.REACT_EXAMPLES, question=question) + 'Thought 1:'
    stops = ['\nObservation', '\nQuestion:']
    response, prompt_cache, generated_cache = llm.generate_first(prompt, max_new_tokens=256, stop_strings=stops)
    del prompt_cache, generated_cache
    lookup = dict(page=None, lookup_keyword=None, lookup_list=None, lookup_cnt=0)
    trajectory = []
    answer, reason = '', 'turn_limit'
    for step in range(1, max_steps + 1):
        thought, action, arg = base.parse_action_original(f'Thought {step}:' + response, step)
        row = dict(step=step, response=response, thought=thought, action_type=action, action_arg=arg)
        trajectory.append(row)
        if action == 'finish':
            answer, reason = arg or '', 'finish'
            break
        if action not in ('search', 'lookup'):
            reason = 'unparsable_action'
            break
        observation, lookup = base.execute_action(action, arg, retriever, lookup)
        observation = observation.replace('\\n', '')
        row['observation'] = observation
        if step < max_steps:
            response = llm.generate_incremental(
                f'\nObservation {step}: {observation}\nThought {step + 1}:',
                max_new_tokens=256, stop_strings=stops)
    # Account for unfinished auxiliary work, but do not evict after answering.
    llm.collect_pending(apply=False, wait=True)
    return answer, reason, trajectory


def summarize(directory, rows, expected):
    n = len(rows)
    events = [e for r in rows for e in r["audit"]["aux_events"]]
    summary = dict(status="complete" if n == expected else "partial", samples=n,
                   expected_samples=expected, exact_match=100 * sum(r["em"] for r in rows) / n if n else None,
                   f1_score=100 * sum(r["f1"] for r in rows) / n if n else None,
                   auxiliary_calls=len(events), invalid_commands=sum(e["parse_error"] is not None for e in events),
                   applied_deletion_events=sum(e.get("before_tokens", 0) > e.get("after_tokens", 0) for e in events),
                   removed_tokens=sum(e.get("before_tokens", 0) - e.get("after_tokens", 0) for e in events),
                   scope="Untrained Qwen adaptation with independent decoder; no fixed ratio; not comparable as a matched-runtime legacy baseline")
    atomic_json(directory / "SUMMARY.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=500)
    parser.add_argument('--dataset', choices=['hotpotqa', '2wiki', 'musique'], required=True)
    parser.add_argument('--purpose', choices=['smoke', 'main'], required=True)
    parser.add_argument('--sample-start', type=int, default=0)
    parser.add_argument('--interval', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=7)
    args = parser.parse_args()
    if args.samples < 1 or args.sample_start < 0 or args.interval < 1:
        parser.error('Positive sizes and nonnegative start required')
    args.output.mkdir(parents=True, exist_ok=True)
    state = args.output / 'status.json'
    llm = None
    try:
        import torch
        import transformers
        from datasets import load_dataset
        import run_all_wiki_experiments_v2 as base
        from retrievers.WikiBM25Retriever import WikiBM25Retriever
        from scripts.sidequest_runtime import SideQuestPilot
        assets = Path('/data/experiment/fengboyu/stepkv')
        metadata = {kind: json.loads((assets / f'{kind}_ready.json').read_text()) for kind in ('model', 'data')}
        original = verify_original_sources(ROOT)
        if args.dataset == 'hotpotqa':
            files = sorted(str(p) for p in (assets / 'datasets/hotpotqa/distractor').glob('validation-*.parquet'))
            data = load_dataset('parquet', data_files={'validation': files}, split='validation',
                                cache_dir=str(assets / 'datasets/arrow_cache'))
        else:
            dataset_asset = json.loads((assets / 'datasets' / args.dataset / 'ready.json').read_text())
            data_path = Path(dataset_asset['path'])
            if hashlib.sha256(data_path.read_bytes()).hexdigest() != dataset_asset['sha256']:
                raise RuntimeError('Dataset checksum mismatch')
            data = json.loads(data_path.read_text())
            metadata['evaluation_dataset'] = dataset_asset
        base.RANDOM_SEED = 233
        base.NUM_SAMPLES = args.sample_start + args.samples
        selected = base.select_samples(data)[args.sample_start:]
        if len(selected) != args.samples:
            raise RuntimeError('Insufficient pilot samples')
        code = ['scripts/run_sidequest_baseline.py', 'scripts/sidequest_runtime.py', 'models/RebuttalLLM.py', 'token_tracker.py']
        manifest = dict(protocol='sidequest_untrained_qwen_baseline_v1', dataset=args.dataset, purpose=args.purpose, budget='adaptive', paper='https://arxiv.org/abs/2602.22603v2',
                        trained=False, assets=metadata, original_source_hashes=original,
                        sources={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in code},
                        ids=[r['id'] for _, r in selected], seed=233, sample_start=args.sample_start,
                        max_steps=args.max_steps, interval=args.interval, aux_max_tokens=128,
                        decoding='shared independent greedy loop; logical RoPE; same cursor labels in both arms',
                        torch=torch.__version__, transformers=transformers.__version__)
        mp = args.output / 'manifest.json'
        if mp.exists() and json.loads(mp.read_text()) != manifest:
            raise RuntimeError('Pilot source/protocol changed: use a new output directory')
        atomic_json(mp, manifest)
        cp = args.output / 'checkpoint.json'
        rows = json.loads(cp.read_text()) if cp.exists() else []
        done = {(r['id'], r['method']) for r in rows}
        atomic_json(state, dict(status='loading', pid=os.getpid(), updated=time.time()))
        retriever = WikiBM25Retriever(metadata['data']['wiki_index'])
        if len(retriever.titles) != 5233235 or retriever.corpus_texts is None:
            raise RuntimeError('Incomplete retrieval corpus')
        llm = SideQuestPilot(metadata['model']['path'], interval=args.interval)
        for index, sample in selected:
            for method in ('sidequest_untrained',):
                if (sample['id'], method) in done:
                    continue
                atomic_json(state, dict(status='running', pid=os.getpid(), updated=time.time(),
                                        method=method, sample_id=sample['id'], saved=len(rows), total=args.samples))
                llm.enabled = method == 'sidequest_untrained'
                torch.cuda.synchronize()
                begin = time.perf_counter()
                answer, termination, trajectory = episode(llm, sample['question'], retriever, args.max_steps)
                torch.cuda.synchronize()
                seconds = time.perf_counter() - begin
                rows.append(dict(id=sample['id'], index=index, method=method, question=sample['question'],
                                 prediction=answer, gold=sample['answer'], termination=termination,
                                 em=base.exact_match(answer, sample['answer']), f1=base.f1_score(answer, sample['answer']),
                                 seconds=seconds, trajectory=trajectory, audit=llm.pilot_audit()))
                atomic_json(cp, rows)
                summarize(args.output, rows, args.samples)
                print(f"Saved {len(rows)}/{args.samples}: {method} {sample['id']} {termination}", flush=True)
        result = summarize(args.output, rows, args.samples)
        atomic_json(args.output / 'results.json', dict(summary=result, results=rows))
        atomic_json(state, dict(status='complete', pid=os.getpid(), updated=time.time(), samples=args.samples))
    except BaseException:
        atomic_json(state, dict(status='failed', pid=os.getpid(), updated=time.time(), traceback=traceback.format_exc()))
        raise
    finally:
        if llm is not None:
            llm.close()


if __name__ == '__main__':
    main()
