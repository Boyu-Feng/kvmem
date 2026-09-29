"""Paired, untrained Qwen SideQuest pilot on a held-out HotpotQA subset."""
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
    completed = []
    arms = {}
    for method in ('fullkv_pilot', 'sidequest_untrained'):
        group = [r for r in rows if r['method'] == method]
        if not group:
            continue
        n = len(group)
        events = [e for r in group for e in r['audit']['aux_events']]
        arms[method] = dict(n=n, em=100 * sum(r['em'] for r in group) / n,
                            f1=100 * sum(r['f1'] for r in group) / n,
                            completion_rate=sum(r['termination'] == 'finish' for r in group) / n,
                            mean_peak_main_kv_MiB=sum(r['audit']['peak_main_cache_bytes'] for r in group) / n / 2**20,
                            mean_episode_seconds=sum(r['seconds'] for r in group) / n,
                            auxiliary_calls=len(events), invalid_commands=sum(e['parse_error'] is not None for e in events),
                            applied_deletion_events=sum(e.get('before_tokens', 0) > e.get('after_tokens', 0) for e in events),
                            removed_tokens=sum(e.get('before_tokens', 0) - e.get('after_tokens', 0) for e in events))
    paired = set(r['id'] for r in rows if r['method'] == 'fullkv_pilot') & set(r['id'] for r in rows if r['method'] == 'sidequest_untrained')
    result = dict(status='complete' if len(paired) == expected else 'partial', paired_samples=len(paired),
                  expected_pairs=expected, arms=arms,
                  scope='Untrained Qwen pilot; not paper reproduction or legacy-table accuracy/efficiency evidence')
    atomic_json(directory / 'SUMMARY.json', result)
    lines = ['# SideQuest 未微调原型：小样本配对验证', '',
             'Qwen2.5-7B，HotpotQA seed233 第503题起；两个实验臂共用带 cursor 标记的独立 greedy 解码器。',
             '未使用论文专门训练的 gpt-oss 权重、训练数据或 SGLang；结果不替换原表。', '',
             '| 方法 | N | EM (%) | F1 (%) | 平均 main KV 峰值 MiB | 辅助调用 | 实际删除事件 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for method, value in arms.items():
        lines.append(f"| {method} | {value['n']} | {value['em']:.2f} | {value['f1']:.2f} | {value['mean_peak_main_kv_MiB']:.2f} | {value['auxiliary_calls']} | {value['applied_deletion_events']} |")
    lines.extend(['', f'完成配对：{len(paired)}/{expected}。运行中的各臂样本数可能不同，不能用不完整汇总作质量结论。',
                  'main KV 指标不含辅助分支；每题记录分支 KV 峰值、辅助 token 数及包含等待的 episode 时间。',
                  '本地 Python 线程与单 GPU 调度不等同于论文的 SGLang 并行服务，耗时仅用于诊断。'])
    (directory / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=20)
    parser.add_argument('--sample-start', type=int, default=502)
    parser.add_argument('--interval', type=int, default=4)
    parser.add_argument('--max-steps', type=int, default=7)
    args = parser.parse_args()
    if args.samples < 1 or args.sample_start < 502 or args.interval < 1:
        parser.error('Positive sizes; pilot must remain outside main and smoke samples')
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
        files = sorted(str(p) for p in (assets / 'datasets/hotpotqa/distractor').glob('validation-*.parquet'))
        data = load_dataset('parquet', data_files={'validation': files}, split='validation',
                            cache_dir=str(assets / 'datasets/arrow_cache'))
        base.RANDOM_SEED = 233
        base.NUM_SAMPLES = args.sample_start + args.samples
        selected = base.select_samples(data)[args.sample_start:]
        if len(selected) != args.samples:
            raise RuntimeError('Insufficient pilot samples')
        code = ['scripts/run_sidequest_pilot.py', 'scripts/sidequest_runtime.py', 'models/RebuttalLLM.py', 'token_tracker.py']
        manifest = dict(protocol='sidequest_untrained_qwen_pilot_v1', paper='https://arxiv.org/abs/2602.22603v2',
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
            for method in ('fullkv_pilot', 'sidequest_untrained'):
                if (sample['id'], method) in done:
                    continue
                atomic_json(state, dict(status='running', pid=os.getpid(), updated=time.time(),
                                        method=method, sample_id=sample['id'], saved=len(rows), total=args.samples * 2))
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
                print(f"Saved {len(rows)}/{2 * args.samples}: {method} {sample['id']} {termination}", flush=True)
        result = summarize(args.output, rows, args.samples)
        atomic_json(args.output / 'results.json', dict(summary=result, results=rows))
        atomic_json(state, dict(status='complete', pid=os.getpid(), updated=time.time(), pairs=args.samples))
    except BaseException:
        atomic_json(state, dict(status='failed', pid=os.getpid(), updated=time.time(), traceback=traceback.format_exc()))
        raise
    finally:
        if llm is not None:
            llm.close()


if __name__ == '__main__':
    main()
