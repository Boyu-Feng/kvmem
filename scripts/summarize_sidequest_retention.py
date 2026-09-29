"""Record SideQuest retention with explicit final/step and macro/pooled denominators."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_rebuttal_legacy import atomic_json

START = '<!-- sidequest-retention:start -->'
END = '<!-- sidequest-retention:end -->'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def trajectory_ratio(retained, logical):
    # An empty trajectory has no deletions. Keep it in the per-question mean.
    if logical == 0 and retained == 0:
        return 1.0
    if not 0 <= retained <= logical or logical <= 0:
        raise ValueError('Invalid trajectory counts')
    return retained / logical


def measure_row(row, prompt_tokens):
    audit = row['audit']
    logical, final = audit['logical_tokens'], audit['final_tokens']
    if not 0 < prompt_tokens <= final <= logical:
        raise ValueError(f"Invalid final lengths: {row['id']}")
    applied = [event for event in audit['aux_events'] if event['applied']]
    for event in applied:
        if not prompt_tokens <= event['after_tokens'] <= event['before_tokens']:
            raise ValueError('Eviction violated prompt protection or increased cache size')
    removed = sum(e['before_tokens'] - e['after_tokens'] for e in applied)
    if logical - final != removed:
        raise ValueError(f"Final counter / eviction ledger mismatch: {row['id']}")
    boundaries = audit['boundaries']
    if not boundaries or boundaries[-1]['tokens'] != final:
        raise ValueError('Final boundary does not match final cache')
    if [b['turn'] for b in boundaries] != list(range(1, len(boundaries) + 1)):
        raise ValueError('Nonconsecutive generation boundaries')
    steps = []
    for boundary in boundaries:
        # collection_turn=t is before generate_incremental for t+1; boundary t
        # was already recorded. The deletion affects boundaries t+1 onward.
        deleted = sum(e['before_tokens'] - e['after_tokens'] for e in applied
                      if e['collection_turn'] < boundary['turn'])
        resident = boundary['tokens']
        reference = resident + deleted
        if prompt_tokens > reference or resident < prompt_tokens:
            raise ValueError('Invalid prompt at a boundary')
        steps.append(dict(id=row['id'], method=row['method'], turn=boundary['turn'],
                          prompt_tokens=prompt_tokens, resident_tokens=resident,
                          logical_tokens=reference, cumulative_deleted_tokens=deleted,
                          trajectory_retention=trajectory_ratio(resident - prompt_tokens, reference - prompt_tokens),
                          total_retention=resident / reference))
    if steps[-1]['logical_tokens'] != logical:
        raise ValueError('Reconstructed final logical length disagrees with audit')
    result = dict(id=row['id'], method=row['method'], prompt_tokens=prompt_tokens,
                  final_logical_tokens=logical, final_resident_tokens=final, deleted_tokens=removed,
                  trajectory_logical_tokens=logical - prompt_tokens,
                  trajectory_retained_tokens=final - prompt_tokens,
                  final_trajectory_retention=trajectory_ratio(final - prompt_tokens, logical - prompt_tokens),
                  final_total_retention=final / logical, recorded_steps=len(steps),
                  step_mean_trajectory_retention=statistics.mean(s['trajectory_retention'] for s in steps),
                  step_mean_total_retention=statistics.mean(s['total_retention'] for s in steps))
    return result, steps


def aggregate(rows):
    if not rows:
        raise ValueError('No measured rows')
    trajectory = sum(r['trajectory_logical_tokens'] for r in rows)
    retained = sum(r['trajectory_retained_tokens'] for r in rows)
    logical = sum(r['final_logical_tokens'] for r in rows)
    final = sum(r['final_resident_tokens'] for r in rows)
    return dict(n=len(rows), questions_with_eviction=sum(r['deleted_tokens'] > 0 for r in rows),
                deleted_tokens=sum(r['deleted_tokens'] for r in rows),
                initial_prompt_tokens=sum(r['prompt_tokens'] for r in rows),
                trajectory_logical_tokens=trajectory, trajectory_retained_tokens=retained,
                final_logical_tokens=logical, final_resident_tokens=final,
                mean_final_trajectory_retention=statistics.mean(r['final_trajectory_retention'] for r in rows),
                pooled_final_trajectory_retention=trajectory_ratio(retained, trajectory),
                mean_step_trajectory_retention=statistics.mean(r['step_mean_trajectory_retention'] for r in rows),
                mean_final_total_retention=statistics.mean(r['final_total_retention'] for r in rows),
                pooled_final_total_retention=final / logical,
                mean_step_total_retention=statistics.mean(r['step_mean_total_retention'] for r in rows))


def write_csv(path, rows):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def analyze(directory, method='sidequest_untrained'):
    import transformers
    from transformers import AutoTokenizer
    import run_all_wiki_experiments_v2 as base

    directory = Path(directory).resolve()
    manifest_path = directory / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    state = json.loads((directory / 'status.json').read_text())
    if state['status'] != 'complete':
        raise ValueError('Only completed conditions can be recorded for the paper')
    required = {
        'run_all_wiki_experiments_v2.py': manifest['original_source_hashes']['run_all_wiki_experiments_v2.py'],
        'models/RebuttalLLM.py': manifest['sources']['models/RebuttalLLM.py'],
        'scripts/sidequest_runtime.py': manifest['sources']['scripts/sidequest_runtime.py'],
    }
    if any(sha256(ROOT / path) != expected for path, expected in required.items()):
        raise ValueError('Prompt/runtime sources changed; reconstruct using the pinned source')
    if transformers.__version__ != manifest['transformers']:
        raise ValueError('Use the experiment Transformers version to reconstruct prompts')
    source = directory / 'results.json'
    rows = [r for r in json.loads(source.read_text())['results'] if r['method'] == method]
    if len(rows) != len(manifest['ids']) or {r['id'] for r in rows} != set(manifest['ids']):
        raise ValueError('Incomplete or duplicated result IDs')
    model_path = Path(manifest['assets']['model']['path'])
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    per_question, per_step = [], []
    for row in rows:
        prompt = base.REACT_KV_INITIAL_PROMPT.format(examples=base.REACT_EXAMPLES, question=row['question']) + 'Thought 1:'
        text = tokenizer.apply_chat_template([
            dict(role='system', content='You are a helpful assistant.'),
            dict(role='user', content=prompt)], tokenize=False, add_generation_prompt=True)
        prompt_tokens = len(tokenizer(text, add_special_tokens=True).input_ids)
        question, steps = measure_row(row, prompt_tokens)
        per_question.append(question)
        per_step.extend(steps)
    metrics = aggregate(per_question)
    tokenizer_files = sorted(set(model_path.glob('tokenizer*')) | set(model_path.glob('*.jinja')) |
                             {p for p in (model_path / 'vocab.json', model_path / 'merges.txt',
                                          model_path / 'special_tokens_map.json') if p.exists()})
    report = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(), method=method,
                  dataset=manifest.get('dataset', 'hotpotqa'), seed=manifest['seed'],
                  trained=manifest['trained'], protocol=manifest['protocol'], sample_start=manifest['sample_start'],
                  scope='main branch only; excludes auxiliary cache; logical denominator uses each arm own realized trajectory',
                  metrics=metrics, definitions={
                      'mean_final_trajectory_retention': 'mean_i((K_i_final - P_i)/(L_i_final - P_i))',
                      'pooled_final_trajectory_retention': 'sum_i(K_i_final - P_i)/sum_i(L_i_final - P_i)',
                      'mean_step_trajectory_retention': 'mean_i(mean_t((K_it - P_i)/(L_it - P_i)))',
                      'mean_final_total_retention': 'mean_i(K_i_final/L_i_final)',
                      'pooled_final_total_retention': 'sum_i(K_i_final)/sum_i(L_i_final)',
                      'mean_step_total_retention': 'mean_i(mean_t(K_it/L_it))',
                      'P': 'protected initial system/user/chat-template prompt tokens',
                      'K': 'resident main-branch KV tokens',
                      'L': 'valid logical main-branch tokens, including evicted entries, excluding cropped stop suffix and auxiliary tokens',
                      't': 'recorded post-generation boundary; not wall-clock or per-token time average',
                      'no_eviction': 'all questions included; no-eviction questions contribute retention 1.0',
                      'empty_trajectory': 'retention defined as 1.0 for zero valid trajectory tokens and zero deletions',
                  }, provenance=dict(results_sha256=sha256(source), manifest_sha256=sha256(manifest_path),
                                     analysis_sha256=sha256(__file__), prompt_and_runtime_sources=required,
                                     tokenizer_sha256={p.name: sha256(p) for p in tokenizer_files if p.is_file()},
                                     transformers=transformers.__version__))
    atomic_json(directory / 'RETENTION_SUMMARY.json', report)
    write_csv(directory / 'retention_per_question.csv', per_question)
    write_csv(directory / 'retention_per_step.csv', per_step)
    m = metrics
    paper_text = (f"在{m['n']}题{report['dataset']}小规模实验中，未微调的SideQuest-inspired适配版的"
                  f"逐题平均终态轨迹KV保留率为{m['mean_final_trajectory_retention']:.2%}"
                  f"（排除受保护的初始prompt）；按轨迹token数加权的终态保留率为{m['pooled_final_trajectory_retention']:.2%}。"
                  f"在已记录的生成块结束边界，先题内再题间平均的轨迹KV保留率为{m['mean_step_trajectory_retention']:.2%}。"
                  "这些指标仅统计主分支，以该方法自身实际轨迹的逻辑token数为分母，不包含辅助分支缓存。")
    lines = ['# SideQuest KV 保留比例记录', '',
             f"方法：{method}；数据集：{report['dataset']}；N={m['n']}；seed={report['seed']}；抽样起点（0-based）={report['sample_start']}。", '',
             '| 指标 | 保留比例 | 平均方式 |', '|---|---:|---|',
             f"| 终态轨迹KV，排除prompt | **{m['mean_final_trajectory_retention']:.2%}** | 每题先算比例，再对题目等权平均 |",
             f"| 终态轨迹KV，排除prompt | {m['pooled_final_trajectory_retention']:.2%} | 合并所有题目的token计数后求比值 |",
             f"| 步末轨迹KV，排除prompt | {m['mean_step_trajectory_retention']:.2%} | 每题内各生成块末比例平均，再题间平均 |",
             f"| 终态全部KV，包含prompt | {m['mean_final_total_retention']:.2%} | 每题等权平均 |",
             f"| 终态全部KV，包含prompt | {m['pooled_final_total_retention']:.2%} | 合并token计数后求比值 |",
             f"| 步末全部KV，包含prompt | {m['mean_step_total_retention']:.2%} | 先题内、再题间平均 |", '',
             f"有效轨迹token共{m['trajectory_logical_tokens']:,}，终态保留{m['trajectory_retained_tokens']:,}，删除{m['deleted_tokens']:,}；"
             f"初始prompt共{m['initial_prompt_tokens']:,}个token。{m['questions_with_eviction']}/{m['n']}题发生实际删除，所有题均进入统计。", '',
             '令P_i为初始受保护prompt长度，L_it为有效主分支逻辑token数，K_it为实际驻留KV长度，则轨迹保留率为(K_it-P_i)/(L_it-P_i)。',
             '终态采用每题结束时的值；步末采用已记录的生成块结束边界。collection_turn=t的删除在下一次生成前生效，因此从边界t+1开始扣除。',
             '分母包括已经淘汰的有效主分支token，不含停止标记裁剪的无效后缀及辅助分支token。初始prompt使用固定版本tokenizer和原始prompt源码重建。若某题有效轨迹为空且无删除，其保留率定义为100%。', '',
             '**报告边界：** 终态比例不等于整个运行过程平均，也不等于峰值显存比例；步末平均不是逐token或墙钟时间加权平均。',
             '本统计不是相对另一条自由生成FullKV轨迹的长度比，也不包含辅助分支的显存开销。当前方法是未微调适配版，不能据此宣称论文版SideQuest复现。', '',
             '可用于实验文字（保留样本量及口径）：', '', paper_text, '',
             '逐题计数与比例见 `retention_per_question.csv`，逐步数据见 `retention_per_step.csv`，完整精度、公式和源文件哈希见 `RETENTION_SUMMARY.json`。']
    (directory / 'RETENTION_SUMMARY.md').write_text('\n'.join(lines) + '\n')
    summary_path = directory / 'SUMMARY.md'
    text = summary_path.read_text() if summary_path.exists() else '# SideQuest 结果\n'
    if START in text:
        prefix, tail = text.split(START, 1)
        if END not in tail:
            raise ValueError('Broken retention summary marker')
        text = prefix + tail.split(END, 1)[1]
    text = text.rstrip() + f'\n\n{START}\n\n' + paper_text + '\n\n[保留比例完整记录](RETENTION_SUMMARY.md)\n\n' + END + '\n'
    summary_path.write_text(text)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.root)
    print(json.dumps(report['metrics'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
