"""Persist the ThinkKV compatibility result and candidate execution status."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_rebuttal_legacy import atomic_json

BASE = ROOT / 'results/rebuttal_20260928'
CALIBRATION = BASE / 'thinkkv_preflight/calibration_qa20'
CANDIDATE = BASE / 'thinkkv_candidate_v1'


def summarize():
    state = json.loads((CALIBRATION / 'status.json').read_text())
    rows = json.loads((CALIBRATION / 'checkpoint.json').read_text()) if (CALIBRATION / 'checkpoint.json').exists() else []
    report = json.loads((CALIBRATION / 'calibration.json').read_text()) if (CALIBRATION / 'calibration.json').exists() else None
    plan_path = CANDIDATE / 'experiment_plan.json'
    plan = json.loads(plan_path.read_text())
    if state['status'] == 'complete' and report is not None:
        stage = ('calibration_passed_adapter_pending' if report['status'] == 'passed'
                 else 'blocked_by_calibration')
    elif state['status'] == 'failed':
        stage = 'calibration_failed'
    else:
        stage = 'calibrating'
    value = dict(status=stage, updated=time.time(), calibration_saved=len(rows), calibration_expected=20,
                 calibration_status=state, calibration_gate=report['status'] if report else None,
                 eligible_layers=report['eligible_layers'] if report else None,
                 required_layers=4, completed_main_jobs=0, planned_main_jobs=len(plan['jobs']),
                 main_runner=None, main_controller=None,
                 reason=report.get('reason') if report else None)
    atomic_json(CANDIDATE / 'status.json', value)
    # Keep registration time and pinned sample IDs; update only execution state.
    plan['status'] = stage
    atomic_json(plan_path, plan)
    lines = ['# ThinkKV 候选实验进展', '',
             f"阶段：{stage}；校准完成 {len(rows)}/20 题；主实验完成 0/6 组。", '',
             '已登记 HotpotQA / 2Wiki / MuSiQue，每个数据集保留 20% / 50% 轨迹 token、每档 500 题、seed233。',
             '主实验适配器和 controller 尚未接入；当前没有 ThinkKV EM/F1 或压缩率结果。', '',
             '校准使用独立 QA 题，不与主实验、smoke 或 SideQuest pilot 重合。']
    if report is not None:
        lines += ['', f"共同三峰层：{report['eligible_layers']}；论文配置要求 4 层。",
                  f"校准结论：{report['status']}。",
                  '该结论仅针对当前 Qwen2.5-7B、20题QA、Scott带宽预检，不证明该方法在论文模型或s1K上不可用。', '',
                  '| 校准题 ID | 有效 decode query 数 | 恰好三峰的层数 |', '|---|---:|---:|']
        for row in rows:
            fits = report['per_prompt'][row['id']]
            count = sum(fit['status'] == 'passed' for fit in fits.values())
            lengths = [len(v) for v in row['sparsity'].values()]
            lines.append(f"| {row['id']} | {min(lengths) if lengths else 0} | {count} |")
        lines += ['', '没有采用关键词标签、分位数标签或未压缩 FullKV 来冒充 ThinkKV。',
                  '完成适配需要获得可用的 thought 校准，并实现和验证 TBQ/TBE；官方实现或作者校准资料尚未找到。']
    lines += ['', '详细协议见仓库 `THINKKV_REBUTTAL.md`。']
    (CANDIDATE / 'SUMMARY.md').write_text('\n'.join(lines) + '\n')
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        value = summarize()
        if not args.watch or value['status'] != 'calibrating':
            print(json.dumps(value, ensure_ascii=False, indent=2))
            return
        time.sleep(15)


if __name__ == '__main__':
    main()
