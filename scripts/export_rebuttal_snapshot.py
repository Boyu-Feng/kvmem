"""Export an immutable, auditable progress snapshot; never alter running outputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
QUEUES = ("legacy_baselines", "legacy_step_controls", "multidataset_baselines",
          "multidataset_step_controls", "sidequest_adaptive_v1")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def normalize_rows(rows, expected_ids):
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids) or not set(ids) <= set(expected_ids):
        raise ValueError("Duplicate or out-of-manifest result IDs")
    exported = []
    for row in rows:
        em, f1 = float(row["em"]), float(row["f1"])
        if em not in (0., 1.) or not 0 <= f1 <= 1:
            raise ValueError("Unexpected score scale; expected EM/F1 in [0, 1]")
        item = {key: row[key] for key in ("id", "index", "num_steps", "sample_time", "peak_memory_mb",
                "prompt_token_count", "step_remaining_tokens", "llm_stats", "termination", "seconds") if key in row}
        item.update(em=em, f1=f1)
        if "audit" in row:
            audit = row["audit"]
            events = audit["aux_events"]
            item["sidequest"] = {key: audit[key] for key in ("interval", "peak_main_cache_tokens",
                    "peak_main_cache_bytes", "final_tokens", "logical_tokens") if key in audit}
            item["sidequest"].update(auxiliary_calls=len(events),
                invalid_commands=sum(e.get("parse_error") is not None for e in events),
                deletion_events=sum(e.get("before_tokens", 0) > e.get("after_tokens", 0) for e in events),
                removed_tokens=sum(e.get("before_tokens", 0) - e.get("after_tokens", 0) for e in events))
        exported.append(item)
    return exported


def render_results_table(summary):
    """Render datasets as rows and methods as columns from audited scores."""
    dataset_labels = {"hotpotqa": "HotpotQA", "2wiki": "2WikiMultihopQA", "musique": "MuSiQue"}
    method_labels = {"h2o_step": "H2O-step", "tova_step": "TOVA-step",
                     "flowkv_style": "FlowKV-style", "lazyeviction": "LazyEviction",
                     "sidequest_untrained": "SideQuest-untrained"}
    conditions = summary["main_conditions"]
    datasets = list(dict.fromkeys(c["dataset"] for c in conditions))
    methods = [m for m in method_labels if any(c["method"] == m for c in conditions)]
    methods += sorted({c["method"] for c in conditions} - set(methods))
    lines = ["# KVMem 实验结果总表", "",
             f"快照时间：{summary['captured_to_utc']}。",
             f"已完成 **{summary['completed_main']}/{summary['planned_main']} 组**，"
             f"已保存 **{summary['saved_main']:,}/{summary['expected_main']:,} 条逐题评测**。", "",
             "模型为 Qwen2.5-7B-Instruct，抽样 seed 为 233；本轮每组计划评测 500 题。",
             "每格为 **EM / F1（%）**；20% 和 50% 表示名义轨迹 token 保留档位。仅列已完成组的分数。", "",
             "| 数据集 | " + " | ".join(method_labels.get(m, m) for m in methods) + " |",
             "|---|" + "---:|" * len(methods)]
    for dataset in datasets:
        cells = []
        for method in methods:
            group = [c for c in conditions if c["dataset"] == dataset and c["method"] == method
                     and c["status"] == "complete"]
            group.sort(key=lambda c: (not isinstance(c["budget"], (int, float)), str(c["budget"])))
            values = []
            for c in group:
                budget = f"{c['budget']:.0%}" if isinstance(c["budget"], (int, float)) else "自适应"
                values.append(f"{budget}：{c['em_percent']:.2f} / {c['f1_percent']:.2f}")
            cells.append("<br>".join(values) or "—")
        lines.append("| " + dataset_labels.get(dataset, dataset) + " | " + " | ".join(cells) + " |")
    lines += ["", "## 读表说明", "",
              "- SideQuest-untrained 使用独立解码器、自适应预算和未微调模型；与其余方法同表展示，但不能作为等预算或等运行协议的排名。",
              "- 20%/50% 是名义 token 比例，不是物理显存比例。LazyEviction 将 4 个原生 KV 头展开为 28 个头，实际 KV 字节需单独比较。",
              "- 本轮只有一个抽样 seed，未计算跨 seed 标准差。所有分数均由对应逐题 EM/F1 重新计算。",
              "- 本表覆盖本批 27 组主实验。StepKV、FullKV 等原论文数值见[历史参考表](../../ORIGINAL_RESULTS_REFERENCE.md)，未混入本轮结果。",
              "- ThinkKV 的 6 组候选主实验和 7 条件效率补测尚未完成；未将缺失结果记作零分。逐题时间仅作诊断，不能作为独占 GPU 效率结论。", "",
              "## 完整结果与证据", "",
              "- [逐题指标与缓存审计](per_question/)：全部已保存主实验记录。",
              "- [机器可读汇总](summary.json)：完整精度分数、题数、预算和状态。",
              "- [运行配置和状态](evidence/)及[源文件 SHA256](source_bindings.json)。",
              "- [详细快照说明](README.md)及[后续计划](../../docs/rebuttal/NEXT_STEPS.md)。"]
    return "\n".join(lines) + "\n"


def export(source, destination):
    if destination.exists():
        raise FileExistsError("Snapshots are immutable; choose a new output directory")
    started = datetime.now(timezone.utc).isoformat()
    bindings = {}

    def read(relative, default=None):
        path = source / relative
        if not path.exists():
            return default
        raw = path.read_bytes()  # Source files use atomic replacement.
        bindings[str(relative)] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                                  "read_at_utc": datetime.now(timezone.utc).isoformat()}
        return json.loads(raw)

    queues, conditions = {}, []
    artifacts = {}
    for queue in QUEUES:
        plan = read(f"{queue}/queue_plan.json")
        state = read(f"{queue}/queue_status.json", {})
        if plan is None:
            continue
        artifacts[f"evidence/{queue}/queue_plan.json"] = plan
        artifacts[f"evidence/{queue}/queue_status.json"] = state
        jobs = []
        for job in plan["jobs"]:
            prefix = f"{queue}/{job['name']}"
            status = read(f"{prefix}/status.json", {"status": "pending"})
            manifest = read(f"{prefix}/manifest.json")
            raw_rows = read(f"{prefix}/checkpoint.json", [])
            count = len(raw_rows)
            item = dict(queue=queue, name=job["name"], dataset=job.get("dataset", "hotpotqa"),
                        method=job["method"], budget=job.get("ratio", job.get("budget")),
                        purpose=job["purpose"], expected=job["samples"], saved=count,
                        status=status.get("status", "pending"), seed=job.get("seed", 233))
            if manifest is not None:
                fingerprint = manifest.get("fingerprint", manifest)
                expected = fingerprint.get("sample_ids", fingerprint.get("ids", []))
                if len(expected) != job["samples"] or len(set(expected)) != len(expected):
                    raise ValueError(f"Invalid registered sample IDs: {prefix}")
                rows = normalize_rows(raw_rows, expected)
                if item["status"] == "complete" and count != job["samples"]:
                    raise ValueError(f"Completed status with incomplete checkpoint: {prefix}")
                if job["purpose"] == "main":
                    artifacts[f"evidence/{prefix}/manifest.json"] = manifest
                    artifacts[f"evidence/{prefix}/status.json"] = status
                    audits = read(f"{prefix}/baseline_audit.json")
                    run_summary = read(f"{prefix}/SUMMARY.json")
                    if run_summary is not None:
                        artifacts[f"evidence/{prefix}/SUMMARY.json"] = run_summary
                    if audits is not None:
                        for row in rows:
                            if row["id"] not in audits:
                                raise ValueError(f"Missing saved row audit: {prefix}/{row['id']}")
                            audit = audits[row["id"]]
                            row["cache_audit"] = {k: v for k, v in audit.items()
                                                   if isinstance(v, (int, float, bool)) or v is None}
                    artifacts[f"per_question/{prefix}.json"] = rows
                    item["per_question"] = f"per_question/{prefix}.json"
                    item["score_status"] = "final" if item["status"] == "complete" else "partial_not_ranked"
                    item["em_percent"] = 100 * statistics.mean(r["em"] for r in rows) if rows else None
                    item["f1_percent"] = 100 * statistics.mean(r["f1"] for r in rows) if rows else None
                    result = read(f"{prefix}/results.json") if item["status"] == "complete" else None
                    if result is not None:
                        result_rows = result["results"] if isinstance(result, dict) else result
                        check = normalize_rows(result_rows, expected)
                        if [(r["id"], r["em"], r["f1"]) for r in check] != [(r["id"], r["em"], r["f1"]) for r in rows]:
                            raise ValueError(f"Final result and checkpoint differ: {prefix}")
                    if rows and "sidequest" in rows[0]:
                        item["sidequest"] = {key: sum(r["sidequest"][key] for r in rows) for key in
                                ("auxiliary_calls", "invalid_commands", "deletion_events", "removed_tokens")}
                        item["sidequest"]["questions_with_eviction"] = sum(r["sidequest"]["removed_tokens"] > 0 for r in rows)
            elif raw_rows:
                raise ValueError(f"Checkpoint has no manifest: {prefix}")
            jobs.append(item)
            if job["purpose"] == "main":
                conditions.append(item)
        mains = [j for j in jobs if j["purpose"] == "main"]
        queues[queue] = dict(status=state, planned_main=len(mains), completed_main=sum(j["status"] == "complete" for j in mains),
                             saved_main=sum(j["saved"] for j in mains), expected_main=sum(j["expected"] for j in mains), jobs=jobs)

    for rel in ("sidequest_pilot_v1/SUMMARY.json", "sidequest_pilot_v1/RETENTION_SUMMARY.json",
                "sidequest_pilot_v1/manifest.json", "sidequest_pilot_v1/status.json",
                "sidequest_adaptive_v1/hotpotqa/main/sidequest_untrained/RETENTION_SUMMARY.json",
                "thinkkv_candidate_v1/status.json", "thinkkv_candidate_v1/experiment_plan.json",
                "thinkkv_preflight/calibration_qa20/calibration.json", "thinkkv_preflight/calibration_qa20/manifest.json",
                "thinkkv_preflight/calibration_qa20/bandwidth_sensitivity.json",
                "efficiency_legacy_v1/queue_status.json", "efficiency_legacy_v1/queue_plan.json",
                "efficiency_legacy_v1/fullkv/r100/status.json",
                "new_baselines/queue_status.json", "legacy_baselines/rkv_cancellation.json",
                "multidataset_baselines/rkv_cancellation.json"):
        value = read(rel)
        if value is not None:
            artifacts[f"evidence/{rel}"] = value
    summary = dict(schema_version=1, captured_from_utc=started, captured_to_utc=datetime.now(timezone.utc).isoformat(),
                   source_directory=str(source), queues=queues, main_conditions=conditions,
                   completed_main=sum(c["status"] == "complete" for c in conditions), planned_main=len(conditions),
                   saved_main=sum(c["saved"] for c in conditions), expected_main=sum(c["expected"] for c in conditions),
                   exclusions="Smoke, pilot, ThinkKV candidates, efficiency profiling and retired protocols excluded from main totals",
                   comparison_limits=["Single sampling seed 233; historical paper table is a separate multi-seed reference",
                       "Nominal retained-token ratios are not equal physical KV byte budgets",
                       "SideQuest uses an independent untrained decoder and adaptive budget",
                       "Partial results are progress only and must not be ranked",
                       "Recorded timings are diagnostic; GPU exclusivity and equal workload were not established"],
                   source_git_base_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    lines = ["# Rebuttal 实验快照", "", f"读取时间：{started} 至 {summary['captured_to_utc']}。",
             "这是上传时的静态快照；导出只读取源结果。各源文件的读取时间和 SHA256 见 `source_bindings.json`。", "",
             "**[按数据集 × 方法查看结果总表](RESULTS.md)**", "",
             f"已完成 **{summary['completed_main']}/{summary['planned_main']} 组**可运行主实验；保存 **{summary['saved_main']}/{summary['expected_main']} 次题目评测**。",
             "主实验总数包含 24 组 legacy 补实验和 3 组 SideQuest 适配；不含 smoke、pilot、ThinkKV 候选和效率补测。", "",
             "## 队列进展", "", "| 队列 | 状态 | 完成主实验 | 保存题数 |", "|---|---|---:|---:|"]
    for name, q in queues.items():
        lines.append(f"| {name} | {q['status'].get('status')} | {q['completed_main']}/{q['planned_main']} | {q['saved_main']}/{q['expected_main']} |")
    lines += ["", "## 已完成的固定预算实验", "", "Qwen2.5-7B-Instruct，seed233。20%/50% 是名义轨迹 token 保留档位，不能当作等物理显存。", "",
              "| 数据集 | 方法 | 名义保留比例 | N | EM (%) | F1 (%) |", "|---|---|---:|---:|---:|---:|"]
    for c in conditions:
        if c["status"] == "complete" and isinstance(c["budget"], (int, float)):
            lines.append(f"| {c['dataset']} | {c['method']} | {c['budget']:.0%} | {c['saved']} | {c['em_percent']:.2f} | {c['f1_percent']:.2f} |")
    lines += ["", "## SideQuest 未微调适配", "", "独立解码协议、自适应预算；不能作为官方实现复现或与 legacy 等预算排名。", "",
              "| 数据集 | 状态 | N | EM (%) | F1 (%) |", "|---|---|---:|---:|---:|"]
    sidequest_notes = []
    for c in conditions:
        if c["method"] == "sidequest_untrained":
            scores = f"{c['em_percent']:.2f} | {c['f1_percent']:.2f}" if c["status"] == "complete" else "— | —"
            lines.append(f"| {c['dataset']} | {c['status']} | {c['saved']}/{c['expected']} | {scores} |")
            if c.get("sidequest"):
                s = c["sidequest"]
                sidequest_notes.append(f"{c['dataset']}：辅助调用 {s['auxiliary_calls']} 次，无效命令 {s['invalid_commands']} 次；{s['questions_with_eviction']} 题实际删除 KV，共 {s['removed_tokens']} tokens。")
    lines += [""] + sidequest_notes
    lines += ["", "## 尚未完成", "", "| 数据集 | 方法 | 预算 | 状态 | 保存题数 |", "|---|---|---|---|---:|"]
    for c in conditions:
        if c["status"] != "complete":
            lines.append(f"| {c['dataset']} | {c['method']} | {c['budget']} | {c['status']} | {c['saved']}/{c['expected']} |")
    sidequest_queue = queues.get("sidequest_adaptive_v1", {})
    if sidequest_queue.get("status", {}).get("status") == "complete":
        sidequest_progress = "- SideQuest 三个数据集均已完成，各 500 题，结果已纳入本快照。"
    else:
        sidequest_progress = "- SideQuest 进展见上方队列表；后两集依赖 `multidataset_baselines` 整条队列完成。"
    lines += ["", "## 已知阻塞与后续", "", sidequest_progress,
              "- ThinkKV：三类 thought 校准 20/20 完成，共同三峰层为 0，主实验 0/6；普通 LLM 单类分支尚未实现，不能据此认定方法不适用。",
              "- 效率补测：0/7，因同卡出现其他计算进程而失败；没有有效独占效率结论。",
              "- LazyEviction：4 个原生 KV 头展开至 28 个，须报告实际 KV 字节、跟踪状态及副本，不能把 slot 比例当成显存比例。",
              "- 原表逐题结果与原环境仍待溯源；原表参考值不与本轮单 seed 拼接为配对统计。", "",
              "详细执行顺序与验收条件见 [后续计划](../../docs/rebuttal/NEXT_STEPS.md)。", "",
              "## 产物", "", "- `RESULTS.md`：数据集为行、方法为列的 EM/F1 结果总表。",
              "- `summary.json`：完整进度、最终/阶段指标和统计口径。",
              "- `per_question/`：已保存主实验的逐题 EM/F1、诊断计时及紧凑缓存审计；不复制题目正文、答案或权重。",
              "- `evidence/`：运行 manifest、队列计划、失败状态、校准与 pilot 结果。",
              "- `source_bindings.json`：所有读取的本地源文件 SHA256。"]
    destination.mkdir(parents=True)
    for rel, value in artifacts.items():
        write_json(destination / rel, value)
    write_json(destination / "summary.json", summary)
    write_json(destination / "source_bindings.json", bindings)
    (destination / "README.md").write_text("\n".join(lines) + "\n")
    (destination / "RESULTS.md").write_text(render_results_table(summary))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/rebuttal_20260928")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.source.resolve(), args.output.resolve())
    print(json.dumps({key: result[key] for key in ("captured_to_utc", "completed_main", "planned_main", "saved_main", "expected_main")}, indent=2))


if __name__ == "__main__":
    main()
