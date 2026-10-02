"""Publish validated, completed multiseed results without raw questions or logs."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/rebuttal_multiseed_20261001"
QUEUES = ("sidequest", "thinkkv_eviction_only", "seed42/fixed_v2", "seed3407/fixed_v2")


def read_json(path: Path, bindings: dict):
    raw = path.read_bytes()
    bindings[str(path.relative_to(SOURCE))] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return json.loads(raw)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def export(destination: Path):
    if destination.exists():
        raise FileExistsError(destination)
    bindings, groups, per_question = {}, [], {}
    for queue in QUEUES:
        base = SOURCE / queue
        plan = read_json(base / "queue_plan.json", bindings)
        read_json(base / "queue_status.json", bindings)
        for job in plan["jobs"]:
            if job["purpose"] != "main":
                continue
            folder = base / job["name"]
            status_file = folder / "status.json"
            status = read_json(status_file, bindings) if status_file.exists() else {"status": "pending"}
            item = {"queue": queue, "dataset": job["dataset"], "method": job.get("method", "thinkkv_eviction_only"),
                    "seed": job["seed"], "budget": job.get("ratio", "adaptive"), "expected": job["samples"],
                    "status": status["status"]}
            if status["status"] == "complete":
                manifest = read_json(folder / "manifest.json", bindings)
                expected = manifest.get("ids", manifest.get("fingerprint", {}).get("sample_ids"))
                rows = read_json(folder / "checkpoint.json", bindings)
                final_document = read_json(folder / "results.json", bindings)
                final = final_document["results"]
                if expected is None or len(expected) != job["samples"] or len(set(expected)) != len(expected):
                    raise ValueError(f"Invalid manifest IDs: {folder}")
                if len(rows) != job["samples"] or len(final) != len(rows):
                    raise ValueError(f"Incomplete final rows: {folder}")
                ids = [r["id"] for r in rows]
                if len(set(ids)) != len(ids) or set(ids) != set(expected):
                    raise ValueError(f"Result IDs do not match manifest: {folder}")
                scores = [(r["id"], float(r["em"]), float(r["f1"])) for r in rows]
                final_scores = [(r["id"], float(r["em"]), float(r["f1"])) for r in final]
                if scores != final_scores or any(em not in (0, 1) or not 0 <= f1 <= 1 for _, em, f1 in scores):
                    raise ValueError(f"Checkpoint/final scores differ or are invalid: {folder}")
                em, f1 = 100 * mean(s[1] for s in scores), 100 * mean(s[2] for s in scores)
                summary_file = folder / "SUMMARY.json"
                if summary_file.exists():
                    summary = read_json(summary_file, bindings)
                    if abs(em - summary["exact_match"]) > 1e-6 or abs(f1 - summary["f1_score"]) > 1e-6:
                        raise ValueError(f"Summary scores differ: {folder}")
                    item["reported_summary"] = summary
                else:
                    result_summary = final_document["summary"]
                    if abs(em - result_summary["exact_match"]) > 1e-6 or abs(f1 - result_summary["f1_score"]) > 1e-6:
                        raise ValueError(f"Result summary scores differ: {folder}")
                item.update(samples=len(rows), em_percent=em, f1_percent=f1)
                key = f"{queue}/{job['name']}"
                per_question[key] = [{"id": r["id"], "index": r["index"], "em": bool(r["em"]), "f1": r["f1"]} for r in rows]
                item["per_question"] = f"per_question/{key}.json"
            groups.append(item)
    complete = [g for g in groups if g["status"] == "complete"]
    snapshot = {"captured_at_utc": datetime.now(timezone.utc).isoformat(), "completed": len(complete),
                "planned": len(groups), "evaluations": sum(g["samples"] for g in complete), "groups": groups,
                "notes": ["只有已完成组发布分数；运行中和待运行组只列状态。",
                          "seed 42、3407 与 seed 233 的题目集合不保证相同。",
                          "SideQuest 使用独立的未微调解码器和自适应预算。",
                          "ThinkKV 是仅按片段淘汰的 QA 适配，没有 KV 量化。",
                          "名义 token 保留比例不等于相同 KV 字节数或延迟。"]}
    lines = ["# 三 seed 扩展实验结果快照", "", f"捕获时间：{snapshot['captured_at_utc']}。",
             f"已完成 **{len(complete)}/{len(groups)} 组**，共 **{snapshot['evaluations']:,} 条逐题评测**。",
             "运行中的组只列状态，不发布阶段分数。", "",
             "| 数据集 | 方法 | seed | 名义预算 | N | EM (%) | F1 (%) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for g in complete:
        budget = f"{g['budget']:.0%}" if isinstance(g["budget"], float) else "自适应"
        lines.append(f"| {g['dataset']} | {g['method']} | {g['seed']} | {budget} | {g['samples']} | "
                     f"{g['em_percent']:.2f} | {g['f1_percent']:.2f} |")
    lines += ["", "## 解读限制", ""] + [f"- {n}" for n in snapshot["notes"]]
    lines += ["", "逐组状态、完整精度指标及机制汇总见 [summary.json](summary.json)；"
              "[逐题分数](per_question/)只包含 ID、索引和 EM/F1；"
              "[源文件哈希](source_bindings.json)用于核对本地原始结果。", ""]
    destination.mkdir(parents=True)
    write_json(destination / "summary.json", snapshot)
    write_json(destination / "source_bindings.json", bindings)
    for key, rows in per_question.items():
        write_json(destination / "per_question" / f"{key}.json", rows)
    (destination / "RESULTS.md").write_text("\n".join(lines))
    return snapshot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = export(args.output.resolve())
    print(f"Exported {result['completed']}/{result['planned']} groups to {args.output}")
