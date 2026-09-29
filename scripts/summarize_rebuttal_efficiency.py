"""Report completed, comparable W3 measurements with explicit absolute units."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_rebuttal_legacy import atomic_json
from scripts.run_rebuttal_efficiency_queue import DEFAULT_OUTPUT, jobs, validate_completed


def summarize_directory(root):
    rows, reference_protocol = [], None
    for job in jobs():
        directory = root / job["name"]
        status_path = directory / "status.json"
        if not status_path.exists() or json.loads(status_path.read_text()).get("status") != "complete":
            continue
        validate_completed(directory, job)
        protocol = json.loads((directory / "manifest.json").read_text())["protocol"]
        if reference_protocol is None:
            reference_protocol = protocol
        for key in ("sample_ids", "warmup_ids", "seed", "repeats", "hardware", "model", "dtype",
                    "assets", "core_hashes", "measurement_hashes", "torch", "transformers"):
            if reference_protocol[key] != protocol[key]:
                raise RuntimeError(f"Incomparable efficiency results ({key}): {directory}")
        data = json.loads((directory / "measurements.json").read_text())
        rows.append({**job, **data["summary"]})
    full = next((r for r in rows if r["method"] == "fullkv"), None)
    for row in rows:
        for key in ("mean_episode_seconds", "mean_question_mean_boundary_total_mib", "max_peak_active_total_mib",
                    "mean_clean_peak_allocated_MiB", "mean_clean_peak_above_parameter_MiB"):
            row[key + "_over_fullkv"] = row[key] / full[key] if full and full[key] else None
    atomic_json(root / "SUMMARY.json", {"normalization": "method absolute aggregate / FullKV same aggregate",
                                       "completed_conditions": len(rows), "target_conditions": 7, "rows": rows})
    lines = ["# W3：原评测实现的独立效率补测", "",
        "Qwen2.5-7B-Instruct，BF16，seed=233，同一张 A100-SXM4-40GB；固定前20题、每题2次主计时。",
        "表中只列已完成且通过 clean/diagnostic 输出与 KV 一致性检查的条件。以下数据不替换原论文 EM/F1。", "",
        "主延迟为同步后的完整 episode 墙钟时间，含检索、解析、评分、KV 管理及原诊断输出；不含加载、预热或测量结果写盘。",
        "KV 均为含 protected prompt/query 的实际 K/V 张量 payload。步末均值先题内、再题间平均；步内峰值为所有题的最大观测值。",
        "Allocated peak 是每题 GPU allocator 峰值的平均值，包含权重与工作张量，不能称为 KV 大小。", "",
        "| 方法 | trajectory 预算 | s/题 | 延迟 / FullKV | 步末 KV tokens | 步末 KV MiB | 步内峰值 KV tokens | 步内峰值 KV MiB | Allocated peak MiB | 平均步数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        norm = row["mean_episode_seconds_over_fullkv"]
        norm_text = f"{norm:.3f}" if norm is not None else "待 FullKV"
        lines.append(f"| {row['method']} | {row['display_budget']:.0%} | {row['mean_episode_seconds']:.3f} | {norm_text} | "
                     f"{row['mean_question_mean_boundary_tokens']:.2f} | {row['mean_question_mean_boundary_total_mib']:.3f} | "
                     f"{row['max_peak_active_tokens']} | {row['max_peak_active_total_mib']:.3f} | "
                     f"{row['mean_clean_peak_allocated_MiB']:.2f} | {row['mean_num_steps']:.2f} |")
    lines += ["", f"已完成 {len(rows)}/7 个条件。", "",
        "归一化统一为 X(method)/X(FullKV)，是同一汇总指标之比。吞吐/加速比若另报，使用反向时间比且明确标注。",
        "trajectory-only、逐题峰值均值、probe cache 与分项诊断见每个条件的 measurements.json；诊断区间有嵌套、含同步开销，不能相加或作为主延迟。",
        "自由生成的轨迹与工作量可能随方法改变；这是实际端到端测量，不能据此单独断言等长 decode kernel 加速。"]
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    summarize_directory(args.root)
