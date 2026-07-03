#!/usr/bin/env python3
"""Write a metrics markdown report from an experiment result JSON."""
import argparse
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def _decode_cache_len(total_len: int, prompt_len: int) -> int:
    return max(0, int(total_len) - int(prompt_len))


def _step_decode_lens_from_result(result: Dict[str, Any]) -> List[int]:
    """Return per-step decode-only cache lengths for one sample."""
    srt = result.get("step_remaining_tokens") or []
    if len(srt) > 1:
        return [int(x) for x in srt[1:]]
    prompt_len = int(result.get("prompt_token_count", 0) or 0)
    out = []
    for t in result.get("step_timings") or []:
        raw = int(t.get("kv_cache_length", 0) or 0)
        if prompt_len > 0 and raw > prompt_len:
            out.append(_decode_cache_len(raw, prompt_len))
        else:
            out.append(raw)
    return out


def compute_derived_stats(data: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate decode-only cache / timing stats from summary + per-sample results."""
    summary = data.get("summary", {}) if isinstance(data, dict) else {}
    results = data.get("results", []) if isinstance(data, dict) else []
    stats: Dict[str, Any] = {}

    final_decode_lens: List[int] = []
    step_decode_lens: List[int] = []
    sample_times: List[float] = []
    peak_mems: List[float] = []

    for r in results:
        if not isinstance(r, dict):
            continue
        final_decode_lens.append(int(r.get("llm_stats", {}).get("final_cache_len", 0) or 0))
        step_decode_lens.extend(_step_decode_lens_from_result(r))
        if isinstance(r.get("sample_time"), (int, float)) and r.get("sample_time", 0) > 0:
            sample_times.append(float(r["sample_time"]))
        if isinstance(r.get("peak_memory_mb"), (int, float)) and r.get("peak_memory_mb", 0) > 0:
            peak_mems.append(float(r["peak_memory_mb"]))

    if final_decode_lens:
        stats["avg_final_decode_cache_len"] = sum(final_decode_lens) / len(final_decode_lens)
        stats["max_final_decode_cache_len"] = max(final_decode_lens)
    if step_decode_lens:
        stats["avg_step_decode_cache_len"] = sum(step_decode_lens) / len(step_decode_lens)
        stats["max_step_decode_cache_len"] = max(step_decode_lens)
    if sample_times:
        stats["avg_sample_time_seconds"] = sum(sample_times) / len(sample_times)
        stats["max_sample_time_seconds"] = max(sample_times)

    timing = summary.get("timing_stats") or {}
    if timing.get("avg_kv_cache_length") is not None:
        stats["avg_step_decode_cache_len"] = timing.get("avg_kv_cache_length")
    if timing.get("max_kv_cache_length") is not None:
        stats["max_step_decode_cache_len"] = timing.get("max_kv_cache_length")
    if timing.get("avg_step_time") is not None:
        stats["avg_step_generation_time_seconds"] = timing.get("avg_step_time")

    if summary.get("avg_final_decode_cache_len") is not None:
        stats["avg_final_decode_cache_len"] = summary.get("avg_final_decode_cache_len")
    if summary.get("max_final_decode_cache_len") is not None:
        stats["max_final_decode_cache_len"] = summary.get("max_final_decode_cache_len")
    if summary.get("avg_sample_time_seconds") is not None:
        stats["avg_sample_time_seconds"] = summary.get("avg_sample_time_seconds")
    if summary.get("max_sample_time_seconds") is not None:
        stats["max_sample_time_seconds"] = summary.get("max_sample_time_seconds")

    return stats


def parse_metrics_markdown(path: str) -> Dict[str, Any]:
    """Parse metrics_*.md written by write_experiment_metrics()."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    out: Dict[str, Any] = {"source_path": path}
    for line in text.splitlines():
        m = re.match(r"^-\s+([A-Za-z0-9_]+):\s*(.+?)\s*$", line.strip())
        if not m:
            continue
        key, raw = m.group(1), m.group(2).strip()
        if raw in ("N/A", "—", "-", ""):
            continue
        try:
            if re.fullmatch(r"-?\d+", raw):
                out[key] = int(raw)
            elif re.fullmatch(r"-?\d+\.\d+", raw):
                out[key] = float(raw)
            else:
                out[key] = raw
        except ValueError:
            out[key] = raw

    json_match = re.search(
        r"## Full Summary JSON\s*```json\s*(.*?)\s*```",
        text,
        flags=re.DOTALL,
    )
    if json_match:
        try:
            out["summary"] = json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    summary = out.get("summary") if isinstance(out.get("summary"), dict) else {}
    if out.get("EM") is None and summary.get("exact_match") is not None:
        out["EM"] = summary.get("exact_match")
    if out.get("F1") is None and summary.get("f1_score") is not None:
        out["F1"] = summary.get("f1_score")
    if out.get("avg_sample_time_seconds") is None and summary.get("avg_sample_time_seconds") is not None:
        out["avg_sample_time_seconds"] = summary.get("avg_sample_time_seconds")
    if out.get("max_sample_time_seconds") is None and summary.get("max_sample_time_seconds") is not None:
        out["max_sample_time_seconds"] = summary.get("max_sample_time_seconds")
    if out.get("avg_step_decode_cache_len") is None and summary.get("avg_step_decode_cache_len") is not None:
        out["avg_step_decode_cache_len"] = summary.get("avg_step_decode_cache_len")
    if out.get("max_step_decode_cache_len") is None and summary.get("max_step_decode_cache_len") is not None:
        out["max_step_decode_cache_len"] = summary.get("max_step_decode_cache_len")
    if out.get("avg_final_decode_cache_len") is None and summary.get("avg_final_decode_cache_len") is not None:
        out["avg_final_decode_cache_len"] = summary.get("avg_final_decode_cache_len")
    if out.get("max_final_decode_cache_len") is None and summary.get("max_final_decode_cache_len") is not None:
        out["max_final_decode_cache_len"] = summary.get("max_final_decode_cache_len")
    if out.get("n_samples") is None:
        n = summary.get("total_samples", summary.get("n_samples"))
        if n is not None:
            out["n_samples"] = n

    return out


def build_metrics_lines(
    data: Dict[str, Any],
    dataset: str,
    method: str,
    cache_ratio: str = "",
) -> List[str]:
    summary = data.get("summary", {})
    derived = compute_derived_stats(data)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    lines = [
        f"# Experiment Metrics ({dataset})",
        "",
        f"- recorded_at_utc: {ts}",
        f"- dataset: {dataset}",
        f"- method: {method}",
    ]
    if cache_ratio:
        lines.append(f"- cache_ratio: {cache_ratio}")

    n_samples = summary.get("total_samples", summary.get("n_samples", "N/A"))
    lines.extend(
        [
            f"- n_samples: {n_samples}",
            f"- EM: {summary.get('exact_match', 'N/A')}",
            f"- F1: {summary.get('f1_score', 'N/A')}",
            "",
            "## Timing",
            f"- total_time_seconds: {summary.get('total_time_seconds', 'N/A')}",
            f"- avg_time_per_sample: {summary.get('avg_time_per_sample', 'N/A')}",
        ]
    )
    if derived.get("avg_sample_time_seconds") is not None:
        lines.append(f"- avg_sample_time_seconds: {derived['avg_sample_time_seconds']:.2f}")
    if derived.get("max_sample_time_seconds") is not None:
        lines.append(f"- max_sample_time_seconds: {derived['max_sample_time_seconds']:.2f}")
    if derived.get("avg_step_generation_time_seconds") is not None:
        lines.append(
            f"- avg_step_generation_time_seconds: {derived['avg_step_generation_time_seconds']:.4f}"
        )

    lines.extend(["", "## KV Cache (decode only, prompt excluded)", ""])
    if derived.get("avg_final_decode_cache_len") is not None:
        lines.append(f"- avg_final_decode_cache_len: {derived['avg_final_decode_cache_len']:.1f}")
    if derived.get("max_final_decode_cache_len") is not None:
        lines.append(f"- max_final_decode_cache_len: {int(derived['max_final_decode_cache_len'])}")
    if derived.get("avg_step_decode_cache_len") is not None:
        lines.append(f"- avg_step_decode_cache_len: {derived['avg_step_decode_cache_len']:.1f}")
    if derived.get("max_step_decode_cache_len") is not None:
        lines.append(f"- max_step_decode_cache_len: {int(derived['max_step_decode_cache_len'])}")

    lines.extend(
        [
            "",
            "## GPU Memory (inference only, model weights excluded)",
            f"- model_param_mb: {summary.get('model_param_mb', 'N/A')}",
            f"- avg_peak_memory_mb: {summary.get('avg_peak_memory_mb', 'N/A')}",
            f"- max_peak_memory_mb: {summary.get('max_peak_memory_mb', 'N/A')}",
            "",
        ]
    )

    if summary.get("total_prune_count") is not None:
        lines.extend([f"- total_prune_count: {summary.get('total_prune_count')}", ""])

    step_dist = summary.get("step_count_distribution")
    if isinstance(step_dist, dict) and step_dist:
        lines.extend([
            "## Step Count Distribution",
            "",
            "| Steps | # Samples |",
            "|-------|-----------|",
        ])
        for steps, count in sorted(step_dist.items(), key=lambda x: int(x[0])):
            lines.append(f"| {steps} | {count} |")
        lines.extend([
            "",
            f"- avg_num_steps: {summary.get('avg_num_steps', 'N/A')}",
            f"- min_num_steps: {summary.get('min_num_steps', 'N/A')}",
            f"- max_num_steps: {summary.get('max_num_steps', 'N/A')}",
            "",
        ])

    lines.extend(
        [
            "## Full Summary JSON",
            "```json",
            json.dumps(summary, ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    )
    return lines


def write_experiment_metrics(
    result_json: str,
    dataset: str,
    method: str,
    output_file: str,
    cache_ratio: str = "",
) -> bool:
    if not os.path.exists(result_json):
        print(f"[WARN] Result file not found, skip metrics record: {result_json}")
        return False

    with open(result_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = build_metrics_lines(data, dataset=dataset, method=method, cache_ratio=cache_ratio)
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"[INFO] Metrics recorded: {output_file}")
    return True


def metrics_output_path(result_json: str, method: str, cache_ratio: str = "") -> str:
    tag = method
    if cache_ratio:
        try:
            ratio_tag = str(int(float(cache_ratio) * 100))
        except (TypeError, ValueError):
            ratio_tag = str(cache_ratio).replace(".", "")
        tag = f"{method}_r{ratio_tag}"
    return os.path.join(os.path.dirname(result_json), f"metrics_{tag}.md")


def main() -> int:
    parser = argparse.ArgumentParser(description="Record EM/F1/time metrics from result JSON.")
    parser.add_argument("--result_json", required=True, type=str)
    parser.add_argument("--dataset", required=True, type=str)
    parser.add_argument("--method", required=True, type=str)
    parser.add_argument("--output_file", required=True, type=str)
    parser.add_argument("--cache_ratio", type=str, default="")
    args = parser.parse_args()

    ok = write_experiment_metrics(
        args.result_json,
        dataset=args.dataset,
        method=args.method,
        output_file=args.output_file,
        cache_ratio=args.cache_ratio,
    )
    return 0 if ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
