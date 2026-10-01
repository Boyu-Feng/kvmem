"""Evaluate ThinKV's one-category segment eviction without KV quantization."""
from __future__ import annotations

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
    prompt = base.REACT_KV_INITIAL_PROMPT.format(examples=base.REACT_EXAMPLES, question=question) + "Thought 1:"
    stops = ["\nObservation", "\nQuestion:"]
    response, prompt_kv, generated_kv = llm.generate_first(prompt, max_new_tokens=256, stop_strings=stops)
    del prompt_kv, generated_kv
    lookup = dict(page=None, lookup_keyword=None, lookup_list=None, lookup_cnt=0)
    trajectory = []
    answer, reason = "", "turn_limit"
    for step in range(1, max_steps + 1):
        thought, action, arg = base.parse_action_original(f"Thought {step}:" + response, step)
        row = dict(step=step, response=response, thought=thought, action_type=action, action_arg=arg)
        trajectory.append(row)
        if action == "finish":
            answer, reason = arg or "", "finish"
            break
        if action not in ("search", "lookup"):
            reason = "unparsable_action"
            break
        observation, lookup = base.execute_action(action, arg, retriever, lookup)
        observation = observation.replace("\\n", "")
        row["observation"] = observation
        if step < max_steps:
            response = llm.generate_incremental(
                f"\nObservation {step}: {observation}\nThought {step + 1}:",
                max_new_tokens=256, stop_strings=stops)
    return answer, reason, trajectory


def summarize(directory, rows, expected):
    n = len(rows)
    audit = [r["audit"] for r in rows]
    report = dict(status="complete" if n == expected else "partial", samples=n,
        expected_samples=expected,
        exact_match=100 * sum(r["em"] for r in rows) / n if n else None,
        f1_score=100 * sum(r["f1"] for r in rows) / n if n else None,
        questions_with_eviction=sum(bool(a["evictions"]) for a in audit),
        eviction_events=sum(len(a["evictions"]) for a in audit),
        mean_trajectory_retention=sum(a["trajectory_retained"] / a["trajectory_logical"]
                                      if a["trajectory_logical"] else 1. for a in audit) / n if n else None,
        total_resident_bytes=sum(a["resident_bytes"] for a in audit),
        mean_seconds=sum(r["seconds"] for r in rows) / n if n else None,
        scope="Single-category Qwen QA adaptation; segment eviction only, native-dtype KV")
    atomic_json(directory / "SUMMARY.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["hotpotqa", "2wiki", "musique"], required=True)
    parser.add_argument("--seed", type=int, choices=[233, 42, 3407], required=True)
    parser.add_argument("--ratio", type=float, choices=[.2, .5], required=True)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=7)
    parser.add_argument("--purpose", choices=["smoke", "main"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.sample_start < 0 or args.max_steps < 1:
        parser.error("Positive samples/max-steps and nonnegative sample-start required")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    state = directory / "status.json"
    llm = None
    try:
        import torch
        import transformers
        from datasets import load_dataset
        import run_all_wiki_experiments_v2 as base
        from retrievers.WikiBM25Retriever import WikiBM25Retriever
        from scripts.thinkkv_eviction_runtime import ThinKVEvictionOnlyLLM

        assets = Path("/data/experiment/fengboyu/stepkv")
        metadata = {kind: json.loads((assets / f"{kind}_ready.json").read_text()) for kind in ("model", "data")}
        original = verify_original_sources(ROOT)
        if args.dataset == "hotpotqa":
            files = sorted(str(p) for p in (assets / "datasets/hotpotqa/distractor").glob("validation-*.parquet"))
            data = load_dataset("parquet", data_files={"validation": files}, split="validation",
                                cache_dir=str(assets / "datasets/arrow_cache"))
            if len(data) != 7405:
                raise RuntimeError("Unexpected HotpotQA split size")
        else:
            dataset_asset = json.loads((assets / "datasets" / args.dataset / "ready.json").read_text())
            data_path = Path(dataset_asset["path"])
            if hashlib.sha256(data_path.read_bytes()).hexdigest() != dataset_asset["sha256"]:
                raise RuntimeError("Dataset SHA256 mismatch")
            data = json.loads(data_path.read_text())
            expected = {"2wiki": 12576, "musique": 2417}[args.dataset]
            if len(data) != expected or len({r["id"] for r in data}) != expected:
                raise RuntimeError("Unexpected dataset length or duplicate IDs")
            metadata["evaluation_dataset"] = dataset_asset
        base.RANDOM_SEED = args.seed
        base.NUM_SAMPLES = args.sample_start + args.samples
        selected = base.select_samples(data)[args.sample_start:]
        if len(selected) != args.samples:
            raise RuntimeError("Incomplete selected sample count")
        code = ["scripts/run_thinkkv_eviction.py", "scripts/thinkkv_eviction_runtime.py",
                "models/RebuttalLLM.py", "token_tracker.py"]
        manifest = dict(protocol="thinkkv_eviction_only_qwen_v1", dataset=args.dataset,
            seed=args.seed, ratio=args.ratio, purpose=args.purpose, samples=args.samples,
            sample_start=args.sample_start, max_steps=args.max_steps,
            ids=[r["id"] for _, r in selected], assets=metadata,
            original_source_hashes=original,
            source_hashes={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in code},
            torch=torch.__version__, transformers=transformers.__version__,
            quantization="none; all retained KV in model native dtype",
            eviction="single 128-token category; oldest completed segment; 64/32/16/8/4; post-RoPE K-means medoid",
            protected_prompt=True, budget="prompt + ratio * logical non-prompt tokens, active segment and min-4 floor",
            decode="independent greedy loop, logical RoPE, same stop/crop as RebuttalLLM",
            kernel="native PyTorch attention; no CT/PagedAttention kernel",
            scope="Eviction-only ablation of paper E.10 for QA; no quantization")
        mp = directory / "manifest.json"
        if mp.exists() and json.loads(mp.read_text()) != manifest:
            raise RuntimeError("Changed ThinkKV manifest; use fresh output directory")
        atomic_json(mp, manifest)
        cp = directory / "checkpoint.json"
        rows = json.loads(cp.read_text()) if cp.exists() else []
        if len({r["id"] for r in rows}) != len(rows) or not {r["id"] for r in rows} <= set(manifest["ids"]):
            raise RuntimeError("Invalid checkpoint IDs")
        atomic_json(state, dict(status="loading", pid=os.getpid(), updated=time.time(), saved=len(rows)))
        retriever = WikiBM25Retriever(metadata["data"]["wiki_index"])
        if len(retriever.titles) != 5233235 or retriever.corpus_texts is None:
            raise RuntimeError("Incomplete Wikipedia retrieval corpus")
        llm = ThinKVEvictionOnlyLLM(metadata["model"]["path"], args.ratio)
        done = {r["id"] for r in rows}
        for index, sample in selected:
            if sample["id"] in done:
                continue
            atomic_json(state, dict(status="running", pid=os.getpid(), updated=time.time(),
                                    saved=len(rows), total=args.samples, sample_id=sample["id"]))
            torch.cuda.synchronize()
            begin = time.perf_counter()
            answer, reason, trajectory = episode(llm, sample["question"], retriever, args.max_steps)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - begin
            audit = llm.thin_audit()
            if audit["prompt_tokens"] < 1 or audit["resident_bytes"] < 1:
                raise RuntimeError("No resident KV recorded")
            rows.append(dict(id=sample["id"], index=index, prediction=answer, gold=sample["answer"],
                termination=reason, em=base.exact_match(answer, sample["answer"]),
                f1=base.f1_score(answer, sample["answer"]), seconds=seconds,
                trajectory=trajectory, audit=audit))
            atomic_json(cp, rows)
            summarize(directory, rows, args.samples)
            print(f"Saved {len(rows)}/{args.samples}: {sample['id']} {reason} "
                  f"retention={audit['trajectory_retained']}/{audit['trajectory_logical']} "
                  f"evictions={len(audit['evictions'])}", flush=True)
        report = summarize(directory, rows, args.samples)
        atomic_json(directory / "results.json", dict(summary=report, results=rows))
        atomic_json(state, dict(status="complete", pid=os.getpid(), updated=time.time(), samples=len(rows)))
    except BaseException:
        atomic_json(state, dict(status="failed", pid=os.getpid(), updated=time.time(),
                                traceback=traceback.format_exc()))
        raise
    finally:
        if llm is not None:
            del llm


if __name__ == "__main__":
    main()
