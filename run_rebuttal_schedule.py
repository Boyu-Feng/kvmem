"""Paired, schedule-controlled HotpotQA experiments with auditable manifests.

H2O/TOVA here name this repository's selectors, using a shared EOS probe.
They are adaptations, not claims of reproducing the official baseline code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parent
METHODS = {
    "fullkv": ("none", "step"),
    "h2o_token": ("h2o", "token"),
    "h2o_step": ("h2o", "step"),
    "tova_token": ("tova", "token"),
    "tova_step": ("tova", "step"),
    "stepkv": ("step_aware_h2o", "step"),
    "stepkv_no_step_score": ("step_aware_h2o", "step"),
}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def source_hashes():
    paths = [ROOT / "run_rebuttal_schedule.py", ROOT / "run_all_wiki_experiments_v2.py",
             ROOT / "token_tracker.py"]
    for directory in ("models", "kv_cache", "retrievers"):
        paths.extend(sorted((ROOT / directory).glob("*.py")))
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def method_config(method, ratio):
    mode, schedule = METHODS[method]
    return {
        "pruning_mode": mode, "pruning_schedule": schedule,
        "cache_ratio": ratio, "attn_mode": "scoring_forward",
        "protect_prompt": True, "prune_every_n": 1,
        "observation_window": 32 if mode == "h2o" else 0,
        "step_aware_alpha": .8,
        "step_aware_beta": 0.0 if method == "stepkv_no_step_score" else .8,
        "step_poolwise_prune": mode == "step_aware_h2o",
        "step_reward_weight": .85, "step_citation_weight": .15,
        "step_repeat_penalty": .3,
        # Preserve the submitted-code utility for this first schedule control.
        # A self-reuse correction must be a separately labeled experiment.
        "step_exclude_self_reuse": False,
        "raise_on_runtime_error": True,
    }


def main():
    raise SystemExit("controlled_v2 is retired by user request. Use run_rebuttal_legacy.py; original results remain authoritative.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--ratio", type=float, choices=[.2, .5], default=.2)
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=7)
    parser.add_argument("--assets", type=Path, default=Path("/data/experiment/fengboyu/stepkv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--purpose", choices=["smoke", "main"], default="main")
    args = parser.parse_args()
    if args.samples <= 0 or args.sample_start < 0 or args.max_steps <= 0:
        parser.error("Sample counts and max steps must be positive; sample start must be nonnegative")
    args.output.mkdir(parents=True, exist_ok=True)
    from datasets import load_dataset
    import torch
    import transformers
    import run_all_wiki_experiments_v2 as base
    from models.RebuttalLLM import RebuttalLLM
    from retrievers.WikiBM25Retriever import WikiBM25Retriever

    assets = {kind: json.loads((args.assets / f"{kind}_ready.json").read_text())
              for kind in ("model", "data")}
    files = sorted(str(p) for p in (args.assets / "datasets/hotpotqa/distractor").glob("validation-*.parquet"))
    data = load_dataset("parquet", data_files={"validation": files}, split="validation",
                        cache_dir=str(args.assets / "datasets/arrow_cache"))
    if len(data) != 7405:
        raise RuntimeError(f"Unexpected HotpotQA validation size: {len(data)}")
    base.MODEL_PATH = assets["model"]["path"]
    base.RANDOM_SEED = args.seed
    base.NUM_SAMPLES = args.sample_start + args.samples
    base.MAX_STEPS = args.max_steps
    base.CHECKPOINT_INTERVAL = 1
    # Keep trajectories and per-step timings through resume for paired analysis.
    base._REACT_KV_CHECKPOINT_DROP_KEYS = ("debug_payload",)
    selected = base.select_samples(data)[args.sample_start:]
    if len(selected) != args.samples:
        raise RuntimeError("Requested sample selection was not fully satisfied")
    config = method_config(args.method, args.ratio)
    fingerprint = {
        "runtime": RebuttalLLM.RUNTIME_VERSION, "purpose": args.purpose,
        "method": args.method, "config_override": config,
        "seed": args.seed, "sample_start": args.sample_start,
        "max_steps": args.max_steps, "assets": assets,
        "sample_ids": [s["id"] for _, s in selected],
        "source_hashes": source_hashes(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "decoding": "raw-logit argmax; no sampling or repetition penalty (1.0); max_new_tokens=256 per block; materialize every emitted token; absolute logical RoPE positions; preserve pre-marker characters in shared BPE tokens",
        "estimator": "shared full-cache EOS attention probe; last three layers",
        "budget": "protected prompt + floor(ratio * cumulative logical non-prompt tokens); resident candidates only",
        "timing_scope": "shared GPU; diagnostic only, not isolated benchmark",
        "baseline_scope": "repository H2O/TOVA selectors with controlled estimator and schedule; not official baseline reproduction",
        "no_step_scope": "beta=0; step pool retention floor still active",
    }
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["fingerprint"] != fingerprint:
            raise RuntimeError("Refusing to resume with changed protocol/source/sample IDs; use a new output directory")
    else:
        gpu = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout
        atomic_json(manifest_path, {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                   "fingerprint": fingerprint, "gpu_at_start": gpu,
                                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                                   "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()})
    status_path = args.output / "status.json"
    atomic_json(status_path, {"status": "loading_retriever", "pid": os.getpid(), "started": time.time()})
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required for these 7B experiments")
        torch.manual_seed(args.seed)
        retriever = WikiBM25Retriever(assets["data"]["wiki_index"])
        if len(retriever.titles) != 5233235 or retriever.corpus_texts is None:
            raise RuntimeError("Full Wikipedia retrieval assets failed validation")
        atomic_json(status_path, {"status": "running", "pid": os.getpid(), "started": time.time()})
        mode = METHODS[args.method][0]
        base.run_react_kv_experiment(data, selected, retriever, mode,
            str(args.output / "results.json"), str(args.output / "checkpoint.json"),
            kv_config_override=config, metrics_method=f"controlled_{args.method}",
            llm_class=RebuttalLLM)
        result = json.loads((args.output / "results.json").read_text())
        rows = result["results"]
        if {r["id"] for r in rows} != {s["id"] for _, s in selected} or len(rows) != args.samples:
            raise RuntimeError("Output sample IDs are incomplete or duplicated")
        atomic_json(status_path, {"status": "complete", "pid": os.getpid(),
                                  "finished": time.time(), "samples": len(rows),
                                  "exact_match": result["summary"]["exact_match"],
                                  "f1_score": result["summary"]["f1_score"]})
    except BaseException:
        atomic_json(status_path, {"status": "failed", "pid": os.getpid(),
                                  "at": time.time(), "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
