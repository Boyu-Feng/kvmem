"""Run the restored evaluator; keep inference behavior in the original source files."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time
import traceback

ROOT = Path(__file__).resolve().parent
SNAPSHOT = ROOT / "results/rebuttal_20260927/source_snapshot/before_changes.tar"
CORE_FILES = (
    "run_all_wiki_experiments_v2.py", "models/QwenLLMWithKVCache.py",
    "models/QwenLLM.py", "models/model_paths.py", "kv_cache/kv_cache_manager.py",
    "kv_cache/pruning_strategy.py", "kv_cache/h2o_scorer.py", "token_tracker.py",
    "retrievers/WikiBM25Retriever.py",
)
CORE_METHODS = {"fullkv": "none", "h2o": "h2o", "tova": "tova",
                "stepkv": "step_aware_h2o", "stepkv_no_step_score": "step_aware_h2o"}
EXTENSIONS = ("flowkv_style", "rkv", "lazyeviction")


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temp.replace(path)


def verify_original_sources(source):
    """Do not silently restore only some of the old inference behavior."""
    verified = {}
    with tarfile.open(SNAPSHOT) as archive:
        for name in CORE_FILES:
            expected = archive.extractfile(name).read()
            actual = (source / name).read_bytes()
            if actual != expected:
                raise RuntimeError(f"Original evaluation source differs from snapshot: {source / name}")
            verified[name] = hashlib.sha256(actual).hexdigest()
    return verified


def source_hashes(source):
    paths = [source / "run_all_wiki_experiments_v2.py", source / "token_tracker.py"]
    for name in ("models", "kv_cache", "retrievers", "vendor"):
        paths.extend(sorted((source / name).rglob("*.py")))
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    hashes["__legacy_runner__"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=[*CORE_METHODS, *EXTENSIONS], required=True)
    parser.add_argument("--source", type=Path, default=ROOT)
    parser.add_argument("--ratio", type=float, choices=[.2, .5], default=.2)
    parser.add_argument("--seed", type=int, choices=[233], default=233)
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--sample-start", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=7)
    parser.add_argument("--assets", type=Path, default=Path("/data/experiment/fengboyu/stepkv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--purpose", choices=["smoke", "main"], default="main")
    args = parser.parse_args()
    if args.samples <= 0 or args.sample_start < 0 or args.max_steps <= 0:
        parser.error("Positive sample/step counts and nonnegative sample start required")
    source = args.source.resolve()
    original_hashes = verify_original_sources(source)
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("checkpoint.json", "results.json")) and not (args.output / "manifest.json").exists():
        raise RuntimeError("Existing results lack a legacy manifest; use a new output directory")
    sys.path.insert(0, str(source))
    import torch
    import transformers
    from datasets import load_dataset
    base = importlib.import_module("run_all_wiki_experiments_v2")
    model_module = importlib.import_module("models.QwenLLMWithKVCache")
    original_class = model_module.QwenLLMWithKVCache
    mode = CORE_METHODS.get(args.method)
    config = {"cache_ratio": args.ratio}
    scope = "Unmodified pre-rebuttal evaluator and inference class; original per-method defaults"
    if args.method == "stepkv_no_step_score":
        config["step_aware_beta"] = 0.0
    if args.method in EXTENSIONS:
        extension = importlib.import_module("models.LegacyBaseline")
        if extension.LEGACY_METHOD != args.method:
            raise RuntimeError("Requested method does not match the isolated baseline source")
        if not issubclass(extension.LegacyBaselineLLM, original_class):
            raise RuntimeError("Extension must inherit the original inference class")
        config = {**extension.LEGACY_CONFIG_OVERRIDE, "cache_ratio": args.ratio}
        mode = config["pruning_mode"]
        scope = extension.LEGACY_SCOPE
        # The original evaluator imports this symbol inside run_react_kv_experiment.
        # Only the new baseline receives the adapter; original methods keep their class.
        model_module.QwenLLMWithKVCache = extension.LegacyBaselineLLM
    assets = {kind: json.loads((args.assets / f"{kind}_ready.json").read_text()) for kind in ("model", "data")}
    model_path = Path(assets["model"]["path"])
    files = sorted(str(p) for p in (args.assets / "datasets/hotpotqa/distractor").glob("validation-*.parquet"))
    data = load_dataset("parquet", data_files={"validation": files}, split="validation",
                        cache_dir=str(args.assets / "datasets/arrow_cache"))
    if len(data) != 7405:
        raise RuntimeError("Unexpected HotpotQA validation size")
    base.MODEL_PATH = str(model_path)
    base.RANDOM_SEED = args.seed
    base.NUM_SAMPLES = args.sample_start + args.samples
    base.MAX_STEPS = args.max_steps
    # I/O frequency only: do not change inference, scorer, stop or exception behavior.
    base.CHECKPOINT_INTERVAL = 1
    selected = base.select_samples(data)[args.sample_start:]
    if len(selected) != args.samples:
        raise RuntimeError("Insufficient selected samples")
    generation_config = json.loads((model_path / "generation_config.json").read_text())
    fingerprint = {
        "runtime": "restored_original_evaluator_v1", "method": args.method,
        "purpose": args.purpose, "source": str(source), "seed": args.seed,
        "sample_start": args.sample_start, "max_steps": args.max_steps,
        "sample_ids": [s["id"] for _, s in selected], "config_override": config,
        "source_hashes": source_hashes(source), "original_source_hashes": original_hashes,
        "assets": assets, "torch": torch.__version__, "transformers": transformers.__version__,
        "generation_config": generation_config, "scope": scope,
        "provenance": "Restores the local before_changes snapshot; exact paper-run assets/environment not yet verified",
        "decoding": "Original generate/token-loop, original positions, stop/crop, first-step behavior and per-method attention defaults",
        "io_changes": "checkpoint every question; optional baseline audit in a separate sidecar",
    }
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())["fingerprint"] != fingerprint:
            raise RuntimeError("Changed protocol/source/sample IDs; use a new output directory")
    else:
        atomic_json(manifest_path, {"created": time.time(), "fingerprint": fingerprint})
    status_path = args.output / "status.json"
    audit_path = args.output / "baseline_audit.json"
    audits = json.loads(audit_path.read_text()) if audit_path.exists() else {}
    ids_by_question = {sample["question"]: sample["id"] for _, sample in selected}
    original_episode = base._run_react_kv_episode

    def audited_episode(question, llm, *episode_args, **episode_kwargs):
        result = original_episode(question, llm, *episode_args, **episode_kwargs)
        if hasattr(llm, "legacy_audit"):
            audits[ids_by_question[question]] = llm.legacy_audit()
            atomic_json(audit_path, audits)
        return result

    if args.method in EXTENSIONS:
        base._run_react_kv_episode = audited_episode
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required")
        atomic_json(status_path, {"status": "loading_retriever", "pid": os.getpid(), "started": time.time()})
        from retrievers.WikiBM25Retriever import WikiBM25Retriever
        retriever = WikiBM25Retriever(assets["data"]["wiki_index"])
        if len(retriever.titles) != 5233235 or retriever.corpus_texts is None:
            raise RuntimeError("Full Wikipedia retrieval assets failed validation")
        atomic_json(status_path, {"status": "running", "pid": os.getpid(), "started": time.time()})
        base.run_react_kv_experiment(data, selected, retriever, mode,
            str(args.output / "results.json"), str(args.output / "checkpoint.json"),
            kv_config_override=config, metrics_method=f"legacy_{args.method}")
        result = json.loads((args.output / "results.json").read_text())
        rows = result["results"]
        if len(rows) != args.samples or {r["id"] for r in rows} != set(fingerprint["sample_ids"]):
            raise RuntimeError("Incomplete or duplicated output samples")
        atomic_json(status_path, {"status": "complete", "pid": os.getpid(), "finished": time.time(),
                    "samples": len(rows), "exact_match": result["summary"]["exact_match"],
                    "f1_score": result["summary"]["f1_score"]})
    except BaseException:
        atomic_json(status_path, {"status": "failed", "pid": os.getpid(), "at": time.time(),
                    "traceback": traceback.format_exc()})
        raise
    finally:
        model_module.QwenLLMWithKVCache = original_class
        base._run_react_kv_episode = original_episode


if __name__ == "__main__":
    main()
