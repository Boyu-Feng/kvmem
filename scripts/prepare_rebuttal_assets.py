"""Download pinned, public experiment assets and build the original full-Wiki index."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["model", "data"])
    parser.add_argument("--root", default="/data/experiment/fengboyu/stepkv")
    args = parser.parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(root / "hf_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from huggingface_hub import HfApi, snapshot_download

    started = time.time()
    api = HfApi()
    if args.kind == "model":
        repo = "Qwen/Qwen2.5-7B-Instruct"
        revision = api.model_info(repo).sha
        target = root / "models" / "Qwen2.5-7B-Instruct"
        snapshot_download(repo, revision=revision, local_dir=target,
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
                          max_workers=2)
        manifest = {"repo": repo, "revision": revision, "path": str(target)}
    else:
        manifest = {}
        for repo, patterns, name in [
            ("hotpotqa/hotpot_qa", ["distractor/validation-*.parquet"], "hotpotqa"),
            ("TIGER-Lab/LongRAG", ["hotpot_qa_wiki/*.parquet"], "longrag"),
        ]:
            revision = api.dataset_info(repo).sha
            target = root / "datasets" / name
            snapshot_download(repo, repo_type="dataset", revision=revision,
                              local_dir=target, allow_patterns=patterns, max_workers=2)
            manifest[name] = {"repo": repo, "revision": revision, "path": str(target)}
        # Keep the original indexing/tokenization procedure, while reading the
        # pinned local parquet files rather than resolving a changing HF branch.
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        import build_wiki_index as builder
        from datasets import load_dataset
        files = sorted(str(p) for p in (root / "datasets/longrag/hotpot_qa_wiki").glob("*.parquet"))
        def load_local(*_args, **_kwargs):
            return load_dataset("parquet", data_files={"train": files}, split="train",
                                cache_dir=str(root / "datasets/arrow_cache"))
        builder.load_dataset = load_local
        builder.build_wiki_index(str(root / "wiki_index"))
        manifest["wiki_index"] = str(root / "wiki_index")
    manifest["elapsed_seconds"] = time.time() - started
    path = root / f"{args.kind}_ready.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    tmp.replace(path)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
