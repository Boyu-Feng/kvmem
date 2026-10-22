"""Prepare a public, immutable CPU embedding encoder without changing dependencies."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_component_ablation_rebuttal import file_hash, read_json
from run_rebuttal_legacy import atomic_json

REPO = "BAAI/bge-small-en-v1.5"


def prepare(output):
    from huggingface_hub import HfApi, snapshot_download
    output = output.resolve()
    ready = output / "encoder_ready.json"
    if ready.exists():
        value = read_json(ready)
        for name, expected in value["config_hashes"].items():
            if file_hash(Path(value["path"]) / name) != expected:
                raise RuntimeError("Existing embedding configuration changed")
        for name, expected in value["weight_inventory"].items():
            if file_hash(Path(value["path"]) / name) != expected["sha256"]:
                raise RuntimeError("Existing embedding weights changed")
        return value
    info = HfApi(token=False).model_info(REPO)
    filenames = {item.rfilename for item in info.siblings}
    if "model.safetensors" not in filenames:
        raise RuntimeError("Public encoder lacks expected safe-format weights")
    model = output / "bge-small-en-v1.5"
    files = ["config.json", "model.safetensors", "tokenizer.json",
             "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"]
    snapshot_download(
        REPO, revision=info.sha, local_dir=model, token=False,
        allow_patterns=[*files, "README.md", "LICENSE"],
    )
    config = read_json(model / "config.json")
    if config["model_type"] != "bert" or config["hidden_size"] != 384:
        raise RuntimeError("Unexpected embedding encoder topology")
    value = {
        "version": 1, "repo": REPO, "revision": info.sha, "path": str(model),
        "pooling": "cls", "normalize": True, "max_tokens": 512,
        "dtype": "float32", "device": "cpu",
        "config_hashes": {name: file_hash(model / name) for name in files
                          if name != "model.safetensors"},
        "weight_inventory": {"model.safetensors": {
            "bytes": (model / "model.safetensors").stat().st_size,
            "sha256": file_hash(model / "model.safetensors"),
        }},
        "preparation_source_sha256": file_hash(Path(__file__)),
        "selection": "Frozen public encoder, not trained or selected on pilot F1",
    }
    atomic_json(ready, value)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = prepare(args.output)
    print({key: value[key] for key in ("repo", "revision", "path", "device", "pooling")})


if __name__ == "__main__":
    main()
