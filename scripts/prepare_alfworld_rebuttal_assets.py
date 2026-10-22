"""Prepare pinned Qwen2.5 weights and the official text-only ALFWorld assets."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from urllib.request import Request, urlopen
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_rebuttal_legacy import atomic_json

MODELS = {
    "7b": ("Qwen/Qwen2.5-7B-Instruct", "a09a35458c702b33eeacc393d103063234e8bc28"),
    "32b": ("Qwen/Qwen2.5-32B-Instruct", "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd"),
}
ARCHIVES = (
    "https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip",
    "https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_pddl.zip",
    "https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(url: str, target: Path) -> None:
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "kvmem-rebuttal-assets"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with urlopen(Request(url, headers=headers), timeout=120) as response:
        append = response.status == 206
        if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise RuntimeError(f"Unexpected partial response for {url}")
        if response.status not in (200, 206):
            raise RuntimeError(f"Download returned HTTP {response.status}: {url}")
        expected = response.headers.get("Content-Length")
        received = 0
        with partial.open("ab" if append else "wb") as handle:
            for block in iter(lambda: response.read(8 * 1024 * 1024), b""):
                handle.write(block)
                received += len(block)
        if expected is not None and received != int(expected):
            raise RuntimeError(f"Incomplete download: {url}: {received}/{expected}")
    partial.replace(target)


def extract_evaluation_files(archive: Path, destination: Path) -> int:
    destination = destination.resolve()
    count = 0
    with zipfile.ZipFile(archive) as zipped:
        for member in zipped.infolist():
            relative = Path(member.filename)
            target = (destination / relative).resolve()
            if relative.is_absolute() or not target.is_relative_to(destination):
                raise ValueError(f"Archive member escapes destination: {member.filename}")
            if (member.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Archive contains a symbolic link: {member.filename}")
            if not {"valid_seen", "valid_unseen"}.intersection(relative.parts) or member.is_dir():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
            count += 1
    if not count:
        raise RuntimeError(f"No official evaluation files found in {archive}")
    return count


def prepare_model(root: Path, size: str) -> dict:
    from huggingface_hub import snapshot_download

    repo, revision = MODELS[size]
    directory = root / "models" / repo.split("/")[1]
    snapshot_download(
        repo, revision=revision, local_dir=directory,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "LICENSE"],
        max_workers=4,
    )
    config = json.loads((directory / "config.json").read_text())
    if config.get("model_type") != "qwen2":
        raise RuntimeError(f"Expected ordinary Qwen2 attention, found {config.get('model_type')}")
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    for shard in shards:
        if not (directory / shard).is_file():
            raise RuntimeError(f"Missing model shard: {shard}")
    return {
        "repo": repo, "revision": revision, "path": str(directory.resolve()),
        "model_type": config["model_type"], "weight_bytes": index["metadata"]["total_size"],
        "shards": shards,
        "config_hashes": {p.name: sha256(p) for p in sorted(directory.glob("*.json"))},
        "purpose": "Shared-GPU accuracy only; not a latency measurement",
    }


def prepare_alfworld(root: Path) -> dict:
    from importlib.metadata import version
    from alfworld.info import ALFRED_PDDL_PATH, ALFRED_TWL2_PATH

    directory = root / "alfworld"
    archive_directory = root / "downloads"
    archives = []
    for url in ARCHIVES:
        target = archive_directory / url.rsplit("/", 1)[1]
        print(f"Downloading {url}", flush=True)
        download(url, target)
        print(f"Extracting evaluation splits from {target.name}", flush=True)
        count = extract_evaluation_files(target, directory)
        archives.append({
            "url": url, "sha256": sha256(target), "bytes": target.stat().st_size,
            "extracted_evaluation_files": count,
        })
    logic = directory / "logic"
    logic.mkdir(parents=True, exist_ok=True)
    for source in (Path(ALFRED_PDDL_PATH), Path(ALFRED_TWL2_PATH)):
        shutil.copy2(source, logic / source.name)
    split_counts = {}
    for split in ("valid_seen", "valid_unseen"):
        games = sorted(directory.glob(f"**/{split}/**/game.tw-pddl"))
        if not games:
            raise RuntimeError(f"ALFWorld text games missing in {split}")
        split_counts[split] = len(games)
    return {
        "path": str(directory.resolve()), "package": "alfworld", "version": version("alfworld"),
        "archives": archives, "split_counts": split_counts,
        "logic_hashes": {p.name: sha256(p) for p in sorted(logic.iterdir()) if p.is_file()},
        "scope": "Official text environments; evaluation splits only; no detector, images or training checkpoints",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["7b", "32b", "alfworld"])
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(root / "hf_cache"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
    os.environ.setdefault("ALFWORLD_DATA", str(root / "alfworld"))
    with (root / f"{args.kind}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status_path = root / f"{args.kind}_status.json"
        started = time.time()
        atomic_json(status_path, {"status": "preparing", "pid": os.getpid(), "started": started})
        try:
            record = prepare_alfworld(root) if args.kind == "alfworld" else prepare_model(root, args.kind)
            record.update(created=time.time(), preparation_source_sha256=sha256(Path(__file__)))
            atomic_json(root / f"{args.kind}_ready.json", record)
            atomic_json(status_path, {"status": "complete", "pid": os.getpid(), "updated": time.time()})
            print(json.dumps(record, indent=2), flush=True)
        except Exception:
            atomic_json(status_path, {
                "status": "failed", "pid": os.getpid(), "updated": time.time(),
                "traceback": traceback.format_exc(),
            })
            raise


if __name__ == "__main__":
    main()
