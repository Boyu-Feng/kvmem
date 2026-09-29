"""Materialize hash-verified frozen baseline sources without changing live runs."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
BUNDLES = ROOT / "rebuttal/adapters"


def checked_bytes(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise RuntimeError(f"Frozen source hash mismatch: {path}")
    return data


def prepare(destination, methods, snapshot_path=None):
    manifest = json.loads((BUNDLES / "manifest.json").read_text())
    # Validate the complete requested input before creating any output.
    for method in methods:
        spec = manifest["methods"][method]
        for rel, info in spec["source_files"].items():
            parent = ROOT if info["from"] == "repository" else BUNDLES / method
            checked_bytes(parent / rel, info["sha256"])
        for rel, digest in spec["extra_files"].items():
            checked_bytes(BUNDLES / method / rel, digest)
        target = destination / spec["historical_directory_name"]
        if target.exists():
            raise FileExistsError(f"Use a fresh destination; refusing to overwrite {target}")
    outputs = []
    for method in methods:
        spec = manifest["methods"][method]
        target = destination / spec["historical_directory_name"]
        for rel, info in spec["source_files"].items():
            parent = ROOT if info["from"] == "repository" else BUNDLES / method
            out = target / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(parent / rel, out)
        for rel in spec["extra_files"]:
            out = target / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(BUNDLES / method / rel, out)
        outputs.append(str(target))
    if snapshot_path is not None:
        core = manifest["methods"][methods[0]]["original_core_hashes"]
        if snapshot_path.exists():
            with tarfile.open(snapshot_path) as archive:
                for rel, digest in core.items():
                    data = archive.extractfile(rel).read()
                    if hashlib.sha256(data).hexdigest() != digest:
                        raise RuntimeError(f"Existing reference archive differs: {rel}")
        else:
            payloads = {rel: checked_bytes(ROOT / rel, digest) for rel, digest in core.items()}
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            with tarfile.open(snapshot_path, "x") as archive:
                for rel, data in payloads.items():
                    info = tarfile.TarInfo(rel)
                    info.size = len(data)
                    info.mode = 0o644
                    archive.addfile(info, io.BytesIO(data))
            # This recreates only the verified reference source members. It is
            # not the historical archive and does not manufacture run results.
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=["flowkv_style", "lazyeviction", "h2o_step", "tova_step"],
                        default=["flowkv_style", "lazyeviction", "h2o_step", "tova_step"])
    parser.add_argument("--prepare-reference", action="store_true",
                        help="Validate/create the nine-core-file archive required by legacy runners")
    args = parser.parse_args()
    snapshot = ROOT / "results/rebuttal_20260927/source_snapshot/before_changes.tar" if args.prepare_reference else None
    print(json.dumps(prepare(args.destination.resolve(), args.methods, snapshot), indent=2))


if __name__ == "__main__":
    main()
