"""Dataset-only overlays for the unchanged, already-running QA latency worker."""
from __future__ import annotations

import contextlib
from pathlib import Path

from scripts import latency_runtime as runtime
from scripts import run_latency_experiment as runner
from scripts.run_latency_experiment_queue import jobs as legacy_jobs
from scripts.run_qa32_rebuttal import DATASETS, ASSET_SPEC, file_hash, read_json, select_legacy_samples

ADDITIONAL_DATASETS = ("2wiki", "musique")
SOURCE_FILES = (
    "scripts/latency_datasets.py", "scripts/run_multidataset_latency.py",
    "scripts/run_multidataset_latency_queue.py", "scripts/run_qa32_rebuttal.py",
    "scripts/run_latency_experiment_queue.py",
    "run_all_2wiki_experiments_v2.py", "run_all_musique_experiments_v2.py",
)


def experiment_hashes():
    return {**runtime.experiment_hashes(),
            **{name: file_hash(runtime.ROOT / name) for name in SOURCE_FILES}}


def normalized_assets(data_ready, dataset):
    if dataset not in ADDITIONAL_DATASETS:
        raise ValueError(f"Unsupported additional latency dataset: {dataset}")
    directory = Path(data_ready).resolve().parent / "datasets" / dataset
    ready_path = directory / "ready.json"
    metadata = read_json(ready_path)
    pinned = read_json(ASSET_SPEC)["additional_datasets"][dataset]
    for key in ("dataset", "repository", "revision", "split", "rows", "sha256",
                "raw_sha256", "normalizer", "normalizer_sha256"):
        if metadata.get(key) != pinned[key]:
            raise RuntimeError(f"{dataset}: pinned normalized data metadata mismatch ({key})")
    path = Path(metadata["path"]).resolve()
    if path != (directory / "normalized_dev.json").resolve():
        raise RuntimeError("Normalized dataset path escapes its selected asset directory")
    if file_hash(path) != pinned["sha256"]:
        raise RuntimeError(f"{dataset}: normalized dataset checksum mismatch")
    normalizer_path = runtime.ROOT / (metadata["normalizer"].split(".", 1)[0] + ".py")
    if file_hash(normalizer_path) != pinned["normalizer_sha256"]:
        raise RuntimeError(f"{dataset}: original normalizer source checksum mismatch")
    rows = read_json(path)
    if not isinstance(rows, list) or len(rows) != DATASETS[dataset]:
        raise RuntimeError(f"{dataset}: unexpected normalized dev row count")
    provenance = {"evaluation_dataset": metadata, "normalized_payload_sha256": pinned["sha256"],
                  "normalized_ready_sha256": file_hash(ready_path),
                  "dataset_config_policy": "Frozen common HotpotQA ReAct defaults, as in QA32; "
                                           "only pinned normalized dev rows change."}
    return path, rows, provenance


def select_samples(base, rows, count, dataset):
    if not 1 <= count <= 500:
        raise ValueError("Use 1..500 main questions; positions 501-502 are reserved for warmup")
    ordered = select_legacy_samples(base, rows, samples=502, sample_start=0,
                                    expected_rows=DATASETS[dataset])
    selected, warmups = ordered[:count], ordered[500:502]
    if len(selected) != count or {row["id"] for _, row in selected} & {row["id"] for _, row in warmups}:
        raise RuntimeError("Main and warmup sample selection is incomplete or overlaps")
    return selected, warmups


@contextlib.contextmanager
def dataset_scope(dataset):
    import datasets

    original_assets = runner.load_assets
    original_loader = datasets.load_dataset
    loaded = {}

    def assets(model_ready, data_ready):
        model, index, _, provenance = original_assets(model_ready, data_ready)
        path, rows, dataset_provenance = normalized_assets(data_ready, dataset)
        loaded.update(path=path, rows=rows)
        provenance = {**provenance, "parquet_sha256": {}, **dataset_provenance}
        return model, index, [path], provenance

    def load(name, *args, **kwargs):
        if name != "parquet":
            return original_loader(name, *args, **kwargs)
        if args or kwargs.get("data_files") != {"validation": [str(loaded["path"])]}:
            raise RuntimeError("Unexpected normalized dataset loader call")
        if kwargs.get("split") != "validation":
            raise RuntimeError("Unexpected internal dataset split selector")
        return loaded["rows"]

    with runtime.patch_attribute(runner, "load_assets", assets), \
            runtime.patch_attribute(runner, "select_samples",
                                    lambda base, rows, count: select_samples(base, rows, count, dataset)), \
            runtime.patch_attribute(runner, "experiment_hashes", experiment_hashes), \
            runtime.patch_attribute(datasets, "load_dataset", load):
        yield
