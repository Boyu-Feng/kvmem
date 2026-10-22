from argparse import Namespace
import copy
import importlib
from types import SimpleNamespace

import pytest

from scripts import run_component_ablation_rebuttal as worker


@pytest.fixture
def base():
    return importlib.import_module("run_all_wiki_experiments_v2")


@pytest.fixture
def rows():
    return [{"id": f"q{i}", "question": f"question {i}", "answer": "answer"} for i in range(800)]


@pytest.mark.parametrize("seed", [233, 3407])
def test_original_seed_selection_and_disjoint_smoke(base, rows, seed):
    previous = base.RANDOM_SEED, base.NUM_SAMPLES
    main = worker.select_samples(base, rows, seed=seed, purpose="main", expected_rows=800)
    smoke = worker.select_samples(base, rows, seed=seed, purpose="smoke", expected_rows=800)
    assert len(main) == 500 and len(smoke) == 2
    assert not {row["id"] for _, row in main} & {row["id"] for _, row in smoke}
    assert (base.RANDOM_SEED, base.NUM_SAMPLES) == previous
    with worker.patch_attribute(base, "RANDOM_SEED", seed), worker.patch_attribute(base, "NUM_SAMPLES", 500):
        assert main == base.select_samples(rows)


def test_requested_seeds_change_sample_selection(base, rows):
    first = worker.select_samples(base, rows, seed=233, purpose="main")
    second = worker.select_samples(base, rows, seed=3407, purpose="main")
    assert first != second


@pytest.mark.parametrize("variant", worker.VARIANTS)
def test_original_component_configuration_preserved(base, variant):
    original = importlib.import_module("run_stepkv_component_ablation")
    expected = {**original._base_kv_override(.5), **original.ABLATION_SPECS[variant]["override"],
                "step_ablation_name": variant}
    assert worker.variant_override(variant) == expected
    before = base._finalize_react_kv_output
    config = worker.component_config(base, variant)
    assert config["pruning_mode"] == "step_aware_h2o" and config["cache_ratio"] == .5
    assert all(config[key] == value for key, value in expected.items())
    assert base._finalize_react_kv_output is before


def test_full_is_stepkv_not_fullkv(base):
    config = worker.component_config(base, "full")
    assert config["step_aware_alpha"] == .8 and config["step_aware_beta"] == .8
    assert config["pruning_mode"] != "none"


def test_fingerprint_protocol_rejects_unrequested_seed():
    with pytest.raises(ValueError, match="Unsupported"):
        worker.protocol(Namespace(dataset="hotpotqa", seed=42, variant="full", purpose="main"))


def test_selection_rejects_duplicate_ids(base, rows):
    rows[1]["id"] = rows[0]["id"]
    with pytest.raises(RuntimeError, match="Duplicate"):
        worker.select_samples(base, rows, seed=233, purpose="main")


def test_scope_restores_after_error():
    obj = SimpleNamespace(value=1)
    with pytest.raises(RuntimeError):
        with worker.patch_attribute(obj, "value", 2):
            raise RuntimeError("failure")
    assert obj.value == 1
    with worker.patch_attribute(obj, "temporary", 3):
        assert obj.temporary == 3
    assert not hasattr(obj, "temporary")


@pytest.fixture
def fingerprint(base, rows):
    selected = worker.select_samples(base, rows, seed=233, purpose="smoke")
    # Synthetic source rows use the same index range as the pinned HotpotQA split.
    content = [[index, sample] for index, sample in selected]
    return {
        "protocol": worker.protocol(Namespace(dataset="hotpotqa", seed=233, variant="full", purpose="smoke")),
        "source_hashes": worker.source_hashes(), "kv_config": worker.component_config(base, "full"),
        "sample_ids": [row["id"] for _, row in selected],
        "sample_indices": [index for index, _ in selected],
        "selected_samples": content, "selected_content_sha256": worker.digest(content),
    }


def result_row(fingerprint, position=0):
    index, sample = fingerprint["selected_samples"][position]
    return {
        "status": "success", "id": sample["id"], "index": index,
        "question": sample["question"], "gold_answer": sample["answer"],
        "predicted_answer": sample["answer"], "em": 1, "f1": 1.,
        "diagnostic_elapsed_seconds": .1, "num_steps": 2,
        "pruning_audit": [{"before": [10, 10], "after": [5, 5]}],
        "has_actual_eviction": True,
    }


def test_durable_resume_and_completed_validation(tmp_path, fingerprint):
    store = worker.ResultStore(tmp_path, fingerprint)
    store.commit(result_row(fingerprint, 0))
    resumed = worker.ResultStore(tmp_path, fingerprint)
    assert len(resumed.rows) == 1
    resumed.commit(result_row(fingerprint, 1))
    resumed.finalize()
    record = worker.validate_completed_run(tmp_path)
    assert record["completed_samples"] == 2
    assert record["sample_ids"] == fingerprint["sample_ids"]
    assert record["has_actual_eviction"]
    assert record["summary"]["exact_match"] == 100


def test_corrupt_checkpoint_is_refused(tmp_path, fingerprint):
    from scripts.run_qa32_rebuttal import atomic_json
    store = worker.ResultStore(tmp_path, fingerprint)
    store.commit(result_row(fingerprint))
    value = worker.read_json(tmp_path / "checkpoint.json")
    value["rows"][0]["predicted_answer"] = "changed"
    atomic_json(tmp_path / "checkpoint.json", value)
    with pytest.raises(RuntimeError, match="Corrupt"):
        worker.ResultStore(tmp_path, fingerprint)


def test_changed_source_is_refused(tmp_path, fingerprint):
    changed = copy.deepcopy(fingerprint)
    changed["source_hashes"]["scripts/run_component_ablation_rebuttal.py"] = "different"
    with pytest.raises(RuntimeError, match="Source/protocol"):
        worker.ResultStore(tmp_path, changed)


def test_suppressed_generation_error_is_not_scored(monkeypatch):
    class BrokenBase:
        @staticmethod
        def _run_react_kv_episode(*args, **kwargs):
            print("[ERROR] failed generation")
            return "", [], [], {}
    import contextlib
    monkeypatch.setattr(worker, "pruning_observer", lambda llm: contextlib.nullcontext(([], [])))
    with pytest.raises(RuntimeError, match="suppressed errors"):
        worker.run_episode(BrokenBase, {"question": "q", "answer": "a"}, None, None)


def test_actual_tensor_eviction_is_observed_and_wrapper_restored():
    import torch
    cache = ((torch.zeros(1, 4, 10, 2), torch.zeros(1, 4, 10, 2)),)
    def original(past_key_values, **kwargs):
        return tuple((k[:, :, :5], v[:, :, :5]) for k, v in past_key_values), None
    strategy = SimpleNamespace(prune=original)
    llm = SimpleNamespace(kv_manager=SimpleNamespace(pruning_strategy=strategy))
    with worker.pruning_observer(llm) as (events, errors):
        result = strategy.prune(cache)
        assert result[0][0][0].shape[-2] == 5
    assert events == [{"before": [10], "after": [5]}] and errors == []
    assert strategy.prune is original


def test_two_model_revisions_and_topologies_are_distinct():
    qwen = worker.MODELS["Qwen/Qwen2.5-7B-Instruct"]
    llama = worker.MODELS["meta-llama/Meta-Llama-3.1-8B-Instruct"]
    assert qwen["config"]["num_hidden_layers"] == 28
    assert llama["config"]["num_hidden_layers"] == 32
    assert llama["config"]["num_key_value_heads"] == 8
    assert qwen["revision"] != llama["revision"]
