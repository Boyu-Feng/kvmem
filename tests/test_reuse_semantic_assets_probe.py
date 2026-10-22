import json
from types import SimpleNamespace
import sys

import pytest

from scripts import reuse_semantic_assets as assets
from scripts import reuse_semantic_probe as probe


def test_public_encoder_assets_pin_revision_weights_and_existing_config(tmp_path, monkeypatch):
    def download(repo, *, revision, local_dir, token, allow_patterns):
        assert token is False and revision == "frozen"
        local_dir.mkdir(parents=True)
        for name in allow_patterns:
            if name == "config.json":
                (local_dir / name).write_text(json.dumps({"model_type": "bert", "hidden_size": 384}))
            elif name == "model.safetensors":
                (local_dir / name).write_bytes(b"fake-safe-format-for-hash-test")
            else:
                (local_dir / name).write_text("{}")

    class Api:
        def __init__(self, *, token):
            assert token is False

        def model_info(self, repo):
            return SimpleNamespace(sha="frozen", siblings=[SimpleNamespace(rfilename="model.safetensors")])

    monkeypatch.setitem(sys.modules, "huggingface_hub",
                        SimpleNamespace(HfApi=Api, snapshot_download=download))
    ready = assets.prepare(tmp_path)
    assert ready["device"] == "cpu" and ready["pooling"] == "cls"
    assert assets.prepare(tmp_path) == ready
    (tmp_path / "bge-small-en-v1.5/model.safetensors").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="weights"):
        assets.prepare(tmp_path)


def test_lexical_probe_matches_original_overlap_normalization():
    result = probe.lexical_pair("alpha beta gamma", "alpha delta")
    assert result["overlap"] == 1 and result["overlap_ratio"] == .5
    assert result["citation_increment"] == .75
    assert probe.lexical_pair("", "")["citation_increment"] == 0


def test_probe_excludes_self_matches_and_never_copies_gold_into_scorer_pairs(tmp_path):
    rows = [{
        "id": "question", "gold_answer": "SECRET_GOLD",
        "trajectory": [
            {"step": 1, "observation": "A physician opened a clinic.", "thought": "Find the clinic."},
            {"step": 2, "observation": "A doctor founded the practice.", "thought": "Continue."},
            {"step": 3, "action_type": "finish", "action_arg": "SECRET_GOLD"},
        ],
    }]
    fingerprint = {"protocol": {"dataset": "hotpotqa", "seed": 233, "ratio": .2}}
    checksum = probe.digest(fingerprint)
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"fingerprint": fingerprint, "fingerprint_sha256": checksum}))
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({"rows": rows, "rows_sha256": probe.digest(rows),
                                     "fingerprint_sha256": checksum}))
    result = probe.prepare_pairs(checkpoint, samples=1)
    actual = [p for p in result["pairs"] if p["source"] == "frozen_default_trajectory"]
    assert len(actual) == 1 and actual[0]["historical_step"] < actual[0]["reference_step"]
    assert "SECRET_GOLD" not in json.dumps(result)
    assert actual[0]["label"] == "unannotated"
