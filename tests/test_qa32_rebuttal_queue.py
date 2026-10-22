import json

import pytest

from scripts import run_qa32_rebuttal_queue as queue


def test_three_datasets_have_both_pruning_schedules_without_duplicate_fullkv():
    plan = queue.jobs(["hotpotqa", "2wiki", "musique"])
    assert len(plan) == 66
    main = [job for job in plan if job["purpose"] == "main"]
    assert len(main) == 33
    assert sum(job["samples"] for job in main) == 16500
    for dataset in ("hotpotqa", "2wiki", "musique"):
        group = [job for job in plan if job["dataset"] == dataset]
        assert all(job["purpose"] == "smoke" and job["sample_start"] == 500 for job in group[:11])
        assert all(job["purpose"] == "main" and job["sample_start"] == 0 for job in group[11:])
        assert sum(job["method"] == "fullkv" for job in group[11:]) == 1
        assert {"h2o", "h2o_step", "tova", "tova_step"} <= {job["method"] for job in group[11:]}


def test_smoke_requires_actual_eviction_but_not_a_positive_accuracy(tmp_path, monkeypatch):
    job = queue.jobs(["hotpotqa"])[1]
    (tmp_path / "manifest.json").write_text(json.dumps({"fingerprint": job}))
    result = {"completed_samples": 2, "sample_ids": ["a", "b"],
              "has_actual_eviction": True, "summary": {"exact_match": 0, "f1_score": 0}}
    monkeypatch.setattr(queue, "validate_completed_run", lambda _: result)
    assert queue.validate_job(tmp_path, job) is result
    result["has_actual_eviction"] = False
    with pytest.raises(RuntimeError, match="eviction"):
        queue.validate_job(tmp_path, job)


def test_job_validation_rejects_wrong_dataset_or_sample_count(tmp_path, monkeypatch):
    job = queue.jobs(["hotpotqa"])[11]
    (tmp_path / "manifest.json").write_text(json.dumps({"fingerprint": {**job, "dataset": "2wiki"}}))
    result = {"completed_samples": 500, "sample_ids": [], "has_actual_eviction": False, "summary": {}}
    monkeypatch.setattr(queue, "validate_completed_run", lambda _: result)
    with pytest.raises(RuntimeError, match="dataset"):
        queue.validate_job(tmp_path, job)
    (tmp_path / "manifest.json").write_text(json.dumps({"fingerprint": job}))
    result["completed_samples"] = 499
    with pytest.raises(RuntimeError, match="sample count"):
        queue.validate_job(tmp_path, job)
