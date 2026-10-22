import contextlib
from pathlib import Path

import pytest

from scripts import run_qa32_thinkkv_latency_handoff as handoff


@pytest.mark.parametrize("blockers", [
    {"owner": {"pid": 1, "start_ticks": 1}},
    {"child": {"pid": 2, "start_ticks": 2}},
])
def test_live_owner_or_orphan_child_defers_mu_on_all_pairs(monkeypatch, blockers):
    jobs = [(Path("/hotpot"), {"dataset": "hotpotqa"}),
            (Path("/2wiki"), {"dataset": "2wiki"}),
            (Path("/mu"), {"dataset": "musique"})]
    monkeypatch.setattr(handoff, "maintenance_blockers", lambda path: blockers)
    monkeypatch.setattr(handoff, "read_maintenance", lambda path: {"output": "/thinkkv"})
    monkeypatch.setattr(handoff.scheduler, "ready_jobs", lambda *args: jobs)
    with contextlib.ExitStack() as scope:
        handoff.install_priority(scope, lambda: None)
        for pair in ((3, 4), (5, 6), (0, 7)):
            result = handoff.scheduler.ready_jobs(pair)
            assert [job["dataset"] for _, job in result] == ["hotpotqa", "2wiki"]
    assert handoff.scheduler.ready_jobs(None) == jobs


def test_mu_automatically_returns_after_lease_release(monkeypatch):
    jobs = [(Path("/mu"), {"dataset": "musique"})]
    blockers = {"owner": {"pid": 1, "start_ticks": 1}}
    monkeypatch.setattr(handoff, "maintenance_blockers", lambda path: dict(blockers))
    monkeypatch.setattr(handoff, "read_maintenance", lambda path: {"output": "/thinkkv"})
    monkeypatch.setattr(handoff.scheduler, "ready_jobs", lambda *args: jobs)
    with contextlib.ExitStack() as scope:
        handoff.install_priority(scope, lambda: None)
        assert handoff.scheduler.ready_jobs(None) == []
        blockers.clear()
        assert handoff.scheduler.ready_jobs(None) == jobs


def test_priority_is_observable_and_original_publish_arguments_preserved(monkeypatch):
    monkeypatch.setattr(handoff, "maintenance_blockers", lambda path: {"owner": {"pid": 1}})
    monkeypatch.setattr(handoff, "read_maintenance", lambda path: {"output": "/thinkkv"})
    calls = []
    monkeypatch.setattr(handoff.scheduler, "publish", lambda *args, **kwargs: calls.append((args, kwargs)))
    with contextlib.ExitStack() as scope:
        handoff.install_priority(scope, lambda: None)
        handoff.scheduler.publish("original", max_workers=8)
    assert calls[0][0] == ("original",)
    assert calls[0][1]["max_workers"] == 8
    assert calls[0][1]["latency_priority"]["held_datasets"] == ["musique"]


def test_overlay_rejects_source_drift_before_admitting_any_job(monkeypatch):
    def verify():
        raise RuntimeError("source drift")
    with contextlib.ExitStack() as scope:
        handoff.install_priority(scope, verify)
        with pytest.raises(RuntimeError, match="source drift"):
            handoff.scheduler.ready_jobs(None)


def test_original_cli_and_scheduler_identity_are_retained(monkeypatch, tmp_path):
    original = str(Path(handoff.scheduler.__file__).resolve())
    import sys
    monkeypatch.setattr(sys, "argv", [
        "run_qa32_thinkkv_latency_handoff.py", "--source-scheduler", original,
        "--output", str(tmp_path), "--pairs", "3,4", "5,6", "0,7", "--max-workers", "8"])
    received = []
    monkeypatch.setattr(handoff.scheduler, "main", lambda: received.append(list(sys.argv)))
    handoff.main()
    assert received == [[original, "--output", str(tmp_path),
                         "--pairs", "3,4", "5,6", "0,7", "--max-workers", "8"]]
    assert (tmp_path / "thinkkv_priority_overlay.json").exists()


def priority_fixture():
    roots = {Path("/" + dataset): {"jobs": [
        {"dataset": dataset, "purpose": "main", "name": dataset + "/main/one"},
        {"dataset": dataset, "purpose": "main", "name": dataset + "/main/two"}]}
        for dataset in ("hotpotqa", "2wiki", "musique")}
    jobs = [(root, job) for root, plan in roots.items() for job in plan["jobs"]]
    return roots, jobs


def test_all_freed_core_slots_focus_first_dataset_then_second_then_mu(monkeypatch):
    roots, jobs = priority_fixture()
    completed = set()
    monkeypatch.setattr(handoff.scheduler, "ready_jobs", lambda *args: jobs)
    monkeypatch.setattr(handoff, "maintenance_blockers", lambda path: {})
    monkeypatch.setattr(handoff, "read_maintenance", lambda path: None)
    order = ("hotpotqa", "2wiki", "musique")
    with contextlib.ExitStack() as scope:
        handoff.install_priority(scope, lambda: None, order)
        for dataset in order:
            selected = handoff.scheduler.ready_jobs(roots, completed)
            assert [job["dataset"] for _, job in selected] == [dataset, dataset]
            completed.update((str(root), job["name"]) for root, job in selected)
        assert handoff.scheduler.ready_jobs(roots, completed) == []


def test_one_completed_condition_cannot_release_dataset_priority():
    roots, _ = priority_fixture()
    order = ("hotpotqa", "2wiki", "musique")
    assert handoff.selected_dataset(roots, {("/hotpotqa", "hotpotqa/main/one")}, order) == "hotpotqa"


def test_mu_remains_deferred_if_both_priority_datasets_finish_before_latency(monkeypatch):
    roots, jobs = priority_fixture()
    completed = {(str(root), job["name"]) for root, job in jobs if job["dataset"] != "musique"}
    monkeypatch.setattr(handoff.scheduler, "ready_jobs", lambda *args: jobs)
    monkeypatch.setattr(handoff, "maintenance_blockers", lambda path: {"child": {"pid": 2}})
    monkeypatch.setattr(handoff, "read_maintenance", lambda path: {"output": "/thinkkv"})
    with contextlib.ExitStack() as scope:
        handoff.install_priority(scope, lambda: None, ("hotpotqa", "2wiki", "musique"))
        assert handoff.scheduler.ready_jobs(roots, completed) == []


def test_original_entrypoint_can_install_overlay_on_its_own_module(monkeypatch, tmp_path):
    import sys
    from scripts import run_parallel_qa32_queue as qa
    original = qa.ready_jobs
    received = []
    monkeypatch.setattr(qa, "execute", lambda args: received.append(
        (args, qa.ready_jobs is not original)))
    monkeypatch.setattr(sys, "argv", [
        str(Path(qa.__file__).resolve()), "--output", str(tmp_path),
        "--python", str(Path(sys.executable).absolute()),
        "--priority-overlay", str(Path(handoff.__file__).resolve()),
        "--dataset-order", "hotpotqa", "2wiki", "musique"])
    qa.main()
    assert len(received) == 1 and received[0][1] is True
    assert received[0][0].dataset_order == ["hotpotqa", "2wiki", "musique"]
    assert qa.ready_jobs is original


def test_free_gpu2_gpu7_pair_is_explicit_and_portable_manifest_is_recognized():
    from scripts import run_parallel_qa32_queue as qa
    assert qa.parser().parse_args(["--pairs", "3,4", "5,6", "2,7"]).pairs == ["3,4", "5,6", "2,7"]
    assert qa.parser().parse_args([]).pairs == ["3,4", "5,6", "0,7"]
    assert qa.manifest_pair({"hardware_attempts": [{"cuda_visible_devices": "2,7"}]}) == (2, 7)


def test_new_pair_cannot_overlap_existing_gpu7_pair(monkeypatch):
    import sys
    from scripts import run_parallel_qa32_queue as qa
    monkeypatch.setattr(sys, "argv", [str(qa.__file__), "--pairs", "0,7", "2,7"])
    with pytest.raises(ValueError, match="disjoint"):
        qa.main()
