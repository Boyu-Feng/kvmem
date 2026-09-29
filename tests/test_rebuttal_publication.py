import hashlib
import json

import pytest

from scripts import export_rebuttal_snapshot as snapshot
from scripts import prepare_rebuttal_sources as sources


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_export_rejects_duplicate_unknown_and_invalid_scores():
    row = dict(id="a", em=1, f1=.5)
    for rows in ([row, row], [dict(row, id="b")], [dict(row, f1=50)]):
        with pytest.raises(ValueError):
            snapshot.normalize_rows(rows, ["a"])


def test_export_keeps_partial_results_out_of_final_table(tmp_path, monkeypatch):
    monkeypatch.setattr(snapshot, "QUEUES", ("queue",))
    source, output = tmp_path / "source", tmp_path / "report"
    jobs = [dict(name=name, dataset="test", method=name, ratio=.2, purpose="main", samples=2)
            for name in ("finished", "unfinished")]
    put(source / "queue/queue_plan.json", dict(jobs=jobs))
    for job in jobs:
        path = source / "queue" / job["name"]
        rows = [dict(id="a", em=1, f1=.75)]
        if job["name"] == "finished":
            rows.append(dict(id="b", em=0, f1=.25))
            put(path / "results.json", dict(results=rows))
        put(path / "checkpoint.json", rows)
        put(path / "manifest.json", dict(fingerprint=dict(sample_ids=["a", "b"])))
        put(path / "status.json", dict(status="complete" if len(rows) == 2 else "running"))
    before = {p: p.read_bytes() for p in source.rglob("*.json")}
    result = snapshot.export(source, output)
    final, partial = result["main_conditions"]
    assert (result["completed_main"], result["saved_main"]) == (1, 3)
    assert (final["em_percent"], final["f1_percent"], final["score_status"]) == (50, 50, "final")
    assert partial["score_status"] == "partial_not_ranked"
    final_table = (output / "README.md").read_text().split("## 已完成的固定预算实验")[1].split("## SideQuest")[0]
    assert "finished" in final_table and "unfinished" not in final_table
    bindings = json.loads((output / "source_bindings.json").read_text())
    for path, raw in before.items():
        assert path.read_bytes() == raw
        assert bindings[str(path.relative_to(source))]["sha256"] == hashlib.sha256(raw).hexdigest()
    with pytest.raises(FileExistsError):
        snapshot.export(source, output)
    put(source / "queue/unfinished/status.json", dict(status="complete"))
    with pytest.raises(ValueError, match="incomplete checkpoint"):
        snapshot.export(source, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


def test_frozen_source_tampering_rejected_before_materialization(tmp_path, monkeypatch):
    root, bundle = tmp_path / "root", tmp_path / "bundle"
    root.mkdir()
    (root / "core.py").write_text("original\n")
    digest = hashlib.sha256((root / "core.py").read_bytes()).hexdigest()
    put(bundle / "manifest.json", dict(methods=dict(example=dict(
        historical_directory_name="example_source",
        source_files={"core.py": {"from": "repository", "sha256": digest}}, extra_files={}))))
    monkeypatch.setattr(sources, "ROOT", root)
    monkeypatch.setattr(sources, "BUNDLES", bundle)
    output = tmp_path / "materialized"
    sources.prepare(output, ["example"])
    assert (output / "example_source/core.py").read_bytes() == (root / "core.py").read_bytes()
    with pytest.raises(FileExistsError):
        sources.prepare(output, ["example"])
    (root / "core.py").write_text("changed\n")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        sources.prepare(tmp_path / "invalid", ["example"])
    assert not (tmp_path / "invalid").exists()
