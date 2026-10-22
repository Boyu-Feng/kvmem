import json
from types import SimpleNamespace

import pytest
import torch

from scripts import run_thinkkv_latency as worker
from test_thinkkv_selection import runtime


def test_bf16_clean_and_diagnostic_episode_parity_and_every_step_export(monkeypatch, tmp_path):
    llm = runtime(monkeypatch)
    llm.model.to(torch.bfloat16)
    llm.tokenizer.apply_chat_template = lambda *args, **kwargs: "XXXX"
    monkeypatch.setattr(llm.tokenizer.__class__, "__call__", lambda self, text, **kwargs: SimpleNamespace(
        input_ids=[2 for _ in text]))
    forward = llm.model.forward

    def force_one_token(*args, **kwargs):
        result = forward(*args, **kwargs)
        result.logits[..., :] = -1000
        result.logits[..., 1 if kwargs["input_ids"].shape[-1] == 1 else 2] = 1000
        return result
    monkeypatch.setattr(llm.model, "forward", force_one_token)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *args: None)
    monkeypatch.setattr(worker, "require_exclusive", lambda: {})
    monkeypatch.setattr(worker, "model_storage_bytes", lambda model: {"cuda:0": 100})
    allocator = {f"{key}_bytes": 1000 for key in ("allocated", "reserved", "peak_allocated", "peak_reserved")}
    allocator.update({f"{key}_above_model_bytes": 900 for key in (
        "allocated", "reserved", "peak_allocated", "peak_reserved", "driver")})
    allocator["model_storage_bytes"] = 100
    monkeypatch.setattr(worker, "cuda_allocator", lambda size: dict(allocator))

    class CPUMonitor(worker.SelectionMonitor):
        def _allocator(self, **kwargs):
            self._current_model_storage = self._model_storage()
            return {"cuda:0": dict(allocator)}
    monkeypatch.setattr(worker, "SelectionMonitor", CPUMonitor)
    base = SimpleNamespace(
        REACT_KV_INITIAL_PROMPT="{examples}{question}", REACT_EXAMPLES="",
        parse_action_original=lambda text, step: (
            "", "finish" if step == 3 else "search", "answer" if step == 3 else "page"),
        execute_action=lambda *args: ("X" * 260, args[-1]),
        exact_match=lambda prediction, gold: float(prediction == gold),
        f1_score=lambda prediction, gold: float(prediction == gold),
    )
    sample = {"id": "tiny", "question": "question", "answer": "answer"}
    clean = [worker.pass_once(base, llm, None, sample, False) for _ in range(2)]
    diagnostic = worker.pass_once(base, llm, None, sample, True)
    assert len({r["semantic_sha256"] for r in [*clean, diagnostic]}) == 1
    assert all(r["audit"]["kv_dtype"] == "torch.bfloat16" for r in [*clean, diagnostic])
    assert diagnostic["diagnostics"]["timing"]["exclusive_seconds"]["token_clustering"] > 0
    assert diagnostic["diagnostics"]["timing"]["exclusive_seconds"]["decode_forward"] > 0
    assert diagnostic["diagnostics"]["workload"]["decode_forward"]["input_positions"] == 3
    assert len(diagnostic["interaction_boundaries"]) == 3
    assert diagnostic["audit"]["native_kv_heads"] == 2
    row = {"id": "tiny", "clean_repeats": clean, "diagnostic": diagnostic,
           "all_passes_semantically_equal": True}
    protocol = {"samples": 1, "sample_ids": ["tiny"], "dtype": "bfloat16", "quantization": False}
    saved = {"protocol_sha256": worker.digest(protocol), "rows_sha256": worker.digest([row]),
             "rows": [row], "summary": worker.summary([row])}
    (tmp_path / "manifest.json").write_text(json.dumps({"protocol": protocol}))
    (tmp_path / "measurements.json").write_text(json.dumps(saved))
    (tmp_path / "status.json").write_text(json.dumps({"status": "complete"}))
    assert worker.validate_run(tmp_path) == (protocol, [row])
    worker.export_points(tmp_path, [row])
    assert len((tmp_path / "memory_points.csv").read_text().splitlines()) == 8
    assert "exclusive_seconds.token_clustering" in (tmp_path / "diagnostic_phases.csv").read_text()
    saved["rows"][0]["diagnostic"]["semantic_sha256"] = "corrupt"
    saved["rows_sha256"] = worker.digest(saved["rows"])
    (tmp_path / "measurements.json").write_text(json.dumps(saved))
    with pytest.raises(RuntimeError, match="semantic hashes"):
        worker.validate_run(tmp_path)


def test_worker_plan_cannot_load_model_without_explicit_execute(monkeypatch, capsys):
    monkeypatch.setattr(worker, "execute", lambda args: pytest.fail("Plan loaded a model"))
    monkeypatch.setattr("sys.argv", [
        "run_thinkkv_latency.py", "--method", "thinkkv_selection_only", "--dataset", "musique",
        "--model-ready", "model.json", "--data-ready", "data.json", "--output", "output"])
    worker.main()
    assert "BF16 KV" in capsys.readouterr().out
