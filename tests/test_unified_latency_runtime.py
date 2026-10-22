from types import SimpleNamespace
import threading

import pytest
import torch

from scripts.unified_latency_runtime import GenerationAudit, UnifiedMonitor
from scripts import run_unified_latency as worker


def fake_llm():
    model = SimpleNamespace(generate=lambda **kwargs: SimpleNamespace(
        sequences=torch.zeros((1, kwargs["input_ids"].shape[-1] + 3), dtype=torch.long)))
    return SimpleNamespace(model=model)


def test_main_generation_audit_preserves_hf_output_and_records_late_stop_work():
    llm = fake_llm()
    original = llm.model.generate
    with GenerationAudit(llm) as audit:
        result = llm.model.generate(input_ids=torch.ones((1, 365), dtype=torch.long),
                                    do_sample=False, use_cache=True, max_new_tokens=256)
        assert result.sequences.shape == (1, 368)
        assert audit.calls == [{"input_positions": 365, "max_new_tokens": 256,
                               "sequence_positions": 368, "returned_new_tokens": 3,
                               "stop_policy": "Original text/KV crop after HF generation returns"}]
    assert llm.model.generate == original


@pytest.mark.parametrize("extra", [
    {"stopping_criteria": [object()]}, {"stop_strings": ["\nObservation"]},
    {"do_sample": True}, {"max_new_tokens": 128}, {"use_cache": False},
])
def test_main_generation_rejects_noncanonical_settings(extra):
    llm = fake_llm()
    kwargs = dict(input_ids=torch.ones((1, 3), dtype=torch.long),
                  do_sample=False, use_cache=True, max_new_tokens=256)
    kwargs.update(extra)
    with GenerationAudit(llm), pytest.raises(RuntimeError):
        llm.model.generate(**kwargs)


def test_auxiliary_generation_is_labeled_algorithm_work_not_main_decoder_work():
    llm = fake_llm()
    with GenerationAudit(llm) as audit:
        thread = threading.Thread(target=lambda: llm.model.generate(
            input_ids=torch.ones((1, 1), dtype=torch.long), max_new_tokens=128))
        thread.start()
        thread.join()
        assert audit.calls == []


def test_extra_cache_inventory_supports_named_caches_and_deduplicates_storage():
    tensor = torch.ones(1, 2, 3, 4)
    cache = [(tensor, tensor)]
    monitor = object.__new__(UnifiedMonitor)
    monitor.llm = SimpleNamespace(live_caches=lambda: {"main": cache, "aux_alias": cache})
    from scripts.latency_metrics import DetailedLatencyMonitor
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(DetailedLatencyMonitor, "_sample", lambda *args, **kwargs: {})
        result = monitor._sample("test")
    assert result["all_live_kv"]["total_bytes"] == 4 * tensor.numel() * tensor.element_size()
    assert result["all_live_kv"]["storage_bytes"] == tensor.untyped_storage().nbytes()


def args(method="stepkv"):
    from pathlib import Path
    return SimpleNamespace(method=method, dataset="hotpotqa", samples=20, repeats=2, ratio=.2,
                           model_ready=Path("model"), data_ready=Path("data"))


def test_all_methods_have_identical_main_generation_policy_and_fixed_sample_protocol():
    plans = [worker.prepare_plan(args(method)) for method in worker.METHODS]
    for key in ("decode_policy", "generation_policy", "generation_overrides", "sample_positions",
                "warmup_positions", "max_steps", "max_new_tokens_per_block",
                "clean_repeats", "diagnostic_repeats", "dtype", "quantization"):
        assert all(plan[key] == plans[0][key] for plan in plans)
    side = next(p for p in plans if p["method"] == "sidequest_unified")
    assert side["ratio"] is None and side["budget"] == "adaptive"


def test_unified_cli_plan_does_not_execute_or_load_assets(monkeypatch, capsys):
    monkeypatch.setattr(worker, "execute", lambda args: pytest.fail("Plan started execution"))
    monkeypatch.setattr("sys.argv", ["run_unified_latency.py", "--method", "stepkv",
                                   "--dataset", "2wiki", "--model-ready", "model",
                                   "--data-ready", "data", "--output", "output"])
    worker.main()
    assert "frozen_stepkv_hf_generate_then_stop_crop_v1" in capsys.readouterr().out


def measurement_row():
    repeat = {
        "episode_seconds": 2., "tool_seconds": .5, "episode_excluding_tool_seconds": 1.5,
        "memory": {"peak_allocated_above_model_bytes": 16},
        "main_generation_calls": [{"max_new_tokens": 256}], "semantic_sha256": "same",
    }
    return {"clean_repeats": [repeat.copy(), repeat.copy()], "diagnostic_wall": repeat.copy(),
            "parity": True, "async_semantic_divergence": False,
            "diagnostics": {}, "interaction_boundaries": [], "generation_blocks": 1}


@pytest.mark.parametrize("change", ["negative_tool", "partition", "semantic_flags"])
def test_validation_rejects_invalid_accounting_even_with_valid_row_hash(monkeypatch, change):
    monkeypatch.setattr(worker, "validate_diagnostics", lambda *args: None)
    row = measurement_row()
    if change == "negative_tool":
        row["diagnostic_wall"]["tool_seconds"] = -.5
    elif change == "partition":
        row["diagnostic_wall"]["episode_excluding_tool_seconds"] = 2.
    else:
        row["diagnostic_wall"]["semantic_sha256"] = "different"
    row["row_sha256"] = worker.digest(row)
    with pytest.raises(RuntimeError):
        worker.validate_rows([row], "fullkv")


def test_clean_pass_uses_original_episode_and_drains_auxiliary_inside_outer_wall(monkeypatch):
    from scripts import latency_metrics
    llm = fake_llm()
    events = []
    llm.reset = lambda: events.append("reset")
    llm.collect_pending = lambda **kwargs: events.append(("drain", kwargs))
    monkeypatch.setattr(worker, "require_exclusive", lambda: None)
    monkeypatch.setattr(latency_metrics, "model_storage_bytes", lambda model: {"cuda:0": 128})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: events.append("sync"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *args: events.append("peak_reset"))
    monkeypatch.setattr(worker.legacy, "cuda_allocator",
                        lambda baseline: {"peak_allocated_above_model_bytes": 16})
    monkeypatch.setattr(worker.legacy, "semantic_state", lambda result, model: {"answer": result[0]})
    base = SimpleNamespace(execute_action=lambda: events.append("tool"),
                           exact_match=lambda a, b: float(a == b), f1_score=lambda a, b: float(a == b))

    def original_episode(question, model, retriever, *, pruning_mode, return_debug):
        assert question == "question" and pruning_mode == "none" and return_debug is True
        model.reset()
        model.model.generate(input_ids=torch.ones((1, 365), dtype=torch.long),
                             do_sample=False, use_cache=True, max_new_tokens=256)
        base.execute_action()
        return "answer", [], [0.]

    base._run_react_kv_episode = original_episode
    _, record = worker.run_once(base, llm, None, SimpleNamespace(mode="none"),
                                {"question": "question", "answer": "answer"})
    assert record["em"] == record["f1"] == 1.
    assert record["main_generation_calls"][0]["returned_new_tokens"] == 3
    assert events.count("reset") == events.count("peak_reset") == 1
    drain = ("drain", {"apply": False, "wait": True})
    assert events.index("tool") < events.index(drain) < len(events) - 1
    assert events[-1] == "sync"
