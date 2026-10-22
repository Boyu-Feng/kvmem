import contextlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts import latency_runtime as runtime
from scripts import run_latency_experiment as runner
from scripts import summarize_latency_experiment as report
from scripts.run_qa32_rebuttal import atomic_json
from test_legacy_efficiency_metrics import model_factory


def test_plan_is_default_and_never_reads_assets_or_initializes_cuda(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(runner, "execute", lambda *args: pytest.fail("Default CLI must not execute inference"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("Planning must not query CUDA"))
    assert runner.main([
        "--method", "lazyeviction", "--model-ready", str(tmp_path / "not_downloaded.json"),
        "--data-ready", str(tmp_path / "data.json"), "--output", str(tmp_path / "results"),
    ]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["status"] == "planned_not_executed"
    assert plan["sample_positions"] == list(range(20)) and plan["warmup_positions"] == [500, 501]
    assert "7x" in plan["lazy_budget_warning"]
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("method", list(runtime.METHODS))
def test_frozen_method_loads_real_cpu_model_without_altering_namespaces(method, model_factory):
    import run_all_wiki_experiments_v2 as base
    from token_tracker import TokenTracker

    before = dict(sys.modules)
    with runtime.frozen_method(method) as spec:
        llm = spec.llm_class("tiny", kv_config=spec.config(base, .2), token_tracker=TokenTracker())
        result = llm.generate_first("question", max_new_tokens=4)
        assert isinstance(result[0], str)
        assert llm.past_key_values.layers[0].keys.shape[-2] > 0
        assert spec.mode == runtime.METHODS[method]
        actual = llm.past_key_values.layers[0].keys.shape[1]
        assert actual == (4 if method == "lazyeviction" else 2)
    for name in ("models.LazyEvictionLLM", f"scripts._latency_adapter_{method}"):
        assert sys.modules.get(name) is before.get(name)


def test_adapter_namespace_restores_on_failure():
    previous = sys.modules.get("models.LazyEvictionLLM")
    with pytest.raises(RuntimeError, match="test"):
        with runtime.frozen_method("lazyeviction"):
            raise RuntimeError("test")
    assert sys.modules.get("models.LazyEvictionLLM") is previous


@pytest.mark.parametrize("method", list(runtime.METHODS))
def test_complete_original_episode_diagnostic_wiring_preserves_outputs_and_state(method, model_factory):
    import run_all_wiki_experiments_v2 as base
    from scripts.latency_metrics import DetailedLatencyMonitor
    from token_tracker import TokenTracker

    previous_trace = sys.gettrace()
    with runtime.frozen_method(method) as spec:
        llm = spec.llm_class("tiny", kv_config=spec.config(base, .2), token_tracker=TokenTracker())
        original_first = llm.generate_first

        def short_first(text, max_new_tokens=256, stop_strings=None):
            return original_first(text, max_new_tokens=4, stop_strings=stop_strings)

        with runtime.patch_attribute(llm, "generate_first", short_first):
            plain = base._run_react_kv_episode("question", llm, None, pruning_mode=spec.mode, return_debug=True)
            reference = runner.semantic_state(plain, llm)
            with DetailedLatencyMonitor(llm) as monitor:
                runtime.instrument_episode(base, spec, llm, monitor)
                with runtime.observe_kv_gathers(llm, monitor), runtime.EpisodeTrace(base._run_react_kv_episode, monitor) as trace:
                    observed = base._run_react_kv_episode(
                        "question", llm, None, pruning_mode=spec.mode, return_debug=True)
                stats = monitor.snapshot()
            assert runner.semantic_state(observed, llm) == reference
            runtime.validate_diagnostics(stats, trace.boundaries, len(observed[2]))
            assert stats["peak_active_kv"]["total_bytes"] > 0
            assert stats["final"]["kv"]["layout"][0]["native_heads"] == 2
            assert stats["final"]["kv"]["layout"][0]["actual_heads"] == (4 if method == "lazyeviction" else 2)
            assert stats["timing"]["exclusive_seconds"]
    assert sys.gettrace() is previous_trace


def test_scoped_gathers_measure_only_active_kv_and_restore_tensor_methods():
    keys = torch.arange(24).reshape(1, 2, 3, 4)
    llm = SimpleNamespace(past_key_values=((keys, keys.clone()),))
    labels = []

    @contextlib.contextmanager
    def measure(phase):
        labels.append(phase)
        yield

    prior = torch.Tensor.gather, torch.Tensor.index_select
    monitor = SimpleNamespace(measure=measure)
    with runtime.observe_kv_gathers(llm, monitor):
        result = keys.index_select(2, torch.tensor([0, 2]))
        assert result.shape == (1, 2, 2, 4)
        keys[..., :2, :].gather(2, torch.zeros((1, 2, 1, 4), dtype=torch.long))
        torch.ones(1, 8).gather(1, torch.tensor([[1, 3]]))
    assert labels == ["cache_gather", "cache_gather"]
    assert (torch.Tensor.gather, torch.Tensor.index_select) == prior
    with pytest.raises(RuntimeError):
        with runtime.observe_kv_gathers(llm, monitor):
            raise RuntimeError("test")
    assert (torch.Tensor.gather, torch.Tensor.index_select) == prior


def test_utility_ranges_cover_inline_reward_and_citation_not_model_call():
    import run_all_wiki_experiments_v2 as base
    ranges = runtime.utility_ranges(base._run_react_kv_episode)
    lines = Path(base.__file__).read_text().splitlines()
    reward = "\n".join(lines[ranges["reward_novelty_repeat"][0] - 1:ranges["reward_novelty_repeat"][1]])
    citation = "\n".join(lines[ranges["citation_history_scan"][0] - 1:ranges["citation_history_scan"][1]])
    assert "novelty_ratio" in reward and "repeat_penalty" in reward
    assert "for sid, meta in step_meta.items()" in citation
    assert "generate_incremental(" not in reward + citation


def _trace_fixture():
    trajectory_log, step_timings = [], []
    enabled = True
    if enabled:
        prev_action_norm = "look"
        reward = len(prev_action_norm)
    if enabled:
        ref_terms = {"apple"}
        citation = len(ref_terms)
    step_timings.append({"step": 1})
    trajectory_log.append({"step": 1})
    step_timings.append({"step": 2})
    return reward + citation


def test_trace_includes_complete_tool_step_and_unobserved_terminal_boundary():
    records = []

    @contextlib.contextmanager
    def measure(phase):
        records.append(("enter", phase))
        yield
        records.append(("exit", phase))

    monitor = SimpleNamespace(measure=measure, sample_memory=lambda site: {"site": site})
    prior = sys.gettrace()
    with runtime.EpisodeTrace(_trace_fixture, monitor) as trace:
        assert _trace_fixture() == 5
    assert sys.gettrace() is prior
    assert [boundary["step"] for boundary in trace.boundaries] == [1, 2]
    assert [boundary["endpoint"] for boundary in trace.boundaries] == [
        "action_and_tool_return", "terminal_generation_no_tool_return",
    ]
    assert records == [("enter", "step_utility"), ("exit", "step_utility"),
                       ("enter", "step_utility"), ("exit", "step_utility")]


def test_byte_reference_never_treats_same_slots_as_same_memory():
    memory = {"kv": {
        "total_bytes": 70 * 10, "protected_bytes": 70 * 4, "trajectory_bytes": 70 * 6,
        "protected_tokens": 4,
        "layout": [{"key_shape": [1, 28, 10, 1], "value_shape": [1, 28, 10, 1],
                    "key_bytes": 350, "value_bytes": 350, "native_heads": 4}],
    }}
    result = runtime.native_equivalent_budget(memory, .2, 30)
    assert result["native_gqa_reference_bytes"] == 100
    assert result["actual_over_native_reference"] == 7
    assert result["matched_byte_budget"] is False
    assert result["protected_prompt_alone_exceeds_reference"] is True


def test_exclusive_gpu_rejects_other_processes_and_records_own_bytes(monkeypatch):
    own = os.getpid()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")

    def query(args):
        if any(arg.startswith("--query-gpu") for arg in args):
            return "GPU-test, Test H200, 100, test-driver\n"
        return f"GPU-test, {own}, 50\n"

    monkeypatch.setattr(runtime, "_query", query)
    assert runtime.process_memory_reader() == {"cuda:0": 50 * 1024 ** 2}
    monkeypatch.setattr(runtime, "gpu_state", lambda: {"processes": [{"pid": own + 1}]})
    with pytest.raises(RuntimeError, match="shared"):
        runtime.require_exclusive()


def test_clean_episode_reset_tool_accounting_and_restoration_without_gpu(monkeypatch):
    from scripts import latency_metrics

    order = []
    original_reset = lambda: order.append("original_reset")
    llm = SimpleNamespace(model=object(), reset=original_reset)
    base = SimpleNamespace(execute_action=lambda *args: "observation")
    original_tool = base.execute_action

    def episode(question, owner, retriever, **kwargs):
        owner.reset()
        assert base.execute_action("search", question) == "observation"
        return "answer", [], [], {}

    base._run_react_kv_episode = episode
    monkeypatch.setattr(runner, "require_exclusive", lambda: order.append("exclusive_check"))
    monkeypatch.setattr(latency_metrics, "model_storage_bytes", lambda model: {"cuda:0": 128})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: order.append("sync"))
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: order.append("peak_reset"))
    for name, value in (("memory_allocated", 512), ("memory_reserved", 1024),
                        ("max_memory_allocated", 2048), ("max_memory_reserved", 4096)):
        monkeypatch.setattr(torch.cuda, name, lambda device, value=value: value)
    result, measurement = runner.run_once(base, llm, None, SimpleNamespace(mode="none"), "question")
    assert result[0] == "answer"
    assert order.index("original_reset") < order.index("peak_reset")
    assert order.count("exclusive_check") == 2
    assert measurement["tool_calls"] == 1
    assert 0 <= measurement["tool_seconds"] <= measurement["episode_seconds"]
    assert measurement["memory"]["peak_allocated_above_model_bytes"] == 2048 - 128
    assert llm.reset is original_reset and base.execute_action is original_tool

    def failed_episode(question, owner, retriever, **kwargs):
        owner.reset()
        print("[ERROR] original inference failed")
        return "fallback", [], [], {}

    base._run_react_kv_episode = failed_episode
    with pytest.raises(RuntimeError, match="suppressed"):
        runner.run_once(base, llm, None, SimpleNamespace(mode="none"), "question")
    assert llm.reset is original_reset and base.execute_action is original_tool


def _memory(bytes_=100, expansion=1):
    return {
        "kv": {"total_bytes": bytes_, "protected_bytes": 40, "trajectory_bytes": bytes_ - 40,
               "tokens_per_head": 10,
               "layout": [{"actual_heads": 4 * expansion, "native_heads": 4}]},
        "allocator": {"cuda:0": {
            "allocated_above_model_bytes": bytes_ + 20, "reserved_above_model_bytes": bytes_ + 50,
            "driver_above_model_bytes": bytes_ + 70,
        }},
    }


def _rows(seconds, expansion=1):
    rows = []
    for index, value in enumerate(seconds):
        memory = _memory(100 * expansion, expansion)
        row = {
            "id": str(index), "parity": True, "generation_blocks": 1, "em": 1, "f1": 1,
            "clean_repeats": [{
                "episode_seconds": value, "tool_seconds": .1,
                "episode_excluding_tool_seconds": value - .1,
                "memory": {"peak_allocated_above_model_bytes": 200 * expansion},
            }] * 2,
            "diagnostics": {
                "steps": [{"memory": memory}], "final": memory,
                "peak_active_kv": memory["kv"],
                "timing": {"exclusive_seconds": {"decode_forward": .2}},
                "workload": {"initial_prefill": {"forward_calls": 1, "input_positions": 6},
                             "decode_forward": {"forward_calls": 3, "input_positions": 3}},
            },
            "interaction_boundaries": [{"step": 1, "endpoint": "terminal_generation_no_tool_return", "memory": memory}],
            "diagnostic_unattributed_seconds": .1,
        }
        row["row_sha256"] = runtime.digest(row)
        rows.append(row)
    return rows


def _completed(directory, method, seconds, expansion=1):
    rows = _rows(seconds, expansion)
    protocol = {key: "same" for key in report.COMPARABLE}
    protocol.update(method=method, ratio=1.0 if method == "fullkv" else .2,
                    model=list(runtime.MODEL), hardware={"name": "actual-test-hardware"},
                    seed=233, sample_ids=[row["id"] for row in rows],
                    clean_repeats_per_question=2)
    atomic_json(directory / "manifest.json", {"protocol": protocol})
    atomic_json(directory / "status.json", {"status": "complete"})
    atomic_json(directory / "measurements.json", {
        "protocol_sha256": runtime.digest(protocol), "rows": rows, "summary": runner.condition_summary(rows),
    })
    return directory


def test_report_matches_ids_reports_actual_bytes_and_paired_latency(tmp_path):
    full = _completed(tmp_path / "full", "fullkv", [2, 4])
    lazy = _completed(tmp_path / "lazy", "lazyeviction", [1, 3], 7)
    output = tmp_path / "report"
    summary = report.summarize_runs([full, lazy], output, bootstrap_samples=100)
    row = summary["conditions"][1]
    assert row["mean_episode_seconds_over_fullkv"] == pytest.approx(2 / 3)
    assert row["mean_final_kv_bytes_over_fullkv"] == 7
    assert row["head_expansion_factors"] == [7]
    assert row["uses_native_kv_head_count"] is False
    assert row["diagnostic_mean_workload"]["decode_forward"]["input_positions"] == 3
    assert row["paired_latency_difference_seconds_vs_fullkv"]["ci95"] == [-1, -1]
    assert "kv_bytes" in (output / "lazyeviction_r20_boundaries.csv").read_text()


def test_report_rejects_mismatched_ids_partial_and_corrupt_rows(tmp_path):
    full = _completed(tmp_path / "full", "fullkv", [2, 4])
    other = _completed(tmp_path / "step", "stepkv", [1, 3])
    manifest = runtime.read_json(other / "manifest.json")
    manifest["protocol"]["hardware"] = {"name": "different"}
    measurements = runtime.read_json(other / "measurements.json")
    measurements["protocol_sha256"] = runtime.digest(manifest["protocol"])
    atomic_json(other / "manifest.json", manifest)
    atomic_json(other / "measurements.json", measurements)
    with pytest.raises(RuntimeError, match="hardware"):
        report.summarize_runs([full, other], tmp_path / "report", 100)
    atomic_json(other / "status.json", {"status": "running"})
    with pytest.raises(RuntimeError, match="not complete"):
        report.validate_run(other)
    corrupt = _completed(tmp_path / "corrupt", "stepkv", [1, 3])
    measurements = runtime.read_json(corrupt / "measurements.json")
    measurements["rows"][0]["clean_repeats"][0]["episode_seconds"] = 999
    atomic_json(corrupt / "measurements.json", measurements)
    with pytest.raises(RuntimeError, match="Corrupt"):
        report.validate_run(corrupt)
