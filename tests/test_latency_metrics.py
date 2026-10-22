"""CPU-only observational parity, timing and memory tests (no model downloads)."""
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding
from transformers import Qwen2Config, Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache

from models.QwenLLMWithKVCache import QwenLLMWithKVCache
from scripts import latency_metrics
from scripts.latency_metrics import DetailedLatencyMonitor, model_storage_bytes
from token_tracker import TokenTracker


BYTES_PER_TOKEN = 3 * 2 * 8 * 2 * 4
ROOT = Path(__file__).resolve().parents[1]


class TinyTokenizer:
    eos_token_id = 63

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        return "prompt"

    def __call__(self, text, return_tensors=None, **kwargs):
        self.calls.append(text)
        ids = [ord(c) - 256 if 256 <= ord(c) < 320 else ord(c) % 61 + 1 for c in text]
        if return_tensors:
            tensor = torch.tensor([ids])
            return BatchEncoding(dict(input_ids=tensor, attention_mask=torch.ones_like(tensor)))
        return SimpleNamespace(input_ids=ids)

    def decode(self, ids, **kwargs):
        return "".join(chr(256 + int(i)) for i in ids)


@pytest.fixture
def tiny_loader(monkeypatch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(233)
    config = Qwen2Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=3, num_attention_heads=4,
                        num_key_value_heads=2, max_position_embeddings=256,
                        eos_token_id=None, pad_token_id=0)
    config._attn_implementation = "sdpa"
    template = Qwen2ForCausalLM(config).eval()
    template.generation_config.repetition_penalty = 1.05
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", lambda *a, **k: copy.deepcopy(template))
    monkeypatch.setattr(AutoTokenizer, "from_pretrained", lambda *a, **k: TinyTokenizer())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    def forbidden(*args, **kwargs):
        pytest.fail("CPU diagnostics must never access a CUDA API")

    for name in ("synchronize", "memory_allocated", "memory_reserved",
                 "max_memory_allocated", "max_memory_reserved",
                 "reset_peak_memory_stats", "empty_cache"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    yield template
    torch.set_num_threads(previous_threads)


def make_llm(mode="none", ratio=.5, **overrides):
    config = dict(pruning_mode=mode, cache_ratio=ratio, observation_window=0,
                  protect_prompt=True, prune_every_n=1, attn_mode="scoring_forward",
                  prompt_prefill_keep_ratio=1.0, step_poolwise_prune=True,
                  step_aware_min_keep=1)
    config.update(overrides)
    return QwenLLMWithKVCache("tiny-latency-test", config, TokenTracker())


def pairs(llm):
    return [(layer.keys, layer.values) for layer in llm.past_key_values.layers]


def assert_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for left, right in zip(a, b):
            assert_equal(left, right)
    else:
        assert a == b


def assert_state(plain, observed):
    assert_equal(pairs(plain), pairs(observed))
    assert plain.current_cache_len == observed.current_cache_len
    assert plain.token_tracker.__dict__ == observed.token_tracker.__dict__
    assert plain._all_token_ids == observed._all_token_ids
    assert plain._global_token_id_log == observed._global_token_id_log
    assert plain.timing_stats.keys() == observed.timing_stats.keys()
    assert plain.model.config._attn_implementation == observed.model.config._attn_implementation
    if plain.kv_manager is not None:
        for name in ("current_cache_len", "protected_prefix_len", "step_count"):
            assert getattr(plain.kv_manager, name) == getattr(observed.kv_manager, name)


def assert_kv(sample, length, protected=6, factor=1):
    kv = sample["kv"]
    assert kv["tokens_per_head"] == length
    assert kv["layer_tokens"] == [length] * 3
    assert kv["total_bytes"] == length * BYTES_PER_TOKEN * factor
    assert kv["protected_tokens"] == min(length, protected)
    assert kv["protected_bytes"] == min(length, protected) * BYTES_PER_TOKEN * factor
    assert kv["trajectory_tokens"] == max(0, length - protected)
    assert kv["trajectory_bytes"] == max(0, length - protected) * BYTES_PER_TOKEN * factor
    assert kv["head_expansion"]["uniform_factor"] == factor
    assert sample["allocator"] == {}


@pytest.mark.parametrize("mode", ["none", "h2o", "tova", "step_aware_h2o"])
def test_real_tiny_model_output_cache_tracker_and_rng_parity(tiny_loader, mode):
    plain, observed = make_llm(mode), make_llm(mode)
    captures = [[], []]
    handles = [llm.model.register_forward_hook(
        lambda module, args, result, target=target: target.append(result.logits.detach().clone()))
        for llm, target in zip((plain, observed), captures)]
    old_hooks = dict(observed.model._forward_hooks)
    with DetailedLatencyMonitor(observed, event_limit=3) as monitor:
        before_rng = torch.get_rng_state().clone()
        expected = plain.generate_first("question", max_new_tokens=8)
        plain_rng = torch.get_rng_state().clone()
        torch.set_rng_state(before_rng)
        actual = observed.generate_first("question", max_new_tokens=8)
        assert_equal(expected, actual)
        assert_equal(plain_rng, torch.get_rng_state())
        assert_state(plain, observed)
        for step, text in enumerate(("observation", "second observation"), 2):
            plain.token_tracker.set_current_step(step)
            observed.token_tracker.set_current_step(step)
            assert plain.generate_incremental(text, max_new_tokens=5) == observed.generate_incremental(text, max_new_tokens=5)
            assert_state(plain, observed)
        assert_equal(captures[0], captures[1])
        snapshot = monitor.snapshot()
        json.dumps(snapshot, allow_nan=False)
        assert len(snapshot["steps"]) == 3
        assert snapshot["instrumentation_errors"] == snapshot["failed_calls"] == []
        assert len(snapshot["events"]) == 3
        assert snapshot["omitted_events"] > 0
        timing = snapshot["timing"]
        assert sum(timing["exclusive_seconds"].values()) == pytest.approx(timing["root_seconds"])
        assert sum(timing["inclusive_seconds"].values()) >= timing["root_seconds"]
        if mode != "none":
            assert snapshot["pruning_events"]
            assert observed.token_tracker.total_discarded > 0
    assert "forward" not in vars(observed.model)
    assert "generate_first" not in vars(observed)
    assert observed.model._forward_hooks == old_hooks
    for handle in handles:
        handle.remove()


def test_boundaries_returned_clones_final_and_actual_workload(tiny_loader):
    llm = make_llm()
    with DetailedLatencyMonitor(llm) as monitor:
        result = llm.generate_first("question", max_new_tokens=5)
        snapshot = monitor.snapshot()
        assert_kv(snapshot["steps"][0]["memory"], 10)
        assert_kv(snapshot["final"], 10)
        boundary = snapshot["steps"][0]["memory"]
        assert boundary["site"] == "generate_first:return"
        assert snapshot["final"]["site"] == "snapshot:final"
        clones = boundary["returned_snapshots"]
        assert clones["total_bytes"] == 10 * BYTES_PER_TOKEN
        assert clones["blocks"][0]["total_bytes"] == 6 * BYTES_PER_TOKEN
        assert clones["blocks"][1]["total_bytes"] == 4 * BYTES_PER_TOKEN
        assert clones["blocks"][0]["layout"][0]["shape"] == [1, 2, 6, 8]
        assert "returned_snapshots" not in snapshot["final"]
        assert llm.current_cache_len == 11
        assert llm.tokenizer.calls == ["prompt"]
        assert snapshot["workload"]["initial_prefill"] == dict(
            forward_calls=1, input_positions=6, query_shapes=[dict(shape=[1, 6], calls=1)],
            failed_forward_calls=0)
        assert snapshot["workload"]["decode_forward"] == dict(
            forward_calls=4, input_positions=4, query_shapes=[dict(shape=[1, 1], calls=4)],
            failed_forward_calls=0)
        llm.generate_incremental("observation", max_new_tokens=3)
        snapshot = monitor.snapshot()
        assert snapshot["workload"]["observation_prefill"]["input_positions"] == 11
        assert snapshot["workload"]["observation_prefill"]["forward_calls"] == 1
        assert snapshot["peak_active_kv"]["total_bytes"] == snapshot["final"]["kv"]["total_bytes"]
        assert snapshot["scope"]["inline_cache_gather_isolated"] is False
        assert "separate clean pass" in snapshot["scope"]["latency"]
        del result


def test_public_interaction_sample_is_detached_and_keeps_wrappers_and_intervals(tiny_loader):
    llm = make_llm()
    calls = []
    utility = SimpleNamespace(call=lambda: calls.append("utility"))
    with DetailedLatencyMonitor(llm) as monitor:
        wrapped = monitor.wrap(utility, "call", "step_utility")
        llm.generate_first("question", max_new_tokens=5)
        assert utility.call is wrapped
        with monitor.measure("tool"):
            boundary = monitor.sample_memory(site="interaction:trajectory_append")
            assert_kv(boundary, 10)
            assert boundary["site"] == "interaction:trajectory_append"
            boundary["kv"]["layout"].clear()
            boundary["kv"]["total_bytes"] = -1
        first_snapshot = monitor.snapshot()
        assert first_snapshot["peak_active_kv"]["total_bytes"] == 10 * BYTES_PER_TOKEN
        assert len(first_snapshot["peak_active_kv"]["layout"]) == 3
        assert first_snapshot["timing"]["calls"]["tool"] == 1
        utility.call()
        llm.generate_first("new episode", max_new_tokens=3)
        assert utility.call is wrapped
        utility.call()
        terminal = monitor.sample_memory("interaction:terminal")
        assert_kv(terminal, 8)
        assert len(terminal["kv"]["layout"]) == 3
        snapshot = monitor.snapshot()
        assert snapshot["timing"]["calls"]["step_utility"] == 1
        assert snapshot["timing"]["calls"]["tool"] == 0
        assert snapshot["peak_active_kv"]["total_bytes"] == 8 * BYTES_PER_TOKEN
        assert len(snapshot["peak_active_kv"]["layout"]) == 3
        assert snapshot["final"]["site"] == "snapshot:final"
        assert snapshot["final"]["kv"]["total_bytes"] == terminal["kv"]["total_bytes"]
    assert calls == ["utility", "utility"]


def test_incremental_with_memory_is_observed_without_semantics_change(tiny_loader):
    plain, observed = make_llm(), make_llm()
    with DetailedLatencyMonitor(observed) as monitor:
        a = plain.generate_first("question", max_new_tokens=5)
        b = observed.generate_first("question", max_new_tokens=5)
        expected = plain.generate_incremental_with_memory("obs", a[1], a[2], None, max_new_tokens=3)
        actual = observed.generate_incremental_with_memory("obs", b[1], b[2], None, max_new_tokens=3)
        assert_equal(expected, actual)
        assert_state(plain, observed)
        step = monitor.snapshot()["steps"][-1]
        assert step["generation_call"] == "generate_incremental_with_memory"
        assert step["memory"]["returned_snapshots"]["total_bytes"] > 0
        assert step["memory"]["returned_snapshots"]["blocks"][0]["layout"][0]["shape"][2] == 3


def test_scoring_probe_is_separate_and_nested_timing_is_disjoint(tiny_loader):
    llm = make_llm()
    with DetailedLatencyMonitor(llm) as monitor:
        llm.generate_first("question", max_new_tokens=5)
        before = copy.deepcopy(pairs(llm))
        tracker = copy.deepcopy(llm.token_tracker.__dict__)
        with monitor.measure("selection_and_cache_management"):
            attentions = llm._get_attention_for_scoring()
        assert attentions[-1].shape[-1] == 11
        assert_equal(pairs(llm), before)
        assert llm.token_tracker.__dict__ == tracker
        snapshot = monitor.snapshot()
        assert snapshot["peak_active_kv"]["total_bytes"] == 10 * BYTES_PER_TOKEN
        assert snapshot["peak_probe_kv"]["total_bytes"] == 11 * BYTES_PER_TOKEN
        assert snapshot["workload"]["scoring_forward"]["input_positions"] == 1
        assert snapshot["workload"]["scoring_forward"]["forward_calls"] == 1
        assert snapshot["timing"]["calls"]["scoring_forward"] == 2
        assert sum(snapshot["timing"]["exclusive_seconds"].values()) == pytest.approx(snapshot["timing"]["root_seconds"])


def test_manual_clock_exact_nested_exclusive_inclusive_and_once_only_wrap(tiny_loader, monkeypatch):
    llm = make_llm()
    clock = iter([0., 1., 2., 3., 4., 7.])
    with DetailedLatencyMonitor(llm) as monitor:
        monkeypatch.setattr(latency_metrics.time, "perf_counter", lambda: next(clock))
        with monitor.measure("tool"):
            with monitor.measure("decode_forward"):
                with monitor.measure("token_saliency"):
                    pass
        timing = monitor.snapshot()["timing"]
        assert timing["root_seconds"] == 7
        assert timing["exclusive_seconds"]["tool"] == 4
        assert timing["exclusive_seconds"]["decode_forward"] == 2
        assert timing["exclusive_seconds"]["token_saliency"] == 1
        assert timing["inclusive_seconds"]["tool"] == 7
        assert timing["inclusive_seconds"]["decode_forward"] == 3
        assert timing["inclusive_seconds"]["token_saliency"] == 1
        counter = []

        def utility(a, *, b):
            counter.append((a, b))
            return a + b

        obj = SimpleNamespace(utility=utility)
        wrapper = monitor.wrap(obj, "utility", "step_utility")
        assert monitor.wrap(obj, "utility", "step_utility") is wrapper
        with pytest.raises(ValueError, match="already wrapped"):
            monitor.wrap(obj, "utility", "tool")
        clock = iter([10., 12.])
        assert obj.utility(3, b=4) == 7
        assert counter == [(3, 4)]
        assert monitor.snapshot()["timing"]["exclusive_seconds"]["step_utility"] == 2
    assert obj.utility is utility


def test_nonfinite_clock_is_explicit_and_restores(tiny_loader, monkeypatch):
    llm = make_llm()
    monitor = DetailedLatencyMonitor(llm)
    clock = iter([0., float("nan")])
    monkeypatch.setattr(latency_metrics.time, "perf_counter", lambda: next(clock))
    with pytest.raises(ValueError, match="latency clock"):
        with monitor.measure("tool"):
            pass
    assert monitor.instrumentation_errors
    assert "generate_first" not in vars(llm)
    assert "forward" not in vars(llm.model)


def test_failed_original_forward_called_once_and_restored(tiny_loader):
    llm = make_llm()
    calls = []
    original_forward = llm.model.forward

    def failed(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("intentional forward failure")

    llm.model.forward = failed
    monitor = DetailedLatencyMonitor(llm)
    with pytest.raises(RuntimeError, match="intentional forward failure"):
        llm.generate_first("question", max_new_tokens=3)
    assert calls == [1]
    assert llm.model.forward is failed
    assert "generate_first" not in vars(llm)
    snapshot = monitor.snapshot()
    assert snapshot["steps"][0]["succeeded"] is False
    assert snapshot["workload"]["initial_prefill"]["failed_forward_calls"] == 1
    assert snapshot["failed_calls"]
    assert snapshot["instrumentation_errors"] == []
    monitor.close()
    llm.model.forward = original_forward


def test_new_episode_and_diagnostic_reset_do_not_mutate_or_inherit_cache(tiny_loader):
    llm = make_llm()
    llm.generate_first("unobserved", max_new_tokens=15)
    with DetailedLatencyMonitor(llm) as monitor:
        llm.generate_first("first", max_new_tokens=5)
        assert monitor.snapshot()["peak_active_kv"]["tokens_per_head"] == 10
        llm.generate_first("second", max_new_tokens=3)
        snapshot = monitor.snapshot()
        assert snapshot["peak_active_kv"]["tokens_per_head"] == 8
        assert len(snapshot["steps"]) == 1
        before = copy.deepcopy(pairs(llm))
        tracker = copy.deepcopy(llm.token_tracker.__dict__)
        monitor.reset_episode()
        assert_equal(pairs(llm), before)
        assert llm.token_tracker.__dict__ == tracker
        assert monitor.steps == []
        assert not any(monitor.calls.values())


def test_storage_dedup_ragged_and_missing_dynamic_layers(tiny_loader):
    llm = make_llm()
    with DetailedLatencyMonitor(llm) as monitor:
        backing = torch.zeros(1, 4, 10, 8)
        llm.past_key_values = tuple((backing[:, :, :n], backing[:, :, :n]) for n in (3, 5, 7))
        pointers = [x.data_ptr() for pair in llm.past_key_values for x in pair]
        snapshot = monitor.snapshot()
        kv = snapshot["final"]["kv"]
        assert kv["tokens_per_head"] is None
        assert kv["layer_tokens"] == [3, 5, 7]
        assert kv["total_bytes"] == 2 * (3 + 5 + 7) * 4 * 8 * 4
        assert kv["storage_bytes"] == backing.untyped_storage().nbytes()
        assert kv["head_expansion"]["uniform_factor"] == 2
        assert [x.data_ptr() for pair in llm.past_key_values for x in pair] == pointers
        llm.past_key_values = DynamicCache(config=llm.model.config)
        kv = monitor.snapshot()["final"]["kv"]
        assert kv["missing_layers"] == [0, 1, 2]
        assert kv["layer_tokens"] == [None, None, None]
        assert kv["tokens_per_head"] is None
        assert kv["complete"] is False
        assert kv["total_bytes"] == 0
        llm.past_key_values.update(backing[:, :2, :3], backing[:, :2, :3], 0)
        kv = monitor.snapshot()["final"]["kv"]
        assert kv["materialized_layers"] == 1
        assert kv["missing_layers"] == [1, 2]


@pytest.mark.parametrize("malformed", [
    ((torch.zeros(1, 2, 3, 8), None),),
    ((torch.zeros(2, 2, 3, 8), torch.zeros(2, 2, 3, 8)),),
    ((torch.zeros(1, 2, 3, 8), torch.zeros(1, 2, 4, 8)),),
    ((torch.zeros(3, 8), torch.zeros(3, 8)),),
])
def test_invalid_cache_metadata_never_silently_reports_zero(tiny_loader, malformed):
    llm = make_llm()
    monitor = DetailedLatencyMonitor(llm)
    llm.past_key_values = malformed
    with pytest.raises((TypeError, ValueError)):
        monitor.snapshot()
    assert monitor.instrumentation_errors
    assert "forward" not in vars(llm.model)
    assert "generate_first" not in vars(llm)


def test_model_parameters_and_buffers_storage_dedup_and_growth(tiny_loader):
    llm = make_llm()
    parameter = next(llm.model.parameters())
    llm.model.register_buffer("shared_storage", parameter.detach())
    exported_baseline = model_storage_bytes(llm.model)
    with DetailedLatencyMonitor(llm) as monitor:
        snapshot = monitor.snapshot()
        storages = {(str(t.device), t.untyped_storage().data_ptr(), t.untyped_storage().nbytes()):
                    t.untyped_storage().nbytes()
                    for t in (*llm.model.parameters(), *llm.model.buffers())}
        baseline = sum(storages.values())
        assert exported_baseline == {"cpu": baseline}
        assert snapshot["model_storage"]["baseline_by_device"]["cpu"] == baseline
        llm.model.register_buffer("growing_buffer", torch.zeros(17, dtype=torch.float32))
        snapshot = monitor.snapshot()
        assert model_storage_bytes(llm.model) == {"cpu": baseline + 68}
        assert snapshot["model_storage"]["current_by_device"]["cpu"] == baseline + 68
        assert snapshot["model_storage"]["baseline_by_device"]["cpu"] == baseline


def test_exported_model_storage_uses_backing_size_for_parameter_buffer_views(tiny_loader):
    model = torch.nn.Module()
    storage = torch.zeros(32, dtype=torch.float32)
    model.register_parameter("partial_parameter", torch.nn.Parameter(storage[:4]))
    model.register_buffer("partial_buffer", storage[8:12])
    model.register_buffer("separate_buffer", torch.zeros(3, dtype=torch.int64))
    assert model_storage_bytes(model) == {"cpu": 32 * 4 + 3 * 8}
    assert model_storage_bytes(torch.nn.Module()) == {}


def test_fake_allocator_reset_order_step_and_episode_peaks_and_driver(tiny_loader, monkeypatch):
    llm = make_llm()
    llm.generate_first("previous", max_new_tokens=20)
    reset = llm.reset
    state = dict(allocated=9999, reserved=12000, peak_a=9999, peak_r=12000)
    order = []
    process_reads = []

    def read_process_memory():
        process_reads.append(1)
        return {"cuda:3": 450}

    def original_reset():
        assert state["peak_a"] == 9999
        reset()
        state.update(allocated=100, reserved=120, peak_a=9999, peak_r=12000)
        order.append("original_reset_completed")

    llm.reset = original_reset
    monitor = DetailedLatencyMonitor(llm, process_memory_reader=read_process_memory)
    monitor._devices = {"cuda:3"}
    monkeypatch.setattr(monitor, "_model_storage", lambda: {"cuda:3": 100})
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: order.append(("sync", str(device))))
    for name, key in (("memory_allocated", "allocated"), ("memory_reserved", "reserved"),
                      ("max_memory_allocated", "peak_a"), ("max_memory_reserved", "peak_r")):
        monkeypatch.setattr(torch.cuda, name, lambda device, key=key: state[key])

    def reset_peaks(device):
        assert "original_reset_completed" in order
        assert llm.past_key_values is None or len(monitor.steps) > 0
        order.append(("reset_peaks", str(device)))
        state["peak_a"], state["peak_r"] = state["allocated"], state["reserved"]

    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", reset_peaks)
    original_forward = llm.model.forward

    def allocate(*args, **kwargs):
        state.update(allocated=200, reserved=300)
        state["peak_a"] = max(state["peak_a"], 500 if monitor._call_name == "generate_first" else 350)
        state["peak_r"] = max(state["peak_r"], 700 if monitor._call_name == "generate_first" else 400)
        return original_forward(*args, **kwargs)

    llm.model.forward = allocate
    try:
        llm.generate_first("first monitored", max_new_tokens=3)
        assert len(process_reads) == 1
        # Simulate returned clones released after the generation-block boundary.
        state.update(allocated=150, reserved=300)
        llm.generate_incremental("obs", max_new_tokens=2)
        assert len(process_reads) == 2
        snapshot = monitor.snapshot()
        assert len(process_reads) == 3
        assert order.index("original_reset_completed") < order.index(("reset_peaks", "cuda:3"))
        assert snapshot["steps"][0]["allocator_peak"]["cuda:3"] == dict(peak_allocated_bytes=500, peak_reserved_bytes=700)
        assert snapshot["steps"][1]["allocator_peak"]["cuda:3"] == dict(peak_allocated_bytes=350, peak_reserved_bytes=400)
        assert snapshot["allocator_episode_peak"]["cuda:3"] == dict(peak_allocated_bytes=500, peak_reserved_bytes=700)
        allocation = snapshot["final"]["allocator"]["cuda:3"]
        assert allocation["model_storage_bytes"] == allocation["model_storage_baseline_bytes"] == 100
        assert allocation["allocated_above_model_bytes"] == 100
        assert allocation["reserved_above_model_bytes"] == 200
        assert allocation["peak_allocated_above_model_bytes"] == 250
        assert allocation["driver_process_used_bytes"] == 450
        assert allocation["driver_above_model_bytes"] == 350
        assert snapshot["final"]["driver_process_sampled"] is True
        monitor.sample_memory("interaction:terminal")
        assert len(process_reads) == 4
        monitor._sample("prune:before")
        monitor._sample("forward:after")
        assert len(process_reads) == 4
    finally:
        monitor.close()


def test_cpu_never_calls_injected_process_memory_reader(tiny_loader):
    llm = make_llm()

    def forbidden_reader():
        pytest.fail("CPU models must not launch a driver process-memory probe")

    with DetailedLatencyMonitor(llm, process_memory_reader=forbidden_reader) as monitor:
        llm.generate_first("question", max_new_tokens=3)
        sample = monitor.sample_memory("interaction:end")
        assert sample["driver_process_sampled"] is False
        assert monitor.snapshot()["final"]["driver_process_sampled"] is False


def load_adapter(monkeypatch, path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_lazy_recurrence_head_expansion_window_floor_and_restoration(tiny_loader, monkeypatch):
    lazy_runtime = load_adapter(monkeypatch, "rebuttal/adapters/lazyeviction/models/LazyEvictionLLM.py",
                                "models.LazyEvictionLLM")
    lazy = load_adapter(monkeypatch, "rebuttal/adapters/lazyeviction/models/LegacyBaseline.py",
                        "latency_test_lazy_baseline")
    config = dict(cache_ratio=.05, lazy_window=3)
    plain = lazy.LegacyBaselineLLM("tiny", config)
    observed = lazy.LegacyBaselineLLM("tiny", config)
    original_functions = [lazy.update_recurrence, lazy._select_at_boundary, lazy.gather_state]
    original_hooks = dict(observed.model._forward_hooks)
    expected_first = plain.generate_first("question", max_new_tokens=8)
    expected = plain.generate_incremental("long observation", max_new_tokens=5)
    with DetailedLatencyMonitor(observed) as monitor:
        monitor.wrap(lazy, "update_recurrence", "token_saliency")
        monitor.wrap(lazy, "_select_at_boundary", "selection_and_cache_management")
        monitor.wrap(lazy, "gather_state", "cache_gather")
        actual_first = observed.generate_first("question", max_new_tokens=8)
        assert_equal(expected_first, actual_first)
        actual = observed.generate_incremental("long observation", max_new_tokens=5)
        assert actual == expected
        assert_equal(pairs(plain), pairs(observed))
        for layer in plain._legacy_recurrence:
            for field in ("ids", "last", "mri"):
                assert_equal(getattr(plain._legacy_recurrence[layer], field),
                             getattr(observed._legacy_recurrence[layer], field))
        snapshot = monitor.snapshot()
        kv = snapshot["final"]["kv"]
        assert kv["head_expansion"]["uniform_factor"] == 2
        assert all(entry["actual_heads"] == 4 and entry["native_heads"] == 2 for entry in kv["layout"])
        assert "not shared logical tokens" in kv["token_scope"]
        recurrence = snapshot["final"]["recurrence"]
        expected_metadata = sum(t.numel() * t.element_size() for state in observed._legacy_recurrence.values()
                                for t in (state.ids, state.last, state.mri))
        assert recurrence["total_bytes"] == expected_metadata
        assert recurrence["storage_bytes"] == observed.legacy_audit()["tracking_bytes"]
        assert snapshot["timing"]["exclusive_seconds"]["token_saliency"] > 0
        assert snapshot["timing"]["calls"]["cache_gather"] > 0
        assert snapshot["workload"].get("scoring_forward", {}).get("forward_calls", 0) == 0
        assert snapshot["steps"][-1]["peak_preprune_kv"]["total_bytes"] > 0
        prunes = [event for event in observed.cache_measurements if event["pruned"]]
        assert prunes
        assert all(event["after_tokens"] >= 6 + min(3, event["query_clock"]) for event in prunes)
        assert any(event["window_floor_added_tokens"] > 0 for event in prunes)
        assert snapshot["final"]["owner_returned_snapshots"]["total_bytes"] > 0
        assert snapshot["scope"]["inline_cache_gather_isolated"] is False
    assert original_functions == [lazy.update_recurrence, lazy._select_at_boundary, lazy.gather_state]
    assert observed.model._forward_hooks == original_hooks
    assert lazy_runtime.RecurrenceState is type(next(iter(observed._legacy_recurrence.values())))


def test_stop_crop_boundaries_and_window_floor(tiny_loader):
    preview = make_llm().generate_first("question", max_new_tokens=8)[0]
    stop = preview[3:5]
    plain, observed = make_llm(), make_llm()
    with DetailedLatencyMonitor(observed) as monitor:
        expected = plain.generate_first("question", max_new_tokens=8, stop_strings=[stop])
        actual = observed.generate_first("question", max_new_tokens=8, stop_strings=[stop])
        assert_equal(expected, actual)
        assert_state(plain, observed)
        snapshot = monitor.snapshot()
        assert snapshot["timing"]["calls"]["stop_crop"] == 1
        assert snapshot["peak_active_kv"]["tokens_per_head"] > snapshot["final"]["kv"]["tokens_per_head"]
        assert snapshot["steps"][0]["memory"]["kv"]["tokens_per_head"] == snapshot["final"]["kv"]["tokens_per_head"]
    llm = make_llm("tova", .05, observation_window=7)
    with DetailedLatencyMonitor(llm) as monitor:
        llm.generate_first("question", max_new_tokens=6)
        llm.generate_incremental("long observation", max_new_tokens=3)
        snapshot = monitor.snapshot()
        assert snapshot["policy"]["recent_window"] == 7
        assert snapshot["final"]["kv"]["tokens_per_head"] >= 6 + 7


def test_exact_existing_attributes_and_external_hooks_restored(tiny_loader):
    llm = make_llm("tova")
    original_prune = llm._do_pruning
    llm._do_pruning = lambda *a, **k: original_prune(*a, **k)
    external = llm.model.register_forward_hook(lambda *args: None)
    objects = (llm, llm.model, llm.kv_manager.pruning_strategy,
               llm.kv_manager.pruning_strategy.h2o_scorer)
    surface = [(obj, dict(vars(obj))) for obj in objects]
    monitor = DetailedLatencyMonitor(llm)
    monitor.close()
    monitor.close()
    for obj, old in surface:
        assert set(vars(obj)) == set(old)
        for name, value in old.items():
            assert vars(obj)[name] is value, name
    external.remove()
