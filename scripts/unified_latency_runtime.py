"""Shared original-StepKV generation audit and observational latency diagnostics."""
from __future__ import annotations

import contextlib
from collections.abc import Mapping
import threading
import time

import torch

from scripts.latency_runtime import Method, frozen_method, patch_attribute
from scripts.latency_metrics import DetailedLatencyMonitor, _cache_pairs, _tensor_group

DECODE_POLICY = "frozen_stepkv_hf_generate_then_stop_crop_v1"
METHODS = ("fullkv", "stepkv", "lazyeviction", "sidequest_unified", "thinkkv_unified")
DATASETS = ("hotpotqa", "2wiki", "musique")


@contextlib.contextmanager
def unified_method(name):
    if name not in METHODS:
        raise ValueError(f"Unknown unified latency method: {name}")
    if name in ("fullkv", "stepkv", "lazyeviction"):
        with frozen_method(name) as method:
            yield method
        return
    with frozen_method("fullkv") as control:
        if name == "sidequest_unified":
            from scripts.unified_sidequest_runtime import UnifiedSideQuestLLM
            cls = UnifiedSideQuestLLM
        else:
            from scripts.unified_thinkkv_runtime import UnifiedThinkKVLLM
            cls = UnifiedThinkKVLLM
        yield Method("fullkv", "none", cls, {}, (),
                     f"{name}: original StepKV HF main generation and posthoc stop crop; "
                     "algorithm-specific cache/auxiliary operations, unquantized KV.")


class GenerationAudit:
    """Metadata-only audit; never add forwards, stop rules or synchronization."""

    def __init__(self, llm):
        self.llm = llm
        self.thread = threading.get_ident()
        self.calls = []
        self.scope = contextlib.ExitStack()

    def __enter__(self):
        original = self.llm.model.generate

        def generate(*args, **kwargs):
            main = threading.get_ident() == self.thread
            if main:
                if kwargs.get("do_sample") is not False or kwargs.get("use_cache") is not True:
                    raise RuntimeError("Main generation must preserve frozen greedy cached generation")
                if kwargs.get("stopping_criteria") is not None or kwargs.get("stop_strings") is not None:
                    raise RuntimeError("Early/custom stopping cannot enter original-StepKV latency")
                cap = kwargs.get("max_new_tokens")
                if cap not in (255, 256):
                    raise RuntimeError(f"Unexpected frozen main generation cap: {cap}")
                ids = kwargs.get("input_ids", args[0] if args else None)
                if not isinstance(ids, torch.Tensor) or ids.ndim != 2 or ids.shape[0] != 1:
                    raise RuntimeError("Unified HF generation requires batch-one token IDs")
                before = int(ids.shape[-1])
            result = original(*args, **kwargs)
            if main:
                self.calls.append({"input_positions": before, "max_new_tokens": cap,
                                   "sequence_positions": int(result.sequences.shape[-1]),
                                   "returned_new_tokens": int(result.sequences.shape[-1]) - before,
                                   "stop_policy": "Original text/KV crop after HF generation returns"})
            return result

        self.scope.enter_context(patch_attribute(self.llm.model, "generate", generate))
        return self

    def __exit__(self, *exc):
        self.scope.close()


def adapter_audit(llm):
    if hasattr(llm, "audit"):
        return llm.audit()
    if hasattr(llm, "legacy_audit"):
        return llm.legacy_audit()
    return {"scope": "Unchanged original QA runtime"}


def drain_auxiliary(llm):
    if hasattr(llm, "collect_pending"):
        llm.collect_pending(apply=False, wait=True)


class UnifiedMonitor(DetailedLatencyMonitor):
    """Main exclusive intervals and separately labeled overlapping auxiliary spans."""

    def __init__(self, llm, **kwargs):
        self.main_thread = threading.get_ident()
        self.auxiliary_events = []
        self.auxiliary_lock = threading.RLock()
        super().__init__(llm, **kwargs)

    def reset_episode(self):
        super().reset_episode()
        with self.auxiliary_lock:
            self.auxiliary_events.clear()

    def _forward(self, original, args, kwargs):
        if threading.get_ident() == self.main_thread:
            return super()._forward(original, args, kwargs)
        inputs = kwargs.get("input_ids", args[0] if args else None)
        if not isinstance(inputs, torch.Tensor) or inputs.ndim != 2 or inputs.shape[0] != 1:
            raise RuntimeError("Auxiliary forward requires batch-one input_ids")
        self._sync()
        begin = time.perf_counter()
        result = original(*args, **kwargs)
        self._sync()
        with self.auxiliary_lock:
            self.auxiliary_events.append({
                "phase": "auxiliary_forward", "input_positions": int(inputs.shape[-1]),
                "seconds": time.perf_counter() - begin,
                "scope": "Overlapping synchronized auxiliary span; do NOT add to main exclusive times"})
        return result

    @contextlib.contextmanager
    def measure(self, phase):
        if threading.get_ident() == self.main_thread:
            with super().measure(phase):
                yield
            return
        self._sync()
        begin = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            with self.auxiliary_lock:
                self.auxiliary_events.append({
                    "phase": "auxiliary_" + phase, "seconds": time.perf_counter() - begin,
                    "scope": "Overlapping auxiliary interval, not a disjoint episode partition"})

    def _sample(self, *args, **kwargs):
        result = super()._sample(*args, **kwargs)
        caches = self.llm.live_caches() if hasattr(self.llm, "live_caches") else [self.llm.past_key_values]
        if isinstance(caches, Mapping):
            caches = list(caches.values())
        tensors = [tensor for cache in caches if cache is not None
                   for pair in _cache_pairs(cache) for tensor in pair if tensor is not None]
        result["all_live_kv"] = _tensor_group(tensors)
        result["all_live_kv"]["scope"] = (
            "All observed main/auxiliary KV payload; storage_bytes deduplicates aliases. "
            "Allocator above model includes workspaces, metadata and unobserved temporary tensors.")
        return result

    def snapshot(self):
        result = super().snapshot()
        with self.auxiliary_lock:
            result["auxiliary_events"] = list(self.auxiliary_events)
        result["scope"]["auxiliary_intervals"] = "Overlapping spans; never sum with main exclusive timing"
        return result
