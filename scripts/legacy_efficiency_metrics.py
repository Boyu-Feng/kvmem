"""Optional diagnostic instrumentation for the unchanged legacy KV evaluator.

No model forward, tokenization, RNG call, cache mutation, or allocator reset is
added. Hooks observe tensor metadata; wrappers call the original methods once
with unchanged arguments. Use a separate clean pass for headline latency.

KV bytes mean K/V tensor payload (numel * element_size), not allocated storage,
allocator reservations, all live Python references, or total GPU memory. In
particular, the legacy runner's prompt/generated clones are outside active KV.
"""
from __future__ import annotations

import copy
import functools
import time
from collections import defaultdict

import torch

MIB = 1024 ** 2


def cache_tensor_pairs(cache):
    """Return materialized per-layer (K, V) tensors without copying them."""
    if cache is None:
        return []
    if hasattr(cache, "layers"):
        pairs = [(getattr(layer, "keys", None), getattr(layer, "values", None))
                 for layer in cache.layers]
    elif hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        pairs = list(zip(cache.key_cache, cache.value_cache))
    else:
        pairs = list(cache)
    return [(keys, values) for keys, values in pairs
            if isinstance(keys, torch.Tensor) and isinstance(values, torch.Tensor)]


class LegacyEfficiencyMonitor:
    """Attach to one loaded LLM; call close() to restore attributes and hooks.

    synchronize=True synchronizes model CUDA devices at whole LLM-call
    boundaries. Component intervals remain host-inclusive unless
    synchronize_components=True, which perturbs execution further. Nested
    intervals must never be added as a disjoint latency decomposition.
    generate_first() starts a fresh episode of measurements automatically.
    """

    def __init__(self, llm, *, synchronize=True, synchronize_components=False,
                 event_limit=256, prompt_length=None):
        self.llm = llm
        self.synchronize = bool(synchronize)
        self.synchronize_components = bool(synchronize_components)
        self.event_limit = max(0, int(event_limit))
        self._explicit_prompt_length = prompt_length
        self._patches = []
        self._hooks = []
        self._devices = set()
        self._closed = False
        for parameter in llm.model.parameters():
            if parameter.device.type == "cuda":
                self._devices.add(parameter.device.index)
        self.reset()
        self._install()

    def reset(self):
        """Reset measurements only; does not reset the model/cache/allocator."""
        self.counts = defaultdict(int)
        self.component_timings = {}
        self.compute_workload = {}
        self.steps = []
        self.pruning_events = []
        self.events = []
        self.instrumentation_errors = []
        self._event_count = 0
        self._call_name = None
        self._decode_depth = 0
        self._probe_depth = 0
        self._first_forward_seen = False
        self._forward_phases = []
        self._step_peak = None
        self._step_preprune_peak = None
        self.kv = {key: None for key in (
            "current", "peak_active", "peak_trajectory", "peak_preprune",
            "peak_post_step", "peak_probe")}
        self.kv["peak_active_plus_probe_bytes"] = 0
        manager = getattr(self.llm, "kv_manager", None)
        configured = int(getattr(manager, "protected_prefix_len", 0) or 0)
        self.prompt_length = (int(self._explicit_prompt_length)
                              if self._explicit_prompt_length is not None else configured)
        self.prompt_length_source = ("explicit" if self._explicit_prompt_length is not None
                                     else "manager_protected_prefix" if configured else "unknown")

    def _error(self, site, error):
        self.instrumentation_errors.append(f"{site}: {type(error).__name__}: {error}")

    def _sync(self):
        # CPU tests never touch any GPU, even when unrelated GPUs are available.
        for device in sorted(self._devices):
            torch.cuda.synchronize(device)

    def _patch(self, obj, name, wrapper_factory):
        if not hasattr(obj, name):
            return
        own = name in obj.__dict__
        prior = obj.__dict__.get(name)
        original = getattr(obj, name)
        wrapper = wrapper_factory(original)
        self._patches.append((obj, name, own, prior))
        setattr(obj, name, wrapper)

    @staticmethod
    def _layers(cache):
        return cache_tensor_pairs(cache)

    def _inspect(self, cache, site):
        manager = getattr(self.llm, "kv_manager", None)
        prefix = int(getattr(manager, "protected_prefix_len", 0) or 0)
        if prefix > 0:
            self.prompt_length = prefix
            self.prompt_length_source = "manager_protected_prefix"
        # The original manager protects a contiguous local prefix. FullKV has
        # no manager: its initial-prefill length provides the same diagnostic
        # prompt/trajectory partition, without claiming a pruning policy.
        protect = bool(getattr(manager, "protect_prompt", True))
        prefix = self.prompt_length if protect else 0
        lengths, protected_lengths, total, protected = [], [], 0, 0
        for keys, values in self._layers(cache):
            if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
                continue
            n = int(keys.shape[-2])
            if int(values.shape[-2]) != n:
                raise ValueError("Key/value sequence lengths disagree")
            p = min(max(0, prefix), n)
            size = keys.numel() * keys.element_size() + values.numel() * values.element_size()
            total += size
            protected += (size // n) * p if n else 0
            lengths.append(n)
            protected_lengths.append(p)
            for tensor in (keys, values):
                if tensor.device.type == "cuda":
                    self._devices.add(tensor.device.index)
        tracker = getattr(self.llm, "token_tracker", None)
        mapper = getattr(tracker, "global_id_mapper", None)
        uniform = not lengths or len(set(lengths)) == 1
        tokens = lengths[0] if lengths and uniform else 0 if not lengths else None
        p_tokens = protected_lengths[0] if protected_lengths and uniform else 0 if not lengths else None
        return dict(site=site, tokens=tokens, layer_tokens=lengths,
                    total_bytes=total, total_mib=total / MIB,
                    protected_tokens=p_tokens,
                    trajectory_tokens=tokens - p_tokens if tokens is not None else None,
                    protected_bytes=protected, protected_mib=protected / MIB,
                    trajectory_bytes=total - protected, trajectory_mib=(total - protected) / MIB,
                    prompt_length=self.prompt_length, prompt_length_source=self.prompt_length_source,
                    partition_basis=("manager_local_protected_prefix" if manager is not None
                                     else "initial_prompt_prefix_diagnostic_only"),
                    mapper_length=len(mapper) if mapper is not None else None,
                    mapper_matches_resident_length=(len(mapper) == tokens
                                                   if mapper is not None and tokens is not None else None),
                    bookkeeping_tokens=getattr(self.llm, "current_cache_len", None))

    @staticmethod
    def _maximum(prior, sample, key="total_bytes"):
        return sample.copy() if prior is None or sample[key] > prior[key] else prior

    def _sample(self, cache, site, *, probe=False, preprune=False, poststep=False):
        try:
            sample = self._inspect(cache, site)
            self._event_count += 1
            if len(self.events) < self.event_limit:
                self.events.append({**sample, "kind": "probe" if probe else "active"})
            if probe:
                self.kv["peak_probe"] = self._maximum(self.kv["peak_probe"], sample)
                active = self._inspect(getattr(self.llm, "past_key_values", None), "active_during_probe")
                self.kv["peak_active_plus_probe_bytes"] = max(
                    self.kv["peak_active_plus_probe_bytes"], active["total_bytes"] + sample["total_bytes"])
            else:
                self.kv["current"] = sample
                self.kv["peak_active"] = self._maximum(self.kv["peak_active"], sample)
                self.kv["peak_trajectory"] = self._maximum(self.kv["peak_trajectory"], sample, "trajectory_bytes")
                self._step_peak = self._maximum(self._step_peak, sample)
                if preprune:
                    self.kv["peak_preprune"] = self._maximum(self.kv["peak_preprune"], sample)
                    self._step_preprune_peak = self._maximum(self._step_preprune_peak, sample)
                if poststep:
                    self.kv["peak_post_step"] = self._maximum(self.kv["peak_post_step"], sample)
            return sample
        except Exception as error:
            self._error(site, error)
            return None

    def _active(self, site, **kwargs):
        return self._sample(getattr(self.llm, "past_key_values", None), site, **kwargs)

    def _timing(self, name, elapsed):
        entry = self.component_timings.setdefault(name, {"calls": 0, "inclusive_seconds": 0.0})
        entry["calls"] += 1
        entry["inclusive_seconds"] += elapsed

    def _method_wrapper(self, name, counter=None, *, probe=False, prune=False, decode=False, crop=False):
        def factory(original):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                if counter:
                    self.counts[counter] += 1
                before = self._active(name + ":before", preprune=prune) if prune or crop or probe else None
                if probe:
                    self._probe_depth += 1
                if decode:
                    self._decode_depth += 1
                if self.synchronize_components:
                    self._sync()
                start = time.perf_counter()
                succeeded = False
                try:
                    result = original(*args, **kwargs)
                    succeeded = True
                    return result
                finally:
                    if self.synchronize_components:
                        self._sync()
                    self._timing(name, time.perf_counter() - start)
                    if probe:
                        self._probe_depth -= 1
                    if decode:
                        self._decode_depth -= 1
                    after = self._active(name + ":after") if prune or crop or probe else None
                    if prune:
                        removed = (before["tokens"] - after["tokens"]
                                   if before and after and before["tokens"] is not None and after["tokens"] is not None else None)
                        self.pruning_events.append(dict(method=name, before=before, after=after,
                            actual_tokens_removed=removed, succeeded=succeeded,
                            single_token_mode=bool(kwargs.get("single_token_mode", False))))
            return wrapped
        return factory

    def _call_wrapper(self, name):
        def factory(original):
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                if name == "generate_first":
                    self.reset()
                    # First prefill will provide the prompt length before its
                    # output KV is sampled, including for FullKV/no manager.
                    if self._explicit_prompt_length is None:
                        self.prompt_length = 0
                        self.prompt_length_source = "unknown"
                previous = self._call_name
                self._call_name = name
                self._step_peak = None
                self._step_preprune_peak = None
                self.counts["llm_calls"] += 1
                # generate_first resets the LLM internally. Its pre-reset cache
                # may belong to a clean pass or previous question, so it must
                # not contaminate this new episode's active-cache peaks.
                if name != "generate_first":
                    self._active(name + ":before")
                if self.synchronize:
                    self._sync()
                start = time.perf_counter()
                succeeded = False
                try:
                    result = original(*args, **kwargs)
                    succeeded = True
                    return result
                finally:
                    if self.synchronize:
                        self._sync()
                    elapsed = time.perf_counter() - start
                    final = self._active(name + ":after", poststep=True)
                    self.steps.append(dict(method=name, elapsed_s=elapsed, succeeded=succeeded,
                        boundary=final, peak_active=self._step_peak,
                        peak_preprune=self._step_preprune_peak))
                    self._timing(name, elapsed)
                    self._call_name = previous
            return wrapped
        return factory

    def _before_forward(self, module, args, kwargs):
        try:
            probe = self._probe_depth > 0
            ids = kwargs.get("input_ids", args[0] if args else None)
            if self._call_name == "generate_first" and not self._first_forward_seen and not probe:
                if ids is not None:
                    self.prompt_length = int(ids.shape[1])
                    self.prompt_length_source = "first_initial_prefill_input_length"
                self._first_forward_seen = True
                phase = "initial_prefill"
            else:
                phase = "eos_probe" if probe else "decode" if self._decode_depth or self._call_name == "generate_first" else "prefill"
            self._forward_phases.append((phase, probe))
            self.counts["model_forwards"] += 1
            self.counts["eos_probe_forwards" if probe else "active_model_forwards"] += 1
            self.counts[phase + "_forwards"] += 1 if phase not in ("eos_probe",) else 0
            positions = kwargs.get("inputs_embeds") if ids is None else ids
            if positions is not None and positions.ndim >= 2:
                batch, sequence = int(positions.shape[0]), int(positions.shape[1])
                work = self.compute_workload.setdefault(phase, dict(
                    forward_calls=0, input_positions=0, sequence_length_sum=0,
                    batch_size_sum=0, max_batch_size=0, max_sequence_length=0))
                work["forward_calls"] += 1
                work["input_positions"] += batch * sequence
                work["sequence_length_sum"] += sequence
                work["batch_size_sum"] += batch
                work["max_batch_size"] = max(work["max_batch_size"], batch)
                work["max_sequence_length"] = max(work["max_sequence_length"], sequence)
                self.counts["probe_input_positions" if probe else "active_input_positions"] += batch * sequence
            cache = kwargs.get("past_key_values")
            self._sample(cache, "forward:" + phase + ":before", probe=probe)
        except Exception as error:
            self._error("before_forward", error)

    def _after_forward(self, module, args, kwargs, output):
        try:
            phase, probe = self._forward_phases.pop() if self._forward_phases else ("unknown", self._probe_depth > 0)
            cache = getattr(output, "past_key_values", None)
            if cache is None and isinstance(output, dict):
                cache = output.get("past_key_values")
            if cache is not None:
                self._sample(cache, "forward:" + phase + ":after", probe=probe)
        except Exception as error:
            self._error("after_forward", error)

    def _install(self):
        for name in ("generate_first", "generate_incremental", "generate_incremental_with_memory"):
            self._patch(self.llm, name, self._call_wrapper(name))
        specs = (
            ("_do_pruning", "pruning_calls", {"prune": True}),
            ("_get_attention_for_scoring", "eos_probe_calls", {"probe": True}),
            ("_resolve_scoring_attentions", "score_resolution_calls", {}),
            ("truncate_cache", "crop_calls", {"crop": True}),
            ("_decode", "decode_calls", {"decode": True}),
            ("_decode_token_by_token_with_pruning", "manual_decode_calls", {"decode": True}),
        )
        for name, counter, flags in specs:
            self._patch(self.llm, name, self._method_wrapper(name, counter, **flags))
        manager = getattr(self.llm, "kv_manager", None)
        strategy = getattr(manager, "pruning_strategy", None)
        if strategy is not None:
            self._patch(strategy, "prune", self._method_wrapper("selector.prune", "selector_calls"))
            scorer = getattr(strategy, "h2o_scorer", None)
            if scorer is not None:
                self._patch(scorer, "compute_scores", self._method_wrapper("scorer.compute_scores", "attention_score_calls"))
        self._hooks.append(self.llm.model.register_forward_pre_hook(self._before_forward, with_kwargs=True))
        self._hooks.append(self.llm.model.register_forward_hook(self._after_forward, with_kwargs=True, always_call=True))

    def snapshot(self):
        self._active("snapshot")
        allocator = {}
        for device in sorted(self._devices):
            allocator[str(device)] = dict(
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
        keys = ("llm_calls", "model_forwards", "active_model_forwards", "eos_probe_forwards",
                "eos_probe_calls", "pruning_calls", "selector_calls", "attention_score_calls",
                "score_resolution_calls", "crop_calls")
        counts = {**{key: 0 for key in keys}, **self.counts}
        return copy.deepcopy(dict(schema_version=1, counts=counts, kv=self.kv,
            prompt_length=self.prompt_length, prompt_length_source=self.prompt_length_source,
            steps=self.steps, pruning_events=self.pruning_events,
            compute_workload=self.compute_workload,
            component_timings=self.component_timings, events=self.events,
            observed_samples=self._event_count, omitted_events=max(0, self._event_count - len(self.events)),
            instrumentation_errors=self.instrumentation_errors, allocator_peaks=allocator,
            timing_scope={"llm_calls_cuda_synchronized": self.synchronize,
                "components_cuda_synchronized": self.synchronize_components,
                "components_are_inclusive_and_overlap": True,
                "headline_latency": "measure a separate clean episode outside this instrumentor"},
            memory_scope={"active": "actual active/normal-forward-output K/V tensor payload; all layers; prompt included",
                "partition": "current manager protected local-prefix boundary; FullKV uses initial-prefill prompt length",
                "probe": "separate EOS-probe input/output cache; excluded from active KV peaks",
                "excluded": "other Python-held KV clones, backing storage beyond tensor view, activations, attention matrices, allocator overhead",
                "allocator": "process CUDA peaks since last external reset; this module never resets them",
                "sampling": "model-forward input/output, prune/crop input/output and LLM-call boundaries; no internal operator peak"},
            count_scope={"pruning_calls": "_do_pruning callbacks, including no-op/failed callbacks",
                "selector_calls": "original pruning_strategy.prune invocations",
                "attention_score_calls": "H2OScorer.compute_scores; TOVA scores inline in its selector",
                "eos_probe_calls": "_get_attention_for_scoring calls; corresponding forward counted separately",
                "compute_workload": "observed input positions batch*sequence per forward phase, including later stop-discarded work; not retained/text tokens or FLOPs"}))

    def close(self):
        if self._closed:
            return
        for hook in reversed(self._hooks):
            hook.remove()
        for obj, name, own, prior in reversed(self._patches):
            if own:
                setattr(obj, name, prior)
            else:
                delattr(obj, name)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def attach_legacy_efficiency(llm, **kwargs):
    return LegacyEfficiencyMonitor(llm, **kwargs)


class LegacyEfficiencyInstallation:
    """Temporary constructor patch for an external unmodified-evaluator wrapper."""
    def __init__(self, llm_class, **kwargs):
        self.llm_class = llm_class
        self.monitors = []
        self._own_init = "__init__" in llm_class.__dict__
        self._prior_init = llm_class.__dict__.get("__init__")
        self._original_init = llm_class.__init__
        self._closed = False
        original = self._original_init
        @functools.wraps(original)
        def initialize(instance, *args, **init_kwargs):
            original(instance, *args, **init_kwargs)
            self.monitors.append(attach_legacy_efficiency(instance, **kwargs))
        llm_class.__init__ = initialize

    def monitor_for(self, llm):
        return next((monitor for monitor in self.monitors if monitor.llm is llm), None)

    def close(self):
        if self._closed:
            return
        for monitor in reversed(self.monitors):
            monitor.close()
        if self._own_init:
            self.llm_class.__init__ = self._prior_init
        else:
            delattr(self.llm_class, "__init__")
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def install_legacy_efficiency(llm_class, **kwargs):
    return LegacyEfficiencyInstallation(llm_class, **kwargs)
