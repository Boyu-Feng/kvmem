"""Observational latency diagnostics; never use these timings as headline latency.

Every measured phase synchronizes the model's actual CUDA devices. This perturbs
execution: headline latency must come from a separate, uninstrumented clean pass.
Exclusive times subtract nested intervals; inclusive times must not be summed.
No forward, tokenization, tensor copy, RNG change, or empty_cache call is added.

KV payload, unique backing storage, recurrence metadata, returned snapshots and
the allocator are different measurements. Allocated-above-model includes *all*
nonmodel allocations (copies, activations, attention, workspaces and metadata),
not just KV or pure algorithm overhead. Peaks are sampled KV peaks, not a claim
to observe every short-lived tensor. CUDA allocator peaks also catch allocations
between samples. CPU models never access the CUDA API.
"""
from __future__ import annotations

import copy
import functools
import math
import time
from contextlib import contextmanager

import torch


PHASES = (
    "initial_prefill", "observation_prefill", "decode_forward", "scoring_forward",
    "token_saliency", "selection_and_cache_management", "cache_gather",
    "stop_crop", "generation_other", "step_utility", "tool",
)


def _tensor_metadata(tensor):
    storage = tensor.untyped_storage()
    return dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                device=str(tensor.device),
                bytes=int(tensor.numel() * tensor.element_size()),
                storage_bytes=int(storage.nbytes()))


def _storage_key(tensor):
    storage = tensor.untyped_storage()
    return str(tensor.device), storage.data_ptr(), storage.nbytes()


def model_storage_bytes(model) -> dict[str, int]:
    """Unique parameter *and buffer* backing-storage bytes per actual device.

    This metadata-only helper is also suitable for an uninstrumented clean pass.
    It neither copies tensors nor touches CUDA APIs. Re-read after the episode
    to observe any growing model buffers; payload numel is not storage size.
    """
    storages = {}
    for tensor in (*model.parameters(), *model.buffers()):
        storages[_storage_key(tensor)] = int(tensor.untyped_storage().nbytes())
    totals = {}
    for (device, _, _), size in storages.items():
        totals[device] = totals.get(device, 0) + size
    return totals


def _tensor_group(tensors):
    layout, storages = [], {}
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Expected a tensor in observed storage")
        layout.append(_tensor_metadata(tensor))
        storages[_storage_key(tensor)] = int(tensor.untyped_storage().nbytes())
    return dict(total_bytes=sum(t["bytes"] for t in layout),
                storage_bytes=sum(storages.values()), layout=layout)


def _cache_pairs(cache):
    """Keep missing layers explicit (the legacy helper filters them out)."""
    if cache is None:
        return []
    if hasattr(cache, "layers"):
        return [(getattr(layer, "keys", None), getattr(layer, "values", None))
                for layer in cache.layers]
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        if len(cache.key_cache) != len(cache.value_cache):
            raise ValueError("Cache key/value layer counts disagree")
        return list(zip(cache.key_cache, cache.value_cache))
    pairs = list(cache)
    if any(not isinstance(pair, (tuple, list)) or len(pair) != 2 for pair in pairs):
        raise ValueError("Cache must contain per-layer (key, value) tensor pairs")
    return pairs


class DetailedLatencyMonitor:
    """Attach to an already loaded LLM; context exit/close restores all wrappers.

    generate_first clears diagnostics before calling the original exactly once.
    The original reset must complete before CUDA peak counters are reset. A
    generation boundary is captured while its returned clones are still live;
    snapshot() samples a distinct final boundary when the caller/episode returns.

    wrap(obj, name, phase) also supports module functions, e.g. LazyEviction's
    update_recurrence, _select_at_boundary and gather_state. Repeated installation
    with the same phase is a no-op. It does not rewrite Tensor methods: inline
    KV gathers remain in the enclosing selection or model-forward remainder.
    """

    def __init__(self, llm, *, process_memory_reader=None, event_limit=0):
        if process_memory_reader is not None and not callable(process_memory_reader):
            raise TypeError("process_memory_reader must be callable")
        if int(event_limit) < 0:
            raise ValueError("event_limit must be nonnegative")
        self.llm = llm
        self.process_memory_reader = process_memory_reader
        self.event_limit = int(event_limit)
        self._patches = []
        self._wrapped = {}
        self._closed = False
        self._stack = []
        self._devices = set()
        self._call_name = None
        self._decode_depth = 0
        self._probe_depth = 0
        self._forward_count = 0
        self._allocator_ready = False
        self._baseline = self._model_storage()
        self.reset_episode()
        try:
            self._install()
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Cannot re-enter a closed monitor")
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is None:
                self._final = self._sample("episode_return", read_process_memory=True)
        finally:
            self.close()
        return False

    def reset_episode(self):
        """Reset diagnostics, not model state or allocator counters."""
        if self._stack:
            raise RuntimeError("Cannot start a new episode inside measure()")
        self.exclusive_seconds = dict.fromkeys(PHASES, 0.0)
        self.inclusive_seconds = dict.fromkeys(PHASES, 0.0)
        self.calls = dict.fromkeys(PHASES, 0)
        self.root_seconds = 0.0
        self.workload = {}
        self.steps = []
        self.events = []
        self.pruning_events = []
        self.instrumentation_errors = []
        self.failed_calls = []
        self._event_count = 0
        self._episode_peak_kv = None
        self._probe_peak_kv = None
        self._step_peak_kv = None
        self._step_preprune_peak = None
        self._episode_allocator_peaks = {}
        self._step_allocator_peaks = {}
        self._allocator_ready = False
        self._final = None
        self.prompt_length = 0
        self.prompt_length_source = "unknown"

    def _sync(self):
        for device in sorted(self._devices):
            torch.cuda.synchronize(torch.device(device))

    @contextmanager
    def measure(self, phase):
        """Measure a host/CUDA-synchronized interval, disjoint from its children."""
        if self._closed:
            raise RuntimeError("Cannot measure with a closed monitor")
        if not isinstance(phase, str) or not phase:
            raise ValueError("phase must be a nonempty string")
        try:
            self._sync()
            start = time.perf_counter()
            if not math.isfinite(start):
                raise ValueError("Nonfinite latency clock")
        except Exception as error:
            self.instrumentation_errors.append(f"measure:start: {type(error).__name__}: {error}")
            self.close()
            raise
        frame = dict(phase=phase, start=start, children=0.0)
        self._stack.append(frame)
        try:
            yield
        except BaseException as error:
            self.failed_calls.append(dict(phase=phase, error_type=type(error).__name__,
                                          message=str(error)))
            self.close()
            raise
        finally:
            try:
                self._sync()
                elapsed = time.perf_counter() - start
                exclusive = elapsed - frame["children"]
                if (not math.isfinite(elapsed) or elapsed < 0
                        or not math.isfinite(exclusive) or exclusive < -1e-9):
                    raise ValueError("Nonfinite, backwards or inconsistent latency clock")
                self.inclusive_seconds[phase] = self.inclusive_seconds.get(phase, 0.0) + elapsed
                self.exclusive_seconds[phase] = self.exclusive_seconds.get(phase, 0.0) + max(0.0, exclusive)
                self.calls[phase] = self.calls.get(phase, 0) + 1
            except Exception as error:
                self.instrumentation_errors.append(f"measure:end: {type(error).__name__}: {error}")
                self.close()
                raise
            finally:
                popped = self._stack.pop()
                if popped is not frame:
                    raise RuntimeError("Latency interval nesting corrupted")
            if self._stack:
                self._stack[-1]["children"] += elapsed
            else:
                self.root_seconds += elapsed

    def _model_storage(self):
        totals = model_storage_bytes(self.llm.model)
        self._devices.update(device for device in totals if torch.device(device).type == "cuda")
        return totals

    @staticmethod
    def _maximum(previous, sample):
        return sample if previous is None or sample["total_bytes"] > previous["total_bytes"] else previous

    def _prefix(self):
        manager = getattr(self.llm, "kv_manager", None)
        lazy = getattr(self.llm, "_legacy_prompt_length", 0)
        configured = getattr(manager, "protected_prefix_len", 0) if manager is not None else 0
        if lazy:
            self.prompt_length = int(lazy)
            self.prompt_length_source = "lazy_initial_prompt"
        elif configured:
            self.prompt_length = int(configured)
            self.prompt_length_source = "manager_protected_prefix"
        protect = getattr(manager, "protect_prompt", getattr(self.llm, "kv_config", {}).get("protect_prompt", True))
        return self.prompt_length if protect else 0

    def _kv(self, cache):
        pairs = _cache_pairs(cache)
        config = self.llm.model.config
        expected = int(getattr(config, "num_hidden_layers", len(pairs)))
        native = getattr(config, "num_key_value_heads", None)
        native = int(native) if native is not None else None
        if native is not None and native <= 0:
            raise ValueError("Invalid native KV-head count")
        prefix = self._prefix()
        layout, lengths, protected_lengths, storages, missing = [], [], [], {}, []
        total, protected = 0, 0
        for layer, (keys, values) in enumerate(pairs):
            if keys is None and values is None:
                missing.append(layer)
                lengths.append(None)
                protected_lengths.append(None)
                continue
            if not isinstance(keys, torch.Tensor) or not isinstance(values, torch.Tensor):
                raise TypeError(f"Layer {layer} has a missing/non-tensor key or value")
            if keys.ndim != 4 or values.ndim != 4 or keys.shape[0] != 1 or values.shape[0] != 1:
                raise ValueError(f"Layer {layer} must have batch-one 4D KV tensors")
            if keys.shape[:3] != values.shape[:3]:
                raise ValueError(f"Layer {layer} key/value batch, head or sequence shapes disagree")
            n, heads = int(keys.shape[2]), int(keys.shape[1])
            p = min(max(prefix, 0), n)
            km, vm = _tensor_metadata(keys), _tensor_metadata(values)
            size = km["bytes"] + vm["bytes"]
            total += size
            protected += (size // n) * p if n else 0
            lengths.append(n)
            protected_lengths.append(p)
            layout.append(dict(layer=layer, key_shape=km["shape"], value_shape=vm["shape"],
                               key_dtype=km["dtype"], value_dtype=vm["dtype"],
                               key_device=km["device"], value_device=vm["device"],
                               key_bytes=km["bytes"], value_bytes=vm["bytes"],
                               key_storage_bytes=km["storage_bytes"], value_storage_bytes=vm["storage_bytes"],
                               actual_heads=heads, native_heads=native,
                               head_expansion_factor=heads / native if native else None))
            for tensor in (keys, values):
                storages[_storage_key(tensor)] = int(tensor.untyped_storage().nbytes())
                if tensor.device.type == "cuda":
                    self._devices.add(str(tensor.device))
        if cache is not None:
            for layer in range(len(pairs), expected):
                missing.append(layer)
                lengths.append(None)
                protected_lengths.append(None)
        uniform = not missing and len(set(lengths)) <= 1
        tokens = (lengths[0] if lengths else 0) if uniform else None
        p_tokens = (protected_lengths[0] if protected_lengths else 0) if uniform else None
        factors = [layer["head_expansion_factor"] for layer in layout]
        per_head = hasattr(self.llm, "_legacy_recurrence")
        return dict(tokens_per_head=tokens, layer_tokens=lengths,
                    total_bytes=total, storage_bytes=sum(storages.values()),
                    protected_tokens=p_tokens,
                    protected_bytes=protected, trajectory_tokens=tokens - p_tokens if uniform else None,
                    trajectory_bytes=total - protected, layout=layout,
                    complete=not missing, missing_layers=missing,
                    materialized_layers=len(layout), expected_layers=expected,
                    token_scope=("physical slots per head; retained IDs differ across heads/layers, not shared logical tokens"
                                 if per_head else "shared physical slots across heads (not logical tokens ever seen)"),
                    partition_basis=("protected_prompt_prefix; recent-window tokens remain trajectory"
                                     if getattr(self.llm, "kv_manager", None) is not None or per_head
                                     else "initial_prompt_prefix_diagnostic_only"),
                    head_expansion=dict(native_heads=native,
                                        actual_heads_per_layer=[layer["actual_heads"] for layer in layout],
                                        factors_per_layer=factors,
                                        uniform_factor=factors[0] if factors and len(set(factors)) == 1 else None),
                    bookkeeping_tokens=getattr(self.llm, "current_cache_len", None))

    def _allocator(self, *, read_process_memory=False):
        current = self._model_storage()
        driver = (self.process_memory_reader()
                  if read_process_memory and self._devices and self.process_memory_reader is not None else {})
        if not isinstance(driver, dict):
            raise TypeError("process_memory_reader must return a device->integer bytes dict")
        for device, used in driver.items():
            if not isinstance(device, str) or not isinstance(used, int) or used < 0:
                raise ValueError("Invalid driver process memory sample")
        result = {}
        for device in sorted(self._devices):
            allocated = int(torch.cuda.memory_allocated(device))
            reserved = int(torch.cuda.memory_reserved(device))
            peak_a = int(torch.cuda.max_memory_allocated(device))
            peak_r = int(torch.cuda.max_memory_reserved(device))
            model = current.get(device, 0)
            entry = dict(allocated_bytes=allocated, reserved_bytes=reserved,
                         peak_allocated_bytes=peak_a, peak_reserved_bytes=peak_r,
                         model_storage_bytes=model,
                         model_storage_baseline_bytes=self._baseline.get(device, 0),
                         allocated_above_model_bytes=max(0, allocated - model),
                         reserved_above_model_bytes=max(0, reserved - model),
                         peak_allocated_above_model_bytes=max(0, peak_a - model),
                         peak_reserved_above_model_bytes=max(0, peak_r - model),
                         peak_counters_episode_initialized=self._allocator_ready)
            if device in driver:
                entry["driver_process_used_bytes"] = driver[device]
                entry["driver_above_model_bytes"] = max(0, driver[device] - model)
            result[device] = entry
            if self._allocator_ready:
                for peaks in (self._episode_allocator_peaks, self._step_allocator_peaks):
                    previous = peaks.setdefault(device, dict(peak_allocated_bytes=0, peak_reserved_bytes=0))
                    previous["peak_allocated_bytes"] = max(previous["peak_allocated_bytes"], peak_a)
                    previous["peak_reserved_bytes"] = max(previous["peak_reserved_bytes"], peak_r)
        self._current_model_storage = current
        return result

    def _reset_allocator_peaks(self):
        for device in sorted(self._devices):
            torch.cuda.reset_peak_memory_stats(device)

    def _after_reset(self):
        self._baseline = self._model_storage()
        self._reset_allocator_peaks()
        self._allocator_ready = True

    def _recurrence(self):
        states = getattr(self.llm, "_legacy_recurrence", {})
        tensors, layers = [], []
        for layer, state in sorted(states.items()):
            group = _tensor_group([state.ids, state.last, state.mri])
            tensors.extend([state.ids, state.last, state.mri])
            layers.append(dict(layer=int(layer), **group,
                               fields=["ids", "last", "mri"]))
        return dict(**_tensor_group(tensors), layers=layers,
                    scope="per-layer/per-head recurrence; IDs are not shared logical tokens")

    def _returned(self, result):
        if not isinstance(result, (tuple, list)) or len(result) != 3:
            return None
        tensors, blocks = [], []
        for index in (1, 2):
            block_tensors = [t for pair in _cache_pairs(result[index]) for t in pair]
            group = _tensor_group(block_tensors)
            tensors.extend(block_tensors)
            blocks.append(dict(path=f"result[{index}]", **group))
        return dict(**_tensor_group(tensors), blocks=blocks,
                    scope="returned KV snapshots/clones, separate from active KV payload")

    def _sample(self, site, *, cache=None, explicit_cache=False, probe=False, returned=None,
                read_process_memory=False):
        try:
            if not explicit_cache:
                cache = getattr(self.llm, "_legacy_active_cache", None)
                if cache is None:
                    cache = getattr(self.llm, "past_key_values", None)
            kv = self._kv(cache)
            allocator = self._allocator(read_process_memory=read_process_memory)
            exported = getattr(self.llm, "_legacy_exported", ())
            memory = dict(site=site, kv=kv, allocator=allocator,
                          driver_process_sampled=bool(read_process_memory and self._devices
                                                      and self.process_memory_reader is not None),
                          recurrence=self._recurrence(),
                          owner_returned_snapshots=_tensor_group([t for pair in _cache_pairs(exported) for t in pair]),
                          prompt_length=self.prompt_length,
                          prompt_length_source=self.prompt_length_source)
            if returned is not None:
                memory["returned_snapshots"] = self._returned(returned)
            if probe:
                self._probe_peak_kv = self._maximum(self._probe_peak_kv, kv)
            else:
                self._episode_peak_kv = self._maximum(self._episode_peak_kv, kv)
                self._step_peak_kv = self._maximum(self._step_peak_kv, kv)
            self._event_count += 1
            if len(self.events) < self.event_limit:
                self.events.append(dict(kind="probe" if probe else "active", memory=memory))
            return memory
        except Exception as error:
            self.instrumentation_errors.append(f"{site}: {type(error).__name__}: {error}")
            self.close()
            raise

    def _timing_copy(self):
        return dict(exclusive_seconds=dict(self.exclusive_seconds),
                    inclusive_seconds=dict(self.inclusive_seconds), calls=dict(self.calls),
                    root_seconds=self.root_seconds)

    def _timing_diff(self, before):
        current = self._timing_copy()
        result = {key: {phase: value - before[key].get(phase, 0)
                        for phase, value in current[key].items()}
                  for key in ("exclusive_seconds", "inclusive_seconds", "calls")}
        result["root_seconds"] = current["root_seconds"] - before["root_seconds"]
        return result

    def _generation(self, original, name, args, kwargs):
        if name == "generate_first":
            self.reset_episode()
        elif self._allocator_ready:
            self._allocator()
            self._reset_allocator_peaks()
        previous = self._call_name
        self._call_name = name
        self._forward_count = 0
        self._step_peak_kv = None
        self._step_preprune_peak = None
        self._step_allocator_peaks = {}
        timing_before = self._timing_copy()
        prune_begin = len(self.pruning_events)
        boundary = None
        succeeded = False
        try:
            with self.measure("generation_other"):
                result = original(*args, **kwargs)
                # This is BEFORE the caller can discard returned KV clones.
                boundary = self._sample(name + ":return", returned=result,
                                        read_process_memory=True)
                succeeded = True
            return result
        finally:
            self.steps.append(dict(generation_call=name, succeeded=succeeded,
                                   memory=boundary, timing=self._timing_diff(timing_before),
                                   peak_active_kv=self._step_peak_kv,
                                   peak_preprune_kv=self._step_preprune_peak,
                                   allocator_peak=copy.deepcopy(self._step_allocator_peaks),
                                   pruning_events=copy.deepcopy(self.pruning_events[prune_begin:])))
            self._call_name = previous

    def _forward(self, original, args, kwargs):
        if self._probe_depth:
            phase = "scoring_forward"
        elif self._decode_depth or (self._call_name == "generate_first" and self._forward_count):
            phase = "decode_forward"
        elif self._call_name == "generate_first":
            phase = "initial_prefill"
        elif self._call_name is not None:
            phase = "observation_prefill" if not self._forward_count else "decode_forward"
        else:
            phase = "unscoped_forward"
        ids = kwargs.get("input_ids", args[0] if args else None)
        inputs = kwargs.get("inputs_embeds") if ids is None else ids
        if not isinstance(inputs, torch.Tensor) or inputs.ndim < 2 or inputs.shape[0] != 1:
            raise ValueError("Observed model forward requires batch-one input_ids/inputs_embeds")
        shape = list(inputs.shape)
        sequence = int(shape[1])
        if phase == "initial_prefill":
            self.prompt_length = sequence
            self.prompt_length_source = "actual_initial_prefill_query_length"
        if not self._probe_depth:
            self._forward_count += 1
        work = self.workload.setdefault(phase, dict(forward_calls=0, input_positions=0,
                                                  query_shapes=[], failed_forward_calls=0))
        work["forward_calls"] += 1
        work["input_positions"] += sequence
        found = next((entry for entry in work["query_shapes"] if entry["shape"] == shape), None)
        if found is None:
            found = dict(shape=shape, calls=0)
            work["query_shapes"].append(found)
        found["calls"] += 1
        self._sample("forward:" + phase + ":before",
                     cache=kwargs.get("past_key_values"), explicit_cache=True, probe=bool(self._probe_depth))
        try:
            with self.measure(phase):
                result = original(*args, **kwargs)
        except BaseException:
            work["failed_forward_calls"] += 1
            raise
        cache = result.get("past_key_values") if isinstance(result, dict) else getattr(result, "past_key_values", None)
        self._sample("forward:" + phase + ":after", cache=cache, explicit_cache=True,
                     probe=bool(self._probe_depth))
        return result

    def wrap(self, obj, name, phase):
        """Install a transparent once-only wrapper; returns the installed callable."""
        if self._closed:
            raise RuntimeError("Cannot wrap with a closed monitor")
        key = id(obj), name
        if key in self._wrapped:
            installed_phase, wrapper = self._wrapped[key]
            if installed_phase != phase:
                raise ValueError(f"{name} is already wrapped as {installed_phase}")
            return wrapper
        original = getattr(obj, name)
        if not callable(original):
            raise TypeError(f"{name} is not callable")
        own, prior = name in vars(obj), vars(obj).get(name)
        prune = name in ("_do_pruning", "_prune_prompt_prefix_once", "_freeze_completed_block", "_select_at_boundary")

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            try:
                if obj is self.llm and name in ("generate_first", "generate_incremental", "generate_incremental_with_memory"):
                    return self._generation(original, name, args, kwargs)
                if obj is self.llm.model and name == "forward":
                    return self._forward(original, args, kwargs)
                before = self._sample(name + ":before") if prune else None
                if before is not None:
                    self._step_preprune_peak = self._maximum(self._step_preprune_peak, before["kv"])
                if name == "_get_attention_for_scoring":
                    self._probe_depth += 1
                if name in ("_decode", "_decode_token_by_token_with_pruning"):
                    self._decode_depth += 1
                succeeded = False
                try:
                    with self.measure(phase):
                        result = original(*args, **kwargs)
                    if obj is self.llm and name == "reset":
                        self._after_reset()
                    succeeded = True
                    return result
                finally:
                    if name == "_get_attention_for_scoring":
                        self._probe_depth -= 1
                    if name in ("_decode", "_decode_token_by_token_with_pruning"):
                        self._decode_depth -= 1
                    if succeeded and (prune or name in ("truncate_cache", "_record_legacy_memory")):
                        after = self._sample(name + ":after")
                        if prune:
                            self.pruning_events.append(dict(method=name, before=before, after=after,
                                                            scope="wrapped selector boundary; inline lazy KV gather occurs later",
                                                            removed_payload_bytes=before["kv"]["total_bytes"] - after["kv"]["total_bytes"]))
            except BaseException:
                self.close()
                raise

        self._patches.append((obj, name, own, prior))
        setattr(obj, name, wrapped)
        self._wrapped[key] = phase, wrapped
        return wrapped

    def _install(self):
        for name in ("generate_first", "generate_incremental", "generate_incremental_with_memory",
                     "reset", "_decode", "_decode_token_by_token_with_pruning"):
            if hasattr(self.llm, name):
                self.wrap(self.llm, name, "generation_other")
        for name, phase in (
            ("_get_attention_for_scoring", "scoring_forward"),
            ("_resolve_scoring_attentions", "scoring_forward"),
            ("_do_pruning", "selection_and_cache_management"),
            ("_prune_prompt_prefix_once", "selection_and_cache_management"),
            ("_freeze_completed_block", "selection_and_cache_management"),
            ("_record_legacy_memory", "selection_and_cache_management"),
            ("truncate_cache", "stop_crop"),
        ):
            if hasattr(self.llm, name):
                self.wrap(self.llm, name, phase)
        manager = getattr(self.llm, "kv_manager", None)
        strategy = getattr(manager, "pruning_strategy", None)
        if strategy is not None:
            self.wrap(strategy, "prune", "selection_and_cache_management")
            scorer = getattr(strategy, "h2o_scorer", None)
            if scorer is not None:
                for name in ("compute_scores", "compute_attention_matrix"):
                    if hasattr(scorer, name):
                        self.wrap(scorer, name, "token_saliency")
        self.wrap(self.llm.model, "forward", "model_forward")

    def sample_memory(self, site="interaction_boundary"):
        """Capture a detached boundary sample without rebuilding the snapshot.

        This may be called during a measured interval, e.g. at a traced episode
        interaction endpoint while local snapshots are still live. It does not
        reset diagnostics or wrap/unwrap anything. Byte totals are explicit in
        kv.total_bytes and each observable tensor group's total_bytes; they must
        not be summed as disjoint storage (views/aliases can overlap).
        The optional process-memory reader runs at explicit/generation/final
        boundaries only, never at individual forward or pruning observations.
        """
        if not isinstance(site, str) or not site:
            raise ValueError("site must be a nonempty string")
        try:
            self._sync()
        except Exception as error:
            self.instrumentation_errors.append(f"{site}:sync: {type(error).__name__}: {error}")
            self.close()
            raise
        return copy.deepcopy(self._sample(site, read_process_memory=True))

    def snapshot(self):
        """JSON-compatible detached metadata, with a fresh post-caller sample."""
        if self._stack:
            raise RuntimeError("snapshot() requires completed measured intervals")
        final = self._sample("snapshot:final", read_process_memory=True)
        self._final = final
        policy = dict(recent_window=getattr(getattr(self.llm, "kv_manager", None), "observation_window", None),
                      lazy_window=getattr(self.llm, "_legacy_window", None),
                      lazy_query_clock=getattr(self.llm, "_legacy_clock", None),
                      lazy_effective_target=getattr(self.llm, "_legacy_forward_capacity", None))
        return copy.deepcopy(dict(
            steps=self.steps, final=final, timing=self._timing_copy(), workload=self.workload,
            peak_active_kv=self._episode_peak_kv, peak_probe_kv=self._probe_peak_kv,
            allocator_episode_peak=self._episode_allocator_peaks,
            model_storage=dict(baseline_by_device=self._baseline,
                               current_by_device=self._current_model_storage),
            pruning_events=self.pruning_events, events=self.events,
            sampled_events=self._event_count, omitted_events=max(0, self._event_count - len(self.events)),
            instrumentation_errors=self.instrumentation_errors, failed_calls=self.failed_calls,
            policy=policy,
            scope=dict(
                latency="Perturbed by per-phase CUDA synchronization and metadata inspection; headline requires a separate clean pass.",
                exclusive="Disjoint nested intervals; inclusive values must not be summed.",
                synchronization="Exit synchronization is inside its phase interval; child-entry synchronization belongs to the enclosing exclusive remainder. Root-entry synchronization and samples outside intervals are diagnostic residual. Metadata/process-reader work inside generation/caller intervals belongs to their enclosing remainder.",
                forwards="Actual original model.forward invocations and query shapes, including EOS scoring probes; no fabricated 30/70 timers.",
                allocator="Above-model is all nonmodel allocation, not KV-only or pure algorithm overhead; peak subtraction uses current model storage.",
                process_memory="Optional driver reader runs only at generation returns, explicit interaction endpoints and final samples, on CUDA-backed models; absent at forward/prune observations.",
                peaks="KV peaks are sampled; allocator peaks catch unsampled CUDA transients. Per-step counters start after the original reset or at incremental entry; episode peaks accumulate step maxima.",
                inline_cache_gather_isolated=False,
                cache_gather="Only explicitly wrapped functions are isolated. Inline KV gather/index_select remains in selection remainder or Lazy model-forward remainder; recurrence gather is metadata, not KV.",
                lazy_saliency="Wrapped recurrence update excludes inline attention-weight matmul/softmax, which remains in model-forward remainder.",
                timing_calls="Intervals, including nested same-phase wrappers; workload.forward_calls counts actual forwards only.",
            ),
        ))

    def close(self):
        """Idempotently restore exact pre-existing instance/module attributes."""
        if self._closed:
            return
        for obj, name, own, prior in reversed(self._patches):
            if own:
                setattr(obj, name, prior)
            else:
                delattr(obj, name)
        self._patches.clear()
        self._wrapped.clear()
        self._closed = True
