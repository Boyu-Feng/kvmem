"""Process-local placement and observation adapters for the frozen QA evaluator.

Only index/score *devices* change. Cache tensors stay on their owning layers;
neither token selection, cache contents, positions, nor decoding is repaired.
In particular, the legacy unmaterialized-final-token counter discrepancy is
observed, not filled in. No source in models/, kv_cache/, or token_tracker is
modified. CPU models are supported by the adapters for regression tests only;
the production model factory requires two CUDA devices and forbids offload.
"""
from __future__ import annotations

import contextlib
import functools
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys

import torch

from scripts.legacy_efficiency_metrics import attach_legacy_efficiency, cache_tensor_pairs
from scripts.run_rebuttal_efficiency import ErrorWatch

GIB = 1024 ** 3
ROOT = Path(__file__).resolve().parents[1]
METHODS = {"fullkv": "none", "stepkv": "step_aware_h2o", "h2o": "h2o", "tova": "tova",
           "h2o_step": "h2o", "tova_step": "tova"}
STEP_METHODS = ("h2o_step", "tova_step")
PLACEMENT_POLICY = {
    "device_map": "balanced",
    "max_memory": {"0": "40GiB", "1": "40GiB"},
    "dtype": "bfloat16",
    "offload": "forbidden (parameters, buffers, and device-map entries)",
    "conditions": "identical fixed loader arguments for every dataset/method/ratio",
}
ADAPTATIONS = {
    "model_factory": "models.QwenLLMWithKVCache.AutoModelForCausalLM only; restored on exit",
    "gather": "original selectors; copy each index to the corresponding K/V tensor device",
    "scores": "original per-layer reductions/order/dtype; colocate reduced scores before stack",
    "stepkv": "put score vectors on the first cache layer's device, matching legacy index creation",
    "step_controls": "h2o_step/tova_step use the SHA-verified frozen timing-only adapters, not a "
                     "new decoder; native monitor callbacks include suppressed no-ops, while the "
                     "separate legacy_adapter_audit identifies completed-block pruning",
    "metrics": "LegacyEfficiencyMonitor(synchronize=False, synchronize_components=False)",
    "failure_policy": "reject swallowed inference/pruning/tracker errors instead of scoring a fallback",
    "unchanged": "selection, prompts, scoring, first-step generate, subsequent decoders, stop/crop, "
                 "attention defaults, generation_config, and legacy cache counters",
    "timing": "diagnostic host intervals on shared, model-sharded GPUs; not headline latency "
              "and not comparable to single-GPU speed",
}
QUANTITIES = ("tokens", "total_bytes", "protected_tokens", "protected_bytes",
              "trajectory_tokens", "trajectory_bytes")


def frozen_step_adapter(method):
    """Import a hash-verified adapter without replacing any models namespace."""
    if method not in STEP_METHODS:
        raise ValueError(f"Not a frozen step control: {method}")
    manifest_path = ROOT / "rebuttal/adapters/manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    spec = json.loads(manifest_bytes)["methods"][method]
    relative = f"rebuttal/adapters/{method}/models/LegacyBaseline.py"
    path = ROOT / relative
    source = path.read_bytes()
    expected = spec["source_files"]["models/LegacyBaseline.py"]
    source_hash = hashlib.sha256(source).hexdigest()
    if expected["from"] != "adapter" or source_hash != expected["sha256"]:
        raise RuntimeError(f"Frozen {method} adapter checksum mismatch")
    for name, expected_hash in spec["original_core_hashes"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected_hash:
            raise RuntimeError(f"Frozen {method} original source checksum mismatch: {name}")
    module_name = f"scripts._qa32_frozen_{method}"
    module = sys.modules.get(module_name)
    if module is None:
        module_spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(module_spec)
        # Execute exactly the bytes just checked, not a second read or a stale
        # bytecode cache. The original filename is retained for tracebacks.
        exec(compile(source, str(path), "exec"), module.__dict__)
        module._qa32_source_hash = source_hash
        sys.modules[module_name] = module
    from models.QwenLLMWithKVCache import QwenLLMWithKVCache
    if (module._qa32_source_hash != source_hash or module.LEGACY_METHOD != method
            or module.BASE_MODE != METHODS[method]
            or not issubclass(module.LegacyBaselineLLM, QwenLLMWithKVCache)):
        raise RuntimeError(f"Unexpected frozen {method} adapter identity")
    module._qa32_source_hashes = {
        relative: source_hash,
        "rebuttal/adapters/manifest.json": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    return module


def llm_class_for_method(method):
    if method in STEP_METHODS:
        return frozen_step_adapter(method).LegacyBaselineLLM
    if method not in METHODS:
        raise ValueError(f"Unknown QA32 method: {method}")
    from models.QwenLLMWithKVCache import QwenLLMWithKVCache
    return QwenLLMWithKVCache


def method_protocol(method):
    if method in STEP_METHODS:
        adapter = frozen_step_adapter(method)
        return {
            "base_mode": adapter.BASE_MODE, "pruning_schedule": "frozen_completed_block_control",
            "scope": adapter.LEGACY_SCOPE, "config_override": dict(adapter.LEGACY_CONFIG_OVERRIDE),
            "adapter_source_hashes": dict(adapter._qa32_source_hashes),
        }
    return {
        "base_mode": METHODS[method],
        "pruning_schedule": ("none" if method == "fullkv" else "step_aware" if method == "stepkv"
                             else "native_generation_time"),
        "scope": "Unmodified frozen legacy method, including original first block, prefill/token "
                 "pruning where applicable, decoder, selector, stop/crop and generation configuration.",
        "config_override": {}, "adapter_source_hashes": {},
    }


def legacy_config(base, method, ratio):
    """Obtain defaults from the original runner's zero-sample finalize path.

    This executes its original configuration dictionary, not a duplicated copy.
    Empty filenames cannot read a checkpoint or write output; the finalizer is
    temporarily replaced with a collector before the no-model fast path runs.
    """
    original = base._finalize_react_kv_output
    overrides = ({**frozen_step_adapter(method).LEGACY_CONFIG_OVERRIDE}
                 if method in STEP_METHODS else {})
    overrides["cache_ratio"] = ratio

    def collect(results, output, mode, config, *args, **kwargs):
        if results or output or mode != METHODS[method]:
            raise RuntimeError("Unexpected legacy configuration collection path")
        return dict(config)

    base._finalize_react_kv_output = collect
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = base.run_react_kv_experiment(
                [], [], None, METHODS[method], "", "",
                kv_config_override=overrides)
        if not isinstance(result, dict) or result.get("pruning_mode") != METHODS[method]:
            raise RuntimeError("Could not collect frozen legacy defaults")
        return result
    finally:
        base._finalize_react_kv_output = original


def configure_cuda(memory_limit_gib=44):
    """Apply the allocator limit before loading any weights; never select GPUs."""
    if memory_limit_gib != 44:
        raise ValueError("This comparison fixes the allocator cap at 44 GiB per device")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("Exactly two externally selected CUDA devices are required")
    devices = {}
    for device in range(2):
        props = torch.cuda.get_device_properties(device)
        total = int(props.total_memory)
        if total < memory_limit_gib * GIB:
            raise RuntimeError(f"cuda:{device} is smaller than the requested allocator cap")
        torch.cuda.set_per_process_memory_fraction(memory_limit_gib * GIB / total, device)
        devices[str(device)] = {
            "name": props.name, "total_bytes": total,
            "allocator_limit_bytes": int(memory_limit_gib * GIB),
            "weight_ceiling_bytes": 40 * GIB,
        }
    return devices


def _cuda_index(value):
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    try:
        device = torch.device(value)
    except (TypeError, RuntimeError):
        raise RuntimeError(f"Invalid/offloaded device-map entry: {value!r}") from None
    if device.type != "cuda" or device.index is None:
        raise RuntimeError(f"Invalid/offloaded device-map entry: {value!r}")
    return device.index


def validate_placement(model):
    mapping = getattr(model, "hf_device_map", None)
    if not isinstance(mapping, dict) or not mapping:
        raise RuntimeError("Loaded model lacks an explicit HF device map")
    normalized = {name: _cuda_index(value) for name, value in mapping.items()}
    if set(normalized.values()) != {0, 1}:
        raise RuntimeError("Model must be truly sharded across cuda:0 and cuda:1")
    parameter_bytes = {"0": 0, "1": 0}
    parameter_devices = set()
    for name, parameter in model.named_parameters():
        device = parameter.device
        if device.type != "cuda" or device.index not in (0, 1):
            raise RuntimeError(f"CPU/disk/meta parameter offload is forbidden: {name} on {device}")
        if parameter.dtype != torch.bfloat16:
            raise RuntimeError(f"Non-BF16 model parameter: {name} ({parameter.dtype})")
        parameter_devices.add(device.index)
        parameter_bytes[str(device.index)] += parameter.numel() * parameter.element_size()
    if parameter_devices != {0, 1}:
        raise RuntimeError("Device map claims sharding but actual parameters do not")
    if any(size > 40 * GIB for size in parameter_bytes.values()):
        raise RuntimeError("Actual model weights exceed the 40 GiB/device ceiling")
    for name, buffer in model.named_buffers():
        if buffer.device.type != "cuda" or buffer.device.index not in (0, 1):
            raise RuntimeError(f"Offloaded model buffer: {name} on {buffer.device}")
    embedding_device = model.get_input_embeddings().weight.device
    if model.device != embedding_device:
        raise RuntimeError("Legacy input device does not match the input embedding device")
    layers = list(model.model.layers)
    layer_devices = []
    for layer in layers:
        devices = {str(parameter.device) for parameter in layer.parameters()}
        if len(devices) != 1:
            raise RuntimeError("A decoder layer is split across devices")
        layer_devices.append(devices.pop())
    return {
        "hf_device_map": normalized, "layer_devices": layer_devices,
        "input_device": str(embedding_device), "parameter_bytes_per_device": parameter_bytes,
        "attention_implementation": model.config._attn_implementation,
    }


class BalancedModelFactory:
    """Override the factory binding in one legacy module, not transformers globally."""
    def __init__(self, module, model_path, on_loaded=None):
        self.module, self.model_path = module, str(model_path)
        self.original = module.AutoModelForCausalLM
        self.on_loaded = on_loaded
        self.placement = None

    def from_pretrained(self, model_path, *args, **kwargs):
        if str(model_path) != self.model_path or args:
            raise RuntimeError("Unexpected model path/positional loader override")
        if kwargs.get("dtype") != torch.bfloat16 or kwargs.get("device_map") != "auto":
            raise RuntimeError("Frozen legacy model-loading signature changed")
        forbidden = {"max_memory", "offload_folder", "offload_state_dict",
                     "quantization_config", "load_in_4bit", "load_in_8bit"}
        if forbidden.intersection(kwargs):
            raise RuntimeError("Unexpected model loading/offload/quantization option")
        kwargs = {**kwargs, "device_map": "balanced", "max_memory": {0: "40GiB", 1: "40GiB"},
                  "local_files_only": True}
        model = self.original.from_pretrained(model_path, **kwargs)
        self.placement = validate_placement(model)
        if self.on_loaded:
            self.on_loaded(self.placement)
        return model

    def __enter__(self):
        self.module.AutoModelForCausalLM = self
        return self

    def __exit__(self, *exc):
        self.module.AutoModelForCausalLM = self.original


class LayerLocalIndex:
    """Read-only selector facade; results remain real, unwrapped layer tensors."""
    def __init__(self, tensor):
        self.tensor = tensor

    def __getattr__(self, name):
        return getattr(self.tensor, name)

    def __getitem__(self, index):
        def local(item):
            return item.to(device=self.tensor.device) if isinstance(item, torch.Tensor) else item
        index = tuple(local(item) for item in index) if isinstance(index, tuple) else local(index)
        return self.tensor[index]


class LayerDeviceAdapter:
    """Instance-only adapters around original selectors, with restoration."""
    def __init__(self, llm):
        self.llm, self.errors, self.patches = llm, [], []
        manager = llm.kv_manager
        if manager is None:
            return
        strategy = manager.pruning_strategy
        if strategy.mode not in ("h2o", "tova", "step_aware_h2o"):
            raise RuntimeError("Unreviewed pruning mode for the two-device adapter")
        get_kv = strategy._get_kv

        def local_kv(cache, layer):
            return tuple(LayerLocalIndex(tensor) for tensor in get_kv(cache, layer))

        self._patch(strategy, "_get_kv", local_kv)
        self._patch(strategy, "_get_device", lambda cache: cache_tensor_pairs(cache)[0][0].device)
        scorer = strategy.h2o_scorer
        self._patch(scorer, "compute_scores", self._scores_wrapper(scorer, False))
        self._patch(scorer, "compute_attention_matrix", self._scores_wrapper(scorer, True))

    def _patch(self, obj, name, replacement):
        self.patches.append((obj, name, name in obj.__dict__, obj.__dict__.get(name)))
        setattr(obj, name, replacement)

    def _scores_wrapper(self, scorer, matrix):
        name = "compute_attention_matrix" if matrix else "compute_scores"
        original = getattr(scorer, name)

        @functools.wraps(original)
        def compute(attentions, *args, **kwargs):
            try:
                pairs = cache_tensor_pairs(self.llm.past_key_values)
                if not pairs:
                    raise RuntimeError("Scoring requested without a materialized active cache")
                target = pairs[0][0].device
                selected = attentions[max(0, len(attentions) - scorer.num_score_layers):]
                if len({tensor.device for tensor in selected}) <= 1:
                    return original(attentions, *args, **kwargs).to(target)
                # Keep the frozen reduction order/dtype; transfer reduced data,
                # not entire multi-head attention tensors or any cache payload.
                start_name, end_name = (("key_start", "key_end") if matrix
                                        else ("start_pos", "end_pos"))
                start = args[0] if args else kwargs.get(start_name)
                end = args[1] if len(args) > 1 else kwargs.get(end_name)
                start = 0 if start is None else start
                end = selected[0].shape[-1] if end is None else end
                if matrix:
                    start, end = max(0, int(start)), min(int(end), int(selected[0].shape[-1]))
                else:
                    end = min(end, selected[0].shape[-1])
                if end <= start:
                    return torch.zeros((0, 0) if matrix else (0,), device=target)
                reduced = [(tensor[0].mean(dim=0)[:, start:end] if matrix
                            else tensor[0].sum(dim=0).sum(dim=0)[start:end]).to(target)
                           for tensor in selected]
                return torch.stack(reduced, dim=0).mean(dim=0)
            except Exception as error:
                self.errors.append(f"{name}: {type(error).__name__}: {error}")
                raise
        return compute

    def close(self):
        for obj, name, own, prior in reversed(self.patches):
            if own:
                setattr(obj, name, prior)
            else:
                delattr(obj, name)
        self.patches.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class QAErrorWatch(ErrorWatch):
    def write(self, value):
        joined = self.tail + value
        if any(marker in joined for marker in (
            "Token tracking failed", "Error in generate_incremental",
            "generate() returned no past_key_values",
            "Failed to set pruning_strategy.num_score_layers",
            "Search hit has no passage text", "corpus_texts.jsonl not found",
        )):
            self.errors.append(joined[-500:])
        return super().write(value)

    def __getattr__(self, name):
        return getattr(self.stream, name)


def allocator_snapshot(devices, *, reset=False):
    result = {}
    for device in devices:
        if reset:
            torch.cuda.reset_peak_memory_stats(device)
        result[str(device)] = {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
    return result


def cache_layout(llm):
    pairs = cache_tensor_pairs(llm.past_key_values)
    layers = list(llm.model.model.layers)
    if len(pairs) != len(layers):
        raise RuntimeError("Materialized cache layer count disagrees with model placement")
    layout = []
    for index, ((keys, values), layer) in enumerate(zip(pairs, layers)):
        device = next(layer.parameters()).device
        if keys.device != device or values.device != device:
            raise RuntimeError(f"Cache layer {index} is not on its owning decoder device")
        layout.append({
            "layer": index, "key_device": str(keys.device), "value_device": str(values.device),
            "key_dtype": str(keys.dtype), "value_dtype": str(values.dtype),
            "key_shape": list(keys.shape), "value_shape": list(values.shape),
            "key_bytes": keys.numel() * keys.element_size(),
            "value_bytes": values.numel() * values.element_size(),
        })
    return layout


def compact_metrics(snapshot):
    if snapshot["instrumentation_errors"]:
        raise RuntimeError(f"KV instrumentation failed: {snapshot['instrumentation_errors']}")
    steps = snapshot["steps"]
    if not steps or any(not step["succeeded"] for step in steps):
        raise RuntimeError("Missing/failed generation-block measurements")
    if snapshot["counts"]["active_model_forwards"] <= 0:
        raise RuntimeError("No actual model forwards were measured")
    samples = [step["boundary"] for step in steps] + [snapshot["kv"]["current"]]
    for sample in samples:
        if not sample or not sample["layer_tokens"] or sample["tokens"] is None:
            raise RuntimeError("Missing or nonuniform real per-layer KV measurements")
        if (sample["total_bytes"] != sample["protected_bytes"] + sample["trajectory_bytes"]
                or sample["tokens"] != sample["protected_tokens"] + sample["trajectory_tokens"]):
            raise RuntimeError("Invalid protected/trajectory KV partition")
    means = {key: sum(step["boundary"][key] for step in steps) / len(steps)
             for key in QUANTITIES}

    def values(sample):
        return {key: sample[key] for key in QUANTITIES} if sample else None

    return {
        "generation_blocks": len(steps), "boundary_means": means,
        "final": values(snapshot["kv"]["current"]),
        "peak_active": values(snapshot["kv"]["peak_active"]),
        "peak_trajectory": values(snapshot["kv"]["peak_trajectory"]),
        "peak_preprune": values(snapshot["kv"]["peak_preprune"]),
        "step_boundaries": [values(step["boundary"]) for step in steps],
        "step_preprune_peaks": [values(step["peak_preprune"]) for step in steps],
        "step_active_peaks": [values(step["peak_active"]) for step in steps],
        "counts": snapshot["counts"],
        "allocator": snapshot.get("qa32_allocator", {}),
    }


def observed_episode(base, sample, llm, retriever, method, monitor, adapter, devices=(0, 1)):
    """Call the original episode exactly once; reject failures before scoring."""
    before = allocator_snapshot(devices, reset=True)
    stdout, stderr = QAErrorWatch(sys.stdout), QAErrorWatch(sys.stderr)
    adapter.errors.clear()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        result = base._run_react_kv_episode(
            sample["question"], llm, retriever, pruning_mode=METHODS[method], return_debug=True)
    errors = stdout.errors + stderr.errors + adapter.errors
    if errors:
        raise RuntimeError(f"Legacy episode reported a suppressed error: {errors[:4]}")
    answer, trajectory, timings, debug = result
    snapshot = monitor.snapshot()
    snapshot["active_cache_layout"] = cache_layout(llm)
    if len(snapshot["steps"]) != len(timings):
        raise RuntimeError("Legacy step timings do not match observed generation blocks")
    snapshot["qa32_allocator"] = {
        "before_episode": before, "after_episode": allocator_snapshot(devices),
        "scope": "process allocations on each device; peaks reset immediately before the original "
                 "episode (previous-question cache can still be live until generate_first resets it)",
    }
    metrics = compact_metrics(snapshot)
    if not isinstance(answer, str):
        raise RuntimeError("Original evaluator returned a non-string answer")
    em, f1 = base.exact_match(answer, sample["answer"]), base.f1_score(answer, sample["answer"])
    if not math.isfinite(float(f1)) or not 0 <= f1 <= 1:
        raise RuntimeError("Invalid legacy F1 score")
    audit = {
        "diagnostics": snapshot, "trajectory": trajectory,
        "legacy_step_timings_bookkeeping_only": timings,
        "step_scores": debug.get("step_scores", {}),
        "step_spans": debug.get("step_spans", []),
        "prompt_token_count": debug.get("prompt_token_count"),
        "adaptations": ADAPTATIONS,
    }
    if method in STEP_METHODS:
        audit["legacy_adapter_audit"] = llm.legacy_audit()
    return {
        "predicted_answer": answer, "em": em, "f1": f1, "num_steps": len(trajectory),
        "num_steps_scope": "original len(trajectory); first-step Finish has zero legacy trajectory entries",
        "kv_metrics": metrics,
    }, audit


def attach_monitor(llm):
    return attach_legacy_efficiency(
        llm, synchronize=False, synchronize_components=False, event_limit=0)
