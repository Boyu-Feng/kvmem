"""Isolated, action-only ALFWorld policy; never imports the frozen QA evaluator.

This is an explicitly adapted experiment, not an official H2O/TOVA reproduction.
Only ordinary full-attention Qwen2/Qwen2.5 models are supported. All conditions
use eager attention and the same exact-token greedy decoder. ``auto`` loading
uses one or two exposed GPUs with an explicit CUDA-only map, never offloading.

A step ends after an emitted action and the environment's returned user message,
including its chat delimiters. The next assistant header is appended *after*
pruning, on the next act(), so there is no replay/scoring forward or stale-logit
action after eviction. Generated tokens, including EOS, are never rolled back or
re-encoded. Absolute RoPE positions count all logical tokens, not resident KV.
StepKV token scores reduce only the latest forward's attention, as in the QA
scorer; only H2O uses episode-cumulative attention for selection.
"""
from __future__ import annotations

import copy
import hashlib
import math
import os
import platform
import re
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, DynamicLayer

from kv_cache import anchor_utils


class UnsupportedCacheError(RuntimeError):
    """The cache cannot safely support this policy's explicit invariants."""


class ContextOverflowError(RuntimeError):
    """The append-only logical transcript exceeded its configured limit."""


class AgentKVPolicy:
    IMPLEMENTATION_VERSION = "alfworld-qwen2-exact-token-model-parallel-v6"
    METHODS = frozenset({"fullkv", "stepkv", "h2o_step", "tova_step", "stepkv_no_step_score"})
    PINNED_REVISIONS = {
        "Qwen/Qwen2.5-32B-Instruct": "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
        "Qwen/Qwen2.5-7B-Instruct": "a09a35458c702b33eeacc393d103063234e8bc28",
    }
    AUTO_WEIGHT_LIMIT_GIB = 40.0
    AUTO_ALLOCATOR_LIMIT_GIB = 44.0
    AUTO_RUNTIME_HEADROOM_GIB = 4.0
    STORAGE_ROOT = Path("/mnt/checkpoints/bruli/fengboyu/kvmem_rebuttal_20260929")
    PREFILL_CHUNK_SIZE = 64
    MAX_PREFILL_QUERY_KEY_PRODUCT = 64 * 8192
    SCORE_LAYERS = 3
    TOKEN_WEIGHT = 0.8
    STEP_WEIGHT = 0.8
    STEP_FLOOR = 0.10
    VALIDITY_NOVELTY_WEIGHT = 0.85
    CITATION_WEIGHT = 0.15
    REPEAT_PENALTY = 0.3
    H2O_RECENT_WINDOW = 32
    _CHAT_MARKER = "__AGENT_KV_ASSISTANT_BOUNDARY_8f16512f__"

    def __init__(
        self,
        model_path: str,
        method: str,
        keep_ratio: float,
        max_new_tokens: int = 128,
        device: str = "cuda:0",
        max_context_tokens: int = 8192,
        memory_limit_gib: float | None = None,
    ):
        self._validate_options(method, keep_ratio, max_new_tokens, max_context_tokens)
        revision = self.PINNED_REVISIONS.get(str(model_path))
        cache_dir = self._loader_cache_dir()
        loading = dict(local_files_only=True, trust_remote_code=False, cache_dir=cache_dir)
        if revision is not None:
            loading["revision"] = revision
        config = AutoConfig.from_pretrained(model_path, **loading)
        self._text_config(config)
        tokenizer = AutoTokenizer.from_pretrained(model_path, **loading)
        if device == "auto":
            plan = self._auto_loading_plan(config, memory_limit_gib)
            dtype = torch.bfloat16
        else:
            target = torch.device(device)
            if target.type not in ("cuda", "cpu"):
                raise ValueError("Use cpu, an explicit CUDA device, or auto for one or two exposed GPUs")
            if target.type == "cuda":
                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA was requested but is unavailable")
                target = torch.device("cuda", target.index if target.index is not None else 0)
            self._set_memory_limit(target, memory_limit_gib)
            plan = {
                "device_map": {"": str(target)}, "max_memory": None,
                "allocator_limits_gib": {target.index: memory_limit_gib} if target.type == "cuda" else {},
                "placement": "explicit_single_device",
            }
            dtype = torch.bfloat16 if target.type == "cuda" else torch.float32
        plan["cache_dir"] = cache_dir
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            config=config,
            dtype=dtype,
            device_map=plan["device_map"],
            max_memory=plan["max_memory"],
            offload_buffers=False,
            attn_implementation="eager",
            **loading,
        )
        self._initialize(
            model, tokenizer, str(model_path), method, keep_ratio, max_new_tokens,
            device, max_context_tokens, memory_limit_gib, revision, injected=False, loading_plan=plan,
        )

    @classmethod
    def _from_components(
        cls, model, tokenizer, method="fullkv", keep_ratio=1.0,
        max_new_tokens=16, max_context_tokens=4096,
    ):
        """CPU-only dependency injection for small real-model tests; no loading."""
        if any(parameter.device.type != "cpu" for parameter in model.parameters()):
            raise ValueError("The test component constructor is CPU-only")
        obj = cls.__new__(cls)
        obj._initialize(
            model, tokenizer, "<injected-cpu-model>", method, keep_ratio,
            max_new_tokens, "cpu", max_context_tokens, None, None,
            injected=True,
        )
        return obj

    @staticmethod
    def _validate_options(method, ratio, max_new_tokens, max_context_tokens):
        if method not in AgentKVPolicy.METHODS:
            raise ValueError(f"Unsupported method: {method!r}")
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)) or not math.isfinite(ratio):
            raise ValueError("keep_ratio must be a finite number")
        if not 0 < ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1]")
        for name, value in (("max_new_tokens", max_new_tokens), ("max_context_tokens", max_context_tokens)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def _loader_cache_dir(cls) -> str:
        configured = os.environ.get("HF_HUB_CACHE") or os.environ.get("HUGGINGFACE_HUB_CACHE")
        if configured is None and os.environ.get("HF_HOME"):
            configured = str(Path(os.environ["HF_HOME"]) / "hub")
        candidate = Path(configured) if configured else cls.STORAGE_ROOT / "cache" / "huggingface" / "hub"
        root, resolved = cls.STORAGE_ROOT.resolve(), candidate.expanduser().resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"Model cache must be under {root}, not a home/root-filesystem cache")
        return str(resolved)

    @staticmethod
    def _set_memory_limit(device: torch.device, limit: float | None):
        if limit is None:
            return
        if isinstance(limit, bool) or not isinstance(limit, (float, int)) or not math.isfinite(limit) or limit <= 0:
            raise ValueError("memory_limit_gib must be finite and positive")
        if device.type != "cuda":
            raise ValueError("memory_limit_gib requires a CUDA device")
        index = device.index if device.index is not None else 0
        if not torch.cuda.is_available() or index >= torch.cuda.device_count():
            raise ValueError("Requested allocator-limit GPU is not exposed")
        total = torch.cuda.get_device_properties(index).total_memory
        fraction = limit * (1024 ** 3) / total
        if fraction > 1:
            raise ValueError("memory_limit_gib exceeds the exposed GPU's physical memory")
        # This bounds this process's PyTorch allocator, not other processes or
        # non-PyTorch CUDA allocations; it does not reserve or clear GPU memory.
        torch.cuda.set_per_process_memory_fraction(fraction, device=index)

    @staticmethod
    def _cuda_map_devices(device_map: dict) -> set[torch.device]:
        if not isinstance(device_map, dict) or not device_map:
            raise RuntimeError("An explicit nonempty CUDA device map is required")
        devices = set()
        for value in device_map.values():
            try:
                target = torch.device("cuda", value) if isinstance(value, int) else torch.device(value)
            except (TypeError, RuntimeError) as exc:
                raise RuntimeError(f"CPU/disk/unsupported model offloading is forbidden: {value!r}") from exc
            if target.type != "cuda" or target.index is None:
                raise RuntimeError(f"CPU/disk/unspecified model offloading is forbidden: {value!r}")
            devices.add(target)
        return devices

    @classmethod
    def _auto_loading_plan(cls, config, memory_limit_gib):
        """Plan on meta tensors, reject offloading, then load real weights once."""
        count = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if count not in (1, 2):
            raise ValueError("device='auto' requires one or two GPUs exposed by CUDA_VISIBLE_DEVICES")
        cap = cls.AUTO_ALLOCATOR_LIMIT_GIB if memory_limit_gib is None else memory_limit_gib
        if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not math.isfinite(cap):
            raise ValueError("memory_limit_gib must be finite")
        if not cls.AUTO_RUNTIME_HEADROOM_GIB < cap <= cls.AUTO_ALLOCATOR_LIMIT_GIB:
            raise ValueError("For shared-GPU auto loading, the per-GPU allocator cap must be in (4, 44] GiB")
        weight_limit = min(cls.AUTO_WEIGHT_LIMIT_GIB, cap - cls.AUTO_RUNTIME_HEADROOM_GIB)
        maxima = {index: int(weight_limit * 1024 ** 3) for index in range(count)}
        for index in range(count):
            cls._set_memory_limit(torch.device("cuda", index), cap)

        from accelerate import infer_auto_device_map, init_empty_weights
        from accelerate.utils import get_balanced_memory

        with init_empty_weights():
            skeleton = AutoModelForCausalLM.from_config(
                copy.deepcopy(config), dtype=torch.bfloat16, attn_implementation="eager",
            )
            skeleton.tie_weights()
        no_split = skeleton._no_split_modules
        balanced = get_balanced_memory(
            skeleton, max_memory=dict(maxima), no_split_module_classes=no_split, dtype=torch.bfloat16,
        )
        balanced = {index: min(maxima[index], balanced[index]) for index in maxima}
        device_map = infer_auto_device_map(
            skeleton, max_memory=balanced, no_split_module_classes=no_split, dtype=torch.bfloat16,
            offload_buffers=False,
        )
        del skeleton
        devices = cls._cuda_map_devices(device_map)
        if devices != {torch.device("cuda", index) for index in range(count)}:
            raise RuntimeError("Auto placement must use exactly the exposed GPUs without offloading")
        return {
            "device_map": device_map, "max_memory": balanced,
            "weight_ceiling_bytes": maxima, "allocator_limits_gib": {index: float(cap) for index in range(count)},
            "placement": "balanced_explicit_two_gpu_no_offload" if count == 2 else "explicit_one_gpu_no_offload",
        }

    @staticmethod
    def _text_config(config):
        if config.model_type != "qwen2":
            raise UnsupportedCacheError(f"Unsupported model type: {config.model_type!r}")
        text = config.get_text_config(decoder=True)
        layer_types = getattr(text, "layer_types", None)
        if layer_types is None:
            if getattr(text, "use_sliding_window", False):
                raise UnsupportedCacheError("An explicit supported layer layout is required")
            layer_types = ["full_attention"] * text.num_hidden_layers
        if len(layer_types) != text.num_hidden_layers or any(
            kind != "full_attention" for kind in layer_types
        ) or not layer_types:
            raise UnsupportedCacheError(f"Unsupported attention layer layout: {layer_types!r}")
        return text, tuple(layer_types)

    def _initialize(
        self, model, tokenizer, model_path, method, ratio, max_new_tokens,
        device, max_context_tokens, memory_limit_gib, revision, *, injected, loading_plan=None,
    ):
        self._validate_options(method, ratio, max_new_tokens, max_context_tokens)
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.method = method
        self.keep_ratio = float(ratio)
        self.max_new_tokens = max_new_tokens
        self.max_context_tokens = max_context_tokens
        self.memory_limit_gib = memory_limit_gib
        self._requested_device = str(device)
        self.device = self.model.get_input_embeddings().weight.device
        self._loading_plan = loading_plan or {
            "device_map": {"": str(self.device)}, "max_memory": None,
            "allocator_limits_gib": {}, "placement": "injected_cpu",
        }
        self._requested_revision = revision
        self._injected = injected
        self.text_config, self.layer_types = self._text_config(model.config)
        self.full_attention_layers = tuple(range(len(self.layer_types)))
        self.model.set_attn_implementation("eager")
        if self.text_config._attn_implementation != "eager":
            raise RuntimeError("Genuine scoring requires eager attention in every condition")
        parameters = list(self.model.parameters())
        cuda_requested = self._requested_device == "auto" or torch.device(self._requested_device).type == "cuda"
        allowed = (
            self._cuda_map_devices(self._loading_plan["device_map"])
            if cuda_requested else {torch.device("cpu")}
        )
        if not parameters or any(p.device not in allowed for p in parameters) or self.device not in allowed:
            raise RuntimeError("Model parameters and input embeddings must remain on the explicit devices; no offloading")
        actual_devices = {parameter.device for parameter in parameters}
        if actual_devices != allowed:
            raise RuntimeError("Actual parameter devices do not match the audited loading plan")
        self._model_devices = tuple(sorted(actual_devices, key=str))
        actual_map = getattr(self.model, "hf_device_map", self._loading_plan["device_map"])
        if self.device.type == "cuda" and not self._cuda_map_devices(actual_map) <= allowed:
            raise RuntimeError("The loaded model device map differs from the audited CUDA-only plan")
        self._actual_device_map = {name: str(value) for name, value in actual_map.items()}
        self._layer_devices = tuple(next(layer.parameters()).device for layer in self.model.model.layers)
        if len(self._layer_devices) != len(self.layer_types):
            raise RuntimeError("Decoder layer/device layout does not match the model config")
        self._dtype = parameters[0].dtype
        if self.device.type == "cuda" and any(p.dtype != torch.bfloat16 for p in parameters if p.is_floating_point()):
            raise RuntimeError("CUDA model parameters must be loaded in BF16 without offloading")
        self._parameter_count = sum(p.numel() for p in parameters)
        self._parameter_bytes = sum(p.numel() * p.element_size() for p in parameters)
        self._chat_template = tokenizer.get_chat_template()
        if not isinstance(self._chat_template, str) or not self._chat_template:
            raise ValueError("A native Qwen ChatML tokenizer template is required")
        self._im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        for marker, token in (("<|im_end|>", self._im_end), ("<|im_start|>", self._im_start)):
            if not isinstance(token, int) or token < 0 or self._encode(marker) != [token]:
                raise ValueError(f"Tokenizer must encode {marker} as its own ChatML token")
        self._eos_ids = {self._im_end}
        for value in (getattr(model.generation_config, "eos_token_id", None), tokenizer.eos_token_id):
            if value is not None:
                self._eos_ids.update(value if isinstance(value, (tuple, list)) else [value])
        if any(
            not isinstance(value, int) or value < 0 or value >= self.text_config.vocab_size
            for value in self._eos_ids
        ):
            raise ValueError("Invalid EOS token IDs")
        self._clear_episode()

    def _clear_episode(self):
        self.past_key_values = DynamicCache(config=self.model.config)
        self._token_ids: list[int] = []
        self._global_ids: list[int] = []
        self._protected_tokens = 0
        self._hh_scores = torch.empty(0, dtype=torch.float32)
        self._last_query_scores = torch.empty(0, dtype=torch.float32)
        self._steps: list[dict[str, Any]] = []
        self._seen_anchors: set[str] = set()
        self._action_counts: dict[str, int] = {}
        self._boundaries: list[dict[str, Any]] = []
        self._prune_events: list[dict[str, Any]] = []
        self._forward_count = 0
        self._attention_query_count = 0
        self._generated_action_count = 0
        self._peak_kv_bytes = 0
        self._peak_cache_bytes = 0
        self._peak_boundary_kv_bytes = 0
        self._peak_boundary_cache_bytes = 0
        self._peak_kv_bytes_by_device: dict[str, int] = {}
        self._peak_boundary_kv_bytes_by_device: dict[str, int] = {}
        self._peak_resident_tokens = 0
        self._step_start = 0
        self._pending_action: str | None = None
        self._pending_generated_ids: list[int] = []
        self._next_header: list[int] = []
        self._logits = None
        self._failure: str | None = None
        self._ready = False
        self._system_prompt: str | None = None
        self._initial_observation_hash: str | None = None
        self._last_forward_positions: list[int] = []

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _encode(self, text: str) -> list[int]:
        ids = self.tokenizer(text, add_special_tokens=False).input_ids
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        if not isinstance(ids, list) or any(isinstance(t, bool) or not isinstance(t, int) for t in ids):
            raise ValueError("Tokenizer must return one unbatched integer token sequence")
        if any(t < 0 or t >= self.text_config.vocab_size for t in ids):
            raise ValueError("Tokenizer produced an out-of-vocabulary token ID")
        return ids

    def _render(self, messages, *, generation_prompt):
        rendered = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=generation_prompt, enable_thinking=False,
        )
        if not isinstance(rendered, str):
            raise ValueError("Chat template must render a string")
        return rendered

    def reset(self, system_prompt: str, initial_observation: str) -> None:
        if not isinstance(system_prompt, str) or not isinstance(initial_observation, str):
            raise TypeError("System prompt and observation must be strings")
        rendered = self._render(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": initial_observation}],
            generation_prompt=True,
        )
        ids = self._encode(rendered)
        if not ids:
            raise ValueError("The initial chat prompt is empty")
        self._clear_episode()
        self._system_prompt = system_prompt
        self._initial_observation_hash = self._hash(initial_observation)
        self._append(ids)
        self._protected_tokens = len(ids)
        self._step_start = len(ids)
        self._ready = True
        self._measure_cache()

    def _assert_healthy(self):
        if self._failure is not None:
            raise RuntimeError(f"Policy state is invalid; reset is required: {self._failure}")

    def _check_cache(self):
        cache = self.past_key_values
        if type(cache) is not DynamicCache or getattr(cache, "offloading", False):
            raise UnsupportedCacheError("Only non-offloaded Transformers DynamicCache is supported")
        if len(cache.layers) != len(self.layer_types):
            raise UnsupportedCacheError("Cache layer count does not match the model; no layer may be omitted")
        resident = len(self._global_ids)
        if self._global_ids != sorted(set(self._global_ids)) or any(
            gid < 0 or gid >= len(self._token_ids) for gid in self._global_ids
        ):
            raise RuntimeError("Resident global IDs must be ordered, unique, and logically valid")
        if self._global_ids[:self._protected_tokens] != list(range(self._protected_tokens)):
            raise RuntimeError("Protected initial prompt tokens are missing")
        for index, layer in enumerate(cache.layers):
            if type(layer) is not DynamicLayer:
                raise UnsupportedCacheError(
                    f"Unsupported cache layer {index}: {type(layer).__name__}; expected DynamicLayer"
                )
            if not layer.is_initialized:
                if resident:
                    raise RuntimeError(f"Full-attention cache layer {index} is uninitialized")
                continue
            k, v = layer.keys, layer.values
            if not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
                raise UnsupportedCacheError("Full-attention layers must expose tensor keys and values")
            if k.ndim != 4 or v.ndim != 4 or k.shape[:3] != v.shape[:3] or k.shape[0] != 1:
                raise UnsupportedCacheError("Only batch-one [batch, heads, tokens, dim] KV is supported")
            if k.shape[-2] != resident or k.device != self._layer_devices[index] or v.device != k.device:
                raise RuntimeError(f"KV/global-ID length or decoder-layer device mismatch at layer {index}")

    def _measure_cache(self) -> dict[str, Any]:
        self._check_cache()
        kv = 0
        per_device = {}
        for index in self.full_attention_layers:
            layer = self.past_key_values.layers[index]
            if layer.is_initialized:
                size = sum(t.numel() * t.element_size() for t in (layer.keys, layer.values))
                kv += size
                device = str(layer.keys.device)
                per_device[device] = per_device.get(device, 0) + size
        for device, size in per_device.items():
            self._peak_kv_bytes_by_device[device] = max(self._peak_kv_bytes_by_device.get(device, 0), size)
        self._peak_kv_bytes = max(self._peak_kv_bytes, kv)
        self._peak_cache_bytes = max(self._peak_cache_bytes, kv)
        self._peak_resident_tokens = max(self._peak_resident_tokens, len(self._global_ids))
        return {
            "full_attention_kv_bytes": kv,
            "full_attention_kv_bytes_by_device": per_device,
            "cache_bytes": kv,
        }

    def _causal_mask(self, queries: int, resident: int):
        new = torch.full(
            (queries, queries), torch.finfo(self._dtype).min, dtype=self._dtype, device=self.device,
        ).triu(diagonal=1)
        prefix = torch.zeros((queries, resident), dtype=self._dtype, device=self.device)
        # Packed order governs causal masking, while logical positions govern
        # RoPE. Accelerate's decoder hooks place this mask on each layer's GPU.
        return torch.cat((prefix, new), dim=-1)[None, None, :, :]

    def _attention_scores(self, attentions, queries: int, keys: int):
        if not isinstance(attentions, (tuple, list)) or not attentions:
            raise RuntimeError("Empty attention output: eager genuine attention is required")
        if len(attentions) != len(self.full_attention_layers):
            raise RuntimeError("Attention outputs do not cover exactly all full-attention layers")
        full = list(attentions)
        for attention in full:
            if not isinstance(attention, torch.Tensor) or attention.ndim != 4 or (
                attention.shape[0] != 1 or attention.shape[1] != self.text_config.num_attention_heads
                or attention.shape[-2:] != (queries, keys)
            ):
                raise RuntimeError("Attention shape does not match genuine queries and resident KV")
        selected = full[-self.SCORE_LAYERS:]
        mass = torch.stack([
            a[0].sum(dim=0, dtype=torch.float32).sum(dim=0).cpu() for a in selected
        ]).mean(dim=0)
        last = full[-1][0, :, -1, :].mean(dim=0, dtype=torch.float32).cpu()
        if not torch.isfinite(mass).all() or not torch.isfinite(last).all() or (mass < 0).any():
            raise RuntimeError("Nonfinite or invalid genuine attention scores")
        return mass, last

    def _prefill_chunk_limit(self, logical_start: int) -> int:
        chunk = self.PREFILL_CHUNK_SIZE
        if chunk < 1:
            raise RuntimeError("The prefill chunk limit must be positive")
        # Logical, rather than resident, length gives every method the same
        # query-block schedule and bounds eager matrices even with 32K FullKV.
        while chunk > 1 and chunk * (logical_start + chunk) > self.MAX_PREFILL_QUERY_KEY_PRODUCT:
            chunk = max(1, chunk // 2)
        return chunk

    @torch.inference_mode()
    def _append(self, ids: list[int]):
        self._assert_healthy()
        if not ids or any(isinstance(t, bool) or not isinstance(t, int) for t in ids):
            raise ValueError("Cannot append an empty or noninteger token block")
        if any(t < 0 or t >= self.text_config.vocab_size for t in ids):
            raise ValueError("Cannot append out-of-vocabulary tokens")
        limit = min(self.max_context_tokens, self.text_config.max_position_embeddings)
        if len(self._token_ids) + len(ids) > limit:
            raise ContextOverflowError(
                f"Logical context overflow: {len(self._token_ids)} + {len(ids)} > {limit}; "
                "compaction never resets positions or truncates the transcript"
            )
        self._check_cache()
        try:
            offset = 0
            while offset < len(ids):
                start, resident = len(self._token_ids), len(self._global_ids)
                block = ids[offset:offset + self._prefill_chunk_limit(start)]
                positions = torch.arange(start, start + len(block), device=self.device)[None, :]
                outputs = self.model(
                    input_ids=torch.tensor([block], device=self.device),
                    position_ids=positions,
                    cache_position=positions[0],
                    attention_mask=self._causal_mask(len(block), resident),
                    past_key_values=self.past_key_values,
                    use_cache=True, output_attentions=True, return_dict=True, logits_to_keep=1,
                )
                if outputs.past_key_values is not self.past_key_values:
                    raise UnsupportedCacheError("The model replaced the live cache unexpectedly")
                if outputs.logits.shape != (1, 1, self.text_config.vocab_size) or not torch.isfinite(outputs.logits).all():
                    raise RuntimeError("Invalid next-token logits")
                mass, last = self._attention_scores(outputs.attentions, len(block), resident + len(block))
                self._token_ids.extend(block)
                self._global_ids.extend(range(start, start + len(block)))
                if self.method in ("stepkv", "stepkv_no_step_score"):
                    self._hh_scores = mass
                else:
                    self._hh_scores = torch.cat((self._hh_scores, torch.zeros(len(block), dtype=torch.float32))) + mass
                self._last_query_scores = last
                self._logits = outputs.logits[:, -1, :].detach()
                self._last_forward_positions = list(range(start, start + len(block)))
                self._forward_count += 1
                self._attention_query_count += len(block)
                self._measure_cache()
                del outputs
                offset += len(block)
        except Exception as exc:
            self._failure = f"{type(exc).__name__}: {exc}"
            raise
        return self._logits

    @torch.inference_mode()
    def act(self) -> str:
        self._assert_healthy()
        if not self._ready or self._pending_action is not None:
            raise RuntimeError("Call reset before act, and observe exactly once between actions")
        if self._next_header:
            self._append(self._next_header)
            self._next_header = []
        generated = []
        try:
            for _ in range(self.max_new_tokens):
                token = int(self._logits.argmax(dim=-1).item())
                self._append([token])
                generated.append(token)
                if token in self._eos_ids:
                    break
        except Exception as exc:
            self._failure = f"{type(exc).__name__}: {exc}"
            raise
        action = self.tokenizer.decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        ).strip()
        self._pending_action = action
        self._pending_generated_ids = generated
        self._generated_action_count += 1
        return action

    def _observation_parts(self, observation: str) -> tuple[list[int], list[int], bool]:
        if self._CHAT_MARKER in observation:
            raise ValueError("Observation collides with the internal chat-template boundary marker")
        messages = [
            {"role": "user", "content": "Previous environment observation."},
            {"role": "assistant", "content": self._CHAT_MARKER},
            {"role": "user", "content": observation},
        ]
        closed = self._render(messages, generation_prompt=False)
        opened = self._render(messages, generation_prompt=True)
        if closed.count(self._CHAT_MARKER) != 1 or not opened.startswith(closed):
            raise ValueError("Unsupported non-appendable assistant generation header in chat template")
        suffix_text = closed.split(self._CHAT_MARKER, 1)[1]
        header_text = opened[len(closed):]
        suffix, header = self._encode(suffix_text), self._encode(header_text)
        if not suffix or suffix[0] != self._im_end or not header or header[0] != self._im_start:
            raise ValueError("Unsupported non-ChatML action/observation delimiter layout")
        if self._encode(suffix_text + header_text) != suffix + header:
            raise ValueError("Chat-template suffix/header boundary changes tokenization")
        reused_eos = bool(self._pending_generated_ids and self._pending_generated_ids[-1] == self._im_end)
        if reused_eos:
            suffix = suffix[1:]
        return suffix, header, not reused_eos

    @classmethod
    def _utility(cls, progress: float, citation: float) -> float:
        return max(0.0, min(8.0, (
            cls.VALIDITY_NOVELTY_WEIGHT * max(-1.0, min(2.0, progress))
            + cls.CITATION_WEIGHT * math.log1p(max(0.0, citation))
        )))

    def _register_step(self, action: str, observation: str, valid: bool, forced_close: bool):
        anchors = anchor_utils.extract_anchor_terms(observation + "\n" + action)
        novelty = len(anchors - self._seen_anchors) / max(1, len(anchors))
        self._seen_anchors.update(anchors)
        normalized = re.sub(r"\s+", " ", action.strip().lower())
        repeats = self._action_counts.get(normalized, 0) if normalized else 0
        if normalized:
            self._action_counts[normalized] = repeats + 1
        progress = float(valid) + novelty - self.REPEAT_PENALTY * repeats
        self._steps.append({
            "id": len(self._steps) + 1,
            "start": self._step_start, "end": len(self._token_ids),
            "action": action, "emitted_action": self._pending_action,
            "action_changed_by_caller": action != self._pending_action,
            "observation_sha256": self._hash(observation),
            "valid": valid, "novelty": novelty, "repeat_count": repeats,
            "progress": progress, "citation": 0.0, "anchors": anchors,
            "generated_token_ids": list(self._pending_generated_ids),
            "forced_assistant_close": forced_close,
        })
        # Preserve the legacy formula's self-reuse convention, explicitly
        # fingerprinted. Future observable references dynamically update old steps.
        for step in self._steps:
            overlap = len(anchors & step["anchors"])
            if overlap:
                denominator = max(1, min(len(anchors), len(step["anchors"])))
                step["citation"] += min(1.0, 1.5 * overlap / denominator)
            step["utility"] = self._utility(step["progress"], step["citation"])

    def _target_nonprompt(self) -> int:
        logical = len(self._token_ids) - self._protected_tokens
        if self.method == "fullkv":
            return logical
        return max(1, math.floor(self.keep_ratio * logical)) if logical else 0

    def _select_keep(self) -> tuple[list[int], dict[str, Any]]:
        protected = list(range(self._protected_tokens))
        candidates = list(range(self._protected_tokens, len(self._global_ids)))
        budget = min(len(candidates), self._target_nonprompt())
        if self._hh_scores.numel() != len(self._global_ids) or self._last_query_scores.numel() != len(self._global_ids):
            raise RuntimeError("Saliency/global-ID mapping is inconsistent")
        if self.method == "fullkv":
            return protected + candidates, {"step_floors": [], "recent_tokens_reserved": 0}
        scores = (self._last_query_scores if self.method == "tova_step" else self._hh_scores).clone()
        selected: set[int] = set()
        floors = []
        recent_count = 0
        step_method = self.method in ("stepkv", "stepkv_no_step_score")
        if step_method:
            scores *= self.TOKEN_WEIGHT
            groups = []
            covered = set()
            for step in self._steps:
                group = [i for i in candidates if step["start"] <= self._global_ids[i] < step["end"]]
                covered.update(group)
                if not group:
                    continue
                if self.method == "stepkv":
                    scores[group] += self.STEP_WEIGHT * step["utility"]
                count = math.ceil(self.STEP_FLOOR * len(group))
                groups.append((step, group, count))
                floors.append({"step_id": step["id"], "resident_before": len(group), "floor": count})
            if covered != set(candidates):
                raise RuntimeError("Boundary pruning found nonprompt tokens outside completed steps")
            if sum(count for _, _, count in groups) > budget:
                raise RuntimeError("The logical ratio budget cannot satisfy the 10% per-step floors")
            for _, group, count in groups:
                selected.update(sorted(group, key=lambda i: (-float(scores[i]), self._global_ids[i]))[:count])
        elif self.method == "h2o_step":
            recent_count = min(self.H2O_RECENT_WINDOW, budget)
            selected.update(candidates[-recent_count:] if recent_count else [])
        if not torch.isfinite(scores).all():
            raise RuntimeError("Nonfinite token/step selection scores")
        ranked = sorted(
            (i for i in candidates if i not in selected),
            key=lambda i: (-float(scores[i]), self._global_ids[i]),
        )
        selected.update(ranked[:budget - len(selected)])
        if len(selected) != budget:
            raise RuntimeError("Selector failed to fill its exact logical budget")
        return protected + sorted(selected), {"step_floors": floors, "recent_tokens_reserved": recent_count}

    @torch.inference_mode()
    def _compact(self, keep: list[int]):
        self._check_cache()
        if keep != sorted(set(keep)) or not keep or keep[0] < 0 or keep[-1] >= len(self._global_ids):
            raise RuntimeError("Invalid shared KV selection")
        if keep[:self._protected_tokens] != list(range(self._protected_tokens)):
            raise RuntimeError("KV selection would evict protected prompt tokens")
        cpu_index = torch.tensor(keep, dtype=torch.long)
        indices = {}
        replacements = []
        for layer_index in self.full_attention_layers:
            layer = self.past_key_values.layers[layer_index]
            if layer.keys.device not in indices:
                indices[layer.keys.device] = cpu_index.to(layer.keys.device)
            index = indices[layer.keys.device]
            replacements.append((
                layer_index, layer.keys.index_select(-2, index), layer.values.index_select(-2, index),
            ))
        new_hh = self._hh_scores.index_select(0, cpu_index)
        new_last = self._last_query_scores.index_select(0, cpu_index)
        for layer_index, keys, values in replacements:
            layer = self.past_key_values.layers[layer_index]
            layer.keys, layer.values = keys, values
        self._global_ids = [self._global_ids[i] for i in keep]
        self._hh_scores, self._last_query_scores = new_hh, new_last
        self._measure_cache()

    def _prune_boundary(self):
        before = self._measure_cache()
        self._peak_boundary_kv_bytes = max(self._peak_boundary_kv_bytes, before["full_attention_kv_bytes"])
        self._peak_boundary_cache_bytes = max(self._peak_boundary_cache_bytes, before["cache_bytes"])
        for device, size in before["full_attention_kv_bytes_by_device"].items():
            self._peak_boundary_kv_bytes_by_device[device] = max(
                self._peak_boundary_kv_bytes_by_device.get(device, 0), size,
            )
        old_ids = list(self._global_ids)
        keep, selection = self._select_keep()
        if len(keep) < len(old_ids):
            self._compact(keep)
        after = self._measure_cache()
        retained = set(self._global_ids)
        dropped = [gid for gid in old_ids if gid not in retained]
        boundary = {
            "step": len(self._steps), "phase": "completed_action_and_environment_return",
            "logical_tokens": len(self._token_ids), "protected_tokens": self._protected_tokens,
            "nonprompt_logical_tokens": len(self._token_ids) - self._protected_tokens,
            "target_nonprompt_tokens": self._target_nonprompt(),
            "before_tokens": len(old_ids), "after_tokens": len(self._global_ids),
            "before_kv_bytes": before["full_attention_kv_bytes"],
            "after_kv_bytes": after["full_attention_kv_bytes"],
            "before_kv_bytes_by_device": before["full_attention_kv_bytes_by_device"],
            "after_kv_bytes_by_device": after["full_attention_kv_bytes_by_device"],
            "before_cache_bytes": before["cache_bytes"], "after_cache_bytes": after["cache_bytes"],
            "evicted_count": len(dropped), "evicted_global_ids": dropped,
            "retained_global_ids": list(self._global_ids),
            **selection,
        }
        if dropped:
            if after["full_attention_kv_bytes"] >= before["full_attention_kv_bytes"]:
                raise RuntimeError("Logical eviction did not reduce actual KV tensor bytes")
            self._prune_events.append(boundary)
        self._boundaries.append(boundary)

    def observe(self, action: str, observation: str, *, valid: bool) -> None:
        self._assert_healthy()
        if not self._ready or self._pending_action is None:
            raise RuntimeError("observe requires one preceding act")
        if not isinstance(action, str) or not isinstance(observation, str) or not isinstance(valid, bool):
            raise TypeError("observe accepts only action/observation strings and an observable bool valid")
        suffix, header, forced_close = self._observation_parts(observation)
        self._append(suffix)
        try:
            self._register_step(action, observation, valid, forced_close)
            self._prune_boundary()
        except Exception as exc:
            self._failure = f"{type(exc).__name__}: {exc}"
            raise
        self._pending_action = None
        self._pending_generated_ids = []
        self._next_header = header
        self._step_start = len(self._token_ids)
        # These pre-pruning logits must not be reused for a new action.
        self._logits = None

    def audit(self) -> dict:
        measured = self._measure_cache()
        logical, resident, protected = len(self._token_ids), len(self._global_ids), self._protected_tokens
        steps = []
        for step in self._steps:
            entry = {key: value for key, value in step.items() if key != "anchors"}
            entry["anchors"] = sorted(step["anchors"])
            entry["resident_tokens"] = sum(step["start"] <= gid < step["end"] for gid in self._global_ids)
            entry["status"] = "completed"
            entry["environment_return_received"] = True
            steps.append(entry)
        pending = self._pending_action is not None and len(self._steps) < self._generated_action_count
        if pending:
            steps.append({
                "id": self._generated_action_count, "start": self._step_start, "end": logical,
                "status": "awaiting_observation", "environment_return_received": False,
                "action": None, "emitted_action": self._pending_action,
                "action_changed_by_caller": None, "observation_sha256": None,
                "valid": None, "novelty": None, "repeat_count": None, "progress": None,
                "citation": None, "utility": None, "anchors": [], "forced_assistant_close": None,
                "generated_token_ids": list(self._pending_generated_ids),
                "resident_tokens": sum(self._step_start <= gid < logical for gid in self._global_ids),
            })
        return copy.deepcopy({
            "implementation_version": self.IMPLEMENTATION_VERSION,
            "method": self.method, "cache_type": type(self.past_key_values).__name__,
            "logical_tokens": logical, "resident_tokens": resident,
            "protected_logical_tokens": protected,
            "protected_resident_tokens": sum(gid < protected for gid in self._global_ids),
            "nonprompt_logical_tokens": logical - protected,
            "nonprompt_resident_tokens": resident - protected,
            "target_nonprompt_tokens": self._target_nonprompt(),
            "retained_global_ids": list(self._global_ids),
            "logical_token_ids_sha256": self._hash(",".join(map(str, self._token_ids))),
            "full_attention_layers": list(self.full_attention_layers),
            "cache_layer_devices": [str(device) for device in self._layer_devices],
            "peak_kv_bytes": self._peak_kv_bytes, "peak_cache_bytes": self._peak_cache_bytes,
            "peak_boundary_kv_bytes": self._peak_boundary_kv_bytes,
            "peak_boundary_cache_bytes": self._peak_boundary_cache_bytes,
            "peak_kv_bytes_by_device": self._peak_kv_bytes_by_device,
            "peak_boundary_kv_bytes_by_device": self._peak_boundary_kv_bytes_by_device,
            "peak_resident_tokens": self._peak_resident_tokens,
            "boundary_count": len(self._boundaries), "prune_event_count": len(self._prune_events),
            "generated_action_count": self._generated_action_count,
            "completed_step_count": len(self._steps), "pending_action_count": int(pending),
            "total_evicted_tokens": sum(event["evicted_count"] for event in self._prune_events),
            "forward_count": self._forward_count, "attention_query_count": self._attention_query_count,
            "extra_scoring_forwards": 0,
            "last_forward_positions": list(self._last_forward_positions),
            "pending_generated_token_ids": list(self._pending_generated_ids),
            "pending_assistant_header_tokens": len(self._next_header),
            "system_prompt": self._system_prompt, "initial_observation_sha256": self._initial_observation_hash,
            "state_failure": self._failure, "steps": steps,
            "boundaries": self._boundaries,
            "prune_events": len(self._prune_events), "prune_history": self._prune_events,
            **measured,
        })

    def fingerprint(self) -> dict:
        revision = getattr(self.model.config, "_commit_hash", None)
        if revision is None:
            parts = Path(self.model_path).parts
            if "snapshots" in parts:
                index = parts.index("snapshots") + 1
                if index < len(parts) and re.fullmatch(r"[0-9a-f]{40}", parts[index]):
                    revision = parts[index]
        return {
            "implementation_version": self.IMPLEMENTATION_VERSION,
            "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "anchor_source_sha256": hashlib.sha256(Path(anchor_utils.__file__).read_bytes()).hexdigest(),
            "model_path": self.model_path, "model_class": type(self.model).__name__,
            "model_revision": revision or self._requested_revision,
            "requested_revision": self._requested_revision,
            "revision_source": "config_or_snapshot_path_or_pinned_load_argument",
            "model_config": self.model.config.to_dict(),
            "parameter_count": self._parameter_count, "parameter_tensor_bytes": self._parameter_bytes,
            "dtype": str(self._dtype), "device": str(self.device), "requested_device": self._requested_device,
            "input_device_source": "model.get_input_embeddings().weight.device",
            "model_devices": [str(device) for device in self._model_devices],
            "cache_layer_devices": [str(device) for device in self._layer_devices],
            "hf_device_map": self._actual_device_map,
            "loading_plan": copy.deepcopy(self._loading_plan),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_names": {
                str(device): torch.cuda.get_device_name(device)
                for device in self._model_devices if device.type == "cuda"
            },
            "versions": {"python": platform.python_version(), "torch": torch.__version__,
                         "transformers": transformers.__version__, "cuda": torch.version.cuda},
            "offline_only": True, "trust_remote_code": False, "injected_cpu_test_model": self._injected,
            "loader_cache_dir": self._loading_plan.get("cache_dir"),
            "runtime_storage_contract": (
                f"HF/Torch/Triton/XDG caches are under {self.STORAGE_ROOT}; "
                "runner supplies executable non-NFS TMPDIR for native planner libraries"
            ),
            "memory_limit_gib": self.memory_limit_gib,
            "memory_limit_scope": "per exposed device, this process's PyTorch allocator only; not a reservation",
            "auto_memory_plan": "one or two GPUs, <=40GiB weights and <=44GiB allocator each, >=4GiB headroom",
            "cpu_disk_offload": "forbidden; explicit CUDA-only device map is validated before loading",
            "method": self.method, "requested_keep_ratio": self.keep_ratio,
            "effective_budget_ratio": 1.0 if self.method == "fullkv" else self.keep_ratio,
            "budget": "protected_prefix + max(1, floor(rho * cumulative_nonprompt_logical_tokens)); zero if empty",
            "budget_scope": "ordinary full-attention KV positions in every decoder layer",
            "pruning_schedule": "after completed action + returned user observation, before next assistant header",
            "terminal_action_audit": "count/report unobserved final emission; no invented validity, utility, or pruning",
            "prune_events_audit": "integer count of actual KV-eviction events; detailed records in prune_history",
            "protected_prefix": "entire initial system + task/observation + assistant generation header",
            "positions": "monotonic logical token IDs for RoPE; explicit packed causal mask for retained KV",
            "shared_selection": "same ordered token IDs in every full-attention layer",
            "full_attention_layers": list(self.full_attention_layers),
            "saliency_device": "CPU float32 after independent reduction on each attention tensor's actual device",
            "attention_implementation": "eager for every forward in every condition",
            "prefill_chunk_size": self.PREFILL_CHUNK_SIZE,
            "prefill_query_key_product_limit": self.MAX_PREFILL_QUERY_KEY_PRODUCT,
            "prefill_chunk_policy": (
                "halve the configured upper bound until q*(logical_start+q) fits the limit; "
                "same logical-position rule for all methods, independent of compacted resident length"
            ),
            "attention_retention": (
                f"one <={self.PREFILL_CHUNK_SIZE}-query forward at a time; "
                "no attention matrices retained between chunks"
            ),
            "timing_scope": "diagnostic only; no comparable latency claims from this eager-attention runtime",
            "token_saliency": (
                "last real query, last full-attention layer, mean over heads" if self.method == "tova_step"
                else "sum over heads and all actual query tokens, mean of last min(3, full_layers) layers"
            ),
            "token_score_history": (
                "latest_forward_only" if self.method in ("stepkv", "stepkv_no_step_score")
                else "latest_query_only" if self.method == "tova_step"
                else "episode_cumulative"
            ),
            "stepkv_token_scoring": (
                "QA H2OScorer.compute_scores reduction on the latest real forward's attention; "
                "CPU float32 accumulation; replace previous scores, including after compaction; "
                "chunked prefill uses only the final chunk, not a chunk/step/episode sum"
            ),
            "score_layers": list(self.full_attention_layers[-self.SCORE_LAYERS:]),
            "h2o_recent_window": self.H2O_RECENT_WINDOW if self.method == "h2o_step" else 0,
            "h2o_recent_window_budget_handling": "reserve min(32, nonprompt budget), inside rather than above budget",
            "tova_saliency": "last real query, last full-attention layer, mean over heads; recent window 0",
            "step_token_weight": self.TOKEN_WEIGHT,
            "step_utility_weight": self.STEP_WEIGHT if self.method == "stepkv" else 0.0,
            "step_floor_ratio": self.STEP_FLOOR if self.method.startswith("stepkv") else 0.0,
            "step_floor": "ceil(0.10 * current resident tokens per completed step); fail if infeasible",
            "step_utility": "clip(0.85 * clip(valid + novelty - 0.3 * prior_action_repeats, -1, 2)"
                            " + 0.15 * log1p(citation), 0, 8)",
            "anchors": "unchanged kv_cache.anchor_utils; observation followed by executed action; cap 64",
            "novelty_reference": "anchors from earlier completed steps; initially empty",
            "citation": "each boundary adds min(1, 1.5 * overlap / min(anchor_set_sizes)); includes current step",
            "score_inputs": ["observable valid", "executed action text", "returned observation text"],
            "selection_tie_break": "ascending logical token ID",
            "hidden_reward_or_success_used": False,
            "decoding": {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0,
                         "logit_processors": [], "max_new_tokens": self.max_new_tokens,
                         "eos_token_ids": sorted(self._eos_ids), "materialize_final_token": True},
            "max_context_tokens": self.max_context_tokens,
            "context_overflow": "raise; no sliding/resetting positions, transcript truncation, or silent fallback",
            "tokenizer_class": type(self.tokenizer).__name__,
            "tokenizer_name": getattr(self.tokenizer, "name_or_path", None),
            "chat_template": self._chat_template, "chat_template_sha256": self._hash(self._chat_template),
            "chat_template_kwargs": {"enable_thinking": False},
            "action_output": "decode generated IDs with skip_special_tokens=True, cleanup=False, then strip",
            "transport": "native template suffixes; reuse emitted im_end, otherwise append explicit im_end",
            "fingerprint_scope": "stable model/runtime/prompt protocol across reset, act, and observe",
            "episode_prompt_record": "audit.system_prompt and audit.initial_observation_sha256",
            "semantic_differences_from_frozen_qa_engine": [
                "Native ALFWorld action-only chat; no thought/search/QA parser or hidden task reward.",
                "All greedy tokens and delimiters preserved; no stop-marker crop, retokenization, or replay.",
                "Genuine query attention replaces synthetic EOS scoring/prefill probes; "
                "StepKV matches the QA score reduction/history semantics, not its attention query source.",
                "Explicit logical RoPE and packed masks replace physical-cache position reuse.",
                "Every completed environment step is eligible, including the first; no token-level pruning.",
                "Step utility keeps legacy weights/self-reuse, substitutes observable validity for QA success heuristics "
                "and executed action text for thought anchors.",
                "H2O-step/TOVA-step are shared-index, step-boundary adaptations, not official reproductions.",
                "Ordinary full-attention Qwen2.5 only; CUDA-only balanced model parallelism is allowed.",
            ],
        }
