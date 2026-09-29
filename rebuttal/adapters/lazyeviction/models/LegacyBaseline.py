"""LazyEviction hooks for the unchanged, pre-rebuttal Qwen KV wrapper.

Generation, initial-turn behavior, stop handling, and HF-supplied RoPE positions
are delegated to QwenLLMWithKVCache. Only this baseline's attention/cache state
is adapted. See LEGACY_LAZY_SCOPE.md for the resulting comparison boundary.
"""
from __future__ import annotations

import math
from types import MethodType

import torch
import torch.nn.functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

from models.QwenLLMWithKVCache import QwenLLMWithKVCache
from models.LazyEvictionLLM import (OFFICIAL_COMMIT, RecurrenceState,
                                  update_recurrence, official_importance, gather_state)

LEGACY_METHOD = "lazyeviction"
LEGACY_SCOPE = (
    "Original QwenLLMWithKVCache generation/stop/RoPE engine with LazyEviction "
    "real-query, per-layer/per-query-head recurrence and selection hooks. Initial "
    "generation records recurrence but never evicts. Batched observation prefill "
    "is unchanged: recurrence observes its causal query rows, with crossed-window "
    "eviction deferred to the end of that prefill. Protected prompt and cumulative "
    "processed nonprompt-query ratio budget are adaptations. Legacy post-generation "
    "stop cropping also crops per-head metadata; it does not rewind already "
    "observed attention or earlier eviction decisions. HF-provided positions, "
    "generation processors and the legacy unmaterialized final-token behavior "
    "are preserved. Physical head-expanded KV bytes are reported separately "
    "(28 query versus 4 native KV heads, a 7x head expansion on Qwen2.5-7B)."
)
LEGACY_CONFIG_OVERRIDE = dict(
    pruning_mode="none", legacy_baseline=LEGACY_METHOD,
    cache_ratio=.2, protect_prompt=True, lazy_window=175, lazy_alpha=.0001,
    attn_mode="scoring_forward", raise_on_runtime_error=True,
)


def _select_at_boundary(state, clock, capacity, window, prompt_length):
    """Official sigmoid/top-k selection; scheduling is handled by the hook."""
    length = state.ids.shape[-1]
    if length <= capacity:
        return None
    recent = min(window, length - prompt_length)
    old_count = capacity - prompt_length - recent
    if old_count < 0:
        raise RuntimeError("LazyEviction target violates the protected-window floor")
    device, shape = state.ids.device, state.ids.shape[:-1]
    prefix = torch.arange(prompt_length, device=device).expand(*shape, -1)
    suffix = torch.arange(length - recent, length, device=device).expand(*shape, -1)
    older = official_importance(state, clock)[..., prompt_length:length - recent]
    selected = older.topk(old_count, dim=-1).indices + prompt_length
    return torch.cat((prefix, selected, suffix), dim=-1)


def _legacy_lazy_attention(attention, hidden_states, position_embeddings,
                           attention_mask, past_key_values=None, cache_position=None, **kwargs):
    owner = attention._legacy_lazy_owner
    if past_key_values is None:
        raise RuntimeError("Legacy LazyEviction requires HF's cache-enabled path")
    owner._legacy_active_cache = past_key_values
    input_shape = hidden_states.shape[:-1]
    q_len = input_shape[1]
    hidden_shape = (*input_shape, -1, attention.head_dim)
    query = attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    keys = attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    values = attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    # Positions come solely from the original wrapper / HF generation path.
    cos, sin = position_embeddings
    query, keys = apply_rotary_pos_emb(query, keys, cos, sin)
    keys = repeat_kv(keys, attention.num_key_value_groups)
    values = repeat_kv(values, attention.num_key_value_groups)
    keys, values = past_key_values.update(keys, values, attention.layer_idx,
        {"sin": sin, "cos": cos, "cache_position": cache_position})
    past_length = keys.shape[2] - q_len
    causal_mask = attention_mask[..., :keys.shape[2]] if attention_mask is not None else None
    output = F.scaled_dot_product_attention(query, keys, values,
        attn_mask=causal_mask, dropout_p=0.0,
        is_causal=causal_mask is None and q_len > 1, scale=attention.scaling)
    layer = attention.layer_idx
    if owner._legacy_first_prefill:
        ids = torch.arange(q_len, device=keys.device).view(1, 1, -1).expand(*keys.shape[:2], -1)
        owner._legacy_recurrence[layer] = RecurrenceState(ids, torch.zeros_like(ids), torch.zeros_like(ids))
    else:
        state = owner._legacy_recurrence[layer]
        if state.ids.shape[-1] != past_length:
            raise RuntimeError("Legacy per-head metadata is not aligned with actual KV")
        for row in range(q_len):
            # The old engine keeps observation prefill batched. These are its
            # genuine causal query rows, never an EOS or synthetic query.
            visible = past_length + row + 1
            weights = torch.matmul(query[:, :, row:row + 1], keys[:, :, :visible].transpose(2, 3)) / math.sqrt(attention.head_dim)
            weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
            new_ids = torch.full((*keys.shape[:2], 1), owner._legacy_next_query_id + row,
                                 dtype=torch.long, device=keys.device)
            state = update_recurrence(state, weights, owner._legacy_clock + row + 1,
                                      new_ids, owner._legacy_alpha)
        owner._legacy_recurrence[layer] = state
        owner._record_legacy_memory()
        if owner._legacy_forward_due and not owner._legacy_initial_generation:
            indices = _select_at_boundary(state, owner._legacy_clock + q_len,
                owner._legacy_forward_capacity, owner._legacy_window, owner._legacy_prompt_length)
            if indices is not None:
                owner._legacy_recurrence[layer] = gather_state(state, indices)
                expanded = indices.unsqueeze(-1).expand(-1, -1, -1, attention.head_dim)
                past_key_values.layers[layer].keys = keys.gather(2, expanded)
                past_key_values.layers[layer].values = values.gather(2, expanded)
                owner._legacy_forward_evictions[layer] = keys.shape[2] - indices.shape[2]
    output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    return attention.o_proj(output), None


class LegacyBaselineLLM(QwenLLMWithKVCache):
    """The old generator with only baseline-specific attention and bookkeeping."""
    RUNTIME_VERSION = "legacy-qwen-lazyeviction-hooks-v1"

    def __init__(self, model_path, kv_config=None, token_tracker=None):
        config = {**LEGACY_CONFIG_OVERRIDE, **(kv_config or {}), "pruning_mode": "none"}
        # No global shared token mapper: different heads retain different IDs.
        super().__init__(model_path, config, token_tracker=None)
        self._install_legacy_hooks()
        self.reset()

    def _install_legacy_hooks(self):
        self._legacy_ratio = float(self.kv_config.get("cache_ratio", .2))
        self._legacy_window = int(self.kv_config.get("lazy_window", 175))
        self._legacy_alpha = float(self.kv_config.get("lazy_alpha", .0001))
        if not 0 < self._legacy_ratio <= 1 or self._legacy_window < 1 or self._legacy_alpha < 0:
            raise ValueError("Invalid legacy LazyEviction configuration")
        if not self.kv_config.get("protect_prompt", True):
            raise ValueError("This baseline adaptation protects the initial prompt")
        if self.model.config.model_type != "qwen2" or (self.model.config.use_sliding_window and self.model.config.sliding_window is not None):
            raise ValueError("Audited legacy hooks support non-sliding Qwen2/Qwen2.5")
        for layer in self.model.model.layers:
            layer.self_attn._legacy_lazy_owner = self
            layer.self_attn.forward = MethodType(_legacy_lazy_attention, layer.self_attn)
        self._legacy_pre_handle = self.model.register_forward_pre_hook(self._legacy_before_forward, with_kwargs=True)
        self._legacy_post_handle = self.model.register_forward_hook(self._legacy_after_forward, with_kwargs=True)

    def reset(self):
        super().reset()
        self._legacy_initial_generation = True
        self._legacy_phase = "initial_generate"
        self._legacy_first_prefill = True
        self._legacy_recurrence = {}
        self._legacy_active_cache = None
        self._legacy_clock = 0
        self._legacy_next_query_id = 0
        self._legacy_prompt_length = 0
        self._legacy_step = 1
        self._legacy_evicted_slots = 0
        self._legacy_exported = ()
        self._legacy_peak_kv_bytes = 0
        self._legacy_peak_tracking_bytes = 0
        self._legacy_peak_total_bytes = 0
        self.cache_measurements = []
        self._legacy_crop_events = []

    def _legacy_before_forward(self, module, args, kwargs):
        inputs = kwargs.get("input_ids", args[0] if args else None)
        if inputs is None or inputs.shape[0] != 1:
            raise ValueError("Legacy LazyEviction supports batch-size-one input_ids")
        q_len = inputs.shape[1]
        self._legacy_forward_q_len = q_len
        self._legacy_first_prefill = not self._legacy_recurrence
        if self._legacy_first_prefill:
            self._legacy_prompt_length = q_len
        self._legacy_forward_due = not self._legacy_first_prefill and (
            (self._legacy_clock + q_len) // self._legacy_window > self._legacy_clock // self._legacy_window)
        count = self._legacy_clock + (0 if self._legacy_first_prefill else q_len)
        self._legacy_forward_capacity = self._legacy_prompt_length + max(min(self._legacy_window, count), int(count * self._legacy_ratio))
        self._legacy_forward_evictions = {}

    def _legacy_after_forward(self, module, args, kwargs, output):
        q_len = self._legacy_forward_q_len
        self._legacy_next_query_id += q_len
        if not self._legacy_first_prefill:
            self._legacy_clock += q_len
        removed = set(self._legacy_forward_evictions.values())
        if len(removed) > 1 or (removed and len(self._legacy_forward_evictions) != len(self.model.model.layers)):
            raise RuntimeError("Legacy per-head selection produced inconsistent layer capacities")
        evicted = next(iter(removed), 0)
        # Preserve the old engine's reported-vs-materialized length discrepancy.
        # Subtract only our actual evictions, without repairing missing final KV.
        self.current_cache_len -= evicted
        self._legacy_evicted_slots += evicted
        memory = self._record_legacy_memory()
        if self._legacy_forward_due:
            length = self._legacy_actual_length()
            nominal = self._legacy_prompt_length + int(self._legacy_clock * self._legacy_ratio)
            self.cache_measurements.append(dict(
                step=self._legacy_step, phase=self._legacy_phase,
                query_clock=self._legacy_clock, query_block_tokens=q_len,
                deferred_prefill_window=q_len > 1,
                initial_generation_eviction_disabled=self._legacy_initial_generation,
                prompt_tokens=self._legacy_prompt_length,
                nominal_ratio_target_tokens=nominal,
                effective_target_tokens=self._legacy_forward_capacity,
                window_floor_added_tokens=self._legacy_forward_capacity - nominal,
                before_tokens=length + evicted, after_tokens=length,
                pruned=evicted > 0, tokens_evicted=evicted,
                token_unit="slots per query head; retained identities differ across layers/heads",
                **memory))
        return output

    def _legacy_pairs(self):
        cache = self._legacy_active_cache
        return [(layer.keys, layer.values) for layer in cache.layers if layer.is_initialized] if cache is not None else []

    def _legacy_actual_length(self):
        pairs = self._legacy_pairs()
        lengths = [k.shape[2] for k, _ in pairs]
        if len(set(lengths)) > 1:
            raise RuntimeError("Legacy LazyEviction layer lengths disagree")
        return lengths[0] if lengths else 0

    @staticmethod
    def _bytes(tensors):
        storage = {(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
                   for t in tensors if t.numel()}
        return sum(storage.values())

    def _record_legacy_memory(self):
        kv_bytes = self._bytes([x for pair in self._legacy_pairs() for x in pair])
        tracking_bytes = self._bytes([x for state in self._legacy_recurrence.values() for x in (state.ids, state.last, state.mri)])
        exported_bytes = self._bytes([x for pair in self._legacy_exported for x in pair])
        self._legacy_peak_kv_bytes = max(self._legacy_peak_kv_bytes, kv_bytes)
        self._legacy_peak_tracking_bytes = max(self._legacy_peak_tracking_bytes, tracking_bytes)
        self._legacy_peak_total_bytes = max(self._legacy_peak_total_bytes, kv_bytes + tracking_bytes + exported_bytes)
        return dict(cache_bytes=kv_bytes, tracking_bytes=tracking_bytes,
                    returned_cache_snapshot_bytes=exported_bytes)

    def generate_first(self, *args, **kwargs):
        result = super().generate_first(*args, **kwargs)
        self._legacy_initial_generation = False
        self._legacy_exported = result[1] + result[2]
        self._record_legacy_memory()
        return result

    def generate_incremental(self, *args, **kwargs):
        self._legacy_phase = "observation_prefill"
        self._legacy_step += 1
        return super().generate_incremental(*args, **kwargs)

    def _decode(self, *args, **kwargs):
        previous = self._legacy_phase
        self._legacy_phase = "decode"
        try:
            return super()._decode(*args, **kwargs)
        finally:
            self._legacy_phase = previous

    def truncate_cache(self, keep_token_count):
        before = self._legacy_actual_length()
        reported_before = self.current_cache_len
        super().truncate_cache(keep_token_count)
        length = self._legacy_actual_length()
        for layer, state in list(self._legacy_recurrence.items()):
            self._legacy_recurrence[layer] = RecurrenceState(
                state.ids[..., :length], state.last[..., :length], state.mri[..., :length])
        self._legacy_crop_events.append(dict(
            requested_keep=keep_token_count, reported_length_before=reported_before,
            actual_before=before, actual_after=length, query_clock=self._legacy_clock,
            recurrence_policy="crop physical prefix exactly with legacy KV; preserve prior attention history and monotonic query clock"))
        self._record_legacy_memory()

    def get_pruning_history(self):
        return self.cache_measurements

    def get_stats(self):
        result = super().get_stats()
        result.update(total_prune_count=sum(e["pruned"] for e in self.cache_measurements))
        return result

    def legacy_audit(self):
        config = self.model.config
        actual = self._legacy_actual_length()
        for layer, (k, _) in enumerate(self._legacy_pairs()):
            if self._legacy_recurrence[layer].ids.shape != k.shape[:3]:
                raise RuntimeError("Legacy audit found cache/identity mismatch")
        return dict(runtime_version=self.RUNTIME_VERSION, method=LEGACY_METHOD,
            scope=LEGACY_SCOPE, official_commit=OFFICIAL_COMMIT,
            hyperparameter_source="official eval_qwen.sh MATH example; unchanged W=175/alpha=1e-4 defaults, not tuned on smoke accuracy",
            window=self._legacy_window, alpha=self._legacy_alpha, ratio=self._legacy_ratio,
            processed_nonprompt_queries=self._legacy_clock,
            actual_cache_slots_per_head=actual, legacy_reported_cache_len=self.current_cache_len,
            legacy_reported_minus_actual=self.current_cache_len - actual,
            stored_heads=config.num_attention_heads, native_kv_heads=config.num_key_value_heads,
            head_expansion_factor=config.num_attention_heads / config.num_key_value_heads,
            total_prune_count=sum(e["pruned"] for e in self.cache_measurements),
            total_evicted_slots_per_head=self._legacy_evicted_slots,
            peak_cache_bytes=self._legacy_peak_kv_bytes,
            peak_tracking_bytes=self._legacy_peak_tracking_bytes,
            peak_cache_tracking_returned_bytes=self._legacy_peak_total_bytes,
            cache_measurements=self.cache_measurements, stop_crops=self._legacy_crop_events,
            recurrence_numeric_adaptation="int64 timestamps/MRI; official-code sigmoid formula in float32",
            **self._record_legacy_memory())
