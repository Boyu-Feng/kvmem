"""LazyEviction official-code selector, adapted to multi-turn Qwen2/HF 5.

Source: Halo-949/LazyEviction, commit 051d663990cd1628d72db3e0e849bb20d4bd0e0c.
The recurrence update and sigmoid MRI score below are ported from
model/temp_cacheobs.py and model/kv_utils.py (Apache-2.0; vendor reference).
Changes: instance-local int64 counters, protected prompt, growing ratio budget,
real-query observation ingestion, absolute RoPE positions, transactional stops.
This is a separate runtime, not the controlled-v2 EOS-probe runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import time
from types import MethodType

import torch
import torch.nn.functional as F
from transformers import DynamicCache
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

from models.QwenLLMWithKVCache import QwenLLMWithKVCache

OFFICIAL_COMMIT = "051d663990cd1628d72db3e0e849bb20d4bd0e0c"


@dataclass(frozen=True)
class RecurrenceState:
    """Per-layer, per-query-head resident IDs and recurrence, never in-place."""
    ids: torch.Tensor
    last: torch.Tensor
    mri: torch.Tensor

    def observe(self, weights, clock, new_ids):
        """Official record_recurrence, with exact integer timestamps.

        weights are the real query's softmax over the ENTIRE resident cache,
        including the protected prompt. No head/layer pooling or EOS probe.
        """
        if weights.shape[-2] != 1 or new_ids.shape[-1] != 1:
            raise ValueError("Recurrence requires one actual input query at a time")
        ids = torch.cat((self.ids, new_ids), dim=-1)
        last = torch.cat((self.last, torch.full_like(new_ids, clock - 1)), dim=-1)
        mri = torch.cat((self.mri, torch.zeros_like(new_ids)), dim=-1)
        return ids, last, mri


def update_recurrence(state, weights, clock, new_ids, alpha):
    ids, old_last, old_mri = state.observe(weights, clock, new_ids)
    if old_last.shape != weights.squeeze(-2).shape:
        raise RuntimeError("Recurrence and real-query attention shape mismatch")
    last = torch.where(weights.squeeze(-2) >= alpha, clock, old_last)
    return RecurrenceState(ids, last, torch.maximum(old_mri, last - old_last))


def official_importance(state, clock):
    """Exact official-code sigmoid formula (different from arXiv v1 Eq. 2)."""
    mri = state.mri.float()
    last = state.last.float()
    s = torch.where(mri == 0, 0.0, 2 / (1 + torch.exp(mri - 1)))
    return 2 / (1 + torch.exp(1 + (clock - last) / (mri + 1e-5))) + s


def retention_indices(state, clock, capacity, window, prompt_length=0):
    """Official per-head top-k + recent W; prompt protection is an adaptation."""
    length = state.ids.shape[-1]
    if length <= capacity or clock % window:
        return None
    recent = min(window, length - prompt_length)
    keep_old = capacity - prompt_length - recent
    if keep_old < 0:
        raise ValueError("Capacity must include protected prompt and recent window")
    prefix = torch.arange(prompt_length, device=state.ids.device)
    suffix = torch.arange(length - recent, length, device=state.ids.device)
    shape = state.ids.shape[:-1]
    top = official_importance(state, clock)[..., prompt_length:length - recent].topk(keep_old, dim=-1).indices
    # Preserve the official top-k order; no shared-head selector or sorting.
    return torch.cat((prefix.expand(*shape, -1), top + prompt_length,
                      suffix.expand(*shape, -1)), dim=-1)


def gather_state(state, indices):
    return RecurrenceState(*(x.gather(-1, indices) for x in (state.ids, state.last, state.mri)))


class LogicalClock:
    """Harness compatibility without pretending one ID map describes all heads."""
    def __init__(self):
        self.next_global_id = 0
        self.current_step = None
        self.cache_length = 0

    def set_current_step(self, step):
        self.current_step = step

    def print_final_summary(self):
        print(f"[LazyEviction] logical_tokens={self.next_global_id}, resident_slots_per_head={self.cache_length}")


def lazy_attention_forward(attention, hidden_states, position_embeddings,
                           attention_mask, past_key_values=None, cache_position=None, **kwargs):
    owner = attention._lazy_owner
    if past_key_values is None:
        raise RuntimeError("LazyEviction runtime requires a live DynamicCache")
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, attention.head_dim)
    query = attention.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    keys = attention.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    values = attention.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    cos, sin = position_embeddings
    query, keys = apply_rotary_pos_emb(query, keys, cos, sin)
    # Canonical implementation stores different resident tokens for each QUERY
    # head, so repeated GQA keys are physically stored. Their bytes are counted.
    keys = repeat_kv(keys, attention.num_key_value_groups)
    values = repeat_kv(values, attention.num_key_value_groups)
    keys, values = past_key_values.update(keys, values, attention.layer_idx,
        {"sin": sin, "cos": cos, "cache_position": cache_position})
    output = F.scaled_dot_product_attention(query, keys, values, attn_mask=attention_mask,
        dropout_p=0.0, is_causal=attention_mask is None and input_shape[1] > 1,
        scale=attention.scaling)
    layer = attention.layer_idx
    new_ids = torch.arange(owner._forward_start, owner._forward_start + input_shape[1],
                           device=keys.device).view(1, 1, -1).expand(*keys.shape[:2], -1)
    if owner._prefilling_prompt:
        if layer in owner.recurrence:
            raise RuntimeError("Prompt prefill cannot overwrite an existing episode")
        owner.recurrence[layer] = RecurrenceState(new_ids, torch.zeros_like(new_ids), torch.zeros_like(new_ids))
    else:
        if query.shape[-2] != 1:
            raise RuntimeError("Observation and decode must use individual real queries")
        weights = torch.matmul(query, keys.transpose(2, 3)) / math.sqrt(attention.head_dim)
        weights = F.softmax(weights, dim=-1, dtype=torch.float32).to(query.dtype)
        state = update_recurrence(owner.recurrence[layer], weights, owner._query_clock,
                                  new_ids, owner.alpha)
        owner.recurrence[layer] = state
        owner._record_memory()
        indices = retention_indices(state, owner._query_clock, owner._forward_capacity,
                                    owner.window, owner.prompt_length)
        if indices is not None:
            owner.recurrence[layer] = gather_state(state, indices)
            expanded = indices.unsqueeze(-1).expand(-1, -1, -1, attention.head_dim)
            past_key_values.layers[layer].keys = keys.gather(2, expanded)
            past_key_values.layers[layer].values = values.gather(2, expanded)
    output = output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
    return attention.o_proj(output), None


class LazyEvictionLLM(QwenLLMWithKVCache):
    RUNTIME_VERSION = "lazyeviction-multiturn-v1"

    def __init__(self, model_path, kv_config=None, token_tracker=None):
        config = dict(kv_config or {})
        # Model loading only; this runtime owns eviction and per-head identity.
        super().__init__(model_path, {**config, "pruning_mode": "none"}, None)
        self.kv_config = config
        self._configure()
        self.reset()

    def _configure(self):
        self.ratio = float(self.kv_config.get("cache_ratio", 0.2))
        self.window = int(self.kv_config.get("lazy_window", 175))
        self.alpha = float(self.kv_config.get("lazy_alpha", 0.0001))
        if not 0 < self.ratio <= 1 or self.window < 1 or self.alpha < 0:
            raise ValueError("Invalid LazyEviction budget, window, or alpha")
        if self.model.config.model_type != "qwen2":
            raise ValueError("This audited adapter supports Qwen2/Qwen2.5 only")
        if not self.kv_config.get("protect_prompt", True):
            raise ValueError("This experiment always protects the initial prompt")
        if self.model.config.sliding_window is not None and self.model.config.use_sliding_window:
            raise ValueError("Sliding-window model configurations are unsupported")
        self.kv_manager = None
        self.pruning_enabled = True
        for layer in self.model.model.layers:
            layer.self_attn._lazy_owner = self
            layer.self_attn.forward = MethodType(lazy_attention_forward, layer.self_attn)

    def reset(self):
        self.past_key_values = None
        self.recurrence = {}
        self.token_tracker = LogicalClock()
        self.current_cache_len = 0
        self.prompt_length = 0
        self._prefilling_prompt = False
        self._global_token_id_log = []
        self._all_token_ids = []
        self._pending_snapshots = {}
        self._exported_cache_snapshot = ()
        self.cache_measurements = []
        self.window_checks = 0
        self.executed_query_count = 0
        self.rewind_count = 0
        self.last_forward_positions = []
        self.peak_cache_tokens = 0
        self.peak_cache_bytes = 0
        self.peak_tracking_bytes = 0
        self.peak_stop_snapshot_extra_bytes = 0
        self.peak_cache_and_tracking_bytes = 0
        self.timing_stats = dict(prefill_time=0.0, decode_time=0.0, scoring_time=0.0, pruning_time=0.0)

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _layers(self):
        return [(x.keys, x.values) for x in self.past_key_values.layers if x.is_initialized] if self.past_key_values else []

    @staticmethod
    def _storage_sizes(tensors):
        return {(str(x.device), x.untyped_storage().data_ptr()): x.untyped_storage().nbytes()
                for x in tensors if x.numel()}

    def _memory(self):
        pairs = self._layers()
        kv = [x for pair in pairs for x in pair]
        tracking = [x for state in self.recurrence.values() for x in (state.ids, state.last, state.mri)]
        active = self._storage_sizes(kv + tracking)
        snapshots = []
        for snapshot in self._pending_snapshots.values():
            snapshots.extend(x for pair in snapshot["kv"] for x in pair)
            snapshots.extend(x for state in snapshot["recurrence"].values() for x in (state.ids, state.last, state.mri))
        snapshot_storage = self._storage_sizes(snapshots)
        extra = sum(size for ptr, size in snapshot_storage.items() if ptr not in active)
        exported = self._storage_sizes([x for pair in self._exported_cache_snapshot for x in pair])
        return dict(cache_bytes=sum(self._storage_sizes(kv).values()),
                    tracking_bytes=sum(self._storage_sizes(tracking).values()),
                    stop_snapshot_extra_bytes=extra, returned_cache_snapshot_bytes=sum(exported.values()))

    def _record_memory(self):
        memory = self._memory()
        self.peak_cache_bytes = max(self.peak_cache_bytes, memory["cache_bytes"])
        self.peak_tracking_bytes = max(self.peak_tracking_bytes, memory["tracking_bytes"])
        self.peak_stop_snapshot_extra_bytes = max(self.peak_stop_snapshot_extra_bytes, memory["stop_snapshot_extra_bytes"])
        self.peak_cache_and_tracking_bytes = max(self.peak_cache_and_tracking_bytes, sum(memory.values()))
        pairs = self._layers()
        self.peak_cache_tokens = max(self.peak_cache_tokens, *(k.shape[2] for k, _ in pairs), 0)
        return memory

    def _record_cache(self):
        pairs = self._layers()
        lengths = [k.shape[2] for k, _ in pairs]
        if len(set(lengths)) > 1:
            raise RuntimeError("Per-layer capacities differ")
        for layer, (keys, values) in enumerate(pairs):
            if keys.shape != values.shape or self.recurrence[layer].ids.shape != keys.shape[:3]:
                raise RuntimeError("Per-head cache/identity/recurrence mismatch")
        self.current_cache_len = lengths[0] if lengths else 0
        self.token_tracker.cache_length = self.current_cache_len
        return self._record_memory()

    @torch.inference_mode()
    def _forward(self, ids, timing_key):
        if not ids:
            raise ValueError("Cannot append an empty input")
        if not self._prefilling_prompt and len(ids) != 1:
            raise ValueError("Multi-turn inputs are ingested one real query at a time")
        start = self.token_tracker.next_global_id
        self._forward_start = start
        self._query_clock = start + len(ids) - self.prompt_length
        nonprompt = max(0, self._query_clock)
        self._forward_capacity = self.prompt_length + max(min(self.window, nonprompt), int(nonprompt * self.ratio))
        before = self.current_cache_len + len(ids)
        if self.past_key_values is None:
            self.past_key_values = DynamicCache(config=self.model.config)
        positions = torch.arange(start, start + len(ids), device=self.device).unsqueeze(0)
        self.last_forward_positions = positions[0].tolist()
        self._sync()
        begin = time.perf_counter()
        outputs = self.model(input_ids=torch.tensor([ids], device=self.device),
            position_ids=positions,
            cache_position=torch.arange(self.current_cache_len, before, device=self.device),
            attention_mask=torch.ones((1, before), dtype=torch.long, device=self.device),
            past_key_values=self.past_key_values, use_cache=True, return_dict=True,
            output_attentions=False)
        self._sync()
        self.timing_stats[timing_key] += time.perf_counter() - begin
        self.token_tracker.next_global_id += len(ids)
        self._global_token_id_log.extend(ids)
        self._all_token_ids.extend(ids)
        memory = self._record_cache()
        if not self._prefilling_prompt:
            self.executed_query_count += 1
            if self._query_clock % self.window == 0:
                self.window_checks += 1
                if self.current_cache_len != min(before, self._forward_capacity):
                    raise RuntimeError("LazyEviction did not meet the feasible per-head window budget")
                self.cache_measurements.append(dict(phase=timing_key, step=self.token_tracker.current_step,
                    clock=self._query_clock, logical_tokens=self.token_tracker.next_global_id,
                    before_tokens=before, after_tokens=self.current_cache_len,
                    tokens_evicted=before - self.current_cache_len,
                    tokens_evicted_unit="slots per query head; selected identities differ by layer/head",
                    target_tokens=self._forward_capacity, protected_tokens=self.prompt_length,
                    nominal_ratio_target_tokens=self.prompt_length + int(nonprompt * self.ratio),
                    recent_window_floor_tokens=min(self.window, nonprompt),
                    budget_floor_added_tokens=max(0, min(self.window, nonprompt) - int(nonprompt * self.ratio)),
                    before_bytes=before * sum((k.numel() * k.element_size() + v.numel() * v.element_size()) // k.shape[2]
                                            for k, v in self._layers()),
                    after_bytes=memory["cache_bytes"],
                    window=self.window, alpha=self.alpha, pruned=self.current_cache_len < before,
                    **memory))
        return outputs.logits[:, -1, :]

    def _append(self, ids, timing_key="decode_time"):
        if self._prefilling_prompt:
            return self._forward(ids, timing_key)
        if not ids:
            raise ValueError("Cannot append an empty token block")
        for token in ids:
            logits = self._forward([token], timing_key)
        return logits

    def _snapshot(self):
        return dict(kv=self._layers(), recurrence=dict(self.recurrence),
                    logical_end=self.token_tracker.next_global_id,
                    events=len(self.cache_measurements), window_checks=self.window_checks)

    def _restore(self, snapshot):
        # All tensor transitions are out-of-place: these pointers preserve both
        # evicted tokens and recurrence values from before a stop-marker query.
        for layer, (keys, values) in zip(self.past_key_values.layers, snapshot["kv"]):
            layer.keys, layer.values = keys, values
        self.recurrence = dict(snapshot["recurrence"])
        end = snapshot["logical_end"]
        self.token_tracker.next_global_id = end
        self._global_token_id_log = self._global_token_id_log[:end]
        self._all_token_ids = self._all_token_ids[:end]
        self.cache_measurements = self.cache_measurements[:snapshot["events"]]
        self.window_checks = snapshot["window_checks"]
        self.rewind_count += 1
        self._record_cache()

    def _rewind_to(self, logical_end):
        if logical_end == self.token_tracker.next_global_id:
            return
        candidates = [s for s in self._pending_snapshots.values() if s["logical_end"] == logical_end]
        if not candidates:
            raise RuntimeError("Rewind requires an exact pre-marker state snapshot")
        self._restore(candidates[0])

    def _generate_block(self, logits, max_new_tokens, stop_strings):
        generated = []
        self._pending_snapshots = {}
        first_id = self.token_tracker.next_global_id
        stops = [s for s in (stop_strings or []) if s]
        eos = self.model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])

        def before_character(cut):
            keep = len(generated)
            while keep and len(self.tokenizer.decode(generated[:keep], skip_special_tokens=True)) > cut:
                keep -= 1
            return keep

        for _ in range(max_new_tokens):
            token = int(logits.argmax(dim=-1).item())
            if token in eos:
                break
            text = self.tokenizer.decode(generated + [token], skip_special_tokens=True)
            boundaries = [text.find(stop) for stop in stops if stop in text]
            if boundaries:
                cut = min(boundaries)
                keep = before_character(cut)
                self._rewind_to(first_id + keep)
                generated = generated[:keep]
                prefix = self.tokenizer.decode(generated, skip_special_tokens=True)
                if not text[:cut].startswith(prefix):
                    raise RuntimeError("Stop boundary cannot be aligned to decoded prefix")
                self._pending_snapshots = {}
                remainder = text[len(prefix):cut]
                if remainder:
                    tail = self.tokenizer(remainder, add_special_tokens=False).input_ids
                    for tail_token in tail:
                        generated.append(tail_token)
                        self._append([tail_token])
                break
            # Keep snapshots only for incomplete stop prefixes, including a BPE
            # token containing action text followed by the marker's first chars.
            starts = {len(text) - n for stop in stops for n in range(1, min(len(stop), len(text) + 1))
                      if text.endswith(stop[:n])}
            needed = {}
            for cut in starts:
                keep = before_character(cut)
                if keep == len(generated):
                    needed[keep] = self._snapshot()
                elif keep in self._pending_snapshots:
                    needed[keep] = self._pending_snapshots[keep]
                else:
                    raise RuntimeError("Tokenizer changed a stop prefix without a recoverable checkpoint")
            self._pending_snapshots = needed
            generated.append(token)
            logits = self._append([token])
        self._pending_snapshots = {}
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate_first(self, prompt_text, max_new_tokens=256, stop_strings=None):
        self.reset()
        self.token_tracker.set_current_step(1)
        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_text}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer(text, add_special_tokens=True).input_ids
        self.prompt_length = len(ids)
        self._prefilling_prompt = True
        try:
            logits = self._append(ids, "prefill_time")
        finally:
            self._prefilling_prompt = False
        response = self._generate_block(logits, max_new_tokens, stop_strings)
        # The legacy harness holds these returned snapshots throughout an
        # episode; count their real allocations separately from the live cache.
        prompt_kv = tuple((k[:, :, :len(ids)].clone(), v[:, :, :len(ids)].clone()) for k, v in self._layers())
        generated_kv = tuple((k[:, :, len(ids):].clone(), v[:, :, len(ids):].clone()) for k, v in self._layers())
        self._exported_cache_snapshot = prompt_kv + generated_kv
        self._record_memory()
        return response, prompt_kv, generated_kv

    def generate_incremental(self, new_text, max_new_tokens=256, stop_strings=None):
        ids = self.tokenizer(new_text, add_special_tokens=False).input_ids
        logits = self._append(ids, "prefill_time")
        return self._generate_block(logits, max_new_tokens, stop_strings)

    def get_pruning_history(self):
        return self.cache_measurements

    def get_step_pruned(self, step):
        return any(event["pruned"] and event["step"] == step for event in self.cache_measurements)

    def get_stats(self):
        heads = self.model.config.num_attention_heads
        kv_heads = self.model.config.num_key_value_heads
        return dict(self.timing_stats, runtime_version=self.RUNTIME_VERSION,
            attention_source="real_query_all_layers_all_query_heads",
            timing_scope="recurrence/scoring/selection included in prefill/decode time; not separately synchronized",
            pruning_schedule="every_W_nonprompt_input_queries",
            official_commit=OFFICIAL_COMMIT, window=self.window, alpha=self.alpha,
            ratio=self.ratio, window_checks=self.window_checks,
            total_prune_count=sum(event["pruned"] for event in self.cache_measurements),
            logical_query_clock=self.token_tracker.next_global_id - self.prompt_length,
            executed_query_count=self.executed_query_count, rewind_count=self.rewind_count,
            current_cache_len=self.current_cache_len,
            cache_tokens_unit="resident token slots per query head",
            stored_heads=heads, native_kv_heads=kv_heads, head_expansion_factor=heads / kv_heads,
            peak_cache_tokens=self.peak_cache_tokens, peak_cache_bytes=self.peak_cache_bytes,
            peak_tracking_bytes=self.peak_tracking_bytes,
            peak_stop_snapshot_extra_bytes=self.peak_stop_snapshot_extra_bytes,
            peak_cache_and_tracking_bytes=self.peak_cache_and_tracking_bytes,
            tracking_dtype="int64 IDs/timestamps/MRI; 24 bytes per resident layer-head-token when dense, actual storage bytes account for shared prompt-ID views",
            cache_measurements=self.cache_measurements, **self._memory())
