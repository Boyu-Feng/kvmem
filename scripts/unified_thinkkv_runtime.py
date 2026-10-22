"""Selection-only ThinKV on the *original* QA/HF generator.

The canonical wrapper uses physical, not absolute logical, RoPE positions after
pruning. New queries therefore start at resident KV length; selected post-RoPE
keys are never re-rotated. Logical IDs are only trajectory/selection accounting.
HF's sequence-slicing index is separate from this physical position.

A stop-bearing generation block is protected until the original posthoc crop.
This conservative, reported stop-crop floor is necessary for arbitrary tokenizer
decode/re-encode boundaries: pruning that block could make an exact uniform-head
crop impossible. Older eligible segments are still annealed after EVERY forward.
Once the posthoc crop is known, its floor is released and the budget is enforced
again; this supplements, rather than replaces, per-forward eviction.
No stop criteria, independent decoder, quantization, or generation-config changes
are installed. Disabled mode delegates completely to the original implementation,
including its sampled-but-not-yet-cached final token counter convention.
"""
from __future__ import annotations

import contextlib
import math
import time

import torch

from models.QwenLLMWithKVCache import QwenLLMWithKVCache
from scripts.thinkkv_selection import SEGMENT_TOKENS, SelectionState
from token_tracker import TokenTracker


class UnifiedThinkKVLLM(QwenLLMWithKVCache):
    RUNTIME_VERSION = "unified-original-qa-thinkkv-selection-v1"
    DIAGNOSTIC_HOOKS = (
        ("_after_forward", "selection_and_cache_management"),
        ("_release_stop_floor", "selection_and_cache_management"),
        ("_plan_selection", "selection_and_cache_management"),
        ("_register_segments", "selection_and_cache_management"),
        ("_sync_tracker", "selection_and_cache_management"),
        ("_gather_selection", "cache_gather"),
        ("truncate_cache", "stop_crop"),
    )

    def __init__(self, model_path, kv_config=None, token_tracker=None, enabled=True, ratio=None):
        """An explicit ratio overrides kv_config.cache_ratio; otherwise default .2."""
        config = dict(kv_config or {})
        if ratio is None:
            ratio = config.get("cache_ratio")
            ratio = .2 if ratio is None else ratio
        ratio = float(ratio)
        if not math.isfinite(ratio) or not 0 < ratio <= 1:
            raise ValueError("ratio must be finite and in (0, 1]")
        if config.get("pruning_mode", "none") not in ("none", "thinkkv"):
            raise ValueError("ThinKV cannot also enable another KV pruning policy")
        config["pruning_mode"] = "none"
        tracker = (token_tracker or TokenTracker()) if enabled else token_tracker
        super().__init__(model_path, config, tracker)
        self.enabled = bool(enabled)
        self.ratio = float(ratio)
        self.reset()

    def reset(self):
        super().reset()
        self.selection = SelectionState()
        self.events = []
        self.forward_events = []
        self.pending_tokens = []
        self.uncached_sampled_tokens = []
        self._logged_end = 0
        self._logical_tokens = []
        self._stop_floor = None
        self._in_hooks = False
        self._live_cache = None
        self._crop_base_physical = 0
        self._crop_base_logical = 0
        self._stop_requested = False
        self._disabled_prompt_length = 0
        if getattr(self, "enabled", False):
            self.token_tracker.set_initial_cache_length(0)

    @staticmethod
    def _length(cache):
        return int(cache.get_seq_length()) if cache is not None else 0

    def _register_segments(self, appended=None, cropped=False):
        state = self.selection
        segments = math.ceil(max(0, state.next_id - state.prompt_length) / SEGMENT_TOKENS)
        if cropped:
            counts = []
            for index in range(segments):
                start = state.prompt_length + index * SEGMENT_TOKENS
                per_head = ((state.ids >= start) & (state.ids < start + SEGMENT_TOKENS)).sum(-1)
                if not torch.equal(per_head, per_head.flatten()[0].expand_as(per_head)):
                    raise RuntimeError("Logical segments have unequal native-head lengths")
                counts.append(int(per_head.flatten()[0]))
            state.counts = counts
        else:
            state.counts.extend([0] * (segments - len(state.counts)))
            if appended is not None:
                start, count = appended
                end = start + count
                start = max(start, state.prompt_length)
                while start < end:
                    segment, offset = divmod(start - state.prompt_length, SEGMENT_TOKENS)
                    added = min(end - start, SEGMENT_TOKENS - offset)
                    state.counts[segment] += added
                    start += added
        state.registered_end = state.next_id

    def _sync_tracker(self):
        state = self.selection
        if state.ids is None:
            return
        # The original tracker can represent only one shared mapper. Its mapper
        # is explicitly layer 0/head 0; audit exposes the authoritative full IDs.
        self.token_tracker.global_id_mapper = state.ids[0, 0].tolist()
        self.token_tracker.cache_length = state.ids.shape[-1]
        self.token_tracker.next_global_id = state.next_id

    def _materialize_plan(self, cache, indices, lengths, event):
        if not event["annealing"]:
            return
        state = self.selection
        old_ids = state.ids[0, 0].tolist()
        self._gather_selection(cache, indices, lengths)
        kept = set(state.ids[0, 0].tolist())
        discarded = [i for i in old_ids if i not in kept]
        tracker = self.token_tracker
        tracker.step_pruning_events.setdefault(tracker.current_step, []).extend(discarded)
        tracker.total_discarded += len(discarded)
        self._sync_tracker()

    def _plan_selection(self, cache, safe_end):
        return self.selection.plan(cache, self.ratio, safe_end)

    def _gather_selection(self, cache, indices, lengths):
        self.selection.materialize(cache, indices, lengths)

    def _release_stop_floor(self):
        self._stop_floor = None
        state = self.selection
        # A final sampled token has no KV yet. Keep its logical segment active.
        safe_end = self.pending_tokens[0]["global_id"] if self.pending_tokens else state.next_id
        t0 = time.perf_counter()
        indices, lengths, event = self._plan_selection(self.past_key_values, safe_end)
        event.update({
            "trigger": "posthoc_stop_floor_release", "forward": None,
            "logical_end": state.next_id, "safe_end": safe_end,
            "before_tokens": self._length(self.past_key_values), "stop_crop_lower_bound": None,
        })
        self._materialize_plan(self.past_key_values, indices, lengths, event)
        self.events.append(event)
        self.timing_stats["pruning_time"] += time.perf_counter() - t0
        self.current_cache_len = self._length(self.past_key_values)
        self._sync_tracker()

    def _after_forward(self, module, args, kwargs, outputs):
        cache = outputs.past_key_values
        if cache is None or not hasattr(cache, "layers"):
            raise RuntimeError("ThinKV requires a mutable HF DynamicCache with native KV layers")
        if any(layer.keys.dtype != torch.bfloat16 or layer.values.dtype != torch.bfloat16
               for layer in cache.layers):
            raise RuntimeError("Selection-only ThinKV requires actual BF16 KV pairs")
        self._live_cache = cache
        ids = kwargs.get("input_ids")
        if ids is None or ids.shape[0] != 1:
            raise RuntimeError("ThinKV requires batch-one token input")
        count = ids.shape[-1]
        state = self.selection
        before = self._length(cache) - count
        if before != (0 if state.ids is None else state.ids.shape[-1]):
            raise RuntimeError("HF appended a different number of KV tokens than input tokens")
        logical_start = state.next_id
        state.append(cache, logical_start, count)
        self.token_tracker.global_id_mapper.extend(range(logical_start, state.next_id))
        self.token_tracker.next_global_id = state.next_id
        if not state.prompt_length:
            state.register_prompt(count)
            self._crop_base_logical = count
            self._crop_base_physical = count
            if self._stop_floor is not None:
                self._stop_floor = count
        self.pending_tokens = []
        self._register_segments(appended=(logical_start, count))
        safe_end = state.next_id if self._stop_floor is None else self._stop_floor
        t0 = time.perf_counter()
        indices, lengths, event = self._plan_selection(cache, safe_end)
        event.update({
            "trigger": "after_forward",
            "forward": len(self.forward_events), "logical_end": state.next_id,
            "safe_end": safe_end, "before_tokens": self._length(cache),
            "stop_crop_lower_bound": self._stop_floor,
        })
        self._materialize_plan(cache, indices, lengths, event)
        self.timing_stats["pruning_time"] += time.perf_counter() - t0
        self.events.append(event)
        self.current_cache_len = self._length(cache)
        self.token_tracker.cache_length = self.current_cache_len
        positions = kwargs.get("position_ids")
        self.forward_events.append({
            "input_tokens": count, "logical_start": logical_start,
            "physical_start": before,
            "positions": None if positions is None else positions.detach().cpu().tolist(),
            "attention_mask_length": None if kwargs.get("attention_mask") is None
            else kwargs["attention_mask"].shape[-1],
            "resident_after": self._length(cache),
        })

    @contextlib.contextmanager
    def _hf_hooks(self):
        if self._in_hooks:
            yield
            return
        self._in_hooks = True
        model = self.model
        names = ("prepare_inputs_for_generation", "_update_model_kwargs_for_generation")
        originals = {name: getattr(model, name) for name in names}
        own = {name: (name in vars(model), vars(model).get(name)) for name in names}
        started = False

        def prepare(input_ids, **kwargs):
            nonlocal started
            cache = kwargs.get("past_key_values")
            length = self._length(cache)
            # Original preparation still handles all HF model-specific options.
            # Once generation has started, sequence indices select its last
            # sampled token, independently of the compacted physical KV length.
            if started:
                kwargs["cache_position"] = torch.tensor(
                    [input_ids.shape[-1] - 1], device=input_ids.device)
            prepared = originals[names[0]](input_ids, **kwargs)
            mask = prepared.get("attention_mask")
            if not started and mask is not None and (mask.ndim != 2 or not bool(mask.all())):
                raise RuntimeError("ThinKV requires an unpadded batch-one causal prefix")
            count = prepared["input_ids"].shape[-1]
            device = prepared["input_ids"].device
            prepared["cache_position"] = torch.arange(length, length + count, device=device)
            prepared["position_ids"] = prepared["cache_position"].unsqueeze(0)
            prepared["attention_mask"] = torch.ones(
                (1, length + count), dtype=torch.long, device=device)
            started = True
            return prepared

        def update(outputs, model_kwargs, **kwargs):
            updated = originals[names[1]](outputs, model_kwargs, **kwargs)
            length = self._length(outputs.past_key_values)
            device = outputs.logits.device
            updated["attention_mask"] = torch.ones(
                (1, length + 1), dtype=torch.long, device=device)
            updated["cache_position"] = torch.tensor([length], device=device)
            # Explicit positions are regenerated from the physical prefix.
            updated.pop("position_ids", None)
            return updated

        model.prepare_inputs_for_generation = prepare
        model._update_model_kwargs_for_generation = update
        handle = model.register_forward_hook(self._after_forward, with_kwargs=True)
        try:
            yield
        finally:
            handle.remove()
            for name in names:
                if own[name][0]:
                    setattr(model, name, own[name][1])
                else:
                    delattr(model, name)
            self._in_hooks = False

    def _extend_global_token_log(self, token_ids):
        super()._extend_global_token_log(token_ids)
        if not self.enabled:
            return
        self._logical_tokens.extend(int(token) for token in token_ids)
        self._logged_end += len(token_ids)
        missing = self._logged_end - self.selection.next_id
        if missing not in (0, 1):
            raise RuntimeError("Original HF sampled/processed token clocks diverged")
        self.pending_tokens = (
            [{"global_id": self.selection.next_id, "token_id": int(token_ids[-1])}]
            if missing else [])
        self.uncached_sampled_tokens.extend(self.pending_tokens)
        self.selection.next_id = self._logged_end
        self._register_segments()
        self.current_cache_len = self._length(self._live_cache)
        self._sync_tracker()

    def _decode(self, last_logits, max_new_tokens):
        if not self.enabled:
            return super()._decode(last_logits, max_new_tokens)
        # Caller already saved this counter for its original posthoc keep_count.
        self._crop_base_physical = self.current_cache_len
        self._crop_base_logical = self.selection.next_id
        self._stop_floor = self.selection.next_id if self._stop_requested else None
        self.current_cache_len = self._length(self.past_key_values)
        return super()._decode(last_logits, max_new_tokens)

    def truncate_cache(self, keep_token_count):
        if not self.enabled:
            return super().truncate_cache(keep_token_count)
        logical_end = self._crop_base_logical + keep_token_count - self._crop_base_physical
        state = self.selection
        if logical_end < state.prompt_length or logical_end > state.next_id:
            raise RuntimeError("Original posthoc crop is outside the logical trajectory")
        lengths = (state.ids < logical_end).sum(-1)
        if not torch.equal(lengths, lengths.flatten()[0].expand_as(lengths)):
            raise RuntimeError("Exact stop crop would produce unequal native-head lengths")
        physical_end = int(lengths.flatten()[0])
        self.past_key_values.crop(physical_end)
        state.ids = state.ids[..., :physical_end]
        state.next_id = logical_end
        self._logged_end = logical_end
        self._logical_tokens = self._logical_tokens[:logical_end]
        self._global_token_id_log = list(self._logical_tokens)
        self._all_token_ids = self._all_token_ids[:logical_end]
        self.pending_tokens = [token for token in self.pending_tokens
                               if token["global_id"] < logical_end]
        self.uncached_sampled_tokens = [
            token for token in self.uncached_sampled_tokens if token["global_id"] < logical_end]
        self._register_segments(cropped=True)
        self.current_cache_len = physical_end
        self._sync_tracker()

    def generate_first(self, prompt_text, max_new_tokens=256, stop_strings=None):
        if not self.enabled:
            result = super().generate_first(prompt_text, max_new_tokens, stop_strings)
            self._disabled_prompt_length = int(result[1][0][0].shape[-2])
            return result
        # The original method calls reset internally, so the floor is established
        # by the first forward rather than by a second prompt-tokenization path.
        self.reset()
        reset_owned = "reset" in vars(self)
        prior_reset = vars(self).get("reset")
        self.reset = lambda: None
        self._stop_floor = 0 if stop_strings else None
        try:
            with self._hf_hooks():
                result = super().generate_first(prompt_text, max_new_tokens, stop_strings)
        finally:
            if reset_owned:
                self.reset = prior_reset
            else:
                del self.reset
        self.current_cache_len = self._length(self.past_key_values)
        self._sync_tracker()
        if stop_strings:
            self._release_stop_floor()
        # Original splits happen before posthoc pruning. Return exact live views.
        prompt = self.selection.prompt_length
        prompt_kv = tuple((layer.keys[:, :, :prompt].detach().clone(),
                           layer.values[:, :, :prompt].detach().clone())
                          for layer in self.past_key_values.layers)
        generated_kv = tuple((layer.keys[:, :, prompt:].detach().clone(),
                              layer.values[:, :, prompt:].detach().clone())
                             for layer in self.past_key_values.layers)
        self._stop_floor = None
        return result[0], prompt_kv, generated_kv

    def generate_incremental(self, new_text, max_new_tokens=256, stop_strings=None):
        if not self.enabled:
            return super().generate_incremental(new_text, max_new_tokens, stop_strings)
        # Observation prefill is eligible normally; the stop floor starts at the
        # decode boundary, established by _decode after observation tokenization.
        self._stop_requested = bool(stop_strings)
        self._stop_floor = None
        with self._hf_hooks():
            result = super().generate_incremental(new_text, max_new_tokens, stop_strings)
        self.current_cache_len = self._length(self.past_key_values)
        self._sync_tracker()
        if stop_strings:
            self._release_stop_floor()
        self._stop_floor = None
        self._stop_requested = False
        return result

    def live_caches(self):
        caches = []
        for cache in (self.past_key_values, self._live_cache):
            if cache is not None and all(cache is not other for other in caches):
                caches.append(cache)
        return caches

    def audit(self):
        state = self.selection
        resident = self._length(self.past_key_values)
        prompt = state.prompt_length if self.enabled else self._disabled_prompt_length
        logical_end = state.next_id if self.enabled else len(self._all_token_ids)
        sampled_end = self._logged_end if self.enabled else logical_end
        trajectory = max(0, logical_end - prompt)
        safe_end = self.pending_tokens[0]["global_id"] if self.pending_tokens else logical_end
        target = prompt + math.floor(self.ratio * trajectory) if self.enabled else None
        floor = prompt + sum(
            4 if prompt + (index + 1) * SEGMENT_TOKENS <= safe_end else count
            for index, count in enumerate(state.counts)) if self.enabled else None
        config = self.model.generation_config.to_dict()
        return {
            "runtime_version": self.RUNTIME_VERSION, "enabled": self.enabled,
            "requested_ratio": self.ratio,
            "actual_trajectory_ratio": (
                (resident - prompt) / trajectory if trajectory else None),
            "resident_tokens": resident, "logical_end": logical_end,
            "prompt_tokens": prompt, "sampled_end": sampled_end,
            "pending_sampled_tokens": list(self.pending_tokens),
            "uncached_sampled_tokens": list(self.uncached_sampled_tokens),
            "global_token_id_log": self.get_global_token_id_log(),
            "per_layer_head_global_ids": None if state.ids is None else state.ids.tolist(),
            "tracker_mapper_scope": "layer0/native_head0",
            "segments": list(state.counts), "events": list(self.events),
            "target_tokens": target, "floor_tokens": floor,
            "budget_excess_tokens": max(0, resident - target) if target is not None else None,
            "forward_events": list(self.forward_events),
            "position_convention": "physical resident prefix; retained post-RoPE keys unchanged",
            "stop_crop_floor": "entire current stop-bearing block (arbitrary BPE-safe)",
            "generation_config": config,
            "attention_saliency": False,
            "scoring_eos_probe": None,
            "quantization": False,
            "native_kv_heads": [layer.keys.shape[1] for layer in
                                getattr(self.past_key_values, "layers", ())],
            "kv_dtypes": [[str(layer.keys.dtype), str(layer.values.dtype)] for layer in
                          getattr(self.past_key_values, "layers", ())],
        }
