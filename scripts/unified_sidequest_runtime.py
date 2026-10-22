"""Untrained asynchronous SideQuest adaptation on the original StepKV HF decoder.

Main prompts, generation, EOS handling and posthoc text/KV crops are inherited
unchanged. Unlike the pilot, cursor labels are sidecar metadata, not added to QA
observations. Completed auxiliary decisions are collected during real forwards
but applied only at the next tool-turn entry, never inside HF's kwargs update or
before its posthoc crop. This scheduling difference is intentional and audited.
"""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
import re
import threading

import torch

from models.QwenLLMWithKVCache import QwenLLMWithKVCache
from scripts.sidequest_runtime import SideQuestPilot, evict_cache, fork_cache


class _AuxiliaryTokenizer:
    """Add cursor identities only to the auxiliary memory-management prompt."""

    def __init__(self, tokenizer, catalogue):
        self.tokenizer = tokenizer
        self.catalogue = catalogue

    def __call__(self, text, **kwargs):
        return self.tokenizer(text + "\nCursor catalogue (untrusted tool text):\n"
                              + json.dumps(self.catalogue) + "\n", **kwargs)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


class UnifiedSideQuestLLM(QwenLLMWithKVCache):
    RUNTIME_VERSION = "sidequest_untrained_original_hf_v1"
    SCHEDULE = ("spawn_after_original_observation_prefill; collect_nonblocking_on_main_forward; "
                "apply_at_next_tool_turn_after_original_posthoc_crop")
    DIAGNOSTIC_HOOKS = (
        ("_remember_action", "selection_and_cache_management"),
        ("spawn_auxiliary", "selection_and_cache_management"),
        ("_apply_ready", "selection_and_cache_management"),
        ("_sync_crop", "selection_and_cache_management"),
        ("collect_pending", "generation_other"),
    )

    def __init__(self, model_path, kv_config=None, token_tracker=None, enabled=True):
        config = dict(kv_config or {})
        if config.get("pruning_mode", "none") != "none":
            raise ValueError("SideQuest cannot be combined with another pruning algorithm")
        self.enabled = enabled
        self.interval = config.get("sidequest_interval", 4)
        self.aux_max_tokens = config.get("sidequest_aux_max_tokens", 128)
        if any(type(value) is not int or value < 1
               for value in (self.interval, self.aux_max_tokens)):
            raise ValueError("SideQuest interval and auxiliary cap must be positive integers")
        super().__init__(model_path, config, token_tracker)
        # No attention/config/dtype switching: both branches use the loaded BF16
        # model. Dynamic RoPE updates shared buffers and is not concurrent-safe.
        rope = (getattr(self.model.config, "rope_parameters", None)
                or getattr(self.model.config, "rope_scaling", None) or {})
        rope_type = rope.get("rope_type", rope.get("type", "default"))
        if enabled and ("dynamic" in rope_type or rope_type == "longrope"):
            raise ValueError("Asynchronous SideQuest requires static RoPE")
        self._lock = threading.RLock()
        self._thread = threading.local()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="unified-sidequest")
        self.pending = None
        self._pending_prefix = None
        self._aux_cache = None
        self._main_cache = None
        self._eviction_caches = {}
        self._closed = False
        self._active_generation = False
        self._hooks = [
            self.model.register_forward_pre_hook(self._before_forward, with_kwargs=True),
            self.model.register_forward_hook(self._after_forward, with_kwargs=True),
        ]
        self.reset()

    def reset(self):
        if self.pending is not None:
            self.collect_pending(apply=False, wait=True)
        super().reset()
        self.turn = 0
        self.prompt_length = 0
        self.cursors = {}
        self.aux_events = []
        self.boundaries = []
        self.forward_positions = []
        self._ready = []
        self._mapper = []
        self._tokens = []
        self._next_id = 0
        self._main_cache = None
        self._observation_prefill = False
        self.last_action_span = None

    def _before_forward(self, module, args, kwargs):
        auxiliary = getattr(self._thread, "auxiliary", False)
        cache = kwargs.get("past_key_values")
        ids = kwargs.get("input_ids", args[0] if args else None)
        if ids is None:
            raise RuntimeError("SideQuest audit requires input_ids, not inputs_embeds")
        resident = int(cache.get_seq_length()) if cache is not None else 0
        positions = kwargs.get("position_ids")
        cache_position = kwargs.get("cache_position")
        mask = kwargs.get("attention_mask")
        event = dict(branch="auxiliary" if auxiliary else "main", turn=self.turn,
                     resident_tokens=resident, input_tokens=int(ids.shape[-1]),
                     position_ids=positions.detach().cpu().tolist() if positions is not None else None,
                     cache_position=cache_position.detach().cpu().tolist() if cache_position is not None else None,
                     attention_mask_shape=list(mask.shape) if mask is not None else None,
                     physical_positions=list(range(resident, resident + ids.shape[-1])))
        with self._lock:
            self.forward_positions.append(event)
            if auxiliary:
                self._aux_cache = cache
            else:
                self._main_cache = cache
                if self.enabled:
                    if resident != len(self._mapper):
                        raise RuntimeError("Main forward cache/sidecar map mismatch")
                    new_ids = ids[0].detach().cpu().tolist()
                    self._mapper.extend(range(self._next_id, self._next_id + len(new_ids)))
                    self._tokens.extend(new_ids)
                    self._next_id += len(new_ids)

    def _after_forward(self, module, args, kwargs, output):
        cache = output.past_key_values
        auxiliary = getattr(self._thread, "auxiliary", False)
        with self._lock:
            if auxiliary:
                self._aux_cache = cache
            else:
                self._main_cache = cache
        if auxiliary or not self.enabled:
            return
        if cache.get_seq_length() != len(self._mapper):
            raise RuntimeError("Main output cache/sidecar map mismatch")
        self.collect_pending(apply=True, wait=False)
        if self._observation_prefill:
            self._observation_prefill = False
            if self.turn % self.interval == 0 and self.pending is None and not self._ready:
                self.spawn_auxiliary()

    def _sync_crop(self):
        physical = int(self.past_key_values.get_seq_length())
        self._main_cache = self.past_key_values
        if self.enabled:
            if physical > len(self._mapper):
                raise RuntimeError("Posthoc crop grew the cache")
            self._mapper = self._mapper[:physical]
            self._next_id = max(self._mapper, default=-1) + 1
            self._tokens = self._tokens[:self._next_id]
        self.boundaries.append(dict(turn=self.turn, tokens=physical,
                                    wrapper_tokens=self.current_cache_len,
                                    global_ids=list(self._mapper)))

    def _remember_action(self, start):
        ids = self._tokens[start:]
        text = self.tokenizer.decode(ids, skip_special_tokens=True)
        match = re.search(r"Action\s*\d*\s*:", text)
        self.last_action_span = None
        if match:
            index = next((i for i in range(len(ids))
                          if len(self.tokenizer.decode(ids[:i], skip_special_tokens=True))
                          >= match.start()), None)
            if index is not None:
                self.last_action_span = (start + index, self._next_id)

    def generate_first(self, *args, **kwargs):
        if self._closed:
            raise RuntimeError("SideQuest is closed")
        self._active_generation = True
        try:
            result = super().generate_first(*args, **kwargs)
        finally:
            self._active_generation = False
        self.turn = 1
        # The initial HF forward records the actual chat-template prompt length.
        self.prompt_length = self.forward_positions[0]["input_tokens"]
        self._sync_crop()
        if self.enabled:
            self._remember_action(self.prompt_length)
        return result

    def generate_incremental(self, new_text, max_new_tokens=256, stop_strings=None):
        if self._closed:
            raise RuntimeError("SideQuest is closed")
        if self.enabled:
            self.collect_pending(apply=True, wait=False)
            self._apply_ready()
            encoded = self.tokenizer(new_text, add_special_tokens=False,
                                     return_offsets_mapping=True)
            thought_start = new_text.rfind("\nThought ")
            if thought_start < 0:
                raise ValueError("Missing original observation/thought boundary")
            obs_count = sum(end <= thought_start for _, end in encoded.offset_mapping)
            intervals = [(self._next_id, self._next_id + obs_count)]
            if self.last_action_span is not None:
                intervals.insert(0, self.last_action_span)
            self.cursors[self.turn - 1] = dict(intervals=intervals, deleted=False,
                                              observation=new_text[:thought_start])
            generated_start = self._next_id + len(encoded.input_ids)
            self._observation_prefill = True
        self._active_generation = True
        try:
            result = super().generate_incremental(new_text, max_new_tokens, stop_strings)
        finally:
            self._active_generation = False
            self._observation_prefill = False
        self.turn += 1
        self._sync_crop()
        if self.enabled:
            self._remember_action(generated_start)
        return result

    def spawn_auxiliary(self):
        if not self.enabled or self.pending is not None:
            return
        available = [i for i, value in self.cursors.items() if not value["deleted"]]
        snapshot = fork_cache(self._main_cache, self.model.config)
        event = dict(spawn_turn=self.turn, available=available,
                     prefix_tokens=int(snapshot.get_seq_length()), applied=False,
                     schedule=self.SCHEDULE, position_basis="physical_snapshot")
        catalogue = {i: self.cursors[i]["observation"] for i in available}
        with self._lock:
            self.aux_events.append(event)
            self._pending_prefix = snapshot
            self.pending = (self.executor.submit(
                self._auxiliary, snapshot, snapshot.get_seq_length(), available, catalogue), event)

    def _auxiliary(self, cache, next_id, available, catalogue):
        # Reuse the pilot's adaptive JSON decoder and validation, not a fixed
        # ratio or a trained SideQuest claim. Only auxiliary prompt labels differ.
        proxy = copy.copy(self)
        proxy.tokenizer = _AuxiliaryTokenizer(self.tokenizer, catalogue)
        self._thread.auxiliary = True
        try:
            return SideQuestPilot._auxiliary(proxy, cache, next_id, available)
        finally:
            with self._lock:
                self._aux_cache = None
            self._thread.auxiliary = False

    def collect_pending(self, apply=True, wait=False):
        if not apply:
            for _, staged_event in self._ready:
                staged_event["unused_at_episode_end"] = True
            self._ready = []
        if self.pending is None:
            return
        future, event = self.pending
        if not wait and not future.done():
            return
        try:
            result = future.result()
        except BaseException as exc:
            event.update(runtime_error=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            self.pending = None
            self._pending_prefix = None
        event.update(result, collection_turn=self.turn,
                     collected_during_original_forward=self._active_generation)
        if apply:
            self._ready.append((result, event))
            event["deferred_until_next_tool_turn"] = True
        else:
            event["unused_at_episode_end"] = True

    def _apply_ready(self):
        if self._active_generation:
            raise RuntimeError("Cannot evict while original HF generation is active")
        for result, event in self._ready:
            ids = result["del_cursors"]
            intervals = [span for cursor in ids if not self.cursors[cursor]["deleted"]
                         for span in self.cursors[cursor]["intervals"]]
            before = int(self.past_key_values.get_seq_length())
            if intervals:
                old = self.past_key_values
                self._eviction_caches = {"eviction_old": old}
                try:
                    pruned, mapper = evict_cache(
                        old, self._mapper, intervals, self.prompt_length, self.model.config)
                    self._eviction_caches["eviction_new"] = pruned
                    removed = before - len(mapper)
                    kept_ids = set(mapper)
                    keep = [i for i, gid in enumerate(self._mapper) if gid in kept_ids]
                    # Preserve the original wrapper's uncached final sampled token.
                    self._all_token_ids = ([self._all_token_ids[i] for i in keep]
                                           + self._all_token_ids[before:])
                    self.current_cache_len -= removed
                    self.past_key_values = self._main_cache = pruned
                    self._mapper = mapper
                    if self.token_tracker is not None:
                        tracker = self.token_tracker
                        # Original first-generation tracking can omit prompt IDs;
                        # reconstruct aligned sidecar IDs only when actually evicting.
                        tracker.global_id_mapper = list(mapper)
                        tracker.cache_length = len(mapper)
                        tracker.next_global_id = self._next_id
                    for cursor in ids:
                        self.cursors[cursor]["deleted"] = True
                finally:
                    self._eviction_caches = {}
            event.update(applied=True, apply_turn=self.turn, before_tokens=before,
                         after_tokens=int(self.past_key_values.get_seq_length()),
                         selection_intervals=[list(span) for span in intervals],
                         kept_global_ids=list(self._mapper))
        self._ready = []

    def live_caches(self):
        """Actual cache objects, including a queued/running branch's shared prefix."""
        with self._lock:
            caches = dict(main=self._main_cache or self.past_key_values,
                          auxiliary=self._aux_cache, pending_prefix=self._pending_prefix,
                          **self._eviction_caches)
            return {name: cache for name, cache in caches.items() if cache is not None}

    def audit(self):
        with self._lock:
            return copy.deepcopy(dict(
                runtime=self.RUNTIME_VERSION, enabled=self.enabled, trained=False,
                schedule=self.SCHEDULE, main_decoder="unchanged_original_greedy_HF_posthoc_crop",
                main_position_basis="unchanged_original_physical_cache",
                interval=self.interval, aux_max_tokens=self.aux_max_tokens,
                aux_events=self.aux_events, cursors=self.cursors, boundaries=self.boundaries,
                forward_positions=self.forward_positions, final_global_ids=self._mapper,
                pending=self.pending is not None, staged_decisions=len(self._ready),
                final_tokens=int(self.past_key_values.get_seq_length())
                if self.past_key_values is not None else 0))

    def close(self):
        if self._closed:
            return
        try:
            self.collect_pending(apply=False, wait=True)
            for _, event in self._ready:
                event["unused_at_episode_end"] = True
            self._ready = []
        finally:
            self.executor.shutdown(wait=True)
            for hook in self._hooks:
                hook.remove()
            self._closed = True
