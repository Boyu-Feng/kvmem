"""Auditable runtime for the schedule-controlled rebuttal experiments.

All arms use the same greedy loop, absolute RoPE positions and full-cache EOS
attention probe. This is an explicitly labeled estimator-controlled experiment,
not a claim to reproduce the submitted table or its prefill-saliency estimator.
The repository's token selectors and step utility are reused unchanged.
"""
from __future__ import annotations

import time
import torch
from transformers import DynamicCache

from models.QwenLLMWithKVCache import QwenLLMWithKVCache
from token_tracker import TokenTracker


class RebuttalLLM(QwenLLMWithKVCache):
    RUNTIME_VERSION = "rebuttal-controlled-v2"

    def __init__(self, model_path, kv_config=None, token_tracker=None):
        super().__init__(model_path, kv_config, token_tracker or TokenTracker())
        self.schedule = self.kv_config.get("pruning_schedule", "step")
        if self.schedule not in ("step", "token"):
            raise ValueError(f"Invalid schedule: {self.schedule}")
        if self.kv_config.get("attn_mode") != "scoring_forward":
            raise ValueError("Controlled v1 requires the explicit shared scoring_forward estimator")
        self.reset()

    def reset(self):
        super().reset()
        self.token_tracker.set_initial_cache_length(0)
        self.peak_cache_tokens = 0
        self.peak_cache_bytes = 0
        self.cache_measurements = []
        self.last_forward_positions = []
        self.prompt_length = 0
        self._initial_decode = False
        if self.kv_manager:
            self.kv_manager.pruning_strategy.h2o_scorer.record_snapshots = False

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _layers(self, cache=None):
        cache = self.past_key_values if cache is None else cache
        if cache is None:
            return []
        return [(layer.keys, layer.values) for layer in cache.layers]

    def _record_cache(self):
        pairs = self._layers()
        length = pairs[0][0].shape[2] if pairs else 0
        if any(k.shape[2] != length or v.shape[2] != length for k, v in pairs):
            raise RuntimeError("KV layers have inconsistent lengths")
        mapper = self.token_tracker.global_id_mapper
        if length != len(mapper):
            raise RuntimeError(f"KV/ID-map mismatch: {length} != {len(mapper)}")
        if len(set(mapper)) != len(mapper) or mapper != sorted(mapper):
            raise RuntimeError("Resident token IDs must be ordered and unique")
        self.current_cache_len = int(length)
        if self.kv_manager:
            self.kv_manager.current_cache_len = int(length)
        size = sum(k.numel() * k.element_size() + v.numel() * v.element_size() for k, v in pairs)
        self.peak_cache_tokens = max(self.peak_cache_tokens, int(length))
        self.peak_cache_bytes = max(self.peak_cache_bytes, int(size))
        return int(length), int(size)

    @torch.inference_mode()
    def _append(self, ids, timing_key="decode_time"):
        if not ids:
            raise ValueError("Cannot append an empty token block")
        start = self.token_tracker.next_global_id
        resident = self.current_cache_len
        if self.past_key_values is None:
            self.past_key_values = DynamicCache(config=self.model.config)
        positions = torch.arange(start, start + len(ids), device=self.device).unsqueeze(0)
        self.last_forward_positions = positions[0].tolist()
        self._sync()
        begin = time.perf_counter()
        outputs = self.model(
            input_ids=torch.tensor([ids], device=self.device),
            position_ids=positions,
            attention_mask=torch.ones((1, resident + len(ids)), device=self.device, dtype=torch.long),
            past_key_values=self.past_key_values,
            use_cache=True,
            return_dict=True,
            output_attentions=False,
        )
        self._sync()
        self.timing_stats[timing_key] += time.perf_counter() - begin
        self.past_key_values = outputs.past_key_values
        self.token_tracker.append_new_tokens(len(ids))
        self._extend_global_token_log(ids)
        self._all_token_ids.extend(ids)
        self._record_cache()
        return outputs.logits[:, -1, :]

    @torch.inference_mode()
    def _get_attention_for_scoring(self):
        """Probe every resident entry without appending to the live KV cache."""
        probe_cache = DynamicCache(config=self.model.config)
        # DynamicCache.update concatenates; the probe can initially share storage
        # with the live tensors without deep-copying the entire cache.
        for idx, (keys, values) in enumerate(self._layers()):
            probe_cache.update(keys, values, idx)
        implementation = self.model.config._attn_implementation
        start = self.token_tracker.next_global_id
        try:
            self.model.config._attn_implementation = "eager"
            outputs = self.model(
                input_ids=torch.tensor([[self.tokenizer.eos_token_id]], device=self.device),
                position_ids=torch.tensor([[start]], device=self.device),
                attention_mask=torch.ones((1, self.current_cache_len + 1), device=self.device, dtype=torch.long),
                past_key_values=probe_cache, use_cache=True,
                return_dict=True, output_attentions=True,
            )
        finally:
            self.model.config._attn_implementation = implementation
        if not outputs.attentions or outputs.attentions[-1].shape[-1] != self.current_cache_len + 1:
            raise RuntimeError("Scoring attention does not cover the entire resident cache")
        self._record_cache()
        return outputs.attentions

    def _maybe_prune(self, phase):
        if self.kv_manager is None:
            return
        # Original code leaves the first generation unpruned for all methods.
        if self._initial_decode:
            return
        before, before_bytes = self._record_cache()
        target = self._compute_target_budget()
        if target is None or before <= target:
            return
        self._sync()
        begin = time.perf_counter()
        attentions = self._get_attention_for_scoring()
        self._sync()
        self.timing_stats["scoring_time"] += time.perf_counter() - begin
        begin = time.perf_counter()
        self.past_key_values, self.current_cache_len = self.kv_manager.prune(self.past_key_values, attentions)
        self._sync()
        self.timing_stats["pruning_time"] += time.perf_counter() - begin
        after, after_bytes = self._record_cache()
        self._all_token_ids = [self._global_token_id_log[i] for i in self.token_tracker.global_id_mapper]
        # A protected suffix can make the target infeasible by one token because
        # the legacy selector requires at least one candidate to survive.
        if abs(after - target) > 1:
            raise RuntimeError(f"Budget mismatch after {phase}: target={target}, actual={after}")
        event = {
            "phase": phase, "step": self.token_tracker.current_step,
            "logical_tokens": self.token_tracker.next_global_id,
            "protected_tokens": self.prompt_length,
            "before_tokens": before, "after_tokens": after,
            "before_bytes": before_bytes, "after_bytes": after_bytes,
            "target_tokens": target,
        }
        self.cache_measurements.append(event)

    def _rewind_to(self, logical_end):
        """Discard a generated stop marker while retaining earlier evictions."""
        mapper = self.token_tracker.global_id_mapper
        count = sum(gid < logical_end for gid in mapper)
        self.past_key_values.crop(count)
        self.token_tracker.global_id_mapper = mapper[:count]
        self.token_tracker.next_global_id = logical_end
        self.token_tracker.cache_length = count
        self._global_token_id_log = self._global_token_id_log[:logical_end]
        self._all_token_ids = [self._global_token_id_log[i] for i in mapper[:count]]
        self._record_cache()

    def _generate_block(self, logits, max_new_tokens, stop_strings):
        generated = []
        first_id = self.token_tracker.next_global_id
        eos = self.model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        for _ in range(max_new_tokens):
            token = int(logits.argmax(dim=-1).item())
            if token in eos:
                break
            proposed = generated + [token]
            text = self.tokenizer.decode(proposed, skip_special_tokens=True)
            boundaries = [text.find(stop) for stop in (stop_strings or []) if stop in text]
            if boundaries:
                cut = min(boundaries)
                keep = len(generated)
                while keep and len(self.tokenizer.decode(generated[:keep], skip_special_tokens=True)) > cut:
                    keep -= 1
                self._rewind_to(first_id + keep)
                generated = generated[:keep]
                # A BPE token can contain both the action's closing bracket and
                # the newline starting the stop marker (e.g. "]\n"). Keep the
                # characters before the marker and materialize their replacement
                # token IDs, otherwise a valid tool call becomes unparseable.
                prefix = self.tokenizer.decode(generated, skip_special_tokens=True)
                if not text[:cut].startswith(prefix):
                    raise RuntimeError("Stop boundary cannot be aligned to the decoded token prefix")
                remainder = text[len(prefix):cut]
                if remainder:
                    tail = self.tokenizer(remainder, add_special_tokens=False).input_ids
                    for tail_token in tail:
                        generated.append(tail_token)
                        self._append([tail_token])
                        if self.schedule == "token":
                            self._maybe_prune("decode_token")
                break
            generated.append(token)
            logits = self._append([token])
            if self.schedule == "token":
                self._maybe_prune("decode_token")
        if self.schedule == "step":
            self._maybe_prune("completed_decode_block")
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()

    def generate_first(self, prompt_text, max_new_tokens=256, stop_strings=None):
        self.reset()
        self.token_tracker.set_current_step(1)
        messages = [{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt_text}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = self.tokenizer(text, add_special_tokens=True).input_ids
        logits = self._append(ids, "prefill_time")
        self.prompt_length = len(ids)
        if self.kv_manager:
            self.kv_manager.register_initial_cache(len(ids))
            self.token_tracker.set_current_step(1)
        self._initial_decode = True
        try:
            response = self._generate_block(logits, max_new_tokens, stop_strings)
        finally:
            self._initial_decode = False
        # Detached copies avoid pinning a superseded full live cache across steps.
        prompt_kv = tuple((k[:, :, :len(ids)].detach().clone(), v[:, :, :len(ids)].detach().clone())
                          for k, v in self._layers())
        generated_kv = tuple((k[:, :, len(ids):].detach().clone(), v[:, :, len(ids):].detach().clone())
                             for k, v in self._layers())
        return response, prompt_kv, generated_kv

    def generate_incremental(self, new_text, max_new_tokens=256, stop_strings=None):
        ids = self.tokenizer(new_text, add_special_tokens=False).input_ids
        logits = self._append(ids, "prefill_time")
        if self.kv_manager:
            self.kv_manager.step_count += 1
        if self.schedule == "token":
            self._maybe_prune("observation_prefill")
        return self._generate_block(logits, max_new_tokens, stop_strings)

    def get_stats(self):
        stats = super().get_stats()
        stats.update(runtime_version=self.RUNTIME_VERSION,
                     attention_source="full_cache_eos_probe",
                     pruning_schedule=self.schedule,
                     peak_cache_tokens=self.peak_cache_tokens,
                     peak_cache_bytes=self.peak_cache_bytes,
                     cache_measurements=self.cache_measurements)
        return stats
