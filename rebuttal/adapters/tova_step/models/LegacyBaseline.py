"""Timing-only step variant of the unchanged pre-rebuttal TOVA engine."""
from __future__ import annotations

import torch

from models.QwenLLMWithKVCache import QwenLLMWithKVCache
from token_tracker import TokenTracker

LEGACY_METHOD = "tova_step"
BASE_MODE = "tova"
RECENT_WINDOW = 0
LEGACY_CONFIG_OVERRIDE = dict(
    pruning_mode=BASE_MODE, observation_window=RECENT_WINDOW,
    attn_mode="scoring_forward", protect_prompt=True, num_score_layers=3,
    prune_every_n=1, step_poolwise_prune=False, prompt_prefill_keep_ratio=1.0,
)
LEGACY_SCOPE = (
    f"Timing-only {LEGACY_METHOD} on before_changes.tar QwenLLMWithKVCache. "
    "Original first-generation post-stop pruning remains unchanged. Later "
    "prefill/token pruning is deferred until inherited generation and stop crop "
    "finish, then the original prefill batch-plus-budget-top-up policy runs. "
    "Original manual token decoder, first model.generate, generation_config, "
    "physical-cache RoPE, EOS probe, selector and cumulative logical ratio budget "
    f"are preserved; recent window={RECENT_WINDOW}, including its original floor. "
    "No step utility, accumulated-attention replacement, or generation rewrite."
)


class LegacyBaselineLLM(QwenLLMWithKVCache):
    def __init__(self, model_path, kv_config=None, token_tracker=None):
        config = {**LEGACY_CONFIG_OVERRIDE, **(kv_config or {})}
        for key in ("pruning_mode", "observation_window", "attn_mode", "protect_prompt"):
            if config[key] != LEGACY_CONFIG_OVERRIDE[key]:
                raise ValueError(f"{LEGACY_METHOD} requires {key}={LEGACY_CONFIG_OVERRIDE[key]!r}")
        super().__init__(model_path, config, token_tracker or TokenTracker())
        self.reset()

    def reset(self):
        super().reset()
        self._step_defer_pruning = False
        self._step_completed_blocks = 0
        self._step_suppressed_calls = 0
        self._step_suppressed_probes = 0
        self.cache_measurements = []
        self._step_boundaries = []
        self._step_boundary_peak_bytes = 0

    def _resident_state(self):
        if self.past_key_values is None:
            return 0, 0
        if hasattr(self.past_key_values, "layers"):
            layers = [(layer.keys, layer.values) for layer in self.past_key_values.layers]
        else:
            layers = list(self.past_key_values)
        count = int(layers[0][0].shape[2]) if layers else 0
        if any(k.shape[2] != count or v.shape[2] != count for k, v in layers):
            raise RuntimeError("Legacy step baseline received inconsistent KV layer lengths")
        size = sum(x.numel() * x.element_size() for pair in layers for x in pair)
        self._step_boundary_peak_bytes = max(self._step_boundary_peak_bytes, size)
        return count, size

    def _phase(self):
        return ("initial_completed_block" if self._step_completed_blocks == 0
                else "incremental_completed_block")

    def generate_first(self, *args, **kwargs):
        # super.generate_first calls reset itself, and performs its ORIGINAL
        # post-stop single-token pruning loop. Never suppress this first round.
        response = super().generate_first(*args, **kwargs)
        self._record_boundary("initial_completed_block")
        self._step_completed_blocks = 1
        return response

    def generate_incremental(self, *args, **kwargs):
        self._step_defer_pruning = True
        try:
            response = super().generate_incremental(*args, **kwargs)
        finally:
            self._step_defer_pruning = False
        # This line is reached only after the original generation and stop crop
        # completed successfully. Do not prune from a finally block on failure.
        self._prune_completed_incremental()
        self._record_boundary("incremental_completed_block")
        self._step_completed_blocks += 1
        return response

    def _resolve_scoring_attentions(self, piggyback_attentions=None):
        if self._step_defer_pruning:
            # The old prefill branch resolves a probe before its prune callback.
            # Defer that read-only probe too; score the final cropped cache.
            self._step_suppressed_probes += 1
            return None
        return super()._resolve_scoring_attentions(piggyback_attentions)

    def _do_pruning(self, *args, **kwargs):
        if self._step_defer_pruning:
            self._step_suppressed_calls += 1
            return None
        before, before_bytes = self._resident_state()
        bookkeeping_before = self.current_cache_len
        history_before = len(self.kv_manager.pruning_history)
        result = super()._do_pruning(*args, **kwargs)
        after, after_bytes = self._resident_state()
        target = self._compute_h2o_budget()
        self.cache_measurements.append(dict(
            method=LEGACY_METHOD, phase=self._phase(),
            block=self._step_completed_blocks + 1,
            step=self.token_tracker.current_step,
            single_token_mode=bool(kwargs.get("single_token_mode", False)),
            pruned=after < before, before_tokens=before, after_tokens=after,
            before_bytes=before_bytes, after_bytes=after_bytes,
            bookkeeping_tokens_before=bookkeeping_before,
            bookkeeping_tokens_after=self.current_cache_len,
            logical_tokens=self.token_tracker.next_global_id,
            protected_tokens=self.kv_manager.protected_prefix_len,
            recent_window=RECENT_WINDOW, target_tokens=target,
            over_nominal_budget=max(0, self.current_cache_len - target) if target is not None else None,
            original_prune_events=len(self.kv_manager.pruning_history) - history_before,
        ))
        return result

    def _prune_completed_incremental(self):
        """Move the original prefill batch/top-up sequence to the stop boundary."""
        self.kv_manager.current_cache_len = self.current_cache_len
        target = self._compute_h2o_budget()
        if target is None or self.current_cache_len <= target:
            return
        # Fresh scoring after crop. Cached-attention top-up below intentionally
        # follows the ORIGINAL prefill path, without changing its selector.
        attentions = self._resolve_scoring_attentions()
        self._do_pruning(single_token_mode=False, attentions_override=attentions)
        max_iters = max(0, int(self.current_cache_len - target) + 8)
        for _ in range(max_iters):
            if self.current_cache_len <= target:
                break
            before = self.current_cache_len
            self._do_pruning(single_token_mode=True, attentions_override=attentions)
            if self.current_cache_len >= before:
                break
        del attentions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _record_boundary(self, phase):
        count, size = self._resident_state()
        self._step_boundaries.append(dict(
            phase=phase, block=self._step_completed_blocks + 1,
            resident_tokens=count, bookkeeping_tokens=self.current_cache_len,
            cache_bytes=size, logical_tokens=self.token_tracker.next_global_id,
            target_tokens=self._compute_h2o_budget(),
            suppressed_prune_calls=self._step_suppressed_calls,
            suppressed_prefill_probes=self._step_suppressed_probes,
        ))

    def legacy_audit(self):
        count, size = self._resident_state()
        return dict(
            method=LEGACY_METHOD, scope=LEGACY_SCOPE,
            base_mode=BASE_MODE, recent_window=RECENT_WINDOW,
            cache_measurements=list(self.cache_measurements),
            step_boundaries=list(self._step_boundaries),
            resident_tokens=count, bookkeeping_tokens=self.current_cache_len,
            cache_bytes=size, boundary_peak_cache_bytes=self._step_boundary_peak_bytes,
            suppressed_prune_calls=self._step_suppressed_calls,
            suppressed_prefill_probes=self._step_suppressed_probes,
            memory_scope="boundary resident KV only; not within-generation or probe-copy peak",
            generation_engine="unchanged before_changes.tar generation/decode/stop/position methods",
            budget_scope="original cumulative logical nonprompt ratio with original recent-window/minimum floor",
        )
