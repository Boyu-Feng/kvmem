"""FlowKV-style frozen-history adaptation on the unchanged legacy engine.

Generation, post-generation stop truncation, physical-cache RoPE positions,
generation-config penalties, and the EOS probe are inherited without rewriting.
Only the FlowKV-specific pruning policy is new. See LEGACY_FLOWKV.md.
"""
from __future__ import annotations

import time

import torch

from models.QwenLLMWithKVCache import QwenLLMWithKVCache
from token_tracker import TokenTracker

LEGACY_METHOD = "flowkv_style"
LEGACY_SCOPE = (
    "FlowKV-style agent-block frozen-history policy with repository TOVA last-layer "
    "last-query head-mean selector, on before_changes.tar QwenLLMWithKVCache. "
    "Legacy generate/penalty/physical RoPE/post-hoc stop/cache semantics retained; "
    "initial generation unpruned and unfrozen, compressed with the first later block. "
    "Prompt protected, cumulative legacy logical non-prompt ratio budget; no step utility. "
    "Method adaptation, not an official FlowKV implementation reproduction."
)
LEGACY_CONFIG_OVERRIDE = {
    "pruning_mode": LEGACY_METHOD,
    "flowkv_style": True,
    "attn_mode": "scoring_forward",
    "protect_prompt": True,
    "observation_window": 0,
    "step_poolwise_prune": False,
    "prune_every_n": 1,
    "prompt_prefill_keep_ratio": 1.0,
}


class LegacyBaselineLLM(QwenLLMWithKVCache):
    """Keep legacy generation; freeze/compress only completed later agent blocks."""

    def __init__(self, model_path, kv_config=None, token_tracker=None):
        config = {**LEGACY_CONFIG_OVERRIDE, **(kv_config or {})}
        if config.get("pruning_mode") != LEGACY_METHOD:
            raise ValueError("Legacy FlowKV requires pruning_mode=flowkv_style")
        if not config.get("protect_prompt") or config.get("observation_window") != 0:
            raise ValueError("Legacy FlowKV requires prompt protection and no recent window")
        if config.get("step_poolwise_prune"):
            raise ValueError("Legacy FlowKV does not use step utility/floors")
        # The unmodified manager only accepts established selector names. Use
        # its TOVA bookkeeping, then dispatch generation under our public name:
        # this reaches the original model.generate path, not the H2O token loop.
        super().__init__(model_path, dict(config, pruning_mode="tova"),
                         token_tracker=token_tracker or TokenTracker())
        self.kv_config = config
        self.reset()

    def reset(self):
        super().reset()
        self._flowkv_frozen_ids = ()
        self.cache_measurements = []
        self._flowkv_boundary_peak_bytes = 0
        self._flowkv_initial_state = None

    def _resident_layers(self):
        if self.past_key_values is None:
            return []
        if hasattr(self.past_key_values, "layers"):
            return [(layer.keys, layer.values) for layer in self.past_key_values.layers]
        return list(self.past_key_values)

    def _resident_state(self):
        layers = self._resident_layers()
        count = int(layers[0][0].shape[2]) if layers else 0
        if any(k.shape[2] != count or v.shape[2] != count for k, v in layers):
            raise RuntimeError("Legacy FlowKV received inconsistent KV layer lengths")
        size = sum(k.numel() * k.element_size() + v.numel() * v.element_size()
                   for k, v in layers)
        self._flowkv_boundary_peak_bytes = max(self._flowkv_boundary_peak_bytes, size)
        return count, size

    def generate_first(self, prompt_text, max_new_tokens=256, stop_strings=None):
        result = super().generate_first(prompt_text, max_new_tokens, stop_strings)
        actual, size = self._resident_state()
        self._flowkv_initial_state = dict(
            resident_tokens=actual, bookkeeping_tokens=self.current_cache_len,
            cache_bytes=size, frozen=False, pruned=False)
        return result

    def _do_pruning(self, *args, **kwargs):
        # Legacy generate_incremental asks the manager to prune after prefill.
        # Flow's method-specific isolation policy waits for the completed block.
        # Do not alter inherited prefill, generate(), stop handling, or probe.
        return None

    def generate_incremental(self, new_text, max_new_tokens=256, stop_strings=None):
        response = super().generate_incremental(new_text, max_new_tokens, stop_strings)
        self._freeze_completed_block()
        return response

    @torch.no_grad()
    def _freeze_completed_block(self):
        before, before_bytes = self._resident_state()
        bookkeeping_before = int(self.current_cache_len)
        mapper = list(self.token_tracker.global_id_mapper)
        if len(mapper) < before:
            raise RuntimeError("Legacy FlowKV tracker is shorter than the resident KV")
        # Preserve the legacy engine's missing-final-token behavior. A generated
        # token that was only counted, never forwarded, has no selectable KV.
        # Do not materialize it or otherwise repair the generation engine.
        resident_ids = mapper[:before]
        prompt_len = int(self.kv_manager.protected_prefix_len)
        prompt_ids = tuple(range(prompt_len))
        if tuple(resident_ids[:prompt_len]) != prompt_ids:
            raise RuntimeError("Legacy FlowKV requires an intact protected prompt")
        frozen_ids = self._flowkv_frozen_ids or prompt_ids
        frozen_count = len(frozen_ids)
        if tuple(resident_ids[:frozen_count]) != frozen_ids:
            raise RuntimeError("Legacy FlowKV frozen history was removed or reordered")

        target = self._compute_target_budget()
        if target is None or target < frozen_count:
            raise RuntimeError("Legacy FlowKV budget cannot contain frozen history")
        new_ids = resident_ids[frozen_count:]
        quota = min(len(new_ids), target - frozen_count)
        pruned = quota < len(new_ids)
        selected = list(range(before))
        if pruned:
            if quota:
                begin = time.time()
                attentions = self._get_attention_for_scoring()  # unchanged legacy EOS probe
                self.timing_stats["scoring_time"] += time.time() - begin
                if not attentions or attentions[-1].shape[-1] < before:
                    raise RuntimeError("Legacy FlowKV did not obtain resident-cache scores")
                scores = attentions[-1][0].mean(dim=0)[-1, frozen_count:before]
                chosen = scores.topk(quota).indices.sort().values + frozen_count
                selected_new = chosen.tolist()
            else:
                selected_new = []
            selected = list(range(frozen_count)) + selected_new
            begin = time.time()
            compressed = []
            for keys, values in self._resident_layers():
                index = torch.tensor(selected, dtype=torch.long, device=keys.device)
                compressed.append((keys.index_select(2, index), values.index_select(2, index)))
            # Reuse the exact legacy cache construction used by its selectors.
            self.past_key_values = self.kv_manager.pruning_strategy._build_cache(compressed)
            self.token_tracker.record_pruning_with_kept_indices(
                step=None, kept_local_indices=selected,
                old_cache_length=len(mapper), prune_start=frozen_count, prune_end=before)
            self.current_cache_len = len(selected)
            self.kv_manager.current_cache_len = len(selected)
            self.kv_manager.total_prune_count += 1
            self.timing_stats["pruning_time"] += time.time() - begin
        # When no resident eviction is needed, do not normalize the legacy
        # cache-length/map bookkeeping: that would change its generation path.
        after, after_bytes = self._resident_state()
        self.kv_manager.last_pruned = pruned
        self._flowkv_frozen_ids = tuple(resident_ids[i] for i in selected)
        selected_set = set(self._flowkv_frozen_ids[frozen_count:])
        event = dict(
            phase="legacy_completed_block", step=self.token_tracker.current_step,
            mode=LEGACY_METHOD, selector="tova_last_layer_last_query_head_mean",
            pruned=pruned, logical_tokens=self.token_tracker.next_global_id,
            protected_tokens=prompt_len, before_tokens=before, after_tokens=after,
            before_bytes=before_bytes, after_bytes=after_bytes, target_tokens=target,
            bookkeeping_tokens_before=bookkeeping_before,
            bookkeeping_tokens_after=self.current_cache_len,
            counted_but_nonresident_before=max(0, len(mapper) - before),
            available_new_budget=target - frozen_count, new_tokens_kept=quota,
            budget_shortfall_tokens=target - after,
            frozen_global_ids_before=list(frozen_ids), new_candidate_global_ids=new_ids,
            selected_new_global_ids=list(self._flowkv_frozen_ids[frozen_count:]),
            evicted_new_global_ids=[gid for gid in new_ids if gid not in selected_set],
            frozen_tokens_after=after)
        self.cache_measurements.append(event)
        self.kv_manager.pruning_history.append({**event,
            "cache_before": before, "new_total_len": after,
            "tokens_evicted": before - after,
            "evicted_abs_indices": sorted(set(range(before)) - set(selected))})

    def legacy_audit(self):
        actual, size = self._resident_state()
        return dict(method=LEGACY_METHOD, scope=LEGACY_SCOPE,
            initial_generation=self._flowkv_initial_state,
            cache_measurements=list(self.cache_measurements),
            frozen_tokens=len(self._flowkv_frozen_ids),
            resident_tokens=actual, bookkeeping_tokens=self.current_cache_len,
            cache_bytes=size, boundary_peak_cache_bytes=self._flowkv_boundary_peak_bytes,
            memory_scope="boundary resident KV only; not a within-block live peak",
            identity_scope="inherited legacy tracker IDs; no rewrite of stop/global-token logging",
            generation_engine="unchanged before_changes.tar QwenLLMWithKVCache")
