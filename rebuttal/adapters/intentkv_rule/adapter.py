"""Isolated rule-only IntentKV adaptation; inherits the frozen HF decoder."""
from __future__ import annotations

from contextlib import contextmanager
import time

import torch

from models.QwenLLMWithKVCache import QwenLLMWithKVCache
from token_tracker import TokenTracker
from scripts.legacy_efficiency_metrics import cache_tensor_pairs
from rebuttal.adapters.intentkv_rule.query_memory import QueryMemory, kept_positions

METHOD = "intentkv_rule"
SCOPE = (
    "IntentKV Phase-1 QueryMemory rule adaptation (arXiv:2606.09916 §3.2 Eq2-3, "
    "lambda=.5); no trained residual; no paged dead-slot or radix/prefix reuse. "
    "Inherited HF physical-cache positions/generate/stop/crop; compact resident gathers. "
    "Whole original prompt protected; latest observation forced, allowing budget overflow. "
    "Initial block unpruned; memory updated once per initial-user/observation prefill. "
    "Not official full reproduction."
)
OVERRIDES = {
    "pruning_mode": METHOD, "protect_prompt": True, "observation_window": 0,
    "step_poolwise_prune": False, "prompt_prefill_keep_ratio": 1.0,
    "attn_mode": "scoring_forward",
}


class IntentKVRuleLLM(QwenLLMWithKVCache):
    def __init__(self, model_path, kv_config=None, token_tracker=None):
        config = {**OVERRIDES, **(kv_config or {})}
        if any(config.get(key) != value for key, value in OVERRIDES.items()):
            raise ValueError("IntentKV rule requires its isolated, prompt-protected configuration")
        if config.get("cache_ratio") not in (.2, .5):
            raise ValueError("Require nominal non-prompt ratio .5 or .2")
        super().__init__(model_path, dict(config, pruning_mode="tova"),
                         token_tracker=token_tracker or TokenTracker())
        self.kv_config = config
        self.reset()

    def reset(self):
        super().reset()
        self.query_memory = QueryMemory()
        self.intent_events = []
        self._forced_query = ()

    @contextmanager
    def _capture_prefill(self, length, span):
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

        if not 0 <= span[0] < span[1] <= length:
            raise ValueError("Empty/invalid actionable span")
        encodings, hooks = {}, []
        try:
            for index, layer in enumerate(self.model.model.layers):
                attention = layer.self_attn
                if attention.__class__.__name__ != "Qwen2Attention":
                    raise RuntimeError("Capture is verified only for the installed Qwen2Attention")
                state = {}

                def before(module, args, kwargs, *, i=index, state=state):
                    if i in encodings:
                        return
                    hidden = kwargs.get("hidden_states", args[0] if args else None)
                    if hidden is None or hidden.shape[:2] != (1, length):
                        raise RuntimeError("First attention call is not the expected actionable prefill")
                    embeddings = kwargs.get("position_embeddings", args[1] if len(args) > 1 else None)
                    if embeddings is None:
                        raise RuntimeError("Missing actual prefill RoPE embeddings")
                    state["embeddings"] = embeddings

                def projected(module, args, output, *, i=index, state=state, attention=attention):
                    if i in encodings:
                        return
                    query = output.view(1, length, -1, attention.head_dim).transpose(1, 2)
                    cos, sin = state.pop("embeddings")
                    query, _ = apply_rotary_pos_emb(query, query, cos, sin)
                    encodings[i] = query[0, :, span[0]:span[1]].detach().float().mean(dim=1)

                hooks.append(attention.register_forward_pre_hook(before, with_kwargs=True))
                hooks.append(attention.q_proj.register_forward_hook(projected))
            yield encodings
            if len(encodings) != len(self.model.model.layers):
                raise RuntimeError("Incomplete all-layer post-RoPE query capture")
        finally:
            for hook in hooks:
                hook.remove()

    def generate_first(self, prompt_text, max_new_tokens=256, stop_strings=None):
        rendered = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": "You are a helpful assistant."},
             {"role": "user", "content": prompt_text}],
            tokenize=False, add_generation_prompt=True)
        start = rendered.find(prompt_text)
        if start < 0 or rendered.count(prompt_text) != 1:
            raise RuntimeError("Cannot resolve initial user span in actual rendered template")
        encoded = self.tokenizer(rendered, return_offsets_mapping=True)
        positions = [i for i, (left, right) in enumerate(encoded["offset_mapping"])
                     if left < start + len(prompt_text) and right > start]
        if not positions:
            raise RuntimeError("Initial user span has no tokens")
        with self._capture_prefill(len(encoded["input_ids"]), (positions[0], positions[-1] + 1)) as rows:
            result = super().generate_first(prompt_text, max_new_tokens, stop_strings)
        self.query_memory.update(rows[index] for index in sorted(rows))
        self.intent_events.append({
            "phase": "initial_unpruned", "memory_updates": 1,
            "resident_tokens": cache_tensor_pairs(self.past_key_values)[0][0].shape[-2],
            "bookkeeping_tokens": self.current_cache_len,
            "prompt_tokens": self.kv_manager.protected_prefix_len,
            "actionable_query_span": [positions[0], positions[-1] + 1],
        })
        return result

    def _do_pruning(self, *args, **kwargs):
        return None

    def generate_incremental(self, new_text, max_new_tokens=256, stop_strings=None):
        before = cache_tensor_pairs(self.past_key_values)[0][0].shape[-2]
        length = len(self.tokenizer(new_text, add_special_tokens=False).input_ids)
        with self._capture_prefill(length, (0, length)) as rows:
            result = super().generate_incremental(new_text, max_new_tokens, stop_strings)
        self.query_memory.update(rows[index] for index in sorted(rows))
        self._forced_query = tuple(range(before, before + length))
        self._compress_completed_block()
        return result

    @torch.no_grad()
    def _compress_completed_block(self):
        pairs = cache_tensor_pairs(self.past_key_values)
        resident = pairs[0][0].shape[-2]
        mapper = list(self.token_tracker.global_id_mapper)
        prompt = self.kv_manager.protected_prefix_len
        if len(mapper) < resident or mapper[:prompt] != list(range(prompt)):
            raise RuntimeError("Missing resident IDs or damaged protected prompt")
        forced = sorted(set(range(prompt)) | set(self._forced_query))
        target = self._compute_target_budget()
        if target is None:
            raise RuntimeError("Missing original cumulative logical budget")
        begin = time.time()
        selected = kept_positions(self.query_memory, (key for key, _ in pairs), forced, target)
        self.timing_stats["scoring_time"] += time.time() - begin
        pruned = len(selected) < resident
        selected_set = set(selected)
        if pruned:
            begin = time.time()
            compressed = []
            for keys, values in pairs:
                index = torch.tensor(selected, device=keys.device, dtype=torch.long)
                compressed.append((keys.index_select(2, index), values.index_select(2, index)))
            self.past_key_values = self.kv_manager.pruning_strategy._build_cache(compressed)
            self.token_tracker.record_pruning_with_kept_indices(
                step=None, kept_local_indices=selected, old_cache_length=len(mapper),
                prune_start=prompt, prune_end=resident)
            self.current_cache_len = len(selected)
            self.kv_manager.current_cache_len = len(selected)
            self.kv_manager.total_prune_count += 1
            self.timing_stats["pruning_time"] += time.time() - begin
        self.kv_manager.last_pruned = pruned
        self.intent_events.append({
            "phase": "completed_observation_block", "pruned": pruned,
            "logical_total": self.token_tracker.next_global_id,
            "bookkeeping_before": len(mapper), "resident_before": resident,
            "resident_after": len(selected), "target": target,
            "forced_positions": forced, "forced_budget_overflow": max(0, len(forced) - target),
            "kept_positions": selected, "kept_inherited_ids": [mapper[i] for i in selected],
            "removed_inherited_ids": [mapper[i] for i in range(resident) if i not in selected_set],
            "counted_but_nonresident_before": len(mapper) - resident,
            "memory_updates": self.query_memory.updates,
        })

    def legacy_audit(self):
        return {"method": METHOD, "scope": SCOPE, "events": self.intent_events,
                "memory_updates": self.query_memory.updates,
                "memory_shapes": [list(row.shape) for row in self.query_memory.rows],
                "learned_residual": "omitted", "layout": "HF compact; no dead-slot/prefix-cache reuse"}
