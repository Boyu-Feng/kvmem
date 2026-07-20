"""
Qwen LLM Wrapper with KV Cache Management
Supports incremental generation with KV cache reuse and pruning.
Provides two attention acquisition modes:
  1. Scoring forward: dedicated forward pass with output_attentions=True at pruning time
  2. Piggyback: record attention during normal prefill of new observation tokens
"""

import math
import os
import torch
import time
import copy
from transformers import AutoModelForCausalLM, AutoTokenizer
from kv_cache.kv_cache_manager import KVCacheManager
from transformers.cache_utils import DynamicCache
from models.model_paths import is_local_model_dir

class QwenLLMWithKVCache:
    """
    Qwen LLM wrapper with KV Cache management for incremental generation.

    Instead of re-encoding the full trajectory every step, this wrapper:
    1. Encodes [System + Question] once, caching KV
    2. For each new step, only encodes new tokens (Observation), reusing cached KV
    3. Optionally prunes the KV cache to control memory
    """

    def __init__(self, model_path, kv_config=None, token_tracker=None):
        """
        Args:
            model_path: path to the pretrained Qwen model
            kv_config: dict with pruning configuration (see KVCacheManager).
                       If None, no pruning is applied (pure KV cache reuse).
            token_tracker: optional TokenTracker instance for tracking pruned tokens
        """
        print(f"[INFO] Loading model from {model_path} (KV Cache mode)...")
        local_only = is_local_model_dir(model_path)
        if local_only:
            print(f"[INFO] Using local model files only (no HuggingFace download).")
        load_kwargs = {"trust_remote_code": True}
        if local_only:
            load_kwargs["local_files_only"] = True

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, **load_kwargs)

        # Always load with default (SDPA) attention for best inference quality.
        # For H2O scoring, we temporarily switch to eager attention only during
        # the scoring forward pass (output_attentions=True requires eager).
        pruning_mode = (kv_config or {}).get("pruning_mode", "none")
        self.needs_attn_scoring = pruning_mode in ("h2o", "tova", "pyramidinfer", "step_anchor_h2o", "step_aware_h2o", "step_inter", "h2o_snapkv")
        self.needs_new_step_kv = pruning_mode in ("ours")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map="auto",
            **load_kwargs,
        )
        self.model.eval()
        self.device = self.model.device
        attn_impl = getattr(self.model.config, '_attn_implementation', 'default')
        print(f"[INFO] Model loaded successfully (KV Cache mode). attn_implementation={attn_impl}")
        if self.needs_attn_scoring:
            print("[INFO] H2O scoring will temporarily switch to eager attention during scoring forward.")
        if self.needs_new_step_kv:
            print("[INFO] Ours pruning will require new step KV during fusion.")

        # KV Cache management
        self.kv_config = kv_config or {}
        self.token_tracker = token_tracker
        self.pruning_enabled = self.kv_config.get("pruning_mode", "none") != "none"
        if self.pruning_enabled:
            self.kv_manager = KVCacheManager(self.kv_config, token_tracker=token_tracker)
        else:
            self.kv_manager = None

        # Attention acquisition mode: "scoring_forward" or "piggyback"
        self.attn_mode = self.kv_config.get("attn_mode", "scoring_forward")
        self.step_kv = None
        self.last_prefill_attentions = None
        # State
        self.past_key_values = None
        self.current_cache_len = 0

        # Timing stats
        self.timing_stats = {
            "prefill_time": 0.0,
            "decode_time": 0.0,
            "scoring_time": 0.0,
            "pruning_time": 0.0,
        }

    def reset(self):
        """Reset all state for a new episode."""
        self.past_key_values = None
        self.current_cache_len = 0
        self._all_token_ids = []  # Track all token ids seen for repetition penalty
        self._global_token_id_log = []  # Append-only: index = global token id
        self.last_prefill_attentions = None
        if self.kv_manager:
            self.kv_manager.register_initial_cache(0)
        self.timing_stats = {
            "prefill_time": 0.0,
            "decode_time": 0.0,
            "scoring_time": 0.0,
            "pruning_time": 0.0,
        }

    def _compute_h2o_budget(self):
        """
        Compute H2O cache budget for current step.
        - Ratio priority: cache_ratio -> target_cache_ratio -> keep_ratio -> 0.5.
        - protect_prompt=False: budget = ratio * original total.
        - protect_prompt=True: budget = prompt_len + ratio * non-prompt total.
        """
        if self.token_tracker is None or not hasattr(self.token_tracker, "next_global_id"):
            return None
        original_total = int(self.token_tracker.next_global_id)
        ratio = 0.5
        if self.kv_manager is not None:
            ratio_cfg = getattr(self.kv_manager, "cache_ratio", None)
            if ratio_cfg is None:
                ratio_cfg = getattr(self.kv_manager, "target_cache_ratio", None)
            if ratio_cfg is None:
                ratio_cfg = getattr(self.kv_manager, "keep_ratio", None)
            try:
                if ratio_cfg is not None:
                    ratio = float(ratio_cfg)
            except Exception:
                ratio = 0.5
        ratio = max(0.0, min(1.0, ratio))
        if self.kv_manager is None:
            return max(1, int(original_total * ratio))

        protect_prompt = bool(getattr(self.kv_manager, "protect_prompt", True))
        prompt_len = int(getattr(self.kv_manager, "protected_prefix_len", 0))
        if not protect_prompt:
            return max(1, int(original_total * ratio))

        non_prompt_total = max(0, original_total - prompt_len)
        # Keep full prompt + ratio of generated tokens.
        if non_prompt_total <= 0:
            return max(1, prompt_len)
        return prompt_len + max(1, int(non_prompt_total * ratio))

    def _compute_target_budget(self):
        """
        Compute target total-cache budget from a stable logical token base.

        Key behavior:
        - Use token_tracker.next_global_id (logical tokens seen so far) as base
          whenever available, so target ratio is stable across steps and does not
          recursively shrink with already-pruned cache length.
        - protect_prompt=True: keep full prompt + ratio * generated(logical) tokens.
        - protect_prompt=False: apply ratio on total logical tokens.
        """
        if self.kv_manager is None:
            return None

        ratio = getattr(self.kv_manager, "cache_ratio", None)
        if ratio is None:
            ratio = getattr(self.kv_manager, "target_cache_ratio", None)
        if ratio is None:
            return None
        try:
            ratio = float(ratio)
        except Exception:
            return None
        ratio = max(0.0, min(1.0, ratio))

        # Keep manager state synced with wrapper state.
        self.kv_manager.current_cache_len = int(self.current_cache_len)
        current_cache = int(self.current_cache_len)
        if current_cache <= 0:
            return None

        protect_prompt = bool(getattr(self.kv_manager, "protect_prompt", True))
        prompt_len = int(getattr(self.kv_manager, "protected_prefix_len", 0))
        prune_start = prompt_len if protect_prompt else 0

        # Stable logical token base (preferred): total tokens seen so far in this sample.
        logical_total = None
        if self.token_tracker is not None and hasattr(self.token_tracker, "next_global_id"):
            try:
                logical_total = int(self.token_tracker.next_global_id)
            except Exception:
                logical_total = None
        if logical_total is None or logical_total <= 0:
            logical_total = current_cache

        trajectory_len = int(self.kv_manager.get_trajectory_len())
        obs_window_cfg = int(getattr(self.kv_manager, "observation_window", 0))
        obs_window = min(obs_window_cfg, max(0, trajectory_len))
        prune_end = current_cache - obs_window

        # If no prunable region exists, current cache size is the effective floor.
        if prune_end <= prune_start:
            return current_cache

        protected_total = prune_start + (current_cache - prune_end)
        prunable_len = max(1, prune_end - prune_start)

        if protect_prompt:
            logical_generated = max(0, logical_total - prompt_len)
            target_generated = int(logical_generated * ratio)
            # Keep at least 1 generated token once generation has started.
            if logical_generated > 0:
                target_generated = max(1, target_generated)
            target_total = prompt_len + target_generated
        else:
            target_total = int(logical_total * ratio)

        # Never target below currently protected tokens (prefix + protected suffix).
        target_total = max(int(target_total), int(protected_total))
        # Keep within feasible current cache bounds for this pruning call.
        return max(1, min(current_cache, int(target_total)))

    def _prune_prompt_prefix_once(self, prompt_len):
        """
        One-shot prompt-only pruning after initial prefill/generation.
        Keep top-k prompt tokens by attention score and keep all non-prompt tokens.

        Returns:
            new_prompt_len, kept_prompt_local_indices(list[int] or None)
        """
        if self.kv_manager is None or prompt_len <= 1:
            return prompt_len, None
        if self.past_key_values is None:
            return prompt_len, None

        # Guard against stale prompt_len: always align with actual cache length.
        try:
            if hasattr(self.past_key_values, "layers"):
                actual_cache_len = int(self.past_key_values.layers[0].keys.shape[2])
            else:
                actual_cache_len = int(self.past_key_values[0][0].shape[2])
        except Exception:
            return prompt_len, None

        prompt_len = int(min(max(0, int(prompt_len)), actual_cache_len))
        if prompt_len <= 1:
            return prompt_len, None
        keep_ratio = float(self.kv_config.get(
            "prompt_prefill_keep_ratio",
            self.kv_config.get("keep_ratio", 1.0),
        ))
        keep_ratio = max(0.0, min(1.0, keep_ratio))
        if keep_ratio >= 1.0:
            return prompt_len, None

        keep_prompt = max(1, min(prompt_len, int(math.ceil(prompt_len * keep_ratio))))
        if keep_prompt >= prompt_len:
            return prompt_len, None

        t0 = time.time()
        attentions = self._get_attention_for_scoring()
        self.timing_stats["scoring_time"] += time.time() - t0
        if attentions is None:
            return prompt_len, None

        scorer = getattr(getattr(self.kv_manager, "pruning_strategy", None), "h2o_scorer", None)
        if scorer is None:
            return prompt_len, None

        scores = scorer.compute_scores(attentions, 0, prompt_len)
        if scores is None or int(scores.shape[0]) != prompt_len:
            return prompt_len, None

        _, top_idx = torch.topk(scores, k=keep_prompt, dim=0)
        kept_prompt = torch.sort(top_idx.to(torch.long)).values

        device = kept_prompt.device
        suffix = torch.arange(prompt_len, actual_cache_len, device=device, dtype=torch.long)
        keep_indices = torch.cat([kept_prompt, suffix], dim=0)
        # Safety: unique + bounds clipping to avoid any CUDA indexing assert.
        keep_indices = torch.unique(keep_indices, sorted=True)
        keep_indices = keep_indices[(keep_indices >= 0) & (keep_indices < actual_cache_len)]
        if keep_indices.numel() <= 0:
            return prompt_len, None

        t1 = time.time()
        new_layers = []
        if hasattr(self.past_key_values, "layers"):
            for layer in self.past_key_values.layers:
                k = layer.keys
                v = layer.values
                # Layer-level safety for rare shape drift.
                layer_keep = keep_indices[(keep_indices >= 0) & (keep_indices < k.size(2))]
                if layer_keep.numel() <= 0:
                    layer_keep = torch.arange(0, min(1, k.size(2)), device=k.device, dtype=torch.long)
                new_layers.append((k[:, :, layer_keep, :], v[:, :, layer_keep, :]))
            # Avoid rebuilding DynamicCache via update() here because update()
            # appends along seq-dim and can trigger shape/device asserts.
            cache_copy = copy.deepcopy(self.past_key_values)
            for layer, (k_new, v_new) in zip(cache_copy.layers, new_layers):
                layer.keys = k_new.detach().clone()
                layer.values = v_new.detach().clone()
            self.past_key_values = cache_copy
        else:
            for k, v in self.past_key_values:
                layer_keep = keep_indices[(keep_indices >= 0) & (keep_indices < k.size(2))]
                if layer_keep.numel() <= 0:
                    layer_keep = torch.arange(0, min(1, k.size(2)), device=k.device, dtype=torch.long)
                new_layers.append((k[:, :, layer_keep, :], v[:, :, layer_keep, :]))
            self.past_key_values = tuple(new_layers)

        # Use true post-prune length from cache tensors.
        if hasattr(self.past_key_values, "layers"):
            self.current_cache_len = int(self.past_key_values.layers[0].keys.shape[2])
        else:
            self.current_cache_len = int(self.past_key_values[0][0].shape[2])
        self.timing_stats["pruning_time"] += time.time() - t1
        if self.kv_manager is not None:
            self.kv_manager.current_cache_len = self.current_cache_len

        kept_prompt_list = kept_prompt.detach().cpu().tolist()
        print(
            f"[PROMPT PREFILL PRUNE] prompt_kept={keep_prompt}/{prompt_len} "
            f"keep_ratio={keep_ratio:.3f} total_after={self.current_cache_len}"
        )
        return keep_prompt, kept_prompt_list

    def truncate_cache(self, keep_token_count):
        """
        Truncate the KV cache to keep only the first `keep_token_count` tokens.
        
        This is CRITICAL for correctness: after generation, the model may have 
        generated tokens beyond the useful Action line (e.g., hallucinated 
        Observation/Thought/Action for future steps). These hallucinated tokens
        must be removed from the KV cache before appending the real Observation,
        otherwise the model sees conflicting context.
        
        Args:
            keep_token_count: number of tokens to keep from the start of the cache
        """
        if self.past_key_values is None or keep_token_count >= self.current_cache_len:
            return
        
        if isinstance(self.past_key_values, DynamicCache):
            # Use DynamicCache's built-in crop method
            self.past_key_values.crop(keep_token_count)
        else:
            # Tuple of (key, value) pairs
            truncated = []
            for k, v in self.past_key_values:
                truncated.append((k[:, :, :keep_token_count, :], v[:, :, :keep_token_count, :]))
            self.past_key_values = tuple(truncated)
        
        old_len = self.current_cache_len
        self.current_cache_len = keep_token_count
        
        # Also truncate tracked token ids
        if len(self._all_token_ids) > keep_token_count:
            self._all_token_ids = self._all_token_ids[:keep_token_count]

        # Keep token tracker aligned with truncated logical trajectory.
        # Tokens removed here are hallucinated continuation beyond stop strings,
        # not pruning events, so we should not count them as discarded.
        if self.token_tracker is not None:
            try:
                if hasattr(self.token_tracker, "global_id_mapper"):
                    self.token_tracker.global_id_mapper = self.token_tracker.global_id_mapper[:keep_token_count]
                    self.token_tracker.cache_length = len(self.token_tracker.global_id_mapper)
                if hasattr(self.token_tracker, "next_global_id"):
                    # Preserve global-id monotonic semantics after truncation:
                    # next id should be max(existing_global_id)+1, not local length.
                    if hasattr(self.token_tracker, "global_id_mapper") and self.token_tracker.global_id_mapper:
                        self.token_tracker.next_global_id = max(self.token_tracker.global_id_mapper) + 1
                    else:
                        self.token_tracker.next_global_id = 0
            except Exception:
                pass
        
        # Update manager
        if self.kv_manager:
            self.kv_manager.current_cache_len = self.current_cache_len

    def generate_first(self, prompt_text, max_new_tokens=256, stop_strings=None):
        """
        First call:
        1. Encode full prompt
        2. Generate first Thought/Action
        3. Split KV into prompt KV and generated KV
        """
        self.reset()
    
        # =========================
        # 1) Build prompt
        # =========================
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt_text},
        ]
    
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]
        original_prompt_len = int(prompt_len)
    
        # =========================
        # 2) Generate
        # =========================
        t0 = time.time()
        with torch.no_grad():
            gen_outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                use_cache=True,
            )
        total_time = time.time() - t0
    
        full_sequence = gen_outputs.sequences
        generated_ids = full_sequence[0][prompt_len:].tolist()
    
        # =========================
        # 3) Get full KV
        # =========================
        if hasattr(gen_outputs, "past_key_values") and gen_outputs.past_key_values is not None:
            self.past_key_values = gen_outputs.past_key_values
        else:
            print("[WARN] generate() returned no past_key_values, fallback forward.")
            with torch.no_grad():
                outputs = self.model(
                    input_ids=full_sequence,
                    use_cache=True,
                    return_dict=True,
                )
            self.past_key_values = outputs.past_key_values
    
        self.current_cache_len = full_sequence.shape[1]

        # 3.5) Prompt-only prefill pruning (optional):
        # keep a ratio of prompt KV, keep all generated tokens.
        kept_prompt_local_indices = None
        if self.pruning_enabled and self.kv_config.get("pruning_mode") in ("step_aware_h2o", "step_inter"):
            prompt_len, kept_prompt_local_indices = self._prune_prompt_prefix_once(prompt_len)
    
        def _split_prompt_and_generated(local_prompt_len):
            prompt_part = []
            generated_part = []
            if hasattr(self.past_key_values, "layers"):
                for layer in self.past_key_values.layers:
                    k = layer.keys
                    v = layer.values
                    pk = k[:, :, :local_prompt_len, :].detach().clone()
                    pv = v[:, :, :local_prompt_len, :].detach().clone()
                    gk = k[:, :, local_prompt_len:, :].detach().clone()
                    gv = v[:, :, local_prompt_len:, :].detach().clone()
                    prompt_part.append((pk, pv))
                    generated_part.append((gk, gv))
            else:
                for k, v in self.past_key_values:
                    pk = k[:, :, :local_prompt_len, :].detach().clone()
                    pv = v[:, :, :local_prompt_len, :].detach().clone()
                    gk = k[:, :, local_prompt_len:, :].detach().clone()
                    gv = v[:, :, local_prompt_len:, :].detach().clone()
                    prompt_part.append((pk, pv))
                    generated_part.append((gk, gv))
            return tuple(prompt_part), tuple(generated_part)

        # =========================
        # 4) Split KV safely
        # =========================
        prompt_kv, generated_kv = _split_prompt_and_generated(prompt_len)
    
        print(f"[DEBUG] prompt layers = {len(prompt_kv)}")
        print(f"[DEBUG] generated layers = {len(generated_kv)}")
    
        if len(prompt_kv) > 0:
            print(f"[DEBUG] prompt_kv[0][0].shape = {prompt_kv[0][0].shape}")
            print(f"[DEBUG] generated_kv[0][0].shape = {generated_kv[0][0].shape}")
    
        # =========================
        # 5) timing
        # =========================
        self.timing_stats["prefill_time"] += total_time * 0.3
        self.timing_stats["decode_time"] += total_time * 0.7
    
        # =========================
        # 6) manager
        # =========================
        if self.kv_manager:
            self.kv_manager.register_initial_cache(prompt_len)
            self.kv_manager.current_cache_len = self.current_cache_len
        if self.token_tracker is not None:
            self.token_tracker.append_new_tokens(max(0, self.current_cache_len - prompt_len))

        all_ids = full_sequence[0].tolist()
        self._extend_global_token_log(all_ids)
        if kept_prompt_local_indices is not None:
            prompt_ids = all_ids[:original_prompt_len]
            kept_prompt_ids = [prompt_ids[i] for i in kept_prompt_local_indices if 0 <= i < len(prompt_ids)]
            self._all_token_ids = kept_prompt_ids + generated_ids
        else:
            self._all_token_ids = all_ids
    
        # =========================
        # 7) decode text
        # =========================
        response_text = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )
    
        # =========================
        # 8) stop truncation
        # =========================
        if stop_strings and response_text:
            truncated_text = response_text
    
            for stop_str in stop_strings:
                idx = truncated_text.find(stop_str)
                if idx != -1:
                    truncated_text = truncated_text[:idx]
    
            if len(truncated_text) < len(response_text):
                truncated_ids = self.tokenizer(
                    truncated_text,
                    add_special_tokens=False
                ).input_ids
    
                keep_count = prompt_len + len(truncated_ids)
                self.truncate_cache(keep_count)
    
                keep_generated = len(truncated_ids)
                new_generated_kv = []
    
                if hasattr(self.past_key_values, "layers"):
                    for layer in self.past_key_values.layers:
                        k = layer.keys
                        v = layer.values
                        gk = k[:, :, prompt_len:prompt_len + keep_generated, :].detach().clone()
                        gv = v[:, :, prompt_len:prompt_len + keep_generated, :].detach().clone()
                        new_generated_kv.append((gk, gv))
                else:
                    for k, v in self.past_key_values:
                        gk = k[:, :, prompt_len:prompt_len + keep_generated, :].detach().clone()
                        gv = v[:, :, prompt_len:prompt_len + keep_generated, :].detach().clone()
                        new_generated_kv.append((gk, gv))
    
                generated_kv = tuple(new_generated_kv)
                response_text = truncated_text

        # 9) For token-level pruning baselines, enforce budget right after first decode.
        # This makes step-1 behavior consistent with later steps (e.g., tova/h2o).
        if self.kv_manager and self.kv_config.get("pruning_mode") in ("h2o", "tova", "streamingllm", "tokenskipping", "pyramidinfer", "step_anchor_h2o"):
            self.kv_manager.current_cache_len = self.current_cache_len
            while True:
                target_budget = self._compute_h2o_budget()
                if target_budget is None or self.current_cache_len <= target_budget:
                    break
                before_len = int(self.current_cache_len)
                self._do_pruning(single_token_mode=True)
                if int(self.current_cache_len) >= before_len:
                    break
            # Re-split after first-step pruning so returned KV views match final cache.
            prompt_kv, generated_kv = _split_prompt_and_generated(prompt_len)
    
        return response_text.strip(), prompt_kv, generated_kv
    
    def generate_incremental_with_memory(self, new_text, prompt_kv, memory_block, recent_kv, max_new_tokens=256, stop_strings=None):
        """
        组合 Prompt, Memory 和 Recent KV，进行增量解码，并返回分离的增量 KV。
        (重写为与 generate_incremental 风格一致的清晰流程，行为与原实现保持一致)
        """
 
        print(f"[INFO] generate_incremental_with_memory called with new_text length={len(new_text)}")
        if prompt_kv is None or len(prompt_kv) == 0:
            print("[ERROR] prompt_kv is None or empty, cannot proceed")
            return "", None, None
        else:
            print(f"[DEBUG] prompt_kv layers = {len(prompt_kv)}")
            if len(prompt_kv) > 0:
                print(f"[DEBUG] prompt_kv[0][0].shape = {prompt_kv[0][0].shape}")

        if memory_block is None or len(memory_block) == 0:
            print("[ERROR] memory_block is None or empty, cannot proceed")
            return "", None, None
        else:
            print(f"[DEBUG] memory_block layers = {len(memory_block)}")
            if len(memory_block) > 0:
                print(f"[DEBUG] memory_block[0][0].shape = {memory_block[0][0].shape}")

        # 2) tokenize new_text
        new_input_ids = self.tokenizer(new_text, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        new_token_count = new_input_ids.shape[1]
        if new_token_count == 0:
            print("[WARN] new_text is empty, skipping prefill")
            return "", None, None

        # 3) 合并 prompt_kv, memory_block, recent_kv -> combined_pkv
        num_layers = len(prompt_kv)
        tuple_parts = []
        for i in range(num_layers):
            p_k, p_v = prompt_kv[i]
            m_k, m_v = memory_block[i]
            parts_k = [p_k, m_k]
            parts_v = [p_v, m_v]
            if recent_kv is not None and len(recent_kv) > i:
                r_k, r_v = recent_kv[i]
                parts_k.append(r_k)
                parts_v.append(r_v)
            layer_k = torch.cat(parts_k, dim=2)
            layer_v = torch.cat(parts_v, dim=2)
            tuple_parts.append((layer_k, layer_v))
        combined_pkv = tuple(tuple_parts)

        # 如果 self.past_key_values 是 DynamicCache，尝试构造相应的 DynamicCache 副本并写入 keys/values
        try:
            use_dynamic = self.past_key_values is not None and hasattr(self.past_key_values, "layers")
        except Exception:
            use_dynamic = False

        if use_dynamic:
            cache_copy = copy.deepcopy(self.past_key_values)
            for i, layer in enumerate(cache_copy.layers):
                k_new, v_new = tuple_parts[i]
                try:
                    layer.keys = k_new.detach().clone()
                    layer.values = v_new.detach().clone()
                except Exception:
                    use_dynamic = False
                    break
            if use_dynamic:
                combined_pkv = cache_copy

        # Ensure we pass a DynamicCache to the model when possible.
        # If combined_pkv is already a DynamicCache (cache_copy above), keep it.
        # Otherwise try to convert from legacy tuple via from_legacy_cache.
        converted_cache = None
        if isinstance(combined_pkv, DynamicCache):
            converted_cache = combined_pkv
        else:
            # First try the native conversion helper (preferred)
            try:
                converted_cache = DynamicCache.from_legacy_cache(combined_pkv)
            except Exception as e:
                # Fallback: build a lightweight DynamicCache-like wrapper that provides
                # the attributes the model expects (layers with keys/values, get_seq_length, crop).
                print(f"[WARN] DynamicCache.from_legacy_cache failed: {e}; constructing fallback DynamicCache-like object.")

                class _SimpleLayer:
                    def __init__(self, k, v):
                        self.keys = k.detach().clone()
                        self.values = v.detach().clone()

                class _SimpleDynamicCache:
                    def __init__(self, kv_tuple):
                        # kv_tuple: tuple of (k, v) per layer
                        self.layers = []
                        for k, v in kv_tuple:
                            self.layers.append(_SimpleLayer(k, v))

                    def get_seq_length(self):
                        if not self.layers:
                            return 0
                        return self.layers[0].keys.shape[2]

                    def crop(self, keep_token_count):
                        for layer in self.layers:
                            layer.keys = layer.keys[:, :, :keep_token_count, :].detach().clone()
                            layer.values = layer.values[:, :, :keep_token_count, :].detach().clone()

                    def get_mask_sizes(self, cache_position=None, layer_idx=None):
                        """
                        Compatibility shim for transformers' masking utilities.
                        Returns (kv_length, kv_offset) where kv_length is the length of
                        the cached key/value sequence and kv_offset is the offset
                        (we return 0 since this simple cache uses absolute positions).
                        """
                        kv_length = self.get_seq_length()
                        kv_offset = 0
                        return kv_length, kv_offset
                    
                    @property
                    def device(self):
                        # return device of underlying tensors if available
                        if not self.layers:
                            return None
                        return self.layers[0].keys.device

                    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
                        """
                        Update the cache with new key and value states for a specific layer.
                        This method is called by the model during forward pass.
                        
                        Args:
                            key_states: new key tensor to append
                            value_states: new value tensor to append
                            layer_idx: index of the layer being updated
                            cache_kwargs: optional kwargs (unused in this simple implementation)
                        
                        Returns:
                            Tuple of (updated_key_states, updated_value_states)
                        """
                        # Ensure layer index is valid
                        while len(self.layers) <= layer_idx:
                            self.layers.append(_SimpleLayer(
                                torch.empty((1, 1, 0, key_states.shape[-1]), 
                                          dtype=key_states.dtype, device=key_states.device),
                                torch.empty((1, 1, 0, value_states.shape[-1]), 
                                          dtype=value_states.dtype, device=value_states.device)
                            ))
                        
                        # Concatenate new key/value with existing cache
                        if self.layers[layer_idx].keys.shape[2] == 0:
                            # Empty cache, initialize with new states
                            self.layers[layer_idx].keys = key_states.detach().clone()
                            self.layers[layer_idx].values = value_states.detach().clone()
                        else:
                            # Append to existing cache
                            self.layers[layer_idx].keys = torch.cat(
                                [self.layers[layer_idx].keys, key_states], dim=2
                            ).detach().clone()
                            self.layers[layer_idx].values = torch.cat(
                                [self.layers[layer_idx].values, value_states], dim=2
                            ).detach().clone()
                        
                        # Return full cached key/value (what the model expects)
                        return self.layers[layer_idx].keys, self.layers[layer_idx].values

                # If combined_pkv is tuple of layers, use it; otherwise try tuple_parts as source
                source_kv = combined_pkv if isinstance(combined_pkv, tuple) else tuple_parts
                try:
                    converted_cache = _SimpleDynamicCache(source_kv)
                except Exception as e2:
                    # catastrophic fallback: leave as tuple but warn (model may still fail)
                    print(f"[ERROR] Failed to construct fallback DynamicCache-like: {e2}. Passing original tuple (may error).")
                    converted_cache = combined_pkv

        combined_pkv = converted_cache
 
         # 4) Prefill Observation（带 past_key_values）
        t0 = time.time()
        with torch.no_grad():
            outputs = self.model(
                input_ids=new_input_ids,
                past_key_values=combined_pkv,
                use_cache=True,
                return_dict=True
            )
        self.timing_stats["prefill_time"] += (time.time() - t0)

        # 5) 提取本次 Observation 的增量 KV (obs_kv_pairs)，兼容 DynamicCache / tuple
        obs_kv_pairs = []
        if outputs.past_key_values is not None:
            if hasattr(outputs.past_key_values, "layers"):
                for layer in outputs.past_key_values.layers:
                    k = layer.keys
                    v = layer.values
                    s_k = k[:, :, -new_token_count:, :].detach().clone()
                    s_v = v[:, :, -new_token_count:, :].detach().clone()
                    obs_kv_pairs.append((s_k, s_v))
            else:
                for k, v in outputs.past_key_values:
                    s_k = k[:, :, -new_token_count:, :].detach().clone()
                    s_v = v[:, :, -new_token_count:, :].detach().clone()
                    obs_kv_pairs.append((s_k, s_v))
        obs_kv_pairs = tuple(obs_kv_pairs)

        # 6) 更新内部 past_key_values / current_cache_len，并更新 manager 状态（保持原逻辑）
        self.past_key_values = outputs.past_key_values
        self.current_cache_len += new_token_count
        # if self.kv_manager:
        #     self.kv_manager.append_step(new_token_count)
        #     self.kv_manager.current_cache_len = self.current_cache_len
        #     if self.kv_manager.should_prune():
        #         # 输出 attentions 仅在需要时传入
        #         need_attention = (self.pruning_enabled and self.attn_mode == "piggyback" and self.kv_config.get("pruning_mode") != "snapkv")
        #         piggyback_attentions = outputs.attentions if need_attention else None
        #         #self._do_pruning(piggyback_attentions, step_kv=obs_kv_pairs, step_token_count=new_token_count)

        # # 7) track token ids and record cache length before decode
        # self._all_token_ids.extend(new_input_ids[0].tolist())
        cache_len_before_decode = self.current_cache_len

        # 8) Decode 下一轮 Thought/Action（调用现有 _decode，保持行为）
        response_text, generated_len = self._decode(outputs.logits, max_new_tokens)

        # 9) 提取模型生成的增量 KV (gen_kv_pairs)，兼容 DynamicCache / tuple
        gen_kv_pairs = []
        if generated_len > 0 and self.past_key_values is not None:
            if hasattr(self.past_key_values, "layers"):
                for layer in self.past_key_values.layers:
                    k = layer.keys
                    v = layer.values
                    g_k = k[:, :, -generated_len:, :].detach().clone()
                    g_v = v[:, :, -generated_len:, :].detach().clone()
                    gen_kv_pairs.append((g_k, g_v))
            else:
                for k, v in self.past_key_values:
                    g_k = k[:, :, -generated_len:, :].detach().clone()
                    g_v = v[:, :, -generated_len:, :].detach().clone()
                    gen_kv_pairs.append((g_k, g_v))
        gen_kv_pairs = tuple(gen_kv_pairs)

        # 10) stop_strings 截断逻辑（保持原有行为：根据需要截断并调整 cache）
        if stop_strings and response_text:
            truncated_text = response_text
            for stop_str in stop_strings:
                idx = truncated_text.find(stop_str)
                if idx != -1:
                    truncated_text = truncated_text[:idx]

            if len(truncated_text) < len(response_text):
                truncated_ids = self.tokenizer(truncated_text, add_special_tokens=False).input_ids
                keep_len = len(truncated_ids)

                # 截断生成的 KV 部分（仅在需要时）
                if keep_len < generated_len and self.past_key_values is not None:
                    new_past_kv = []
                    if hasattr(self.past_key_values, "layers"):
                        # 如果当前 past 是 DynamicCache，无法直接用 tuple 替换层对象，回退为 tuple 截断
                        for layer in self.past_key_values.layers:
                            k = layer.keys
                            v = layer.values
                            new_k = k[:, :, :-(generated_len - keep_len), :] if generated_len > keep_len else k
                            new_v = v[:, :, :-(generated_len - keep_len), :] if generated_len > keep_len else v
                            new_past_kv.append((new_k, new_v))
                    else:
                        for k, v in self.past_key_values:
                            new_k = k[:, :, :-(generated_len - keep_len), :] if generated_len > keep_len else k
                            new_v = v[:, :, :-(generated_len - keep_len), :] if generated_len > keep_len else v
                            new_past_kv.append((new_k, new_v))
                    self.past_key_values = tuple(new_past_kv)

                response_text = truncated_text

        # 11) 返回（原先错误地只返回了 keys-only，为兼容拼接改回返回 (k,v) pairs）
        return response_text.strip(), obs_kv_pairs, gen_kv_pairs
    
    def generate_incremental(self, new_text, max_new_tokens=256, stop_strings=None):
        """
        Subsequent calls: only encode new_text, reuse cached KV.

        IMPORTANT: To match baseline behavior, we do NOT wrap new_text with role markers.
        In baseline, the entire trajectory (Thought/Action/Observation sequence) is inside
        a single user message, and the model generates continuously. Here, we:
        1. Keep all content (initial prompt + model output + observation) in the same
           KV cache without role switching
        2. Append observation directly, allowing the model to continue as if it were
           continuing from within the same user message context

        The KV cache ends at the model's generated tokens (Thought/Action).
        We append Observation directly, and the model generates the next Thought/Action.
        This matches baseline's behavior where observation is part of the running trajectory.

        After generation, the response is truncated at stop_strings to prevent
        hallucinated future content from polluting the KV cache.

        Args:
            new_text: new text to append (e.g., Observation content)
            max_new_tokens: maximum tokens to generate
            stop_strings: list of strings that should trigger truncation

        Returns:
            response_text: generated text (truncated to useful portion)
        """
        assert self.past_key_values is not None, \
            "Must call generate_first() before generate_incremental()"

        # CRITICAL FIX: Do NOT wrap with role markers to match baseline behavior.
        # In baseline, the trajectory (Thought/Action/Observation) is all in one user message.
        # The model sees Observation as continuation of the reasoning trace, NOT a new turn.
        # Simply append the new_text (Observation) directly.
        # This ensures the model's attention pattern matches the baseline.
        wrapped_text = new_text

        # Tokenize only the wrapped text
        new_input_ids = self.tokenizer(
            wrapped_text, return_tensors="pt", add_special_tokens=False
        ).input_ids.to(self.device)
        new_token_count = new_input_ids.shape[1]

        # Prefill new tokens with existing KV cache
        t0 = time.time()
        need_attention = (self.pruning_enabled and
                          self.attn_mode == "piggyback" and
                          self.kv_config.get("pruning_mode") != "snapkv")

        with torch.no_grad():
            outputs = self.model(
                input_ids=new_input_ids,
                past_key_values=self.past_key_values,
                use_cache=True,
                return_dict=True,
                output_attentions=need_attention,
            )
        self.last_prefill_attentions = outputs.attentions if need_attention else None
        self.past_key_values = outputs.past_key_values
        self.current_cache_len += new_token_count
        if self.token_tracker is not None:
            self.token_tracker.append_new_tokens(new_token_count)
        prefill_time = time.time() - t0
        self.timing_stats["prefill_time"] += prefill_time

        # Update manager
        if self.kv_manager:
            self.kv_manager.append_step(new_token_count)
            self.kv_manager.current_cache_len = self.current_cache_len
            if self.kv_config.get("pruning_mode") == "step_inter" and need_attention:
                self.kv_manager.update_step_scores_from_attention(
                    outputs.attentions,
                    query_token_count=new_token_count,
                )

        # Check if pruning is needed
        if self.kv_manager:
            piggyback_attentions = outputs.attentions if need_attention else None
            if self.kv_config.get("pruning_mode") in ("h2o", "tova", "streamingllm", "tokenskipping", "pyramidinfer", "step_anchor_h2o"):
                target_budget = self._compute_h2o_budget()
                if target_budget is not None and self.current_cache_len > target_budget:
                    # Score once per step, batch-prune, then token top-up with cached attentions.
                    step_attentions = self._resolve_scoring_attentions(piggyback_attentions)
                    self._do_pruning(
                        piggyback_attentions=piggyback_attentions,
                        single_token_mode=False,
                        attentions_override=step_attentions,
                    )
                    max_iters = max(0, int(self.current_cache_len - target_budget) + 8)
                    for _ in range(max_iters):
                        if self.current_cache_len <= target_budget:
                            break
                        before_len = self.current_cache_len
                        self._do_pruning(
                            attentions_override=step_attentions,
                            single_token_mode=True,
                        )
                        if self.current_cache_len >= before_len:
                            break
                    del step_attentions
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            elif self.kv_config.get("pruning_mode") not in ("step_aware_h2o", "step_inter") and self.kv_manager.should_prune():
                self._do_pruning(piggyback_attentions)

        # Track new token ids for repetition penalty
        new_ids = new_input_ids[0].tolist()
        self._extend_global_token_log(new_ids)
        self._all_token_ids.extend(new_ids)

        # Record cache length before decode, needed for truncation
        cache_len_before_decode = self.current_cache_len

        # Decode using fast model.generate() - no per-token pruning
        # Pruning happens at step boundaries (after prefill or after complete generation)
        response_text, generated_len = self._decode(
            outputs.logits, max_new_tokens
        )
        
        # After decode completes, if additional pruning is needed during generation,
        # it would be handled here. For now, pruning is only at step boundaries.
        if self.kv_manager:
            self.kv_manager.current_cache_len = self.current_cache_len

        # Truncate at stop_strings if found (remove hallucinated future content)
        if stop_strings and response_text:
            truncated_text = response_text
            for stop_str in stop_strings:
                idx = truncated_text.find(stop_str)
                if idx != -1:
                    truncated_text = truncated_text[:idx]
            
            if len(truncated_text) < len(response_text):
                # Re-tokenize the truncated text to find how many tokens to keep
                truncated_ids = self.tokenizer(
                    truncated_text, add_special_tokens=False
                ).input_ids
                keep_count = cache_len_before_decode + len(truncated_ids)
                self.truncate_cache(keep_count)
                response_text = truncated_text

        # Step-aware mode: prune once at step boundary (after full decode + truncation).
        if self.kv_manager and self.kv_config.get("pruning_mode") in ("step_aware_h2o", "step_inter"):
            piggyback_attentions = outputs.attentions if need_attention else None
            self._do_pruning(piggyback_attentions=piggyback_attentions, single_token_mode=False)
            # Budget top-up: keep pruning one token until the configured target ratio is met.
            target_budget = self._compute_target_budget()
            if target_budget is not None and self.current_cache_len > target_budget:
                # Cap iterations defensively to avoid accidental infinite loops.
                max_iters = max(0, int(self.current_cache_len - target_budget) + 8)
                for _ in range(max_iters):
                    if self.current_cache_len <= target_budget:
                        break
                    before_len = int(self.current_cache_len)
                    self._do_pruning(piggyback_attentions=piggyback_attentions, single_token_mode=True)
                    # Stop if prune did not make progress.
                    if int(self.current_cache_len) >= before_len:
                        break

        return response_text.strip() if response_text else response_text
    

    def _resolve_scoring_attentions(self, piggyback_attentions=None):
        """Resolve attention weights for one prune step (score at most once)."""
        if self.attn_mode == "piggyback" and piggyback_attentions is not None:
            return piggyback_attentions
        t0 = time.time()
        attentions = self._get_attention_for_scoring()
        self.timing_stats["scoring_time"] += time.time() - t0
        return attentions

    def _do_pruning(self, piggyback_attentions=None, step_kv=None, step_token_count=None, single_token_mode=False, attentions_override=None):
        """
        执行 KV cache 剪枝/压缩。
        
        Args:
            piggyback_attentions: 预填充阶段捕获的注意力权重
            step_kv: 本次 Observation 新增的 KV 矩阵 (元组形式)
            step_token_count: 本次新增 Token 的数量
        """
        attentions = None
        mode = self.kv_config.get("pruning_mode", "h2o_snapkv")

        # --- 1. 获取注意力权重 (原有逻辑保留) ---
        if attentions_override is not None:
            attentions = attentions_override
        elif mode not in ("snapkv", "streamingllm", "tokenskipping"):
            if self.attn_mode == "scoring_forward":
                t0 = time.time()
                attentions = self._get_attention_for_scoring()
                self.timing_stats["scoring_time"] += time.time() - t0
            elif self.attn_mode == "piggyback" and piggyback_attentions is not None:
                attentions = piggyback_attentions
            else:
                t0 = time.time()
                attentions = self._get_attention_for_scoring()
                self.timing_stats["scoring_time"] += time.time() - t0

        # --- 2. 同步状态 ---
        self.kv_manager.current_cache_len = self.current_cache_len

        # --- 3. 调用核心算法 ---
        t0 = time.time()
        # 确保 pruning_strategy 上存在 num_score_layers（kv_cache_manager 可能直接访问该属性）
        try:
            if self.kv_manager and hasattr(self.kv_manager, "pruning_strategy"):
                ps = self.kv_manager.pruning_strategy
                if not hasattr(ps, "num_score_layers"):
                    setattr(ps, "num_score_layers", self.kv_config.get("num_score_layers", 1))
        except Exception as e:
            print(f"[WARN] Failed to set pruning_strategy.num_score_layers: {e}")

        # 尝试多种 prune() 签名以兼容不同实现
        prune_call_variants = []

        # 常见带 step_token_count 的签名
        prune_call_variants.append({
            "past_key_values": self.past_key_values,
            "attentions": attentions,
            "step_kv": step_kv,
            "step_token_count": step_token_count
        })
        # 有实现可能使用 new_step_token_count
        prune_call_variants.append({
            "past_key_values": self.past_key_values,
            "attentions": attentions,
            "step_kv": step_kv,
            "new_step_token_count": step_token_count
        })
        # 没有 step count 的签名
        prune_call_variants.append({
            "past_key_values": self.past_key_values,
            "attentions": attentions,
            "step_kv": step_kv
        })
        # 最简签名（仅 past_key_values, attentions）
        prune_call_variants.append({
            "past_key_values": self.past_key_values,
            "attentions": attentions
        })

        new_kv = None
        new_len = self.current_cache_len
        info = None
        prune_success = False
        last_exception = None

        for kwargs in prune_call_variants:
            try:
                if single_token_mode and hasattr(self.kv_manager, "prune_one_token"):
                    result = self.kv_manager.prune_one_token(
                        past_key_values=self.past_key_values,
                        attentions=attentions,
                    )
                else:
                    result = self.kv_manager.prune(**kwargs)
                # 支持返回 (new_kv, new_len, info) 或 (new_kv, new_len)
                if isinstance(result, tuple) or isinstance(result, list):
                    if len(result) == 3:
                        new_kv, new_len, info = result
                    elif len(result) == 2:
                        new_kv, new_len = result
                        info = None
                    else:
                        # 不常见的返回，尝试按前两个元素解释
                        new_kv = result[0]
                        new_len = result[1] if len(result) > 1 else self.current_cache_len
                        info = result[2] if len(result) > 2 else None
                else:
                    # 如果返回单个对象，尝试从对象属性读取
                    try:
                        new_kv = result.past_key_values if hasattr(result, "past_key_values") else result
                        new_len = getattr(result, "new_len", self.current_cache_len)
                        info = getattr(result, "info", None)
                    except Exception:
                        new_kv = result
                        new_len = self.current_cache_len
                        info = None
                prune_success = True
                break
            except TypeError as te:
                # 参数不匹配，尝试下一个签名
                last_exception = te
                continue
            except Exception as e:
                # 记录错误并继续尝试其它签名
                last_exception = e
                print(f"[WARN] Pruning strategy failed for kwargs={list(kwargs.keys())}: {e}")
                continue

        if not prune_success:
            print(f"[WARN] Prune failed for all tried signatures. Last exception: {last_exception}")
            # 记录耗时并返回（不改变状态）
            self.timing_stats["pruning_time"] += time.time() - t0
            return

        # --- 4. 更新状态 ---
        try:
            self.past_key_values = new_kv
            self.current_cache_len = new_len
        except Exception as e:
            print(f"[WARN] Failed to apply prune results: {e}")
        self.timing_stats["pruning_time"] += time.time() - t0
        
        # 记录剪枝历史（可选）
        if hasattr(self, 'pruning_history'):
            self.pruning_history.append(info)

    def _get_attention_for_scoring(self):
        """
        Run a forward pass with output_attentions=True to get attention weights.

        Uses the last token in the cache as the query.
        Attention shape per layer: (1, num_heads, 1, cache_len+1),
        so memory overhead is only O(num_layers * num_heads * cache_len).

        IMPORTANT: We must NOT update self.past_key_values with the dummy token.
        We deep-copy the cache before the forward pass so the original is untouched.

        We temporarily switch to eager attention for this pass since SDPA
        does not support output_attentions=True.
        """
        dummy_input = torch.tensor([[self.tokenizer.eos_token_id]], device=self.device)

        # Deep copy the cache so the dummy token doesn't pollute it
        if isinstance(self.past_key_values, DynamicCache):
            cache_copy = copy.deepcopy(self.past_key_values)
        else:
            cache_copy = tuple(
                (k.clone(), v.clone()) for k, v in self.past_key_values
            )

        try:
            # Temporarily switch to eager attention for output_attentions support.
            # All attention layers read from self.config._attn_implementation dynamically
            # in their forward(), so we only need to change the config object.
            original_impl = getattr(self.model.config, '_attn_implementation', None)
            self.model.config._attn_implementation = 'eager'

            with torch.no_grad():
                outputs = self.model(
                    input_ids=dummy_input,
                    past_key_values=cache_copy,
                    use_cache=True,
                    return_dict=True,
                    output_attentions=True,
                )

            # Restore original attention implementation
            self.model.config._attn_implementation = original_impl

            attentions = outputs.attentions
            if attentions is None or len(attentions) == 0:
                print("[WARN] output_attentions=True returned None even with eager attention.")
                return None
            return attentions
        except Exception as e:
            # Restore on failure
            try:
                self.model.config._attn_implementation = original_impl
            except Exception:
                pass
            print(f"[WARN] Scoring forward failed: {e}")
            return None


    def _decode(self, last_logits, max_new_tokens):
        """
        Decode using model.generate() with past_key_values for efficient generation.

        After the caller has done a manual prefill (encoding new observation tokens
        into the KV cache), this method handles the autoregressive decode phase
        using model.generate() — which is much faster than a manual Python
        token-by-token loop.

        Strategy:
        - Get the first decoded token from last_logits (argmax)
        - Pass this single token to model.generate() along with past_key_values
          AND an explicit cache_position to avoid HF's internal cache_position
          computation bug (which produces an empty tensor when past_key_values
          length exceeds input_ids length)
        - model.generate() processes this token (adding it to KV cache) then
          continues decoding efficiently using its optimized internal loop
        - The first token in the output IS the token we passed in, so the full
          generated sequence is returned correctly

        Args:
            last_logits: logits from the prefill step, shape (1, seq_len, vocab_size).
                         We use the last position's logits to get the first decode token.
            max_new_tokens: maximum tokens to generate

        Returns:
            response_text: decoded string
            generated_len: number of tokens generated
        """
        if self.kv_config.get("pruning_mode") in ("h2o", "tova", "streamingllm", "tokenskipping", "pyramidinfer", "step_anchor_h2o"):
            return self._decode_token_by_token_with_pruning(last_logits, max_new_tokens)

        t0 = time.time()

        # Get the first decode token from the prefill logits
        first_token_logits = last_logits[:, -1, :]  # (1, vocab_size)
        first_token_id = first_token_logits.argmax(dim=-1, keepdim=True)  # (1, 1)

        # Check if the first token is already an EOS token
        gen_config = getattr(self.model, 'generation_config', None)
        if gen_config and gen_config.eos_token_id is not None:
            eos_ids = gen_config.eos_token_id
            if isinstance(eos_ids, int):
                eos_ids = {eos_ids}
            else:
                eos_ids = set(eos_ids)
        else:
            eos_ids = {self.tokenizer.eos_token_id}

        if first_token_id.item() in eos_ids:
            # Model wants to stop immediately
            self.timing_stats["decode_time"] += time.time() - t0
            return "", 0

        # Use model.generate() with the first token + past_key_values.
        # generate() will:
        # 1. Process first_token_id (1 token prefill, adding it to KV cache)
        # 2. Run optimized autoregressive decode for remaining tokens
        #
        # We need attention_mask covering: existing KV cache + this 1 new token
        attention_mask = torch.ones(
            (1, self.current_cache_len + 1), dtype=torch.long, device=self.device
        )

        # CRITICAL: Provide cache_position explicitly.
        # HF's _get_initial_cache_position computes:
        #   cache_position = [0, ..., seq_len-1][past_length:]
        # When seq_len=1 (our single token) and past_length=N (KV cache),
        # this gives an EMPTY tensor → IndexError on cache_position[-1].
        # By passing cache_position ourselves, HF skips that computation.
        # The first token's position is self.current_cache_len (right after
        # all cached tokens).
        cache_position = torch.tensor(
            [self.current_cache_len], dtype=torch.long, device=self.device
        )

        with torch.no_grad():
            gen_outputs = self.model.generate(
                input_ids=first_token_id,
                attention_mask=attention_mask,
                past_key_values=self.past_key_values,
                cache_position=cache_position,
                max_new_tokens=max_new_tokens - 1,  # -1 because first token is already selected
                do_sample=False,
                temperature=None,
                top_p=None,
                return_dict_in_generate=True,
                use_cache=True,
            )

        # gen_outputs.sequences = [first_token_id, generated_token_1, ...]
        # The first token in sequences is our input (first_token_id)
        all_generated = gen_outputs.sequences[0].tolist()  # includes first_token_id

        # Update past_key_values from generation output
        if hasattr(gen_outputs, 'past_key_values') and gen_outputs.past_key_values is not None:
            self.past_key_values = gen_outputs.past_key_values

        # Update cache length: add all generated tokens (including first_token_id)
        self.current_cache_len += len(all_generated)

        self.timing_stats["decode_time"] += time.time() - t0

        # Track generated ids for future repetition penalty
        self._extend_global_token_log(all_generated)
        self._all_token_ids.extend(all_generated)
        if self.token_tracker is not None:
            self.token_tracker.append_new_tokens(len(all_generated))

        # Update manager cache len
        if self.kv_manager:
            self.kv_manager.current_cache_len = self.current_cache_len

        response_text = self.tokenizer.decode(all_generated, skip_special_tokens=True)
        return response_text.strip(), len(all_generated)

    def _decode_token_by_token_with_pruning(self, last_logits, max_new_tokens):
        """
        H2O path: autoregressive decode with per-token prune checks.
        """
        t0 = time.time()
        generated_ids = []

        gen_config = getattr(self.model, 'generation_config', None)
        if gen_config and gen_config.eos_token_id is not None:
            eos_ids = gen_config.eos_token_id
            if isinstance(eos_ids, int):
                eos_ids = {eos_ids}
            else:
                eos_ids = set(eos_ids)
        else:
            eos_ids = {self.tokenizer.eos_token_id}

        next_token_id = last_logits[:, -1, :].argmax(dim=-1, keepdim=True)

        for _ in range(max_new_tokens):
            if next_token_id.item() in eos_ids:
                break

            with torch.no_grad():
                out = self.model(
                    input_ids=next_token_id,
                    past_key_values=self.past_key_values,
                    use_cache=True,
                    return_dict=True,
                )

            self.past_key_values = out.past_key_values
            self.current_cache_len += 1
            if self.kv_manager:
                self.kv_manager.append_step(1)
                self.kv_manager.current_cache_len = self.current_cache_len
            if self.token_tracker is not None:
                self.token_tracker.append_new_tokens(1)

            token_id_int = int(next_token_id.item())
            generated_ids.append(token_id_int)
            self._extend_global_token_log([token_id_int])
            self._all_token_ids.append(token_id_int)

            # Token-level budget control for H2O.
            if self.kv_manager:
                target_budget = self._compute_h2o_budget()
                if target_budget is not None and self.current_cache_len > target_budget:
                    self._do_pruning(single_token_mode=True)
                    self.kv_manager.current_cache_len = self.current_cache_len

            next_token_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

        self.timing_stats["decode_time"] += time.time() - t0
        response_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response_text.strip(), len(generated_ids)

    def get_cache_len(self):
        """Get current KV cache length."""
        return self.current_cache_len
    
    def get_last_pruning_info(self):
        """Get the last pruning event information."""
        if self.kv_manager and self.kv_manager.pruning_history:
            return self.kv_manager.pruning_history[-1]
        return None
    
    def get_pruning_history(self):
        """Get complete pruning history."""
        if self.kv_manager:
            return self.kv_manager.pruning_history
        return []

    def _extend_global_token_log(self, token_ids) -> None:
        """Append vocab ids in global-id order (never truncated after pruning)."""
        if not token_ids:
            return
        if not hasattr(self, "_global_token_id_log"):
            self._global_token_id_log = []
        self._global_token_id_log.extend(int(x) for x in token_ids)

    def get_global_token_id_log(self):
        """Return append-only global_id -> vocab token id mapping."""
        return list(getattr(self, "_global_token_id_log", []) or [])

    def get_stats(self):
        """Get combined timing and pruning stats."""
        stats = dict(self.timing_stats)
        if self.kv_manager:
            stats.update(self.kv_manager.get_stats())
        stats["current_cache_len"] = self.current_cache_len
        return stats
