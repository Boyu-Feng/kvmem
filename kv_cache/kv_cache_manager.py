"""
KV Cache Manager
Top-level manager for Agent trajectory KV cache memory.
Handles cache segmentation, pruning scheduling, and position remapping.
"""

import torch
from .pruning_strategy import PruningStrategy
from .position_remapper import PositionRemapper


class KVCacheManager:
    """
    Top-level KV Cache memory manager for Agent trajectories.

    KV Cache Layout:
        [Sink Tokens] [System+Question Tokens] [Trajectory Tokens]
        <-- protected prefix -----------------> <-- prunable ---->

    The observation_window protects the last M tokens of the trajectory
    from being pruned, ensuring the model always sees recent context.
    """

    def __init__(self, config, token_tracker=None):
        """
        Args:
            config: dict with keys:
                - pruning_mode: "h2o" | "snapkv" | "h2o_snapkv" | "ours" 
                - prune_every_n: trigger pruning every N ReAct steps (default: 2)
                - max_trajectory_tokens: hard budget for trajectory region (default: 1024)
                - cache_ratio: unified cache retention ratio (default: 0.5)
                - pool_window: SnapKV pooling window size (default: 4)
                - sink_size: Attention Sink token count (default: 4)
                - observation_window: recent tokens protected from pruning (default: 128)
                - num_score_layers: layers used for H2O scoring (default: 3)
            token_tracker: optional TokenTracker instance for tracking pruned tokens
        """
        self.pruning_mode = config.get("pruning_mode", "h2o")
        self.prune_every_n = config.get("prune_every_n", 1)
        self.max_trajectory_tokens = config.get("max_trajectory_tokens", 1024)
        self.cache_ratio = float(
            config.get(
                "cache_ratio",
                config.get("target_cache_ratio", config.get("keep_ratio", 0.5)),
            )
        )
        self.cache_ratio = max(0.0, min(1.0, self.cache_ratio))
        # Backward-compatible aliases for legacy code paths.
        self.keep_ratio = self.cache_ratio
        self.target_cache_ratio = self.cache_ratio
        # If False, pruning can start from token 0 (including prompt/system tokens).
        self.protect_prompt = config.get("protect_prompt", True)
        self.observation_window = config.get("observation_window", 128)
        self.sink_size = config.get("sink_size", 4)
        self.memory_rank = config.get("memory_rank", 128)
        self.token_tracker = token_tracker
        self.step_anchor_keep_last_obs = config.get("step_anchor_keep_last_obs", 1)
        self.step_aware_alpha = float(config.get("step_aware_alpha", 0.7))
        self.step_aware_beta = float(config.get("step_aware_beta", 0.3))
        self.step_aware_min_keep = int(config.get("step_aware_min_keep", 5))
        self.step_aware_min_keep_ratio = float(config.get("step_aware_min_keep_ratio", 0.0))
        self.step_aware_bonus = float(config.get("step_aware_bonus", 0.0))
        self.step_poolwise_prune = bool(config.get("step_poolwise_prune", False))
        self.step_inter_ema = float(config.get("step_inter_ema", 0.7))
        self.pyramid_h2o_weight = float(config.get("pyramid_h2o_weight", 0.7))
        self.pyramid_recency_power = float(config.get("pyramid_recency_power", 1.5))
        self.step_spans = []
        self.step_scores = {}
        self.step_force_drop_ids = set(int(x) for x in config.get("step_force_drop_ids", []) or [])

        self.pruning_strategy = PruningStrategy(
            mode=self.pruning_mode,
            num_score_layers=config.get("num_score_layers", 3),
            pool_window=config.get("pool_window", 4),
            memory_rank=self.memory_rank,
            token_tracker=token_tracker,
            pyramid_h2o_weight=self.pyramid_h2o_weight,
            pyramid_recency_power=self.pyramid_recency_power,
        )
        self.position_remapper = PositionRemapper(sink_size=self.sink_size)

        # State tracking
        self.step_count = 0
        self.total_prune_count = 0
        self.protected_prefix_len = 0  # Length of [sink + system + question]
        self.current_cache_len = 0
        self.last_pruned = False
        self.new_step_kv_lengths = 0
        self.pruning_history = []

        # Per-step segments recorded for future fusion strategies.
        # Each entry: {"id": optional id, "len": token_count}
        # The actual per-step KV blobs are provided to merge_step_kv(...) when needed.
        self.step_segments = []
        # NOTE: We do NOT keep strong references to raw KV tensors here by default
        # to avoid unexpected memory growth; callers should pass step KV into
        # merge_step_kv(...) when they want it merged into the global cache.

    def register_initial_cache(self, cache_len):
        """
        Called after the first forward pass (System Prompt + Question).
        The entire initial cache is treated as protected prefix.

        Args:
            cache_len: total length of KV cache after encoding system + question
        """
        self.protected_prefix_len = cache_len
        self.current_cache_len = cache_len
        self.step_count = 0
        self.total_prune_count = 0
        self.last_pruned = False
        self.pruning_history = []
        self.step_spans = []
        self.step_scores = {}
        
        # Initialize token tracker if available
        if self.token_tracker is not None:
            self.token_tracker.set_initial_cache_length(cache_len)

    def append_step(self, new_tokens_len):
        """
        Called after each Thought/Action/Observation is appended to the cache.

        Args:
            new_tokens_len: number of new tokens added in this step
        """
        self.current_cache_len += new_tokens_len
        self.step_count += 1

    def update_step_spans(self, step_spans):
        """Update externally tracked step spans (global token IDs)."""
        self.step_spans = step_spans or []

    def update_step_scores(self, step_scores):
        """Update externally tracked step importance scores by step_id."""
        self.step_scores = step_scores or {}

    def update_step_force_drop_ids(self, drop_ids):
        """Update step ids whose spans should be force-dropped during pruning."""
        try:
            self.step_force_drop_ids = set(int(x) for x in (drop_ids or []))
        except Exception:
            self.step_force_drop_ids = set()

    def update_step_scores_from_attention(self, attentions, query_token_count=0):
        """
        Build internal step scores from piggyback attentions.

        For each step span in the current prunable region, compute average
        attention mass from the current query tokens to that span and update
        step_scores with EMA smoothing.
        """
        if attentions is None or not len(attentions):
            return
        if self.pruning_mode != "step_inter":
            return
        if self.current_cache_len <= 0:
            return

        trajectory_len = self.get_trajectory_len()
        if trajectory_len <= 0:
            return

        prune_start = self.protected_prefix_len if self.protect_prompt else 0
        obs_window = min(self.observation_window, trajectory_len)
        prune_end = self.current_cache_len - obs_window
        if prune_end <= prune_start:
            return

        local_spans = self._build_step_aware_local_spans(prune_start, prune_end)
        if not local_spans:
            return

        # Use the same layer selection policy as H2O scoring.
        num_layers = len(attentions)
        num_score_layers = int(getattr(self.pruning_strategy, "num_score_layers", 3))
        selected_layers = attentions[max(0, num_layers - num_score_layers):]
        if not selected_layers:
            return

        # query_len is attention dim-2; clamp by caller hint if provided.
        query_len = int(selected_layers[0].shape[-2])
        if query_token_count and query_token_count > 0:
            query_len = min(query_len, int(query_token_count))
        if query_len <= 0:
            return

        kv_len = int(selected_layers[0].shape[-1])
        ema = float(getattr(self, "step_inter_ema", 0.7))
        ema = max(0.0, min(0.99, ema))

        for sp in local_spans:
            if not isinstance(sp, dict):
                continue
            sid = sp.get("step_id", None)
            if sid is None:
                continue
            left = max(0, int(sp.get("start", -1)))
            right = min(kv_len - 1, int(sp.get("end", -1)))
            if right < left:
                continue

            layer_vals = []
            for attn in selected_layers:
                # attn shape: (batch, heads, query_len, kv_len)
                a = attn[0, :, -query_len:, left:right + 1]
                if a.numel() <= 0:
                    continue
                layer_vals.append(float(a.mean().item()))
            if not layer_vals:
                continue

            raw_score = float(sum(layer_vals) / len(layer_vals))
            prev = float(self.step_scores.get(sid, raw_score))
            self.step_scores[sid] = float(ema * prev + (1.0 - ema) * raw_score)

    def _build_step_anchor_protected_indices(self, prune_start, prune_end):
        """
        Build protected indices from recent observation spans for step-anchor mode.
        """
        if not self.step_spans:
            return []
        obs_spans = [s for s in self.step_spans if isinstance(s, dict) and s.get("type") == "obs"]
        if self.step_anchor_keep_last_obs > 0:
            obs_spans = obs_spans[-self.step_anchor_keep_last_obs:]
        protected = set()
        for span in obs_spans:
            s = int(span.get("start", -1))
            e = int(span.get("end", -1))
            if e < s:
                continue
            left = max(prune_start, s)
            right = min(prune_end - 1, e)
            if right >= left:
                protected.update(range(left, right + 1))
        return sorted(protected)

    def _build_step_aware_local_spans(self, prune_start, prune_end):
        """
        Map global-ID step spans to current local cache indices.
        This keeps step-aware pools aligned after pruning remaps cache positions.
        """
        if not self.step_spans:
            return []
        mapper = getattr(self.token_tracker, "global_id_mapper", None) if self.token_tracker is not None else None
        if not mapper:
            # Fallback: assume spans are already local.
            return self.step_spans

        spans_source = list(self.step_spans)
        local_spans = []
        for sp in spans_source:
            if not isinstance(sp, dict):
                continue
            if sp.get("type") not in ("step", "prompt"):
                continue
            g_start = int(sp.get("start", -1))
            g_end = int(sp.get("end", -1))
            if g_end < g_start:
                continue
            step_id = sp.get("step_id", None)
            orig_len = int(sp.get("orig_len", g_end - g_start + 1))

            matched_local = []
            for local_idx, gid in enumerate(mapper):
                if local_idx < prune_start or local_idx >= prune_end:
                    continue
                if g_start <= int(gid) <= g_end:
                    matched_local.append(local_idx)
            if not matched_local:
                continue

            run_start = matched_local[0]
            run_prev = matched_local[0]
            for idx in matched_local[1:]:
                if idx == run_prev + 1:
                    run_prev = idx
                    continue
                local_spans.append({
                    "type": sp.get("type", "step"),
                    "step_id": step_id,
                    "orig_len": orig_len,
                    "start": int(run_start),
                    "end": int(run_prev),
                })
                run_start = idx
                run_prev = idx
            local_spans.append({
                "type": sp.get("type", "step"),
                "step_id": step_id,
                "orig_len": orig_len,
                "start": int(run_start),
                "end": int(run_prev),
            })
        return local_spans

    def merge_step_kv(self, past_key_values, step_key_values, step_token_count, step_id=None):
        """
        Merge KV chunk produced for the current ReAct step into the global past_key_values.

        Args:
            past_key_values: current full KV cache (tuple of (k,v) per layer or other supported type)
            step_key_values: KV corresponding to the current step tokens (same format as past_key_values for the step portion)
            step_token_count: number of tokens in step_key_values (sequence length)
            step_id: optional identifier for this step (for bookkeeping)

        Returns:
            new_past_key_values, new_cache_len, info
            - new_past_key_values: fused KV cache (may be the same object if fusion skipped)
            - new_cache_len: updated total cache length after merging
            - info: dict with metadata (merged: bool, reason, step_id, added_tokens, new_cache_len)

        NOTE: 当前实现为简单的按序列维度 concat（占位实现）。如果使用 DynamicCache 类型或结构不兼容，
        本方法会返回原始 past_key_values 并在 info 中说明未合并的原因。
        """
        info = {"merged": False, "step_id": step_id, "added_tokens": 0, "new_cache_len": self.current_cache_len}

        if step_token_count <= 0 or step_key_values is None:
            info["reason"] = "empty_step"
            return past_key_values, self.current_cache_len, info

        # If no existing cache, step KV becomes the cache
        if past_key_values is None:
            # Accept the step KV as the whole cache
            self.current_cache_len = step_token_count
            self.step_segments.append({"id": step_id, "len": step_token_count})
            info.update({"merged": True, "added_tokens": step_token_count, "new_cache_len": self.current_cache_len})
            return step_key_values, self.current_cache_len, info

        # Handle tuple-of-(k,v) format by concatenating along sequence dim (dim=2)
        if isinstance(past_key_values, tuple) and isinstance(step_key_values, tuple):
            try:
                new_layers = []
                for (k_old, v_old), (k_step, v_step) in zip(past_key_values, step_key_values):
                    # expected shapes:
                    # k_old: (batch, num_heads, seq_old, head_dim)
                    # k_step: (batch, num_heads, seq_step, head_dim)
                    k_new = torch.cat([k_old, k_step], dim=2)
                    v_new = torch.cat([v_old, v_step], dim=2)
                    new_layers.append((k_new, v_new))

                new_kv = tuple(new_layers)
                new_len = self.current_cache_len + step_token_count
                self.current_cache_len = new_len
                self.step_segments.append({"id": step_id, "len": step_token_count})
                info.update({"merged": True, "added_tokens": step_token_count, "new_cache_len": new_len})
                return new_kv, new_len, info
            except Exception as e:
                info["reason"] = f"concat_failed: {e}"
                return past_key_values, self.current_cache_len, info

        # DynamicCache or unknown types: do not attempt merging here (placeholder)
        info["reason"] = "unsupported_cache_type"
        return past_key_values, self.current_cache_len, info

    def get_trajectory_len(self):
        """Get current trajectory region length."""
        if not self.protect_prompt:
            return self.current_cache_len
        return self.current_cache_len - self.protected_prefix_len

    def should_prune(self):
        """
        Check if pruning should be triggered.

        Triggers when:
        1. step_count is a multiple of prune_every_n, OR
        2. trajectory region exceeds max_trajectory_tokens budget
        """
        trajectory_len = self.get_trajectory_len()

        if trajectory_len <= 0:
            return False

        # Budget exceeded: must prune
        if trajectory_len > self.max_trajectory_tokens:
            return True

        # Periodic pruning
        if self.step_count > 0 and self.step_count % self.prune_every_n == 0:
            return True

        return False

    def should_prune(self):
        """
        Check if pruning should be triggered.

        Triggers when:
        1. step_count is a multiple of prune_every_n, OR
        2. trajectory region exceeds max_trajectory_tokens budget
        """
        trajectory_len = self.get_trajectory_len()

        if trajectory_len <= 0:
            return False

        # Budget exceeded: must prune
        if trajectory_len > self.max_trajectory_tokens:
            return True

        # Periodic pruning
        if self.step_count > 0 and self.step_count % self.prune_every_n == 0:
            return True

        return False

    def prune(self, past_key_values, attentions=None):
        """
        Execute pruning on the trajectory region of the KV cache.

        Args:
            past_key_values: full KV cache tuple
            attentions: attention weights (needed for h2o/h2o_snapkv modes)

        Returns:
            new_past_key_values: pruned KV cache
            new_cache_len: new total cache length
        """
        trajectory_len = self.get_trajectory_len()
        if trajectory_len <= 0:
            self.last_pruned = False
            return past_key_values, self.current_cache_len

        # Determine prunable region boundaries.
        # By default prompt is protected; can be disabled via protect_prompt=False.
        prune_start = self.protected_prefix_len if self.protect_prompt else 0

        # Protect the observation window (recent tokens)
        obs_window = min(self.observation_window, trajectory_len)
        prune_end = self.current_cache_len - obs_window

        if prune_end <= prune_start:
            # Not enough tokens to prune (all within observation window)
            self.last_pruned = False
            return past_key_values, self.current_cache_len

        # Execute pruning
        cache_before = self.current_cache_len  # Track cache size before pruning
        effective_keep_ratio = self.cache_ratio
        if self.pruning_mode in ("h2o", "tova", "streamingllm", "tokenskipping", "pyramidinfer", "step_aware_h2o", "step_inter"):
            # Ratio semantics should be based on logical token totals (stable across steps),
            # not recursively on already-pruned current cache.
            protected_suffix_len = max(0, self.current_cache_len - prune_end)
            protected_total = prune_start + protected_suffix_len
            prunable_len = max(1, prune_end - prune_start)
            ratio = float(self.cache_ratio)

            logical_total = None
            if self.token_tracker is not None and hasattr(self.token_tracker, "next_global_id"):
                try:
                    logical_total = int(self.token_tracker.next_global_id)
                except Exception:
                    logical_total = None
            if logical_total is None or logical_total <= 0:
                logical_total = int(cache_before)

            if self.protect_prompt:
                logical_generated = max(0, logical_total - self.protected_prefix_len)
                target_generated = int(logical_generated * ratio)
                if logical_generated > 0:
                    target_generated = max(1, target_generated)
                # Generated budget includes protected suffix window; allocate the remainder to prunable region.
                desired_prunable_kept = max(0, target_generated - protected_suffix_len)
                desired_prunable_kept = min(prunable_len, desired_prunable_kept)
                # If there is prunable region and non-empty generated trajectory, keep at least 1.
                if logical_generated > 0 and prunable_len > 0:
                    desired_prunable_kept = max(1, desired_prunable_kept)
            else:
                target_total = int(logical_total * ratio)
                target_total = max(target_total, protected_total)
                desired_prunable_kept = target_total - protected_total
                desired_prunable_kept = max(1, min(prunable_len, desired_prunable_kept))

            effective_keep_ratio = desired_prunable_kept / prunable_len

        protected_indices = None
        if self.pruning_mode == "step_anchor_h2o":
            protected_indices = self._build_step_anchor_protected_indices(prune_start, prune_end)
        step_spans = None
        if self.pruning_mode in ("step_aware_h2o", "step_inter"):
            step_spans = self._build_step_aware_local_spans(prune_start, prune_end)
        step_scores = self.step_scores if self.pruning_mode in ("step_aware_h2o", "step_inter") else None

        new_kv, new_total_len, info = self.pruning_strategy.prune(
            past_key_values=past_key_values,
            attentions=attentions,
            prune_start=prune_start,
            prune_end=prune_end,
            observation_window=obs_window,
            keep_ratio=effective_keep_ratio,
            protected_indices=protected_indices,
            step_spans=step_spans,
            step_scores=step_scores,
            step_force_drop_ids=self.step_force_drop_ids if self.step_force_drop_ids else None,
            step_alpha=self.step_aware_alpha,
            step_beta=self.step_aware_beta,
            step_min_keep=self.step_aware_min_keep,
            step_min_keep_ratio=self.step_aware_min_keep_ratio,
            step_bonus=self.step_aware_bonus,
            step_poolwise=self.step_poolwise_prune,
        )

        # Add cache_before to info for tracking
        info["cache_before"] = cache_before
        if self.token_tracker is not None and hasattr(self.token_tracker, "current_step"):
            info["react_step"] = self.token_tracker.current_step
        
        # Update state
        self.current_cache_len = new_total_len
        self.total_prune_count += 1
        self.last_pruned = True
        self.pruning_history.append(info)

        return new_kv, new_total_len

    def prune_one_token(self, past_key_values, attentions=None):
        """
        Prune exactly one token from the current prunable region (if possible).
        Useful for token-by-token budget control.
        """
        trajectory_len = self.get_trajectory_len()
        if trajectory_len <= 0:
            self.last_pruned = False
            return past_key_values, self.current_cache_len

        prune_start = self.protected_prefix_len if self.protect_prompt else 0
        obs_window = min(self.observation_window, trajectory_len)
        prune_end = self.current_cache_len - obs_window
        if prune_end <= prune_start:
            self.last_pruned = False
            return past_key_values, self.current_cache_len

        prunable_len = prune_end - prune_start
        if prunable_len <= 1:
            self.last_pruned = False
            return past_key_values, self.current_cache_len

        # Keep all but one token in prunable region.
        one_token_keep_ratio = (prunable_len - 1) / prunable_len
        cache_before = self.current_cache_len
        protected_indices = None
        if self.pruning_mode == "step_anchor_h2o":
            protected_indices = self._build_step_anchor_protected_indices(prune_start, prune_end)
        step_spans = None
        if self.pruning_mode in ("step_aware_h2o", "step_inter"):
            step_spans = self._build_step_aware_local_spans(prune_start, prune_end)
        step_scores = self.step_scores if self.pruning_mode in ("step_aware_h2o", "step_inter") else None

        new_kv, new_total_len, info = self.pruning_strategy.prune(
            past_key_values=past_key_values,
            attentions=attentions,
            prune_start=prune_start,
            prune_end=prune_end,
            observation_window=obs_window,
            keep_ratio=one_token_keep_ratio,
            protected_indices=protected_indices,
            step_spans=step_spans,
            step_scores=step_scores,
            step_force_drop_ids=self.step_force_drop_ids if self.step_force_drop_ids else None,
            step_alpha=self.step_aware_alpha,
            step_beta=self.step_aware_beta,
            step_min_keep=self.step_aware_min_keep,
            step_min_keep_ratio=self.step_aware_min_keep_ratio,
            step_bonus=self.step_aware_bonus,
            step_poolwise=self.step_poolwise_prune,
        )
        info["cache_before"] = cache_before
        info["single_token_mode"] = True
        if self.token_tracker is not None and hasattr(self.token_tracker, "current_step"):
            info["react_step"] = self.token_tracker.current_step
        self.current_cache_len = new_total_len
        self.total_prune_count += 1
        self.last_pruned = True
        self.pruning_history.append(info)
        return new_kv, new_total_len

    def get_stats(self):
        """Get summary statistics about cache management."""
        return {
            "total_prune_count": self.total_prune_count,
            "current_cache_len": self.current_cache_len,
            "protected_prefix_len": self.protected_prefix_len,
            "trajectory_len": self.get_trajectory_len(),
            "step_count": self.step_count,
            "pruning_history": self.pruning_history,
        }
