"""
Pruning Strategy
Unified pruning interface supporting multiple modes:
1. H2O-only: Score -> Evict low-scoring tokens (hard deletion)
2. PyramidInfer-style: H2O score + recency pyramid prior (hard deletion)
3. SnapKV-only: Pool non-observation-window tokens (no attention scores needed)
4. H2O + SnapKV: Score -> Keep heavy hitters as-is, Pool evicted tokens

Reference: H2O (Zhang et al., 2023), SnapKV (Li et al., 2024)
"""

import math
import torch
from transformers import DynamicCache
from .h2o_scorer import H2OScorer
from .ours import OurCompressor  # 确保已导入用户的融合器类


def _build_token_score_snapshot(
    h2o_scorer,
    attentions,
    prune_start,
    prune_end,
    hh_scores,
    display_scores,
    step_scores_per_token=None,
    combined_scores=None,
    alpha=None,
    beta=None,
):
    snap = {
        "prune_start": int(prune_start),
        "prune_end": int(prune_end),
        "hh_scores": hh_scores.detach().cpu().tolist(),
        "display_scores": display_scores.detach().cpu().tolist(),
    }
    if step_scores_per_token is not None:
        snap["step_scores"] = step_scores_per_token.detach().cpu().tolist()
    if combined_scores is not None:
        snap["combined_scores"] = combined_scores.detach().cpu().tolist()
        snap["display_scores"] = combined_scores.detach().cpu().tolist()
    if alpha is not None:
        snap["alpha"] = float(alpha)
    if beta is not None:
        snap["beta"] = float(beta)
    if attentions is not None:
        try:
            attn_mat = h2o_scorer.compute_attention_matrix(attentions)
            q_len, kv_len = int(attn_mat.shape[0]), int(attn_mat.shape[1])
            snap["attention_matrix"] = attn_mat.detach().cpu().tolist()
            snap["query_base"] = int(kv_len - q_len)
            snap["query_len"] = int(q_len)
            snap["kv_len"] = int(kv_len)
        except Exception:
            pass
    return snap


class PruningStrategy:
    """
    Unified pruning interface for KV cache compression.

    Modes:
        "h2o":        Use attention scores to identify heavy hitters; hard-delete the rest.
        "snapkv":     Pool all non-observation-window tokens; no attention scores needed.
        "h2o_snapkv": Use attention scores to keep heavy hitters intact;
                      pool the evicted (low-score) tokens instead of deleting them.
        "ours":       Our proposed method that integrates scoring, pooling, and optional step KV fusion.
    """

    def __init__(
        self,
        mode="ours",
        num_score_layers=3,
        pool_window=4,
        memory_rank=128,
        token_tracker=None,
        pyramid_h2o_weight=0.7,
        pyramid_recency_power=1.5,
    ):
        """
        Args:
            mode: one of "h2o", "pyramidinfer", "snapkv", "h2o_snapkv", "ours"
            num_score_layers: layers used for H2O scoring (default: 3)
            pool_window: SnapKV pooling window size (default: 4)
            memory_rank: fixed rank for Memory base (used in "ours" mode)
            token_tracker: optional TokenTracker instance for tracking pruned tokens
        """
        # 支持用户自定义的 "ours" 模式（融合/压缩占位实现）
        assert mode in ("h2o", "tova", "streamingllm", "tokenskipping", "pyramidinfer", "step_anchor_h2o", "step_aware_h2o", "step_inter", "snapkv", "h2o_snapkv", "ours"), \
            f"Unknown pruning mode: {mode}. Must be one of: h2o, tova, streamingllm, tokenskipping, pyramidinfer, step_anchor_h2o, step_aware_h2o, step_inter, snapkv, h2o_snapkv, ours"

        self.mode = mode
        self.h2o_scorer = H2OScorer(num_score_layers=num_score_layers)
        self.memory_rank =  memory_rank # 仅在 "ours" 模式下使用，表示 Memory 基底的固定秩
        self.token_tracker = token_tracker
        self.pyramid_h2o_weight = float(max(0.0, min(1.0, pyramid_h2o_weight)))
        self.pyramid_recency_power = float(max(0.25, pyramid_recency_power))

    def prune(self, past_key_values, attentions, prune_start, prune_end, new_step_kv=None,step_token_count=0, keep_ratio=0.5,
              observation_window=128,is_initial=False, protected_indices=None, step_spans=None, step_scores=None,
              step_force_drop_ids=None,
              step_alpha=0.7, step_beta=0.3, step_min_keep=5, step_bonus=0.0, step_min_keep_ratio=0.0, step_poolwise=False):
        """
        Apply the selected pruning strategy on a trajectory segment within KV cache.

        The KV cache layout is:
            [0 ... prune_start-1] = protected region (sink + question)
            [prune_start ... prune_end-1] = trajectory region (subject to pruning)
            [prune_end ... total_len-1] = observation window (recent tokens, protected)

        Args:
            past_key_values: tuple of (num_layers,) tuples, each (key, value)
                             key/value shape: (batch, num_heads, seq_len, head_dim)
            attentions: tuple of attention tensors (needed for h2o/h2o_snapkv modes).
                        Can be None for snapkv mode.
            prune_start: start index of the prunable trajectory region
            prune_end: end index of the prunable trajectory region (exclusive)
            observation_window: number of recent tokens that are never pruned
            keep_ratio: for H2O, fraction of heavy hitters to keep

        Returns:
            new_past_key_values: pruned KV cache (same format as input)
            new_total_len: new total sequence length after pruning
            pruning_info: dict with stats about the pruning operation
        """
        total_len = self._get_cache_len(past_key_values)
        # Defensive boundary clamp: keep pruning range consistent with current KV length.
        # This avoids invalid arange/index ranges when external counters drift.
        prune_start = int(max(0, min(int(prune_start), int(total_len))))
        prune_end = int(max(prune_start, min(int(prune_end), int(total_len))))
        if prune_end <= prune_start:
            return past_key_values, total_len, {"pruned": False, "reason": "empty_or_invalid_range"}

        num_layers = self._get_num_layers(past_key_values)

        # Protected prefix: [0, prune_start)
        # Prunable region: [prune_start, prune_end)
        # Protected suffix (observation window): [prune_end, total_len)
        prunable_len = prune_end - prune_start

        if prunable_len <= 0:
            return past_key_values, total_len, {"pruned": False}

        if self.mode == "h2o":
            return self._prune_h2o(
                past_key_values, attentions,
                prune_start, prune_end, total_len,
                keep_ratio, num_layers
            )
        elif self.mode == "tova":
            return self._prune_tova(
                past_key_values, attentions,
                prune_start, prune_end, total_len,
                keep_ratio, num_layers
            )
        elif self.mode == "streamingllm":
            return self._prune_streamingllm(
                past_key_values,
                prune_start, prune_end, total_len,
                keep_ratio, num_layers
            )
        elif self.mode == "tokenskipping":
            return self._prune_tokenskipping(
                past_key_values,
                prune_start, prune_end, total_len,
                keep_ratio, num_layers
            )
        elif self.mode == "pyramidinfer":
            return self._prune_pyramidinfer(
                past_key_values, attentions,
                prune_start, prune_end, total_len,
                keep_ratio, num_layers
            )
        elif self.mode == "step_anchor_h2o":
            return self._prune_step_anchor_h2o(
                past_key_values, attentions,
                prune_start, prune_end, total_len,
                keep_ratio, num_layers,
                protected_indices=protected_indices,
            )
        elif self.mode in ("step_aware_h2o", "step_inter"):
            return self._prune_step_aware_h2o(
                past_key_values, attentions,
                prune_start, prune_end, total_len,
                keep_ratio, num_layers,
                step_spans=step_spans,
                step_scores=step_scores,
                force_drop_step_ids=step_force_drop_ids,
                alpha=step_alpha,
                beta=step_beta,
                min_keep_per_step=step_min_keep,
                bonus=step_bonus,
                min_keep_ratio=step_min_keep_ratio,
                poolwise=step_poolwise,
            )
        elif self.mode == "snapkv":
            return self._prune_snapkv(
                past_key_values,
                prune_start, prune_end, total_len,
                num_layers
            )
        elif self.mode == "h2o_snapkv":
            return self._prune_h2o_snapkv(
                past_key_values, attentions,
                prune_start, prune_end, total_len,
                keep_ratio, num_layers
            )
        elif self.mode == "ours":
            return self._prune_ours(
                past_key_values, attentions,
                prune_start, prune_end, new_step_kv, total_len,
                keep_ratio, num_layers,window_size=observation_window,is_initial=is_initial
            )

    @staticmethod
    def _get_cache_len(past_key_values):
        """Get the sequence length from past_key_values (works with both tuple and DynamicCache)."""
        if isinstance(past_key_values, DynamicCache):
            return past_key_values.layers[0].keys.shape[2]
        else:
            k, v = past_key_values[0]
            return k.shape[2]

    @staticmethod
    def _get_kv(past_key_values, layer_idx):
        """Get (key, value) tensors for a given layer from past_key_values."""
        if isinstance(past_key_values, DynamicCache):
            return past_key_values.layers[layer_idx].keys, past_key_values.layers[layer_idx].values
        else:
            return past_key_values[layer_idx]

    @staticmethod
    def _get_device(past_key_values):
        """Get the device of the cache tensors."""
        if isinstance(past_key_values, DynamicCache):
            return past_key_values.layers[0].device
        else:
            return past_key_values[0][0].device

    @staticmethod
    def _get_num_layers(past_key_values):
        """Get the number of layers in past_key_values."""
        if isinstance(past_key_values, DynamicCache):
            return len(past_key_values.layers)
        else:
            return len(past_key_values)

    @staticmethod
    def _build_cache(new_kv_list):
        """Build a DynamicCache from a list of (key, value) tuples."""
        cache = DynamicCache()
        for layer_idx, (k, v) in enumerate(new_kv_list):
            cache.update(k, v, layer_idx)
        return cache

    def _prune_ours(self, past_key_values, attentions,
                    prune_start, prune_end, new_step_kv=None, total_len=None,
                    new_step_token_count=0,
                    keep_ratio=None, window_size=512, num_layers=None,
                    is_initial=False # 初始传入为 False
                    ):
        
        if window_size is None:
            window_size = 512

        # memory_rank 是你预设的固定记忆矩阵长度（如 128）
        memory_rank = self.memory_rank
        device = self._get_device(past_key_values)
        
        layers_data = []
        
        for layer_idx in range(num_layers):
            k, v = self._get_kv(past_key_values, layer_idx)
            
            # --- 【变量 A：静态前缀 (Prompt)】 ---
            # 绝对不动，保持模型对指令的遵循能力
            prefix_k = k[:, :, :prune_start, :]
            prefix_v = v[:, :, :prune_start, :]

            # --- 【变量 B：长期记忆块 (Memory Block M)】 ---
            if not is_initial:
                # 情况 1：第一次压缩初始化
                # 此时：[Prompt] + [第一次输出的内容] + [新 Step] + [Window]
                # 我们要把“第一次输出的内容”池化成 M
                # 范围计算：从 Prompt 结束到 (窗口前 - 新 Step 前)
                hist_end = total_len - window_size - new_step_token_count
                
                if hist_end > prune_start:
                    prunable_indices = torch.arange(prune_start, hist_end, device=device)
                    # 利用池化把第一次生成的 Thought/Action 压成固定秩 r
                    memory_m_k, memory_m_v = self.snapkv_pooler.pool_region(k, v, prunable_indices)
                else:
                    # 容错：如果第一次输出太短，直接截取原始段作为初始 M
                    memory_m_k = k[:, :, prune_start:hist_end, :]
                    memory_m_v = v[:, :, prune_start:hist_end, :]
                
                layer_initialized_status = False # 标记此层刚完成初始化，待融合
            else:
                # 情况 2：增量更新模式
                # 此时 M 已经在固定位置：[prune_start : prune_start + memory_rank]
                m_start, m_end = prune_start, prune_start + memory_rank
                memory_m_k = k[:, :, m_start:m_end, :]
                memory_m_v = v[:, :, m_start:m_end, :]
                
                layer_initialized_status = True

            # --- 【变量 C：新步进信号 (Step Signal S)】 ---
            # 这是本次 Incremental Step 产生的 Observation
            if new_step_kv is not None:
                new_s_k, new_s_v = new_step_kv[layer_idx]
            else:
                # 如果外部没传，从 Window 之前精准切出这一步的长度
                s_end = total_len - window_size
                s_start = max(0, s_end - new_step_token_count)
                new_s_k = k[:, :, s_start:s_end, :].detach()
                new_s_v = v[:, :, s_start:s_end, :].detach()

            # --- 【变量 D：原始滑动窗口 (Window W)】 ---
            # 保持 100% 精度，不做任何处理
            window_k = k[:, :, -window_size:, :]
            window_v = v[:, :, -window_size:, :]
            
            # 打包本层数据，准备喂给融合函数
            layers_data.append({
                "prefix": (prefix_k, prefix_v),
                "memory_m": (memory_m_k, memory_m_v), # 长期记忆 M (旧)
                "new_step_s": (new_s_k, new_s_v),     # 本次增量 S
                "window": (window_k, window_v),       # 滑动窗口 W
                "is_initialized": layer_initialized_status
            })

        # --- 【核心：融合与更新】 ---
        # 调用融合函数：M_new = f(M_old, S)
        # 你可以在这个函数里写 Delta Rule 或者简单的 Concat
        updated_layers = self._fuse_memory_and_signal(layers_data)

        # --- 【重新拼接 (Reconstruct)】 ---
        reconstructed_layers = []
        for data in updated_layers:
            pk, pv = data["prefix"]
            mk, mv = data["memory_m"] # 融合后的新记忆 M_new
            wk, wv = data["window"]
            
            # 布局：[Prompt] + [Memory Block] + [Sliding Window]
            merged_k = torch.cat([pk, mk, wk], dim=2)
            merged_v = torch.cat([pv, mv, wv], dim=2)
            reconstructed_layers.append((merged_k, merged_v))

        # 构建新的 Cache 对象
        new_kv = self._build_cache(reconstructed_layers)
        
        # 逻辑总长度：前缀长 + 记忆秩 + 窗口长
        new_total_len = prune_start + memory_rank + window_size

        # 告知外部：初始化已完成，下次请设为 True
        return new_kv, new_total_len, {"is_initial": True, "note": "Memory Initialized and Fused"}


    def _fuse_memory_and_signal(self, layers_data):
        """
        简单的记忆融合函数（用于跑通架构）。
        逻辑：如果已初始化，则将新信号 S 池化到 memory_rank 长度后与 M 相加。
        """
        updated_layers = []

        for layer_idx, data in enumerate(layers_data):
            # 提取变量
            (m_k, m_v) = data["memory_m"]      # 长期记忆 [B, H, rank, D]
            (s_k, s_v) = data["new_step_s"]    # 新步进信号 [B, H, step_len, D]
            is_init = data["is_initialized"]   # 是否已经建立过记忆

            # 如果是第一次初始化 (is_init 为 False)，memory_m 已经是池化好的初始基底
            # 我们直接使用它，不做额外融合
            if not is_init:
                new_m_k, new_m_v = m_k, m_v
            else:
                # 如果是增量更新 (is_init 为 True)
                # 为了跑通逻辑，我们将新信号 S 也池化成 memory_rank 的长度
                if s_k is not None and s_k.size(2) > 0:
                    # 简单的池化对齐：将 S 压缩到与 M 相同的 rank
                    # 这里的 pool_tensor 是你 SnapKV 里的基础操作
                    s_k_pooled = self.snapkv_pooler.pool_tensor(s_k, self.memory_rank)
                    s_v_pooled = self.snapkv_pooler.pool_tensor(s_v, self.memory_rank)
                    
                    # 执行最简单的融合：M_new = M_old + S_pooled
                    new_m_k = m_k + s_k_pooled
                    new_m_v = m_v + s_v_pooled
                else:
                    # 如果没有新信号，保持原样
                    new_m_k, new_m_v = m_k, m_v

            # 更新本层数据
            data["memory_m"] = (new_m_k, new_m_v)
            updated_layers.append(data)

        return updated_layers

    def _prune_h2o(self, past_key_values, attentions,
                   prune_start, prune_end, total_len,
                   keep_ratio, num_layers):
        """
        H2O-only: compute scores over entire prunable region, keep heavy hitters, hard-delete the rest.
        
        KEY CHANGE: H2O should score the ENTIRE prunable region [prune_start, prune_end),
        not just the most recent tokens. This is the core H2O algorithm: evaluate all
        tokens and keep those with highest attention scores globally.
        """
        # ================================================================
        # H2O scores the entire prunable region for importance evaluation
        # ================================================================
        scores = self.h2o_scorer.compute_scores(attentions, prune_start, prune_end)
        heavy_indices, evicted_indices = self.h2o_scorer.select_heavy_hitters(scores, keep_ratio)
        
        # Convert relative indices (within prunable region) to absolute indices (in full cache)
        abs_heavy = heavy_indices + prune_start
        
        # Build final index: [prefix] + [selected heavy hitters] + [suffix]
        # prefix: [0, prune_start) - protected prefix (system prompt, question)
        # selected: chosen heavy hitter tokens from [prune_start, prune_end)
        # suffix: [prune_end, total_len) - observation window (recent tokens, protected)
        prefix_indices = torch.arange(prune_start, device=scores.device)
        suffix_indices = torch.arange(prune_end, total_len, device=scores.device)
        keep_indices = torch.cat([prefix_indices, abs_heavy, suffix_indices])

        # Reconstruct KV cache with selected tokens
        new_kv = []
        for layer_idx in range(num_layers):
            k, v = self._get_kv(past_key_values, layer_idx)
            new_k = k[:, :, keep_indices, :]
            new_v = v[:, :, keep_indices, :]
            new_kv.append((new_k, new_v))
        new_kv = self._build_cache(new_kv)

        new_total_len = keep_indices.shape[0]
        
        # Convert keep_indices to a CPU list for token tracker
        kept_indices_list = keep_indices.cpu().tolist() if hasattr(keep_indices, 'cpu') else list(keep_indices)
        
        # Record to token tracker if available
        if self.token_tracker is not None:
            try:
                self.token_tracker.record_pruning_with_kept_indices(
                    step=None,  # Step number will be set by caller
                    kept_local_indices=kept_indices_list,
                    old_cache_length=total_len,
                    prune_start=prune_start,
                    prune_end=prune_end,
                )
            except Exception as e:
                # Token tracking should never break pruning.
                print(f"[WARN] Token tracking failed during H2O prune: {e}")
        
        info = {
            "pruned": True,
            "mode": "h2o",
            "prunable_region_size": prune_end - prune_start,
            "heavy_hitters_kept": len(heavy_indices),
            "tokens_evicted": len(evicted_indices),
            "compression_ratio": len(heavy_indices) / max(1, prune_end - prune_start),
            "new_total_len": new_total_len,
            # Absolute cache positions evicted in this pruning event.
            # Stored for downstream visualization/analysis.
            "evicted_abs_indices": (evicted_indices + prune_start).detach().cpu().tolist(),
            "token_score_snapshot": _build_token_score_snapshot(
                self.h2o_scorer,
                attentions,
                prune_start,
                prune_end,
                scores,
                scores,
            ),
        }
        return new_kv, new_total_len, info

    def _prune_pyramidinfer(self, past_key_values, attentions,
                            prune_start, prune_end, total_len,
                            keep_ratio, num_layers):
        """
        PyramidInfer-style baseline:
        fuse H2O attention score with a recency pyramid prior, then keep top-k.

        - h2o_score: token importance from attentions
        - pyramid_prior: favors recent tokens using a power-law curve
        - final_score = w * norm(h2o_score) + (1-w) * pyramid_prior
        """
        raw_scores = self.h2o_scorer.compute_scores(attentions, prune_start, prune_end)
        n = int(raw_scores.shape[0])
        if n <= 0:
            return past_key_values, total_len, {"pruned": False, "mode": "pyramidinfer", "reason": "empty_prunable"}

        # Normalize H2O score to [0, 1] for stable fusion across samples.
        s_min = torch.min(raw_scores)
        s_max = torch.max(raw_scores)
        denom = (s_max - s_min).clamp(min=1e-8)
        h2o_norm = (raw_scores - s_min) / denom

        # Recency prior: old->0, recent->1, with power shaping.
        if n == 1:
            recency = torch.ones(1, device=raw_scores.device, dtype=raw_scores.dtype)
        else:
            pos = torch.arange(n, device=raw_scores.device, dtype=raw_scores.dtype)
            recency = pos / float(n - 1)
        recency = torch.pow(recency, self.pyramid_recency_power)

        w = float(self.pyramid_h2o_weight)
        fused_scores = w * h2o_norm + (1.0 - w) * recency

        heavy_indices, evicted_indices = self.h2o_scorer.select_heavy_hitters(fused_scores, keep_ratio)
        abs_heavy = heavy_indices + prune_start

        prefix_indices = torch.arange(prune_start, device=raw_scores.device)
        suffix_indices = torch.arange(prune_end, total_len, device=raw_scores.device)
        keep_indices = torch.cat([prefix_indices, abs_heavy, suffix_indices])

        new_kv = []
        for layer_idx in range(num_layers):
            k, v = self._get_kv(past_key_values, layer_idx)
            new_k = k[:, :, keep_indices, :]
            new_v = v[:, :, keep_indices, :]
            new_kv.append((new_k, new_v))
        new_kv = self._build_cache(new_kv)

        new_total_len = keep_indices.shape[0]
        kept_indices_list = keep_indices.cpu().tolist() if hasattr(keep_indices, "cpu") else list(keep_indices)
        if self.token_tracker is not None:
            try:
                self.token_tracker.record_pruning_with_kept_indices(
                    step=None,
                    kept_local_indices=kept_indices_list,
                    old_cache_length=total_len,
                    prune_start=prune_start,
                    prune_end=prune_end,
                )
            except Exception as e:
                print(f"[WARN] Token tracking failed during PyramidInfer prune: {e}")

        info = {
            "pruned": True,
            "mode": "pyramidinfer",
            "prunable_region_size": int(prune_end - prune_start),
            "heavy_hitters_kept": int(len(heavy_indices)),
            "tokens_evicted": int(len(evicted_indices)),
            "compression_ratio": float(len(heavy_indices) / max(1, prune_end - prune_start)),
            "new_total_len": int(new_total_len),
            "evicted_abs_indices": (evicted_indices + prune_start).detach().cpu().tolist(),
            "pyramid_h2o_weight": float(self.pyramid_h2o_weight),
            "pyramid_recency_power": float(self.pyramid_recency_power),
        }
        return new_kv, new_total_len, info

    def _prune_tova(self, past_key_values, attentions,
                    prune_start, prune_end, total_len,
                    keep_ratio, num_layers):
        """
        TOVA-style baseline:
        score KV tokens using the latest query-token attention (last layer, head-mean),
        then keep top-k tokens in the prunable region.
        """
        if attentions is None or len(attentions) == 0:
            # Fallback to H2O scorer when attentions are unavailable.
            return self._prune_h2o(
                past_key_values, attentions,
                prune_start, prune_end, total_len,
                keep_ratio, num_layers
            )

        last_layer_attn = attentions[-1]  # (B, H, Q, KV)
        if last_layer_attn is None or last_layer_attn.numel() == 0:
            return self._prune_h2o(
                past_key_values, attentions,
                prune_start, prune_end, total_len,
                keep_ratio, num_layers
            )

        kv_len = int(last_layer_attn.shape[-1])
        left = max(0, int(prune_start))
        right = min(int(prune_end), kv_len)
        if right <= left:
            return past_key_values, total_len, {"pruned": False, "mode": "tova", "reason": "empty_prunable"}

        # Mean over heads, use latest query row to mimic token omission via current attention.
        # Shape after mean/select: (prunable_len,)
        head_mean = last_layer_attn[0].mean(dim=0)  # (Q, KV)
        latest_query_scores = head_mean[-1, left:right]
        if latest_query_scores.numel() <= 0:
            return past_key_values, total_len, {"pruned": False, "mode": "tova", "reason": "no_scores"}

        heavy_indices, evicted_indices = self.h2o_scorer.select_heavy_hitters(latest_query_scores, keep_ratio)
        abs_heavy = heavy_indices + left

        prefix_indices = torch.arange(prune_start, device=latest_query_scores.device)
        suffix_indices = torch.arange(prune_end, total_len, device=latest_query_scores.device)
        keep_indices = torch.cat([prefix_indices, abs_heavy, suffix_indices])

        new_kv = []
        for layer_idx in range(num_layers):
            k, v = self._get_kv(past_key_values, layer_idx)
            new_k = k[:, :, keep_indices, :]
            new_v = v[:, :, keep_indices, :]
            new_kv.append((new_k, new_v))
        new_kv = self._build_cache(new_kv)

        new_total_len = keep_indices.shape[0]
        kept_indices_list = keep_indices.cpu().tolist() if hasattr(keep_indices, "cpu") else list(keep_indices)
        if self.token_tracker is not None:
            try:
                self.token_tracker.record_pruning_with_kept_indices(
                    step=None,
                    kept_local_indices=kept_indices_list,
                    old_cache_length=total_len,
                    prune_start=prune_start,
                    prune_end=prune_end,
                )
            except Exception as e:
                print(f"[WARN] Token tracking failed during TOVA prune: {e}")

        info = {
            "pruned": True,
            "mode": "tova",
            "prunable_region_size": int(prune_end - prune_start),
            "heavy_hitters_kept": int(len(heavy_indices)),
            "tokens_evicted": int(len(evicted_indices)),
            "compression_ratio": float(len(heavy_indices) / max(1, prune_end - prune_start)),
            "new_total_len": int(new_total_len),
            "evicted_abs_indices": (evicted_indices + left).detach().cpu().tolist(),
            "token_score_snapshot": _build_token_score_snapshot(
                self.h2o_scorer,
                attentions,
                left,
                right,
                latest_query_scores,
                latest_query_scores,
            ),
        }
        return new_kv, new_total_len, info

    def _prune_streamingllm(self, past_key_values,
                            prune_start, prune_end, total_len,
                            keep_ratio, num_layers):
        """
        StreamingLLM-style baseline: keep sink tokens in the protected prefix and
        retain only the most recent tokens inside the prunable trajectory region.
        No attention scores are used.
        """
        prunable_len = prune_end - prune_start
        if prunable_len <= 0:
            return past_key_values, total_len, {"pruned": False, "mode": "streamingllm", "reason": "empty_prunable"}

        num_keep = max(1, int(prunable_len * keep_ratio))
        if num_keep >= prunable_len:
            return past_key_values, total_len, {
                "pruned": False,
                "mode": "streamingllm",
                "reason": "no_compression_needed",
            }

        keep_start = prune_end - num_keep
        device = self._get_device(past_key_values)
        prefix_indices = torch.arange(prune_start, device=device)
        abs_keep = torch.arange(keep_start, prune_end, device=device)
        suffix_indices = torch.arange(prune_end, total_len, device=device)
        keep_indices = torch.cat([prefix_indices, abs_keep, suffix_indices])

        new_kv = []
        for layer_idx in range(num_layers):
            k, v = self._get_kv(past_key_values, layer_idx)
            new_k = k[:, :, keep_indices, :]
            new_v = v[:, :, keep_indices, :]
            new_kv.append((new_k, new_v))
        new_kv = self._build_cache(new_kv)

        new_total_len = keep_indices.shape[0]
        kept_indices_list = keep_indices.cpu().tolist() if hasattr(keep_indices, "cpu") else list(keep_indices)
        if self.token_tracker is not None:
            try:
                self.token_tracker.record_pruning_with_kept_indices(
                    step=None,
                    kept_local_indices=kept_indices_list,
                    old_cache_length=total_len,
                    prune_start=prune_start,
                    prune_end=prune_end,
                )
            except Exception as e:
                print(f"[WARN] Token tracking failed during StreamingLLM prune: {e}")

        evicted_abs = torch.arange(prune_start, keep_start, device=device)
        info = {
            "pruned": True,
            "mode": "streamingllm",
            "prunable_region_size": int(prunable_len),
            "heavy_hitters_kept": int(num_keep),
            "tokens_evicted": int(prunable_len - num_keep),
            "compression_ratio": float(num_keep / max(1, prunable_len)),
            "new_total_len": int(new_total_len),
            "evicted_abs_indices": evicted_abs.detach().cpu().tolist(),
        }
        return new_kv, new_total_len, info

    def _compute_key_norm_scores(self, past_key_values, prune_start, prune_end, num_layers):
        """TokenSkipping: L2 norm of key vectors, averaged over last score layers."""
        start_layer = max(0, num_layers - self.h2o_scorer.num_score_layers)
        layer_scores = []
        for layer_idx in range(start_layer, num_layers):
            k, _ = self._get_kv(past_key_values, layer_idx)
            region = k[:, :, prune_start:prune_end, :]
            # Mean over heads -> L2 norm per token position.
            per_head_norm = torch.linalg.vector_norm(region, dim=-1)  # (B, H, T)
            layer_scores.append(per_head_norm[0].mean(dim=0))
        if not layer_scores:
            device = self._get_device(past_key_values)
            return torch.zeros(max(0, prune_end - prune_start), device=device)
        return torch.stack(layer_scores, dim=0).mean(dim=0)

    def _prune_tokenskipping(self, past_key_values,
                             prune_start, prune_end, total_len,
                             keep_ratio, num_layers):
        """
        TokenSkipping baseline (Hongthai & Chuangsuwanich, ICIT 2025):
        retain high L2-norm key tokens under a fixed budget; anchor/prefix
        and observation suffix stay protected by the manager.
        """
        prunable_len = prune_end - prune_start
        if prunable_len <= 0:
            return past_key_values, total_len, {"pruned": False, "mode": "tokenskipping", "reason": "empty_prunable"}

        scores = self._compute_key_norm_scores(past_key_values, prune_start, prune_end, num_layers)
        if scores.numel() <= 0:
            return past_key_values, total_len, {"pruned": False, "mode": "tokenskipping", "reason": "no_scores"}

        heavy_indices, evicted_indices = self.h2o_scorer.select_heavy_hitters(scores, keep_ratio)
        abs_heavy = heavy_indices + prune_start

        device = scores.device
        prefix_indices = torch.arange(prune_start, device=device)
        suffix_indices = torch.arange(prune_end, total_len, device=device)
        keep_indices = torch.cat([prefix_indices, abs_heavy, suffix_indices])

        new_kv = []
        for layer_idx in range(num_layers):
            k, v = self._get_kv(past_key_values, layer_idx)
            new_k = k[:, :, keep_indices, :]
            new_v = v[:, :, keep_indices, :]
            new_kv.append((new_k, new_v))
        new_kv = self._build_cache(new_kv)

        new_total_len = keep_indices.shape[0]
        kept_indices_list = keep_indices.cpu().tolist() if hasattr(keep_indices, "cpu") else list(keep_indices)
        if self.token_tracker is not None:
            try:
                self.token_tracker.record_pruning_with_kept_indices(
                    step=None,
                    kept_local_indices=kept_indices_list,
                    old_cache_length=total_len,
                    prune_start=prune_start,
                    prune_end=prune_end,
                )
            except Exception as e:
                print(f"[WARN] Token tracking failed during TokenSkipping prune: {e}")

        info = {
            "pruned": True,
            "mode": "tokenskipping",
            "prunable_region_size": int(prunable_len),
            "heavy_hitters_kept": int(len(heavy_indices)),
            "tokens_evicted": int(len(evicted_indices)),
            "compression_ratio": float(len(heavy_indices) / max(1, prunable_len)),
            "new_total_len": int(new_total_len),
            "evicted_abs_indices": (evicted_indices + prune_start).detach().cpu().tolist(),
            "token_score_snapshot": {
                "prune_start": int(prune_start),
                "prune_end": int(prune_end),
                "display_scores": scores.detach().cpu().tolist(),
            },
        }
        return new_kv, new_total_len, info

    def _prune_step_anchor_h2o(self, past_key_values, attentions,
                               prune_start, prune_end, total_len,
                               keep_ratio, num_layers, protected_indices=None):
        """
        Step-Anchor H2O:
        protect selected step spans (e.g., recent observations) inside the
        prunable region, then apply H2O heavy-hitter selection to the rest.
        """
        device = self._get_device(past_key_values)
        protected_set = set()
        if protected_indices is not None:
            protected_set = set(int(i) for i in protected_indices if 0 <= int(i) < total_len)

        # Candidate indices in prunable region excluding protected anchors.
        candidate_abs = [i for i in range(prune_start, prune_end) if i not in protected_set]
        if len(candidate_abs) == 0:
            return past_key_values, total_len, {"pruned": False, "mode": "step_anchor_h2o", "reason": "all_anchor"}

        scores = self.h2o_scorer.compute_scores(attentions, prune_start, prune_end)
        # Relative protected indices within [prune_start, prune_end)
        rel_mask = torch.ones(scores.shape[0], dtype=torch.bool, device=device)
        if protected_set:
            rel_protected = [i - prune_start for i in protected_set if prune_start <= i < prune_end]
            if rel_protected:
                rel_protected_t = torch.tensor(rel_protected, dtype=torch.long, device=device)
                rel_mask[rel_protected_t] = False

        candidate_rel = torch.arange(scores.shape[0], device=device)[rel_mask]
        candidate_scores = scores[rel_mask]
        if candidate_scores.numel() == 0:
            return past_key_values, total_len, {"pruned": False, "mode": "step_anchor_h2o", "reason": "no_candidate"}

        heavy_rel, evicted_rel = self.h2o_scorer.select_heavy_hitters(candidate_scores, keep_ratio)
        kept_abs_from_candidates = candidate_rel[heavy_rel] + prune_start
        evicted_from_candidates = int(evicted_rel.numel())

        prefix_indices = torch.arange(prune_start, device=device)
        suffix_indices = torch.arange(prune_end, total_len, device=device)
        protected_abs = torch.tensor(sorted([i for i in protected_set if prune_start <= i < prune_end]),
                                     dtype=torch.long, device=device) if protected_set else torch.empty(0, dtype=torch.long, device=device)

        keep_indices = torch.cat([prefix_indices, protected_abs, kept_abs_from_candidates, suffix_indices])
        keep_indices = torch.unique(keep_indices, sorted=True)

        new_kv = []
        for layer_idx in range(num_layers):
            k, v = self._get_kv(past_key_values, layer_idx)
            new_k = k[:, :, keep_indices, :]
            new_v = v[:, :, keep_indices, :]
            new_kv.append((new_k, new_v))
        new_kv = self._build_cache(new_kv)
        new_total_len = int(keep_indices.shape[0])

        kept_indices_list = keep_indices.cpu().tolist() if hasattr(keep_indices, "cpu") else list(keep_indices)
        if self.token_tracker is not None:
            try:
                self.token_tracker.record_pruning_with_kept_indices(
                    step=None,
                    kept_local_indices=kept_indices_list,
                    old_cache_length=total_len,
                    prune_start=prune_start,
                    prune_end=prune_end,
                )
            except Exception as e:
                print(f"[WARN] Token tracking failed during Step-Anchor H2O prune: {e}")

        info = {
            "pruned": True,
            "mode": "step_anchor_h2o",
            "prunable_region_size": prune_end - prune_start,
            "protected_anchor_count": int(protected_abs.numel()),
            "tokens_evicted": evicted_from_candidates,
            "new_total_len": new_total_len,
        }
        return new_kv, new_total_len, info

    def _prune_step_aware_h2o(self, past_key_values, attentions,
                              prune_start, prune_end, total_len, keep_ratio, num_layers,
                              step_spans=None, step_scores=None, force_drop_step_ids=None,
                              alpha=0.7, beta=0.3, min_keep_per_step=5, bonus=0.0,
                              min_keep_ratio=0.0, poolwise=False):
        """
        Step-aware H2O (two-stage selection):
        1) token-level HH scores
        2) combined score = alpha * HH + beta * StepScore(step)
        3) Stage-1: per-step pool retention floor (ratio/min-keep)
        4) Stage-2: global fill to budget B
        """
        device = self._get_device(past_key_values)
        scores = self.h2o_scorer.compute_scores(attentions, prune_start, prune_end)
        n = int(scores.shape[0])
        if n <= 0:
            return past_key_values, total_len, {"pruned": False, "mode": self.mode, "reason": "empty_prunable"}

        # Budget B over prunable region
        B = max(1, min(n, int(n * float(keep_ratio))))
        combined = alpha * scores
        step_score_per_token = torch.zeros_like(scores)

        spans = step_spans or []
        forced_ids = set()
        if force_drop_step_ids:
            try:
                forced_ids = set(int(x) for x in force_drop_step_ids)
            except Exception:
                forced_ids = set()
        step_groups = {}
        external_step_score_count = 0
        for i, sp in enumerate(spans):
            if not isinstance(sp, dict):
                continue
            s = int(sp.get("start", -1))
            e = int(sp.get("end", -1))
            if e < s:
                continue
            left = max(prune_start, s)
            right = min(prune_end - 1, e)
            if right < left:
                continue
            rs = left - prune_start
            re = right - prune_start
            sid = sp.get("step_id", i)
            grp = step_groups.setdefault(sid, {
                "indices": set(),
                "step_id": sid,
                "orig_len": int(sp.get("orig_len", re - rs + 1)),
                "type": sp.get("type", "step"),
            })
            grp["orig_len"] = max(int(grp.get("orig_len", 0)), int(sp.get("orig_len", re - rs + 1)))
            grp["indices"].update(range(rs, re + 1))

        for grp in step_groups.values():
            idx_list = sorted(list(grp["indices"]))
            if not idx_list:
                continue
            idx_t = torch.tensor(idx_list, dtype=torch.long, device=device)
            sid = grp.get("step_id", None)
            if step_scores is not None and sid is not None and sid in step_scores:
                step_val = float(step_scores[sid])
                step_score_per_token[idx_t] = step_val
                external_step_score_count += 1
            else:
                step_val = torch.mean(scores[idx_t]).item()
                step_score_per_token[idx_t] = float(step_val)

        combined = combined + beta * step_score_per_token
        forbidden_rel = set()
        if forced_ids:
            for sid, grp in step_groups.items():
                try:
                    sid_int = int(sid)
                except Exception:
                    continue
                if sid_int in forced_ids:
                    forbidden_rel.update(int(i) for i in grp.get("indices", []))

        # Force-drop ablation (no backfill):
        # The dropped step's tokens must truly vanish from the cache without
        # being compensated by Stage-2 backfill. We shrink the effective budget
        # by exactly the number of would-be-kept tokens (top-B) that fall in the
        # dropped step, so the cache genuinely loses that step's contribution.
        # Physically removing these positions is equivalent to a hard attention
        # mask: remaining cached keys keep their RoPE rotation, so no position
        # shift occurs for the surviving tokens.
        effective_budget = min(n, B)
        if forbidden_rel:
            topB_pos = torch.topk(combined, effective_budget, dim=0).indices.tolist()
            drop_in_budget = len(set(int(i) for i in topB_pos) & forbidden_rel)
            effective_budget = max(0, effective_budget - int(drop_in_budget))

        if bonus > 0 and step_groups:
            for grp in step_groups.values():
                idx_list = sorted(list(grp["indices"]))
                if not idx_list:
                    continue
                combined[idx_list[0]] = combined[idx_list[0]] + bonus
                combined[idx_list[-1]] = combined[idx_list[-1]] + bonus

        # Stage-1:
        # - poolwise=True: per-span independent pre-selection
        #   keep 20% of each CURRENT span (user-requested behavior).
        # - poolwise=False: keep original floor-based behavior.
        requested_B = B
        m = max(0, int(min_keep_per_step))
        ratio = max(0.0, float(min_keep_ratio))
        keep_rel_set = set()
        rel_to_pool = {}
        pool_stats = []
        stage1_span_ratio = 0.10
        if poolwise and step_groups:
            for grp in step_groups.values():
                idx_list = sorted(list(grp["indices"]))
                if not idx_list:
                    continue
                sid = grp.get("step_id", None)
                try:
                    sid_int = int(sid) if sid is not None else None
                except Exception:
                    sid_int = None
                if sid_int is not None and sid_int in forced_ids:
                    continue
                pool_name = "prompt" if sid == "prompt" else (f"step{sid}" if sid is not None else "unknown")
                seg_len_current = len(idx_list)
                seg_orig_len = max(seg_len_current, int(grp.get("orig_len", seg_len_current)))
                for ridx in idx_list:
                    rel_to_pool[int(ridx)] = pool_name
                # Stage1 fixed-ratio keep on current span.
                k = int(math.ceil(float(seg_len_current) * stage1_span_ratio))
                k = max(1, min(seg_len_current, k))
                idx_t = torch.tensor(idx_list, dtype=torch.long, device=device)
                seg_scores = combined[idx_t]
                _, top_pos = torch.topk(seg_scores, k, dim=0)
                keep_idx = idx_t[top_pos].tolist()
                for ridx in keep_idx:
                    keep_rel_set.add(int(ridx))
                pool_stats.append({
                    "pool": pool_name,
                    "orig_len": int(seg_orig_len),
                    "current_len": int(seg_len_current),
                    "kept_stage1": int(k),
                    "kept_stage2": 0,
                    "kept_final": int(k),
                })
        else:
            for grp in step_groups.values():
                idx_list = sorted(list(grp["indices"]))
                if not idx_list:
                    continue
                sid = grp.get("step_id", None)
                try:
                    sid_int = int(sid) if sid is not None else None
                except Exception:
                    sid_int = None
                if sid_int is not None and sid_int in forced_ids:
                    continue
                seg_len_current = len(idx_list)
                seg_orig_len = max(seg_len_current, int(grp.get("orig_len", seg_len_current)))
                ratio_floor = int(math.ceil(seg_orig_len * ratio)) if ratio > 0.0 else 0
                base_quota = int(math.ceil(seg_orig_len * float(keep_ratio))) if poolwise else 0
                k = min(seg_len_current, max(m, ratio_floor, base_quota))
                if k <= 0:
                    continue
                idx_t = torch.tensor(idx_list, dtype=torch.long, device=device)
                seg_scores = combined[idx_t]
                _, top_pos = torch.topk(seg_scores, k, dim=0)
                keep_idx = idx_t[top_pos].tolist()
                for ridx in keep_idx:
                    keep_rel_set.add(int(ridx))
                pool_name = "prompt" if sid == "prompt" else (f"step{sid}" if sid is not None else "unknown")
                for ridx in idx_list:
                    rel_to_pool[int(ridx)] = pool_name
                pool_stats.append({
                    "pool": pool_name,
                    "orig_len": int(seg_orig_len),
                    "current_len": int(seg_len_current),
                    "kept_stage1": int(len(keep_idx)),
                    "kept_stage2": 0,
                    "kept_final": int(len(keep_idx)),
                })

        # Hard budget closure:
        # keep current step-aware logic, but never exceed global budget B.
        # If Stage-1 already exceeds B, trim Stage-1 keeps by combined score.
        all_rank = torch.argsort(combined, descending=True)
        keep_rel = sorted(list(keep_rel_set))
        stage1_overflow_trimmed = 0
        if len(keep_rel) > effective_budget:
            keep_rel_t = torch.tensor(keep_rel, dtype=torch.long, device=device)
            keep_scores = combined[keep_rel_t]
            _, top_pos = torch.topk(keep_scores, k=effective_budget, dim=0)
            keep_rel = sorted(keep_rel_t[top_pos].tolist())
            stage1_overflow_trimmed = int(len(keep_rel_set) - len(keep_rel))
            keep_rel_set = set(int(i) for i in keep_rel)

        # Stage-2: global fill to (possibly drop-reduced) budget.
        # effective_B equals effective_budget, which is shrunk by the dropped
        # step's would-be-kept tokens so the cache is not refilled to compensate.
        effective_B = effective_budget
        if len(keep_rel) < effective_B:
            keep_rel_lookup = set(keep_rel)
            for idx in all_rank.tolist():
                idx = int(idx)
                if idx in keep_rel_lookup:
                    continue
                if idx in forbidden_rel:
                    continue
                keep_rel.append(idx)
                keep_rel_lookup.add(idx)
                if len(keep_rel) >= effective_B:
                    break

        stage1_total = int(len(keep_rel_set))
        final_set = set(int(i) for i in keep_rel)
        stage2_total = max(0, int(len(final_set) - stage1_total))

        # Per-pool stage-2/final accounting to inspect cross-pool competition.
        pool_index = {ps.get("pool", f"pool{idx}"): idx for idx, ps in enumerate(pool_stats)}
        final_count_by_pool = {}
        for ridx in final_set:
            pname = rel_to_pool.get(int(ridx), "unpooled")
            final_count_by_pool[pname] = int(final_count_by_pool.get(pname, 0) + 1)
        for pname, fcnt in final_count_by_pool.items():
            if pname not in pool_index:
                pool_stats.append({
                    "pool": pname,
                    "orig_len": int(fcnt),
                    "current_len": int(fcnt),
                    "kept_stage1": 0,
                    "kept_stage2": int(fcnt),
                    "kept_final": int(fcnt),
                })
                continue
            ps = pool_stats[pool_index[pname]]
            s1 = int(ps.get("kept_stage1", 0))
            ps["kept_final"] = int(fcnt)
            ps["kept_stage2"] = max(0, int(fcnt - s1))

        keep_rel_t = torch.tensor(sorted(keep_rel), dtype=torch.long, device=device)
        abs_keep = keep_rel_t + prune_start

        prefix_indices = torch.arange(prune_start, device=device)
        suffix_indices = torch.arange(prune_end, total_len, device=device)
        keep_indices = torch.cat([prefix_indices, abs_keep, suffix_indices])

        new_kv = []
        for layer_idx in range(num_layers):
            k, v = self._get_kv(past_key_values, layer_idx)
            new_k = k[:, :, keep_indices, :]
            new_v = v[:, :, keep_indices, :]
            new_kv.append((new_k, new_v))
        new_kv = self._build_cache(new_kv)
        new_total_len = int(keep_indices.shape[0])

        kept_indices_list = keep_indices.cpu().tolist() if hasattr(keep_indices, "cpu") else list(keep_indices)
        if self.token_tracker is not None:
            try:
                self.token_tracker.record_pruning_with_kept_indices(
                    step=None,
                    kept_local_indices=kept_indices_list,
                    old_cache_length=total_len,
                    prune_start=prune_start,
                    prune_end=prune_end,
                )
            except Exception as e:
                print(f"[WARN] Token tracking failed during Step-aware H2O prune: {e}")

        info = {
            "pruned": True,
            "mode": self.mode,
            "prunable_region_size": n,
            "poolwise": bool(poolwise),
            "step_span_count": len(step_groups),
            "step_external_score_count": int(external_step_score_count),
            "forced_drop_step_ids": sorted(int(x) for x in forced_ids) if forced_ids else [],
            "forbidden_rel_tokens": int(len(forbidden_rel)),
            "force_drop_no_backfill": bool(forbidden_rel),
            "budget_B": requested_B,
            "budget_B_effective": effective_B,
            "budget_reduced_by": int(max(0, min(n, requested_B) - effective_B)),
            "stage1_overflow_trimmed": int(stage1_overflow_trimmed),
            "stage1_kept_total": int(stage1_total),
            "stage2_added_total": int(stage2_total),
            "step_min_keep": m,
            "step_min_keep_ratio": float(ratio),
            "stage1_span_ratio": float(stage1_span_ratio),
            "pool_target_ratio": float(keep_ratio),
            "pool_stats": pool_stats,
            "tokens_evicted": max(0, n - len(keep_rel)),
            "new_total_len": new_total_len,
            "alpha": float(alpha),
            "beta": float(beta),
            "bonus": float(bonus),
            "token_score_snapshot": _build_token_score_snapshot(
                self.h2o_scorer,
                attentions,
                prune_start,
                prune_end,
                scores,
                combined,
                step_scores_per_token=step_score_per_token,
                combined_scores=combined,
                alpha=float(alpha),
                beta=float(beta),
            ),
        }
        return new_kv, new_total_len, info

    def _prune_snapkv(self, past_key_values,
                      prune_start, prune_end, total_len,
                      num_layers):
        """
        SnapKV-only: pool all tokens in the prunable region.
        No attention scores needed.
        """
        prunable_len = prune_end - prune_start
        prunable_indices = torch.arange(prune_start, prune_end,
                                        device=self._get_device(past_key_values))

        new_kv = []
        for layer_idx in range(num_layers):
            k, v = self._get_kv(past_key_values, layer_idx)
            # Pool the prunable region
            pooled_k, pooled_v = self.snapkv_pooler.pool_region(k, v, prunable_indices)
            # Reconstruct: [prefix] + [pooled] + [suffix]
            prefix_k = k[:, :, :prune_start, :]
            suffix_k = k[:, :, prune_end:, :]
            prefix_v = v[:, :, :prune_start, :]
            suffix_v = v[:, :, prune_end:, :]

            new_k = torch.cat([prefix_k, pooled_k, suffix_k], dim=2)
            new_v = torch.cat([prefix_v, pooled_v, suffix_v], dim=2)
            new_kv.append((new_k, new_v))
        new_kv = self._build_cache(new_kv)

        num_pooled = self.snapkv_pooler.get_num_pooled_tokens(prunable_len)
        new_total_len = prune_start + num_pooled + (total_len - prune_end)
        info = {
            "pruned": True,
            "mode": "snapkv",
            "original_prunable": prunable_len,
            "pooled_to": num_pooled,
            "new_total_len": new_total_len,
        }
        return new_kv, new_total_len, info

    def _prune_h2o_snapkv(self, past_key_values, attentions,
                          prune_start, prune_end, total_len,
                          keep_ratio, num_layers):
        """
        H2O + SnapKV: score tokens, keep heavy hitters as-is, pool the evicted ones.
        """
        # Compute scores for the prunable region
        scores = self.h2o_scorer.compute_scores(attentions, prune_start, prune_end)
        heavy_indices, evicted_indices = self.h2o_scorer.select_heavy_hitters(scores, keep_ratio)

        # Convert to absolute indices
        abs_heavy = heavy_indices + prune_start
        abs_evicted = evicted_indices + prune_start

        new_kv = []
        for layer_idx in range(num_layers):
            k, v = self._get_kv(past_key_values, layer_idx)
            # Keep heavy hitters as-is
            heavy_k = k[:, :, abs_heavy, :]
            heavy_v = v[:, :, abs_heavy, :]

            # Pool evicted tokens
            if len(abs_evicted) > 0:
                pooled_k, pooled_v = self.snapkv_pooler.pool_region(k, v, abs_evicted)
            else:
                pooled_k = k[:, :, :0, :]
                pooled_v = v[:, :, :0, :]

            # Reconstruct: [prefix] + [pooled evicted] + [heavy hitters] + [suffix]
            prefix_k = k[:, :, :prune_start, :]
            suffix_k = k[:, :, prune_end:, :]
            prefix_v = v[:, :, :prune_start, :]
            suffix_v = v[:, :, prune_end:, :]

            new_k = torch.cat([prefix_k, pooled_k, heavy_k, suffix_k], dim=2)
            new_v = torch.cat([prefix_v, pooled_v, heavy_v, suffix_v], dim=2)
            new_kv.append((new_k, new_v))
        new_kv = self._build_cache(new_kv)

        num_pooled = self.snapkv_pooler.get_num_pooled_tokens(len(evicted_indices))
        new_total_len = prune_start + num_pooled + len(heavy_indices) + (total_len - prune_end)
        info = {
            "pruned": True,
            "mode": "h2o_snapkv",
            "original_prunable": prune_end - prune_start,
            "kept_heavy_hitters": len(heavy_indices),
            "evicted": len(evicted_indices),
            "pooled_to": num_pooled,
            "new_total_len": new_total_len,
        }
        return new_kv, new_total_len, info
