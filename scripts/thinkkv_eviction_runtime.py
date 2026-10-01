"""ThinKV single-category segment eviction adapted to QA, without quantization.

The protected prompt and all retained trajectory KV stay in the model's native
dtype. Completed 128-token segments are progressively halved to a four-token
floor when the nominal trajectory budget is exceeded. Token representatives
are chosen by K-means on post-RoPE keys, with one index set shared by layers.
"""
from __future__ import annotations

import time

import torch
from transformers import DynamicCache

from models.RebuttalLLM import RebuttalLLM


SEGMENT = 128
MIN_RETAIN = 4


def kmeans_medoids(keys, count, iterations=4):
    """Choose distinct existing tokens nearest to deterministic K-means centers."""
    n = keys.shape[0]
    if count >= n:
        return list(range(n))
    data = keys.float()
    centers = data[torch.linspace(0, n - 1, count, device=keys.device).round().long()].clone()
    for _ in range(iterations):
        distance = torch.cdist(data, centers)
        labels = distance.argmin(dim=1)
        for cluster in range(count):
            assigned = data[labels == cluster]
            if assigned.numel():
                centers[cluster] = assigned.mean(dim=0)
    distance = torch.cdist(data, centers)
    selected = []
    used = set()
    for cluster in range(count):
        for candidate in distance[:, cluster].argsort().tolist():
            if candidate not in used:
                selected.append(candidate)
                used.add(candidate)
                break
    return sorted(selected)


class ThinKVEvictionOnlyLLM(RebuttalLLM):
    RUNTIME_VERSION = "thinkkv_eviction_only_qwen_v1"

    def __init__(self, model_path, ratio):
        if ratio not in (.2, .5):
            raise ValueError("Registered ratios are 0.2 and 0.5")
        self.ratio = ratio
        super().__init__(model_path, dict(pruning_mode="none", attn_mode="scoring_forward"))

    def reset(self):
        super().reset()
        self.eviction_events = []
        self.eviction_seconds = 0.

    @torch.inference_mode()
    def _append(self, ids, timing_key="decode_time"):
        logits = super()._append(ids, timing_key)
        # The initial prefill becomes the immutable prompt after generate_first
        # sets prompt_length. All subsequent appends may cross a segment edge.
        if self.prompt_length:
            self._enforce_budget()
        return logits

    @torch.inference_mode()
    def _enforce_budget(self):
        logical_generated = max(0, self.token_tracker.next_global_id - self.prompt_length)
        target = self.prompt_length + max(1, int(logical_generated * self.ratio))
        if self.current_cache_len <= target or not logical_generated:
            return
        active_segment = (logical_generated - 1) // SEGMENT
        while self.current_cache_len > target:
            mapper = self.token_tracker.global_id_mapper
            groups = {}
            for index, gid in enumerate(mapper):
                if gid >= self.prompt_length:
                    groups.setdefault((gid - self.prompt_length) // SEGMENT, []).append(index)
            eligible = [(segment, positions) for segment, positions in sorted(groups.items())
                        if segment < active_segment and len(positions) > MIN_RETAIN]
            if not eligible:
                return  # The active segment and four-token floors are protected.

            begin = time.perf_counter()
            segment, positions = eligible[0]
            length = len(positions)
            retain = max(MIN_RETAIN, 1 << (length.bit_length() - 1))
            if retain == length:
                retain //= 2
            retain = max(MIN_RETAIN, retain)
            before, before_bytes = self._record_cache()
            keys = self.past_key_values.layers[-1].keys[0, :, positions, :].float().mean(dim=0)
            selected = {positions[i] for i in kmeans_medoids(keys, retain)}
            keep = [i for i in range(before) if i not in positions or i in selected]
            replacement = DynamicCache(config=self.model.config)
            for index, layer in enumerate(self.past_key_values.layers):
                indices = torch.tensor(keep, device=layer.keys.device, dtype=torch.long)
                replacement.update(layer.keys.index_select(2, indices),
                                   layer.values.index_select(2, indices), index)
            self.past_key_values = replacement
            self.token_tracker.global_id_mapper = [mapper[i] for i in keep]
            self.token_tracker.cache_length = len(keep)
            self._all_token_ids = [self._global_token_id_log[i] for i in self.token_tracker.global_id_mapper]
            after, after_bytes = self._record_cache()
            self._sync()
            elapsed = time.perf_counter() - begin
            self.eviction_seconds += elapsed
            self.eviction_events.append(dict(segment=segment, before_tokens=before,
                after_tokens=after, target_tokens=target,
                logical_tokens=self.token_tracker.next_global_id,
                before_bytes=before_bytes, after_bytes=after_bytes,
                selected_tokens=retain, seconds=elapsed,
                policy="post-RoPE K-means medoid, shared token indices across layers"))

    def thin_audit(self):
        retained, bytes_used = self._record_cache()
        logical = self.token_tracker.next_global_id
        return dict(runtime=self.RUNTIME_VERSION, ratio=self.ratio,
                    quantization="none", segment_tokens=SEGMENT,
                    minimum_tokens_per_completed_segment=MIN_RETAIN,
                    prompt_tokens=self.prompt_length, logical_tokens=logical,
                    retained_tokens=retained,
                    trajectory_logical=max(0, logical - self.prompt_length),
                    trajectory_retained=max(0, retained - self.prompt_length),
                    resident_bytes=bytes_used,
                    peak_resident_bytes=self.peak_cache_bytes,
                    eviction_seconds=self.eviction_seconds,
                    evictions=self.eviction_events)
