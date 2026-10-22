"""Single-category ThinKV TBE adaptation: representative KV pairs, no TBQ/CT."""
from __future__ import annotations

import math

import torch

from models.RebuttalLLM import RebuttalLLM

SEGMENT_TOKENS = 128
RETENTION = (64, 32, 16, 8, 4)
KMEANS_ITERATIONS = 10


@torch.inference_mode()
def cluster_representatives(keys, keep, iterations=KMEANS_ITERATIONS):
    """Lloyd clustering per native KV head, retaining distinct actual KV pairs."""
    if keys.ndim != 3 or not 1 <= keep <= keys.shape[1] or iterations < 1:
        raise ValueError("Expected group/token/channel keys and a valid cluster count")
    if not torch.isfinite(keys).all():
        raise ValueError("Cannot cluster nonfinite post-RoPE keys")
    groups, tokens, _ = keys.shape
    points = keys.float()
    seeds = torch.linspace(0, tokens - 1, keep, device=keys.device).round().long()
    centers = points[:, seeds].clone()
    prior = None
    for _ in range(iterations):
        distances = (points.square().sum(-1, keepdim=True)
                     + centers.square().sum(-1).unsqueeze(1)
                     - 2 * points @ centers.transpose(1, 2))
        assignments = distances.argmin(-1)
        if prior is not None and torch.equal(assignments, prior):
            break
        membership = torch.nn.functional.one_hot(assignments, keep).float()
        counts = membership.sum(1)
        means = membership.transpose(1, 2) @ points / counts.clamp_min(1).unsqueeze(-1)
        centers = torch.where(counts.unsqueeze(-1) > 0, means, centers)
        prior = assignments
    distances = (points.square().sum(-1, keepdim=True)
                 + centers.square().sum(-1).unsqueeze(1)
                 - 2 * points @ centers.transpose(1, 2)).transpose(1, 2)
    belongs = assignments.unsqueeze(1) == torch.arange(keep, device=keys.device)[None, :, None]
    occupied = belongs.any(-1)
    representatives = distances.masked_fill(~belongs, math.inf).argmin(-1)
    used = torch.zeros((groups, tokens), dtype=torch.long, device=keys.device)
    used.scatter_add_(1, representatives, occupied.long())
    # Identical keys can leave empty clusters; fill them with unclaimed tokens.
    for column in (~occupied).any(0).nonzero().flatten().tolist():
        replacement = distances[:, column].masked_fill(used.bool(), math.inf).argmin(-1)
        missing = ~occupied[:, column]
        representatives[:, column] = torch.where(missing, replacement, representatives[:, column])
        used.scatter_add_(1, replacement[:, None], missing.long()[:, None])
    return representatives.sort(-1).values


class SelectionState:
    def __init__(self):
        self.ids = None
        self.next_id = 0
        self.counts = []
        self.prompt_length = 0
        self.registered_end = 0

    def append(self, cache, start, count):
        if start != self.next_id or count < 1:
            raise RuntimeError("TBE logical IDs must append contiguously")
        layers = cache.layers
        heads = layers[0].keys.shape[1]
        if any(layer.keys.shape[0] != 1 or layer.keys.shape[1] != heads
               or layer.keys.shape[-2] != layers[0].keys.shape[-2] for layer in layers):
            raise RuntimeError("TBE requires batch one and uniform native KV-head lengths")
        new = torch.arange(start, start + count, device=layers[0].keys.device)
        new = new.expand(len(layers), heads, count)
        self.ids = new.clone() if self.ids is None else torch.cat((self.ids, new), -1)
        self.next_id += count
        if self.ids.shape[-1] != layers[0].keys.shape[-2]:
            raise RuntimeError("TBE IDs and physical KV lengths disagree")

    def register_prompt(self, length):
        if self.prompt_length not in (0, length) or not 0 < length <= self.next_id:
            raise RuntimeError("Protected prompt changed during the episode")
        self.prompt_length = length
        self.registered_end = max(self.registered_end, length)
        while self.registered_end < self.next_id:
            segment, extent = divmod(self.registered_end - length, SEGMENT_TOKENS)
            added = min(self.next_id - self.registered_end, SEGMENT_TOKENS - extent)
            if segment == len(self.counts):
                self.counts.append(0)
            if self.counts[segment] != extent:
                raise RuntimeError("Cannot append into an already compressed segment")
            self.counts[segment] += added
            self.registered_end += added

    def plan(self, cache, ratio, safe_end):
        if not 0 < ratio <= 1 or self.ids is None:
            raise ValueError("TBE requires initialized IDs and a positive token ratio")
        original = self.next_id - self.prompt_length
        target = self.prompt_length + math.floor(ratio * original)
        lengths = list(self.counts)
        live = self.ids.shape[-1]
        indices = torch.arange(live, device=self.ids.device).expand(*self.ids.shape[:2], live)
        events = []
        offset = self.prompt_length
        for segment, count in enumerate(lengths):
            end = self.prompt_length + (segment + 1) * SEGMENT_TOKENS
            while live > target and end <= safe_end and count > RETENTION[-1]:
                keep = next(level for level in RETENTION if level < count)
                blocks = []
                for layer_index, layer in enumerate(cache.layers):
                    slots = indices[layer_index, :, offset:offset + count]
                    blocks.append(layer.keys[0].gather(
                        1, slots.unsqueeze(-1).expand(-1, -1, layer.keys.shape[-1])))
                keys = torch.stack(blocks).flatten(0, 1)
                selected = cluster_representatives(keys, keep).reshape(*indices.shape[:2], keep)
                kept = indices[..., offset:offset + count].gather(-1, selected)
                indices = torch.cat((indices[..., :offset], kept, indices[..., offset + count:]), -1)
                events.append({"segment": segment, "from": count, "to": keep})
                live -= count - keep
                count = keep
            lengths[segment] = count
            offset += count
        if offset != self.ids.shape[-1] - sum(event["from"] - event["to"] for event in events):
            raise RuntimeError("TBE segment accounting does not cover the resident cache")
        floor = self.prompt_length + sum(
            RETENTION[-1] if self.prompt_length + (i + 1) * SEGMENT_TOKENS <= safe_end else count
            for i, count in enumerate(lengths))
        if live > max(target, floor):
            raise RuntimeError("TBE failed to satisfy a feasible segment-level budget")
        return indices, lengths, {"target_tokens": target, "floor_tokens": floor,
                                 "after_tokens": live, "budget_excess_tokens": max(0, live - target),
                                 "annealing": events}

    def materialize(self, cache, indices, lengths):
        for index, layer in enumerate(cache.layers):
            slots = indices[index].unsqueeze(0)
            layer.keys = layer.keys.gather(2, slots.unsqueeze(-1).expand(
                -1, -1, -1, layer.keys.shape[-1]))
            layer.values = layer.values.gather(2, slots.unsqueeze(-1).expand(
                -1, -1, -1, layer.values.shape[-1]))
        self.ids = self.ids.gather(-1, indices)
        self.counts = lengths

    def rewind(self, cache, end):
        counts = (self.ids < end).sum(-1)
        if not torch.equal(counts, counts.flatten()[0].expand_as(counts)):
            raise RuntimeError("Stop crop crossed a compressed segment differently across KV heads")
        count = int(counts.flatten()[0])
        cache.crop(count)
        self.ids = self.ids[..., :count]
        self.next_id = end
        self.registered_end = end
        segments = math.ceil((end - self.prompt_length) / SEGMENT_TOKENS)
        self.counts = self.counts[:segments]
        if segments and (end - self.prompt_length) % SEGMENT_TOKENS:
            self.counts[-1] = (end - self.prompt_length) % SEGMENT_TOKENS


class ThinkKVSelectionLLM(RebuttalLLM):
    RUNTIME_VERSION = "thinkkv-single-category-selection-only-bf16-v2"

    def __init__(self, model_path, enabled=True, ratio=.2):
        if not 0 < ratio <= 1:
            raise ValueError("Require a positive trajectory token retention ratio")
        self.enabled, self.ratio = enabled, ratio
        super().__init__(model_path, {"pruning_mode": "none", "pruning_schedule": "token",
                                      "attn_mode": "scoring_forward"})

    def reset(self):
        super().reset()
        self.selection = SelectionState()
        self._generation_start = None
        self._stop_strings = ()

    def _append(self, ids, timing_key="decode_time"):
        start = self.token_tracker.next_global_id
        logits = super()._append(ids, timing_key)
        self.selection.append(self.past_key_values, start, len(ids))
        return self._repetition_logits(logits) if self._generation_start is not None else logits

    def _sync(self):
        # Clean latency synchronizes only outer boundaries; the diagnostic adds phase syncs.
        return None

    @torch.inference_mode()
    def _repetition_logits(self, logits):
        from transformers import RepetitionPenaltyLogitsProcessor
        penalty = self.model.generation_config.repetition_penalty
        if not isinstance(penalty, (int, float)) or not math.isfinite(penalty) or penalty <= 0:
            raise ValueError("Invalid pinned repetition penalty")
        if penalty == 1:
            return logits
        start = 0 if self._initial_decode else self._generation_start
        if start is None:
            raise RuntimeError("Repetition history requested outside generation")
        history = self._global_token_id_log[start:self.token_tracker.next_global_id]
        if not history:
            return logits
        ids = torch.tensor([history], device=logits.device)
        return RepetitionPenaltyLogitsProcessor(float(penalty))(ids, logits.clone())

    def _safe_end(self):
        end = self.token_tracker.next_global_id
        if self._generation_start is None or not self._stop_strings:
            return end
        generated = self._global_token_id_log[self._generation_start:end]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        pending = max((n for stop in self._stop_strings for n in range(1, len(stop))
                       if text.endswith(stop[:n])), default=0)
        if not pending:
            return end
        cut = len(text) - pending
        keep = len(generated)
        while keep and len(self.tokenizer.decode(generated[:keep], skip_special_tokens=True)) > cut:
            keep -= 1
        return self._generation_start + keep

    def _maybe_prune(self, phase):
        self.selection.register_prompt(self.prompt_length)
        if not self.enabled:
            return
        before, before_bytes = self._record_cache()
        target = self.prompt_length + math.floor(
            self.ratio * (self.token_tracker.next_global_id - self.prompt_length))
        if before <= target:
            return
        indices, lengths, event = self.selection.plan(self.past_key_values, self.ratio, self._safe_end())
        if event["annealing"]:
            self.selection.materialize(self.past_key_values, indices, lengths)
            mapper = self.selection.ids[0, 0].tolist()
            self.token_tracker.global_id_mapper = mapper
            self.token_tracker.cache_length = len(mapper)
            self._all_token_ids = [self._global_token_id_log[i] for i in mapper]
        after, after_bytes = self._record_cache()
        self.cache_measurements.append({**event, "phase": phase,
                                       "step": self.token_tracker.current_step,
                                       "logical_tokens": self.token_tracker.next_global_id,
                                       "protected_tokens": self.prompt_length,
                                       "before_tokens": before, "before_bytes": before_bytes,
                                       "after_tokens": after, "after_bytes": after_bytes})

    def _rewind_to(self, logical_end):
        self.selection.rewind(self.past_key_values, logical_end)
        self.token_tracker.global_id_mapper = self.selection.ids[0, 0].tolist()
        self.token_tracker.next_global_id = logical_end
        self.token_tracker.cache_length = self.selection.ids.shape[-1]
        self._global_token_id_log = self._global_token_id_log[:logical_end]
        self._all_token_ids = [self._global_token_id_log[i] for i in self.token_tracker.global_id_mapper]
        self._record_cache()

    def _generate_block(self, logits, max_new_tokens, stop_strings):
        self._generation_start = self.token_tracker.next_global_id
        self._stop_strings = tuple(stop_strings or ())
        try:
            if self._initial_decode:
                logits = self._repetition_logits(logits)
            result = super()._generate_block(logits, max_new_tokens, stop_strings)
        finally:
            self._generation_start = None
            self._stop_strings = ()
        self._maybe_prune("completed_decode_block")
        return result

    def get_stats(self):
        return {**self.audit(), "timing_scope": "Use separate clean wall time and diagnostic GPU "
                "phases; inherited CPU-enqueue timing estimates are intentionally not reported."}

    def audit(self):
        tokens, size = self._record_cache()
        seen = self.token_tracker.next_global_id - self.prompt_length
        return {"runtime": self.RUNTIME_VERSION, "enabled": self.enabled,
                "quantization": False, "kv_dtype": str(self._layers()[0][0].dtype),
                "logical_trajectory_tokens": seen, "protected_prompt_tokens": self.prompt_length,
                "final_tokens_per_head": tokens, "final_kv_bytes": size,
                "actual_trajectory_ratio": (tokens - self.prompt_length) / seen if seen else None,
                "segment_counts": list(self.selection.counts),
                "native_kv_heads": self._layers()[0][0].shape[1],
                "selection_events": self.cache_measurements,
                "position_scope": "Absolute logical RoPE positions; IDs differ per native KV head/layer"}
