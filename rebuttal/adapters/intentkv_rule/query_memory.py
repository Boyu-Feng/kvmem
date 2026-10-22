"""IntentKV §3.2, equations (2)-(3), without the learned residual."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class QueryMemory:
    def __init__(self, decay=0.5):
        if decay != 0.5:
            raise ValueError("Appendix A fixes lambda=0.5")
        self.decay = decay
        self.rows = ()
        self.updates = 0

    @torch.no_grad()
    def update(self, encodings):
        encodings = tuple(encodings)
        if not encodings or (self.rows and len(encodings) != len(self.rows)):
            raise ValueError("Query encoding layer count changed")
        updated = []
        for index, encoding in enumerate(encodings):
            if encoding.ndim != 2 or not torch.isfinite(encoding).all():
                raise ValueError("Expected finite [query heads, head dimension] encoding")
            value = encoding.detach().float()
            if self.rows:
                previous = self.rows[index]
                if previous.shape != value.shape or previous.device != value.device:
                    raise ValueError("Query memory layout changed")
                value = math.exp(-self.decay) * previous + value
            updated.append(F.normalize(value, dim=-1))
        self.rows = tuple(updated)
        self.updates += 1

    @torch.no_grad()
    def scores(self, keys, candidates):
        """Candidate-only softmax, then sum every layer and expanded GQA head."""
        keys = tuple(keys)
        if not self.rows or len(keys) != len(self.rows):
            raise ValueError("Missing or mismatched QueryMemory")
        if not candidates:
            return torch.empty(0, device=self.rows[0].device)
        total = None
        for memory, key in zip(self.rows, keys):
            if key.ndim != 4 or key.shape[0] != 1 or key.shape[-1] != memory.shape[-1]:
                raise ValueError("Expected batch-one resident [1, KV heads, tokens, D] keys")
            heads = memory.shape[0]
            if heads % key.shape[1]:
                raise ValueError("Query heads must be divisible by KV heads")
            index = torch.tensor(candidates, device=key.device, dtype=torch.long)
            expanded = key[0].index_select(1, index).float().repeat_interleave(
                heads // key.shape[1], dim=0)
            logits = torch.einsum("hd,hnd->hn", memory, expanded) / math.sqrt(key.shape[-1])
            score = logits.softmax(dim=-1).sum(dim=0)
            if not torch.isfinite(score).all():
                raise ValueError("Nonfinite rule score")
            total = score if total is None else total + score.to(total.device)
        return total


def kept_positions(memory, keys, forced, target):
    """Equation (1), restricted to resident rows; forced set may exceed budget."""
    keys = tuple(keys)
    length = keys[0].shape[-2]
    forced = sorted(set(forced))
    if target < 0 or any(index < 0 or index >= length for index in forced):
        raise ValueError("Invalid target or forced position")
    if any(key.shape[-2] != length for key in keys):
        raise ValueError("Nonuniform resident cache")
    candidates = sorted(set(range(length)) - set(forced))
    quota = min(len(candidates), max(0, target - len(forced)))
    if quota == len(candidates):
        return list(range(length))
    if not quota:
        return forced
    scores = memory.scores(keys, candidates)
    # Stable physical-order ties are an explicit deterministic adaptation.
    chosen = torch.argsort(scores, descending=True, stable=True)[:quota].tolist()
    return sorted(forced + [candidates[index] for index in chosen])
