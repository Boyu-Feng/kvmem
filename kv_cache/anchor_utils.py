"""Anchor term extraction for StepKV step scores (novelty / citation)."""

from __future__ import annotations

import re

ANCHOR_STOPWORDS = {
    "thought", "action", "observation", "search", "lookup", "find", "information",
    "invalid", "could", "would", "should", "about", "which", "what", "where",
    "when", "from", "with", "this", "that", "then", "than", "into", "have",
    "has", "been", "were", "also", "there", "their", "them", "they", "because",
    "movie", "film", "series", "director", "actor", "american", "british",
    "the", "and", "for", "was", "are", "who", "after", "before", "had", "not",
    "but", "finish",
}

_ANCHOR_PATTERN = re.compile(r"[a-z][a-z0-9_-]{3,}|\d+")
_ANCHOR_MAX = 64


def extract_anchor_terms(text: str) -> set[str]:
    """Extract anchors in text order: all numbers kept; words need len > 3 and not stopword."""
    if not text:
        return set()
    lowered = text.lower()
    ordered: list[str] = []
    seen: set[str] = set()

    for m in _ANCHOR_PATTERN.finditer(lowered):
        tok = m.group(0)
        if tok.isdigit():
            pass  # keep every numeric sequence
        else:
            if tok in ANCHOR_STOPWORDS or len(tok) <= 3:
                continue
        if tok not in seen:
            seen.add(tok)
            ordered.append(tok)

    return set(ordered[:_ANCHOR_MAX])
