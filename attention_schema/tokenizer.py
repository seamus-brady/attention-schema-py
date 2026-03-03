"""Shared tokenization and token-overlap utilities."""

from __future__ import annotations

import re


def tokenize(text: str) -> list[str]:
    """Lowercase split, stripping punctuation."""
    return re.findall(r"[a-z0-9]+", text.lower())


def token_overlap(a: str, b: str) -> float:
    """Jaccard-like token overlap in [0, 1]."""
    tokens_a = set(tokenize(a))
    tokens_b = set(tokenize(b))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
