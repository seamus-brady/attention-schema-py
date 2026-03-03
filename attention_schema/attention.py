from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from .tokenizer import tokenize, token_overlap


@dataclass
class AttentionTarget:
    """A single item in the competitive attention pool with activation level."""

    content: str
    activation: float
    source: str


class AttentionMechanism:
    """Scores and selects context items based on relevance to the current query."""

    def __init__(
        self,
        use_recency_weight: bool = True,
        lateral_inhibition: float = 0.3,
    ) -> None:
        self.use_recency_weight = use_recency_weight
        self.lateral_inhibition = lateral_inhibition
        self.activation_pool: list[AttentionTarget] = []

    def score_relevance(
        self, items: list[str], query: str, weights: Optional[list[float]] = None
    ) -> list[float]:
        """Return a relevance score in [0, 1] for each *item* against *query*."""
        query_vec = _term_freq(query)
        if not query_vec:
            return [0.0] * len(items)

        scores: list[float] = []
        for i, item in enumerate(items):
            item_vec = _term_freq(item)
            relevance = _cosine(query_vec, item_vec)

            if weights and i < len(weights):
                relevance *= weights[i]

            scores.append(relevance)
        return scores

    def compete(
        self,
        items: list[str],
        query: str,
        focus_bias: str | None = None,
    ) -> list[AttentionTarget]:
        """Score items and apply winner-take-most lateral inhibition.

        The top scorer gets a boost (x1.3, capped at 1.0) and runners-up
        are suppressed (x0.7). The strength of suppression is controlled
        by ``lateral_inhibition``.
        """
        if not items:
            self.activation_pool = []
            return []

        weights = None
        if self.use_recency_weight and len(items) > 1:
            n = len(items)
            weights = [0.2 + 0.8 * (i / (n - 1)) for i in range(n)]

        scores = self.score_relevance(items, query, weights)

        # Apply focus bias
        if focus_bias:
            for i, item in enumerate(items):
                overlap = token_overlap(item, focus_bias)
                if overlap > 0:
                    scores[i] += 0.2 * overlap

        if not any(s > 0 for s in scores):
            self.activation_pool = [
                AttentionTarget(content=item, activation=0.0, source="competition")
                for item in items
            ]
            return list(self.activation_pool)

        # Winner-take-most: find the top scorer
        max_score = max(scores)
        li = self.lateral_inhibition

        targets: list[AttentionTarget] = []
        for item, score in zip(items, scores):
            if score == max_score and max_score > 0:
                # Winner gets a boost
                activation = min(score * (1.0 + li), 1.0)
            else:
                # Runners-up get suppressed
                activation = score * (1.0 - li)
            targets.append(
                AttentionTarget(content=item, activation=activation, source="competition")
            )

        self.activation_pool = targets
        return list(targets)

    def get_ground_truth(self) -> list[AttentionTarget]:
        """Return the current activation pool (ground truth of what mechanism attends to)."""
        return list(self.activation_pool)

    def select(
        self,
        items: list[str],
        query: str,
        top_k: int = 3,
        use_recency: bool = True,
        focus_bias: str | None = None,
    ) -> list[str]:
        """Return the *top_k* most relevant items (preserving original order among ties).

        Internally uses competitive attention via ``compete()``.
        Return type unchanged (list[str]) for backward compatibility.
        """
        # Temporarily override recency setting if caller disables it
        orig_recency = self.use_recency_weight
        if not use_recency:
            self.use_recency_weight = False

        targets = self.compete(items, query, focus_bias=focus_bias)

        self.use_recency_weight = orig_recency

        # Rank by activation, pick top-k, preserve original order
        ranked = sorted(
            enumerate(targets), key=lambda pair: pair[1].activation, reverse=True
        )
        indices = [i for i, _ in ranked[:top_k]]
        indices.sort()  # preserve original order
        return [items[i] for i in indices]


# ------------------------------------------------------------------
# Lightweight TF helpers
# ------------------------------------------------------------------

def _term_freq(text: str) -> Counter:
    tokens = tokenize(text)
    total = len(tokens) or 1
    return Counter({t: c / total for t, c in Counter(tokens).items()})


def _cosine(a: Counter, b: Counter) -> float:
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[k] * b[k] for k in common)
    mag_a = math.sqrt(sum(v * v for v in a.values()))
    mag_b = math.sqrt(sum(v * v for v in b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)
