"""Compute aggregate metrics from judge scores."""

from __future__ import annotations

import math
from collections import defaultdict

from .judge import JudgeScore
from .runner import EvalResult


def _sign_test_p(wins: int, losses: int) -> float:
    """Two-sided sign test p-value using exact binomial calculation."""
    n = wins + losses
    if n == 0:
        return 1.0
    # Count probability of observing result at least as extreme under H0: p=0.5
    k = min(wins, losses)
    # Sum of binomial probabilities for 0..k on both tails
    p = 0.0
    for i in range(k + 1):
        p += math.comb(n, i) * (0.5 ** n)
    return min(p * 2, 1.0)  # two-sided


def compute_metrics(eval_results: list[EvalResult]) -> dict:
    """Compute aggregate metrics from evaluation results.

    Returns a dict with:
        - per_variant: {variant_name: {win_rate, per_category, per_dimension, sign_test_p}}
    """
    output: dict = {"per_variant": {}}

    for result in eval_results:
        scores = result.scores
        if not scores:
            output["per_variant"][result.variant_name] = {
                "n_probes": 0,
                "win_rate": 0.0,
            }
            continue

        n = len(scores)
        wins = sum(1 for s in scores if s.preferred == "schema")
        losses = sum(1 for s in scores if s.preferred == "baseline")
        ties = sum(1 for s in scores if s.preferred == "tie")

        # Per-category breakdown
        by_category: dict[str, dict] = defaultdict(lambda: {"wins": 0, "losses": 0, "ties": 0, "n": 0})
        for s in scores:
            cat = by_category[s.category]
            cat["n"] += 1
            if s.preferred == "schema":
                cat["wins"] += 1
            elif s.preferred == "baseline":
                cat["losses"] += 1
            else:
                cat["ties"] += 1

        category_rates = {}
        for cat_name, cat_data in by_category.items():
            cat_total = cat_data["n"]
            category_rates[cat_name] = {
                "win_rate": cat_data["wins"] / cat_total if cat_total else 0,
                "loss_rate": cat_data["losses"] / cat_total if cat_total else 0,
                "tie_rate": cat_data["ties"] / cat_total if cat_total else 0,
                "n": cat_total,
            }

        # Per-dimension mean score deltas
        dimensions = ["coherence", "transition", "awareness"]
        dim_deltas: dict[str, float] = {}
        for dim in dimensions:
            deltas = []
            for s in scores:
                schema_val = s.schema_scores.get(dim, 3)
                baseline_val = s.baseline_scores.get(dim, 3)
                deltas.append(schema_val - baseline_val)
            dim_deltas[dim] = sum(deltas) / len(deltas) if deltas else 0.0

        # A/B order balance check
        a_count = sum(1 for s in scores if s.schema_was_a)

        output["per_variant"][result.variant_name] = {
            "n_probes": n,
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "win_rate": wins / n,
            "loss_rate": losses / n,
            "per_category": dict(category_rates),
            "per_dimension_delta": dim_deltas,
            "sign_test_p": _sign_test_p(wins, losses),
            "schema_was_a_pct": a_count / n,
        }

    return output


def print_summary(metrics: dict) -> None:
    """Print a human-readable summary table."""
    for variant, data in metrics.get("per_variant", {}).items():
        n = data.get("n_probes", 0)
        print(f"\n{'=' * 60}")
        print(f"  Variant: {variant} vs baseline  ({n} probe turns)")
        print(f"{'=' * 60}")

        if n == 0:
            print("  No data.")
            continue

        w = data["wins"]
        l = data["losses"]
        t = data["ties"]
        print(f"  Win/Loss/Tie: {w}/{l}/{t}  "
              f"(win rate: {data['win_rate']:.1%})")
        print(f"  Sign test p-value: {data['sign_test_p']:.4f}")
        print(f"  A/B balance: schema was 'A' {data['schema_was_a_pct']:.1%} of the time")

        print(f"\n  Per-category breakdown:")
        for cat, cat_data in data.get("per_category", {}).items():
            print(f"    {cat:15s}  win {cat_data['win_rate']:.0%}  "
                  f"loss {cat_data['loss_rate']:.0%}  "
                  f"tie {cat_data['tie_rate']:.0%}  "
                  f"(n={cat_data['n']})")

        print(f"\n  Mean score delta (schema - baseline):")
        for dim, delta in data.get("per_dimension_delta", {}).items():
            sign = "+" if delta >= 0 else ""
            print(f"    {dim:15s}  {sign}{delta:.2f}")
