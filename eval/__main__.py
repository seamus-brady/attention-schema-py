"""CLI entry point: python -m eval"""

from __future__ import annotations

import argparse
import sys

from .scenarios import SCENARIOS
from .runner import run_evaluation, VARIANT_CONTROLLERS
from .metrics import compute_metrics, print_summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="A/B evaluation harness for the attention schema chatbot."
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use MockLLMClient instead of real API (tests the harness).",
    )
    parser.add_argument(
        "--category",
        choices=["coherence", "shift", "self_report"],
        default=None,
        help="Run only scenarios in this category.",
    )
    parser.add_argument(
        "--variants",
        type=str,
        default="full",
        help="Comma-separated variant names to test against baseline "
             "(e.g. 'full,claims_only,attention_only').",
    )
    args = parser.parse_args(argv)

    # Filter scenarios by category
    scenarios = SCENARIOS
    if args.category:
        scenarios = [s for s in scenarios if s.category == args.category]
        if not scenarios:
            print(f"No scenarios found for category '{args.category}'.")
            sys.exit(1)

    # Parse variants
    variant_names = [v.strip() for v in args.variants.split(",")]
    for v in variant_names:
        if v not in VARIANT_CONTROLLERS:
            print(f"Unknown variant '{v}'. Available: {list(VARIANT_CONTROLLERS.keys())}")
            sys.exit(1)
    # Always include baseline
    all_variants = list(dict.fromkeys(["baseline"] + variant_names))

    mode = "MOCK" if args.mock else "LIVE"
    print(f"Attention Schema Evaluation ({mode})")
    print(f"  Scenarios: {len(scenarios)}")
    print(f"  Variants:  {', '.join(all_variants)}")
    print()

    results = run_evaluation(
        scenarios=scenarios,
        variants=all_variants,
        use_real_llm=not args.mock,
    )

    metrics = compute_metrics(results)
    print_summary(metrics)


if __name__ == "__main__":
    main()
