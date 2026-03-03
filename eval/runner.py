"""Runs both controllers through scenarios and collects judge results."""

from __future__ import annotations

from dataclasses import dataclass, field

from attention_schema.controller import Controller
from attention_schema.llm import LLMClient, MockLLMClient, LLMClientProtocol

from .baseline import BaselineController, ClaimsOnlyController, AttentionOnlyController, LegacySchemaController
from .judge import Judge, JudgeScore
from .scenarios import Scenario, SCENARIOS


# Map of variant names to controller classes
VARIANT_CONTROLLERS: dict[str, type] = {
    "full": Controller,
    "baseline": BaselineController,
    "claims_only": ClaimsOnlyController,
    "attention_only": AttentionOnlyController,
    "legacy": LegacySchemaController,
}


@dataclass
class EvalResult:
    """All judge scores for one variant pair across all scenarios."""

    variant_name: str
    scores: list[JudgeScore] = field(default_factory=list)


def _build_transcript(turns: list[str], responses: dict[str, list[str]], up_to: int) -> str:
    """Build a conversation transcript up to (and including) the given turn index."""
    lines: list[str] = []
    for i in range(up_to):
        lines.append(f"User: {turns[i]}")
        # Use the schema variant responses for transcript context
        if i < len(responses.get("full", responses.get("baseline", []))):
            first_variant = next(iter(responses.values()))
            if i < len(first_variant):
                lines.append(f"Assistant: {first_variant[i]}")
    # Include the final user turn (the probe)
    lines.append(f"User: {turns[up_to]}")
    return "\n".join(lines)


def run_evaluation(
    scenarios: list[Scenario] | None = None,
    variants: list[str] | None = None,
    use_real_llm: bool = True,
    judge_llm: LLMClientProtocol | None = None,
) -> list[EvalResult]:
    """Run evaluation across scenarios for specified variants vs baseline.

    Args:
        scenarios: Scenarios to evaluate. Defaults to all SCENARIOS.
        variants: Variant names to test against baseline. Defaults to ["full"].
        use_real_llm: If False, use MockLLMClient for controllers.
        judge_llm: LLM client for the judge. Defaults to real LLM or mock.

    Returns:
        List of EvalResult, one per non-baseline variant.
    """
    if scenarios is None:
        scenarios = SCENARIOS
    if variants is None:
        variants = ["full", "baseline"]

    # Ensure baseline is always included
    if "baseline" not in variants:
        variants = ["baseline"] + list(variants)

    non_baseline = [v for v in variants if v != "baseline"]

    # Set up LLM clients
    if use_real_llm:
        controller_llm: LLMClientProtocol = LLMClient()
        if judge_llm is None:
            judge_llm = LLMClient(model="claude-sonnet-4-5-20250929")
    else:
        controller_llm = MockLLMClient()
        if judge_llm is None:
            judge_llm = MockLLMClient()

    judge = Judge(llm=judge_llm)
    results = {v: EvalResult(variant_name=v) for v in non_baseline}

    for scenario in scenarios:
        print(f"  Running scenario: {scenario.name} ({scenario.category})")

        # Build fresh controllers for this scenario
        controllers: dict[str, Controller] = {}
        for v in variants:
            cls = VARIANT_CONTROLLERS[v]
            controllers[v] = cls(llm=controller_llm)

        # Collect responses per variant
        responses: dict[str, list[str]] = {v: [] for v in variants}

        for i, user_msg in enumerate(scenario.turns):
            for v in variants:
                resp = controllers[v].run(user_msg)
                responses[v].append(resp)

            # Judge probe turns
            if i in scenario.probe_turns:
                transcript = _build_transcript(scenario.turns, responses, i)
                for v in non_baseline:
                    score = judge.score(
                        scenario_name=scenario.name,
                        category=scenario.category,
                        turn_index=i,
                        transcript=transcript,
                        schema_response=responses[v][i],
                        baseline_response=responses["baseline"][i],
                        expected_topic=scenario.expected_topic,
                    )
                    results[v].scores.append(score)

    return list(results.values())
