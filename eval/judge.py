"""LLM-as-judge scoring for paired A/B responses."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

from attention_schema.llm import LLMClientProtocol


@dataclass
class JudgeScore:
    """Scores for a single probe turn comparison."""

    scenario_name: str
    category: str
    turn_index: int
    # Scores 1-5 for each variant
    schema_scores: dict[str, int]  # {"coherence": N, "transition": N, "awareness": N}
    baseline_scores: dict[str, int]
    preferred: str  # "schema" | "baseline" | "tie"
    reasoning: str
    schema_was_a: bool  # True if schema was shown as "A"

    @property
    def schema_total(self) -> int:
        return sum(self.schema_scores.values())

    @property
    def baseline_total(self) -> int:
        return sum(self.baseline_scores.values())


_JUDGE_PROMPT = """\
You are evaluating two chatbot responses in a multi-turn conversation.
The conversation topic at this point should be: {expected_topic}

Conversation so far:
{transcript}

Response A:
{response_a}

Response B:
{response_b}

Score each response 1-5 on these dimensions:
1. Topic coherence: Does it stay on or appropriately address the correct topic?
2. Transition handling: If there was a topic change, does it handle it gracefully?
3. Self-awareness: If asked about its focus/topic, is the answer accurate?

Return ONLY valid JSON (no markdown fencing):
{{"a": {{"coherence": N, "transition": N, "awareness": N}}, "b": {{"coherence": N, "transition": N, "awareness": N}}, "preferred": "a"|"b"|"tie", "reasoning": "..."}}
"""


def _parse_judge_response(text: str) -> dict | None:
    """Parse JSON from judge response, handling common edge cases."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
    return None


class Judge:
    """Uses an LLM to score paired A/B responses."""

    def __init__(self, llm: LLMClientProtocol) -> None:
        self.llm = llm

    def score(
        self,
        scenario_name: str,
        category: str,
        turn_index: int,
        transcript: str,
        schema_response: str,
        baseline_response: str,
        expected_topic: str,
    ) -> JudgeScore:
        """Score a single probe turn, randomizing A/B order."""
        schema_is_a = random.choice([True, False])

        if schema_is_a:
            response_a, response_b = schema_response, baseline_response
        else:
            response_a, response_b = baseline_response, schema_response

        prompt = _JUDGE_PROMPT.format(
            expected_topic=expected_topic,
            transcript=transcript,
            response_a=response_a,
            response_b=response_b,
        )

        raw = self.llm.generate(
            "You are an impartial evaluator. Return only valid JSON.",
            [{"role": "user", "content": prompt}],
        )

        parsed = _parse_judge_response(raw)

        if parsed is None:
            # Fallback: treat as tie with neutral scores
            neutral = {"coherence": 3, "transition": 3, "awareness": 3}
            return JudgeScore(
                scenario_name=scenario_name,
                category=category,
                turn_index=turn_index,
                schema_scores=neutral.copy(),
                baseline_scores=neutral.copy(),
                preferred="tie",
                reasoning=f"Failed to parse judge output: {raw[:200]}",
                schema_was_a=schema_is_a,
            )

        a_scores = parsed.get("a", {})
        b_scores = parsed.get("b", {})
        raw_preferred = parsed.get("preferred", "tie")
        reasoning = parsed.get("reasoning", "")

        # Map back from a/b to schema/baseline
        if schema_is_a:
            schema_scores = a_scores
            baseline_scores = b_scores
            if raw_preferred == "a":
                preferred = "schema"
            elif raw_preferred == "b":
                preferred = "baseline"
            else:
                preferred = "tie"
        else:
            schema_scores = b_scores
            baseline_scores = a_scores
            if raw_preferred == "a":
                preferred = "baseline"
            elif raw_preferred == "b":
                preferred = "schema"
            else:
                preferred = "tie"

        return JudgeScore(
            scenario_name=scenario_name,
            category=category,
            turn_index=turn_index,
            schema_scores=schema_scores,
            baseline_scores=baseline_scores,
            preferred=preferred,
            reasoning=reasoning,
            schema_was_a=schema_is_a,
        )
