from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DissociationReport:
    """Measures the gap between what the mechanism attends to and what the schema reports."""

    attended_not_aware: list[str]
    aware_not_attended: list[str]
    gap_score: float
    turn: int


class DissociationTracker:
    """Tracks attention/awareness dissociation over time.

    In Graziano's theory, the schema (awareness) is a lossy model of
    attention. This tracker measures when they diverge: items that the
    mechanism attends to but the schema doesn't report (attended-not-aware),
    and items the schema claims to focus on but the mechanism doesn't
    actually attend to (aware-not-attended).
    """

    def __init__(self, awareness_threshold: float = 0.3) -> None:
        self.awareness_threshold = awareness_threshold
        self.history: list[DissociationReport] = []

    def measure(
        self,
        ground_truth: list,  # list[AttentionTarget]
        schema_focus: str,
        schema_confidence: float,
        turn: int,
    ) -> DissociationReport:
        """Measure dissociation between mechanism ground truth and schema report."""
        from .tokenizer import token_overlap

        # High-activation targets the mechanism actually attends to
        high_activation = [
            t for t in ground_truth if t.activation >= self.awareness_threshold
        ]

        # Attended but not in schema focus (attended-not-aware)
        attended_not_aware = []
        for target in high_activation:
            overlap = token_overlap(target.content, schema_focus) if schema_focus else 0.0
            if overlap < 0.2:
                attended_not_aware.append(target.content)

        # Schema focus not matching any high-activation target (aware-not-attended)
        aware_not_attended = []
        if schema_focus and schema_confidence > 0.3:
            matched = False
            for target in high_activation:
                if token_overlap(schema_focus, target.content) >= 0.2:
                    matched = True
                    break
            if not matched:
                aware_not_attended.append(schema_focus)

        # Compute gap score
        total_items = len(high_activation) + (1 if schema_focus else 0)
        dissociated_items = len(attended_not_aware) + len(aware_not_attended)
        gap_score = dissociated_items / max(total_items, 1)

        report = DissociationReport(
            attended_not_aware=attended_not_aware,
            aware_not_attended=aware_not_attended,
            gap_score=gap_score,
            turn=turn,
        )
        self.history.append(report)
        return report
