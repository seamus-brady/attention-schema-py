from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from .tokenizer import tokenize, token_overlap


@dataclass
class AttentionState:
    """Snapshot of what the system is attending to."""

    focus_target: str
    confidence: float
    reason: str
    history: list[dict] = field(default_factory=list)


@dataclass
class GroundTruth:
    """Actual mechanism state received from the attention mechanism."""

    targets: list  # list[AttentionTarget] — kept generic to avoid circular import
    timestamp: int


@dataclass
class AwarenessClaims:
    """Structured claims about the system's current awareness state."""

    current_focus: str
    focus_confidence: str  # "high", "moderate", "low"
    awareness_of_shift: str | None = None
    predicted_next: str | None = None
    user_model_summary: str | None = None
    dissociation_note: str | None = None

    def to_prompt_block(self) -> str:
        lines = [
            "=== AWARENESS STATE ===",
            f"Focus: {self.current_focus}",
            f"Confidence: {self.focus_confidence}",
        ]
        if self.awareness_of_shift:
            lines.append(f"Shift: {self.awareness_of_shift}")
        if self.predicted_next:
            lines.append(f"Predicted next focus: {self.predicted_next}")
        if self.user_model_summary:
            lines.append(f"User model: {self.user_model_summary}")
        if self.dissociation_note:
            lines.append(f"Dissociation note: {self.dissociation_note}")
        lines.append("=== END AWARENESS STATE ===")
        return "\n".join(lines)


class AttentionSchema:
    """Internal self-model of the system's own attention — the core AST idea."""

    def __init__(
        self,
        shift_overlap_threshold: float = 0.2,
        shift_confidence_threshold: float = 0.4,
        confidence_increment: float = 0.05,
        max_history: int = 50,
        schema_inertia: float = 0.0,
        reconciliation_threshold: float = 0.3,
    ) -> None:
        self.state = AttentionState(
            focus_target="",
            confidence=0.0,
            reason="No input yet.",
        )
        self.shift_overlap_threshold = shift_overlap_threshold
        self.shift_confidence_threshold = shift_confidence_threshold
        self.confidence_increment = confidence_increment
        self.max_history = max_history
        self.schema_inertia = schema_inertia
        self.reconciliation_threshold = reconciliation_threshold

        # Explicit shift markers in user input
        self._shift_markers = {"actually", "new topic", "forget that", "never mind", "nevermind", "change of subject", "switching gears"}

        # Ground truth from the actual attention mechanism
        self._ground_truth: GroundTruth | None = None

        # Prediction state
        self.prediction: str | None = None
        self.prediction_error_history: list[float] = []
        self.prediction_learning_rate: float = 0.1

        # Transition history for prediction: maps focus -> next_focus
        self._transition_history: list[tuple[str, str]] = []

    def update(self, focus_target: str, confidence: float, reason: str) -> None:
        """Record a new attention state, pushing the old one into history."""
        if self.state.focus_target:
            self.state.history.append(
                {
                    "focus_target": self.state.focus_target,
                    "confidence": self.state.confidence,
                    "reason": self.state.reason,
                }
            )
            if len(self.state.history) > self.max_history:
                self.state.history = self.state.history[-self.max_history :]
        self.state.focus_target = focus_target
        self.state.confidence = confidence
        self.state.reason = reason

    def receive_ground_truth(self, targets: list, turn: int) -> None:
        """Store actual mechanism state for later reconciliation."""
        self._ground_truth = GroundTruth(targets=targets, timestamp=turn)

    def update_from_competition(self, targets: list) -> None:
        """Lossy update from competitive attention results.

        Takes only the top target's content (discards runner-up activations).
        Applies inertia: only shifts if new target activation exceeds current
        confidence by ``schema_inertia``.
        """
        if not targets:
            return

        # Find top target by activation
        top = max(targets, key=lambda t: t.activation)

        old_focus = self.state.focus_target

        # Apply inertia: only shift if the new top target's activation
        # exceeds current confidence by the inertia margin
        if (
            old_focus
            and self.schema_inertia > 0
            and top.activation < self.state.confidence + self.schema_inertia
            and token_overlap(top.content, old_focus) < self.shift_overlap_threshold
        ):
            # Inertia prevents the shift — stay on current focus
            self.state.reason = "Schema inertia maintained current focus."
            return

        # Record transition for prediction
        if old_focus and top.content != old_focus:
            self._transition_history.append((old_focus, top.content))
            # Keep bounded
            if len(self._transition_history) > 100:
                self._transition_history = self._transition_history[-100:]

        # Lossy update: only top target content, simplified confidence
        if old_focus and token_overlap(top.content, old_focus) >= self.shift_overlap_threshold:
            # Continuing same topic
            confidence = min(
                0.6 * max(top.activation, 0.3) + 0.4 * self.state.confidence,
                1.0,
            )
            reason = "Continuing with current focus."
        else:
            # New topic
            confidence = top.activation
            reason = "New topic detected — shifting attention."

        self.update(top.content[:120], confidence, reason)

    def reconcile(self) -> dict:
        """Compare schema focus vs ground truth and correct if mismatch is large."""
        if self._ground_truth is None or not self._ground_truth.targets:
            return {"mismatch": 0.0, "corrected": False, "old_focus": self.state.focus_target, "new_focus": self.state.focus_target}

        # Find the top ground truth target
        gt_top = max(self._ground_truth.targets, key=lambda t: t.activation)

        # Measure mismatch between schema focus and ground truth
        if not self.state.focus_target or not gt_top.content:
            mismatch = 1.0
        else:
            overlap = token_overlap(self.state.focus_target, gt_top.content)
            mismatch = 1.0 - overlap

        old_focus = self.state.focus_target

        corrected = False
        if mismatch > self.reconciliation_threshold and gt_top.activation > 0.3:
            # Schema is significantly wrong — correct it
            self.state.focus_target = gt_top.content[:120]
            self.state.confidence = gt_top.activation
            self.state.reason = "Reconciliation corrected schema to match ground truth."
            corrected = True

        return {
            "mismatch": mismatch,
            "corrected": corrected,
            "old_focus": old_focus,
            "new_focus": self.state.focus_target,
        }

    def predict_next_focus(self, context_items: list[str]) -> str:
        """Predict what the next focus will be based on transition history and context."""
        if not context_items:
            self.prediction = self.state.focus_target
            return self.prediction

        current = self.state.focus_target

        # Check transition history for A -> B patterns
        candidates: dict[str, float] = {}
        for src, dst in self._transition_history:
            if token_overlap(src, current) > 0.3:
                candidates[dst] = candidates.get(dst, 0) + 1.0

        # Also score context items by overlap with current focus
        for item in context_items:
            overlap = token_overlap(item, current)
            if overlap > 0.1:
                candidates[item[:120]] = candidates.get(item[:120], 0) + overlap

        if candidates:
            self.prediction = max(candidates, key=candidates.get)
        else:
            self.prediction = current

        return self.prediction

    def compute_prediction_error(self, actual_focus: str) -> float:
        """Compute error between prediction and actual focus. 0=perfect, 1=total miss."""
        if self.prediction is None:
            return 1.0
        overlap = token_overlap(self.prediction, actual_focus)
        error = 1.0 - overlap
        self.prediction_error_history.append(error)
        return error

    def learn_from_error(self, error: float) -> None:
        """Adjust schema_inertia based on prediction error magnitude."""
        if error > 0.7:
            # High error: schema is too rigid, decrease inertia
            self.schema_inertia = max(0.05, self.schema_inertia - self.prediction_learning_rate)
        elif error < 0.3:
            # Low error: schema predictions are good, increase inertia (more stable)
            self.schema_inertia = min(0.8, self.schema_inertia + self.prediction_learning_rate)

    def generate_claims(self) -> AwarenessClaims:
        """Generate structured awareness claims from current schema state."""
        # Confidence level
        if self.state.confidence >= 0.7:
            conf_label = "high"
        elif self.state.confidence >= 0.4:
            conf_label = "moderate"
        else:
            conf_label = "low"

        # Awareness of shift
        shift_note = None
        if self.state.reason and "shifting" in self.state.reason.lower():
            if self.state.history:
                old = self.state.history[-1]["focus_target"]
                shift_note = f"Shifted from '{old}' to '{self.state.focus_target}'"

        return AwarenessClaims(
            current_focus=self.state.focus_target or "(none)",
            focus_confidence=conf_label,
            awareness_of_shift=shift_note,
            predicted_next=self.prediction,
        )

    def compute_response_strategy(self, user_input: str, user_model_summary: str, shifting: bool) -> dict:
        """Return a dict of instructions that shape the system prompt."""
        strategy: dict[str, str | None] = {}

        if shifting:
            lower_input = user_input.lower()
            has_explicit_marker = any(marker in lower_input for marker in self._shift_markers)
            if has_explicit_marker:
                strategy["topic_instruction"] = "The user has changed topics. Respond to the new topic directly."
            else:
                strategy["topic_instruction"] = (
                    f"The user's question may relate to the prior discussion about {self.state.focus_target}. "
                    "Address it in that context unless it clearly doesn't fit."
                )

        if self.state.confidence < 0.4:
            strategy["framing"] = "Keep your response focused and concise."

        if user_model_summary and user_model_summary != "No model of user attention yet.":
            strategy["user_adaptation"] = user_model_summary

        return strategy

    def select_context(self, context_items: list[str], user_input: str, attention, top_k: int = 3) -> list[str]:
        """Schema-controlled context selection.

        Reorders items when prediction is relevant, adjusts focus_bias usage
        based on confidence, and widens top_k when confidence is low.
        """
        if not context_items:
            return []

        items = list(context_items)

        # If prediction overlaps with user_input, reorder to put prediction-relevant items first
        if self.prediction and token_overlap(self.prediction, user_input) > 0.3:
            def prediction_relevance(item: str) -> float:
                return token_overlap(item, self.prediction)
            items.sort(key=prediction_relevance, reverse=True)

        # Only pass focus_bias when confidence is high enough
        focus_bias = self.state.focus_target if self.state.confidence > 0.6 else None

        # Widen top_k when confidence is low (hedge by including more context)
        effective_k = top_k + 2 if self.state.confidence < 0.4 else top_k

        return attention.select(
            items,
            user_input,
            top_k=effective_k,
            focus_bias=focus_bias,
        )

    def should_shift(
        self, new_input: str, context_items: list[str] | None = None
    ) -> bool:
        """Heuristic: does *new_input* warrant shifting attention away from the current focus?"""
        if not self.state.focus_target:
            return True

        overlap = token_overlap(new_input, self.state.focus_target)
        if overlap >= self.shift_overlap_threshold:
            return False

        if context_items:
            recent = context_items[-3:]
            for item in recent:
                if token_overlap(new_input, item) >= self.shift_overlap_threshold:
                    return False

        if self.state.confidence < self.shift_confidence_threshold:
            return True
        return False

    def summary(self) -> str:
        """Natural-language self-report of the current attention state."""
        if not self.state.focus_target:
            return "I am not currently focused on anything."
        return (
            f"I am currently attending to '{self.state.focus_target}' "
            f"(confidence {self.state.confidence:.2f}). "
            f"Reason: {self.state.reason}"
        )

    def to_dict(self) -> dict:
        """Serialize schema state to a dictionary (for persistence)."""
        return {
            "focus_target": self.state.focus_target,
            "confidence": self.state.confidence,
            "reason": self.state.reason,
            "history": self.state.history,
            "schema_inertia": self.schema_inertia,
            "prediction": self.prediction,
            "prediction_error_history": self.prediction_error_history,
            "config": {
                "shift_overlap_threshold": self.shift_overlap_threshold,
                "shift_confidence_threshold": self.shift_confidence_threshold,
                "confidence_increment": self.confidence_increment,
                "max_history": self.max_history,
                "schema_inertia": self.schema_inertia,
                "reconciliation_threshold": self.reconciliation_threshold,
            },
        }

    def save(self, filepath: str | Path) -> None:
        """Save schema state to a JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str | Path) -> AttentionSchema:
        """Load schema state from a JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)

        config = data.get("config", {})
        schema = cls(**config)
        schema.state.focus_target = data["focus_target"]
        schema.state.confidence = data["confidence"]
        schema.state.reason = data["reason"]
        schema.state.history = data.get("history", [])
        schema.schema_inertia = data.get("schema_inertia", config.get("schema_inertia", 0.0))
        schema.prediction = data.get("prediction")
        schema.prediction_error_history = data.get("prediction_error_history", [])
        return schema
