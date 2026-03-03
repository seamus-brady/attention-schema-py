from __future__ import annotations

from .schema import AttentionSchema
from .tokenizer import tokenize, token_overlap


class UserAttentionModel:
    """Model of the user's attention, built with the same schema machinery.

    Graziano's theory says the same mechanism that models self-attention
    can model others' attention (Theory of Mind). This class wraps an
    AttentionSchema instance with lower inertia to be responsive to
    user input signals.
    """

    def __init__(self, schema_inertia: float = 0.1) -> None:
        self.schema = AttentionSchema(schema_inertia=schema_inertia)

    def update_from_input(self, user_input: str, context_items: list[str]) -> None:
        """Update the user model based on user input.

        The user's input IS their attention signal — what they type about
        is what they're attending to.
        """
        focus = user_input[:120]

        # Confidence based on input specificity: longer, more specific inputs
        # signal stronger attention
        tokens = tokenize(user_input)
        if len(tokens) >= 5:
            confidence = 0.8
        elif len(tokens) >= 2:
            confidence = 0.6
        else:
            confidence = 0.4

        self.schema.update(focus, confidence, "Inferred from user input.")

    def predict_user_interest(self, context_items: list[str]) -> str:
        """Predict what the user will attend to next."""
        return self.schema.predict_next_focus(context_items)

    def get_response_adaptation(self) -> str | None:
        """Return an instruction based on user input patterns."""
        history = self.schema.state.history
        if len(history) < 2:
            return None

        # Check recent input lengths from focus_target (which stores user input)
        recent = history[-3:]
        avg_len = sum(len(h["focus_target"]) for h in recent) / len(recent)

        if avg_len < 30:
            return "Match their style with focused answers."
        elif avg_len > 80:
            return "Respond with matching depth."
        return None

    def summary(self) -> str:
        """Natural language summary of the user's modeled attention."""
        focus = self.schema.state.focus_target
        conf = self.schema.state.confidence
        if not focus:
            return "No model of user attention yet."
        return f"The user appears focused on '{focus}' (confidence {conf:.2f})"
