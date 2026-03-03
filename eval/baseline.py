"""Variant controllers for A/B evaluation."""

from __future__ import annotations

from collections import defaultdict

from attention_schema.controller import Controller, SYSTEM_TEMPLATE
from attention_schema.llm import LLMClientProtocol
from attention_schema.tokenizer import token_overlap

LEGACY_SYSTEM_TEMPLATE = """\
You are an assistant whose cognition is guided by an internal attention schema \
(based on Michael Graziano's Attention Schema Theory).

{awareness_claims}

{user_model_summary}

IMPORTANT: Your self-reports about your own attention, focus, and awareness \
MUST be derived from the AWARENESS STATE block above. Do not fabricate \
metacognitive statements independent of the schema. If you mention what you \
are focused on, use the focus stated above. If you note a shift, reference \
the shift described above."""


class BaselineController(Controller):
    """Same LLM, same conversation history, no schema injection."""

    def run(self, user_input: str) -> str:
        self.turn_count += 1
        self.messages.append({"role": "user", "content": user_input})

        max_msgs = self.max_history_turns * 2
        if len(self.messages) > max_msgs:
            self.messages = self.messages[-max_msgs:]

        system = "You are a helpful assistant."
        response = self.llm.generate(system, self.messages)
        self.messages.append({"role": "assistant", "content": response})
        return response


class ClaimsOnlyController(Controller):
    """Injects awareness claims block but skips competitive attention / context scoring."""

    def run(self, user_input: str) -> str:
        self.turn_count += 1

        # Update schema from user input directly (no competitive attention)
        self.schema.update(
            focus_target=user_input[:120],
            confidence=0.5,
            reason="direct_update",
        )

        claims = self.schema.generate_claims()
        claims.user_model_summary = self.user_model.summary()
        self.user_model.update_from_input(user_input, self.context_items)

        template_vars = defaultdict(str, {
            "awareness_claims": claims.to_prompt_block(),
            "user_model_summary": self.user_model.summary(),
            "schema_summary": self.schema.summary(),
        })
        system_prompt = self.system_template.format_map(template_vars)

        self.messages.append({"role": "user", "content": user_input})

        max_msgs = self.max_history_turns * 2
        if len(self.messages) > max_msgs:
            self.messages = self.messages[-max_msgs:]

        response = self.llm.generate(system_prompt, self.messages)
        self.messages.append({"role": "assistant", "content": response})
        self._add_context(response)
        return response


class AttentionOnlyController(Controller):
    """Runs competitive attention / context scoring but no awareness claims in prompt."""

    def run(self, user_input: str) -> str:
        self.turn_count += 1
        self.user_model.update_from_input(user_input, self.context_items)

        # Run competitive attention for context selection
        selected = self.attention.select(
            self.context_items, user_input, top_k=self.top_k_context
        )

        if selected:
            context_block = "\n".join(f"- {item}" for item in selected)
            full_user = f"Relevant context:\n{context_block}\n\nUser: {user_input}"
        else:
            full_user = user_input

        self.messages.append({"role": "user", "content": full_user})

        max_msgs = self.max_history_turns * 2
        if len(self.messages) > max_msgs:
            self.messages = self.messages[-max_msgs:]

        # Plain system prompt, no awareness block
        system = "You are a helpful assistant."
        response = self.llm.generate(system, self.messages)
        self.messages.append({"role": "assistant", "content": response})
        self._add_context(response)
        return response


class LegacySchemaController(Controller):
    """Old awareness-claims flow for comparison against refactored controller."""

    def __init__(self, **kwargs):
        kwargs.setdefault("system_template", LEGACY_SYSTEM_TEMPLATE)
        super().__init__(**kwargs)

    def run(self, user_input: str) -> str:
        self.turn_count += 1

        # 1. Update user model
        self.user_model.update_from_input(user_input, self.context_items)

        # 2. Decide shift (old buggy behavior: always shifts on low overlap)
        shifting = self._legacy_should_shift(user_input)

        # 3. Score context via competitive attention
        if self.context_items:
            selected = self.attention.select(
                self.context_items,
                user_input,
                top_k=self.top_k_context,
                focus_bias=self.schema.state.focus_target or None,
            )
        else:
            selected = []

        # 4. Ground truth
        ground_truth = self.attention.get_ground_truth()
        self.schema.receive_ground_truth(ground_truth, self.turn_count)

        # 5. Update schema
        if ground_truth and not shifting:
            self.schema.update_from_competition(ground_truth)
        else:
            if shifting:
                self.schema.update(user_input[:120], 0.8, "New topic detected — shifting attention.")
            else:
                focus_target = self.schema.state.focus_target
                overlap = token_overlap(user_input, focus_target)
                confidence = min(
                    0.6 * max(overlap, 0.3) + 0.4 * self.schema.state.confidence, 1.0
                )
                self.schema.update(focus_target, confidence, "Continuing with current focus.")

        self.schema.reconcile()

        # 6. Prediction
        if self.turn_count > 1 and self.schema.prediction is not None:
            error = self.schema.compute_prediction_error(self.schema.state.focus_target)
            self.schema.learn_from_error(error)
        self.schema.predict_next_focus(self.context_items)

        # 7. Generate awareness claims (old flow)
        claims = self.schema.generate_claims()
        claims.user_model_summary = self.user_model.summary()

        template_vars = defaultdict(str, {
            "awareness_claims": claims.to_prompt_block(),
            "user_model_summary": self.user_model.summary(),
            "schema_summary": self.schema.summary(),
        })
        system_prompt = self.system_template.format_map(template_vars)

        # 8. Build user message
        if selected:
            context_block = "\n".join(f"- {item}" for item in selected)
            full_user = f"Relevant context:\n{context_block}\n\nUser: {user_input}"
        else:
            full_user = user_input

        self.messages.append({"role": "user", "content": full_user})
        max_msgs = self.max_history_turns * 2
        if len(self.messages) > max_msgs:
            self.messages = self.messages[-max_msgs:]

        response = self.llm.generate(system_prompt, self.messages)
        self.messages.append({"role": "assistant", "content": response})
        self._add_context(user_input)
        self._add_context(response[:500])
        return response

    def _legacy_should_shift(self, new_input: str) -> bool:
        """Old buggy should_shift: always returns True when overlap is low."""
        if not self.schema.state.focus_target:
            return True
        overlap = token_overlap(new_input, self.schema.state.focus_target)
        if overlap >= self.schema.shift_overlap_threshold:
            return False
        if self.context_items:
            for item in self.context_items[-3:]:
                if token_overlap(new_input, item) >= self.schema.shift_overlap_threshold:
                    return False
        # Bug: always shifts regardless of confidence
        return True
