from __future__ import annotations

from pathlib import Path
from typing import Optional

from .attention import AttentionMechanism
from .dissociation import DissociationTracker
from .llm import LLMClientProtocol, LLMClient
from .schema import AttentionSchema
from .social import UserAttentionModel
from .tokenizer import token_overlap


SYSTEM_TEMPLATE = "You are a helpful assistant.{strategy_instructions}"


class Controller:
    """Main agent loop: score context -> update schema -> call LLM."""

    def __init__(
        self,
        llm: LLMClientProtocol | None = None,
        context_items: list[str] | None = None,
        schema: AttentionSchema | None = None,
        max_context_items: int = 100,
        top_k_context: int = 3,
        system_template: str = SYSTEM_TEMPLATE,
        max_history_turns: int = 20,
    ) -> None:
        self.schema = schema or AttentionSchema()
        self.attention = AttentionMechanism()
        self.llm = llm or LLMClient()
        self.context_items: list[str] = context_items or []
        self.max_context_items = max_context_items
        self.top_k_context = top_k_context
        self.system_template = system_template
        self.max_history_turns = max_history_turns
        self.messages: list[dict] = []
        self.turn_count: int = 0
        self.dissociation = DissociationTracker()
        self.user_model = UserAttentionModel()

    def run(self, user_input: str) -> str:
        """Execute one attention-schema cycle and return the LLM response."""
        self.turn_count += 1

        # 1. Update user model from input (Theory of Mind)
        self.user_model.update_from_input(user_input, self.context_items)

        # 2. Check previous prediction, learn from error
        if self.turn_count > 1 and self.schema.prediction is not None:
            error = self.schema.compute_prediction_error(self.schema.state.focus_target)
            self.schema.learn_from_error(error)

        # 3. Decide whether to shift attention (bug fixed: high confidence + low overlap → stay)
        shifting = self.schema.should_shift(user_input, context_items=self.context_items)

        # 4. Schema-controlled context selection
        selected = self.schema.select_context(
            self.context_items, user_input, self.attention, top_k=self.top_k_context
        )

        # 5. Get ground truth and pass to schema
        ground_truth = self.attention.get_ground_truth()
        self.schema.receive_ground_truth(ground_truth, self.turn_count)

        # 6. Update schema from competition results (lossy)
        if ground_truth and not shifting:
            self.schema.update_from_competition(ground_truth)
        else:
            if shifting:
                focus_target = user_input[:120]
                confidence = 0.8
                reason = "New topic detected — shifting attention."
            else:
                focus_target = self.schema.state.focus_target
                overlap = token_overlap(user_input, focus_target)
                confidence = min(
                    0.6 * max(overlap, 0.3) + 0.4 * self.schema.state.confidence,
                    1.0,
                )
                reason = "Continuing with current focus."
            self.schema.update(focus_target, confidence, reason)

        # 7. Reconciliation step
        self.schema.reconcile()

        # 8. Dissociation measurement (still tracked, just not injected into prompt)
        self.dissociation.measure(
            ground_truth,
            self.schema.state.focus_target,
            self.schema.state.confidence,
            self.turn_count,
        )

        # 9. Predict next focus (consumed by step 4 on next turn)
        self.schema.predict_next_focus(self.context_items)

        # 10. Compute response strategy — replaces generate_claims()
        strategy = self.schema.compute_response_strategy(
            user_input, self.user_model.summary(), shifting
        )

        # Add social adaptation
        adaptation = self.user_model.get_response_adaptation()
        if adaptation:
            strategy["user_adaptation"] = adaptation

        # 11. Build system prompt — no awareness block
        system_prompt = self._build_system_prompt(strategy)

        # 12. Build user message — include selected context if any
        if selected:
            context_block = "\n".join(f"- {item}" for item in selected)
            full_user = (
                f"Relevant context:\n{context_block}\n\nUser: {user_input}"
            )
        else:
            full_user = user_input

        # 13. Append user turn to conversation history
        self.messages.append({"role": "user", "content": full_user})

        # Window conversation history
        max_msgs = self.max_history_turns * 2
        if len(self.messages) > max_msgs:
            self.messages = self.messages[-max_msgs:]

        # 14. Call the LLM
        response = self.llm.generate(system_prompt, self.messages)

        # 15. Append assistant turn
        self.messages.append({"role": "assistant", "content": response})

        # 16. Store user input and LLM response as context items
        self._add_context(user_input)
        self._add_context(response[:500])

        return response

    def _build_system_prompt(self, strategy: dict) -> str:
        """Construct system prompt from strategy dict."""
        parts = ["You are a helpful assistant."]
        if strategy.get("topic_instruction"):
            parts.append(strategy["topic_instruction"])
        if strategy.get("user_adaptation"):
            parts.append(strategy["user_adaptation"])
        if strategy.get("framing"):
            parts.append(strategy["framing"])
        return " ".join(parts)

    def _add_context(self, item: str) -> None:
        """Add a context item, respecting the max_context_items limit."""
        self.context_items.append(item)
        if len(self.context_items) > self.max_context_items:
            self.context_items = self.context_items[-self.max_context_items :]

    def prune_context_by_schema(self) -> None:
        """Remove context items that are no longer relevant to current focus."""
        if not self.schema.state.focus_target or not self.context_items:
            return

        threshold = 0.1
        pruned = [
            item
            for item in self.context_items
            if token_overlap(item, self.schema.state.focus_target) >= threshold
        ]
        if len(pruned) < 5:
            pruned = self.context_items[-5:]
        self.context_items = pruned

    def save_state(self, filepath: str | Path) -> None:
        """Save schema state and context to a file."""
        import json

        data = {
            "schema": self.schema.to_dict(),
            "context_items": self.context_items,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load_state(self, filepath: str | Path) -> None:
        """Load schema state and context from a file."""
        import json

        with open(filepath, "r") as f:
            data = json.load(f)

        config = data["schema"].get("config", {})
        self.schema = AttentionSchema(**config)
        self.schema.state.focus_target = data["schema"]["focus_target"]
        self.schema.state.confidence = data["schema"]["confidence"]
        self.schema.state.reason = data["schema"]["reason"]
        self.schema.state.history = data["schema"].get("history", [])
        self.schema.schema_inertia = data["schema"].get(
            "schema_inertia", config.get("schema_inertia", 0.0)
        )
        self.schema.prediction = data["schema"].get("prediction")
        self.schema.prediction_error_history = data["schema"].get(
            "prediction_error_history", []
        )

        self.context_items = data.get("context_items", [])

    def loop(self) -> None:
        """Interactive REPL with persistence support."""
        print("Attention Schema Agent (type 'quit', 'save', or 'load' to control)\n")
        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit"):
                print("Goodbye!")
                break

            if user_input.lower() == "save":
                self.save_state("schema_state.json")
                print("Schema state saved to schema_state.json")
                continue

            if user_input.lower() == "load":
                try:
                    self.load_state("schema_state.json")
                    print("Schema state loaded from schema_state.json")
                except FileNotFoundError:
                    print("No saved state found.")
                continue

            response = self.run(user_input)
            print(f"\nAgent: {response}")
            print(f"  [schema: {self.schema.summary()}]")
            print(f"  [context items: {len(self.context_items)}]\n")
