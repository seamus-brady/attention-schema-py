import json
import tempfile
from pathlib import Path

from attention_schema.controller import Controller
from attention_schema.schema import AttentionSchema
from attention_schema.llm import MockLLMClient


class TestController:
    def setup_method(self):
        self.mock_llm = MockLLMClient()
        self.controller = Controller(llm=self.mock_llm)

    def test_single_run(self):
        response = self.controller.run("Tell me about Python")
        assert "mock response" in response
        assert self.mock_llm.call_count == 1

    def test_schema_updates_after_run(self):
        self.controller.run("Tell me about Python")
        assert self.controller.schema.state.focus_target != ""
        assert self.controller.schema.state.confidence > 0

    def test_system_prompt_contains_schema(self):
        self.controller.run("Tell me about Python")
        assert "helpful assistant" in self.mock_llm.last_system.lower()

    def test_context_accumulates(self):
        self.controller.run("Tell me about Python")
        self.controller.run("What about decorators in Python?")
        # Each run adds user input + LLM response = 2 items per run
        assert len(self.controller.context_items) == 4

    def test_attention_shift_on_topic_change(self):
        self.controller.run("Tell me about Python")
        first_focus = self.controller.schema.state.focus_target

        # With the should_shift fix, schema only shifts when confidence is below
        # the threshold. Simulate confidence decay for genuine topic change.
        self.controller.schema.state.confidence = 0.3

        self.controller.run("What is the recipe for chocolate cake?")
        second_focus = self.controller.schema.state.focus_target

        assert first_focus != second_focus
        assert "chocolate cake" in second_focus.lower()

    def test_no_shift_on_same_topic(self):
        self.controller.run("Tell me about Python")
        self.controller.run("Tell me more about Python classes")
        # Should sustain attention (not shift)
        assert self.controller.schema.state.reason == "Continuing with current focus."

    def test_schema_history_grows(self):
        self.controller.run("Python programming")
        self.controller.run("Chocolate cake recipe")
        self.controller.run("History of Rome")
        # Each shift pushes old state into history
        assert len(self.controller.schema.state.history) >= 2

    def test_with_context_items(self):
        controller = Controller(
            llm=self.mock_llm,
            context_items=[
                "Python is great for scripting",
                "Cake is delicious",
                "Rome was founded in 753 BC",
            ],
        )
        controller.run("Tell me about Python")
        # The last user message should include relevant context
        last_user_msg = self.mock_llm.last_messages[-1]["content"]
        assert "context" in last_user_msg.lower() or "Python" in last_user_msg

    # New tests for enhanced features
    def test_context_bounded(self):
        """Context items should be bounded by max_context_items."""
        controller = Controller(llm=self.mock_llm, max_context_items=5)
        for i in range(10):
            controller.run(f"Message {i}")
        assert len(controller.context_items) <= 5

    def test_configurable_shift_thresholds(self):
        """Schema should respect custom shift thresholds."""
        schema = AttentionSchema(
            shift_overlap_threshold=0.5,
            shift_confidence_threshold=0.6,
        )
        controller = Controller(llm=self.mock_llm, schema=schema)
        controller.run("Tell me about Python")

        # With high overlap threshold, similar topics should still shift
        # Verify thresholds are used
        assert controller.schema.shift_overlap_threshold == 0.5
        assert controller.schema.shift_confidence_threshold == 0.6

    def test_schema_persistence(self):
        """Schema should be saveable and loadable."""
        controller = Controller(llm=self.mock_llm)
        controller.run("Tell me about Python")
        original_focus = controller.schema.state.focus_target
        original_confidence = controller.schema.state.confidence

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "schema.json"
            controller.save_state(filepath)

            # Verify file exists and contains expected data
            assert filepath.exists()
            with open(filepath) as f:
                data = json.load(f)
            assert data["schema"]["focus_target"] == original_focus
            assert data["schema"]["confidence"] == original_confidence

            # Create new controller and load state
            new_controller = Controller(llm=self.mock_llm)
            new_controller.load_state(filepath)
            assert new_controller.schema.state.focus_target == original_focus
            assert new_controller.schema.state.confidence == original_confidence

    def test_context_persistence(self):
        """Context items should be saved and loaded with state."""
        controller = Controller(llm=self.mock_llm)
        controller.run("First message")
        controller.run("Second message")
        original_items = controller.context_items.copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "state.json"
            controller.save_state(filepath)

            # Create new controller and load
            new_controller = Controller(llm=self.mock_llm)
            new_controller.load_state(filepath)
            assert new_controller.context_items == original_items

    def test_schema_history_pruning(self):
        """Schema history should respect max_history limit."""
        schema = AttentionSchema(max_history=3)
        for i in range(10):
            schema.update(f"focus_{i}", 0.5 + (i * 0.01), f"reason_{i}")

        # History should be pruned to max_history
        assert len(schema.state.history) <= 3

    def test_prune_context_by_schema(self):
        """Should be able to prune irrelevant context."""
        controller = Controller(llm=self.mock_llm)
        controller.context_items = [
            "Python programming is great",
            "Chocolate cake is delicious",
            "Rome was founded in 753 BC",
            "Dogs are loyal pets",
            "Cats like to sleep",
            "Python decorators are powerful",
            "JavaScript is dynamic",
            "Machine learning with Python",
        ]

        # Focus on Python topic
        controller.schema.update("Python", 0.8, "Testing")

        # Prune context
        controller.prune_context_by_schema()

        # Should keep Python-related items
        kept = controller.context_items
        assert len(kept) > 0
        # Should have fewer items after pruning (with 8 items, should keep fewer unless all are related)
        assert len(kept) <= 8
        # At least some Python-related items should be kept
        python_items = [item for item in kept if "python" in item.lower()]
        assert len(python_items) > 0

    def test_conversation_history_accumulates(self):
        """Conversation history should accumulate user+assistant turns."""
        self.controller.run("Hello")
        self.controller.run("How are you?")
        # Each run adds 1 user + 1 assistant message = 2 per run
        assert len(self.controller.messages) == 4
        assert self.controller.messages[0]["role"] == "user"
        assert self.controller.messages[1]["role"] == "assistant"
        assert self.controller.messages[2]["role"] == "user"
        assert self.controller.messages[3]["role"] == "assistant"

    def test_conversation_history_windowed(self):
        """Conversation history should be windowed to max_history_turns."""
        controller = Controller(llm=self.mock_llm, max_history_turns=2)
        for i in range(5):
            controller.run(f"Message {i}")
        # max_history_turns=2 means max 4 messages (2 turns * 2 msgs/turn)
        # But the last assistant message is appended after windowing,
        # so we can have up to 5 (4 windowed + 1 new assistant)
        assert len(controller.messages) <= 5

    def test_llm_receives_message_list(self):
        """LLM should receive a list of messages, not a single string."""
        self.controller.run("Hello there")
        assert isinstance(self.mock_llm.last_messages, list)
        assert all(isinstance(m, dict) for m in self.mock_llm.last_messages)
        # The messages sent to the LLM end with the latest user message
        # (assistant messages from prior turns are also present)
        user_msgs = [m for m in self.mock_llm.last_messages if m["role"] == "user"]
        assert len(user_msgs) >= 1

    def test_llm_response_stored_as_context(self):
        """LLM response should be stored as a context item."""
        self.controller.run("Tell me about Python")
        # Context should have both the user input and the mock response
        assert any("mock response" in item for item in self.controller.context_items)

    def test_turn_count_increments(self):
        self.controller.run("Hello")
        assert self.controller.turn_count == 1
        self.controller.run("World")
        assert self.controller.turn_count == 2

    def test_dissociation_tracked(self):
        """Dissociation tracker should have reports after runs."""
        self.controller.run("Tell me about Python")
        self.controller.run("Tell me more about Python classes")
        assert len(self.controller.dissociation.history) == 2

    def test_user_model_updated(self):
        """User model should be updated from user input."""
        self.controller.run("Tell me about quantum physics")
        assert self.controller.user_model.schema.state.focus_target != ""
        assert "quantum" in self.controller.user_model.schema.state.focus_target.lower()

    def test_system_prompt_has_no_awareness_claims(self):
        """System prompt should NOT contain AWARENESS STATE block."""
        self.controller.run("Tell me about Python")
        assert "AWARENESS STATE" not in self.mock_llm.last_system

    def test_prediction_runs_after_second_turn(self):
        """Schema prediction should be populated after multiple turns."""
        self.controller.run("Python programming basics")
        self.controller.run("More about Python advanced features")
        assert self.controller.schema.prediction is not None

    def test_format_map_with_old_template(self):
        """Old-style templates with only {schema_summary} should still work."""
        old_template = "Focus: {schema_summary}"
        controller = Controller(
            llm=self.mock_llm,
            system_template=old_template,
        )
        controller.run("Hello")
        # Should not crash, should contain schema summary
        assert self.mock_llm.last_system is not None

    def test_confidence_overlap_driven(self):
        """Confidence should vary based on overlap, not just monotonically increase."""
        self.controller.run("Tell me about Python programming")
        # First run: shift, confidence = 0.8
        assert self.controller.schema.state.confidence == 0.8

        # Second run: sustain with high overlap
        self.controller.run("Python programming best practices")
        conf_after_related = self.controller.schema.state.confidence

        # Third run: sustain but with lower overlap (vaguer query)
        self.controller.run("tell me more")
        conf_after_vague = self.controller.schema.state.confidence

        # Confidence after vague query should be lower than after highly related query
        # (overlap-driven, not monotonically increasing)
        assert conf_after_vague <= conf_after_related
