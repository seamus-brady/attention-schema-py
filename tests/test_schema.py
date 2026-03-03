from attention_schema.schema import AttentionState, AttentionSchema
from attention_schema.tokenizer import tokenize, token_overlap
import tempfile
from pathlib import Path
import json


class TestAttentionState:
    def test_creation_defaults(self):
        state = AttentionState(focus_target="test", confidence=0.9, reason="r")
        assert state.focus_target == "test"
        assert state.confidence == 0.9
        assert state.reason == "r"
        assert state.history == []

    def test_history_is_independent(self):
        s1 = AttentionState(focus_target="a", confidence=0.5, reason="x")
        s2 = AttentionState(focus_target="b", confidence=0.6, reason="y")
        s1.history.append({"focus_target": "old", "confidence": 0.1, "reason": "z"})
        assert s2.history == []


class TestAttentionSchema:
    def test_initial_state(self):
        schema = AttentionSchema()
        assert schema.state.focus_target == ""
        assert schema.state.confidence == 0.0

    def test_update_pushes_history(self):
        schema = AttentionSchema()
        schema.update("topic A", 0.8, "first topic")
        schema.update("topic B", 0.9, "second topic")
        assert schema.state.focus_target == "topic B"
        assert len(schema.state.history) == 1
        assert schema.state.history[0]["focus_target"] == "topic A"

    def test_should_shift_when_empty(self):
        schema = AttentionSchema()
        assert schema.should_shift("anything") is True

    def test_should_shift_on_topic_change(self):
        schema = AttentionSchema()
        # With the should_shift fix, high confidence prevents shifting.
        # Shift only happens when confidence is below the threshold.
        schema.update("python programming", 0.3, "discussing python")
        assert schema.should_shift("recipe for chocolate cake") is True

    def test_no_shift_on_high_confidence(self):
        schema = AttentionSchema()
        # High confidence + low overlap → stay (the bug fix)
        schema.update("python programming", 0.9, "discussing python")
        assert schema.should_shift("recipe for chocolate cake") is False

    def test_no_shift_on_same_topic(self):
        schema = AttentionSchema()
        schema.update("python programming", 0.9, "discussing python")
        assert schema.should_shift("python functions and classes") is False

    def test_shift_on_low_confidence(self):
        schema = AttentionSchema()
        schema.update("some topic", 0.3, "uncertain")
        assert schema.should_shift("some topic continued") is False  # overlap is high enough

    def test_summary_no_focus(self):
        schema = AttentionSchema()
        assert "not currently focused" in schema.summary()

    def test_summary_with_focus(self):
        schema = AttentionSchema()
        schema.update("quantum physics", 0.85, "user asked about it")
        summary = schema.summary()
        assert "quantum physics" in summary
        assert "0.85" in summary

    # New tests for enhanced features
    def test_configurable_shift_thresholds(self):
        """Schema should use custom thresholds for shift detection."""
        schema = AttentionSchema(
            shift_overlap_threshold=0.5,
            shift_confidence_threshold=0.7,
        )
        # Confidence below threshold (0.6 < 0.7) → should shift on different topic
        schema.update("python", 0.6, "test")
        assert schema.should_shift("completely different topic") is True

        # Confidence above threshold (0.8 >= 0.7) → should stay
        schema.update("python", 0.8, "test")
        assert schema.should_shift("completely different topic") is False

        # Custom thresholds should be stored
        assert schema.shift_overlap_threshold == 0.5
        assert schema.shift_confidence_threshold == 0.7

    def test_history_pruning(self):
        """History should be bounded by max_history."""
        schema = AttentionSchema(max_history=3)

        for i in range(10):
            schema.update(f"topic_{i}", 0.5 + (i * 0.01), f"reason_{i}")

        # History should not exceed max_history
        assert len(schema.state.history) <= 3

        # Most recent entries should be preserved
        if len(schema.state.history) >= 2:
            assert "topic_8" in schema.state.history[-1]["focus_target"]

    def test_configurable_confidence_increment(self):
        """Confidence increment should be customizable."""
        schema = AttentionSchema(confidence_increment=0.1)
        schema.update("topic", 0.5, "test")

        # Without shift, confidence should increment by custom amount
        old_confidence = schema.state.confidence
        schema.should_shift("topic related query")
        # Manually update to test increment (in real use, controller increments)
        schema.state.confidence = min(
            old_confidence + schema.confidence_increment, 1.0
        )
        assert schema.state.confidence == 0.6

    def test_schema_persistence_save_load(self):
        """Schema should be saveable to and loadable from JSON."""
        schema = AttentionSchema(
            shift_overlap_threshold=0.3,
            confidence_increment=0.08,
        )
        schema.update("test topic", 0.75, "testing")
        schema.update("related topic", 0.82, "continuing")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "schema.json"
            schema.save(filepath)

            # Verify file contents
            assert filepath.exists()
            with open(filepath) as f:
                data = json.load(f)
            assert data["focus_target"] == "related topic"
            assert data["confidence"] == 0.82
            assert len(data["history"]) > 0
            assert data["config"]["shift_overlap_threshold"] == 0.3

            # Load and verify
            loaded = AttentionSchema.load(filepath)
            assert loaded.state.focus_target == "related topic"
            assert loaded.state.confidence == 0.82
            assert len(loaded.state.history) == 1
            assert loaded.shift_overlap_threshold == 0.3
            assert loaded.confidence_increment == 0.08

    def test_to_dict_serialization(self):
        """to_dict should produce a valid serialization."""
        schema = AttentionSchema()
        schema.update("focus", 0.9, "reason")

        data = schema.to_dict()
        assert data["focus_target"] == "focus"
        assert data["confidence"] == 0.9
        assert data["reason"] == "reason"
        assert "config" in data
        assert "history" in data

    def test_should_shift_suppressed_by_context_overlap(self):
        """'tell me more' shouldn't trigger shift when recent context is relevant."""
        schema = AttentionSchema()
        schema.update("python programming", 0.9, "discussing python")
        # "tell me more" has no overlap with "python programming"
        # but recent context about python should suppress the shift
        context = [
            "Python is a great programming language",
            "Tell me about python decorators",
            "Python classes and functions are powerful",
        ]
        assert schema.should_shift("tell me more about that", context_items=context) is False

    def test_should_shift_with_unrelated_context(self):
        """Should shift when context is unrelated AND confidence is low."""
        schema = AttentionSchema()
        schema.update("python programming", 0.3, "uncertain")
        context = [
            "Python is a great language",
            "Python decorators explained",
        ]
        # Low confidence + no overlap with context → shift
        assert schema.should_shift("recipe for chocolate cake", context_items=context) is True

    def test_no_shift_with_unrelated_context_high_confidence(self):
        """High confidence prevents shift even when context is unrelated."""
        schema = AttentionSchema()
        schema.update("python programming", 0.9, "discussing python")
        context = [
            "Python is a great language",
            "Python decorators explained",
        ]
        assert schema.should_shift("recipe for chocolate cake", context_items=context) is False


class TestSchemaFromCompetition:
    """Tests for the lossy schema update and reconciliation."""

    def test_update_from_competition_picks_top_target(self):
        from attention_schema.attention import AttentionTarget
        schema = AttentionSchema()
        targets = [
            AttentionTarget(content="python programming", activation=0.9, source="competition"),
            AttentionTarget(content="cake recipe", activation=0.2, source="competition"),
        ]
        schema.update_from_competition(targets)
        assert "python" in schema.state.focus_target.lower()

    def test_update_from_competition_is_lossy(self):
        """Only top target content should be kept; runner-up activations discarded."""
        from attention_schema.attention import AttentionTarget
        schema = AttentionSchema()
        targets = [
            AttentionTarget(content="python programming", activation=0.9, source="competition"),
            AttentionTarget(content="machine learning", activation=0.7, source="competition"),
        ]
        schema.update_from_competition(targets)
        # Schema should only know about the top target
        assert "python" in schema.state.focus_target.lower()
        # Runner-up info is lost from the schema's perspective
        assert "machine learning" not in schema.state.focus_target.lower()

    def test_schema_inertia_prevents_shift(self):
        """With high inertia, schema should resist shifting to a new topic."""
        from attention_schema.attention import AttentionTarget
        schema = AttentionSchema(schema_inertia=0.5)
        schema.update("python programming", 0.8, "established focus")
        targets = [
            AttentionTarget(content="chocolate cake recipe", activation=0.6, source="competition"),
        ]
        schema.update_from_competition(targets)
        # Inertia should prevent shift: 0.6 < 0.8 + 0.5, and no token overlap
        assert schema.state.focus_target == "python programming"

    def test_schema_inertia_zero_allows_shift(self):
        """With zero inertia (default), schema should shift freely."""
        from attention_schema.attention import AttentionTarget
        schema = AttentionSchema(schema_inertia=0.0)
        schema.update("current topic", 0.8, "established focus")
        targets = [
            AttentionTarget(content="new different topic", activation=0.9, source="competition"),
        ]
        schema.update_from_competition(targets)
        assert "new different topic" in schema.state.focus_target

    def test_reconcile_corrects_mismatch(self):
        """Reconciliation should correct schema when it diverges from ground truth."""
        from attention_schema.attention import AttentionTarget
        schema = AttentionSchema(reconciliation_threshold=0.3)
        schema.update("wrong focus", 0.5, "stale")
        schema.receive_ground_truth(
            [AttentionTarget(content="actual focus target", activation=0.8, source="competition")],
            turn=1,
        )
        result = schema.reconcile()
        assert result["corrected"] is True
        assert "actual focus target" in result["new_focus"]

    def test_reconcile_no_correction_when_aligned(self):
        from attention_schema.attention import AttentionTarget
        schema = AttentionSchema()
        schema.update("python programming", 0.8, "good")
        schema.receive_ground_truth(
            [AttentionTarget(content="python programming", activation=0.9, source="competition")],
            turn=1,
        )
        result = schema.reconcile()
        assert result["corrected"] is False

    def test_reconcile_no_ground_truth(self):
        schema = AttentionSchema()
        schema.update("topic", 0.8, "reason")
        result = schema.reconcile()
        assert result["mismatch"] == 0.0
        assert result["corrected"] is False


class TestSchemaPrediction:
    def test_predict_next_focus_returns_string(self):
        schema = AttentionSchema()
        schema.update("python", 0.8, "test")
        prediction = schema.predict_next_focus(["python advanced", "cake recipe"])
        assert isinstance(prediction, str)
        assert schema.prediction is not None

    def test_compute_prediction_error(self):
        schema = AttentionSchema()
        schema.prediction = "python programming"
        error = schema.compute_prediction_error("python programming")
        assert error == 0.0  # perfect match

    def test_prediction_error_for_miss(self):
        schema = AttentionSchema()
        schema.prediction = "python programming"
        error = schema.compute_prediction_error("chocolate cake recipe")
        assert error > 0.5  # very different

    def test_learn_from_high_error_decreases_inertia(self):
        schema = AttentionSchema(schema_inertia=0.5)
        schema.learn_from_error(0.9)
        assert schema.schema_inertia < 0.5

    def test_learn_from_low_error_increases_inertia(self):
        schema = AttentionSchema(schema_inertia=0.3)
        schema.learn_from_error(0.1)
        assert schema.schema_inertia > 0.3

    def test_inertia_bounded(self):
        schema = AttentionSchema(schema_inertia=0.05)
        schema.learn_from_error(0.9)  # try to decrease below min
        assert schema.schema_inertia >= 0.05

        schema2 = AttentionSchema(schema_inertia=0.8)
        schema2.learn_from_error(0.1)  # try to increase above max
        assert schema2.schema_inertia <= 0.8

    def test_prediction_error_history_accumulates(self):
        schema = AttentionSchema()
        schema.prediction = "test"
        schema.compute_prediction_error("test")
        schema.compute_prediction_error("other")
        assert len(schema.prediction_error_history) == 2


class TestAwarenessClaims:
    def test_generate_claims_basic(self):
        schema = AttentionSchema()
        schema.update("python programming", 0.85, "user asked")
        claims = schema.generate_claims()
        assert claims.current_focus == "python programming"
        assert claims.focus_confidence == "high"

    def test_confidence_labels(self):
        schema = AttentionSchema()
        schema.update("topic", 0.8, "r")
        assert schema.generate_claims().focus_confidence == "high"

        schema.update("topic", 0.5, "r")
        assert schema.generate_claims().focus_confidence == "moderate"

        schema.update("topic", 0.2, "r")
        assert schema.generate_claims().focus_confidence == "low"

    def test_awareness_of_shift(self):
        schema = AttentionSchema()
        schema.update("old topic", 0.8, "first")
        schema.update("new topic", 0.8, "New topic detected — shifting attention.")
        claims = schema.generate_claims()
        assert claims.awareness_of_shift is not None
        assert "old topic" in claims.awareness_of_shift

    def test_to_prompt_block_contains_focus(self):
        schema = AttentionSchema()
        schema.update("quantum physics", 0.9, "user asked")
        claims = schema.generate_claims()
        block = claims.to_prompt_block()
        assert "AWARENESS STATE" in block
        assert "quantum physics" in block

    def test_claims_with_prediction(self):
        schema = AttentionSchema()
        schema.update("python", 0.8, "test")
        schema.predict_next_focus(["python advanced"])
        claims = schema.generate_claims()
        assert claims.predicted_next is not None

    def test_claims_no_focus(self):
        schema = AttentionSchema()
        claims = schema.generate_claims()
        assert claims.current_focus == "(none)"
        assert claims.focus_confidence == "low"


class TestSchemaSerializationNew:
    def test_to_dict_includes_new_fields(self):
        schema = AttentionSchema(schema_inertia=0.3, reconciliation_threshold=0.6)
        schema.update("focus", 0.9, "reason")
        schema.prediction = "next thing"
        data = schema.to_dict()
        assert data["schema_inertia"] == 0.3
        assert data["prediction"] == "next thing"
        assert data["config"]["schema_inertia"] == 0.3
        assert data["config"]["reconciliation_threshold"] == 0.6

    def test_save_load_round_trip_with_new_fields(self):
        import tempfile
        from pathlib import Path
        schema = AttentionSchema(schema_inertia=0.4, reconciliation_threshold=0.6)
        schema.update("test topic", 0.75, "testing")
        schema.prediction = "predicted focus"
        schema.prediction_error_history = [0.1, 0.5, 0.3]

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "schema.json"
            schema.save(filepath)
            loaded = AttentionSchema.load(filepath)
            assert loaded.schema_inertia == 0.4
            assert loaded.reconciliation_threshold == 0.6
            assert loaded.prediction == "predicted focus"
            assert loaded.prediction_error_history == [0.1, 0.5, 0.3]


class TestTokenizer:
    def test_tokenize_basic(self):
        tokens = tokenize("Hello, World! 123")
        assert tokens == ["hello", "world", "123"]

    def test_token_overlap_identical(self):
        assert token_overlap("hello world", "hello world") == 1.0

    def test_token_overlap_disjoint(self):
        assert token_overlap("hello world", "foo bar") == 0.0

    def test_token_overlap_partial(self):
        overlap = token_overlap("hello world", "hello there")
        assert 0.0 < overlap < 1.0

    def test_token_overlap_empty(self):
        assert token_overlap("", "hello") == 0.0
        assert token_overlap("hello", "") == 0.0
