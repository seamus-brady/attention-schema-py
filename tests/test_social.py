from attention_schema.social import UserAttentionModel


class TestUserAttentionModel:
    def setup_method(self):
        self.model = UserAttentionModel()

    def test_initial_state(self):
        assert self.model.schema.state.focus_target == ""
        summary = self.model.summary()
        assert "No model" in summary

    def test_update_from_input(self):
        self.model.update_from_input("Tell me about Python programming", [])
        assert "Python" in self.model.schema.state.focus_target
        assert self.model.schema.state.confidence > 0

    def test_confidence_scales_with_specificity(self):
        """Longer, more specific inputs should produce higher confidence."""
        short_model = UserAttentionModel()
        long_model = UserAttentionModel()
        short_model.update_from_input("hi", [])
        long_model.update_from_input("Tell me about advanced Python decorators please", [])
        assert long_model.schema.state.confidence > short_model.schema.state.confidence

    def test_summary_after_input(self):
        self.model.update_from_input("quantum physics experiments", [])
        summary = self.model.summary()
        assert "quantum physics" in summary.lower()
        assert "confidence" in summary.lower()

    def test_predict_user_interest(self):
        self.model.update_from_input("Python programming basics", [])
        context = ["Python advanced topics", "cake recipe", "history of Rome"]
        prediction = self.model.predict_user_interest(context)
        assert isinstance(prediction, str)
        assert len(prediction) > 0

    def test_tracks_topic_changes(self):
        """User model should track when the user shifts topics."""
        self.model.update_from_input("Python programming", [])
        first_focus = self.model.schema.state.focus_target

        self.model.update_from_input("chocolate cake recipe", [])
        second_focus = self.model.schema.state.focus_target

        assert first_focus != second_focus
        assert "chocolate" in second_focus.lower()

    def test_low_inertia_is_responsive(self):
        """User model with low inertia should quickly adopt new focus."""
        model = UserAttentionModel(schema_inertia=0.1)
        model.update_from_input("Python programming", [])
        model.update_from_input("cooking recipes for dinner tonight", [])
        assert "cooking" in model.schema.state.focus_target.lower()
