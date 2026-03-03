from attention_schema.attention import AttentionMechanism, AttentionTarget


class TestCompetitiveAttention:
    def setup_method(self):
        self.attn = AttentionMechanism()
        self.items = [
            "Python is a popular programming language",
            "Chocolate cake recipe with frosting",
            "Machine learning with Python and TensorFlow",
            "History of the Roman Empire",
        ]

    def test_compete_returns_attention_targets(self):
        targets = self.attn.compete(self.items, "python programming")
        assert len(targets) == len(self.items)
        assert all(isinstance(t, AttentionTarget) for t in targets)

    def test_compete_winner_gets_boost(self):
        """Top scorer should have higher activation due to winner-take-most."""
        targets = self.attn.compete(self.items, "python programming")
        activations = [(t.content, t.activation) for t in targets]
        # Sort by activation
        activations.sort(key=lambda x: x[1], reverse=True)
        # Winner (Python item) should have highest activation
        assert "Python" in activations[0][0] or "python" in activations[0][0].lower()

    def test_lateral_inhibition_suppresses_runners_up(self):
        """Runners-up should be suppressed relative to the winner."""
        attn = AttentionMechanism(lateral_inhibition=0.3)
        targets = attn.compete(self.items, "python programming")
        # Find winner and a runner-up with nonzero score
        winner = max(targets, key=lambda t: t.activation)
        runners = [t for t in targets if t.activation > 0 and t is not winner]
        if runners:
            # Winner's activation should be meaningfully higher
            best_runner = max(runners, key=lambda t: t.activation)
            assert winner.activation > best_runner.activation

    def test_lateral_inhibition_strength_is_configurable(self):
        """Higher lateral inhibition should create a bigger gap."""
        attn_low = AttentionMechanism(lateral_inhibition=0.1)
        attn_high = AttentionMechanism(lateral_inhibition=0.5)
        items = [
            "python basics tutorial",
            "python advanced features",
        ]
        targets_low = attn_low.compete(items, "python")
        targets_high = attn_high.compete(items, "python")

        # With higher inhibition, the gap between winner and runner should be larger
        gap_low = abs(targets_low[0].activation - targets_low[1].activation)
        gap_high = abs(targets_high[0].activation - targets_high[1].activation)
        assert gap_high >= gap_low

    def test_get_ground_truth_returns_pool(self):
        self.attn.compete(self.items, "python")
        truth = self.attn.get_ground_truth()
        assert len(truth) == len(self.items)
        assert all(isinstance(t, AttentionTarget) for t in truth)

    def test_compete_empty_items(self):
        targets = self.attn.compete([], "python")
        assert targets == []
        assert self.attn.get_ground_truth() == []

    def test_compete_empty_query(self):
        targets = self.attn.compete(self.items, "")
        assert len(targets) == len(self.items)
        assert all(t.activation == 0.0 for t in targets)

    def test_select_uses_compete_internally(self):
        """select() should populate the activation pool via compete()."""
        self.attn.select(self.items, "python", top_k=2)
        pool = self.attn.get_ground_truth()
        assert len(pool) == len(self.items)
        # At least one item should have nonzero activation
        assert any(t.activation > 0 for t in pool)

    def test_activation_capped_at_one(self):
        """Winner activation should never exceed 1.0."""
        attn = AttentionMechanism(lateral_inhibition=0.9)
        items = ["python python python", "unrelated stuff"]
        targets = attn.compete(items, "python")
        assert all(t.activation <= 1.0 for t in targets)
