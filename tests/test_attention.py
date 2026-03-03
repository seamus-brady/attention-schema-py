from attention_schema.attention import AttentionMechanism


class TestAttentionMechanism:
    def setup_method(self):
        self.attn = AttentionMechanism()
        self.items = [
            "Python is a popular programming language",
            "Chocolate cake recipe with frosting",
            "Machine learning with Python and TensorFlow",
            "History of the Roman Empire",
        ]

    def test_score_relevance_length(self):
        scores = self.attn.score_relevance(self.items, "python")
        assert len(scores) == len(self.items)

    def test_score_relevance_values(self):
        scores = self.attn.score_relevance(self.items, "python programming")
        # Items 0 and 2 mention python, should score higher than 1 and 3
        assert scores[0] > scores[1]
        assert scores[0] > scores[3]
        assert scores[2] > scores[1]

    def test_select_top_k(self):
        selected = self.attn.select(self.items, "python programming", top_k=2)
        assert len(selected) == 2
        assert any("Python" in s for s in selected)

    def test_select_preserves_order(self):
        selected = self.attn.select(self.items, "python", top_k=2)
        # The original order among selected items should be preserved
        original_indices = [self.items.index(s) for s in selected]
        assert original_indices == sorted(original_indices)

    def test_empty_query(self):
        scores = self.attn.score_relevance(self.items, "")
        assert all(s == 0.0 for s in scores)

    def test_empty_items(self):
        scores = self.attn.score_relevance([], "python")
        assert scores == []

    # New tests for enhanced features
    def test_recency_weight_enabled(self):
        """With recency weighting, recent items should get higher scores."""
        attn = AttentionMechanism(use_recency_weight=True)
        items = [
            "old python discussion",
            "medium python discussion",
            "recent python discussion",
        ]

        selected = attn.select(items, "python", top_k=1, use_recency=True)
        # Should prefer the most recent item
        assert selected[0] == items[-1]

    def test_recency_weight_disabled(self):
        """With recency weighting disabled, should use pure relevance."""
        attn = AttentionMechanism(use_recency_weight=False)
        items = [
            "python is great",
            "chocolate cake",
            "python programming",
        ]

        selected = attn.select(items, "python", top_k=1, use_recency=False)
        # With no recency weight, both python items should be equally scored
        # but we should get a consistent selection
        assert "python" in selected[0].lower()

    def test_custom_weights(self):
        """score_relevance should apply custom weights correctly."""
        attn = AttentionMechanism()
        items = [
            "python topic",
            "python topic",
            "python topic",
        ]

        # All items are identical, but weights should affect scores
        weights = [0.1, 0.5, 1.0]
        scores = attn.score_relevance(items, "python", weights=weights)

        # Scores should be proportional to weights (later items higher)
        assert scores[2] > scores[1] > scores[0]

    def test_select_with_recency_boost(self):
        """Recent items should be prioritized when recency is enabled."""
        attn = AttentionMechanism(use_recency_weight=True)
        items = [
            "Python basics discussed",
            "Python advanced topics",
            "Python best practices",
        ]

        # All are equally relevant for "python", but recency should break ties
        selected = attn.select(items, "python", top_k=1, use_recency=True)
        assert selected[0] == items[-1]  # Most recent should win

    def test_select_less_than_top_k(self):
        """Should handle when fewer items exist than top_k."""
        attn = AttentionMechanism()
        items = ["item 1", "item 2"]
        selected = attn.select(items, "item", top_k=5)
        assert len(selected) == 2

    def test_identical_relevance_maintains_recency(self):
        """When relevance is equal, recency should dominate."""
        attn = AttentionMechanism(use_recency_weight=True)
        items = [
            "same query match",
            "same query match",
            "same query match",
        ]

        selected = attn.select(items, "match", top_k=1, use_recency=True)
        # Should pick the last item (most recent) due to recency weight
        assert selected[0] == items[-1]

    def test_recency_weight_floor(self):
        """First item should have weight > 0 (floor of 0.2)."""
        attn = AttentionMechanism(use_recency_weight=True)
        items = [
            "python topic first",
            "unrelated stuff here",
            "unrelated other thing",
        ]
        # With the floor, the first item still has weight 0.2 (not 0)
        # so it should still get a non-zero score
        scores = attn.score_relevance(items, "python")
        # Without recency weights, item 0 should be relevant
        assert scores[0] > 0

        # With recency weights applied via select, first item should still be selectable
        selected = attn.select(items, "python", top_k=1, use_recency=True)
        assert "python" in selected[0].lower()

    def test_focus_bias_boosts_on_topic_items(self):
        """Items overlapping with focus_bias should get a scoring boost."""
        attn = AttentionMechanism(use_recency_weight=False)
        items = [
            "Python decorators are useful",
            "JavaScript frameworks comparison",
            "Python machine learning basics",
        ]

        # Without focus_bias, query "frameworks" should prefer item 1
        selected_no_bias = attn.select(
            items, "frameworks", top_k=1, use_recency=False
        )
        assert "JavaScript" in selected_no_bias[0]

        # With focus_bias="Python programming", Python items get a boost
        selected_with_bias = attn.select(
            items, "frameworks", top_k=2, use_recency=False, focus_bias="Python programming"
        )
        # At least one Python item should appear due to the bias
        python_items = [s for s in selected_with_bias if "Python" in s]
        assert len(python_items) >= 1
