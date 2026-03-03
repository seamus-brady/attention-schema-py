from attention_schema.attention import AttentionTarget
from attention_schema.dissociation import DissociationTracker, DissociationReport


class TestDissociationTracker:
    def setup_method(self):
        self.tracker = DissociationTracker()

    def test_no_dissociation_when_aligned(self):
        """When schema focus matches top ground truth target, gap should be low."""
        ground_truth = [
            AttentionTarget(content="python programming", activation=0.9, source="competition"),
            AttentionTarget(content="cake recipe", activation=0.1, source="competition"),
        ]
        report = self.tracker.measure(ground_truth, "python programming", 0.9, turn=1)
        assert report.gap_score < 0.5
        assert len(report.attended_not_aware) == 0

    def test_attended_not_aware(self):
        """High-activation targets not in schema focus should be reported."""
        ground_truth = [
            AttentionTarget(content="python programming", activation=0.8, source="competition"),
            AttentionTarget(content="machine learning basics", activation=0.7, source="competition"),
        ]
        # Schema only knows about "python programming"
        report = self.tracker.measure(ground_truth, "python programming", 0.8, turn=1)
        assert "machine learning basics" in report.attended_not_aware

    def test_aware_not_attended(self):
        """Schema focus not matching any high-activation target should be reported."""
        ground_truth = [
            AttentionTarget(content="cake recipe", activation=0.5, source="competition"),
        ]
        # Schema thinks we're focused on "python" but mechanism is on "cake"
        report = self.tracker.measure(ground_truth, "python programming", 0.8, turn=1)
        assert "python programming" in report.aware_not_attended

    def test_gap_score_increases_with_dissociation(self):
        """More dissociated items should produce a higher gap score."""
        aligned_truth = [
            AttentionTarget(content="python", activation=0.8, source="competition"),
        ]
        dissociated_truth = [
            AttentionTarget(content="cake", activation=0.8, source="competition"),
            AttentionTarget(content="rome", activation=0.7, source="competition"),
        ]
        aligned = self.tracker.measure(aligned_truth, "python", 0.8, turn=1)
        dissociated = self.tracker.measure(dissociated_truth, "python", 0.8, turn=2)
        assert dissociated.gap_score > aligned.gap_score

    def test_history_accumulates(self):
        ground_truth = [
            AttentionTarget(content="python", activation=0.5, source="competition"),
        ]
        self.tracker.measure(ground_truth, "python", 0.8, turn=1)
        self.tracker.measure(ground_truth, "python", 0.8, turn=2)
        assert len(self.tracker.history) == 2

    def test_empty_ground_truth(self):
        report = self.tracker.measure([], "python", 0.8, turn=1)
        assert report.gap_score >= 0.0
        assert len(report.attended_not_aware) == 0

    def test_custom_awareness_threshold(self):
        """Only targets above the threshold should count as 'attended'."""
        tracker = DissociationTracker(awareness_threshold=0.6)
        ground_truth = [
            AttentionTarget(content="python", activation=0.5, source="competition"),
            AttentionTarget(content="cake", activation=0.7, source="competition"),
        ]
        report = tracker.measure(ground_truth, "other topic", 0.8, turn=1)
        # python (0.5) is below threshold (0.6), shouldn't appear as attended
        assert "python" not in report.attended_not_aware
        # cake (0.7) is above threshold, should appear
        assert "cake" in report.attended_not_aware
