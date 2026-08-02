"""Tests for predict.py helper functions."""

import pytest

from predict import compute_confidence


class TestComputeConfidence:
    """Confidence label should reflect both probability clarity and signal strength."""

    def test_low_probability_no_signal_is_not_high(self):
        """A 17% 24h probability with no evidence must not be labeled high."""
        assert compute_confidence(
            prob_24h=0.17,
            has_history=True,
            llm_confidence=0.0,
            evidence_score=0.0,
        ) in ("low", "medium")

    def test_low_probability_no_signal_without_history_is_low(self):
        """No history, no signal, low probability -> low confidence."""
        assert compute_confidence(
            prob_24h=0.15,
            has_history=False,
            llm_confidence=0.0,
            evidence_score=0.0,
        ) == "low"

    def test_high_probability_with_strong_evidence_is_high(self):
        """High probability backed by strong evidence -> high confidence."""
        assert compute_confidence(
            prob_24h=0.85,
            has_history=True,
            llm_confidence=0.9,
            evidence_score=0.8,
        ) == "high"

    def test_medium_probability_with_moderate_evidence_is_medium(self):
        """Elevated probability with moderate evidence should be medium."""
        assert compute_confidence(
            prob_24h=0.55,
            has_history=True,
            llm_confidence=0.6,
            evidence_score=0.4,
        ) == "medium"

    def test_near_fifty_without_evidence_is_low(self):
        """A coin-flip probability with weak evidence is not confident."""
        assert compute_confidence(
            prob_24h=0.45,
            has_history=True,
            llm_confidence=0.5,
            evidence_score=0.2,
        ) == "low"

    def test_very_high_probability_without_evidence_is_medium_or_high(self):
        """Extreme probability close to 1 is still clear even with weak evidence."""
        label = compute_confidence(
            prob_24h=0.98,
            has_history=True,
            llm_confidence=0.2,
            evidence_score=0.1,
        )
        assert label in ("medium", "high")

    @pytest.mark.parametrize(
        "prob_24h, evidence_score, expected",
        [
            (0.05, 0.0, "low"),
            (0.25, 0.1, "low"),
            (0.65, 0.2, "medium"),
            (0.75, 0.6, "high"),
        ],
    )
    def test_confidence_levels(self, prob_24h, evidence_score, expected):
        """Sanity-check confidence label across probability/evidence combinations."""
        label = compute_confidence(
            prob_24h=prob_24h,
            has_history=True,
            llm_confidence=0.5,
            evidence_score=evidence_score,
        )
        assert label == expected
