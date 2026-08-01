"""Tests for V2 Adaptive Bayesian Evidence Model"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from model.data_models import PredictionExplanation, PredictionFeatures, SignalScores, Tweet
from model.survival_model import (
    BASE_PROBABILITY,
    DEFAULT_RESET_INTERVAL_HOURS,
    MAX_EVIDENCE_MULTIPLIER,
    MAX_PROBABILITY_NO_SIGNAL,
    PREDICTION_HORIZONS,
    ResetPredictor,
    _aggregate_weighted_evidence,
    _base_probability,
    _bayesian_update,
    _evidence_multiplier,
    _probabilities,
    _sigmoid,
    build_features,
)


# ──────────────────────────────────────────────
# Helper function tests
# ──────────────────────────────────────────────

class TestSigmoid:
    """Tests for sigmoid function"""

    def test_zero(self):
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive(self):
        assert _sigmoid(100.0) == pytest.approx(1.0)

    def test_large_negative(self):
        assert _sigmoid(-100.0) == pytest.approx(0.0)


class TestBayesianHelpers:
    """Tests for V2 Bayesian helper functions"""

    def test_base_probability_time_pressure_monotonic(self):
        """Higher time pressure raises baseline probability but stays within cap"""
        for horizon in PREDICTION_HORIZONS:
            p_low = _base_probability(horizon, 0.0)
            p_mid = _base_probability(horizon, 0.5)
            p_high = _base_probability(horizon, 1.0)
            assert p_low < p_high
            assert p_mid <= MAX_PROBABILITY_NO_SIGNAL[horizon]
            assert p_high <= MAX_PROBABILITY_NO_SIGNAL[horizon]

    def test_evidence_multiplier_range(self):
        assert _evidence_multiplier(0.0) == pytest.approx(1.0)
        assert _evidence_multiplier(1.0) == pytest.approx(MAX_EVIDENCE_MULTIPLIER)

    def test_bayesian_update_increases_probability(self):
        prior = 0.2
        posterior = _bayesian_update(prior, 0.5)
        assert posterior > prior
        assert posterior < 1.0

    def test_probabilities_window_ordering(self):
        """5h <= 24h <= 48h"""
        prob = _probabilities(0.5, 0.0, PREDICTION_HORIZONS)
        assert prob["5h"] <= prob["24h"] <= prob["48h"]

    def test_probabilities_no_signal_caps(self):
        """Probability is capped when no signal is present"""
        prob = _probabilities(1.0, 0.0, PREDICTION_HORIZONS)
        assert prob["5h"] <= MAX_PROBABILITY_NO_SIGNAL[5]
        assert prob["24h"] <= MAX_PROBABILITY_NO_SIGNAL[24]
        assert prob["48h"] <= MAX_PROBABILITY_NO_SIGNAL[48]

    def test_probabilities_evidence_boost(self):
        """Evidence boosts probability"""
        base = _probabilities(0.5, 0.0, PREDICTION_HORIZONS)
        strong = _probabilities(0.5, 0.9, PREDICTION_HORIZONS)
        for h in PREDICTION_HORIZONS:
            assert strong[f"{h}h"] >= base[f"{h}h"]


# ──────────────────────────────────────────────
# Data model tests
# ──────────────────────────────────────────────

class TestPredictionFeatures:
    """Tests for PredictionFeatures model"""

    def test_create_with_all_fields(self):
        features = PredictionFeatures(
            hours_since_last_reset=20.0,
            average_reset_interval=24.0,
            tibo_signal=0.8,
            community_signal=0.3,
            release_signal=0.1,
            evidence_score=0.5,
        )
        assert features.hours_since_last_reset == 20.0
        assert features.evidence_score == 0.5

    def test_signal_out_of_range(self):
        with pytest.raises(ValidationError):
            PredictionFeatures(tibo_signal=1.5)
        with pytest.raises(ValidationError):
            PredictionFeatures(evidence_score=-0.1)


class TestPredictionExplanation:
    """Tests for PredictionExplanation model"""

    def test_create_valid(self):
        exp = PredictionExplanation(
            probability={"5h": 0.12, "24h": 0.45, "48h": 0.62},
            reasons=["reason1"],
            evidence_score=0.5,
            hazard_rate=0.05,
        )
        assert exp.evidence_score == 0.5
        assert "5h" in exp.probability


# ──────────────────────────────────────────────
# Evidence aggregation tests
# ──────────────────────────────────────────────

class TestEvidenceAggregation:
    """Tests for evidence aggregation and source weighting"""

    def test_no_signals_zero_evidence(self):
        result = _aggregate_weighted_evidence([], [])
        assert result["overall"] == 0.0
        assert result["tibo_signal"] == 0.0

    def test_tibo_confirmation_strong_evidence(self):
        now = datetime.now(timezone.utc)
        tweets = [
            Tweet(
                timestamp=now,
                author="tibo",
                text="Resetting limits now",
                source="tibo_rss",
                authority_score=1.0,
            )
        ]
        scores = [SignalScores(reset_confirmation=1.0, confidence=1.0)]
        result = _aggregate_weighted_evidence(tweets, scores, now=now)
        assert result["tibo"] > 0.7
        assert result["overall"] > 0.5

    def test_source_priority_tibo_over_community(self):
        now = datetime.now(timezone.utc)
        tibo_tweet = Tweet(
            timestamp=now, author="tibo", text="t", source="tibo_rss"
        )
        community_tweet = Tweet(
            timestamp=now, author="user", text="c", source="community_mock"
        )
        score = SignalScores(reset_confirmation=0.8, confidence=0.9)
        tibo_result = _aggregate_weighted_evidence([tibo_tweet], [score], now=now)
        community_result = _aggregate_weighted_evidence(
            [community_tweet], [score], now=now
        )
        assert tibo_result["overall"] > community_result["overall"]


# ──────────────────────────────────────────────
# build_features tests
# ──────────────────────────────────────────────

class TestBuildFeatures:
    """Tests for build_features helper"""

    def test_no_signals(self):
        features = build_features(20.0, 24.0, signal_scores=None, interval_count=100)
        assert features.hours_since_last_reset == 20.0
        assert features.evidence_score == 0.0
        assert features.tibo_signal == 0.0

    def test_with_signals(self):
        scores = [
            SignalScores(
                reset_intent=0.8,
                limit_complaint=0.6,
                official_change=0.2,
                reset_confirmation=0.7,
                confidence=0.9,
            ),
        ]
        features = build_features(10.0, 24.0, signal_scores=scores)
        assert features.evidence_score > 0.0
        assert features.tibo_signal > 0.0

    def test_no_history(self):
        features = build_features(None, None)
        assert features.hours_since_last_reset == pytest.approx(
            DEFAULT_RESET_INTERVAL_HOURS
        )
        assert features.evidence_score == 0.0


# ──────────────────────────────────────────────
# ResetPredictor core tests
# ──────────────────────────────────────────────

class TestResetPredictorBasic:
    """Tests for ResetPredictor basic functionality"""

    def setup_method(self):
        self.predictor = ResetPredictor()

    def test_model_version(self):
        assert "evidence" in self.predictor.model_version

    def test_predict_returns_prediction_explanation(self):
        features = PredictionFeatures(
            hours_since_last_reset=20.0,
            average_reset_interval=24.0,
        )
        result = self.predictor.predict(features)
        assert isinstance(result, PredictionExplanation)

    def test_probability_keys(self):
        features = PredictionFeatures(
            hours_since_last_reset=10.0, average_reset_interval=24.0
        )
        result = self.predictor.predict(features)
        for key in ["5h", "24h", "48h"]:
            assert key in result.probability

    def test_window_ordering(self):
        features = PredictionFeatures(
            hours_since_last_reset=10.0, average_reset_interval=24.0
        )
        result = self.predictor.predict(features)
        assert result.probability["5h"] <= result.probability["24h"]
        assert result.probability["24h"] <= result.probability["48h"]

    def test_main_factors_present(self):
        features = PredictionFeatures(
            hours_since_last_reset=10.0, average_reset_interval=24.0
        )
        result = self.predictor.predict(features)
        assert len(result.main_factors) >= 1


# ──────────────────────────────────────────────
# V2 scenario tests
# ──────────────────────────────────────────────

class TestResetPredictorScenarios:
    """V2 user-requested scenario tests"""

    def setup_method(self):
        self.predictor = ResetPredictor()

    def test_case_1_no_signal_long_interval(self):
        """No signal, just long since reset: probability must not be too high"""
        features = PredictionFeatures(
            hours_since_last_reset=100.0,
            average_reset_interval=24.0,
        )
        result = self.predictor.predict(features)
        assert result.probability["5h"] <= MAX_PROBABILITY_NO_SIGNAL[5]
        assert result.probability["24h"] <= MAX_PROBABILITY_NO_SIGNAL[24]
        assert result.probability["48h"] <= MAX_PROBABILITY_NO_SIGNAL[48]

    def test_case_2_tibo_strong_confirmation(self):
        """Tibo explicitly announces reset: probability rises quickly"""
        features = PredictionFeatures(
            hours_since_last_reset=10.0,
            average_reset_interval=24.0,
            evidence_score=0.95,
            tibo_signal=0.95,
        )
        result = self.predictor.predict(features)
        assert result.probability["24h"] > 0.80
        assert result.probability["48h"] > 0.85
        assert result.evidence_score > 0.0

    def test_case_3_only_community_complaints(self):
        """Only community complaints: moderate increase, should not approach confirmation"""
        features = PredictionFeatures(
            hours_since_last_reset=20.0,
            average_reset_interval=24.0,
            evidence_score=0.25,
            community_signal=0.7,
        )
        result = self.predictor.predict(features)
        # 24h should be higher than no-signal but not exceed 0.70
        assert 0.15 < result.probability["24h"] < 0.70

    def test_case_4_just_reset_no_signal(self):
        """Just reset + no signal: low probability"""
        features = PredictionFeatures(
            hours_since_last_reset=1.0,
            average_reset_interval=24.0,
        )
        result = self.predictor.predict(features)
        assert result.probability["5h"] < 0.20
        assert result.probability["24h"] < 0.50

    def test_evidence_dominates_time(self):
        """Strong evidence should dominate weak time prior: recent reset + strong signal > overdue + no signal"""
        recent_strong = PredictionFeatures(
            hours_since_last_reset=2.0,
            average_reset_interval=24.0,
            evidence_score=0.95,
            tibo_signal=0.95,
        )
        overdue_no_signal = PredictionFeatures(
            hours_since_last_reset=100.0,
            average_reset_interval=24.0,
            evidence_score=0.0,
        )
        recent_result = self.predictor.predict(recent_strong)
        overdue_result = self.predictor.predict(overdue_no_signal)
        assert recent_result.probability["24h"] > overdue_result.probability["24h"]

    def test_tibo_signal_boosts_probability(self):
        """Tibo signal yields higher probability than no signal"""
        base = PredictionFeatures(
            hours_since_last_reset=15.0,
            average_reset_interval=24.0,
            evidence_score=0.0,
        )
        with_signal = PredictionFeatures(
            hours_since_last_reset=15.0,
            average_reset_interval=24.0,
            evidence_score=0.7,
            tibo_signal=0.7,
        )
        base_result = self.predictor.predict(base)
        signal_result = self.predictor.predict(with_signal)
        assert signal_result.probability["24h"] > base_result.probability["24h"]


# ──────────────────────────────────────────────
# Edge-case tests
# ──────────────────────────────────────────────

class TestEdgeCases:
    """Tests for edge cases"""

    def setup_method(self):
        self.predictor = ResetPredictor()

    def test_no_history_no_signals(self):
        features = PredictionFeatures()
        result = self.predictor.predict(features)
        assert 0.0 < result.probability["5h"] < 0.50
        assert result.prior_applied is True

    def test_all_max_signals(self):
        features = PredictionFeatures(
            hours_since_last_reset=48.0,
            average_reset_interval=24.0,
            evidence_score=1.0,
            tibo_signal=1.0,
            community_signal=1.0,
            release_signal=1.0,
        )
        result = self.predictor.predict(features)
        assert result.probability["48h"] > 0.85
        assert result.probability["24h"] > 0.80
