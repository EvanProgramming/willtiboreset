"""Tests for data models"""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from model.data_models import (
    HorizonPrediction,
    PredictionResult,
    ResetEvent,
    SignalScores,
    SignalSource,
    Tweet,
)


class TestResetEvent:
    """Tests for ResetEvent model"""

    def test_create_valid_event(self):
        """Create a valid reset event"""
        event = ResetEvent(
            reset_time=datetime(2025, 7, 1, 12, 0, tzinfo=timezone.utc),
            source=SignalSource.TWITTER,
            confidence=0.9,
            notes="User reported quota reset",
        )
        assert event.source == SignalSource.TWITTER
        assert event.confidence == 0.9
        assert event.notes == "User reported quota reset"

    def test_default_confidence(self):
        """Default confidence is 1.0 when not specified"""
        event = ResetEvent(
            reset_time=datetime(2025, 7, 1, 12, 0, tzinfo=timezone.utc),
            source=SignalSource.MANUAL,
        )
        assert event.confidence == 1.0

    def test_confidence_out_of_range(self):
        """Confidence outside [0, 1] should raise validation error"""
        with pytest.raises(ValidationError):
            ResetEvent(
                reset_time=datetime(2025, 7, 1, 12, 0, tzinfo=timezone.utc),
                source=SignalSource.MANUAL,
                confidence=1.5,
            )
        with pytest.raises(ValidationError):
            ResetEvent(
                reset_time=datetime(2025, 7, 1, 12, 0, tzinfo=timezone.utc),
                source=SignalSource.MANUAL,
                confidence=-0.1,
            )

    def test_json_roundtrip(self):
        """JSON serialization/deserialization"""
        event = ResetEvent(
            reset_time=datetime(2025, 7, 1, 12, 0, tzinfo=timezone.utc),
            source=SignalSource.OPENAI_STATUS,
            confidence=0.8,
            notes="test",
        )
        json_str = event.model_dump_json()
        restored = ResetEvent.model_validate_json(json_str)
        assert restored.source == SignalSource.OPENAI_STATUS
        assert restored.confidence == 0.8


class TestTweet:
    """Tests for Tweet model"""

    def test_create_valid_tweet(self):
        """Create a valid tweet"""
        tweet = Tweet(
            timestamp=datetime(2025, 7, 1, 10, 30, tzinfo=timezone.utc),
            author="tibo_user",
            text="Codex quota seems to have reset!",
            url="https://twitter.com/tibo_user/status/123",
        )
        assert tweet.author == "tibo_user"
        assert tweet.text == "Codex quota seems to have reset!"

    def test_empty_author_rejected(self):
        """Empty author name should be rejected"""
        with pytest.raises(ValidationError):
            Tweet(
                timestamp=datetime(2025, 7, 1, 10, 30, tzinfo=timezone.utc),
                author="",
                text="some text",
            )

    def test_empty_text_rejected(self):
        """Empty text should be rejected"""
        with pytest.raises(ValidationError):
            Tweet(
                timestamp=datetime(2025, 7, 1, 10, 30, tzinfo=timezone.utc),
                author="user",
                text="",
            )

    def test_url_optional(self):
        """url is optional"""
        tweet = Tweet(
            timestamp=datetime(2025, 7, 1, 10, 30, tzinfo=timezone.utc),
            author="user",
            text="hello",
        )
        assert tweet.url is None

    def test_source_default(self):
        """source defaults to unknown"""
        tweet = Tweet(
            timestamp=datetime(2025, 7, 1, 10, 30, tzinfo=timezone.utc),
            author="user",
            text="hello",
        )
        assert tweet.source == "unknown"

    def test_source_custom(self):
        """Custom source can be specified"""
        tweet = Tweet(
            timestamp=datetime(2025, 7, 1, 10, 30, tzinfo=timezone.utc),
            author="user",
            text="hello",
            source="tibo_rss",
        )
        assert tweet.source == "tibo_rss"

    def test_authority_score_default(self):
        """authority_score defaults to 1.0"""
        tweet = Tweet(
            timestamp=datetime(2025, 7, 1, 10, 30, tzinfo=timezone.utc),
            author="user",
            text="hello",
        )
        assert tweet.authority_score == 1.0

    def test_authority_score_range(self):
        """authority_score outside [0, 1] should raise validation error"""
        with pytest.raises(ValidationError):
            Tweet(
                timestamp=datetime(2025, 7, 1, 10, 30, tzinfo=timezone.utc),
                author="user",
                text="hello",
                authority_score=1.5,
            )

    def test_json_roundtrip_with_source(self):
        """JSON round-trip including source and authority_score"""
        tweet = Tweet(
            timestamp=datetime(2025, 7, 1, 10, 30, tzinfo=timezone.utc),
            author="user",
            text="hello",
            source="openai_rss",
            url="https://example.com",
            authority_score=0.9,
        )
        json_str = tweet.model_dump_json()
        restored = Tweet.model_validate_json(json_str)
        assert restored.source == "openai_rss"
        assert restored.authority_score == 0.9


class TestSignalScores:
    """Tests for SignalScores model"""

    def test_create_valid_scores(self):
        """Create valid signal scores"""
        scores = SignalScores(
            reset_intent=0.8,
            limit_complaint=0.6,
            official_change=0.2,
            reset_confirmation=0.7,
            confidence=0.9,
            reason=["Reset keyword detected", "Limit keyword detected"],
        )
        assert scores.reset_intent == 0.8
        assert scores.confidence == 0.9
        assert len(scores.reason) == 2

    def test_scores_out_of_range(self):
        """Scores outside [0, 1] should raise validation error"""
        with pytest.raises(ValidationError):
            SignalScores(
                reset_intent=1.5,
                limit_complaint=0.0,
                official_change=0.0,
                reset_confirmation=0.0,
                confidence=0.5,
            )

    def test_default_reason(self):
        """reason defaults to empty list"""
        scores = SignalScores(
            reset_intent=0.5,
            limit_complaint=0.5,
            official_change=0.5,
            reset_confirmation=0.5,
            confidence=0.5,
        )
        assert scores.reason == []

    def test_to_features(self):
        """to_features returns feature dict"""
        scores = SignalScores(
            reset_intent=0.8,
            limit_complaint=0.6,
            official_change=0.2,
            reset_confirmation=0.7,
            confidence=0.9,
        )
        features = scores.to_features()
        assert isinstance(features, dict)
        assert features["reset_intent"] == 0.8
        assert features["limit_complaint"] == 0.6
        assert features["official_change"] == 0.2
        assert features["reset_confirmation"] == 0.7
        assert features["confidence"] == 0.9

    def test_json_roundtrip(self):
        """JSON serialization/deserialization"""
        scores = SignalScores(
            reset_intent=0.8,
            limit_complaint=0.6,
            official_change=0.2,
            reset_confirmation=0.7,
            confidence=0.9,
            reason=["test reason"],
        )
        json_str = scores.model_dump_json()
        restored = SignalScores.model_validate_json(json_str)
        assert restored.reset_intent == 0.8
        assert restored.reason == ["test reason"]


class TestPredictionResult:
    """Tests for PredictionResult model"""

    def test_empty_result(self):
        """Empty prediction result"""
        result = PredictionResult()
        assert result.predictions == []
        assert result.signals_used == []
        assert result.timestamp is not None

    def test_with_predictions(self):
        """Prediction with multiple time horizons"""
        result = PredictionResult(
            predictions=[
                HorizonPrediction(horizon_hours=5, will_reset=True, confidence=0.7),
                HorizonPrediction(horizon_hours=24, will_reset=False, confidence=0.6),
                HorizonPrediction(horizon_hours=48, will_reset=False, confidence=0.8),
            ],
            signals_used=["tweet signal", "historical pattern"],
            model_version="test-1.0",
        )
        assert len(result.predictions) == 3
        assert result.get_prediction(5).will_reset is True
        assert result.get_prediction(24).will_reset is False

    def test_duplicate_horizons_rejected(self):
        """Duplicate time horizons should be rejected"""
        with pytest.raises(ValidationError):
            PredictionResult(
                predictions=[
                    HorizonPrediction(horizon_hours=5, will_reset=True, confidence=0.7),
                    HorizonPrediction(horizon_hours=5, will_reset=False, confidence=0.3),
                ],
            )

    def test_get_prediction_not_found(self):
        """Querying non-existent horizon returns None"""
        result = PredictionResult()
        assert result.get_prediction(99) is None

    def test_json_roundtrip(self):
        """JSON serialization/deserialization"""
        result = PredictionResult(
            predictions=[
                HorizonPrediction(
                    horizon_hours=5, will_reset=True, confidence=0.65,
                    reasoning="Many tweets report reset",
                ),
            ],
            signals_used=["tweet surge"],
            model_version="test-1.0",
            notes="test",
        )
        json_str = result.model_dump_json()
        restored = PredictionResult.model_validate_json(json_str)
        assert len(restored.predictions) == 1
        assert restored.predictions[0].reasoning == "Many tweets report reset"
