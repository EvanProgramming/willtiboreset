"""测试数据模型"""

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
    """ResetEvent 模型测试"""

    def test_create_valid_event(self):
        """创建有效的重置事件"""
        event = ResetEvent(
            reset_time=datetime(2025, 7, 1, 12, 0, tzinfo=timezone.utc),
            source=SignalSource.TWITTER,
            confidence=0.9,
            notes="用户报告额度重置",
        )
        assert event.source == SignalSource.TWITTER
        assert event.confidence == 0.9
        assert event.notes == "用户报告额度重置"

    def test_default_confidence(self):
        """未指定 confidence 时默认为 1.0"""
        event = ResetEvent(
            reset_time=datetime(2025, 7, 1, 12, 0, tzinfo=timezone.utc),
            source=SignalSource.MANUAL,
        )
        assert event.confidence == 1.0

    def test_confidence_out_of_range(self):
        """confidence 超出 [0, 1] 范围应报错"""
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
        """JSON 序列化/反序列化"""
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
    """Tweet 模型测试"""

    def test_create_valid_tweet(self):
        """创建有效的推文"""
        tweet = Tweet(
            timestamp=datetime(2025, 7, 1, 10, 30, tzinfo=timezone.utc),
            author="tibo_user",
            text="Codex 额度好像重置了！",
            url="https://twitter.com/tibo_user/status/123",
        )
        assert tweet.author == "tibo_user"
        assert tweet.text == "Codex 额度好像重置了！"

    def test_empty_author_rejected(self):
        """空作者名应报错"""
        with pytest.raises(ValidationError):
            Tweet(
                timestamp=datetime(2025, 7, 1, 10, 30, tzinfo=timezone.utc),
                author="",
                text="some text",
            )

    def test_empty_text_rejected(self):
        """空文本应报错"""
        with pytest.raises(ValidationError):
            Tweet(
                timestamp=datetime(2025, 7, 1, 10, 30, tzinfo=timezone.utc),
                author="user",
                text="",
            )

    def test_url_optional(self):
        """url 为可选字段"""
        tweet = Tweet(
            timestamp=datetime(2025, 7, 1, 10, 30, tzinfo=timezone.utc),
            author="user",
            text="hello",
        )
        assert tweet.url is None

    def test_source_default(self):
        """source 默认为 unknown"""
        tweet = Tweet(
            timestamp=datetime(2025, 7, 1, 10, 30, tzinfo=timezone.utc),
            author="user",
            text="hello",
        )
        assert tweet.source == "unknown"

    def test_source_custom(self):
        """可以指定 source"""
        tweet = Tweet(
            timestamp=datetime(2025, 7, 1, 10, 30, tzinfo=timezone.utc),
            author="user",
            text="hello",
            source="tibo_rss",
        )
        assert tweet.source == "tibo_rss"

    def test_json_roundtrip_with_source(self):
        """包含 source 字段的 JSON 往返"""
        tweet = Tweet(
            timestamp=datetime(2025, 7, 1, 10, 30, tzinfo=timezone.utc),
            author="user",
            text="hello",
            source="openai_rss",
            url="https://example.com",
        )
        json_str = tweet.model_dump_json()
        restored = Tweet.model_validate_json(json_str)
        assert restored.source == "openai_rss"


class TestSignalScores:
    """SignalScores 模型测试"""

    def test_create_valid_scores(self):
        """创建有效的信号分数"""
        scores = SignalScores(
            reset_signal=0.8,
            limit_discussion=0.6,
            release_signal=0.2,
            community_pressure=0.7,
            confidence=0.9,
            reason=["检测到重置关键词", "检测到限制关键词"],
        )
        assert scores.reset_signal == 0.8
        assert scores.confidence == 0.9
        assert len(scores.reason) == 2

    def test_scores_out_of_range(self):
        """分数超出 [0, 1] 范围应报错"""
        with pytest.raises(ValidationError):
            SignalScores(
                reset_signal=1.5,
                limit_discussion=0.0,
                release_signal=0.0,
                community_pressure=0.0,
                confidence=0.5,
            )

    def test_default_reason(self):
        """reason 默认为空列表"""
        scores = SignalScores(
            reset_signal=0.5,
            limit_discussion=0.5,
            release_signal=0.5,
            community_pressure=0.5,
            confidence=0.5,
        )
        assert scores.reason == []

    def test_to_features(self):
        """to_features 返回特征字典"""
        scores = SignalScores(
            reset_signal=0.8,
            limit_discussion=0.6,
            release_signal=0.2,
            community_pressure=0.7,
            confidence=0.9,
        )
        features = scores.to_features()
        assert isinstance(features, dict)
        assert features["reset_signal"] == 0.8
        assert features["limit_discussion"] == 0.6
        assert features["release_signal"] == 0.2
        assert features["community_pressure"] == 0.7
        assert features["confidence"] == 0.9

    def test_json_roundtrip(self):
        """JSON 序列化/反序列化"""
        scores = SignalScores(
            reset_signal=0.8,
            limit_discussion=0.6,
            release_signal=0.2,
            community_pressure=0.7,
            confidence=0.9,
            reason=["测试原因"],
        )
        json_str = scores.model_dump_json()
        restored = SignalScores.model_validate_json(json_str)
        assert restored.reset_signal == 0.8
        assert restored.reason == ["测试原因"]


class TestPredictionResult:
    """PredictionResult 模型测试"""

    def test_empty_result(self):
        """空预测结果"""
        result = PredictionResult()
        assert result.predictions == []
        assert result.signals_used == []
        assert result.timestamp is not None

    def test_with_predictions(self):
        """包含多个时间窗口的预测"""
        result = PredictionResult(
            predictions=[
                HorizonPrediction(horizon_hours=5, will_reset=True, confidence=0.7),
                HorizonPrediction(horizon_hours=24, will_reset=False, confidence=0.6),
                HorizonPrediction(horizon_hours=48, will_reset=False, confidence=0.8),
            ],
            signals_used=["推文信号", "历史模式"],
            model_version="test-1.0",
        )
        assert len(result.predictions) == 3
        assert result.get_prediction(5).will_reset is True
        assert result.get_prediction(24).will_reset is False

    def test_duplicate_horizons_rejected(self):
        """重复时间窗口应报错"""
        with pytest.raises(ValidationError):
            PredictionResult(
                predictions=[
                    HorizonPrediction(horizon_hours=5, will_reset=True, confidence=0.7),
                    HorizonPrediction(horizon_hours=5, will_reset=False, confidence=0.3),
                ],
            )

    def test_get_prediction_not_found(self):
        """查询不存在的时间窗口返回 None"""
        result = PredictionResult()
        assert result.get_prediction(99) is None

    def test_json_roundtrip(self):
        """JSON 序列化/反序列化"""
        result = PredictionResult(
            predictions=[
                HorizonPrediction(
                    horizon_hours=5, will_reset=True, confidence=0.65,
                    reasoning="大量推文报告重置",
                ),
            ],
            signals_used=["推文激增"],
            model_version="test-1.0",
            notes="测试",
        )
        json_str = result.model_dump_json()
        restored = PredictionResult.model_validate_json(json_str)
        assert len(restored.predictions) == 1
        assert restored.predictions[0].reasoning == "大量推文报告重置"
