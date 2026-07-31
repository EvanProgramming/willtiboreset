"""测试信号分析器"""

from datetime import datetime, timedelta, timezone

from analyzer import SignalAnalyzer
from model.data_models import ResetEvent, SignalSource, Tweet


class TestSignalAnalyzer:
    """SignalAnalyzer 测试"""

    def test_empty_inputs(self):
        """空输入返回零值特征"""
        analyzer = SignalAnalyzer()
        now = datetime(2025, 7, 1, 12, 0, tzinfo=timezone.utc)
        features = analyzer.analyze([], [], now=now)

        assert features.tweet_count == 0
        assert features.recent_tweet_count == 0
        assert features.unique_authors == 0
        assert features.total_reset_events == 0
        assert features.hours_since_last_reset is None
        assert features.avg_reset_interval_hours is None

    def test_tweet_features(self):
        """推文特征提取"""
        now = datetime(2025, 7, 1, 12, 0, tzinfo=timezone.utc)
        tweets = [
            Tweet(
                timestamp=now - timedelta(hours=2),
                author="user1",
                text="Codex reset!",
            ),
            Tweet(
                timestamp=now - timedelta(hours=48),
                author="user2",
                text="额度重置了",
            ),
            Tweet(
                timestamp=now - timedelta(hours=1),
                author="user1",
                text="又重置了",
            ),
        ]
        analyzer = SignalAnalyzer()
        features = analyzer.analyze(tweets, [], now=now)

        assert features.tweet_count == 3
        assert features.recent_tweet_count == 2  # 2 条在 24h 内
        assert features.unique_authors == 2

    def test_reset_history_features(self):
        """重置历史特征提取"""
        now = datetime(2025, 7, 1, 12, 0, tzinfo=timezone.utc)
        events = [
            ResetEvent(
                reset_time=now - timedelta(hours=72),
                source=SignalSource.MANUAL,
            ),
            ResetEvent(
                reset_time=now - timedelta(hours=24),
                source=SignalSource.TWITTER,
            ),
        ]
        analyzer = SignalAnalyzer()
        features = analyzer.analyze([], events, now=now)

        assert features.total_reset_events == 2
        assert features.hours_since_last_reset == pytest_approx(24.0)
        # 间隔 = 72 - 24 = 48 小时
        assert features.avg_reset_interval_hours == pytest_approx(48.0)

    def test_signal_descriptions(self):
        """信号描述列表生成"""
        now = datetime(2025, 7, 1, 12, 0, tzinfo=timezone.utc)
        tweets = [Tweet(timestamp=now, author="u", text="reset")]
        events = [
            ResetEvent(
                reset_time=now - timedelta(hours=10),
                source=SignalSource.MANUAL,
            ),
        ]
        analyzer = SignalAnalyzer()
        features = analyzer.analyze(tweets, events, now=now)
        descriptions = features.to_signal_descriptions()

        assert len(descriptions) > 0
        assert any("推文" in d for d in descriptions)
        assert any("距上次重置" in d for d in descriptions)


def pytest_approx(expected, rel=1e-1):
    """简单近似比较（避免 import pytest 在模块顶层）"""
    class _Approx:
        def __init__(self, expected, rel):
            self.expected = expected
            self.rel = rel

        def __eq__(self, actual):
            return abs(actual - self.expected) <= self.rel * abs(self.expected)

        def __repr__(self):
            return f"≈{self.expected}"

    return _Approx(expected, rel)
