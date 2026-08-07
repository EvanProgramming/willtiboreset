"""Tests for the signal analyzer"""

from datetime import datetime, timedelta, timezone

from analyzer import SignalAnalyzer
from model.data_models import ResetEvent, SignalSource, Tweet


class TestSignalAnalyzer:
    """Tests for SignalAnalyzer"""

    def test_empty_inputs(self):
        """Empty inputs return zero-valued features"""
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
        """Tweet feature extraction"""
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
                text="Quota has reset",
            ),
            Tweet(
                timestamp=now - timedelta(hours=1),
                author="user1",
                text="Reset again",
            ),
        ]
        analyzer = SignalAnalyzer()
        features = analyzer.analyze(tweets, [], now=now)

        assert features.tweet_count == 3
        assert features.recent_tweet_count == 2  # 2 tweets within 24h
        assert features.unique_authors == 2

    def test_reset_history_features(self):
        """Reset history feature extraction"""
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
        # interval = 72 - 24 = 48 hours
        assert features.avg_reset_interval_hours == pytest_approx(48.0)
        assert features.median_reset_interval_hours == pytest_approx(48.0)
        assert features.min_reset_interval_hours == pytest_approx(48.0)
        assert features.max_reset_interval_hours == pytest_approx(48.0)
        assert features.std_reset_interval_hours == pytest_approx(0.0)
        assert features.reset_interval_count == 1
        assert features.interval_confidence > 0.0

    def test_signal_descriptions(self):
        """Signal description list generation"""
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
        assert any("tweets" in d for d in descriptions)
        assert any("hours since last reset" in d for d in descriptions)

    def test_next_reset_on_schedule(self):
        """Next reset = last reset + 7 days when not yet passed"""
        now = datetime(2025, 7, 1, 12, 0, tzinfo=timezone.utc)
        events = [
            ResetEvent(
                reset_time=now - timedelta(hours=24),
                source=SignalSource.TWITTER,
            ),
        ]
        analyzer = SignalAnalyzer()
        features = analyzer.analyze([], events, now=now)

        assert features.next_reset_time is not None
        # last reset + 7 days = now - 24h + 168h = now + 144h
        assert features.hours_until_next_reset == pytest_approx(144.0)
        assert features.reset_schedule_status == "on_schedule"

    def test_next_reset_overdue(self):
        """When the weekly window has passed, roll to next cycle and mark overdue"""
        now = datetime(2025, 7, 1, 12, 0, tzinfo=timezone.utc)
        events = [
            ResetEvent(
                reset_time=now - timedelta(days=8),  # 8 days ago, > 7d window
                source=SignalSource.TWITTER,
            ),
        ]
        analyzer = SignalAnalyzer()
        features = analyzer.analyze([], events, now=now)

        assert features.next_reset_time is not None
        assert features.reset_schedule_status == "overdue"
        # last reset + 7d = now - 1d (passed); roll forward one more week:
        # next = last reset + 14d = now + 6d = 144h
        assert features.hours_until_next_reset == pytest_approx(144.0)


def pytest_approx(expected, rel=1e-1):
    """Simple approximate comparison (avoids importing pytest at module top level)"""
    class _Approx:
        def __init__(self, expected, rel):
            self.expected = expected
            self.rel = rel

        def __eq__(self, actual):
            return abs(actual - self.expected) <= self.rel * abs(self.expected)

        def __repr__(self):
            return f"≈{self.expected}"

    return _Approx(expected, rel)
