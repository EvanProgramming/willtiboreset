"""Tests for data collectors"""

import json
from datetime import datetime, timezone
from pathlib import Path

from collectors import ResetHistoryCollector, TweetCollector
from model.data_models import ResetEvent, SignalSource, Tweet


class TestTweetCollector:
    """Tests for TweetCollector"""

    def test_collect_empty(self, tmp_path):
        """Return empty list when file does not exist"""
        collector = TweetCollector(tmp_path / "tweets.json")
        assert collector.collect() == []

    def test_save_and_collect(self, tmp_path):
        """Reload after saving"""
        path = tmp_path / "tweets.json"
        collector = TweetCollector(path)
        tweets = [
            Tweet(
                timestamp=datetime(2025, 7, 1, 10, 0, tzinfo=timezone.utc),
                author="user1",
                text="hello",
                url="https://example.com/1",
            ),
            Tweet(
                timestamp=datetime(2025, 7, 1, 11, 0, tzinfo=timezone.utc),
                author="user2",
                text="world",
            ),
        ]
        collector.save(tweets)

        loaded = collector.collect()
        assert len(loaded) == 2
        assert loaded[0].author == "user1"
        assert loaded[1].text == "world"

    def test_json_format(self, tmp_path):
        """Saved JSON format is correct"""
        path = tmp_path / "tweets.json"
        collector = TweetCollector(path)
        collector.save([
            Tweet(
                timestamp=datetime(2025, 7, 1, 10, 0, tzinfo=timezone.utc),
                author="user",
                text="test",
            ),
        ])
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(raw, list)
        assert raw[0]["author"] == "user"
        assert "timestamp" in raw[0]


class TestResetHistoryCollector:
    """Tests for ResetHistoryCollector"""

    def test_collect_empty(self, tmp_path):
        """Return empty list when file does not exist"""
        collector = ResetHistoryCollector(tmp_path / "reset_history.json")
        assert collector.collect() == []

    def test_add_event(self, tmp_path):
        """Add event and persist"""
        path = tmp_path / "reset_history.json"
        collector = ResetHistoryCollector(path)

        event = collector.add_event(
            reset_time=datetime(2025, 7, 1, 12, 0, tzinfo=timezone.utc),
            source=SignalSource.TWITTER,
            confidence=0.85,
            notes="Test event",
        )
        assert event.confidence == 0.85

        # Reload and verify
        loaded = collector.collect()
        assert len(loaded) == 1
        assert loaded[0].source == SignalSource.TWITTER
        assert loaded[0].notes == "Test event"

    def test_add_multiple_events(self, tmp_path):
        """Add multiple events"""
        path = tmp_path / "reset_history.json"
        collector = ResetHistoryCollector(path)

        collector.add_event(
            reset_time=datetime(2025, 7, 1, 12, 0, tzinfo=timezone.utc),
            source=SignalSource.MANUAL,
        )
        collector.add_event(
            reset_time=datetime(2025, 7, 2, 12, 0, tzinfo=timezone.utc),
            source=SignalSource.TWITTER,
        )

        loaded = collector.collect()
        assert len(loaded) == 2
