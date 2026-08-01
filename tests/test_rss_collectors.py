"""Tests for RSS collectors"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

from collectors.community import CommunityCollector
from collectors.rss_base import (
    BaseRSSCollector,
    _build_dedup_key,
    _parse_entry_time,
    _strip_html,
)
from collectors.tibo_rss import TiboRSSCollector
from collectors.openai_rss import OpenAIRSSCollector
from model.data_models import Tweet


def _make_entry(
    title: str = "",
    link: str = "",
    summary: str = "",
    author: str = "",
    published_parsed=None,
) -> SimpleNamespace:
    """Create a mock feedparser entry"""
    return SimpleNamespace(
        title=title,
        link=link,
        summary=summary,
        description=summary,
        author=author,
        published_parsed=published_parsed,
        updated_parsed=None,
    )


class TestStripHtml:
    """Tests for HTML tag stripping"""

    def test_plain_text(self):
        assert _strip_html("hello world") == "hello world"

    def test_with_tags(self):
        assert _strip_html("<p>hello <b>world</b></p>") == "hello world"

    def test_with_entities(self):
        assert _strip_html("<a href='#'>link</a>") == "link"

    def test_empty(self):
        assert _strip_html("") == ""


class TestParseEntryTime:
    """Tests for RSS time parsing"""

    def test_with_published_parsed(self):
        st = time.struct_time((2025, 7, 30, 10, 0, 0, 2, 211, 0))
        dt = _parse_entry_time(_make_entry(published_parsed=st))
        assert dt.year == 2025
        assert dt.month == 7
        assert dt.day == 30
        assert dt.hour == 10
        assert dt.tzinfo is not None

    def test_without_time_fields(self):
        """Fall back to current time when no time fields are present"""
        entry = _make_entry(published_parsed=None)
        dt = _parse_entry_time(entry)
        assert dt.tzinfo is not None


class TestBaseRSSCollector:
    """Tests for BaseRSSCollector"""

    def test_entry_to_tweet(self):
        """Convert an RSS entry to Tweet"""
        st = time.struct_time((2025, 7, 30, 10, 0, 0, 2, 211, 0))
        collector = BaseRSSCollector([], "test_rss")
        entry = _make_entry(
            title="Codex quota reset",
            link="https://example.com/1",
            summary="<p>Quota has reset</p>",
            author="user1",
            published_parsed=st,
        )
        tweet = collector._entry_to_tweet(entry)
        assert tweet is not None
        assert tweet.text == "Codex quota reset - Quota has reset"
        assert tweet.url == "https://example.com/1"
        assert tweet.author == "user1"
        assert tweet.source == "test_rss"

    def test_entry_to_tweet_no_author(self):
        """Use source_name when author is missing"""
        collector = BaseRSSCollector([], "test_rss")
        entry = _make_entry(title="Hello", summary="World")
        tweet = collector._entry_to_tweet(entry)
        assert tweet is not None
        assert tweet.author == "test_rss"

    def test_entry_to_tweet_empty_text(self):
        """Return None when both title and summary are empty"""
        collector = BaseRSSCollector([], "test_rss")
        entry = _make_entry(title="", summary="")
        tweet = collector._entry_to_tweet(entry)
        assert tweet is None

    def test_deduplicate_by_url(self):
        """Deduplicate based on URL"""
        collector = BaseRSSCollector([], "test_rss")
        base_time = datetime(2025, 7, 30, tzinfo=timezone.utc)
        tweets = [
            Tweet(timestamp=base_time, author="a", text="text1", url="https://example.com/1"),
            Tweet(timestamp=base_time, author="b", text="text2", url="https://example.com/1"),
            Tweet(timestamp=base_time, author="c", text="text3", url="https://example.com/2"),
        ]
        result = collector._deduplicate(tweets)
        assert len(result) == 2
        assert result[0].text == "text1"
        assert result[1].text == "text3"

    def test_deduplicate_by_text_hash(self):
        """Deduplicate based on text hash when no URL is present"""
        collector = BaseRSSCollector([], "test_rss")
        base_time = datetime(2025, 7, 30, tzinfo=timezone.utc)
        tweets = [
            Tweet(timestamp=base_time, author="a", text="same text"),
            Tweet(timestamp=base_time, author="b", text="same text"),
            Tweet(timestamp=base_time, author="c", text="different"),
        ]
        result = collector._deduplicate(tweets)
        assert len(result) == 2

    def test_collect_empty_urls(self):
        """Return empty list when no URLs are configured"""
        collector = BaseRSSCollector([], "test_rss")
        assert collector.collect() == []


class TestTiboRSSCollector:
    """Tests for TiboRSSCollector"""

    def test_source_name(self):
        collector = TiboRSSCollector(feed_urls=[])
        assert collector._source_name == "tibo_rss"

    def test_custom_urls(self):
        urls = ["https://example.com/rss1", "https://example.com/rss2"]
        collector = TiboRSSCollector(feed_urls=urls)
        assert collector._feed_urls == urls


class TestOpenAIRSSCollector:
    """Tests for OpenAIRSSCollector"""

    def test_source_name(self):
        collector = OpenAIRSSCollector(feed_urls=[])
        assert collector._source_name == "openai_rss"

    def test_custom_urls(self):
        urls = ["https://openai.com/blog/rss"]
        collector = OpenAIRSSCollector(feed_urls=urls)
        assert collector._feed_urls == urls


class TestCommunityCollector:
    """Tests for CommunityCollector"""

    def test_mock_data_loading(self, tmp_path):
        """Load mock data from JSON file"""
        mock_data = [
            {
                "timestamp": "2025-07-30T10:00:00Z",
                "author": "user1",
                "text": "test tweet",
                "source": "sample",
                "url": "https://example.com/1",
            },
        ]
        mock_path = tmp_path / "sample.json"
        mock_path.write_text(json.dumps(mock_data), encoding="utf-8")

        collector = CommunityCollector(
            feed_urls=[],
            mock_data_path=mock_path,
            use_mock=True,
        )
        tweets = collector.collect()
        assert len(tweets) == 1
        assert tweets[0].source == "community_mock"
        assert tweets[0].text == "test tweet"

    def test_no_mock_file(self, tmp_path):
        """Return empty list when mock file does not exist"""
        collector = CommunityCollector(
            feed_urls=[],
            mock_data_path=tmp_path / "nonexistent.json",
            use_mock=True,
        )
        tweets = collector.collect()
        assert tweets == []

    def test_dedup_with_mock_data(self, tmp_path):
        """Deduplicate mock data with RSS data"""
        mock_data = [
            {
                "timestamp": "2025-07-30T10:00:00Z",
                "author": "user1",
                "text": "duplicate text",
                "url": "https://example.com/1",
            },
            {
                "timestamp": "2025-07-30T11:00:00Z",
                "author": "user2",
                "text": "unique text",
                "url": "https://example.com/2",
            },
        ]
        mock_path = tmp_path / "sample.json"
        mock_path.write_text(json.dumps(mock_data), encoding="utf-8")

        collector = CommunityCollector(
            feed_urls=[],
            mock_data_path=mock_path,
            use_mock=True,
        )
        tweets = collector.collect()
        assert len(tweets) == 2

    def test_mock_data_invalid_entries_skipped(self, tmp_path):
        """Invalid mock data entries are skipped"""
        mock_data = [
            {
                "timestamp": "2025-07-30T10:00:00Z",
                "author": "user1",
                "text": "valid",
            },
            {
                "invalid": "entry",
            },
        ]
        mock_path = tmp_path / "sample.json"
        mock_path.write_text(json.dumps(mock_data), encoding="utf-8")

        collector = CommunityCollector(
            feed_urls=[],
            mock_data_path=mock_path,
            use_mock=True,
        )
        tweets = collector.collect()
        assert len(tweets) == 1
        assert tweets[0].text == "valid"


class TestBuildDedupKey:
    """Tests for deduplication key generation"""

    def test_with_url(self):
        tweet = Tweet(
            timestamp=datetime(2025, 7, 30, tzinfo=timezone.utc),
            author="u",
            text="text",
            url="https://example.com/1",
        )
        assert _build_dedup_key(tweet) == "https://example.com/1"

    def test_without_url(self):
        tweet = Tweet(
            timestamp=datetime(2025, 7, 30, tzinfo=timezone.utc),
            author="u",
            text="some text",
        )
        key = _build_dedup_key(tweet)
        assert len(key) == 32  # MD5 hex digest
