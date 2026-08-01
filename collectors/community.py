"""
Community signal collector.

Designed as an extensible interface for future integration with community signals
such as Reddit, HackerNews, etc.
Currently supports:
  - RSS feeds (configured via COMMUNITY_RSS_URLS)
  - Mock data (loaded only when USE_MOCK_DATA=true or explicitly enabled, for testing)

New community data sources can be added by inheriting from BaseCollector,
without modifying downstream analyzer / model modules.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from collectors import BaseCollector
from collectors.rss_base import BaseRSSCollector, _build_dedup_key
from model.data_models import Tweet


class CommunityCollector(BaseCollector):
    """
    Community signal collector.

    Aggregates multiple community data sources:
    1. RSS feeds (URLs read from config, main source in production)
    2. Mock data (loaded only when explicitly enabled, for testing)

    Returns a unified Tweet list; downstream modules do not need to know the source.
    """

    def __init__(
        self,
        feed_urls: list[str] | None = None,
        mock_data_path: Optional[Path] = None,
        timeout: int = 30,
        use_mock: Optional[bool] = None,
    ):
        if feed_urls is None:
            from config import config
            feed_urls = config.rss_feeds.get("community", [])
        if mock_data_path is None:
            from config import config
            mock_data_path = config.sample_tweets_path

        self._rss = BaseRSSCollector(
            feed_urls=feed_urls,
            source_name="community_rss",
            timeout=timeout,
            authority_score=0.5,
        )
        self._mock_data_path = mock_data_path
        # When use_mock is explicitly None, read the USE_MOCK_DATA environment variable
        if use_mock is None:
            use_mock = os.getenv("USE_MOCK_DATA", "false").lower() == "true"
        self._use_mock = use_mock

    def collect(self) -> list[Tweet]:
        """Collect community signals (RSS + optional mock)"""
        tweets: list[Tweet] = []

        # 1. RSS feeds
        if self._rss._feed_urls:
            tweets.extend(self._rss.collect())

        # 2. Mock data (only in tests or when explicitly enabled)
        if self._use_mock:
            mock_tweets = self._load_mock_data()
            tweets.extend(mock_tweets)

        # Deduplicate
        return self._deduplicate(tweets)

    def _load_mock_data(self) -> list[Tweet]:
        """Load mock data from a local JSON file"""
        if not self._mock_data_path or not self._mock_data_path.exists():
            return []
        raw = json.loads(self._mock_data_path.read_text(encoding="utf-8"))
        tweets: list[Tweet] = []
        for item in raw:
            try:
                # Ensure mock data is tagged with source and authority
                if "source" not in item or item["source"] == "sample":
                    item["source"] = "community_mock"
                if "authority_score" not in item:
                    item["authority_score"] = 0.5
                tweets.append(Tweet(**item))
            except Exception:
                continue
        return tweets

    def _deduplicate(self, tweets: list[Tweet]) -> list[Tweet]:
        """Deduplicate"""
        seen: set[str] = set()
        result: list[Tweet] = []
        for tweet in tweets:
            key = _build_dedup_key(tweet)
            if key not in seen:
                seen.add(key)
                result.append(tweet)
        return result
