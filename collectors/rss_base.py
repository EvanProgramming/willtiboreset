"""
Base RSS collector.

Provides common RSS feed parsing, entry conversion, and deduplication logic.
All RSS-based collectors (TiboRSSCollector, OpenAIRSSCollector, etc.)
inherit from this class to reuse shared logic.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any, Optional

import feedparser

from collectors import BaseCollector
from model.data_models import Tweet


def _strip_html(text: str) -> str:
    """Remove HTML tags and keep plain text"""
    clean = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", clean).strip()


def _parse_entry_time(entry: Any) -> datetime:
    """
    Parse the publication time from an RSS entry.

    Prefer published_parsed (time.struct_time),
    fall back to updated_parsed, and finally use the current time.
    """
    for field_name in ("published_parsed", "updated_parsed"):
        st = getattr(entry, field_name, None)
        if st:
            try:
                return datetime(
                    st.tm_year, st.tm_mon, st.tm_mday,
                    st.tm_hour, st.tm_min, st.tm_sec,
                    tzinfo=timezone.utc,
                )
            except (TypeError, ValueError):
                continue
    return datetime.now(timezone.utc)


def _build_dedup_key(tweet: Tweet) -> str:
    """
    Generate a deduplication key.

    Prefer URL; otherwise use the text hash.
    """
    if tweet.url:
        return tweet.url
    return hashlib.md5(tweet.text.encode("utf-8")).hexdigest()


class BaseRSSCollector(BaseCollector):
    """
    Base RSS collector.

    Supports configuring multiple RSS URLs, automatically parses publication time,
    extracts title and summary, and removes duplicate content.

    Subclasses only need to specify feed_urls and source_name.
    """

    def __init__(
        self,
        feed_urls: list[str],
        source_name: str,
        timeout: int = 30,
        authority_score: float = 1.0,
    ):
        self._feed_urls = feed_urls
        self._source_name = source_name
        self._timeout = timeout
        self._authority_score = authority_score

    def collect(self) -> list[Tweet]:
        """Collect data from all configured RSS URLs"""
        all_tweets: list[Tweet] = []
        for url in self._feed_urls:
            try:
                tweets = self._fetch_feed(url)
                all_tweets.extend(tweets)
            except Exception as e:
                print(f"  ⚠ RSS fetch failed [{url}]: {e}")
        return self._deduplicate(all_tweets)

    def _fetch_feed(self, url: str) -> list[Tweet]:
        """Fetch and parse a single RSS feed"""
        feed = feedparser.parse(
            url,
            request_headers={"User-Agent": "WillTiboReset/1.0"},
        )
        tweets: list[Tweet] = []
        for entry in feed.entries:
            tweet = self._entry_to_tweet(entry)
            if tweet:
                tweets.append(tweet)
        return tweets

    def _entry_to_tweet(self, entry: Any) -> Optional[Tweet]:
        """Convert a feedparser entry into a Tweet"""
        title = getattr(entry, "title", "") or ""
        summary = (
            getattr(entry, "summary", "")
            or getattr(entry, "description", "")
            or ""
        )

        text_parts: list[str] = []
        if title:
            text_parts.append(_strip_html(title))
        if summary:
            text_parts.append(_strip_html(summary))
        text = " - ".join(text_parts)

        if not text:
            return None

        link = getattr(entry, "link", "") or None
        author = getattr(entry, "author", "") or self._source_name

        return Tweet(
            timestamp=_parse_entry_time(entry),
            author=author,
            text=text,
            source=self._source_name,
            url=link,
            authority_score=self._authority_score,
        )

    def _deduplicate(self, tweets: list[Tweet]) -> list[Tweet]:
        """Deduplicate based on URL or text hash"""
        seen: set[str] = set()
        result: list[Tweet] = []
        for tweet in tweets:
            key = _build_dedup_key(tweet)
            if key not in seen:
                seen.add(key)
                result.append(tweet)
        return result
