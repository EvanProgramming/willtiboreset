"""
OpenAI RSS collector.

Fetches official OpenAI news and announcements via RSS feed.
RSS URLs are read from config.rss_feeds["openai"] and are not hardcoded.
"""

from __future__ import annotations

from collectors.rss_base import BaseRSSCollector


class OpenAIRSSCollector(BaseRSSCollector):
    """
    OpenAI RSS collector.

    Collects official OpenAI news and announcements from configured RSS feeds.
    URL list is configured via the OPENAI_RSS_URLS environment variable.
    """

    def __init__(self, feed_urls: list[str] | None = None, timeout: int = 30):
        if feed_urls is None:
            from config import config
            feed_urls = config.rss_feeds.get("openai", [])
        super().__init__(
            feed_urls=feed_urls,
            source_name="openai_rss",
            timeout=timeout,
            authority_score=0.9,
        )
