"""
Tibo RSS collector.

Fetches public Tibo-related updates via RSS feed.
RSS URLs are read from config.rss_feeds["tibo"] and are not hardcoded.
"""

from __future__ import annotations

from collectors.rss_base import BaseRSSCollector


class TiboRSSCollector(BaseRSSCollector):
    """
    Tibo RSS collector.

    Collects public Tibo-related updates from configured RSS feeds.
    URL list is configured via the TIBO_RSS_URLS environment variable.
    """

    def __init__(self, feed_urls: list[str] | None = None, timeout: int = 30):
        if feed_urls is None:
            from config import config
            feed_urls = config.rss_feeds.get("tibo", [])
        super().__init__(
            feed_urls=feed_urls,
            source_name="tibo_rss",
            timeout=timeout,
            authority_score=1.0,
        )
