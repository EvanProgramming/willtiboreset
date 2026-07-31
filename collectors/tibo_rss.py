"""
Tibo RSS 收集器。

通过 RSS Feed 获取 Tibo 相关公开动态。
RSS URL 从 config.rss_feeds["tibo"] 读取，不在代码中硬编码。
"""

from __future__ import annotations

from collectors.rss_base import BaseRSSCollector


class TiboRSSCollector(BaseRSSCollector):
    """
    Tibo RSS 收集器。

    从配置的 Tibo 相关 RSS Feed 收集公开动态。
    URL 列表通过环境变量 TIBO_RSS_URLS 配置。
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
