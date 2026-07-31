"""
OpenAI RSS 收集器。

通过 RSS Feed 获取 OpenAI 官方相关新闻和公告。
RSS URL 从 config.rss_feeds["openai"] 读取，不在代码中硬编码。
"""

from __future__ import annotations

from collectors.rss_base import BaseRSSCollector


class OpenAIRSSCollector(BaseRSSCollector):
    """
    OpenAI RSS 收集器。

    从配置的 OpenAI 相关 RSS Feed 收集官方新闻和公告。
    URL 列表通过环境变量 OPENAI_RSS_URLS 配置。
    """

    def __init__(self, feed_urls: list[str] | None = None, timeout: int = 30):
        if feed_urls is None:
            from config import config
            feed_urls = config.rss_feeds.get("openai", [])
        super().__init__(
            feed_urls=feed_urls,
            source_name="openai_rss",
            timeout=timeout,
        )
