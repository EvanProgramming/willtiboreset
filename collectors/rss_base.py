"""
RSS 基础收集器。

提供通用的 RSS Feed 解析、条目转换、去重逻辑。
所有 RSS 类收集器（TiboRSSCollector、OpenAIRSSCollector 等）
继承此类以复用公共逻辑。
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
    """移除 HTML 标签，保留纯文本"""
    clean = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\s+", " ", clean).strip()


def _parse_entry_time(entry: Any) -> datetime:
    """
    从 RSS 条目解析发布时间。

    优先使用 published_parsed（time.struct_time），
    回退到 updated_parsed，最后使用当前时间。
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
    生成去重键。

    优先使用 URL，否则使用正文哈希。
    """
    if tweet.url:
        return tweet.url
    return hashlib.md5(tweet.text.encode("utf-8")).hexdigest()


class BaseRSSCollector(BaseCollector):
    """
    RSS 基础收集器。

    支持配置多个 RSS URL，自动解析发布时间，
    提取标题和正文，去除重复内容。

    子类只需指定 feed_urls 和 source_name。
    """

    def __init__(
        self,
        feed_urls: list[str],
        source_name: str,
        timeout: int = 30,
    ):
        self._feed_urls = feed_urls
        self._source_name = source_name
        self._timeout = timeout

    def collect(self) -> list[Tweet]:
        """从所有配置的 RSS URL 收集数据"""
        all_tweets: list[Tweet] = []
        for url in self._feed_urls:
            try:
                tweets = self._fetch_feed(url)
                all_tweets.extend(tweets)
            except Exception as e:
                print(f"  ⚠ RSS 获取失败 [{url}]: {e}")
        return self._deduplicate(all_tweets)

    def _fetch_feed(self, url: str) -> list[Tweet]:
        """获取并解析单个 RSS Feed"""
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
        """将 feedparser 条目转换为 Tweet"""
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
        )

    def _deduplicate(self, tweets: list[Tweet]) -> list[Tweet]:
        """基于 URL 或正文哈希去重"""
        seen: set[str] = set()
        result: list[Tweet] = []
        for tweet in tweets:
            key = _build_dedup_key(tweet)
            if key not in seen:
                seen.add(key)
                result.append(tweet)
        return result
