"""
社区信号收集器。

设计为可扩展接口，用于未来接入 Reddit、HackerNews 等社区信号。
当前支持：
  - RSS Feed（通过 COMMUNITY_RSS_URLS 配置）
  - Mock 数据（仅在 USE_MOCK_DATA=true 或显式启用时加载，用于测试）

后续可继承 BaseCollector 添加新的社区数据源，
无需修改下游 analyzer / model 模块。
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
    社区信号收集器。

    聚合多个社区数据源：
    1. RSS Feed（从配置读取 URL，生产环境主要来源）
    2. Mock 数据（仅在显式启用时加载，用于测试）

    返回统一的 Tweet 列表，下游模块无需关心数据来源。
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
        # use_mock 显式为 None 时，读取环境变量 USE_MOCK_DATA
        if use_mock is None:
            use_mock = os.getenv("USE_MOCK_DATA", "false").lower() == "true"
        self._use_mock = use_mock

    def collect(self) -> list[Tweet]:
        """收集社区信号（RSS + 可选 Mock）"""
        tweets: list[Tweet] = []

        # 1. RSS Feed
        if self._rss._feed_urls:
            tweets.extend(self._rss.collect())

        # 2. Mock 数据（仅在测试或显式启用时）
        if self._use_mock:
            mock_tweets = self._load_mock_data()
            tweets.extend(mock_tweets)

        # 去重
        return self._deduplicate(tweets)

    def _load_mock_data(self) -> list[Tweet]:
        """从本地 JSON 文件加载 mock 数据"""
        if not self._mock_data_path or not self._mock_data_path.exists():
            return []
        raw = json.loads(self._mock_data_path.read_text(encoding="utf-8"))
        tweets: list[Tweet] = []
        for item in raw:
            try:
                # 确保 mock 数据标记来源和权威性
                if "source" not in item or item["source"] == "sample":
                    item["source"] = "community_mock"
                if "authority_score" not in item:
                    item["authority_score"] = 0.5
                tweets.append(Tweet(**item))
            except Exception:
                continue
        return tweets

    def _deduplicate(self, tweets: list[Tweet]) -> list[Tweet]:
        """去重"""
        seen: set[str] = set()
        result: list[Tweet] = []
        for tweet in tweets:
            key = _build_dedup_key(tweet)
            if key not in seen:
                seen.add(key)
                result.append(tweet)
        return result
