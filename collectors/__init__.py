"""
collectors 模块 - 数据收集器

负责从各公开互联网信号源收集原始数据。
当前提供框架和文件存储实现，
后续将接入 Twitter API 等真实数据源。
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model.data_models import ResetEvent, SignalSource, Tweet


class BaseCollector(ABC):
    """数据收集器抽象基类"""

    @abstractmethod
    def collect(self) -> list[Any]:
        """
        收集数据并返回模型对象列表。
        子类必须实现此方法。
        """
        ...


class TweetCollector(BaseCollector):
    """
    推文收集器。

    当前实现：从本地 JSON 文件加载已存储的推文。
    后续扩展：接入 Twitter/X API 实时收集。
    """

    def __init__(self, data_path: Path):
        self._data_path = data_path

    def collect(self) -> list[Tweet]:
        """从本地文件加载推文"""
        if not self._data_path.exists():
            return []
        raw = json.loads(self._data_path.read_text(encoding="utf-8"))
        return [Tweet(**item) for item in raw]

    def save(self, tweets: list[Tweet]) -> None:
        """将推文保存到本地文件"""
        self._data_path.parent.mkdir(parents=True, exist_ok=True)
        data = [t.model_dump(mode="json") for t in tweets]
        self._data_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class ResetHistoryCollector(BaseCollector):
    """
    重置历史收集器。

    当前实现：从本地 JSON 文件加载历史重置事件。
    后续扩展：支持手动录入、社区报告自动归档等。
    """

    def __init__(self, data_path: Path):
        self._data_path = data_path

    def collect(self) -> list[ResetEvent]:
        """从本地文件加载重置历史"""
        if not self._data_path.exists():
            return []
        raw = json.loads(self._data_path.read_text(encoding="utf-8"))
        return [ResetEvent(**item) for item in raw]

    def save(self, events: list[ResetEvent]) -> None:
        """将重置事件保存到本地文件"""
        self._data_path.parent.mkdir(parents=True, exist_ok=True)
        data = [e.model_dump(mode="json") for e in events]
        self._data_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def add_event(
        self,
        reset_time: datetime,
        source: SignalSource,
        confidence: float = 1.0,
        notes: str = "",
    ) -> ResetEvent:
        """添加一条新的重置事件并持久化"""
        events = self.collect()
        event = ResetEvent(
            reset_time=reset_time,
            source=source,
            confidence=confidence,
            notes=notes,
        )
        events.append(event)
        self.save(events)
        return event


__all__ = [
    "BaseCollector",
    "TweetCollector",
    "ResetHistoryCollector",
]
