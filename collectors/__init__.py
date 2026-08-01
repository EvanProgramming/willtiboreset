"""
collectors module - data collectors

Responsible for collecting raw data from various public internet signal sources.
Currently supports RSS feeds and community mock data;
may be extended to API data sources in the future.

All Collectors follow a unified interface:
    collect() -> list[Tweet]

Downstream modules only depend on the Tweet structure, not on specific data sources.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from model.data_models import ResetEvent, SignalSource, Tweet


class BaseCollector(ABC):
    """Abstract base class for data collectors"""

    @abstractmethod
    def collect(self) -> list[Any]:
        """
        Collect data and return a list of model objects.
        Subclasses must implement this method.
        """
        ...


class TweetCollector(BaseCollector):
    """
    Tweet collector.

    Current implementation: loads stored tweets from a local JSON file.
    Future extension: connect to Twitter/X API for real-time collection.
    """

    def __init__(self, data_path: Path):
        self._data_path = data_path

    def collect(self) -> list[Tweet]:
        """Load tweets from local file"""
        if not self._data_path.exists():
            return []
        raw = json.loads(self._data_path.read_text(encoding="utf-8"))
        return [Tweet(**item) for item in raw]

    def save(self, tweets: list[Tweet]) -> None:
        """Save tweets to local file"""
        self._data_path.parent.mkdir(parents=True, exist_ok=True)
        data = [t.model_dump(mode="json") for t in tweets]
        self._data_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class ResetHistoryCollector(BaseCollector):
    """
    Reset history collector.

    Current implementation: loads historical reset events from a local JSON file.
    Future extension: support manual entry, automatic archival of community reports, etc.
    """

    def __init__(self, data_path: Path):
        self._data_path = data_path

    def collect(self) -> list[ResetEvent]:
        """Load reset history from local file"""
        if not self._data_path.exists():
            return []
        raw = json.loads(self._data_path.read_text(encoding="utf-8"))
        return [ResetEvent(**item) for item in raw]

    def save(self, events: list[ResetEvent]) -> None:
        """Save reset events to local file"""
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
        """Add a new reset event and persist it"""
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
    # RSS collectors
    "BaseRSSCollector",
    "TiboRSSCollector",
    "OpenAIRSSCollector",
    "CommunityCollector",
]


# RSS collector exports (placed at the end to avoid circular imports)
from collectors.rss_base import BaseRSSCollector
from collectors.tibo_rss import TiboRSSCollector
from collectors.openai_rss import OpenAIRSSCollector
from collectors.community import CommunityCollector
