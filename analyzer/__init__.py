"""
analyzer module - signal analyzer

Responsible for preprocessing and feature extraction on collected raw signals,
providing structured analysis input for predictors.
Currently provides a framework implementation; LLM semantic analysis will be integrated later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional

from model.data_models import ResetEvent, Tweet
from analyzer.llm_signal import (
    GeminiAnalyzer,
    LLMAnalyzer,
    MockLLMAnalyzer,
)


# 每周重置规则：默认重置间隔为 7 天。
# 若一周内 Tibo 未提前重置，则下一次重置日期 = 上一次实际重置时间 + 7 天；
# 若 Tibo 提前重置，则已记录的最新重置时间作为基准重新推算。
WEEKLY_RESET_INTERVAL_DAYS: float = 7.0


@dataclass
class AnalysisFeatures:
    """
    Analysis features extracted from raw signals.

    These features serve as input to predictors;
    additional dimensions can be extended as needed.
    """
    # Tweet features
    tweet_count: int = 0
    recent_tweet_count: int = 0  # Number of tweets within the last 24 hours
    unique_authors: int = 0
    sample_texts: list[str] = field(default_factory=list)

    # Reset history features
    total_reset_events: int = 0
    last_reset_time: Optional[datetime] = None
    hours_since_last_reset: Optional[float] = None
    avg_reset_interval_hours: Optional[float] = None
    median_reset_interval_hours: Optional[float] = None
    std_reset_interval_hours: Optional[float] = None
    min_reset_interval_hours: Optional[float] = None
    max_reset_interval_hours: Optional[float] = None
    reset_interval_count: int = 0
    interval_confidence: float = 0.0

    # 每周重置规则相关（下次重置日期 = 上次实际重置 + 7 天）
    next_reset_time: Optional[datetime] = None
    hours_until_next_reset: Optional[float] = None
    reset_schedule_status: str = "no_history"  # "on_schedule" / "overdue" / "no_history"

    # Metadata
    analysis_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_signal_descriptions(self) -> list[str]:
        """Convert to a human-readable list of signal descriptions (for predictor reference)."""
        signals: list[str] = []
        signals.append(f"Collected {self.tweet_count} relevant tweets ({self.unique_authors} authors)")
        signals.append(f"{self.recent_tweet_count} tweets in the last 24 hours")

        if self.hours_since_last_reset is not None:
            signals.append(
                f"{self.hours_since_last_reset:.1f} hours since last reset"
            )
        else:
            signals.append("No known historical reset record")

        if self.next_reset_time is not None:
            next_str = self.next_reset_time.strftime("%Y-%m-%d %H:%M UTC")
            if self.reset_schedule_status == "overdue":
                signals.append(
                    f"Expected reset window already passed (last reset + 7d); "
                    f"next estimated reset {next_str} (overdue)"
                )
            else:
                signals.append(
                    f"Next estimated reset (last reset + 7d): {next_str} "
                    f"in {self.hours_until_next_reset:.1f}h"
                )

        if self.avg_reset_interval_hours is not None:
            signals.append(
                f"Historical average reset interval ~{self.avg_reset_interval_hours:.1f} hours"
            )
        if self.median_reset_interval_hours is not None:
            signals.append(
                f"Historical median reset interval ~{self.median_reset_interval_hours:.1f} hours"
            )
        if self.interval_confidence > 0:
            signals.append(
                f"Historical interval estimate confidence {self.interval_confidence:.0%}"
            )

        if self.sample_texts:
            preview = self.sample_texts[:3]
            signals.append(f"Representative tweet summaries: {' | '.join(preview)}")

        return signals


class SignalAnalyzer:
    """
    Signal analyzer.

    Receives raw tweets and historical reset events,
    extracting structured features for predictors.
    """

    def analyze(
        self,
        tweets: list[Tweet],
        reset_events: list[ResetEvent],
        now: Optional[datetime] = None,
    ) -> AnalysisFeatures:
        """
        Analyze raw signals and extract features.

        Args:
            tweets: List of collected tweets
            reset_events: List of historical reset events
            now: Reference time point (defaults to current UTC time)

        Returns:
            AnalysisFeatures containing extracted features
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # --- Tweet features ---
        tweet_count = len(tweets)
        recent_cutoff = now - timedelta(hours=24)
        recent_tweets = [
            t for t in tweets
            if _to_aware(t.timestamp) >= recent_cutoff
        ]
        unique_authors = len({t.author for t in tweets})
        sample_texts = [
            t.text[:120] for t in recent_tweets[:5]
        ]

        # --- Reset history features ---
        total = len(reset_events)
        last_reset_time: Optional[datetime] = None
        hours_since: Optional[float] = None
        avg_interval: Optional[float] = None
        median_interval: Optional[float] = None
        std_interval: Optional[float] = None
        min_interval: Optional[float] = None
        max_interval: Optional[float] = None
        interval_count = 0
        interval_confidence = 0.0
        next_reset_time: Optional[datetime] = None
        hours_until_next: Optional[float] = None
        schedule_status = "no_history"

        if reset_events:
            sorted_events = sorted(
                reset_events,
                key=lambda e: _to_aware(e.reset_time),
            )
            last_reset_time = sorted_events[-1].reset_time
            hours_since = (
                now - _to_aware(last_reset_time)
            ).total_seconds() / 3600.0

            if len(sorted_events) >= 2:
                intervals = []
                for i in range(1, len(sorted_events)):
                    delta = (
                        _to_aware(sorted_events[i].reset_time)
                        - _to_aware(sorted_events[i - 1].reset_time)
                    )
                    intervals.append(delta.total_seconds() / 3600.0)
                avg_interval = sum(intervals) / len(intervals)
                median_interval = _median(intervals)
                std_interval = _std(intervals)
                min_interval = min(intervals)
                max_interval = max(intervals)
                interval_count = len(intervals)
                # Larger sample size and smaller coefficient of variation yield higher confidence
                interval_confidence = _interval_confidence(
                    interval_count, std_interval, avg_interval
                )

            # 每周重置规则：预计下次重置日期 = 上次实际重置时间 + 7 天。
            # 若已超过该日期仍未重置（Tibo 未按周重置或声明未被捕获），
            # 则顺延到下一个周期作为"预计下次"，并标记为 overdue。
            if last_reset_time is not None:
                aware_last = _to_aware(last_reset_time)
                interval = timedelta(days=WEEKLY_RESET_INTERVAL_DAYS)
                candidate = aware_last + interval
                while candidate <= now:
                    candidate += interval
                next_reset_time = candidate
                hours_until_next = (
                    (candidate - now).total_seconds() / 3600.0
                )
                schedule_status = (
                    "overdue"
                    if (aware_last + interval) <= now
                    else "on_schedule"
                )

        return AnalysisFeatures(
            tweet_count=tweet_count,
            recent_tweet_count=len(recent_tweets),
            unique_authors=unique_authors,
            sample_texts=sample_texts,
            total_reset_events=total,
            last_reset_time=last_reset_time,
            hours_since_last_reset=hours_since,
            avg_reset_interval_hours=avg_interval,
            median_reset_interval_hours=median_interval,
            std_reset_interval_hours=std_interval,
            min_reset_interval_hours=min_interval,
            max_reset_interval_hours=max_interval,
            reset_interval_count=interval_count,
            interval_confidence=interval_confidence,
            next_reset_time=next_reset_time,
            hours_until_next_reset=hours_until_next,
            reset_schedule_status=schedule_status,
            analysis_timestamp=now,
        )


def _median(values: list[float]) -> float:
    """Compute median"""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


def _std(values: list[float]) -> float:
    """Compute sample standard deviation"""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return variance ** 0.5


def _interval_confidence(
    count: int,
    std: Optional[float],
    mean: Optional[float],
) -> float:
    """
    Compute interval confidence based on sample size and coefficient of variation.

    More samples and smaller relative variation make the average interval estimate more reliable.
    """
    if count < 1 or not mean or mean <= 0:
        return 0.0
    sample_factor = min(1.0, count / 10.0)
    cv = std / mean if std else 0.0
    stability_factor = max(0.0, 1.0 - cv)
    return round(sample_factor * 0.6 + stability_factor * 0.4, 4)


def _to_aware(dt: datetime) -> datetime:
    """Convert naive datetime to UTC aware datetime"""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


__all__ = [
    "AnalysisFeatures",
    "SignalAnalyzer",
    # LLM signal analysis
    "LLMAnalyzer",
    "GeminiAnalyzer",
    "MockLLMAnalyzer",
]
