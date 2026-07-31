"""
analyzer 模块 - 信号分析器

负责对收集到的原始信号进行预处理和特征提取，
为预测器提供结构化的分析输入。
当前提供框架实现，后续将接入 LLM 语义分析。
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


@dataclass
class AnalysisFeatures:
    """
    从原始信号中提取的分析特征。

    这些特征将作为预测器的输入，
    后续可按需扩展更多维度。
    """
    # 推文特征
    tweet_count: int = 0
    recent_tweet_count: int = 0  # 最近 24 小时内的推文数
    unique_authors: int = 0
    sample_texts: list[str] = field(default_factory=list)

    # 重置历史特征
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

    # 元信息
    analysis_timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    def to_signal_descriptions(self) -> list[str]:
        """转换为人类可读的信号描述列表（供预测器引用）"""
        signals: list[str] = []
        signals.append(f"收集到 {self.tweet_count} 条相关推文（{self.unique_authors} 位作者）")
        signals.append(f"最近 24 小时内有 {self.recent_tweet_count} 条推文")

        if self.hours_since_last_reset is not None:
            signals.append(
                f"距上次重置已 {self.hours_since_last_reset:.1f} 小时"
            )
        else:
            signals.append("无已知历史重置记录")

        if self.avg_reset_interval_hours is not None:
            signals.append(
                f"历史平均重置间隔约 {self.avg_reset_interval_hours:.1f} 小时"
            )
        if self.median_reset_interval_hours is not None:
            signals.append(
                f"历史中位重置间隔约 {self.median_reset_interval_hours:.1f} 小时"
            )
        if self.interval_confidence > 0:
            signals.append(
                f"历史间隔估计置信度 {self.interval_confidence:.0%}"
            )

        if self.sample_texts:
            preview = self.sample_texts[:3]
            signals.append(f"代表性推文摘要: {' | '.join(preview)}")

        return signals


class SignalAnalyzer:
    """
    信号分析器。

    接收原始推文和历史重置事件，
    提取结构化特征供预测器使用。
    """

    def analyze(
        self,
        tweets: list[Tweet],
        reset_events: list[ResetEvent],
        now: Optional[datetime] = None,
    ) -> AnalysisFeatures:
        """
        分析原始信号并提取特征。

        Args:
            tweets: 收集到的推文列表
            reset_events: 历史重置事件列表
            now: 参考时间点（默认为当前 UTC 时间）

        Returns:
            AnalysisFeatures 包含提取出的特征
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # --- 推文特征 ---
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

        # --- 重置历史特征 ---
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
                # 样本量越大、变异系数越小，置信度越高
                interval_confidence = _interval_confidence(
                    interval_count, std_interval, avg_interval
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
            analysis_timestamp=now,
        )


def _median(values: list[float]) -> float:
    """计算中位数"""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


def _std(values: list[float]) -> float:
    """计算样本标准差"""
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
    基于样本量和变异系数计算 interval 置信度。

    样本越多、相对波动越小，对平均间隔的估计越可信。
    """
    if count < 1 or not mean or mean <= 0:
        return 0.0
    sample_factor = min(1.0, count / 10.0)
    cv = std / mean if std else 0.0
    stability_factor = max(0.0, 1.0 - cv)
    return round(sample_factor * 0.6 + stability_factor * 0.4, 4)


def _to_aware(dt: datetime) -> datetime:
    """将 naive datetime 转换为 UTC aware datetime"""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


__all__ = [
    "AnalysisFeatures",
    "SignalAnalyzer",
    # LLM 信号分析
    "LLMAnalyzer",
    "GeminiAnalyzer",
    "MockLLMAnalyzer",
]
