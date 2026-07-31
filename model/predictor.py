"""
WillTiboReset - 预测器框架

定义预测器接口和占位实现。
真正的预测逻辑将在后续 Phase 中由 LLM 分析器
或统计模型填充，此处仅提供扩展骨架。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone

from model.data_models import (
    PredictionResult,
    ResetEvent,
    Tweet,
)


class BasePredictor(ABC):
    """
    预测器抽象基类。

    所有预测器（LLM 预测器、统计模型预测器等）
    都应继承此类并实现 predict 方法。
    """

    @property
    def model_version(self) -> str:
        """返回模型版本标识，子类可覆盖"""
        return "base-0.1.0"

    @abstractmethod
    def predict(
        self,
        tweets: list[Tweet],
        reset_events: list[ResetEvent],
        horizons: list[int],
    ) -> PredictionResult:
        """
        根据输入信号生成预测结果。

        Args:
            tweets: 收集到的相关推文列表
            reset_events: 历史重置事件列表
            horizons: 预测时间窗口列表（小时），如 [5, 24, 48]

        Returns:
            PredictionResult 包含各时间窗口的预测
        """
        ...


class PlaceholderPredictor(BasePredictor):
    """
    占位预测器。

    不包含任何真实预测逻辑，仅用于验证管道连通性。
    调用 predict 会抛出 NotImplementedError，
    明确提示预测逻辑尚未实现。
    """

    @property
    def model_version(self) -> str:
        return "placeholder-0.1.0"

    def predict(
        self,
        tweets: list[Tweet],
        reset_events: list[ResetEvent],
        horizons: list[int],
    ) -> PredictionResult:
        raise NotImplementedError(
            "预测逻辑尚未实现。"
            "请在后续 Phase 中实现 LLMPredictor 或 StatisticalPredictor。"
        )


# ──────────────────────────────────────────────
# 预留扩展接口（后续 Phase 实现）
# ──────────────────────────────────────────────

class LLMPredictor(BasePredictor):
    """
    LLM 预测器（预留）。

    将使用 OpenAI API 对收集到的信号进行自然语言分析，
    输出结构化预测结果。需在后续 Phase 中实现。
    """

    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self._api_key = api_key
        self._model = model

    @property
    def model_version(self) -> str:
        return f"llm-{self._model}-0.1.0"

    def predict(
        self,
        tweets: list[Tweet],
        reset_events: list[ResetEvent],
        horizons: list[int],
    ) -> PredictionResult:
        raise NotImplementedError("LLMPredictor 尚未实现，将在后续 Phase 中完成。")


class StatisticalPredictor(BasePredictor):
    """
    统计模型预测器（预留）。

    将基于历史重置事件的时间序列模式
    构建统计预测模型。需在后续 Phase 中实现。
    """

    @property
    def model_version(self) -> str:
        return "statistical-0.1.0"

    def predict(
        self,
        tweets: list[Tweet],
        reset_events: list[ResetEvent],
        horizons: list[int],
    ) -> PredictionResult:
        raise NotImplementedError("StatisticalPredictor 尚未实现，将在后续 Phase 中完成。")
