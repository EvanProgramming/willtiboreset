"""
WillTiboReset - 数据模型定义

使用 pydantic v2 定义所有核心数据结构。
这些模型贯穿收集、分析、预测和输出全流程，
为后续 LLM 分析和预测模型提供统一的类型契约。
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ──────────────────────────────────────────────
# 枚举
# ──────────────────────────────────────────────

class SignalSource(str, Enum):
    """信号来源类型"""
    TWITTER = "twitter"
    REDDIT = "reddit"
    MANUAL = "manual"
    OPENAI_STATUS = "openai_status"
    OTHER = "other"


class PredictionHorizon(int, Enum):
    """预测时间窗口（小时）"""
    HOURS_5 = 5
    HOURS_24 = 24
    HOURS_48 = 48


# ──────────────────────────────────────────────
# 核心数据模型
# ──────────────────────────────────────────────

class ResetEvent(BaseModel):
    """
    历史重置事件记录。

    表示一次已确认或疑似的使用额度重置事件，
    用于训练预测模型和建立历史基线模式。
    """
    reset_time: datetime = Field(
        ..., description="重置发生的时间（UTC）"
    )
    source: SignalSource = Field(
        ..., description="该事件的信息来源"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="该重置事件的可信度，1.0 = 完全确认，0.0 = 纯猜测"
    )
    notes: str = Field(
        default="", description="补充说明"
    )


class Tweet(BaseModel):
    """
    Tibo / OpenAI 相关推文。

    从 Twitter/X 收集的原始信号数据，
    作为预测模型的输入特征之一。
    """
    timestamp: datetime = Field(
        ..., description="推文发布时间（UTC）"
    )
    author: str = Field(
        ..., min_length=1, description="推文作者用户名"
    )
    text: str = Field(
        ..., min_length=1, description="推文正文"
    )
    url: Optional[str] = Field(
        default=None, description="推文链接"
    )


# ──────────────────────────────────────────────
# 预测相关模型
# ──────────────────────────────────────────────

class HorizonPrediction(BaseModel):
    """
    单一时间窗口的预测结果。

    对应一个具体的预测时间跨度（如 5 小时）。
    """
    horizon_hours: int = Field(
        ..., description="预测时间窗口（小时）"
    )
    will_reset: bool = Field(
        ..., description="预测是否会发生重置"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="预测置信度"
    )
    reasoning: str = Field(
        default="", description="预测依据（LLM 分析或模型特征说明）"
    )


class PredictionResult(BaseModel):
    """
    完整预测结果。

    包含多个时间窗口的预测、使用的信号列表、
    以及模型版本等元信息。
    """
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="预测生成时间（UTC）"
    )
    predictions: list[HorizonPrediction] = Field(
        default_factory=list,
        description="各时间窗口的预测结果列表"
    )
    signals_used: list[str] = Field(
        default_factory=list,
        description="本次预测使用的信号描述列表"
    )
    model_version: str = Field(
        default="unknown", description="生成此预测的模型版本标识"
    )
    notes: str = Field(
        default="", description="额外说明"
    )

    @field_validator("predictions")
    @classmethod
    def validate_predictions(cls, v: list[HorizonPrediction]) -> list[HorizonPrediction]:
        """确保每个时间窗口只出现一次"""
        horizons = [p.horizon_hours for p in v]
        if len(horizons) != len(set(horizons)):
            raise ValueError("存在重复的预测时间窗口")
        return v

    def get_prediction(self, horizon_hours: int) -> Optional[HorizonPrediction]:
        """按时间窗口获取预测结果"""
        for p in self.predictions:
            if p.horizon_hours == horizon_hours:
                return p
        return None
