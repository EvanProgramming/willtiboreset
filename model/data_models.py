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
    统一信号数据单元。

    所有 Collector（RSS、社区、API 等）输出的统一数据结构。
    后续模块只依赖此结构，不依赖具体数据来源。
    """
    timestamp: datetime = Field(
        ..., description="发布时间（UTC）"
    )
    author: str = Field(
        ..., min_length=1, description="作者用户名或站点名"
    )
    text: str = Field(
        ..., min_length=1, description="正文内容（标题 + 摘要）"
    )
    source: str = Field(
        default="unknown",
        description="数据来源标识，如 tibo_rss / openai_rss / community_mock"
    )
    url: Optional[str] = Field(
        default=None, description="原始链接"
    )
    authority_score: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="数据来源权威性评分（Tibo=1.0, OpenAI=0.9, Community=0.5）"
    )


# ──────────────────────────────────────────────
# LLM 信号分析模型
# ──────────────────────────────────────────────

class SignalScores(BaseModel):
    """
    LLM 信号分析输出（V1.5）。

    将自然语言文本转换为结构化机器学习特征。
    这些特征将传递给 model/survival_model.py 作为预测输入。

    LLM 不负责直接预测 reset，只负责信号提取。
    """
    reset_intent: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="文本讨论/暗示即将发生额度重置的信号强度"
    )
    limit_complaint: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="用户抱怨使用限制/额度耗尽的信号强度"
    )
    official_change: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="官方发布产品变更、更新或政策调整的信号强度"
    )
    reset_confirmation: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="明确确认 reset 已经发生或即将发生的信号强度（最高权重）"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="LLM 对以上评分的整体置信度"
    )
    reason: list[str] = Field(
        default_factory=list,
        description="评分依据列表"
    )

    def to_features(self) -> dict[str, float]:
        """转换为特征字典，供 survival_model.py 使用"""
        return {
            "reset_intent": self.reset_intent,
            "limit_complaint": self.limit_complaint,
            "official_change": self.official_change,
            "reset_confirmation": self.reset_confirmation,
            "confidence": self.confidence,
        }


# ──────────────────────────────────────────────
# 生存模型特征与输出
# ──────────────────────────────────────────────

class PredictionFeatures(BaseModel):
    """
    生存模型预测输入特征（V1.5）。

    由 AnalysisFeatures（统计特征）和 SignalScores（LLM 信号）
    合并而成，作为 ResetPredictor.predict() 的输入。
    """
    hours_since_last_reset: Optional[float] = Field(
        default=None, ge=0.0,
        description="距上次 reset 的小时数，None 表示无历史记录"
    )
    average_reset_interval: Optional[float] = Field(
        default=None, gt=0.0,
        description="历史平均 reset 间隔（小时），None 表示无历史记录"
    )
    median_reset_interval: Optional[float] = Field(
        default=None, gt=0.0,
        description="历史中位 reset 间隔（小时），None 表示无历史记录"
    )
    interval_uncertainty: Optional[float] = Field(
        default=None, ge=0.0,
        description="reset 间隔估计的不确定性（小时），用于平滑 time_pressure"
    )
    time_pressure: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="基于 median interval 的平滑时间压力（0=刚 reset，1=明显超期）"
    )
    tibo_signal: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Tibo/Reset 相关信号强度（reset_confirmation + reset_intent 加权）"
    )
    community_signal: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="社区压力信号强度（limit_complaint 加权）"
    )
    release_signal: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="官方变更信号强度（official_change 加权）"
    )
    evidence_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="综合证据强度（V2，由 LLM 信号与来源权重聚合得到）"
    )


class FactorImpact(BaseModel):
    """
    影响最终概率的单一因素。

    用于 prediction.json 中的 main_factors，让用户理解预测依据。
    """
    factor: str = Field(..., description="因素描述")
    impact: str = Field(..., description="对概率的影响，如 +35% / -10%")
    score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="该因素的原始强度（0-1）"
    )


class PredictionExplanation(BaseModel):
    """
    生存模型预测输出（含可解释说明，V2）。

    返回各时间窗口的 reset 概率、驱动概率的关键原因列表
    以及结构化的 main_factors。
    """
    probability: dict[str, float] = Field(
        ..., description='各时间窗口的 reset 概率，如 {"5h": 0.42, "24h": 0.76, "48h": 0.91}'
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="驱动概率的关键原因（人类可读）"
    )
    main_factors: list[FactorImpact] = Field(
        default_factory=list,
        description="对最终概率影响最大的结构化因素列表"
    )
    evidence_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="综合证据强度（0=无证据，1=非常强证据）"
    )
    hazard_rate: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="当前每小时 hazard rate（模型内部状态，V2 保留用于兼容性）"
    )
    time_pressure: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="平滑后的时间压力（0=刚 reset，1=明显超期）"
    )
    time_ratio: Optional[float] = Field(
        default=None,
        description="hours_since_last_reset / average_reset_interval，None 表示无历史"
    )
    average_interval_used: Optional[float] = Field(
        default=None,
        description="模型实际使用的平均 reset 周期（含先验）"
    )
    median_interval_used: Optional[float] = Field(
        default=None,
        description="模型实际使用的中位 reset 周期（含先验）"
    )
    prior_applied: bool = Field(
        default=False,
        description="是否使用了先验默认周期"
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


# ──────────────────────────────────────────────
# 预测历史与校准模型（V2）
# ──────────────────────────────────────────────

class PredictionHistoryEntry(BaseModel):
    """
    单次预测的历史记录。

    用于后续校准、性能评估以及未来训练数据收集。
    """
    prediction_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="预测生成时间（UTC）"
    )
    prediction: dict[str, float] = Field(
        ..., description='各窗口预测概率，如 {"within_5h": 0.2, "within_24h": 0.5, "within_48h": 0.7}'
    )
    signals: dict = Field(
        default_factory=dict,
        description="本次预测使用的信号快照"
    )
    actual_result: Optional[bool] = Field(
        default=None,
        description="实际是否发生 reset（True=发生，False=未发生，None=未确认）"
    )
    resolved_at: Optional[datetime] = Field(
        default=None,
        description="结果确认时间（UTC）"
    )


class CalibrationBin(BaseModel):
    """概率校准分箱"""
    bin_start: float = Field(..., ge=0.0, le=1.0, description="概率区间起点")
    bin_end: float = Field(..., ge=0.0, le=1.0, description="概率区间终点")
    predicted_mean: float = Field(..., ge=0.0, le=1.0, description="区间内平均预测概率")
    actual_frequency: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="区间内实际发生频率"
    )
    count: int = Field(..., ge=0, description="区间内样本数")


class HorizonPerformance(BaseModel):
    """单个时间窗口的性能指标"""
    horizon_hours: int = Field(..., description="时间窗口（小时）")
    total: int = Field(..., ge=0, description="已确认结果的总预测数")
    brier_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Brier score（越小越好）"
    )
    accuracy: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="二分类准确率（以 0.5 为阈值）"
    )
    calibration_error: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="平均校准误差"
    )
    bins: list[CalibrationBin] = Field(
        default_factory=list, description="校准分箱明细"
    )


class ModelPerformance(BaseModel):
    """
    模型整体性能报告。

    由 calibration.py 根据 prediction_history.json 生成。
    """
    total_predictions: int = Field(..., ge=0, description="历史预测总条数")
    resolved_predictions: int = Field(..., ge=0, description="已确认结果的预测条数")
    overall_brier_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="所有窗口平均 Brier score"
    )
    overall_accuracy: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="所有窗口平均准确率"
    )
    horizons: list[HorizonPerformance] = Field(
        default_factory=list, description="各窗口性能指标"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="报告更新时间（UTC）"
    )
