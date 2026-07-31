"""
WillTiboReset - 核心预测引擎

基于 Discrete-Time Survival Model 的可解释概率模型。
不使用大型神经网络，适合历史数据较少的场景。

模型原理：
    1. 基线 hazard 由"距上次 reset 的时间比率"驱动
       time_ratio = hours_since_last_reset / average_reset_interval
       当 time_ratio > 1（超过平均间隔）时 hazard 显著上升

    2. LLM 信号通过 logistic 线性组合调整 hazard：
       tibo_signal      → reset/limit 讨论，直接推高 hazard
       community_signal → 社区压力，间接推高 hazard
       release_signal   → 产品发布信号，轻微推高 hazard

    3. 每小时 hazard rate：
       h = sigmoid(α + β_time * time_ratio + β_tibo * s_tibo
                       + β_community * s_community + β_release * s_release)

    4. 窗口概率（假设窗口内 hazard 近似恒定）：
       P(reset within T hours) = 1 - (1 - h)^T

输入：
    PredictionFeatures {
        hours_since_last_reset,
        average_reset_interval,
        tibo_signal,
        community_signal,
        release_signal
    }

输出：
    PredictionExplanation {
        probability: {"5h": 0.42, "24h": 0.76, "48h": 0.91},
        reasons: ["Reset interval longer than average", "Tibo mentioned limits"],
        hazard_rate: 0.12,
        time_ratio: 1.8
    }

所有概率范围：0-1。
"""

from __future__ import annotations

import math
from typing import Optional

from model.data_models import PredictionExplanation, PredictionFeatures


# ──────────────────────────────────────────────
# 模型默认参数
# ──────────────────────────────────────────────

DEFAULT_PARAMS: dict[str, float] = {
    "alpha": -4.0,          # 截距：控制基线 hazard（无信号、刚 reset 时很低）
    "beta_time": 1.5,       # time_ratio 权重：超过平均间隔后 hazard 快速上升
    "beta_tibo": 1.0,       # Tibo reset/limit 信号权重
    "beta_community": 0.8,  # 社区压力信号权重
    "beta_release": 0.5,    # 产品发布信号权重
}

DEFAULT_INTERVAL_HOURS: float = 24.0  # 无历史数据时的默认平均间隔
PREDICTION_HORIZONS: list[int] = [5, 24, 48]


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """数值稳定的 sigmoid 函数"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _prob_within_window(hazard: float, hours: int) -> float:
    """
    在恒定每小时 hazard 下，T 小时内发生至少一次 reset 的概率。

    P = 1 - (1 - h)^T
    """
    if hazard <= 0.0:
        return 0.0
    if hazard >= 1.0:
        return 1.0
    return 1.0 - (1.0 - hazard) ** hours


# ──────────────────────────────────────────────
# 特征构建器
# ──────────────────────────────────────────────

def build_features(
    hours_since_last_reset: Optional[float],
    average_reset_interval: Optional[float],
    signal_scores: Optional[list] = None,
) -> PredictionFeatures:
    """
    从分析特征和 LLM 信号分数构建 PredictionFeatures。

    Args:
        hours_since_last_reset: 距上次 reset 的小时数（None = 无历史）
        average_reset_interval: 平均 reset 间隔小时数（None = 无历史）
        signal_scores: SignalScores 列表，将取平均值后融合

    Returns:
        PredictionFeatures 可直接传给 ResetPredictor.predict()
    """
    tibo_signal = 0.0
    community_signal = 0.0
    release_signal = 0.0

    if signal_scores:
        n = len(signal_scores)
        tibo_signal = sum(
            0.6 * s.reset_signal + 0.4 * s.limit_discussion
            for s in signal_scores
        ) / n
        community_signal = sum(s.community_pressure for s in signal_scores) / n
        release_signal = sum(s.release_signal for s in signal_scores) / n

    return PredictionFeatures(
        hours_since_last_reset=hours_since_last_reset,
        average_reset_interval=average_reset_interval,
        tibo_signal=min(tibo_signal, 1.0),
        community_signal=min(community_signal, 1.0),
        release_signal=min(release_signal, 1.0),
    )


# ──────────────────────────────────────────────
# 核心预测器
# ──────────────────────────────────────────────

class ResetPredictor:
    """
    可解释的 Discrete-Time Survival Model 预测器。

    基于 logistic hazard rate 模型，融合时间特征和 LLM 信号，
    输出各时间窗口的 reset 概率及可读解释。

    适用场景：
        - 历史 reset 数据较少（参数可手工调优，无需训练）
        - 需要可解释性（每个特征贡献可量化）
        - 需要概率输出（非二分类）

    用法：
        predictor = ResetPredictor()
        features = PredictionFeatures(
            hours_since_last_reset=20.0,
            average_reset_interval=24.0,
            tibo_signal=0.8,
            community_signal=0.3,
            release_signal=0.1,
        )
        result = predictor.predict(features)
        print(result.probability)  # {"5h": ..., "24h": ..., "48h": ...}
        print(result.reasons)      # ["...", "..."]
    """

    def __init__(
        self,
        params: Optional[dict[str, float]] = None,
        horizons: Optional[list[int]] = None,
        default_interval: float = DEFAULT_INTERVAL_HOURS,
    ):
        """
        Args:
            params: 模型参数覆盖， keys: alpha, beta_time, beta_tibo,
                    beta_community, beta_release
            horizons: 预测时间窗口列表（小时），默认 [5, 24, 48]
            default_interval: 无历史数据时使用的默认平均间隔
        """
        self._params = {**DEFAULT_PARAMS}
        if params:
            self._params.update(params)
        self._horizons = horizons if horizons is not None else list(PREDICTION_HORIZONS)
        self._default_interval = default_interval

    @property
    def model_version(self) -> str:
        return "survival-logistic-1.0.0"

    @property
    def params(self) -> dict[str, float]:
        """返回当前模型参数（只读副本）"""
        return dict(self._params)

    def predict(self, features: PredictionFeatures) -> PredictionExplanation:
        """
        根据输入特征预测各时间窗口的 reset 概率。

        Args:
            features: PredictionFeatures 包含时间特征和信号特征

        Returns:
            PredictionExplanation 包含概率、原因、hazard rate 和 time_ratio
        """
        # 1. 计算 time_ratio
        time_ratio = self._compute_time_ratio(features)

        # 2. 计算线性预测值（logit）
        logit = self._compute_logit(features, time_ratio)

        # 3. 转换为每小时 hazard rate
        hazard = _sigmoid(logit)

        # 4. 计算各时间窗口概率
        probability: dict[str, float] = {}
        for h in self._horizons:
            prob = _prob_within_window(hazard, h)
            probability[f"{h}h"] = round(prob, 4)

        # 5. 生成解释
        reasons = self._generate_reasons(features, time_ratio, hazard, logit)

        return PredictionExplanation(
            probability=probability,
            reasons=reasons,
            hazard_rate=round(hazard, 6),
            time_ratio=round(time_ratio, 4) if time_ratio is not None else None,
        )

    # ──────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────

    def _compute_time_ratio(self, features: PredictionFeatures) -> Optional[float]:
        """计算 time_ratio = hours_since_last_reset / average_reset_interval"""
        if features.hours_since_last_reset is None:
            return None

        interval = features.average_reset_interval
        if interval is None or interval <= 0:
            interval = self._default_interval

        return features.hours_since_last_reset / interval

    def _compute_logit(
        self, features: PredictionFeatures, time_ratio: Optional[float]
    ) -> float:
        """计算线性预测值（logit）"""
        p = self._params

        logit = p["alpha"]

        if time_ratio is not None:
            logit += p["beta_time"] * time_ratio

        logit += p["beta_tibo"] * features.tibo_signal
        logit += p["beta_community"] * features.community_signal
        logit += p["beta_release"] * features.release_signal

        return logit

    def _generate_reasons(
        self,
        features: PredictionFeatures,
        time_ratio: Optional[float],
        hazard: float,
        logit: float,
    ) -> list[str]:
        """生成人类可读的预测原因列表"""
        reasons: list[str] = []

        # --- 时间相关原因 ---
        if time_ratio is None:
            reasons.append("无历史 reset 记录，基于默认基线 hazard 预测")
        elif time_ratio > 1.5:
            reasons.append(
                f"距上次 reset 已 {features.hours_since_last_reset:.1f} 小时，"
                f"远超平均间隔 {self._format_interval(features)}（比率 {time_ratio:.1f}x）"
            )
        elif time_ratio > 1.0:
            reasons.append(
                f"距上次 reset 已 {features.hours_since_last_reset:.1f} 小时，"
                f"超过平均间隔 {self._format_interval(features)}（比率 {time_ratio:.1f}x）"
            )
        elif time_ratio < 0.3:
            reasons.append(
                f"刚 reset 不久（{features.hours_since_last_reset:.1f} 小时），"
                f"短期 reset 概率较低"
            )
        elif time_ratio < 0.7:
            reasons.append(
                f"距上次 reset {features.hours_since_last_reset:.1f} 小时，"
                f"接近平均间隔的 {time_ratio:.0%}，概率适中"
            )
        else:
            reasons.append(
                f"距上次 reset {features.hours_since_last_reset:.1f} 小时，"
                f"接近平均间隔（比率 {time_ratio:.1f}x）"
            )

        # --- Tibo 信号原因 ---
        if features.tibo_signal > 0.5:
            reasons.append(
                f"Tibo/社区讨论 reset 或额度限制（信号强度 {features.tibo_signal:.2f}）"
            )
        elif features.tibo_signal > 0.2:
            reasons.append(
                f"检测到少量 reset/limit 相关讨论（信号强度 {features.tibo_signal:.2f}）"
            )

        # --- 社区压力原因 ---
        if features.community_signal > 0.5:
            reasons.append(
                f"社区压力较高（信号强度 {features.community_signal:.2f}），"
                f"可能加速 reset 决策"
            )

        # --- 产品发布信号原因 ---
        if features.release_signal > 0.5:
            reasons.append(
                f"检测到产品发布/更新信号（信号强度 {features.release_signal:.2f}），"
                f"reset 常伴随发布发生"
            )

        # --- Hazard 总结 ---
        if hazard > 0.15:
            reasons.append(
                f"综合 hazard rate 较高（{hazard:.1%}/h），短期内 reset 概率显著"
            )
        elif hazard < 0.02:
            reasons.append(
                f"综合 hazard rate 较低（{hazard:.1%}/h），短期内 reset 概率不大"
            )

        return reasons

    def _format_interval(self, features: PredictionFeatures) -> str:
        """格式化平均间隔显示"""
        interval = features.average_reset_interval
        if interval is not None and interval > 0:
            return f"{interval:.0f} 小时"
        return f"{self._default_interval:.0f} 小时（默认）"


__all__ = [
    "ResetPredictor",
    "build_features",
    "DEFAULT_PARAMS",
    "PREDICTION_HORIZONS",
]
