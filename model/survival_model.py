"""
WillTiboReset - 核心预测引擎 V1.5

基于 Adaptive Bayesian / Survival-inspired 的可解释概率模型。
不使用大型神经网络，适合历史数据较少的场景。

V1.5 改进：
    1. 时间压力使用 sigmoid 平滑函数，避免小幅超期导致概率爆炸。
    2. 使用 median_interval 作为中心基准，降低极端值影响。
    3. LLM 信号聚合引入 authority_score 和 recency_weight。
    4. 新增 probability cap，避免 24h/48h 轻易达到 99%。
    5. SignalScores 语义拆分：reset_intent / limit_complaint / official_change / reset_confirmation。

模型原理：
    1. 时间压力（time_pressure）平滑计算：
       time_pressure = sigmoid(
           (hours_since_last_reset - median_interval) / max(uncertainty, min_uncertainty)
       )

       刚 reset 后 time_pressure ≈ 0
       接近 median interval 时 time_pressure ≈ 0.5
       明显超过周期时 time_pressure → 1

    2. LLM 信号通过 authority_score * recency_weight 加权聚合：
       weighted_signal = sum(signal * authority * recency) / sum(authority * recency)

       tibo_signal      → reset_confirmation + reset_intent（官方确认权重最高）
       community_signal → limit_complaint（社区抱怨）
       release_signal   → official_change（官方变更）

    3. 每小时 hazard rate：
       h_raw = sigmoid(α + β_time * time_pressure + β_tibo * s_tibo
                          + β_community * s_community + β_release * s_release)
       h = min(h_raw, max_hazard)

    4. 窗口概率（假设窗口内 hazard 近似恒定，并施加 cap）：
       P(reset within T hours) = min(cap, 1 - (1 - h)^T)

输入：
    PredictionFeatures {
        hours_since_last_reset,
        average_reset_interval,
        median_reset_interval,
        interval_uncertainty,
        time_pressure,
        tibo_signal,
        community_signal,
        release_signal
    }

输出：
    PredictionExplanation {
        probability: {"5h": 0.42, "24h": 0.76, "48h": 0.91},
        reasons: [...],
        hazard_rate: 0.08,
        time_pressure: 0.65,
        time_ratio: 1.2,
        average_interval_used: 48.0,
        median_interval_used: 42.0,
        prior_applied: False
    }

所有概率范围：0-1。
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from model.data_models import PredictionExplanation, PredictionFeatures, SignalScores, Tweet
from model.model_state import ModelState, ModelStateManager


# ──────────────────────────────────────────────
# 模型默认参数
# ──────────────────────────────────────────────

DEFAULT_PARAMS: dict[str, float] = {
    "alpha": -4.5,          # 截距：刚 reset 后概率足够低
    "beta_time": 2.5,       # time_pressure 权重（time_pressure ∈ [0,1]）
    "beta_tibo": 1.2,       # Tibo reset 信号权重
    "beta_community": 0.4,  # 社区抱怨权重（低于直接 reset 信号）
    "beta_release": 0.3,    # 官方变更权重
}

DEFAULT_RESET_INTERVAL_HOURS: float = 48.0  # 无历史数据时的先验默认周期
INTERVAL_PRIOR_STRENGTH: float = 2.0        # 先验伪样本数，比 V1 更强以平滑早期数据
PREDICTION_HORIZONS: list[int] = [5, 24, 48]

# 概率 cap：避免 long horizon 轻易接近 100%
MAX_HAZARD: float = 0.08                    # 每小时 hazard 上限 8%
MAX_PROBABILITY: dict[int, float] = {
    5: 0.35,
    24: 0.80,
    48: 0.92,
}

# recency 衰减参数
RECENCY_DECAY_HOURS: float = 24.0           # 24 小时后权重衰减到 1/e
MIN_UNCERTAINTY_HOURS: float = 6.0          # 避免 uncertainty 过小时 time_pressure 过于陡峭


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """数值稳定的 sigmoid 函数"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _prob_within_window(hazard: float, hours: int, cap: float = 1.0) -> float:
    """
    在恒定每小时 hazard 下，T 小时内发生至少一次 reset 的概率。
    并施加全局 cap。

    P = min(cap, 1 - (1 - h)^T)
    """
    if hazard <= 0.0:
        return 0.0
    if hazard >= 1.0:
        return min(cap, 1.0)
    prob = 1.0 - (1.0 - hazard) ** hours
    return min(cap, prob)


def _compute_posterior_value(
    empirical_value: Optional[float],
    prior_value: float,
    interval_count: Optional[int] = None,
) -> float:
    """
    用 Bayesian shrinkage 融合先验与观测值。

    posterior = (prior_strength * prior + n * empirical)
                / (prior_strength + n)
    """
    if empirical_value is None:
        return prior_value
    n = interval_count if interval_count is not None and interval_count > 0 else 1
    return (
        INTERVAL_PRIOR_STRENGTH * prior_value + n * empirical_value
    ) / (INTERVAL_PRIOR_STRENGTH + n)


def _compute_time_pressure(
    hours_since: float,
    median_interval: float,
    uncertainty: float,
) -> float:
    """
    平滑时间压力函数。

    当 hours_since 远小于 median 时 → 0
    当 hours_since 接近 median 时 → 0.5
    当 hours_since 远大于 median 时 → 1
    """
    if median_interval <= 0:
        return 0.0
    denom = max(uncertainty, MIN_UNCERTAINTY_HOURS)
    x = (hours_since - median_interval) / denom
    return _sigmoid(x)


def _recency_weight(tweet_timestamp: datetime, now: datetime) -> float:
    """
    根据消息年龄计算 recency weight。

    weight = exp(-hours_old / RECENCY_DECAY_HOURS)
    """
    if tweet_timestamp.tzinfo is None:
        tweet_timestamp = tweet_timestamp.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    hours_old = (now - tweet_timestamp).total_seconds() / 3600.0
    hours_old = max(0.0, hours_old)
    return math.exp(-hours_old / RECENCY_DECAY_HOURS)


def _aggregate_weighted_signals(
    tweets: list[Tweet],
    signal_scores: list[SignalScores],
    now: Optional[datetime] = None,
) -> dict[str, float]:
    """
    使用 authority_score 和 recency_weight 加权聚合 LLM 信号。

    返回：
        {
            "tibo_signal": float,
            "community_signal": float,
            "release_signal": float,
        }
    """
    if not signal_scores:
        return {"tibo_signal": 0.0, "community_signal": 0.0, "release_signal": 0.0}

    if now is None:
        now = datetime.now(timezone.utc)

    weighted_sums = {
        "tibo": 0.0,
        "community": 0.0,
        "release": 0.0,
    }
    weight_sum = 0.0

    use_default_weights = not tweets or len(tweets) != len(signal_scores)

    for i, score in enumerate(signal_scores):
        if use_default_weights:
            authority = 1.0
            recency = 1.0
        else:
            tweet = tweets[i]
            authority = max(0.0, min(1.0, tweet.authority_score))
            recency = _recency_weight(tweet.timestamp, now)
        w = authority * recency

        # Tibo 信号：reset_confirmation 权重最高，reset_intent 次之
        tibo = 0.6 * score.reset_confirmation + 0.3 * score.reset_intent + 0.1 * score.official_change
        # 社区信号：用户抱怨 limit
        community = score.limit_complaint
        # 发布/变更信号：官方变更
        release = score.official_change

        weighted_sums["tibo"] += tibo * w
        weighted_sums["community"] += community * w
        weighted_sums["release"] += release * w
        weight_sum += w

    if weight_sum <= 0:
        return {"tibo_signal": 0.0, "community_signal": 0.0, "release_signal": 0.0}

    return {
        "tibo_signal": min(weighted_sums["tibo"] / weight_sum, 1.0),
        "community_signal": min(weighted_sums["community"] / weight_sum, 1.0),
        "release_signal": min(weighted_sums["release"] / weight_sum, 1.0),
    }


# ──────────────────────────────────────────────
# 特征构建器
# ──────────────────────────────────────────────

def build_features(
    hours_since_last_reset: Optional[float],
    average_reset_interval: Optional[float],
    median_reset_interval: Optional[float] = None,
    interval_uncertainty: Optional[float] = None,
    signal_scores: Optional[list[SignalScores]] = None,
    tweets: Optional[list[Tweet]] = None,
    interval_count: Optional[int] = None,
    model_state: Optional[ModelState] = None,
    now: Optional[datetime] = None,
) -> PredictionFeatures:
    """
    从分析特征和 LLM 信号分数构建 PredictionFeatures（V1.5）。

    Args:
        hours_since_last_reset: 距上次 reset 的小时数（None = 无历史）
        average_reset_interval: 观测到的平均 reset 间隔小时数（None = 无历史）
        median_reset_interval: 观测到的中位 reset 间隔小时数（None = 无历史）
        interval_uncertainty: 间隔估计不确定性（None = 无历史）
        signal_scores: SignalScores 列表
        tweets: 与 signal_scores 对应的 Tweet 列表，用于 authority/recency 加权
        interval_count: 用于计算间隔的样本数
        model_state: 可选的已加载模型状态
        now: 当前时间，用于 recency 计算

    Returns:
        PredictionFeatures 可直接传给 ResetPredictor.predict()
    """
    prior_interval = DEFAULT_RESET_INTERVAL_HOURS

    # 若提供了 model_state，优先使用其统计量
    if model_state is not None:
        posterior_avg = model_state.average_interval_hours
        posterior_median = model_state.median_interval_hours or prior_interval
        posterior_uncertainty = model_state.interval_uncertainty or MIN_UNCERTAINTY_HOURS
    else:
        posterior_avg = _compute_posterior_value(
            average_reset_interval, prior_interval, interval_count
        )
        posterior_median = _compute_posterior_value(
            median_reset_interval, prior_interval, interval_count
        )
        # 无观测时用先验的不确定性（默认周期的 25%）
        empirical_uncertainty = interval_uncertainty if interval_uncertainty is not None else None
        posterior_uncertainty = _compute_posterior_value(
            empirical_uncertainty, prior_interval * 0.25, interval_count
        )

    # 时间因素必须始终参与
    hours_since = (
        hours_since_last_reset
        if hours_since_last_reset is not None
        else posterior_median
    )

    time_pressure = _compute_time_pressure(
        hours_since, posterior_median, posterior_uncertainty
    )

    # 信号聚合
    signals = _aggregate_weighted_signals(
        tweets or [], signal_scores or [], now=now
    )

    return PredictionFeatures(
        hours_since_last_reset=hours_since,
        average_reset_interval=posterior_avg,
        median_reset_interval=posterior_median,
        interval_uncertainty=posterior_uncertainty,
        time_pressure=round(time_pressure, 4),
        tibo_signal=signals["tibo_signal"],
        community_signal=signals["community_signal"],
        release_signal=signals["release_signal"],
    )


# ──────────────────────────────────────────────
# 核心预测器
# ──────────────────────────────────────────────

class ResetPredictor:
    """
    可解释的 Adaptive Bayesian Survival Model 预测器（V1.5）。

    基于平滑 time_pressure 和 capped hazard rate，融合加权 LLM 信号，
    输出各时间窗口的 reset 概率及可读解释。

    适用场景：
        - 历史 reset 数据较少
        - 需要可解释性
        - 需要概率输出（非二分类）
    """

    def __init__(
        self,
        params: Optional[dict[str, float]] = None,
        horizons: Optional[list[int]] = None,
        default_interval: float = DEFAULT_RESET_INTERVAL_HOURS,
        model_state: Optional[ModelState] = None,
        model_state_path: Optional[Path] = None,
    ):
        """
        Args:
            params: 模型参数覆盖
            horizons: 预测时间窗口列表（小时），默认 [5, 24, 48]
            default_interval: 无历史数据时使用的先验默认周期
            model_state: 已加载的模型状态（优先使用）
            model_state_path: model_state.json 路径，若提供则自动加载
        """
        self._default_interval = default_interval
        self._model_state = self._load_model_state(model_state, model_state_path)

        # 参数优先级：传入 params > model_state.params > DEFAULT_PARAMS
        self._params = {**DEFAULT_PARAMS}
        if self._model_state is not None and self._model_state.params:
            self._params.update(self._model_state.params)
        if params:
            self._params.update(params)

        self._horizons = horizons if horizons is not None else list(PREDICTION_HORIZONS)

    def _load_model_state(
        self,
        model_state: Optional[ModelState],
        model_state_path: Optional[Path],
    ) -> Optional[ModelState]:
        """解析 model_state 来源：直接对象优先，否则从路径加载。"""
        if model_state is not None:
            return model_state
        if model_state_path is not None:
            return ModelStateManager(model_state_path).load()
        return None

    @property
    def model_version(self) -> str:
        return "adaptive-bayesian-survival-1.5.0"

    @property
    def model_state(self) -> Optional[ModelState]:
        """返回当前使用的模型状态（可能为 None）"""
        return self._model_state

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
            PredictionExplanation 包含概率、原因、hazard rate、time_pressure 等
        """
        # 1. 补齐 median / uncertainty / time_pressure，确保 predictor 自洽
        self._ensure_time_features(features)

        # 2. 计算 time_ratio（信息展示用）
        time_ratio = self._compute_time_ratio(features)

        # 3. 计算线性预测值（logit）
        logit = self._compute_logit(features)

        # 4. 转换为每小时 hazard rate 并施加上限
        hazard = min(_sigmoid(logit), MAX_HAZARD)

        # 5. 计算各时间窗口概率（带 cap）
        probability: dict[str, float] = {}
        for h in self._horizons:
            cap = MAX_PROBABILITY.get(h, 1.0)
            prob = _prob_within_window(hazard, h, cap)
            probability[f"{h}h"] = round(prob, 4)

        # 6. 生成解释
        reasons = self._generate_reasons(features, time_ratio, hazard, logit)

        # 7. 记录是否使用了先验默认周期
        if self._model_state is not None:
            prior_applied = self._model_state.prior_weight > 0.0
        else:
            prior_applied = (
                features.average_reset_interval is None
                or features.average_reset_interval == self._default_interval
            )

        return PredictionExplanation(
            probability=probability,
            reasons=reasons,
            hazard_rate=round(hazard, 6),
            time_pressure=round(features.time_pressure, 4),
            time_ratio=round(time_ratio, 4) if time_ratio is not None else None,
            average_interval_used=round(features.average_reset_interval, 2)
            if features.average_reset_interval is not None
            else None,
            median_interval_used=round(features.median_reset_interval, 2)
            if features.median_reset_interval is not None
            else None,
            prior_applied=prior_applied,
        )

    # ──────────────────────────────────────────
    # 内部方法
    # ──────────────────────────────────────────

    def _ensure_time_features(self, features: PredictionFeatures) -> None:
        """
        当 PredictionFeatures 直接构造时可能缺少 median / uncertainty / time_pressure，
        这里根据 average_reset_interval 或先验默认值补齐，保证 predictor 自洽。
        """
        median = features.median_reset_interval
        if median is None or median <= 0:
            median = features.average_reset_interval
        if median is None or median <= 0:
            median = self._default_interval

        avg = features.average_reset_interval
        if avg is None or avg <= 0:
            avg = median

        uncertainty = features.interval_uncertainty
        if uncertainty is None or uncertainty <= 0:
            uncertainty = max(median * 0.25, MIN_UNCERTAINTY_HOURS)

        hours_since = features.hours_since_last_reset
        if hours_since is None:
            hours_since = median

        features.average_reset_interval = avg
        features.median_reset_interval = median
        features.interval_uncertainty = uncertainty
        features.time_pressure = _compute_time_pressure(hours_since, median, uncertainty)

    def _compute_time_ratio(self, features: PredictionFeatures) -> Optional[float]:
        """计算 time_ratio（仅用于展示），无历史时使用先验默认值。"""
        hours_since = features.hours_since_last_reset
        interval = features.average_reset_interval
        if interval is None or interval <= 0:
            interval = self._default_interval
        if hours_since is None:
            hours_since = self._default_interval
        return hours_since / interval

    def _compute_logit(self, features: PredictionFeatures) -> float:
        """计算线性预测值（logit），time_pressure 始终参与"""
        p = self._params

        logit = p["alpha"]
        logit += p["beta_time"] * features.time_pressure
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
        if features.hours_since_last_reset is None:
            reasons.append(
                f"无历史 reset 记录，使用先验默认周期 {self._default_interval:.0f}h 作为基线"
            )
        elif features.time_pressure < 0.2:
            reasons.append(
                f"刚 reset 不久（{features.hours_since_last_reset:.1f} 小时），"
                f"时间压力较低（{features.time_pressure:.2f}）"
            )
        elif features.time_pressure < 0.5:
            reasons.append(
                f"距上次 reset {features.hours_since_last_reset:.1f} 小时，"
                f"接近历史中位间隔 {self._format_interval(features)}（time_pressure={features.time_pressure:.2f}）"
            )
        elif features.time_pressure < 0.8:
            reasons.append(
                f"距上次 reset 已 {features.hours_since_last_reset:.1f} 小时，"
                f"超过历史中位间隔 {self._format_interval(features)}（time_pressure={features.time_pressure:.2f}）"
            )
        else:
            reasons.append(
                f"距上次 reset 已 {features.hours_since_last_reset:.1f} 小时，"
                f"明显超过历史中位间隔 {self._format_interval(features)}（time_pressure={features.time_pressure:.2f}）"
            )

        # --- 信号原因 ---
        if features.tibo_signal > 0.5:
            reasons.append(
                f"检测到较强的 reset 确认信号（{features.tibo_signal:.2f}）"
            )
        elif features.tibo_signal > 0.2:
            reasons.append(
                f"检测到一定的 reset 讨论/确认信号（{features.tibo_signal:.2f}）"
            )

        if features.community_signal > 0.5:
            reasons.append(
                f"社区对 limit 的抱怨较高（{features.community_signal:.2f}）"
            )
        elif features.community_signal > 0.2:
            reasons.append(
                f"社区有一定 limit 抱怨（{features.community_signal:.2f}），但不足以确认 reset"
            )

        if features.release_signal > 0.5:
            reasons.append(
                f"检测到官方变更/发布信号（{features.release_signal:.2f}）"
            )

        # --- Hazard 总结 ---
        if hazard >= MAX_HAZARD * 0.9:
            reasons.append(
                f"综合 hazard rate 达到上限（{hazard:.1%}/h），长期概率受 cap 限制"
            )
        elif hazard > 0.04:
            reasons.append(
                f"综合 hazard rate 较高（{hazard:.1%}/h），reset 概率上升"
            )
        elif hazard < 0.01:
            reasons.append(
                f"综合 hazard rate 较低（{hazard:.1%}/h），短期内 reset 概率不大"
            )

        return reasons

    def _format_interval(self, features: PredictionFeatures) -> str:
        """格式化中位间隔显示"""
        interval = features.median_reset_interval
        if interval is not None and interval > 0:
            return f"{interval:.0f} 小时"
        return f"{self._default_interval:.0f} 小时（默认）"


__all__ = [
    "ResetPredictor",
    "build_features",
    "DEFAULT_PARAMS",
    "PREDICTION_HORIZONS",
    "DEFAULT_RESET_INTERVAL_HOURS",
    "INTERVAL_PRIOR_STRENGTH",
    "MAX_HAZARD",
    "MAX_PROBABILITY",
]
