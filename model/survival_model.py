"""
WillTiboReset - 核心预测引擎 V2

Adaptive Bayesian Evidence Model。

V2 核心变化：
    1. 从"时间驱动"改为"证据驱动 + 弱时间先验"。
    2. LLM 信号按来源权威性（authority_score）和时效性（recency_weight）聚合，
       生成综合 evidence_score。
    3. 使用 Bayesian odds update 将先验基线概率提升为后验概率。
    4. 时间因素仅作为弱先验修正：刚 reset 降低概率，超期小幅提升，
       但无信号时不会导致高概率。
    5. 输出结构化 main_factors，说明每个因素对概率的贡献。

输入：
    PredictionFeatures {
        hours_since_last_reset,
        average_reset_interval,
        median_reset_interval,
        interval_uncertainty,
        time_pressure,
        tibo_signal,
        community_signal,
        release_signal,
        evidence_score
    }

输出：
    PredictionExplanation {
        probability: {"5h": 0.12, "24h": 0.45, "48h": 0.62},
        reasons: [...],
        main_factors: [FactorImpact, ...],
        evidence_score: 0.72,
        time_pressure: 0.65,
        ...
    }
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from model.data_models import (
    FactorImpact,
    PredictionExplanation,
    PredictionFeatures,
    SignalScores,
    Tweet,
)
from model.model_state import ModelState, ModelStateManager


# ──────────────────────────────────────────────
# 模型默认参数
# ──────────────────────────────────────────────

DEFAULT_RESET_INTERVAL_HOURS: float = 48.0
INTERVAL_PRIOR_STRENGTH: float = 2.0
PREDICTION_HORIZONS: list[int] = [5, 24, 48]

# 无信号时的先验基线概率（弱时间先验）
BASE_PROBABILITY: dict[int, float] = {
    5: 0.05,
    24: 0.18,
    48: 0.28,
}

# 时间因素对基线的最大调整幅度（±30%）
TIME_ADJUSTMENT_STRENGTH: float = 0.30

# 证据乘数：evidence_score=1 时，odds 最大放大倍数
MAX_EVIDENCE_MULTIPLIER: float = 25.0

# 各窗口概率上限：避免无证据时因时间自然膨胀
MAX_PROBABILITY_NO_SIGNAL: dict[int, float] = {
    5: 0.20,
    24: 0.50,
    48: 0.70,
}

# 强证据情况下的上限
MAX_PROBABILITY_STRONG_EVIDENCE: dict[int, float] = {
    5: 0.75,
    24: 0.95,
    48: 0.98,
}

# recency 衰减参数
RECENCY_DECAY_HOURS: float = 24.0
MIN_UNCERTAINTY_HOURS: float = 6.0


# ──────────────────────────────────────────────
# 辅助函数
# ──────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """数值稳定的 sigmoid 函数"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _compute_posterior_value(
    empirical_value: Optional[float],
    prior_value: float,
    interval_count: Optional[int] = None,
) -> float:
    """用 Bayesian shrinkage 融合先验与观测值。"""
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
    """根据消息年龄计算 recency weight。"""
    if tweet_timestamp.tzinfo is None:
        tweet_timestamp = tweet_timestamp.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    hours_old = (now - tweet_timestamp).total_seconds() / 3600.0
    hours_old = max(0.0, hours_old)
    return math.exp(-hours_old / RECENCY_DECAY_HOURS)


def _source_priority(source: str) -> float:
    """来源优先级权重：Tibo > OpenAI > Community。"""
    source_lower = source.lower()
    if "tibo" in source_lower:
        return 1.0
    if "openai" in source_lower:
        return 0.8
    return 0.4


def _per_tweet_evidence(score: SignalScores) -> float:
    """
    根据单条 LLM 信号计算证据强度（0-1）。

    强调 reset_confirmation，抑制 limit_complaint。
    """
    evidence = 0.0

    if score.reset_confirmation >= 0.8:
        evidence += 0.5 + 0.4 * score.reset_confirmation
    elif score.reset_confirmation >= 0.5:
        evidence += 0.25 + 0.25 * score.reset_confirmation

    if score.reset_intent >= 0.5:
        evidence += 0.1 + 0.15 * score.reset_intent

    if score.official_change >= 0.5:
        evidence += 0.05 + 0.1 * score.official_change

    # 用户抱怨 limit 本身不是 reset 证据，但大量抱怨可微弱提升证据
    if score.limit_complaint >= 0.7:
        evidence += 0.05 + 0.05 * score.limit_complaint

    return min(evidence, 1.0)


def _aggregate_weighted_evidence(
    tweets: list[Tweet],
    signal_scores: list[SignalScores],
    now: Optional[datetime] = None,
) -> dict:
    """
    使用 authority_score、recency_weight 和来源优先级聚合证据。

    返回：
        {
            "tibo": float,
            "openai": float,
            "community": float,
            "overall": float,
            "tibo_signal": float,
            "community_signal": float,
            "release_signal": float,
        }
    """
    if not signal_scores:
        return {
            "tibo": 0.0,
            "openai": 0.0,
            "community": 0.0,
            "overall": 0.0,
            "tibo_signal": 0.0,
            "community_signal": 0.0,
            "release_signal": 0.0,
        }

    if now is None:
        now = datetime.now(timezone.utc)

    use_default_weights = not tweets or len(tweets) != len(signal_scores)

    category_sums: dict[str, float] = {
        "tibo": 0.0,
        "openai": 0.0,
        "community": 0.0,
    }
    category_weights: dict[str, float] = {
        "tibo": 0.0,
        "openai": 0.0,
        "community": 0.0,
    }

    signal_sums = {
        "tibo": 0.0,
        "community": 0.0,
        "release": 0.0,
    }
    signal_weight_sum = 0.0

    for i, score in enumerate(signal_scores):
        if use_default_weights:
            authority = 1.0
            recency = 1.0
            source = "unknown"
        else:
            tweet = tweets[i]
            authority = max(0.0, min(1.0, tweet.authority_score))
            recency = _recency_weight(tweet.timestamp, now)
            source = tweet.source

        priority = _source_priority(source)
        w = authority * recency * priority
        evidence = _per_tweet_evidence(score) * score.confidence

        # 按来源聚合证据
        if "tibo" in source.lower():
            category = "tibo"
        elif "openai" in source.lower():
            category = "openai"
        else:
            category = "community"

        category_sums[category] += evidence * w
        category_weights[category] += w

        # 同时保留语义信号（用于解释和兼容）
        tibo = (
            0.6 * score.reset_confirmation
            + 0.3 * score.reset_intent
            + 0.1 * score.official_change
        )
        community = score.limit_complaint
        release = score.official_change
        signal_sums["tibo"] += tibo * w
        signal_sums["community"] += community * w
        signal_sums["release"] += release * w
        signal_weight_sum += w

    category_scores: dict[str, float] = {}
    for cat in ["tibo", "openai", "community"]:
        # 保留来源优先级的影响：使用加权证据和，而非按权重归一化
        category_scores[cat] = min(category_sums[cat], 1.0)

    # 综合证据：按来源优先级加权，并仅按实际存在的来源归一化
    source_weights = {
        "tibo": 1.0,
        "openai": 0.8,
        "community": 0.3,
    }
    active_weight_sum = 0.0
    weighted_overall = 0.0
    for source, weight in source_weights.items():
        score = category_scores[source]
        if score > 0.0:
            weighted_overall += score * weight
            active_weight_sum += weight
    if active_weight_sum > 0.0:
        category_scores["overall"] = min(weighted_overall / active_weight_sum, 1.0)
    else:
        category_scores["overall"] = 0.0

    signal_results = {
        "tibo_signal": 0.0,
        "community_signal": 0.0,
        "release_signal": 0.0,
    }
    if signal_weight_sum > 0:
        signal_results["tibo_signal"] = min(
            signal_sums["tibo"] / signal_weight_sum, 1.0
        )
        signal_results["community_signal"] = min(
            signal_sums["community"] / signal_weight_sum, 1.0
        )
        signal_results["release_signal"] = min(
            signal_sums["release"] / signal_weight_sum, 1.0
        )

    return {
        **category_scores,
        **signal_results,
    }


def _base_probability(horizon: int, time_pressure: float) -> float:
    """
    无信号时的弱时间先验概率。

    时间压力低 → 概率降低；时间压力高 → 概率小幅提升。
    但 24h/48h 不会因此超过 0.5/0.7。
    """
    base = BASE_PROBABILITY.get(horizon, 0.1)
    # time_pressure ∈ [0,1]，调整幅度 ±TIME_ADJUSTMENT_STRENGTH
    adjustment = TIME_ADJUSTMENT_STRENGTH * (time_pressure - 0.5)
    adjusted = base * (1.0 + adjustment)
    return max(0.01, min(adjusted, MAX_PROBABILITY_NO_SIGNAL[horizon]))


def _evidence_multiplier(evidence_score: float) -> float:
    """证据分数 → odds 乘数。"""
    return 1.0 + evidence_score * (MAX_EVIDENCE_MULTIPLIER - 1.0)


def _bayesian_update(prior: float, evidence_score: float) -> float:
    """
    Bayesian odds update。

    posterior_odds = prior_odds * evidence_multiplier
    """
    if prior <= 0.0:
        prior = 0.001
    if prior >= 1.0:
        return prior

    odds = prior / (1.0 - prior)
    multiplier = _evidence_multiplier(evidence_score)
    posterior = (odds * multiplier) / (1.0 + odds * multiplier)
    return posterior


def _probabilities(
    time_pressure: float,
    evidence_score: float,
    horizons: list[int],
) -> dict[str, float]:
    """计算各时间窗口的后验概率。"""
    probability: dict[str, float] = {}
    for h in horizons:
        prior = _base_probability(h, time_pressure)
        posterior = _bayesian_update(prior, evidence_score)

        # 根据证据强度选择 cap
        if evidence_score >= 0.7:
            cap = MAX_PROBABILITY_STRONG_EVIDENCE[h]
        elif evidence_score >= 0.4:
            cap = MAX_PROBABILITY_NO_SIGNAL[h] + (
                MAX_PROBABILITY_STRONG_EVIDENCE[h] - MAX_PROBABILITY_NO_SIGNAL[h]
            ) * (evidence_score - 0.4) / 0.3
        else:
            cap = MAX_PROBABILITY_NO_SIGNAL[h]

        prob = min(cap, posterior)
        probability[f"{h}h"] = round(prob, 4)
    return probability


def _format_interval(interval: Optional[float], default: float) -> str:
    """格式化间隔显示。"""
    if interval is not None and interval > 0:
        return f"{interval:.0f} 小时"
    return f"{default:.0f} 小时（默认）"


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
    从分析特征和 LLM 信号分数构建 PredictionFeatures（V2）。
    """
    prior_interval = DEFAULT_RESET_INTERVAL_HOURS

    if model_state is not None:
        posterior_avg = model_state.average_interval_hours
        posterior_median = model_state.median_interval_hours or prior_interval
        posterior_uncertainty = (
            model_state.interval_uncertainty or MIN_UNCERTAINTY_HOURS
        )
    else:
        posterior_avg = _compute_posterior_value(
            average_reset_interval, prior_interval, interval_count
        )
        posterior_median = _compute_posterior_value(
            median_reset_interval, prior_interval, interval_count
        )
        empirical_uncertainty = (
            interval_uncertainty if interval_uncertainty is not None else None
        )
        posterior_uncertainty = _compute_posterior_value(
            empirical_uncertainty, prior_interval * 0.25, interval_count
        )

    hours_since = (
        hours_since_last_reset
        if hours_since_last_reset is not None
        else posterior_median
    )

    time_pressure = _compute_time_pressure(
        hours_since, posterior_median, posterior_uncertainty
    )

    evidence = _aggregate_weighted_evidence(
        tweets or [], signal_scores or [], now=now
    )

    return PredictionFeatures(
        hours_since_last_reset=hours_since,
        average_reset_interval=posterior_avg,
        median_reset_interval=posterior_median,
        interval_uncertainty=posterior_uncertainty,
        time_pressure=round(time_pressure, 4),
        tibo_signal=evidence["tibo_signal"],
        community_signal=evidence["community_signal"],
        release_signal=evidence["release_signal"],
        evidence_score=evidence["overall"],
    )


# ──────────────────────────────────────────────
# 核心预测器
# ──────────────────────────────────────────────

class ResetPredictor:
    """
    Adaptive Bayesian Evidence Model 预测器（V2）。

    信号证据主导，时间因素仅作为弱先验修正。
    """

    def __init__(
        self,
        params: Optional[dict[str, float]] = None,
        horizons: Optional[list[int]] = None,
        default_interval: float = DEFAULT_RESET_INTERVAL_HOURS,
        model_state: Optional[ModelState] = None,
        model_state_path: Optional[Path] = None,
    ):
        self._default_interval = default_interval
        self._model_state = self._load_model_state(model_state, model_state_path)
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
        return "adaptive-bayesian-evidence-2.0.0"

    @property
    def model_state(self) -> Optional[ModelState]:
        """返回当前使用的模型状态（可能为 None）"""
        return self._model_state

    def predict(self, features: PredictionFeatures) -> PredictionExplanation:
        """根据输入特征预测各时间窗口的 reset 概率。"""
        self._ensure_time_features(features)

        time_ratio = self._compute_time_ratio(features)
        probability = _probabilities(
            features.time_pressure,
            features.evidence_score,
            self._horizons,
        )

        main_factors = self._build_main_factors(features)
        reasons = self._generate_reasons(features, time_ratio, main_factors)

        if self._model_state is not None:
            prior_applied = self._model_state.prior_weight > 0.0
        else:
            prior_applied = (
                features.average_reset_interval is None
                or features.average_reset_interval == self._default_interval
            )

        # hazard_rate 保留用于兼容性，用 posterior 24h 概率反推等效每小时 hazard
        prob_24h = probability.get("24h", 0.0)
        hazard = self._equivalent_hazard(prob_24h, 24)

        return PredictionExplanation(
            probability=probability,
            reasons=reasons,
            main_factors=main_factors,
            evidence_score=round(features.evidence_score, 4),
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

    def _ensure_time_features(self, features: PredictionFeatures) -> None:
        """补齐 median / uncertainty / time_pressure。"""
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
        features.time_pressure = _compute_time_pressure(
            hours_since, median, uncertainty
        )

    def _compute_time_ratio(self, features: PredictionFeatures) -> Optional[float]:
        """计算 time_ratio（仅用于展示），无历史时使用先验默认值。"""
        hours_since = features.hours_since_last_reset
        interval = features.average_reset_interval
        if interval is None or interval <= 0:
            interval = self._default_interval
        if hours_since is None:
            hours_since = self._default_interval
        return hours_since / interval

    def _equivalent_hazard(self, prob: float, hours: int) -> float:
        """从 T 小时概率反推等效恒定每小时 hazard。"""
        if prob <= 0.0:
            return 0.0
        if prob >= 1.0:
            return 1.0
        return 1.0 - (1.0 - prob) ** (1.0 / hours)

    def _build_main_factors(self, features: PredictionFeatures) -> list[FactorImpact]:
        """构建对最终概率影响最大的结构化因素列表。"""
        factors: list[FactorImpact] = []

        # 时间因素
        if features.hours_since_last_reset is None:
            factors.append(
                FactorImpact(
                    factor="无历史 reset 记录",
                    impact="使用默认先验周期",
                )
            )
        else:
            if features.time_pressure < 0.2:
                impact = "-5%"
            elif features.time_pressure < 0.5:
                impact = "+0%"
            elif features.time_pressure < 0.8:
                impact = "+5%"
            else:
                impact = "+10%"
            factors.append(
                FactorImpact(
                    factor=f"距上次 reset {features.hours_since_last_reset:.1f} 小时",
                    impact=impact,
                    score=round(features.time_pressure, 2),
                )
            )

        # 信号因素
        if features.evidence_score > 0.0:
            if features.tibo_signal >= 0.5:
                factors.append(
                    FactorImpact(
                        factor="Tibo/Reset 确认信号强",
                        impact=f"+{int(min(features.tibo_signal * 50, 50))}%",
                        score=round(features.tibo_signal, 2),
                    )
                )
            elif features.tibo_signal > 0.0:
                factors.append(
                    FactorImpact(
                        factor="存在一定 reset 讨论",
                        impact=f"+{int(features.tibo_signal * 20)}%",
                        score=round(features.tibo_signal, 2),
                    )
                )

            if features.community_signal >= 0.5:
                factors.append(
                    FactorImpact(
                        factor="社区 limit 抱怨较高",
                        impact=f"+{int(min(features.community_signal * 15, 15))}%",
                        score=round(features.community_signal, 2),
                    )
                )
            elif features.community_signal > 0.0:
                factors.append(
                    FactorImpact(
                        factor="社区存在少量 limit 抱怨",
                        impact=f"+{int(features.community_signal * 8)}%",
                        score=round(features.community_signal, 2),
                    )
                )

            if features.release_signal >= 0.5:
                factors.append(
                    FactorImpact(
                        factor="官方发布/变更信号",
                        impact=f"+{int(min(features.release_signal * 20, 20))}%",
                        score=round(features.release_signal, 2),
                    )
                )
        else:
            factors.append(
                FactorImpact(
                    factor="无显著 reset 信号",
                    impact="概率受时间先验限制",
                )
            )

        return factors

    def _generate_reasons(
        self,
        features: PredictionFeatures,
        time_ratio: Optional[float],
        main_factors: list[FactorImpact],
    ) -> list[str]:
        """生成人类可读的预测原因列表。"""
        reasons: list[str] = []

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
                f"接近历史中位间隔 {_format_interval(features.median_reset_interval, self._default_interval)}"
            )
        elif features.time_pressure < 0.8:
            reasons.append(
                f"距上次 reset 已 {features.hours_since_last_reset:.1f} 小时，"
                f"超过历史中位间隔 {_format_interval(features.median_reset_interval, self._default_interval)}，"
                f"但概率仍由信号证据主导"
            )
        else:
            reasons.append(
                f"距上次 reset 已 {features.hours_since_last_reset:.1f} 小时，"
                f"明显超过历史中位间隔 {_format_interval(features.median_reset_interval, self._default_interval)}，"
                f"时间因素小幅提升基线概率"
            )

        if features.evidence_score >= 0.7:
            reasons.append(
                f"检测到强 reset 证据（evidence_score={features.evidence_score:.2f}），"
                f"概率显著上升"
            )
        elif features.evidence_score >= 0.4:
            reasons.append(
                f"检测到中等 reset 证据（evidence_score={features.evidence_score:.2f}）"
            )
        elif features.evidence_score > 0.0:
            reasons.append(
                f"检测到微弱 reset 证据（evidence_score={features.evidence_score:.2f}），"
                f"不足以确认"
            )
        else:
            reasons.append("未检测到显著 reset 信号，概率受时间先验限制")

        if main_factors:
            top = main_factors[0]
            reasons.append(f"主要因素：{top.factor}（{top.impact}）")

        return reasons


__all__ = [
    "ResetPredictor",
    "build_features",
    "PREDICTION_HORIZONS",
    "DEFAULT_RESET_INTERVAL_HOURS",
    "INTERVAL_PRIOR_STRENGTH",
    "MAX_EVIDENCE_MULTIPLIER",
]
