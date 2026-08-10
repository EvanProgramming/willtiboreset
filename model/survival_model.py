"""
WillTiboReset - Core Prediction Engine V2

Adaptive Bayesian Evidence Model.

V2 core changes:
    1. Switched from "time-driven" to "evidence-driven + weak time prior".
    2. LLM signals are aggregated by source authority (authority_score) and recency (recency_weight),
       producing a combined evidence_score.
    3. Uses Bayesian odds update to lift the prior baseline probability to a posterior probability.
    4. Time factor only serves as a weak prior correction: recent reset lowers probability, overdue slightly raises it,
       but without signals it will not cause high probability.
    5. Outputs structured main_factors explaining each factor's contribution to the probability.

Inputs:
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

Outputs:
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
# Model default parameters
# ──────────────────────────────────────────────

DEFAULT_RESET_INTERVAL_HOURS: float = 48.0
INTERVAL_PRIOR_STRENGTH: float = 2.0
PREDICTION_HORIZONS: list[int] = [5, 24, 48]

# Prior baseline probability when no signal is present (weak time prior)
BASE_PROBABILITY: dict[int, float] = {
    5: 0.05,
    24: 0.18,
    48: 0.28,
}

# Maximum adjustment range of time factor on baseline (±30%)
TIME_ADJUSTMENT_STRENGTH: float = 0.30

# Evidence multiplier: maximum odds amplification when evidence_score=1
# Evidence_score is combined with a nonlinear curve so that only genuinely
# strong signals produce large amplification, while weak signals stay modest.
MAX_EVIDENCE_MULTIPLIER: float = 25.0

# Probability caps per horizon: prevent natural inflation over time without evidence
MAX_PROBABILITY_NO_SIGNAL: dict[int, float] = {
    5: 0.20,
    24: 0.50,
    48: 0.70,
}

# Upper limits under strong evidence
MAX_PROBABILITY_STRONG_EVIDENCE: dict[int, float] = {
    5: 0.75,
    24: 0.95,
    48: 0.98,
}

# Recency decay parameters
RECENCY_DECAY_HOURS: float = 24.0
MIN_UNCERTAINTY_HOURS: float = 6.0

# Weekly cycle: how much the day-of-week factor can boost base probability
WEEKLY_CYCLE_BOOST_STRENGTH: float = 0.6  # Max 60% boost on high-factor days


# ──────────────────────────────────────────────
# Helper functions
# ──────────────────────────────────────────────

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid function"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


def _compute_posterior_value(
    empirical_value: Optional[float],
    prior_value: float,
    interval_count: Optional[int] = None,
) -> float:
    """Blend prior and observed values using Bayesian shrinkage."""
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
    Smooth time pressure function.

    When hours_since is much smaller than median → 0
    When hours_since is close to median → 0.5
    When hours_since is much larger than median → 1
    """
    if median_interval <= 0:
        return 0.0
    denom = max(uncertainty, MIN_UNCERTAINTY_HOURS)
    x = (hours_since - median_interval) / denom
    return _sigmoid(x)


def _recency_weight(tweet_timestamp: datetime, now: datetime) -> float:
    """Compute recency weight based on message age."""
    if tweet_timestamp.tzinfo is None:
        tweet_timestamp = tweet_timestamp.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    hours_old = (now - tweet_timestamp).total_seconds() / 3600.0
    hours_old = max(0.0, hours_old)
    return math.exp(-hours_old / RECENCY_DECAY_HOURS)


def _source_priority(source: str) -> float:
    """Source priority weights: Tibo > OpenAI > Community."""
    source_lower = source.lower()
    if "tibo" in source_lower:
        return 1.0
    if "openai" in source_lower:
        return 0.8
    return 0.4


def _per_tweet_evidence(score: SignalScores) -> float:
    """
    Compute evidence strength (0-1) from a single LLM signal.

    Emphasizes reset_confirmation and suppresses limit_complaint.
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

    # User complaints about limit themselves are not reset evidence, but many complaints can slightly raise evidence
    if score.limit_complaint >= 0.7:
        evidence += 0.05 + 0.05 * score.limit_complaint

    return min(evidence, 1.0)


def _aggregate_weighted_evidence(
    tweets: list[Tweet],
    signal_scores: list[SignalScores],
    now: Optional[datetime] = None,
    recent_reset_time: Optional[datetime] = None,
) -> dict:
    """
    Aggregate evidence using authority_score, recency_weight, and source priority.

    recent_reset_time: timestamp of the most recent confirmed reset. Tweets that
    merely confirm a reset which has ALREADY been recorded (past-tense
    announcements, e.g. "I have reset usage limits") must not be counted as
    evidence for a FUTURE reset, so their contribution is heavily dampened.

    Returns:
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
            tweet_time = None
        else:
            tweet = tweets[i]
            authority = max(0.0, min(1.0, tweet.authority_score))
            recency = _recency_weight(tweet.timestamp, now)
            source = tweet.source
            tweet_time = tweet.timestamp

        priority = _source_priority(source)
        w = authority * recency * priority
        evidence = _per_tweet_evidence(score) * score.confidence

        # Dampen evidence from tweets that confirm a reset which has already
        # been recorded in history. A past-tense confirmation ("I have reset
        # usage limits") describes a reset that ALREADY happened, so it must not
        # push up the probability of ANOTHER reset in the future.
        past_confirmation_dampen = 1.0
        if (
            recent_reset_time is not None
            and tweet_time is not None
            and score.reset_confirmation >= 0.5
        ):
            ts = tweet_time
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            rs = recent_reset_time
            if rs.tzinfo is None:
                rs = rs.replace(tzinfo=timezone.utc)
            # Tweet published within 24h before the confirmed reset, or 6h after it,
            # is very likely the announcement of that same reset (not a new one).
            before = (rs - ts).total_seconds() / 3600.0
            after = (ts - rs).total_seconds() / 3600.0
            if -24.0 <= after <= 6.0 and before <= 24.0:
                past_confirmation_dampen = 0.1

        evidence *= past_confirmation_dampen

        # Aggregate evidence by source
        if "tibo" in source.lower():
            category = "tibo"
        elif "openai" in source.lower():
            category = "openai"
        else:
            category = "community"

        category_sums[category] += evidence * w
        category_weights[category] += w

        # Also retain semantic signals (for explanation and compatibility)
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
        # Preserve source priority effect: use weighted evidence sum rather than normalizing by weight
        category_scores[cat] = min(category_sums[cat], 1.0)

    # Combined evidence: weighted by source priority and normalized only by active sources
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


def _base_probability(
    horizon: int,
    time_pressure: float,
    weekly_cycle_factor: float = 0.0,
) -> float:
    """
    Weak time prior probability when no signal is present.

    Low time pressure → lower probability; high time pressure → slightly higher probability.
    Weekly cycle factor boosts the base when current day is a typical reset day.
    """
    base = BASE_PROBABILITY.get(horizon, 0.1)
    # time_pressure ∈ [0,1]; adjustment range ±TIME_ADJUSTMENT_STRENGTH
    adjustment = TIME_ADJUSTMENT_STRENGTH * (time_pressure - 0.5)
    adjusted = base * (1.0 + adjustment)
    # Weekly cycle boost: up to WEEKLY_CYCLE_BOOST_STRENGTH multiplicative boost
    cycle_boost = 1.0 + weekly_cycle_factor * WEEKLY_CYCLE_BOOST_STRENGTH
    adjusted *= cycle_boost
    return max(0.01, min(adjusted, MAX_PROBABILITY_NO_SIGNAL[horizon]))


def _evidence_multiplier(evidence_score: float) -> float:
    """Evidence score → odds multiplier."""
    return 1.0 + evidence_score * (MAX_EVIDENCE_MULTIPLIER - 1.0)


def _bayesian_update(prior: float, evidence_score: float) -> float:
    """
    Bayesian odds update.

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
    weekly_cycle_factor: float = 0.0,
) -> dict[str, float]:
    """Compute posterior probability for each time horizon."""
    probability: dict[str, float] = {}
    for h in horizons:
        prior = _base_probability(h, time_pressure, weekly_cycle_factor)
        posterior = _bayesian_update(prior, evidence_score)

        # Raise no-signal cap when weekly cycle factor is high:
        # on a typical reset day, probability should not be artificially
        # capped at the same level as a non-reset day.
        cycle_cap_boost = weekly_cycle_factor * 0.5  # up to 50% of gap
        base_cap = MAX_PROBABILITY_NO_SIGNAL[h] + (
            MAX_PROBABILITY_STRONG_EVIDENCE[h] - MAX_PROBABILITY_NO_SIGNAL[h]
        ) * cycle_cap_boost

        # Choose cap based on evidence strength
        if evidence_score >= 0.7:
            cap = MAX_PROBABILITY_STRONG_EVIDENCE[h]
        elif evidence_score >= 0.4:
            cap = base_cap + (
                MAX_PROBABILITY_STRONG_EVIDENCE[h] - base_cap
            ) * (evidence_score - 0.4) / 0.3
        else:
            cap = base_cap

        prob = min(cap, posterior)
        probability[f"{h}h"] = round(prob, 4)
    return probability


def _format_interval(interval: Optional[float], default: float) -> str:
    """Format interval for display."""
    if interval is not None and interval > 0:
        return f"{interval:.0f} hours"
    return f"{default:.0f} hours (default)"


# ──────────────────────────────────────────────
# Feature builder
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
    recent_reset_time: Optional[datetime] = None,
    now: Optional[datetime] = None,
    expected_weekly_interval_hours: Optional[float] = None,
    weekly_cycle_factor: float = 0.0,
) -> PredictionFeatures:
    """
    Build PredictionFeatures (V2) from analysis features and LLM signal scores.
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
        tweets or [], signal_scores or [], now=now,
        recent_reset_time=recent_reset_time,
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
        expected_weekly_interval_hours=expected_weekly_interval_hours,
        weekly_cycle_factor=weekly_cycle_factor,
    )


# ──────────────────────────────────────────────
# Core predictor
# ──────────────────────────────────────────────

class ResetPredictor:
    """
    Adaptive Bayesian Evidence Model predictor (V2).

    Signal evidence dominates; time factor only serves as a weak prior correction.
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
        """Resolve model_state source: direct object takes priority, otherwise load from path."""
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
        """Return the currently used model state (may be None)."""
        return self._model_state

    def predict(self, features: PredictionFeatures) -> PredictionExplanation:
        """Predict reset probability for each time window based on input features."""
        self._ensure_time_features(features)

        time_ratio = self._compute_time_ratio(features)
        probability = _probabilities(
            features.time_pressure,
            features.evidence_score,
            self._horizons,
            features.weekly_cycle_factor,
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

        # hazard_rate kept for compatibility: back out equivalent hourly hazard from 24h posterior probability
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
        """Fill in median / uncertainty / time_pressure."""
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

        # Use expected weekly interval for time_pressure when available.
        # The historical median interval includes short intervals from early
        # double-reset days, which makes 145.7h seem "extremely overdue"
        # when it is actually right on schedule for a weekly pattern.
        time_interval = (
            features.expected_weekly_interval_hours
            if features.expected_weekly_interval_hours is not None
            else median
        )
        features.time_pressure = _compute_time_pressure(
            hours_since, time_interval, uncertainty
        )

    def _compute_time_ratio(self, features: PredictionFeatures) -> Optional[float]:
        """Compute time_ratio (for display only); use prior default when no history."""
        hours_since = features.hours_since_last_reset
        interval = features.average_reset_interval
        if interval is None or interval <= 0:
            interval = self._default_interval
        if hours_since is None:
            hours_since = self._default_interval
        return hours_since / interval

    def _equivalent_hazard(self, prob: float, hours: int) -> float:
        """Back out equivalent constant hourly hazard from T-hour probability."""
        if prob <= 0.0:
            return 0.0
        if prob >= 1.0:
            return 1.0
        return 1.0 - (1.0 - prob) ** (1.0 / hours)

    def _build_main_factors(self, features: PredictionFeatures) -> list[FactorImpact]:
        """Build a structured list of factors with the largest impact on the final probability."""
        factors: list[FactorImpact] = []

        # Time factor
        if features.hours_since_last_reset is None:
            factors.append(
                FactorImpact(
                    factor="No historical reset record",
                    impact="Using default prior interval",
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
                    factor=f"{features.hours_since_last_reset:.1f} hours since last reset",
                    impact=impact,
                    score=round(features.time_pressure, 2),
                )
            )

        # Weekly cycle factor
        if features.weekly_cycle_factor >= 0.5:
            impact_pct = int(features.weekly_cycle_factor * 30)
            factors.append(
                FactorImpact(
                    factor="Weekly cycle: today is a typical reset day (US Monday)",
                    impact=f"+{impact_pct}%",
                    score=round(features.weekly_cycle_factor, 2),
                )
            )
        elif features.weekly_cycle_factor >= 0.2:
            factors.append(
                FactorImpact(
                    factor="Weekly cycle: moderate reset-day proximity",
                    impact=f"+{int(features.weekly_cycle_factor * 15)}%",
                    score=round(features.weekly_cycle_factor, 2),
                )
            )

        # Signal factors
        if features.evidence_score > 0.0:
            if features.tibo_signal >= 0.5:
                factors.append(
                    FactorImpact(
                        factor="Strong Tibo/Reset confirmation signal",
                        impact=f"+{int(min(features.tibo_signal * 50, 50))}%",
                        score=round(features.tibo_signal, 2),
                    )
                )
            elif features.tibo_signal > 0.0:
                factors.append(
                    FactorImpact(
                        factor="Some reset discussion detected",
                        impact=f"+{int(features.tibo_signal * 20)}%",
                        score=round(features.tibo_signal, 2),
                    )
                )

            if features.community_signal >= 0.5:
                factors.append(
                    FactorImpact(
                        factor="High community limit complaints",
                        impact=f"+{int(min(features.community_signal * 15, 15))}%",
                        score=round(features.community_signal, 2),
                    )
                )
            elif features.community_signal > 0.0:
                factors.append(
                    FactorImpact(
                        factor="Minor community limit complaints",
                        impact=f"+{int(features.community_signal * 8)}%",
                        score=round(features.community_signal, 2),
                    )
                )

            if features.release_signal >= 0.5:
                factors.append(
                    FactorImpact(
                        factor="Official release/change signal",
                        impact=f"+{int(min(features.release_signal * 20, 20))}%",
                        score=round(features.release_signal, 2),
                    )
                )
        else:
            factors.append(
                FactorImpact(
                    factor="No significant reset signal",
                    impact="Probability constrained by time prior",
                )
            )

        return factors

    def _generate_reasons(
        self,
        features: PredictionFeatures,
        time_ratio: Optional[float],
        main_factors: list[FactorImpact],
    ) -> list[str]:
        """Generate human-readable prediction reasons."""
        reasons: list[str] = []

        if features.hours_since_last_reset is None:
            reasons.append(
                f"No historical reset record; using prior default interval {self._default_interval:.0f}h as baseline"
            )
        elif features.time_pressure < 0.2:
            reasons.append(
                f"Recently reset ({features.hours_since_last_reset:.1f} hours ago), "
                f"low time pressure ({features.time_pressure:.2f})"
            )
        elif features.time_pressure < 0.5:
            reasons.append(
                f"{features.hours_since_last_reset:.1f} hours since last reset, "
                f"close to historical median interval {_format_interval(features.median_reset_interval, self._default_interval)}"
            )
        elif features.time_pressure < 0.8:
            reasons.append(
                f"{features.hours_since_last_reset:.1f} hours since last reset, "
                f"exceeds historical median interval {_format_interval(features.median_reset_interval, self._default_interval)}, "
                f"but probability is still dominated by signal evidence"
            )
        else:
            reasons.append(
                f"{features.hours_since_last_reset:.1f} hours since last reset, "
                f"well beyond historical median interval {_format_interval(features.median_reset_interval, self._default_interval)}, "
                f"time factor slightly raises baseline probability"
            )

        if features.evidence_score >= 0.7:
            reasons.append(
                f"Strong reset evidence detected (evidence_score={features.evidence_score:.2f}), "
                f"probability rises significantly"
            )
        elif features.evidence_score >= 0.4:
            reasons.append(
                f"Moderate reset evidence detected (evidence_score={features.evidence_score:.2f})"
            )
        elif features.evidence_score > 0.0:
            reasons.append(
                f"Weak reset evidence detected (evidence_score={features.evidence_score:.2f}), "
                f"insufficient to confirm"
            )
        else:
            reasons.append("No significant reset signal detected; probability constrained by time prior")

        if features.weekly_cycle_factor >= 0.5:
            reasons.append(
                f"Weekly cycle boost active (factor={features.weekly_cycle_factor:.2f}): "
                f"today is a typical reset day, base probability and probability cap raised"
            )

        if main_factors:
            top = main_factors[0]
            reasons.append(f"Main factor: {top.factor} ({top.impact})")

        return reasons


__all__ = [
    "ResetPredictor",
    "build_features",
    "PREDICTION_HORIZONS",
    "DEFAULT_RESET_INTERVAL_HOURS",
    "INTERVAL_PRIOR_STRENGTH",
    "MAX_EVIDENCE_MULTIPLIER",
]
