"""
WillTiboReset - Data model definitions

Defines all core data structures using pydantic v2.
These models are used throughout collection, analysis, prediction, and output,
providing a unified type contract for LLM analysis and prediction models.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ──────────────────────────────────────────────
# Enums
# ──────────────────────────────────────────────

class SignalSource(str, Enum):
    """Signal source type"""
    TWITTER = "twitter"
    REDDIT = "reddit"
    MANUAL = "manual"
    OPENAI_STATUS = "openai_status"
    OTHER = "other"


class PredictionHorizon(int, Enum):
    """Prediction time horizon (hours)"""
    HOURS_5 = 5
    HOURS_24 = 24
    HOURS_48 = 48


# ──────────────────────────────────────────────
# Core data models
# ──────────────────────────────────────────────

class ResetEvent(BaseModel):
    """
    Historical reset event record.

    Represents a confirmed or suspected usage quota reset event,
    used to train the prediction model and establish historical baseline patterns.
    """
    reset_time: datetime = Field(
        ..., description="Time the reset occurred (UTC)"
    )
    source: SignalSource = Field(
        ..., description="Information source of this event"
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Confidence of this reset event, 1.0 = fully confirmed, 0.0 = pure guess"
    )
    notes: str = Field(
        default="", description="Additional notes"
    )


class Tweet(BaseModel):
    """
    Unified signal data unit.

    The common data structure output by all Collectors (RSS, community, API, etc.).
    Downstream modules only depend on this structure, not on specific data sources.
    """
    timestamp: datetime = Field(
        ..., description="Publication time (UTC)"
    )
    author: str = Field(
        ..., min_length=1, description="Author username or site name"
    )
    text: str = Field(
        ..., min_length=1, description="Body content (title + summary)"
    )
    source: str = Field(
        default="unknown",
        description="Data source identifier, e.g. tibo_rss / openai_rss / community_mock"
    )
    url: Optional[str] = Field(
        default=None, description="Original link"
    )
    authority_score: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Data source authority score (Tibo=1.0, OpenAI=0.9, Community=0.5)"
    )


# ──────────────────────────────────────────────
# LLM signal analysis models
# ──────────────────────────────────────────────

class SignalScores(BaseModel):
    """
    LLM signal analysis output (V1.5).

    Converts natural language text into structured machine-learning features.
    These features are passed to model/survival_model.py as prediction input.

    The LLM does not directly predict reset; it only extracts signals.
    """
    reset_intent: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Signal strength that the text discusses/implies an upcoming quota reset"
    )
    limit_complaint: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Signal strength of user complaints about usage limits/quota exhaustion"
    )
    official_change: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Signal strength of official product changes, updates, or policy adjustments"
    )
    reset_confirmation: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Signal strength of explicit confirmation that reset has occurred or will occur (highest weight)"
    )
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Overall confidence of the LLM in the above scores"
    )
    reason: list[str] = Field(
        default_factory=list,
        description="List of scoring rationales"
    )

    def to_features(self) -> dict[str, float]:
        """Convert to a feature dict for survival_model.py"""
        return {
            "reset_intent": self.reset_intent,
            "limit_complaint": self.limit_complaint,
            "official_change": self.official_change,
            "reset_confirmation": self.reset_confirmation,
            "confidence": self.confidence,
        }


# ──────────────────────────────────────────────
# Survival model features and outputs
# ──────────────────────────────────────────────

class PredictionFeatures(BaseModel):
    """
    Survival model prediction input features (V1.5).

    Merged from AnalysisFeatures (statistical features) and SignalScores (LLM signals),
    used as input to ResetPredictor.predict().
    """
    hours_since_last_reset: Optional[float] = Field(
        default=None, ge=0.0,
        description="Hours since last reset, None means no historical record"
    )
    average_reset_interval: Optional[float] = Field(
        default=None, gt=0.0,
        description="Historical average reset interval (hours), None means no historical record"
    )
    median_reset_interval: Optional[float] = Field(
        default=None, gt=0.0,
        description="Historical median reset interval (hours), None means no historical record"
    )
    interval_uncertainty: Optional[float] = Field(
        default=None, ge=0.0,
        description="Uncertainty of reset interval estimate (hours), used to smooth time_pressure"
    )
    time_pressure: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Smoothed time pressure based on median interval (0=just reset, 1=significantly overdue)"
    )
    tibo_signal: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Tibo/Reset related signal strength (weighted reset_confirmation + reset_intent)"
    )
    community_signal: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Community pressure signal strength (weighted limit_complaint)"
    )
    release_signal: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Official change signal strength (weighted official_change)"
    )
    evidence_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Combined evidence strength (V2, aggregated from LLM signals and source weights)"
    )
    expected_weekly_interval_hours: Optional[float] = Field(
        default=None, gt=0.0,
        description="Expected weekly reset interval (168h); overrides median_interval for time_pressure calculation when set"
    )
    weekly_cycle_factor: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Day-of-week cycle factor (0=unlikely day, 1=very likely reset day); boosts base probability and raises no-signal cap"
    )
    explicit_future_reset: bool = Field(
        default=False,
        description="True when Tibo has explicitly announced an upcoming reset; pushes probability near max regardless of other signals"
    )


class FactorImpact(BaseModel):
    """
    A single factor influencing the final probability.

    Used in main_factors of prediction.json to help users understand the prediction rationale.
    """
    factor: str = Field(..., description="Factor description")
    impact: str = Field(..., description="Impact on probability, e.g. +35% / -10%")
    score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Raw strength of this factor (0-1)"
    )


class PredictionExplanation(BaseModel):
    """
    Survival model prediction output (with explanations, V2).

    Returns reset probabilities per time window, a list of key reasons driving the probability,
    and structured main_factors.
    """
    probability: dict[str, float] = Field(
        ..., description='Reset probability per time window, e.g. {"5h": 0.42, "24h": 0.76, "48h": 0.91}'
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Key reasons driving the probability (human-readable)"
    )
    main_factors: list[FactorImpact] = Field(
        default_factory=list,
        description="Structured list of factors with the largest impact on the final probability"
    )
    evidence_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Combined evidence strength (0=no evidence, 1=very strong evidence)"
    )
    hazard_rate: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Current hourly hazard rate (model internal state, kept in V2 for compatibility)"
    )
    time_pressure: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Smoothed time pressure (0=just reset, 1=significantly overdue)"
    )
    time_ratio: Optional[float] = Field(
        default=None,
        description="hours_since_last_reset / average_reset_interval, None means no history"
    )
    average_interval_used: Optional[float] = Field(
        default=None,
        description="Average reset interval actually used by the model (including prior)"
    )
    median_interval_used: Optional[float] = Field(
        default=None,
        description="Median reset interval actually used by the model (including prior)"
    )
    prior_applied: bool = Field(
        default=False,
        description="Whether the prior default interval was used"
    )


# ──────────────────────────────────────────────
# Prediction-related models
# ──────────────────────────────────────────────

class HorizonPrediction(BaseModel):
    """
    Prediction result for a single time window.

    Corresponds to a specific prediction horizon (e.g. 5 hours).
    """
    horizon_hours: int = Field(
        ..., description="Prediction time horizon (hours)"
    )
    will_reset: bool = Field(
        ..., description="Whether a reset is predicted to occur"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Prediction confidence"
    )
    reasoning: str = Field(
        default="", description="Prediction rationale (LLM analysis or model feature explanation)"
    )


class PredictionResult(BaseModel):
    """
    Complete prediction result.

    Contains predictions for multiple time windows, the list of signals used,
    and metadata such as model version.
    """
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Prediction generation time (UTC)"
    )
    predictions: list[HorizonPrediction] = Field(
        default_factory=list,
        description="List of prediction results per time window"
    )
    signals_used: list[str] = Field(
        default_factory=list,
        description="List of signal descriptions used in this prediction"
    )
    model_version: str = Field(
        default="unknown", description="Model version identifier that generated this prediction"
    )
    notes: str = Field(
        default="", description="Additional notes"
    )

    @field_validator("predictions")
    @classmethod
    def validate_predictions(cls, v: list[HorizonPrediction]) -> list[HorizonPrediction]:
        """Ensure each time window appears only once"""
        horizons = [p.horizon_hours for p in v]
        if len(horizons) != len(set(horizons)):
            raise ValueError("Duplicate prediction time windows")
        return v

    def get_prediction(self, horizon_hours: int) -> Optional[HorizonPrediction]:
        """Get prediction result by time window"""
        for p in self.predictions:
            if p.horizon_hours == horizon_hours:
                return p
        return None


# ──────────────────────────────────────────────
# Prediction history and calibration models (V2)
# ──────────────────────────────────────────────

class PredictionHistoryEntry(BaseModel):
    """
    Historical record of a single prediction.

    Used for subsequent calibration, performance evaluation, and future training data collection.
    """
    prediction_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Prediction generation time (UTC)"
    )
    prediction: dict[str, float] = Field(
        ..., description='Prediction probabilities per window, e.g. {"within_5h": 0.2, "within_24h": 0.5, "within_48h": 0.7}'
    )
    signals: dict = Field(
        default_factory=dict,
        description="Signal snapshot used for this prediction"
    )
    actual_result: Optional[bool] = Field(
        default=None,
        description="Whether reset actually occurred (True=occurred, False=did not occur, None=unconfirmed)"
    )
    resolved_at: Optional[datetime] = Field(
        default=None,
        description="Time the result was confirmed (UTC)"
    )


class CalibrationBin(BaseModel):
    """Probability calibration bin"""
    bin_start: float = Field(..., ge=0.0, le=1.0, description="Start of probability interval")
    bin_end: float = Field(..., ge=0.0, le=1.0, description="End of probability interval")
    predicted_mean: float = Field(..., ge=0.0, le=1.0, description="Mean predicted probability within interval")
    actual_frequency: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Actual occurrence frequency within interval"
    )
    count: int = Field(..., ge=0, description="Sample count within interval")


class HorizonPerformance(BaseModel):
    """Performance metrics for a single time window"""
    horizon_hours: int = Field(..., description="Time horizon (hours)")
    total: int = Field(..., ge=0, description="Total predictions with confirmed outcomes")
    brier_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Brier score (lower is better)"
    )
    accuracy: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Binary classification accuracy (threshold 0.5)"
    )
    calibration_error: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Mean calibration error"
    )
    bins: list[CalibrationBin] = Field(
        default_factory=list, description="Calibration bin details"
    )


class ModelPerformance(BaseModel):
    """
    Overall model performance report.

    Generated by calibration.py from prediction_history.json.
    """
    total_predictions: int = Field(..., ge=0, description="Total historical predictions")
    resolved_predictions: int = Field(..., ge=0, description="Predictions with confirmed outcomes")
    overall_brier_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Average Brier score across all windows"
    )
    overall_accuracy: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="Average accuracy across all windows"
    )
    horizons: list[HorizonPerformance] = Field(
        default_factory=list, description="Performance metrics per window"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Report update time (UTC)"
    )
