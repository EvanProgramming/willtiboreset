"""
model module - data models and predictors

Re-exports all data models and predictor classes
for use by other modules.
"""

from model.data_models import (
    HorizonPrediction,
    PredictionExplanation,
    PredictionFeatures,
    PredictionHorizon,
    PredictionResult,
    ResetEvent,
    SignalScores,
    SignalSource,
    Tweet,
)
from model.predictor import (
    BasePredictor,
    LLMPredictor,
    PlaceholderPredictor,
    StatisticalPredictor,
)
from model.survival_model import (
    BASE_PROBABILITY,
    DEFAULT_RESET_INTERVAL_HOURS,
    MAX_EVIDENCE_MULTIPLIER,
    PREDICTION_HORIZONS,
    ResetPredictor,
    build_features,
)

__all__ = [
    # Data models
    "ResetEvent",
    "Tweet",
    "PredictionResult",
    "HorizonPrediction",
    "SignalScores",
    "SignalSource",
    "PredictionHorizon",
    "PredictionFeatures",
    "PredictionExplanation",
    # Predictors
    "BasePredictor",
    "PlaceholderPredictor",
    "LLMPredictor",
    "StatisticalPredictor",
    # Survival model
    "ResetPredictor",
    "build_features",
    "BASE_PROBABILITY",
    "DEFAULT_RESET_INTERVAL_HOURS",
    "MAX_EVIDENCE_MULTIPLIER",
    "PREDICTION_HORIZONS",
]
