"""
model 模块 - 数据模型与预测器

统一导出所有数据模型和预测器类，
供其他模块引用。
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
    # 数据模型
    "ResetEvent",
    "Tweet",
    "PredictionResult",
    "HorizonPrediction",
    "SignalScores",
    "SignalSource",
    "PredictionHorizon",
    "PredictionFeatures",
    "PredictionExplanation",
    # 预测器
    "BasePredictor",
    "PlaceholderPredictor",
    "LLMPredictor",
    "StatisticalPredictor",
    # 生存模型
    "ResetPredictor",
    "build_features",
    "BASE_PROBABILITY",
    "DEFAULT_RESET_INTERVAL_HOURS",
    "MAX_EVIDENCE_MULTIPLIER",
    "PREDICTION_HORIZONS",
]
