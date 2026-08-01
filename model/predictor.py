"""
WillTiboReset - Predictor framework

Defines predictor interfaces and placeholder implementations.
The real prediction logic will be filled in later phases by the LLM analyzer
or statistical model; this module only provides an extension skeleton.
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
    Abstract base class for predictors.

    All predictors (LLM predictor, statistical model predictor, etc.)
    should inherit from this class and implement the predict method.
    """

    @property
    def model_version(self) -> str:
        """Return the model version identifier; subclasses may override"""
        return "base-0.1.0"

    @abstractmethod
    def predict(
        self,
        tweets: list[Tweet],
        reset_events: list[ResetEvent],
        horizons: list[int],
    ) -> PredictionResult:
        """
        Generate predictions from input signals.

        Args:
            tweets: List of collected relevant tweets
            reset_events: List of historical reset events
            horizons: Prediction time window list (hours), e.g. [5, 24, 48]

        Returns:
            PredictionResult containing predictions per time window
        """
        ...


class PlaceholderPredictor(BasePredictor):
    """
    Placeholder predictor.

    Contains no real prediction logic; used only to verify pipeline connectivity.
    Calling predict raises NotImplementedError to clearly indicate that
    prediction logic is not yet implemented.
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
            "Prediction logic is not yet implemented. "
            "Please implement LLMPredictor or StatisticalPredictor in a later phase."
        )


# ──────────────────────────────────────────────
# Reserved extension interfaces (to be implemented in later phases)
# ──────────────────────────────────────────────

class LLMPredictor(BasePredictor):
    """
    LLM predictor (reserved).

    Will use the OpenAI API to perform natural-language analysis on collected signals
    and output structured prediction results. To be implemented in a later phase.
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
        raise NotImplementedError("LLMPredictor is not yet implemented; it will be completed in a later phase.")


class StatisticalPredictor(BasePredictor):
    """
    Statistical model predictor (reserved).

    Will build a statistical prediction model based on
    time-series patterns of historical reset events. To be implemented in a later phase.
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
        raise NotImplementedError("StatisticalPredictor is not yet implemented; it will be completed in a later phase.")
