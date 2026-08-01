"""
WillTiboReset - Model state persistence

ModelState stores adaptive parameters learned from reset_history.json,
so ResetPredictor does not always have to use hardcoded coefficients.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class ModelState(BaseModel):
    """
    Persistent state for the Adaptive Bayesian Survival Model.

    Contains interval statistics computed from historical reset data,
    prior weight, and model coefficients.
    """

    average_interval_hours: float = Field(
        ..., description="Observed average reset interval (hours)"
    )
    median_interval_hours: Optional[float] = Field(
        default=None, description="Median reset interval (hours)"
    )
    std_interval_hours: Optional[float] = Field(
        default=None, description="Reset interval standard deviation (hours)"
    )
    min_interval_hours: Optional[float] = Field(
        default=None, description="Minimum reset interval (hours)"
    )
    max_interval_hours: Optional[float] = Field(
        default=None, description="Maximum reset interval (hours)"
    )
    interval_uncertainty: Optional[float] = Field(
        default=None, ge=0.0,
        description="Uncertainty of reset interval estimate (hours), used to smooth time_pressure"
    )
    sample_count: int = Field(
        ..., ge=0, description="Number of interval samples used for estimation"
    )
    interval_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Confidence in the average interval estimate"
    )
    prior_weight: float = Field(
        default=1.0, ge=0.0, le=1.0,
        description="Weight of the prior default interval in the posterior"
    )
    params: dict[str, float] = Field(
        default_factory=dict,
        description="Model parameters (e.g. alpha, beta_time, beta_tibo, etc.)"
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="State update time (UTC)"
    )

    def get_param(self, name: str, default: float) -> float:
        """Read a parameter, returning default if missing."""
        return self.params.get(name, default)


class ModelStateManager:
    """Manages reading and writing of model_state.json."""

    def __init__(self, state_path: Path):
        self._state_path = state_path

    def load(self) -> Optional[ModelState]:
        """Load ModelState from disk; return None if file does not exist."""
        if not self._state_path.exists():
            return None
        try:
            return ModelState.model_validate_json(
                self._state_path.read_text(encoding="utf-8")
            )
        except Exception:
            return None

    def save(self, state: ModelState) -> None:
        """Save ModelState to disk."""
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(
            state.model_dump_json(indent=2),
            encoding="utf-8",
        )
