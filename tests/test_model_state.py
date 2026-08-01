"""Tests for model state management"""

from datetime import datetime, timezone
from pathlib import Path

from model.model_state import ModelState, ModelStateManager
from model.survival_model import DEFAULT_RESET_INTERVAL_HOURS, ResetPredictor


class TestModelState:
    """Tests for the ModelState data model"""

    def test_create_minimal(self):
        """Create a minimal model_state"""
        state = ModelState(
            average_interval_hours=48.0,
            sample_count=0,
        )
        assert state.average_interval_hours == 48.0
        assert state.sample_count == 0
        assert state.prior_weight == 1.0

    def test_get_param_default(self):
        """Reading a missing parameter returns the default value"""
        state = ModelState(
            average_interval_hours=48.0,
            sample_count=0,
        )
        assert state.get_param("time_adjustment_strength", 0.30) == 0.30

    def test_get_param_from_state(self):
        """Reading an existing parameter"""
        state = ModelState(
            average_interval_hours=48.0,
            sample_count=5,
            params={"time_adjustment_strength": 0.35},
        )
        assert state.get_param("time_adjustment_strength", 0.30) == 0.35


class TestModelStateManager:
    """Tests for ModelStateManager read/write"""

    def test_save_and_load(self, tmp_path):
        """Save and load model_state"""
        state_path = tmp_path / "model_state.json"
        manager = ModelStateManager(state_path)

        state = ModelState(
            average_interval_hours=42.0,
            median_interval_hours=40.0,
            std_interval_hours=5.0,
            sample_count=10,
            interval_confidence=0.8,
            prior_weight=0.5,
            params={"time_adjustment_strength": 0.35},
        )
        manager.save(state)

        loaded = manager.load()
        assert loaded is not None
        assert loaded.average_interval_hours == 42.0
        assert loaded.sample_count == 10
        assert loaded.prior_weight == 0.5
        assert loaded.get_param("time_adjustment_strength", 0.30) == 0.35

    def test_load_missing(self, tmp_path):
        """Return None when file does not exist"""
        manager = ModelStateManager(tmp_path / "missing.json")
        assert manager.load() is None


class TestResetPredictorWithModelState:
    """Tests for ResetPredictor using model_state"""

    def test_uses_state_interval(self):
        """Prefer interval statistics from model_state"""
        state = ModelState(
            average_interval_hours=36.0,
            median_interval_hours=34.0,
            sample_count=10,
            interval_uncertainty=6.0,
        )
        predictor = ResetPredictor(model_state=state)
        assert predictor.model_state is not None
        assert predictor.model_state.average_interval_hours == 36.0

    def test_load_from_path(self, tmp_path):
        """Auto-load model_state from path"""
        state = ModelState(
            average_interval_hours=36.0,
            sample_count=20,
            prior_weight=0.0,
        )
        state_path = tmp_path / "model_state.json"
        ModelStateManager(state_path).save(state)

        predictor = ResetPredictor(model_state_path=state_path)
        assert predictor.model_state is not None
        assert predictor.model_state.sample_count == 20
