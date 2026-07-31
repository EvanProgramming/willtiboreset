"""测试模型状态管理"""

from datetime import datetime, timezone
from pathlib import Path

from model.model_state import ModelState, ModelStateManager
from model.survival_model import DEFAULT_PARAMS, ResetPredictor


class TestModelState:
    """ModelState 数据模型测试"""

    def test_create_minimal(self):
        """创建最小 model_state"""
        state = ModelState(
            average_interval_hours=48.0,
            sample_count=0,
        )
        assert state.average_interval_hours == 48.0
        assert state.sample_count == 0
        assert state.prior_weight == 1.0

    def test_get_param_default(self):
        """读取不存在的参数返回默认值"""
        state = ModelState(
            average_interval_hours=48.0,
            sample_count=0,
        )
        assert state.get_param("alpha", -4.0) == -4.0

    def test_get_param_from_state(self):
        """读取已存在的参数"""
        state = ModelState(
            average_interval_hours=48.0,
            sample_count=5,
            params={"alpha": -3.5, "beta_time": 1.8},
        )
        assert state.get_param("alpha", -4.0) == -3.5
        assert state.get_param("beta_time", 1.5) == 1.8


class TestModelStateManager:
    """ModelStateManager 读写测试"""

    def test_save_and_load(self, tmp_path):
        """保存并加载 model_state"""
        state_path = tmp_path / "model_state.json"
        manager = ModelStateManager(state_path)

        state = ModelState(
            average_interval_hours=42.0,
            median_interval_hours=40.0,
            std_interval_hours=5.0,
            sample_count=10,
            interval_confidence=0.8,
            prior_weight=0.5,
            params={"alpha": -3.5},
        )
        manager.save(state)

        loaded = manager.load()
        assert loaded is not None
        assert loaded.average_interval_hours == 42.0
        assert loaded.sample_count == 10
        assert loaded.prior_weight == 0.5
        assert loaded.get_param("alpha", -4.0) == -3.5

    def test_load_missing(self, tmp_path):
        """文件不存在时返回 None"""
        manager = ModelStateManager(tmp_path / "missing.json")
        assert manager.load() is None


class TestResetPredictorWithModelState:
    """ResetPredictor 使用 model_state 的测试"""

    def test_uses_state_params(self):
        """优先使用 model_state 中的参数"""
        state = ModelState(
            average_interval_hours=48.0,
            sample_count=10,
            params={"alpha": -2.0},
        )
        predictor = ResetPredictor(model_state=state)
        assert predictor.params["alpha"] == -2.0
        assert predictor.params["beta_time"] == DEFAULT_PARAMS["beta_time"]

    def test_explicit_params_override_state(self):
        """传入 params 覆盖 model_state"""
        state = ModelState(
            average_interval_hours=48.0,
            sample_count=10,
            params={"alpha": -2.0},
        )
        predictor = ResetPredictor(model_state=state, params={"alpha": -1.0})
        assert predictor.params["alpha"] == -1.0

    def test_load_from_path(self, tmp_path):
        """从路径自动加载 model_state"""
        state = ModelState(
            average_interval_hours=36.0,
            sample_count=20,
            prior_weight=0.0,
            params={"beta_time": 2.0},
        )
        state_path = tmp_path / "model_state.json"
        ModelStateManager(state_path).save(state)

        predictor = ResetPredictor(model_state_path=state_path)
        assert predictor.model_state is not None
        assert predictor.model_state.sample_count == 20
        assert predictor.params["beta_time"] == 2.0
