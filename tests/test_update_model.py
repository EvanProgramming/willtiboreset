"""测试模型状态更新脚本"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from update_model import update_model_state


class TestUpdateModelState:
    """update_model_state 测试"""

    def test_no_history_file(self, tmp_path):
        """无历史数据时使用先验"""
        history_path = tmp_path / "reset_history.json"

        state = update_model_state(reset_history_path=history_path)
        assert state.sample_count == 0
        assert state.prior_weight == 1.0
        assert state.average_interval_hours == 48.0

    def test_with_history(self, tmp_path):
        """有历史数据时计算统计量"""
        history_path = tmp_path / "reset_history.json"
        base = datetime(2025, 7, 1, 12, 0, tzinfo=timezone.utc)
        events = [
            {"reset_time": (base - timedelta(hours=96)).isoformat(), "confidence": 1.0},
            {"reset_time": (base - timedelta(hours=48)).isoformat(), "confidence": 1.0},
            {"reset_time": base.isoformat(), "confidence": 1.0},
        ]
        history_path.write_text(json.dumps(events), encoding="utf-8")

        state = update_model_state(reset_history_path=history_path)
        assert state.sample_count == 2
        assert state.average_interval_hours == 48.0
        assert state.median_interval_hours == 48.0
        assert state.min_interval_hours == 48.0
        assert state.max_interval_hours == 48.0
        assert state.interval_confidence > 0.0

    def test_prior_weight_decreases_with_samples(self, tmp_path):
        """样本越多先验权重越低"""
        history_path = tmp_path / "reset_history.json"
        base = datetime(2025, 7, 1, 0, 0, tzinfo=timezone.utc)
        events = [
            {"reset_time": (base - timedelta(hours=i * 48)).isoformat()}
            for i in range(25, -1, -1)
        ]
        history_path.write_text(json.dumps(events), encoding="utf-8")

        state = update_model_state(reset_history_path=history_path)
        assert state.sample_count == 25
        assert state.prior_weight == 0.0
