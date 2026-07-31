"""测试生存模型预测引擎"""

import math

import pytest
from pydantic import ValidationError

from model.data_models import PredictionExplanation, PredictionFeatures, SignalScores
from model.survival_model import (
    DEFAULT_PARAMS,
    DEFAULT_RESET_INTERVAL_HOURS,
    PREDICTION_HORIZONS,
    ResetPredictor,
    _prob_within_window,
    _sigmoid,
    build_features,
)


# ──────────────────────────────────────────────
# 辅助函数测试
# ──────────────────────────────────────────────

class TestSigmoid:
    """sigmoid 函数测试"""

    def test_zero(self):
        """sigmoid(0) = 0.5"""
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_large_positive(self):
        """sigmoid(大正数) → 1"""
        assert _sigmoid(100.0) == pytest.approx(1.0)

    def test_large_negative(self):
        """sigmoid(大负数) → 0"""
        assert _sigmoid(-100.0) == pytest.approx(0.0)

    def test_symmetric(self):
        """sigmoid(-x) = 1 - sigmoid(x)"""
        for x in [0.5, 1.0, 2.0, 5.0]:
            assert _sigmoid(-x) == pytest.approx(1.0 - _sigmoid(x))

    def test_monotonic(self):
        """sigmoid 单调递增"""
        values = [-5, -2, -1, 0, 1, 2, 5]
        results = [_sigmoid(v) for v in values]
        for i in range(len(results) - 1):
            assert results[i] < results[i + 1]


class TestProbWithinWindow:
    """窗口概率函数测试"""

    def test_zero_hazard(self):
        """hazard=0 时概率为 0"""
        assert _prob_within_window(0.0, 24) == 0.0

    def test_full_hazard(self):
        """hazard=1 时概率为 1"""
        assert _prob_within_window(1.0, 5) == 1.0

    def test_monotonic_in_hazard(self):
        """固定窗口下，hazard 越大概率越高"""
        for h1, h2 in [(0.01, 0.05), (0.05, 0.1), (0.1, 0.3)]:
            assert _prob_within_window(h1, 24) < _prob_within_window(h2, 24)

    def test_monotonic_in_window(self):
        """固定 hazard 下，窗口越大概率越高"""
        h = 0.05
        assert _prob_within_window(h, 5) < _prob_within_window(h, 24)
        assert _prob_within_window(h, 24) < _prob_within_window(h, 48)

    def test_known_value(self):
        """已知值验证：h=0.1, T=5 → 1-(0.9)^5 ≈ 0.4095"""
        expected = 1 - (0.9 ** 5)
        assert _prob_within_window(0.1, 5) == pytest.approx(expected, rel=1e-4)


# ──────────────────────────────────────────────
# 数据模型测试
# ──────────────────────────────────────────────

class TestPredictionFeatures:
    """PredictionFeatures 模型测试"""

    def test_create_with_all_fields(self):
        """所有字段都提供"""
        features = PredictionFeatures(
            hours_since_last_reset=20.0,
            average_reset_interval=24.0,
            tibo_signal=0.8,
            community_signal=0.3,
            release_signal=0.1,
        )
        assert features.hours_since_last_reset == 20.0
        assert features.tibo_signal == 0.8

    def test_defaults(self):
        """默认值：时间字段为 None，信号为 0"""
        features = PredictionFeatures()
        assert features.hours_since_last_reset is None
        assert features.average_reset_interval is None
        assert features.tibo_signal == 0.0
        assert features.community_signal == 0.0
        assert features.release_signal == 0.0

    def test_signal_out_of_range(self):
        """信号超出 [0, 1] 应报错"""
        with pytest.raises(ValidationError):
            PredictionFeatures(tibo_signal=1.5)
        with pytest.raises(ValidationError):
            PredictionFeatures(community_signal=-0.1)

    def test_negative_hours_rejected(self):
        """负小时数应报错"""
        with pytest.raises(ValidationError):
            PredictionFeatures(hours_since_last_reset=-1.0)


class TestPredictionExplanation:
    """PredictionExplanation 模型测试"""

    def test_create_valid(self):
        """创建有效的解释对象"""
        exp = PredictionExplanation(
            probability={"5h": 0.42, "24h": 0.76, "48h": 0.91},
            reasons=["原因1", "原因2"],
            hazard_rate=0.12,
            time_ratio=1.8,
        )
        assert exp.probability["5h"] == 0.42
        assert len(exp.reasons) == 2
        assert exp.hazard_rate == 0.12

    def test_hazard_out_of_range(self):
        """hazard_rate 超出 [0, 1] 应报错"""
        with pytest.raises(ValidationError):
            PredictionExplanation(
                probability={"5h": 0.5},
                hazard_rate=1.5,
            )


# ──────────────────────────────────────────────
# build_features 测试
# ──────────────────────────────────────────────

class TestBuildFeatures:
    """build_features 辅助函数测试"""

    def test_no_signals(self):
        """无信号时信号字段全为 0，观测充足时 posterior 接近经验均值"""
        features = build_features(20.0, 24.0, signal_scores=None, interval_count=100)
        assert features.hours_since_last_reset == 20.0
        assert features.average_reset_interval == pytest.approx(24.0, rel=1e-2)
        assert features.tibo_signal == 0.0
        assert features.community_signal == 0.0
        assert features.release_signal == 0.0

    def test_with_signals(self):
        """有信号时正确融合"""
        scores = [
            SignalScores(
                reset_signal=0.8,
                limit_discussion=0.6,
                release_signal=0.2,
                community_pressure=0.7,
                confidence=0.9,
            ),
            SignalScores(
                reset_signal=0.4,
                limit_discussion=0.2,
                release_signal=0.1,
                community_pressure=0.3,
                confidence=0.5,
            ),
        ]
        features = build_features(10.0, 24.0, signal_scores=scores)
        # tibo_signal = avg(0.6*0.8+0.4*0.6, 0.6*0.4+0.4*0.2) = avg(0.72, 0.32) = 0.52
        assert features.tibo_signal == pytest.approx(0.52, rel=1e-2)
        # community_signal = avg(0.7, 0.3) = 0.5
        assert features.community_signal == pytest.approx(0.5)
        # release_signal = avg(0.2, 0.1) = 0.15
        assert features.release_signal == pytest.approx(0.15)

    def test_no_history(self):
        """无历史数据时使用先验默认周期补齐，time_ratio 不空"""
        features = build_features(None, None)
        default_interval = DEFAULT_RESET_INTERVAL_HOURS
        assert features.hours_since_last_reset == pytest.approx(default_interval)
        assert features.average_reset_interval == pytest.approx(default_interval)
        assert features.hours_since_last_reset / features.average_reset_interval == pytest.approx(1.0)


# ──────────────────────────────────────────────
# ResetPredictor 核心测试
# ──────────────────────────────────────────────

class TestResetPredictorBasic:
    """ResetPredictor 基础功能测试"""

    def setup_method(self):
        self.predictor = ResetPredictor()

    def test_model_version(self):
        """模型版本标识"""
        assert "survival" in self.predictor.model_version

    def test_predict_returns_prediction_explanation(self):
        """predict 返回 PredictionExplanation"""
        features = PredictionFeatures(
            hours_since_last_reset=20.0,
            average_reset_interval=24.0,
        )
        result = self.predictor.predict(features)
        assert isinstance(result, PredictionExplanation)

    def test_probability_keys(self):
        """概率字典包含 5h/24h/48h 键"""
        features = PredictionFeatures(hours_since_last_reset=10.0, average_reset_interval=24.0)
        result = self.predictor.predict(features)
        assert "5h" in result.probability
        assert "24h" in result.probability
        assert "48h" in result.probability

    def test_probability_range(self):
        """所有概率在 [0, 1] 范围内"""
        features = PredictionFeatures(
            hours_since_last_reset=100.0,
            average_reset_interval=24.0,
            tibo_signal=1.0,
            community_signal=1.0,
            release_signal=1.0,
        )
        result = self.predictor.predict(features)
        for key, prob in result.probability.items():
            assert 0.0 <= prob <= 1.0, f"{key}={prob} 超出 [0,1]"

    def test_window_ordering(self):
        """5h < 24h < 48h（概率递增）"""
        features = PredictionFeatures(hours_since_last_reset=10.0, average_reset_interval=24.0)
        result = self.predictor.predict(features)
        assert result.probability["5h"] <= result.probability["24h"]
        assert result.probability["24h"] <= result.probability["48h"]

    def test_reasons_non_empty(self):
        """预测结果包含至少一个原因"""
        features = PredictionFeatures(hours_since_last_reset=10.0, average_reset_interval=24.0)
        result = self.predictor.predict(features)
        assert len(result.reasons) >= 1

    def test_hazard_rate_in_range(self):
        """hazard_rate 在 [0, 1] 范围内"""
        features = PredictionFeatures(hours_since_last_reset=10.0, average_reset_interval=24.0)
        result = self.predictor.predict(features)
        assert 0.0 <= result.hazard_rate <= 1.0


# ──────────────────────────────────────────────
# 场景测试（用户要求的测试案例）
# ──────────────────────────────────────────────

class TestResetPredictorScenarios:
    """
    多个测试场景，验证输出合理性。

    场景列表：
        1. 刚 reset 后 → 概率应很低
        2. 很久没 reset → 概率应很高
        3. Tibo 发布 limit 相关消息 → 概率应升高
        4. 无历史数据 → 仍能输出合理概率
        5. 高社区压力 → 概率应升高
        6. 刚 reset + 高信号 → 时间主导，概率仍较低
    """

    def setup_method(self):
        self.predictor = ResetPredictor()

    def test_scenario_1_just_reset(self):
        """场景 1：刚 reset 后（1 小时前），概率应很低"""
        features = PredictionFeatures(
            hours_since_last_reset=1.0,
            average_reset_interval=24.0,
        )
        result = self.predictor.predict(features)

        # 刚 reset，5h 内概率应低于 15%
        assert result.probability["5h"] < 0.15, (
            f"刚 reset 后 5h 概率 {result.probability['5h']} 过高"
        )
        # 24h 内概率也应较低
        assert result.probability["24h"] < 0.40
        # time_ratio 应 < 0.1
        assert result.time_ratio < 0.1
        # 原因应提到"刚 reset"
        assert any("刚 reset" in r or "较低" in r for r in result.reasons)

    def test_scenario_2_long_time_no_reset(self):
        """场景 2：很久没 reset（48 小时），概率应很高"""
        features = PredictionFeatures(
            hours_since_last_reset=48.0,
            average_reset_interval=24.0,
        )
        result = self.predictor.predict(features)

        # 超过平均间隔 2 倍，48h 概率应很高
        assert result.probability["48h"] > 0.85, (
            f"48h 无 reset 后 48h 概率 {result.probability['48h']} 过低"
        )
        # 24h 概率也应较高
        assert result.probability["24h"] > 0.70
        # time_ratio = 2.0
        assert result.time_ratio == pytest.approx(2.0, rel=1e-2)
        # 原因应提到"超过"或"远超"
        assert any("超" in r for r in result.reasons)

    def test_scenario_3_tibo_limit_message(self):
        """场景 3：Tibo 发布 limit 相关消息，概率应比无信号时高"""
        base_features = PredictionFeatures(
            hours_since_last_reset=20.0,
            average_reset_interval=24.0,
            tibo_signal=0.0,
        )
        signal_features = PredictionFeatures(
            hours_since_last_reset=20.0,
            average_reset_interval=24.0,
            tibo_signal=0.85,
        )
        base_result = self.predictor.predict(base_features)
        signal_result = self.predictor.predict(signal_features)

        # 有 Tibo 信号时概率应明显高于无信号
        assert signal_result.probability["5h"] > base_result.probability["5h"], (
            "Tibo limit 信号应推高 5h 概率"
        )
        assert signal_result.probability["24h"] > base_result.probability["24h"]
        # hazard rate 也应更高
        assert signal_result.hazard_rate > base_result.hazard_rate
        # 原因应提到 "reset" 或 "limit"
        assert any("reset" in r.lower() or "limit" in r.lower() or "讨论" in r
                      for r in signal_result.reasons)

    def test_time_pressure_progression(self):
        """时间压力递进：刚 reset < 接近平均 < 超过平均"""
        base = dict(average_reset_interval=24.0, tibo_signal=0.0)
        just_reset = PredictionFeatures(hours_since_last_reset=1.0, **base)
        near_average = PredictionFeatures(hours_since_last_reset=20.0, **base)
        exceed_average = PredictionFeatures(hours_since_last_reset=36.0, **base)

        p_just = self.predictor.predict(just_reset).probability["24h"]
        p_near = self.predictor.predict(near_average).probability["24h"]
        p_exceed = self.predictor.predict(exceed_average).probability["24h"]

        assert p_just < p_near < p_exceed, (
            f"时间压力递进失败: just={p_just:.3f}, near={p_near:.3f}, exceed={p_exceed:.3f}"
        )
        # 刚 reset 后概率应较低
        assert p_just < 0.40
        # 超过平均间隔后概率应较高
        assert p_exceed > 0.70

    def test_scenario_4_no_history(self):
        """场景 4：无历史 reset 记录，仍能输出合理概率"""
        features = PredictionFeatures(
            hours_since_last_reset=None,
            average_reset_interval=None,
            tibo_signal=0.3,
        )
        result = self.predictor.predict(features)

        # 应有概率输出
        assert "5h" in result.probability
        assert "24h" in result.probability
        assert "48h" in result.probability
        # 所有概率在合理范围
        for prob in result.probability.values():
            assert 0.0 < prob < 1.0
        # time_ratio 应使用先验周期计算，不为空
        assert result.time_ratio is not None
        assert result.time_ratio == pytest.approx(1.0, rel=1e-2)
        # 使用了先验
        assert result.prior_applied is True
        # 原因应提到"无历史"
        assert any("无历史" in r or "默认" in r for r in result.reasons)

    def test_scenario_5_high_community_pressure(self):
        """场景 5：高社区压力，概率应比无压力时高"""
        base_features = PredictionFeatures(
            hours_since_last_reset=18.0,
            average_reset_interval=24.0,
            community_signal=0.0,
        )
        pressure_features = PredictionFeatures(
            hours_since_last_reset=18.0,
            average_reset_interval=24.0,
            community_signal=0.9,
        )
        base_result = self.predictor.predict(base_features)
        pressure_result = self.predictor.predict(pressure_features)

        assert pressure_result.probability["24h"] > base_result.probability["24h"], (
            "社区压力应推高概率"
        )
        # 原因应提到 "社区"
        assert any("社区" in r for r in pressure_result.reasons)

    def test_scenario_6_recent_reset_with_signals(self):
        """场景 6：刚 reset + 高 Tibo 信号，时间主导，概率应低于无信号但超时的场景"""
        recent_with_signal = PredictionFeatures(
            hours_since_last_reset=2.0,
            average_reset_interval=24.0,
            tibo_signal=0.8,
            community_signal=0.5,
        )
        overdue_no_signal = PredictionFeatures(
            hours_since_last_reset=40.0,
            average_reset_interval=24.0,
        )
        recent_result = self.predictor.predict(recent_with_signal)
        overdue_result = self.predictor.predict(overdue_no_signal)

        # 超时（无信号）的概率仍应高于刚 reset（有信号）
        assert overdue_result.probability["24h"] > recent_result.probability["24h"], (
            "时间因素应主导：超时无信号 > 刚 reset 有信号"
        )


# ──────────────────────────────────────────────
# 单调性测试
# ──────────────────────────────────────────────

class TestMonotonicity:
    """验证模型单调性：特征增大时概率不应下降"""

    def setup_method(self):
        self.predictor = ResetPredictor()

    def test_time_monotonic(self):
        """时间增加 → 概率不降"""
        probabilities = []
        for hours in [1, 5, 10, 15, 20, 24, 30, 40, 48]:
            features = PredictionFeatures(
                hours_since_last_reset=float(hours),
                average_reset_interval=24.0,
            )
            result = self.predictor.predict(features)
            probabilities.append(result.probability["24h"])
        for i in range(len(probabilities) - 1):
            assert probabilities[i] <= probabilities[i + 1] + 1e-6, (
                f"时间增加但概率下降: {probabilities[i]} -> {probabilities[i+1]}"
            )

    def test_tibo_signal_monotonic(self):
        """tibo_signal 增加 → 概率不降"""
        probabilities = []
        for signal in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            features = PredictionFeatures(
                hours_since_last_reset=15.0,
                average_reset_interval=24.0,
                tibo_signal=signal,
            )
            result = self.predictor.predict(features)
            probabilities.append(result.probability["24h"])
        for i in range(len(probabilities) - 1):
            assert probabilities[i] <= probabilities[i + 1] + 1e-6

    def test_community_signal_monotonic(self):
        """community_signal 增加 → 概率不降"""
        probabilities = []
        for signal in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            features = PredictionFeatures(
                hours_since_last_reset=15.0,
                average_reset_interval=24.0,
                community_signal=signal,
            )
            result = self.predictor.predict(features)
            probabilities.append(result.probability["24h"])
        for i in range(len(probabilities) - 1):
            assert probabilities[i] <= probabilities[i + 1] + 1e-6


# ──────────────────────────────────────────────
# 自定义参数测试
# ──────────────────────────────────────────────

class TestCustomParams:
    """自定义模型参数测试"""

    def test_custom_horizons(self):
        """自定义预测窗口"""
        predictor = ResetPredictor(horizons=[3, 12, 72])
        features = PredictionFeatures(hours_since_last_reset=10.0, average_reset_interval=24.0)
        result = predictor.predict(features)
        assert "3h" in result.probability
        assert "12h" in result.probability
        assert "72h" in result.probability
        assert "5h" not in result.probability

    def test_custom_params_affect_output(self):
        """自定义参数影响输出"""
        default_predictor = ResetPredictor()
        aggressive_predictor = ResetPredictor(params={
            "alpha": -2.0,
            "beta_time": 2.0,
            "beta_tibo": 1.5,
            "beta_community": 1.0,
            "beta_release": 0.8,
        })
        features = PredictionFeatures(
            hours_since_last_reset=20.0,
            average_reset_interval=24.0,
            tibo_signal=0.5,
        )
        default_result = default_predictor.predict(features)
        aggressive_result = aggressive_predictor.predict(features)
        # 激进参数应产生更高概率
        assert aggressive_result.probability["24h"] > default_result.probability["24h"]

    def test_params_property(self):
        """params 属性返回当前参数"""
        predictor = ResetPredictor(params={"alpha": -3.0})
        assert predictor.params["alpha"] == -3.0
        assert predictor.params["beta_time"] == DEFAULT_PARAMS["beta_time"]

    def test_default_interval(self):
        """无历史时使用自定义默认间隔"""
        predictor = ResetPredictor(default_interval=48.0)
        features = PredictionFeatures(
            hours_since_last_reset=24.0,
            average_reset_interval=None,
        )
        result = predictor.predict(features)
        # time_ratio = 24/48 = 0.5
        assert result.time_ratio == pytest.approx(0.5, rel=1e-2)


# ──────────────────────────────────────────────
# 边界情况测试
# ──────────────────────────────────────────────

class TestEdgeCases:
    """边界情况测试"""

    def setup_method(self):
        self.predictor = ResetPredictor()

    def test_all_zero_signals(self):
        """所有信号为 0，仅有时间因素"""
        features = PredictionFeatures(
            hours_since_last_reset=24.0,
            average_reset_interval=24.0,
        )
        result = self.predictor.predict(features)
        # time_ratio = 1.0，概率适中
        assert 0.0 < result.probability["24h"] < 1.0

    def test_all_max_signals(self):
        """所有信号拉满"""
        features = PredictionFeatures(
            hours_since_last_reset=48.0,
            average_reset_interval=24.0,
            tibo_signal=1.0,
            community_signal=1.0,
            release_signal=1.0,
        )
        result = self.predictor.predict(features)
        # 概率应接近 1
        assert result.probability["48h"] > 0.95
        assert result.probability["5h"] > 0.5

    def test_no_history_no_signals(self):
        """无历史且无信号：使用先验周期，time_ratio 不空"""
        features = PredictionFeatures()
        result = self.predictor.predict(features)
        # 基线 hazard，概率处于中等偏低水平但非零
        assert 0.0 < result.probability["5h"] < 0.50
        # time_ratio 使用先验默认周期计算
        assert result.time_ratio is not None
        assert result.time_ratio == pytest.approx(1.0, rel=1e-2)
        assert result.prior_applied is True

    def test_hours_zero(self):
        """hours_since_last_reset = 0（刚刚 reset）"""
        features = PredictionFeatures(
            hours_since_last_reset=0.0,
            average_reset_interval=24.0,
        )
        result = self.predictor.predict(features)
        assert result.probability["5h"] < 0.10

    def test_very_short_interval(self):
        """极短的平均间隔（频繁 reset）"""
        features = PredictionFeatures(
            hours_since_last_reset=3.0,
            average_reset_interval=4.0,
        )
        result = self.predictor.predict(features)
        # time_ratio = 0.75，接近平均间隔
        assert result.time_ratio == pytest.approx(0.75, rel=1e-2)
