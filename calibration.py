"""
WillTiboReset - 模型校准与性能评估。

根据 prediction_history.json 中已确认 actual_result 的记录：
    - 计算 Brier Score
    - 计算二分类准确率
    - 计算校准误差（calibration error）
    - 输出 model_performance.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from model.data_models import (
    CalibrationBin,
    HorizonPerformance,
    ModelPerformance,
    PredictionHistoryEntry,
)


# 校准分箱边界
_BIN_BOUNDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def load_history(history_path: Path) -> list[PredictionHistoryEntry]:
    """加载预测历史。"""
    if not history_path.exists():
        return []

    try:
        raw = json.loads(history_path.read_text(encoding="utf-8"))
        if isinstance(raw, list):
            return [PredictionHistoryEntry.model_validate(item) for item in raw]
    except Exception:
        return []
    return []


def save_history(
    history_path: Path,
    history: list[PredictionHistoryEntry],
) -> None:
    """保存预测历史。"""
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(
            [entry.model_dump(mode="json") for entry in history],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def append_prediction(
    history_path: Path,
    prediction: dict[str, float],
    signals: dict,
    actual_result: Optional[bool] = None,
) -> PredictionHistoryEntry:
    """向历史记录追加一条新预测。"""
    history = load_history(history_path)
    entry = PredictionHistoryEntry(
        prediction_time=datetime.now(timezone.utc),
        prediction=prediction,
        signals=signals,
        actual_result=actual_result,
    )
    history.append(entry)
    save_history(history_path, history)
    return entry


def _resolved_entries(
    history: list[PredictionHistoryEntry],
) -> list[PredictionHistoryEntry]:
    """过滤出已确认结果的记录。"""
    return [h for h in history if h.actual_result is not None]


def _brier_score(probs: list[float], actuals: list[bool]) -> Optional[float]:
    """计算 Brier score。"""
    if not probs:
        return None
    actual_floats = [1.0 if a else 0.0 for a in actuals]
    return sum((p - a) ** 2 for p, a in zip(probs, actuals)) / len(probs)


def _accuracy(probs: list[float], actuals: list[bool]) -> Optional[float]:
    """以 0.5 为阈值计算准确率。"""
    if not probs:
        return None
    correct = sum(
        1 for p, a in zip(probs, actuals)
        if (p >= 0.5) == a
    )
    return correct / len(probs)


def _calibration_bins(
    probs: list[float],
    actuals: list[bool],
) -> tuple[list[CalibrationBin], Optional[float]]:
    """计算校准分箱及平均校准误差。"""
    bins: list[CalibrationBin] = []
    total_error = 0.0
    total_count = 0

    for i in range(len(_BIN_BOUNDS) - 1):
        start = _BIN_BOUNDS[i]
        end = _BIN_BOUNDS[i + 1]

        # 左闭右开，最后一个区间包含 1.0
        if i == len(_BIN_BOUNDS) - 2:
            indices = [
                j for j, p in enumerate(probs)
                if start <= p <= end
            ]
        else:
            indices = [
                j for j, p in enumerate(probs)
                if start <= p < end
            ]

        count = len(indices)
        if count == 0:
            bins.append(
                CalibrationBin(
                    bin_start=start,
                    bin_end=end,
                    predicted_mean=(start + end) / 2.0,
                    actual_frequency=None,
                    count=0,
                )
            )
            continue

        bin_probs = [probs[j] for j in indices]
        bin_actuals = [actuals[j] for j in indices]
        predicted_mean = sum(bin_probs) / count
        actual_frequency = sum(1 for a in bin_actuals if a) / count
        total_error += abs(predicted_mean - actual_frequency) * count
        total_count += count

        bins.append(
            CalibrationBin(
                bin_start=start,
                bin_end=end,
                predicted_mean=predicted_mean,
                actual_frequency=actual_frequency,
                count=count,
            )
        )

    calibration_error = total_error / total_count if total_count > 0 else None
    return bins, calibration_error


def evaluate_horizon(
    history: list[PredictionHistoryEntry],
    key: str,
    horizon_hours: int,
) -> HorizonPerformance:
    """评估单个时间窗口的性能。"""
    resolved = _resolved_entries(history)
    probs = [h.prediction.get(key, 0.0) for h in resolved]
    actuals = [bool(h.actual_result) for h in resolved]

    bins, calibration_error = _calibration_bins(probs, actuals)

    return HorizonPerformance(
        horizon_hours=horizon_hours,
        total=len(probs),
        brier_score=_brier_score(probs, actuals),
        accuracy=_accuracy(probs, actuals),
        calibration_error=calibration_error,
        bins=bins,
    )


def evaluate_performance(
    history: list[PredictionHistoryEntry],
) -> ModelPerformance:
    """评估整体模型性能。"""
    resolved = _resolved_entries(history)

    # 跨所有窗口计算平均 Brier score 和准确率
    all_probs: list[float] = []
    all_actuals: list[bool] = []
    for h in resolved:
        for key, val in h.prediction.items():
            all_probs.append(val)
            all_actuals.append(bool(h.actual_result))

    overall_brier = _brier_score(all_probs, all_actuals)
    overall_accuracy = _accuracy(all_probs, all_actuals)

    horizons = [
        evaluate_horizon(history, "within_5h", 5),
        evaluate_horizon(history, "within_24h", 24),
        evaluate_horizon(history, "within_48h", 48),
    ]

    return ModelPerformance(
        total_predictions=len(history),
        resolved_predictions=len(resolved),
        overall_brier_score=overall_brier,
        overall_accuracy=overall_accuracy,
        horizons=horizons,
        updated_at=datetime.now(timezone.utc),
    )


def update_performance(
    history_path: Path,
    performance_path: Path,
) -> ModelPerformance:
    """
    根据历史记录更新并保存模型性能报告。

    如果还没有已确认结果的记录，则生成空报告。
    """
    history = load_history(history_path)
    performance = evaluate_performance(history)

    performance_path.parent.mkdir(parents=True, exist_ok=True)
    performance_path.write_text(
        performance.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return performance


def mark_actual_result(
    history_path: Path,
    prediction_time: datetime,
    actual_result: bool,
) -> bool:
    """
    为某条历史预测标记实际结果。

    返回是否找到并更新该记录。
    """
    history = load_history(history_path)
    updated = False
    for entry in history:
        if entry.prediction_time == prediction_time:
            entry.actual_result = actual_result
            entry.resolved_at = datetime.now(timezone.utc)
            updated = True
            break

    if updated:
        save_history(history_path, history)
    return updated


__all__ = [
    "load_history",
    "save_history",
    "append_prediction",
    "update_performance",
    "evaluate_performance",
    "mark_actual_result",
]
