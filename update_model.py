"""
WillTiboReset - 模型状态更新脚本

读取 data/reset_history.json，计算 interval statistics 和 adaptive model params，
保存到 data/model_state.json。

建议通过 GitHub Actions 每日/每周运行，实现模型随数据自我改进。
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from model.model_state import ModelState, ModelStateManager
from model.survival_model import DEFAULT_RESET_INTERVAL_HOURS


def _to_aware(dt: datetime) -> datetime:
    """将 naive datetime 转换为 UTC aware datetime。"""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _median(values: list[float]) -> float:
    """计算中位数。"""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


def _std(values: list[float]) -> float:
    """计算样本标准差。"""
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return variance ** 0.5


def _interval_confidence(
    count: int,
    std: float,
    mean: float,
) -> float:
    """基于样本量和变异系数计算 interval 置信度。"""
    if count < 1 or mean <= 0:
        return 0.0
    sample_factor = min(1.0, count / 10.0)
    cv = std / mean if std else 0.0
    stability_factor = max(0.0, 1.0 - cv)
    return round(sample_factor * 0.6 + stability_factor * 0.4, 4)


def _compute_prior_weight(sample_count: int) -> float:
    """
    根据样本量计算先验权重。

    样本越多，先验权重越低。20 个 interval 以上时先验权重降为 0。
    """
    return max(0.0, 1.0 - sample_count / 20.0)


def _compute_adaptive_params(
    interval_confidence: float,
) -> dict[str, float]:
    """
    根据 interval 置信度微调模型参数（V2 保留参数字段供未来使用）。

    当历史数据足够时，略微增强时间压力信号；
    当历史数据不足时，保持保守参数。
    """
    params: dict[str, float] = {}
    # interval 越可信，时间因素的可信度略微提高
    params["time_adjustment_strength"] = round(0.30 + 0.1 * interval_confidence, 4)
    return params


def update_model_state(reset_history_path: Optional[Path] = None) -> ModelState:
    """
    从 reset_history.json 计算并保存模型状态。

    若历史数据不足，仍生成基于先验的 model_state。
    """
    config.ensure_dirs()

    if reset_history_path is None:
        reset_history_path = config.reset_history_path
    if not reset_history_path.exists():
        # 无历史数据：生成基于先验的 model_state
        return ModelState(
            average_interval_hours=DEFAULT_RESET_INTERVAL_HOURS,
            sample_count=0,
            prior_weight=1.0,
            params={},
        )

    import json
    raw = json.loads(reset_history_path.read_text(encoding="utf-8"))

    events: list[dict] = []
    for item in raw:
        try:
            reset_time = datetime.fromisoformat(item["reset_time"])
            events.append({"reset_time": reset_time, "confidence": item.get("confidence", 1.0)})
        except Exception:
            continue

    events.sort(key=lambda e: _to_aware(e["reset_time"]))

    if len(events) < 2:
        return ModelState(
            average_interval_hours=DEFAULT_RESET_INTERVAL_HOURS,
            sample_count=0,
            prior_weight=1.0,
            params={},
        )

    intervals: list[float] = []
    for i in range(1, len(events)):
        delta = (
            _to_aware(events[i]["reset_time"])
            - _to_aware(events[i - 1]["reset_time"])
        )
        intervals.append(delta.total_seconds() / 3600.0)

    sample_count = len(intervals)
    avg = sum(intervals) / sample_count
    median = _median(intervals)
    std = _std(intervals)
    min_interval = min(intervals)
    max_interval = max(intervals)
    interval_conf = _interval_confidence(sample_count, std, avg)
    prior_weight = _compute_prior_weight(sample_count)
    # uncertainty: 用 std 除以 sqrt(sample_count) 表示均值的估计不确定性
    interval_uncertainty = std / (sample_count ** 0.5) if sample_count > 0 else std

    # 后验平均间隔：先验与观测的加权融合
    posterior_avg = (
        prior_weight * DEFAULT_RESET_INTERVAL_HOURS
        + (1.0 - prior_weight) * avg
    )

    params = _compute_adaptive_params(interval_conf)

    state = ModelState(
        average_interval_hours=round(posterior_avg, 2),
        median_interval_hours=round(median, 2),
        std_interval_hours=round(std, 2),
        min_interval_hours=round(min_interval, 2),
        max_interval_hours=round(max_interval, 2),
        interval_uncertainty=round(interval_uncertainty, 2),
        sample_count=sample_count,
        interval_confidence=interval_conf,
        prior_weight=round(prior_weight, 4),
        params=params,
    )

    return state


def main() -> int:
    """命令行入口。"""
    print("WillTiboReset - 更新模型状态")
    print("=" * 40)

    state = update_model_state()
    manager = ModelStateManager(config.model_state_path)
    manager.save(state)

    print(f"保存路径: {config.model_state_path}")
    print(f"样本 interval 数: {state.sample_count}")
    print(f"后验平均间隔: {state.average_interval_hours:.2f}h")
    print(f"中位间隔: {state.median_interval_hours}")
    print(f"标准差: {state.std_interval_hours}")
    print(f"间隔不确定性: {state.interval_uncertainty}")
    print(f"间隔置信度: {state.interval_confidence:.0%}")
    print(f"先验权重: {state.prior_weight:.2%}")
    print(f"参数: {state.params}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
