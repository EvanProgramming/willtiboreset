"""
WillTiboReset - Model state update script

Reads data/reset_history.json, computes interval statistics and adaptive model params,
and saves to data/model_state.json.

Recommended to run daily/weekly via GitHub Actions so the model improves as data accumulates.
"""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import config
from model.model_state import ModelState, ModelStateManager
from model.survival_model import DEFAULT_RESET_INTERVAL_HOURS


def _to_aware(dt: datetime) -> datetime:
    """Convert naive datetime to UTC aware datetime."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt


def _median(values: list[float]) -> float:
    """Compute median."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    if n % 2 == 1:
        return sorted_vals[mid]
    return (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0


def _std(values: list[float]) -> float:
    """Compute sample standard deviation."""
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
    """Compute interval confidence based on sample size and coefficient of variation."""
    if count < 1 or mean <= 0:
        return 0.0
    sample_factor = min(1.0, count / 10.0)
    cv = std / mean if std else 0.0
    stability_factor = max(0.0, 1.0 - cv)
    return round(sample_factor * 0.6 + stability_factor * 0.4, 4)


def _compute_prior_weight(sample_count: int) -> float:
    """
    Compute prior weight based on sample count.

    More samples lead to lower prior weight. Prior weight reaches 0 with 20 or more intervals.
    """
    return max(0.0, 1.0 - sample_count / 20.0)


def _compute_adaptive_params(
    interval_confidence: float,
) -> dict[str, float]:
    """
    Fine-tune model parameters based on interval confidence (V2 reserves the params field for future use).

    When sufficient historical data exists, slightly strengthen the time pressure signal;
    when data is insufficient, keep conservative parameters.
    """
    params: dict[str, float] = {}
    # More credible intervals slightly increase the credibility of the time factor
    params["time_adjustment_strength"] = round(0.30 + 0.1 * interval_confidence, 4)
    return params


def update_model_state(reset_history_path: Optional[Path] = None) -> ModelState:
    """
    Compute and save model state from reset_history.json.

    Even with insufficient historical data, a prior-based model_state is still generated.
    """
    config.ensure_dirs()

    if reset_history_path is None:
        reset_history_path = config.reset_history_path
    if not reset_history_path.exists():
        # No historical data: generate a prior-based model_state
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
    # uncertainty: use std / sqrt(sample_count) to represent estimation uncertainty of the mean
    interval_uncertainty = std / (sample_count ** 0.5) if sample_count > 0 else std

    # Posterior average interval: weighted blend of prior and observed intervals
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
    """Command-line entry point."""
    print("WillTiboReset - Update Model State")
    print("=" * 40)

    state = update_model_state()
    manager = ModelStateManager(config.model_state_path)
    manager.save(state)

    print(f"Save path: {config.model_state_path}")
    print(f"Sample interval count: {state.sample_count}")
    print(f"Posterior average interval: {state.average_interval_hours:.2f}h")
    print(f"Median interval: {state.median_interval_hours}")
    print(f"Standard deviation: {state.std_interval_hours}")
    print(f"Interval uncertainty: {state.interval_uncertainty}")
    print(f"Interval confidence: {state.interval_confidence:.0%}")
    print(f"Prior weight: {state.prior_weight:.2%}")
    print(f"Params: {state.params}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
