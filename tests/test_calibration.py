"""Tests for calibration module."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from calibration import (
    append_prediction,
    evaluate_performance,
    load_history,
    resolve_history,
    update_performance,
)
from model.data_models import PredictionHistoryEntry, ResetEvent, SignalSource


def _dt(hours_ago: float) -> datetime:
    return datetime.now(timezone.utc) - timedelta(hours=hours_ago)


def test_resolve_history_marks_true_when_reset_occurs(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    pred_time = _dt(50)
    append_prediction(
        history_path,
        {"within_5h": 0.1, "within_24h": 0.3, "within_48h": 0.6},
        {"tweet_count": 1},
        actual_result=None,
    )
    # Override prediction_time for deterministic testing
    history = load_history(history_path)
    history[0].prediction_time = pred_time
    history_path.write_text(
        "[" + ",".join(h.model_dump_json() for h in history) + "]",
        encoding="utf-8",
    )

    reset_events = [
        ResetEvent(
            reset_time=pred_time + timedelta(hours=10),
            source=SignalSource.TWITTER,
            confidence=1.0,
            notes="Confirmed reset",
        ),
    ]

    resolved, newly = resolve_history(history_path, reset_events, now=_dt(0))
    assert resolved == 1
    assert newly == 1
    history = load_history(history_path)
    assert history[0].actual_result is True
    assert history[0].resolved_at is not None


def test_resolve_history_marks_false_when_window_elapses(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    pred_time = _dt(60)
    append_prediction(
        history_path,
        {"within_5h": 0.1, "within_24h": 0.3, "within_48h": 0.6},
        {"tweet_count": 1},
        actual_result=None,
    )
    history = load_history(history_path)
    history[0].prediction_time = pred_time
    history_path.write_text(
        "[" + ",".join(h.model_dump_json() for h in history) + "]",
        encoding="utf-8",
    )

    resolved, newly = resolve_history(history_path, [], now=_dt(0))
    assert resolved == 1
    assert newly == 1
    history = load_history(history_path)
    assert history[0].actual_result is False


def test_resolve_history_skips_unexpired_predictions(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    pred_time = _dt(10)
    append_prediction(
        history_path,
        {"within_48h": 0.6},
        {"tweet_count": 1},
        actual_result=None,
    )
    history = load_history(history_path)
    history[0].prediction_time = pred_time
    history_path.write_text(
        "[" + ",".join(h.model_dump_json() for h in history) + "]",
        encoding="utf-8",
    )

    resolved, newly = resolve_history(history_path, [], now=_dt(0))
    assert newly == 0
    history = load_history(history_path)
    assert history[0].actual_result is None


def test_evaluate_performance_with_resolved_entries(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    perf_path = tmp_path / "performance.json"

    entry = PredictionHistoryEntry(
        prediction_time=_dt(50),
        prediction={"within_5h": 0.1, "within_24h": 0.8, "within_48h": 0.9},
        signals={"tweet_count": 1},
        actual_result=True,
    )
    history_path.write_text(
        "[" + entry.model_dump_json() + "]",
        encoding="utf-8",
    )

    performance = update_performance(history_path, perf_path)
    assert performance.total_predictions == 1
    assert performance.resolved_predictions == 1
    assert performance.overall_brier_score is not None
    assert performance.overall_accuracy == pytest.approx(2 / 3)


def test_evaluate_performance_empty_history(tmp_path: Path) -> None:
    history_path = tmp_path / "history.json"
    perf_path = tmp_path / "performance.json"

    performance = update_performance(history_path, perf_path)
    assert performance.total_predictions == 0
    assert performance.resolved_predictions == 0
    assert performance.overall_brier_score is None
    assert performance.horizons[0].brier_score is None
