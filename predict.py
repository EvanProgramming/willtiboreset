#!/usr/bin/env python3
"""
WillTiboReset - One-click prediction script

Run: python predict.py
Output: output/prediction.json

Automatically completes:
  1. Fetch latest data (Tibo / OpenAI / Community RSS)
  2. Analyze text signals (Gemini LLM)
  3. Run prediction model (Adaptive Bayesian Survival Model)
  4. Generate final prediction file output/prediction.json

Output format:
{
  "updated_at": "2025-07-31T12:00:00+00:00",
  "prediction": {
    "within_5h": 0.42,
    "within_24h": 0.76,
    "within_48h": 0.91
  },
  "confidence": "medium",
  "signals": { ... },
  "reasons": ["...", "..."]
}
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyzer import SignalAnalyzer
from analyzer.llm_signal import DeepSeekAnalyzer, LLMAnalyzer, MockLLMAnalyzer
from calibration import append_prediction, resolve_history, update_performance
from collectors import (
    CommunityCollector,
    OpenAIRSSCollector,
    ResetHistoryCollector,
    TiboRSSCollector,
    TweetCollector,
)
from collectors.rss_base import _build_dedup_key
from config import config
from model.data_models import Tweet
from model.model_state import ModelStateManager
from model.survival_model import ResetPredictor, build_features


# ──────────────────────────────────────────────
# Data collection
# ──────────────────────────────────────────────

def collect_data() -> list[Tweet]:
    """
    Fetch latest data: run all collectors and deduplicate.

    Sources:
      - TiboRSSCollector: Tibo-related RSS feeds
      - OpenAIRSSCollector: OpenAI official RSS feeds
      - CommunityCollector: community RSS + mock data
    """
    all_tweets: list[Tweet] = []

    tibo = TiboRSSCollector(timeout=config.rss_request_timeout)
    all_tweets.extend(tibo.collect())

    openai = OpenAIRSSCollector(timeout=config.rss_request_timeout)
    all_tweets.extend(openai.collect())

    community = CommunityCollector(timeout=config.rss_request_timeout)
    all_tweets.extend(community.collect())

    # Global deduplication
    seen: set[str] = set()
    result: list[Tweet] = []
    for t in all_tweets:
        key = _build_dedup_key(t)
        if key not in seen:
            seen.add(key)
            result.append(t)

    return result


def create_analyzer() -> LLMAnalyzer:
    """Create an LLM analyzer based on configuration. DeepSeek is preferred when configured; otherwise fall back to Mock."""
    if config.has_deepseek_credentials:
        return DeepSeekAnalyzer(
            api_key=config.deepseek_api_key,
            model=config.deepseek_model,
        )
    print("  ⚠ DEEPSEEK_API_KEY not configured; using MockLLMAnalyzer for local validation")
    return MockLLMAnalyzer()


def validate_configuration() -> None:
    """Warn about missing configuration before running, allowing local validation to fall back to mock/existing data."""
    if not config.has_deepseek_credentials:
        print("  ⚠ DEEPSEEK_API_KEY not configured; will use MockLLMAnalyzer")
    if not config.rss_feeds.get("tibo"):
        print("  ⚠ TIBO_RSS_URLS not configured; will rely on existing data or community mock data")


# ──────────────────────────────────────────────
# Confidence computation
# ──────────────────────────────────────────────

def compute_confidence(
    prob_24h: float,
    has_history: bool,
    llm_confidence: float,
) -> str:
    """
    Compute prediction confidence.

    Combines 3 factors:
      - Probability clarity (farther from 0.5 is more certain)
      - LLM confidence
      - Whether historical data exists
    """
    clarity = abs(prob_24h - 0.5) * 2  # 0-1
    score = clarity * 0.6 + llm_confidence * 0.3
    if has_history:
        score += 0.1
    score = min(score, 1.0)

    if score > 0.5:
        return "high"
    elif score > 0.25:
        return "medium"
    return "low"


# ──────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────

def main() -> int:
    config.ensure_dirs()
    validate_configuration()

    print("WillTiboReset — Prediction Engine")
    print("=" * 50)

    # ── 1. Fetch latest data ──
    print("\n[1/4] Fetching latest data...")
    tweets = collect_data()

    # Save to tweets.json
    tweet_collector = TweetCollector(config.tweets_path)
    tweet_collector.save(tweets)
    print(f"  Signals collected: {len(tweets)} tweets")

    # Load historical reset events
    reset_collector = ResetHistoryCollector(config.reset_history_path)
    events = reset_collector.collect()
    print(f"  Historical reset events: {len(events)} events")

    # ── 2. Analyze text signals ──
    print("\n[2/4] Analyzing text signals...")
    analyzer = create_analyzer()
    print(f"  Analyzer: {analyzer.__class__.__name__}")

    signal_scores = []
    batch_scores = None

    if tweets:
        signal_scores = analyzer.analyze_tweets(tweets)
        batch_scores = analyzer.analyze_batch([t.text for t in tweets])
        print(f"  reset_intent:        {batch_scores.reset_intent:.2f}")
        print(f"  reset_confirmation:  {batch_scores.reset_confirmation:.2f}")
        print(f"  limit_complaint:     {batch_scores.limit_complaint:.2f}")
        print(f"  official_change:     {batch_scores.official_change:.2f}")
        print(f"  llm_confidence:      {batch_scores.confidence:.2f}")

    # Statistical feature extraction
    signal_analyzer = SignalAnalyzer()
    analysis_features = signal_analyzer.analyze(tweets, events)

    if analysis_features.hours_since_last_reset is not None:
        print(f"  Hours since last reset: {analysis_features.hours_since_last_reset:.1f}h")
    else:
        print("  Hours since last reset: no historical record")

    # ── 3. Load model state and run prediction model ──
    print("\n[3/4] Loading model state...")
    state_manager = ModelStateManager(config.model_state_path)
    model_state = state_manager.load()
    if model_state is not None:
        print(f"  Loaded model_state: {model_state.sample_count} intervals")
        print(f"  Posterior average interval: {model_state.average_interval_hours:.1f}h")
    else:
        print("  model_state.json not found; will use default prior parameters")

    print("\n[3/4] Running prediction model...")
    pred_features = build_features(
        hours_since_last_reset=analysis_features.hours_since_last_reset,
        average_reset_interval=analysis_features.avg_reset_interval_hours,
        median_reset_interval=analysis_features.median_reset_interval_hours,
        interval_uncertainty=analysis_features.std_reset_interval_hours,
        signal_scores=signal_scores if signal_scores else None,
        tweets=tweets if tweets else None,
        interval_count=analysis_features.reset_interval_count,
        model_state=model_state,
    )

    predictor = ResetPredictor(
        horizons=config.prediction_horizons,
        default_interval=config.default_reset_interval_hours,
        model_state=model_state,
    )
    explanation = predictor.predict(pred_features)

    print(f"  Model: {predictor.model_version}")
    print(f"  Hazard rate:   {explanation.hazard_rate:.4f}/h")
    print(f"  Time pressure: {explanation.time_pressure:.2f}")
    if explanation.time_ratio is not None:
        print(f"  Time ratio:    {explanation.time_ratio:.2f}x")

    for horizon, prob in explanation.probability.items():
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {horizon:>4s}: {prob:.1%}  {bar}")

    print("\n  Main factors:")
    for factor in explanation.main_factors:
        print(f"    • {factor.factor}: {factor.impact}")

    # ── 4. Generate prediction file ──
    print("\n[4/4] Generating prediction file...")

    prob_24h = explanation.probability.get("24h", 0.0)
    has_history = analysis_features.hours_since_last_reset is not None
    llm_conf = batch_scores.confidence if batch_scores else 0.0
    confidence = compute_confidence(prob_24h, has_history, llm_conf)
    prior_applied = analysis_features.avg_reset_interval_hours is None

    prediction_dict = {
        "within_5h": explanation.probability.get("5h", 0.0),
        "within_24h": explanation.probability.get("24h", 0.0),
        "within_48h": explanation.probability.get("48h", 0.0),
    }

    signals_snapshot = {
        "tweet_count": len(tweets),
        "hours_since_last_reset": pred_features.hours_since_last_reset,
        "average_reset_interval": pred_features.average_reset_interval,
        "median_reset_interval": pred_features.median_reset_interval,
        "interval_uncertainty": pred_features.interval_uncertainty,
        "std_reset_interval": analysis_features.std_reset_interval_hours,
        "min_reset_interval": analysis_features.min_reset_interval_hours,
        "max_reset_interval": analysis_features.max_reset_interval_hours,
        "interval_confidence": analysis_features.interval_confidence,
        "time_ratio": explanation.time_ratio,
        "time_pressure": explanation.time_pressure,
        "hazard_rate": explanation.hazard_rate,
        "evidence_score": explanation.evidence_score,
        "tibo_signal": round(pred_features.tibo_signal, 4),
        "community_signal": round(pred_features.community_signal, 4),
        "release_signal": round(pred_features.release_signal, 4),
        "llm_scores": (
            batch_scores.model_dump(mode="json") if batch_scores else None
        ),
        "prior_applied": prior_applied,
        "interval_count": analysis_features.reset_interval_count,
    }

    output = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "prediction": prediction_dict,
        "confidence": confidence,
        "signals": signals_snapshot,
        "main_factors": [
            f.model_dump(mode="json") for f in explanation.main_factors
        ],
        "reasons": explanation.reasons,
    }

    output_path = config.output_dir / "prediction.json"
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  Saved: {output_path}")

    # ── 5. Update prediction history and performance report ──
    print("\n[5/4] Updating prediction history and calibration report...")
    append_prediction(
        config.prediction_history_path,
        prediction_dict,
        signals_snapshot,
        actual_result=None,
    )
    print(f"  Saved: {config.prediction_history_path}")

    resolved, newly_resolved = resolve_history(
        config.prediction_history_path,
        events,
    )
    print(
        f"  Resolved predictions: {resolved} total, "
        f"{newly_resolved} newly closed"
    )

    update_performance(
        config.prediction_history_path,
        config.model_performance_path,
    )
    print(f"  Saved: {config.model_performance_path}")

    print("\n" + "=" * 50)
    print("Prediction complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
