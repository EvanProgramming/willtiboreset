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
from auto_confirm import auto_confirm_reset, detect_future_reset_signal
from calibration import (
    append_prediction,
    resolve_history,
    update_performance,
)
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

    # Tibo RSS (via self-hosted RSSHub with Twitter cookies)
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
        try:
            analyzer = DeepSeekAnalyzer(
                api_key=config.deepseek_api_key,
                model=config.deepseek_model,
            )
            # Test the API key with a minimal request
            analyzer._client.models.list()
            return analyzer
        except Exception as e:
            print(f"  ⚠ DeepSeek API key invalid or unreachable: {e}")
            print("  ⚠ Falling back to MockLLMAnalyzer")
            return MockLLMAnalyzer()
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
    evidence_score: float = 0.0,
) -> str:
    """
    Compute prediction confidence.

    Combines 4 factors:
      - Probability clarity (closer to 0 or 1 is more certain)
      - LLM confidence
      - Strength of reset evidence
      - Whether historical data exists

    Low-probability predictions without strong signals must not be labeled
    "high", because that only reflects certainty that a reset is NOT imminent.
    """
    # Clarity: 1 when prob is 0 or 1, 0 when prob is 0.5
    clarity = abs(prob_24h - 0.5) * 2

    # Weighted combination. Evidence score has the largest weight so that
    # low-probability, no-signal scenarios are not labeled high confidence.
    score = clarity * 0.2 + llm_confidence * 0.25 + evidence_score * 0.45
    if has_history:
        score += 0.1

    # Strong reset evidence combined with elevated probability further raises
    # confidence that the prediction direction is correct.
    if evidence_score >= 0.5 and prob_24h >= 0.5:
        score += 0.15

    # Penalize very low probability without meaningful evidence; clarity alone
    # is not enough to claim medium confidence when nothing signals a reset.
    if evidence_score < 0.15 and prob_24h <= 0.25:
        score -= 0.15

    score = max(0.0, min(score, 1.0))

    if score > 0.7:
        return "high"
    elif score > 0.35:
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

    # Auto-confirm a reset from Tibo's explicit announcement. The confirmed
    # reset is recorded in history and events are reloaded BEFORE the model
    # runs, so the model sees "just reset" and naturally lowers short-term
    # probabilities instead of treating the announcement as a future signal.
    print("\n[1/4] Checking for auto-confirmation...")
    confirmed_event = auto_confirm_reset(tweets)
    if confirmed_event:
        print(
            f"  AUTO-CONFIRMED reset at {confirmed_event.reset_time.isoformat()}"
        )
        print(f"  Reason: {confirmed_event.notes[:100]}")
        # Reload events so the new reset is used in feature computation
        reset_collector = ResetHistoryCollector(config.reset_history_path)
        events = reset_collector.collect()
        print(f"  Historical reset events: {len(events)} events")
    else:
        print("  No explicit reset announcement detected")

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

    # Most recent confirmed reset time, used to dampen past-tense reset
    # confirmations ("I have reset...") so they don't inflate future probability.
    recent_reset_time = (
        max((e.reset_time for e in events), default=None) if events else None
    )

    # 每周重置间隔：有历史记录时，预计间隔为 7 天（168h）。
    # 注意：不能直接用 hours_since + hours_until，因为当窗口已过被顺延后,
    # hours_until 会变成下一周期的值，导致 sum = 336h（14天）而非 168h。
    expected_weekly_interval = (
        7 * 24
        if analysis_features.hours_since_last_reset is not None
        else None
    )

    # Check for explicit FUTURE reset announcement from Tibo
    # (e.g., "I will reset usage limits tonight")
    future_reset = detect_future_reset_signal(tweets)
    explicit_future_reset = future_reset is not None
    # Also check manual override via environment variable
    if not explicit_future_reset and config.explicit_future_reset:
        explicit_future_reset = True
        print("  EXPLICIT_FUTURE_RESET override enabled via environment variable")
    if future_reset:
        print(f"  FUTURE RESET ANNOUNCEMENT DETECTED: {future_reset[1][:100]}")

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
        recent_reset_time=recent_reset_time,
        expected_weekly_interval_hours=expected_weekly_interval,
        weekly_cycle_factor=analysis_features.weekly_cycle_factor,
        explicit_future_reset=explicit_future_reset,
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
    confidence = compute_confidence(
        prob_24h, has_history, llm_conf, explanation.evidence_score
    )
    prior_applied = analysis_features.avg_reset_interval_hours is None

    prediction_dict = {
        "within_5h": explanation.probability.get("5h", 0.0),
        "within_24h": explanation.probability.get("24h", 0.0),
        "within_48h": explanation.probability.get("48h", 0.0),
    }

    # 每周重置规则：预计下次重置日期（上次实际重置 + 7 天）
    next_reset_info = None
    if analysis_features.next_reset_time is not None:
        next_reset_info = {
            "expected_time": analysis_features.next_reset_time.isoformat(),
            "hours_until": round(analysis_features.hours_until_next_reset, 1),
            "status": analysis_features.reset_schedule_status,
        }

    signals_snapshot = {
        "tweet_count": len(tweets),
        "hours_since_last_reset": pred_features.hours_since_last_reset,
        "next_reset": next_reset_info,
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
        "weekly_cycle_factor": analysis_features.weekly_cycle_factor,
        "explicit_future_reset": explicit_future_reset,
    }

    output = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "prediction": prediction_dict,
        "confidence": confidence,
        "next_reset": next_reset_info,
        "signals": signals_snapshot,
        "main_factors": [
            f.model_dump(mode="json") for f in explanation.main_factors
        ],
        "reasons": explanation.reasons,
    }

    # 若已超过预期重置窗口（上周重置 + 7 天）仍未确认新重置，
    # 在 reasons 中补充说明，便于用户理解概率为何由时间因素主导。
    if next_reset_info and next_reset_info["status"] == "overdue":
        output["reasons"].insert(
            0,
            "Expected weekly reset window (last reset + 7 days) has already "
            "passed; no new reset confirmed yet, probability driven mainly by "
            "time pressure",
        )

    # If a reset was auto-confirmed, annotate the output WITHOUT overwriting
    # the model probabilities. Because events were reloaded before modeling,
    # the probabilities already reflect "just reset" (lower short-term odds).
    if confirmed_event:
        output["auto_confirmed"] = True
        output["confirmed_reset_time"] = confirmed_event.reset_time.isoformat()
        output["reasons"].insert(
            0,
            "Auto-confirmed reset announced by Tibo at "
            f"{confirmed_event.reset_time.isoformat()}; reset already occurred, "
            "so short-term probabilities are lowered",
        )

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
