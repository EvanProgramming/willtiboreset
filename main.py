#!/usr/bin/env python3
"""
WillTiboReset - Main entry point

Predict whether Tibo/OpenAI will reset ChatGPT/Codex usage quota
within the next 5h / 24h / 48h.

Phase 3: full prediction pipeline (collect → analyze → LLM signals → survival model prediction)

Usage:
    python main.py              # Run the full prediction pipeline
    python main.py --status     # Show project status only
    python main.py --analyze    # Run signal analysis only (no prediction)
    python main.py --predict    # Run prediction only (using already collected data)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyzer import SignalAnalyzer
from analyzer.llm_signal import DeepSeekAnalyzer, MockLLMAnalyzer
from collectors import ResetHistoryCollector, TweetCollector
from config import config
from model.data_models import PredictionResult
from model.model_state import ModelStateManager
from model.survival_model import ResetPredictor, build_features
from output import OutputFormatter


def _validate_prediction_config() -> None:
    """Validate required configuration before running prediction."""
    if not config.has_deepseek_credentials:
        raise RuntimeError(
            "DEEPSEEK_API_KEY is not configured. Set it in .env or GitHub Actions Secrets."
        )
    if not config.rss_feeds.get("tibo"):
        raise RuntimeError(
            "TIBO_RSS_URLS is not configured. Tibo is the core data source; at least one RSS URL is required."
        )


def print_separator(title: str = "") -> None:
    """Print a separator line"""
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"{'─' * pad} {title} {'─' * (width - len(title) - 2 - pad)}")
    else:
        print("─" * width)


def show_status() -> None:
    """Display project status"""
    print()
    print("=" * 60)
    print("  WillTiboReset - Project Status")
    print("=" * 60)
    print()

    # Configuration
    print_separator("Configuration")
    print(f"  Prediction horizons: {config.prediction_horizons} hours")
    print(f"  Confidence threshold: {config.confidence_threshold}")
    print(f"  Data directory: {config.data_dir}")
    print(f"  Output directory: {config.output_dir}")
    print()

    # API credentials
    print_separator("API Credentials")
    print(f"  Twitter:  {'✓ configured' if config.has_twitter_credentials else '✗ not configured'}")
    print(f"  OpenAI:   {'✓ configured' if config.has_openai_credentials else '✗ not configured'}")
    print()

    # Data files
    print_separator("Data Files")
    tweet_collector = TweetCollector(config.tweets_path)
    reset_collector = ResetHistoryCollector(config.reset_history_path)
    tweets = tweet_collector.collect()
    events = reset_collector.collect()
    print(f"  Tweets:       {len(tweets)} tweets")
    print(f"  Reset events: {len(events)} events")
    print()

    print("=" * 60)
    print()


def run_analysis() -> None:
    """Run signal analysis and print results"""
    print()
    print("=" * 60)
    print("  WillTiboReset - Signal Analysis")
    print("=" * 60)
    print()

    # 1. Load data
    print_separator("Data Collection")
    tweet_collector = TweetCollector(config.tweets_path)
    reset_collector = ResetHistoryCollector(config.reset_history_path)
    tweets = tweet_collector.collect()
    events = reset_collector.collect()
    print(f"  Loaded tweets:      {len(tweets)} tweets")
    print(f"  Loaded reset events: {len(events)} events")
    print()

    # 2. Analyze signals
    print_separator("Feature Extraction")
    analyzer = SignalAnalyzer()
    features = analyzer.analyze(tweets, events)
    print(f"  Total tweets:          {features.tweet_count}")
    print(f"  Tweets in last 24h:    {features.recent_tweet_count}")
    print(f"  Unique authors:        {features.unique_authors}")
    print(f"  Historical reset events: {features.total_reset_events}")
    if features.hours_since_last_reset is not None:
        print(f"  Hours since last reset: {features.hours_since_last_reset:.1f} hours")
    if features.avg_reset_interval_hours is not None:
        print(f"  Average reset interval: {features.avg_reset_interval_hours:.1f} hours")
    print()

    # 3. Signal descriptions
    print_separator("Signal Summary")
    for desc in features.to_signal_descriptions():
        print(f"  • {desc}")
    print()

    print("=" * 60)
    print()


def run_prediction(tweets=None, events=None) -> None:
    """
    Run the prediction step: LLM signal analysis → feature building → survival model prediction.

    If tweets/events are not provided, load them from data files.
    """
    print()
    print("=" * 60)
    print("  WillTiboReset - Prediction Engine")
    print("=" * 60)
    print()

    config.ensure_dirs()
    _validate_prediction_config()

    # Load data if not provided
    if tweets is None or events is None:
        print_separator("Data Loading")
        tweet_collector = TweetCollector(config.tweets_path)
        reset_collector = ResetHistoryCollector(config.reset_history_path)
        tweets = tweet_collector.collect()
        events = reset_collector.collect()
        print(f"  Tweets:      {len(tweets)} tweets")
        print(f"  Reset events: {len(events)} events")
        print()

    # Step 1: Statistical feature extraction
    print_separator("Step 1: Statistical Feature Extraction")
    analyzer = SignalAnalyzer()
    analysis_features = analyzer.analyze(tweets, events)
    print(f"  Total tweets:          {analysis_features.tweet_count}")
    print(f"  Tweets in last 24h:    {analysis_features.recent_tweet_count}")
    if analysis_features.hours_since_last_reset is not None:
        print(f"  Hours since last reset: {analysis_features.hours_since_last_reset:.1f} hours")
    else:
        print(f"  Hours since last reset: no historical record")
    if analysis_features.avg_reset_interval_hours is not None:
        print(f"  Average reset interval: {analysis_features.avg_reset_interval_hours:.1f} hours")
    print()

    # Step 2: LLM signal analysis
    print_separator("Step 2: LLM Signal Analysis")
    llm_analyzer = DeepSeekAnalyzer(
        api_key=config.deepseek_api_key,
        model=config.deepseek_model,
    )
    print(f"  Analyzer: {llm_analyzer.__class__.__name__}")
    if tweets:
        signal_scores = llm_analyzer.analyze_tweets(tweets)
        batch_scores = llm_analyzer.analyze_batch([t.text for t in tweets])
        print(f"  Analyzed tweets:       {len(signal_scores)}")
        print(f"  Aggregated reset_intent:       {batch_scores.reset_intent:.2f}")
        print(f"  Aggregated reset_confirmation: {batch_scores.reset_confirmation:.2f}")
        print(f"  Aggregated limit_complaint:    {batch_scores.limit_complaint:.2f}")
        print(f"  Aggregated official_change:    {batch_scores.official_change:.2f}")
    else:
        signal_scores = []
        batch_scores = None
        print("  No tweets to analyze")
    print()

    # Step 3: Load model state and build prediction features
    print_separator("Step 3: Feature Building")
    state_manager = ModelStateManager(config.model_state_path)
    model_state = state_manager.load()
    if model_state is not None:
        print(f"  Loaded model_state: {model_state.sample_count} intervals")
    else:
        print("  model_state.json not found, using default prior parameters")

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
    print(f"  hours_since_last_reset: {pred_features.hours_since_last_reset}")
    print(f"  average_reset_interval: {pred_features.average_reset_interval}")
    print(f"  median_reset_interval:  {pred_features.median_reset_interval}")
    print(f"  time_pressure:          {pred_features.time_pressure:.3f}")
    print(f"  tibo_signal:            {pred_features.tibo_signal:.3f}")
    print(f"  community_signal:       {pred_features.community_signal:.3f}")
    print(f"  release_signal:         {pred_features.release_signal:.3f}")
    print()

    # Step 4: Survival model prediction
    print_separator("Step 4: Survival Model Prediction")
    predictor = ResetPredictor(
        horizons=config.prediction_horizons,
        default_interval=config.default_reset_interval_hours,
        model_state=model_state,
    )
    print(f"  Model: {predictor.model_version}")
    explanation = predictor.predict(pred_features)
    print(f"  Hazard rate:   {explanation.hazard_rate:.4f}/h")
    print(f"  Time pressure: {explanation.time_pressure:.2f}")
    if explanation.time_ratio is not None:
        print(f"  Time ratio:    {explanation.time_ratio:.2f}x")
    print()
    print("  Predicted probabilities:")
    for horizon, prob in explanation.probability.items():
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"    {horizon:>4s}: {prob:.2%}  {bar}")
    print()

    print_separator("Explanation")
    for reason in explanation.reasons:
        print(f"  • {reason}")
    print()

    # Step 5: Save results
    print_separator("Step 5: Save Results")
    import json
    from datetime import datetime, timezone

    prior_applied = analysis_features.avg_reset_interval_hours is None
    output_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_version": predictor.model_version,
        "features": pred_features.model_dump(mode="json"),
        "prediction": explanation.model_dump(mode="json"),
        "meta": {
            "prior_applied": prior_applied,
            "interval_count": analysis_features.reset_interval_count,
        },
    }
    json_path = config.output_dir / "prediction_latest.json"
    json_path.write_text(json.dumps(output_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  Saved: {json_path}")
    print()

    print_separator("Done")
    print("  Prediction pipeline complete.")
    print()
    print("=" * 60)
    print()


def run_pipeline() -> None:
    """
    Run the full pipeline: collect → analyze → predict → output
    """
    print()
    print("=" * 60)
    print("  WillTiboReset - Full Prediction Pipeline")
    print("=" * 60)
    print()

    config.ensure_dirs()

    # 1. Collect
    print_separator("Step 1: Data Collection")
    tweet_collector = TweetCollector(config.tweets_path)
    reset_collector = ResetHistoryCollector(config.reset_history_path)
    tweets = tweet_collector.collect()
    events = reset_collector.collect()
    print(f"  Tweets:      {len(tweets)} tweets")
    print(f"  Reset events: {len(events)} events")
    print()

    # 2. Statistical signal analysis
    print_separator("Step 2: Signal Analysis")
    analyzer = SignalAnalyzer()
    features = analyzer.analyze(tweets, events)
    for desc in features.to_signal_descriptions():
        print(f"  • {desc}")
    print()

    # 3. Predict
    run_prediction(tweets=tweets, events=events)


def main() -> int:
    """CLI entry point"""
    parser = argparse.ArgumentParser(
        description="WillTiboReset - AI prediction for Tibo/OpenAI quota resets",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show project status only",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run signal analysis only (no prediction)",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="Run prediction only (using already collected data)",
    )

    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.analyze:
        run_analysis()
    elif args.predict:
        run_prediction()
    else:
        run_pipeline()

    return 0


if __name__ == "__main__":
    sys.exit(main())
