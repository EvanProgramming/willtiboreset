"""
Entry point for the collectors module.

Run: python -m collectors

Pipeline:
  1. Run all collectors (Tibo RSS, OpenAI RSS, Community)
  2. Merge, deduplicate, and save to data/tweets.json
  3. Run LLM signal analysis (Mock or Gemini)
  4. Output structured signal scores
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from collectors import (
    CommunityCollector,
    OpenAIRSSCollector,
    TiboRSSCollector,
    TweetCollector,
)
from config import config
from model.data_models import Tweet
from analyzer.llm_signal import LLMAnalyzer, GeminiAnalyzer, MockLLMAnalyzer


def _print_sep(title: str = "") -> None:
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"{'─' * pad} {title} {'─' * (width - len(title) - 2 - pad)}")
    else:
        print("─" * width)


def _deduplicate(tweets: list[Tweet]) -> list[Tweet]:
    """Global deduplication"""
    from collectors.rss_base import _build_dedup_key
    seen: set[str] = set()
    result: list[Tweet] = []
    for t in tweets:
        key = _build_dedup_key(t)
        if key not in seen:
            seen.add(key)
            result.append(t)
    return result


def _create_analyzer() -> LLMAnalyzer:
    """Create an LLM analyzer based on configuration"""
    if config.has_gemini_credentials:
        print("  Using Gemini API for signal analysis")
        return GeminiAnalyzer(
            api_key=config.gemini_api_key,
            model=config.gemini_model,
        )
    else:
        print("  Using Mock analyzer (keyword matching)")
        print("  Set GEMINI_API_KEY to enable Gemini API")
        return MockLLMAnalyzer()


def main() -> int:
    print()
    print("=" * 60)
    print("  WillTiboReset - Data Collection & Signal Analysis")
    print("=" * 60)
    print()

    config.ensure_dirs()

    # ── Step 1: Data collection ──
    _print_sep("Step 1: Data Collection")
    all_tweets: list[Tweet] = []

    # Tibo RSS (via self-hosted RSSHub with Twitter cookies)
    tibo = TiboRSSCollector(timeout=config.rss_request_timeout)
    tibo_tweets = tibo.collect()
    print(f"  TiboRSS:       {len(tibo_tweets)} tweets")
    all_tweets.extend(tibo_tweets)

    # OpenAI RSS
    openai = OpenAIRSSCollector(timeout=config.rss_request_timeout)
    openai_tweets = openai.collect()
    print(f"  OpenAI RSS:    {len(openai_tweets)} tweets")
    all_tweets.extend(openai_tweets)

    # Community (RSS + Mock)
    community = CommunityCollector(timeout=config.rss_request_timeout)
    community_tweets = community.collect()
    print(f"  Community:     {len(community_tweets)} tweets")
    all_tweets.extend(community_tweets)

    # Merge and deduplicate
    all_tweets = _deduplicate(all_tweets)
    print(f"  Total after dedup: {len(all_tweets)} tweets")
    print()

    # ── Step 2: Save to tweets.json ──
    _print_sep("Step 2: Save Data")
    tweet_collector = TweetCollector(config.tweets_path)
    tweet_collector.save(all_tweets)
    print(f"  Saved to: {config.tweets_path}")
    print()

    # ── Step 3: LLM signal analysis ──
    _print_sep("Step 3: LLM Signal Analysis")
    if not all_tweets:
        print("  ⚠ No data to analyze")
        print()
        print("=" * 60)
        return 0

    analyzer = _create_analyzer()
    print(f"  Analyzing {len(all_tweets)} tweets...")
    print()

    # Analyze tweet by tweet
    scores = analyzer.analyze_tweets(all_tweets)

    _print_sep("Per-tweet Signal Scores")
    for i, (tweet, score) in enumerate(zip(all_tweets, scores), 1):
        preview = tweet.text[:60].replace("\n", " ")
        print(f"  [{i}] {preview}...")
        print(f"      source: {tweet.source} | author: {tweet.author}")
        print(
            f"      reset={score.reset_signal:.2f}  "
            f"limit={score.limit_discussion:.2f}  "
            f"release={score.release_signal:.2f}  "
            f"pressure={score.community_pressure:.2f}  "
            f"conf={score.confidence:.2f}"
        )
        if score.reason:
            print(f"      reasons: {'; '.join(score.reason[:2])}")
        print()

    # Aggregate signals
    _print_sep("Aggregated Signals")
    batch = analyzer.analyze_batch([t.text for t in all_tweets])
    print(f"  reset_signal:       {batch.reset_signal:.2f}")
    print(f"  limit_discussion:   {batch.limit_discussion:.2f}")
    print(f"  release_signal:     {batch.release_signal:.2f}")
    print(f"  community_pressure: {batch.community_pressure:.2f}")
    print(f"  confidence:         {batch.confidence:.2f}")
    if batch.reason:
        print(f"  reasons: {'; '.join(batch.reason[:3])}")
    print()

    # Save analysis results
    analysis_path = config.output_dir / "signal_analysis.json"
    analysis_data = {
        "timestamp": __import__("datetime").datetime.now(
            __import__("datetime").timezone.utc
        ).isoformat(),
        "tweet_count": len(all_tweets),
        "batch_scores": batch.model_dump(mode="json"),
        "per_tweet_scores": [
            {
                "tweet": t.model_dump(mode="json"),
                "scores": s.model_dump(mode="json"),
            }
            for t, s in zip(all_tweets, scores)
        ],
    }
    analysis_path.write_text(
        json.dumps(analysis_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  Analysis saved: {analysis_path}")
    print()

    _print_sep("Done")
    print("  Data collection and signal analysis complete.")
    print()
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
