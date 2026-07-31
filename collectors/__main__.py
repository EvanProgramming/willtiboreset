"""
collectors 模块入口。

运行: python -m collectors

流程:
  1. 运行所有 Collector（Tibo RSS、OpenAI RSS、Community）
  2. 合并、去重，保存到 data/tweets.json
  3. 运行 LLM 信号分析（Mock 或 Gemini）
  4. 输出结构化信号分数
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
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
    """全局去重"""
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
    """根据配置创建 LLM 分析器"""
    if config.has_gemini_credentials:
        print("  使用 Gemini API 进行信号分析")
        return GeminiAnalyzer(
            api_key=config.gemini_api_key,
            model=config.gemini_model,
        )
    else:
        print("  使用 Mock 分析器（基于关键词匹配）")
        print("  配置 GEMINI_API_KEY 可启用 Gemini API")
        return MockLLMAnalyzer()


def main() -> int:
    print()
    print("=" * 60)
    print("  WillTiboReset - 数据采集 & 信号分析")
    print("=" * 60)
    print()

    config.ensure_dirs()

    # ── Step 1: 数据采集 ──
    _print_sep("Step 1: 数据采集")
    all_tweets: list[Tweet] = []

    # Tibo RSS
    tibo = TiboRSSCollector(timeout=config.rss_request_timeout)
    tibo_tweets = tibo.collect()
    print(f"  TiboRSS:       {len(tibo_tweets)} 条")
    all_tweets.extend(tibo_tweets)

    # OpenAI RSS
    openai = OpenAIRSSCollector(timeout=config.rss_request_timeout)
    openai_tweets = openai.collect()
    print(f"  OpenAI RSS:    {len(openai_tweets)} 条")
    all_tweets.extend(openai_tweets)

    # Community (RSS + Mock)
    community = CommunityCollector(timeout=config.rss_request_timeout)
    community_tweets = community.collect()
    print(f"  Community:     {len(community_tweets)} 条")
    all_tweets.extend(community_tweets)

    # 合并去重
    all_tweets = _deduplicate(all_tweets)
    print(f"  去重后总计:    {len(all_tweets)} 条")
    print()

    # ── Step 2: 保存到 tweets.json ──
    _print_sep("Step 2: 保存数据")
    tweet_collector = TweetCollector(config.tweets_path)
    tweet_collector.save(all_tweets)
    print(f"  已保存到: {config.tweets_path}")
    print()

    # ── Step 3: LLM 信号分析 ──
    _print_sep("Step 3: LLM 信号分析")
    if not all_tweets:
        print("  ⚠ 无数据可分析")
        print()
        print("=" * 60)
        return 0

    analyzer = _create_analyzer()
    print(f"  分析 {len(all_tweets)} 条文本...")
    print()

    # 逐条分析
    scores = analyzer.analyze_tweets(all_tweets)

    _print_sep("逐条信号分数")
    for i, (tweet, score) in enumerate(zip(all_tweets, scores), 1):
        preview = tweet.text[:60].replace("\n", " ")
        print(f"  [{i}] {preview}...")
        print(f"      来源: {tweet.source} | 作者: {tweet.author}")
        print(
            f"      reset={score.reset_signal:.2f}  "
            f"limit={score.limit_discussion:.2f}  "
            f"release={score.release_signal:.2f}  "
            f"pressure={score.community_pressure:.2f}  "
            f"conf={score.confidence:.2f}"
        )
        if score.reason:
            print(f"      依据: {'; '.join(score.reason[:2])}")
        print()

    # 聚合信号
    _print_sep("聚合信号")
    batch = analyzer.analyze_batch([t.text for t in all_tweets])
    print(f"  reset_signal:       {batch.reset_signal:.2f}")
    print(f"  limit_discussion:   {batch.limit_discussion:.2f}")
    print(f"  release_signal:     {batch.release_signal:.2f}")
    print(f"  community_pressure: {batch.community_pressure:.2f}")
    print(f"  confidence:         {batch.confidence:.2f}")
    if batch.reason:
        print(f"  依据: {'; '.join(batch.reason[:3])}")
    print()

    # 保存分析结果
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
    print(f"  分析结果已保存: {analysis_path}")
    print()

    _print_sep("完成")
    print("  数据采集和信号分析完毕。")
    print()
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
