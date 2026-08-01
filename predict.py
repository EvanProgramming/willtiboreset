#!/usr/bin/env python3
"""
WillTiboReset - 一键预测脚本

运行: python predict.py
输出: output/prediction.json

自动完成：
  1. 获取最新数据（Tibo / OpenAI / Community RSS）
  2. 分析文本信号（Gemini LLM）
  3. 运行预测模型（Adaptive Bayesian Survival Model）
  4. 生成最终预测文件 output/prediction.json

输出格式：
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

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyzer import SignalAnalyzer
from analyzer.llm_signal import DeepSeekAnalyzer, LLMAnalyzer
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
# 数据采集
# ──────────────────────────────────────────────

def collect_data() -> list[Tweet]:
    """
    获取最新数据：运行所有 Collector 并去重。

    采集源：
      - TiboRSSCollector: Tibo 相关 RSS Feed
      - OpenAIRSSCollector: OpenAI 官方 RSS Feed
      - CommunityCollector: 社区 RSS + mock 数据
    """
    all_tweets: list[Tweet] = []

    tibo = TiboRSSCollector(timeout=config.rss_request_timeout)
    all_tweets.extend(tibo.collect())

    openai = OpenAIRSSCollector(timeout=config.rss_request_timeout)
    all_tweets.extend(openai.collect())

    community = CommunityCollector(timeout=config.rss_request_timeout)
    all_tweets.extend(community.collect())

    # 全局去重
    seen: set[str] = set()
    result: list[Tweet] = []
    for t in all_tweets:
        key = _build_dedup_key(t)
        if key not in seen:
            seen.add(key)
            result.append(t)

    return result


def create_analyzer() -> LLMAnalyzer:
    """根据配置创建 DeepSeek LLM 分析器。DEEPSEEK_API_KEY 为必须配置。"""
    if not config.has_deepseek_credentials:
        raise RuntimeError(
            "DEEPSEEK_API_KEY 未配置。请在 .env 文件或 GitHub Actions Secrets 中设置。"
        )
    return DeepSeekAnalyzer(
        api_key=config.deepseek_api_key,
        model=config.deepseek_model,
    )


def validate_configuration() -> None:
    """在运行前校验必须配置项。缺失时直接报错，禁止静默 fallback。"""
    if not config.has_deepseek_credentials:
        raise RuntimeError(
            "DEEPSEEK_API_KEY 未配置。请在 .env 文件或 GitHub Actions Secrets 中设置。"
        )
    if not config.rss_feeds.get("tibo"):
        raise RuntimeError(
            "TIBO_RSS_URLS 未配置。Tibo 是核心数据源，必须至少配置一个 RSS URL。"
        )


# ──────────────────────────────────────────────
# 置信度计算
# ──────────────────────────────────────────────

def compute_confidence(
    prob_24h: float,
    has_history: bool,
    llm_confidence: float,
) -> str:
    """
    计算预测置信度。

    综合 3 个因素：
      - 概率清晰度（距离 0.5 越远越确定）
      - LLM 置信度
      - 是否有历史数据
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
# 主流程
# ──────────────────────────────────────────────

def main() -> int:
    config.ensure_dirs()
    validate_configuration()

    print("WillTiboReset — 预测引擎")
    print("=" * 50)

    # ── 1. 获取最新数据 ──
    print("\n[1/4] 获取最新数据...")
    tweets = collect_data()

    # 保存到 tweets.json
    tweet_collector = TweetCollector(config.tweets_path)
    tweet_collector.save(tweets)
    print(f"  收集信号: {len(tweets)} 条")

    # 加载历史 reset 事件
    reset_collector = ResetHistoryCollector(config.reset_history_path)
    events = reset_collector.collect()
    print(f"  历史重置事件: {len(events)} 条")

    # ── 2. 分析文本信号 ──
    print("\n[2/4] 分析文本信号...")
    analyzer = create_analyzer()
    print(f"  分析器: {analyzer.__class__.__name__}")

    signal_scores = []
    batch_scores = None

    if tweets:
        signal_scores = analyzer.analyze_tweets(tweets)
        batch_scores = analyzer.analyze_batch([t.text for t in tweets])
        print(f"  reset_signal:       {batch_scores.reset_signal:.2f}")
        print(f"  limit_discussion:   {batch_scores.limit_discussion:.2f}")
        print(f"  release_signal:     {batch_scores.release_signal:.2f}")
        print(f"  community_pressure: {batch_scores.community_pressure:.2f}")
        print(f"  llm_confidence:     {batch_scores.confidence:.2f}")

    # 统计特征提取
    signal_analyzer = SignalAnalyzer()
    analysis_features = signal_analyzer.analyze(tweets, events)

    if analysis_features.hours_since_last_reset is not None:
        print(f"  距上次重置: {analysis_features.hours_since_last_reset:.1f}h")
    else:
        print("  距上次重置: 无历史记录")

    # ── 3. 加载模型状态并运行预测模型 ──
    print("\n[3/4] 加载模型状态...")
    state_manager = ModelStateManager(config.model_state_path)
    model_state = state_manager.load()
    if model_state is not None:
        print(f"  已加载 model_state: {model_state.sample_count} 个 interval")
        print(f"  后验平均间隔: {model_state.average_interval_hours:.1f}h")
    else:
        print("  未找到 model_state.json，将使用默认先验参数")

    print("\n[3/4] 运行预测模型...")
    pred_features = build_features(
        hours_since_last_reset=analysis_features.hours_since_last_reset,
        average_reset_interval=analysis_features.avg_reset_interval_hours,
        signal_scores=signal_scores if signal_scores else None,
        interval_count=analysis_features.reset_interval_count,
        model_state=model_state,
    )

    predictor = ResetPredictor(
        horizons=config.prediction_horizons,
        default_interval=config.default_reset_interval_hours,
        model_state=model_state,
    )
    explanation = predictor.predict(pred_features)

    print(f"  模型: {predictor.model_version}")
    print(f"  Hazard rate: {explanation.hazard_rate:.4f}/h")
    if explanation.time_ratio is not None:
        print(f"  Time ratio:  {explanation.time_ratio:.2f}x")

    for horizon, prob in explanation.probability.items():
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  {horizon:>4s}: {prob:.1%}  {bar}")

    # ── 4. 生成预测文件 ──
    print("\n[4/4] 生成预测文件...")

    prob_24h = explanation.probability.get("24h", 0.0)
    has_history = analysis_features.hours_since_last_reset is not None
    llm_conf = batch_scores.confidence if batch_scores else 0.0
    confidence = compute_confidence(prob_24h, has_history, llm_conf)
    prior_applied = analysis_features.avg_reset_interval_hours is None

    output = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "prediction": {
            "within_5h": explanation.probability.get("5h", 0.0),
            "within_24h": explanation.probability.get("24h", 0.0),
            "within_48h": explanation.probability.get("48h", 0.0),
        },
        "confidence": confidence,
        "signals": {
            "tweet_count": len(tweets),
            "hours_since_last_reset": pred_features.hours_since_last_reset,
            "average_reset_interval": pred_features.average_reset_interval,
            "median_reset_interval": analysis_features.median_reset_interval_hours,
            "std_reset_interval": analysis_features.std_reset_interval_hours,
            "min_reset_interval": analysis_features.min_reset_interval_hours,
            "max_reset_interval": analysis_features.max_reset_interval_hours,
            "interval_confidence": analysis_features.interval_confidence,
            "time_ratio": explanation.time_ratio,
            "hazard_rate": explanation.hazard_rate,
            "tibo_signal": round(pred_features.tibo_signal, 4),
            "community_signal": round(pred_features.community_signal, 4),
            "release_signal": round(pred_features.release_signal, 4),
            "llm_scores": (
                batch_scores.model_dump(mode="json") if batch_scores else None
            ),
            "prior_applied": prior_applied,
            "interval_count": analysis_features.reset_interval_count,
        },
        "reasons": explanation.reasons,
    }

    output_path = config.output_dir / "prediction.json"
    output_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"  已保存: {output_path}")

    print("\n" + "=" * 50)
    print("预测完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
