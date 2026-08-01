#!/usr/bin/env python3
"""
WillTiboReset - 主入口

预测 Tibo/OpenAI 是否会在未来 5h / 24h / 48h 内
重置 ChatGPT/Codex 使用额度。

Phase 3：完整预测管道（收集 → 分析 → LLM 信号 → 生存模型预测）

用法:
    python main.py              # 运行完整预测管道
    python main.py --status     # 仅显示项目状态
    python main.py --analyze    # 仅运行信号分析（不含预测）
    python main.py --predict    # 仅运行预测（使用已收集数据）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
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
    """运行预测前校验必须配置项。"""
    if not config.has_deepseek_credentials:
        raise RuntimeError(
            "DEEPSEEK_API_KEY 未配置。请在 .env 文件或 GitHub Actions Secrets 中设置。"
        )
    if not config.rss_feeds.get("tibo"):
        raise RuntimeError(
            "TIBO_RSS_URLS 未配置。Tibo 是核心数据源，必须至少配置一个 RSS URL。"
        )


def print_separator(title: str = "") -> None:
    """打印分隔线"""
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"{'─' * pad} {title} {'─' * (width - len(title) - 2 - pad)}")
    else:
        print("─" * width)


def show_status() -> None:
    """显示项目状态"""
    print()
    print("=" * 60)
    print("  WillTiboReset - 项目状态")
    print("=" * 60)
    print()

    # 配置
    print_separator("配置")
    print(f"  预测窗口:    {config.prediction_horizons} 小时")
    print(f"  置信度阈值:  {config.confidence_threshold}")
    print(f"  数据目录:    {config.data_dir}")
    print(f"  输出目录:    {config.output_dir}")
    print()

    # API 凭证状态
    print_separator("API 凭证")
    print(f"  Twitter:  {'✓ 已配置' if config.has_twitter_credentials else '✗ 未配置'}")
    print(f"  OpenAI:   {'✓ 已配置' if config.has_openai_credentials else '✗ 未配置'}")
    print()

    # 数据文件
    print_separator("数据文件")
    tweet_collector = TweetCollector(config.tweets_path)
    reset_collector = ResetHistoryCollector(config.reset_history_path)
    tweets = tweet_collector.collect()
    events = reset_collector.collect()
    print(f"  推文:        {len(tweets)} 条")
    print(f"  重置事件:    {len(events)} 条")
    print()

    print("=" * 60)
    print()


def run_analysis() -> None:
    """运行信号分析并打印结果"""
    print()
    print("=" * 60)
    print("  WillTiboReset - 信号分析")
    print("=" * 60)
    print()

    # 1. 收集数据
    print_separator("数据收集")
    tweet_collector = TweetCollector(config.tweets_path)
    reset_collector = ResetHistoryCollector(config.reset_history_path)
    tweets = tweet_collector.collect()
    events = reset_collector.collect()
    print(f"  加载推文:      {len(tweets)} 条")
    print(f"  加载重置事件:  {len(events)} 条")
    print()

    # 2. 分析信号
    print_separator("特征提取")
    analyzer = SignalAnalyzer()
    features = analyzer.analyze(tweets, events)
    print(f"  推文总数:          {features.tweet_count}")
    print(f"  最近 24h 推文:     {features.recent_tweet_count}")
    print(f"  独立作者数:        {features.unique_authors}")
    print(f"  历史重置事件:      {features.total_reset_events}")
    if features.hours_since_last_reset is not None:
        print(f"  距上次重置:        {features.hours_since_last_reset:.1f} 小时")
    if features.avg_reset_interval_hours is not None:
        print(f"  平均重置间隔:      {features.avg_reset_interval_hours:.1f} 小时")
    print()

    # 3. 信号描述
    print_separator("信号摘要")
    for desc in features.to_signal_descriptions():
        print(f"  • {desc}")
    print()

    print("=" * 60)
    print()


def run_prediction(tweets=None, events=None) -> None:
    """
    运行预测步骤：LLM 信号分析 → 特征构建 → 生存模型预测。

    如果未提供 tweets/events，则从数据文件加载。
    """
    print()
    print("=" * 60)
    print("  WillTiboReset - 预测引擎")
    print("=" * 60)
    print()

    config.ensure_dirs()
    _validate_prediction_config()

    # 加载数据（如果未传入）
    if tweets is None or events is None:
        print_separator("数据加载")
        tweet_collector = TweetCollector(config.tweets_path)
        reset_collector = ResetHistoryCollector(config.reset_history_path)
        tweets = tweet_collector.collect()
        events = reset_collector.collect()
        print(f"  推文:      {len(tweets)} 条")
        print(f"  重置事件:  {len(events)} 条")
        print()

    # Step 1: 统计特征提取
    print_separator("Step 1: 统计特征提取")
    analyzer = SignalAnalyzer()
    analysis_features = analyzer.analyze(tweets, events)
    print(f"  推文总数:          {analysis_features.tweet_count}")
    print(f"  最近 24h 推文:     {analysis_features.recent_tweet_count}")
    if analysis_features.hours_since_last_reset is not None:
        print(f"  距上次重置:        {analysis_features.hours_since_last_reset:.1f} 小时")
    else:
        print(f"  距上次重置:        无历史记录")
    if analysis_features.avg_reset_interval_hours is not None:
        print(f"  平均重置间隔:      {analysis_features.avg_reset_interval_hours:.1f} 小时")
    print()

    # Step 2: LLM 信号分析
    print_separator("Step 2: LLM 信号分析")
    llm_analyzer = DeepSeekAnalyzer(
        api_key=config.deepseek_api_key,
        model=config.deepseek_model,
    )
    print(f"  分析器: {llm_analyzer.__class__.__name__}")
    if tweets:
        signal_scores = llm_analyzer.analyze_tweets(tweets)
        batch_scores = llm_analyzer.analyze_batch([t.text for t in tweets])
        print(f"  分析推文数:        {len(signal_scores)}")
        print(f"  聚合 reset_intent:       {batch_scores.reset_intent:.2f}")
        print(f"  聚合 reset_confirmation: {batch_scores.reset_confirmation:.2f}")
        print(f"  聚合 limit_complaint:    {batch_scores.limit_complaint:.2f}")
        print(f"  聚合 official_change:    {batch_scores.official_change:.2f}")
    else:
        signal_scores = []
        batch_scores = None
        print("  无推文可分析")
    print()

    # Step 3: 加载模型状态并构建预测特征
    print_separator("Step 3: 特征构建")
    state_manager = ModelStateManager(config.model_state_path)
    model_state = state_manager.load()
    if model_state is not None:
        print(f"  已加载 model_state: {model_state.sample_count} 个 interval")
    else:
        print("  未找到 model_state.json，使用默认先验参数")

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

    # Step 4: 生存模型预测
    print_separator("Step 4: 生存模型预测")
    predictor = ResetPredictor(
        horizons=config.prediction_horizons,
        default_interval=config.default_reset_interval_hours,
        model_state=model_state,
    )
    print(f"  模型: {predictor.model_version}")
    explanation = predictor.predict(pred_features)
    print(f"  Hazard rate:   {explanation.hazard_rate:.4f}/h")
    print(f"  Time pressure: {explanation.time_pressure:.2f}")
    if explanation.time_ratio is not None:
        print(f"  Time ratio:    {explanation.time_ratio:.2f}x")
    print()
    print("  预测概率:")
    for horizon, prob in explanation.probability.items():
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"    {horizon:>4s}: {prob:.2%}  {bar}")
    print()

    print_separator("解释")
    for reason in explanation.reasons:
        print(f"  • {reason}")
    print()

    # Step 5: 保存结果
    print_separator("Step 5: 保存结果")
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
    print(f"  已保存: {json_path}")
    print()

    print_separator("完成")
    print("  预测管道执行完毕。")
    print()
    print("=" * 60)
    print()


def run_pipeline() -> None:
    """
    运行完整管道：收集 → 分析 → 预测 → 输出
    """
    print()
    print("=" * 60)
    print("  WillTiboReset - 完整预测管道")
    print("=" * 60)
    print()

    config.ensure_dirs()

    # 1. 收集
    print_separator("Step 1: 数据收集")
    tweet_collector = TweetCollector(config.tweets_path)
    reset_collector = ResetHistoryCollector(config.reset_history_path)
    tweets = tweet_collector.collect()
    events = reset_collector.collect()
    print(f"  推文:      {len(tweets)} 条")
    print(f"  重置事件:  {len(events)} 条")
    print()

    # 2. 统计信号分析
    print_separator("Step 2: 信号分析")
    analyzer = SignalAnalyzer()
    features = analyzer.analyze(tweets, events)
    for desc in features.to_signal_descriptions():
        print(f"  • {desc}")
    print()

    # 3. 预测
    run_prediction(tweets=tweets, events=events)


def main() -> int:
    """CLI 入口"""
    parser = argparse.ArgumentParser(
        description="WillTiboReset - AI 预测 Tibo/OpenAI 额度重置",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="仅显示项目状态",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="仅运行信号分析（不含预测）",
    )
    parser.add_argument(
        "--predict",
        action="store_true",
        help="仅运行预测（使用已收集数据）",
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
