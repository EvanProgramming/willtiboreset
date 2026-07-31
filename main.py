#!/usr/bin/env python3
"""
WillTiboReset - 主入口

预测 Tibo/OpenAI 是否会在未来 5h / 24h / 48h 内
重置 ChatGPT/Codex 使用额度。

Phase 1：数据收集 + 信号分析管道
预测逻辑将在后续 Phase 中接入 LLM / 统计模型。

用法:
    python main.py              # 运行完整管道（收集→分析→输出）
    python main.py --status     # 仅显示项目状态
    python main.py --analyze    # 仅运行信号分析
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
from collectors import ResetHistoryCollector, TweetCollector
from config import config
from model.data_models import PredictionResult
from model.predictor import PlaceholderPredictor
from output import OutputFormatter


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


def run_pipeline() -> None:
    """
    运行完整管道：收集 → 分析 → 预测 → 输出

    预测步骤使用 PlaceholderPredictor，当前会报告
    预测逻辑尚未实现，管道其余部分正常执行。
    """
    print()
    print("=" * 60)
    print("  WillTiboReset - 预测管道")
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

    # 2. 分析
    print_separator("Step 2: 信号分析")
    analyzer = SignalAnalyzer()
    features = analyzer.analyze(tweets, events)
    signals = features.to_signal_descriptions()
    for desc in signals:
        print(f"  • {desc}")
    print()

    # 3. 预测
    print_separator("Step 3: 预测")
    predictor = PlaceholderPredictor()
    print(f"  预测器: {predictor.model_version}")
    try:
        result = predictor.predict(tweets, events, config.prediction_horizons)
        # 如果预测成功，输出结果
        formatter = OutputFormatter(config.output_dir)
        json_path = formatter.save(result)
        print(formatter.to_text(result))
        print(f"  结果已保存: {json_path}")
    except NotImplementedError:
        print("  ⚠ 预测逻辑尚未实现（Phase 1）")
        print("  后续 Phase 将接入 LLM 分析器或统计预测模型。")
        print()

        # 仍然输出分析摘要
        summary = PredictionResult(
            predictions=[],
            signals_used=signals,
            model_version=predictor.model_version,
            notes="Phase 1: 预测逻辑尚未实现，仅输出信号分析摘要。",
        )
        formatter = OutputFormatter(config.output_dir)
        json_path = formatter.save(summary)
        print(f"  信号分析摘要已保存: {json_path}")
    print()

    # 4. 完成
    print_separator("完成")
    print("  管道执行完毕。")
    print()
    print("=" * 60)
    print()


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
        help="仅运行信号分析",
    )

    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.analyze:
        run_analysis()
    else:
        run_pipeline()

    return 0


if __name__ == "__main__":
    sys.exit(main())
