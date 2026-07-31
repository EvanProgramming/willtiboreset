"""
output 模块 - 输出格式化

将预测结果格式化为 JSON / 文本等格式，
并持久化到 output 目录。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from model.data_models import PredictionResult


class OutputFormatter:
    """
    输出格式化器。

    支持将 PredictionResult 输出为 JSON 文件和可读文本。
    """

    def __init__(self, output_dir: Path):
        self._output_dir = output_dir

    def to_json(self, result: PredictionResult) -> str:
        """转换为 JSON 字符串"""
        return result.model_dump_json(indent=2)

    def to_text(self, result: PredictionResult) -> str:
        """转换为人类可读文本"""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("WillTiboReset - 预测结果")
        lines.append("=" * 60)
        lines.append(f"生成时间: {result.timestamp.isoformat()}")
        lines.append(f"模型版本: {result.model_version}")
        lines.append("")

        if result.signals_used:
            lines.append("── 使用信号 ──")
            for s in result.signals_used:
                lines.append(f"  • {s}")
            lines.append("")

        lines.append("── 预测结果 ──")
        if not result.predictions:
            lines.append("  （无预测数据）")
        else:
            for p in result.predictions:
                verdict = "可能重置" if p.will_reset else "不太可能重置"
                lines.append(
                    f"  {p.horizon_hours}h 内: {verdict} "
                    f"(置信度: {p.confidence:.0%})"
                )
                if p.reasoning:
                    lines.append(f"    依据: {p.reasoning}")
        lines.append("")

        if result.notes:
            lines.append(f"备注: {result.notes}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def save(
        self,
        result: PredictionResult,
        filename: str | None = None,
    ) -> Path:
        """
        保存预测结果到 output 目录。

        同时保存 JSON 和文本格式。

        Returns:
            JSON 文件路径
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp_str = result.timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"prediction_{timestamp_str}"

        json_path = self._output_dir / f"{filename}.json"
        text_path = self._output_dir / f"{filename}.txt"

        json_path.write_text(self.to_json(result), encoding="utf-8")
        text_path.write_text(self.to_text(result), encoding="utf-8")

        return json_path


__all__ = ["OutputFormatter"]
