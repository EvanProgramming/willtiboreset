"""
output module - Output formatting

Formats prediction results as JSON / text and persists them to the output directory.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from model.data_models import PredictionResult


class OutputFormatter:
    """
    Output formatter.

    Supports writing PredictionResult to JSON files and human-readable text.
    """

    def __init__(self, output_dir: Path):
        self._output_dir = output_dir

    def to_json(self, result: PredictionResult) -> str:
        """Convert to JSON string"""
        return result.model_dump_json(indent=2)

    def to_text(self, result: PredictionResult) -> str:
        """Convert to human-readable text"""
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("WillTiboReset - Prediction Result")
        lines.append("=" * 60)
        lines.append(f"Generated at: {result.timestamp.isoformat()}")
        lines.append(f"Model version: {result.model_version}")
        lines.append("")

        if result.signals_used:
            lines.append("-- Signals Used --")
            for s in result.signals_used:
                lines.append(f"  • {s}")
            lines.append("")

        lines.append("-- Prediction Result --")
        if not result.predictions:
            lines.append("  (No prediction data)")
        else:
            for p in result.predictions:
                verdict = "Likely reset" if p.will_reset else "Unlikely reset"
                lines.append(
                    f"  Within {p.horizon_hours}h: {verdict} "
                    f"(confidence: {p.confidence:.0%})"
                )
                if p.reasoning:
                    lines.append(f"    Basis: {p.reasoning}")
        lines.append("")

        if result.notes:
            lines.append(f"Notes: {result.notes}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def save(
        self,
        result: PredictionResult,
        filename: str | None = None,
    ) -> Path:
        """
        Save prediction result to the output directory.

        Saves both JSON and text formats.

        Returns:
            Path to the JSON file
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
