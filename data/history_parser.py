"""
WillTiboReset - 历史重置数据解析模块

将自然语言 / 半结构化文本解析为标准 reset_history.json。

支持的输入格式：
    1. 半结构化日志（用户提供的格式）：
       每个事件包含日期、时间、CONFIRMED RESET 标记、Scope、Source 等字段

    2. 纯自然语言：
       "2026年7月10日，Tibo在X表示limit已经reset"

解析策略：
    - 优先匹配结构化标记（CONFIRMED RESET 等）
    - 对自然语言使用日期和关键词启发式
    - 不确定的事件标记 confidence 并存入 uncertain_events.json
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ──────────────────────────────────────────────
# 月份映射
# ──────────────────────────────────────────────

_MONTH_MAP: dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    # 中文月份
    "1月": 1, "2月": 2, "3月": 3, "4月": 4, "5月": 5, "6月": 6,
    "7月": 7, "8月": 8, "9月": 9, "10月": 10, "11月": 11, "12月": 12,
}

# 结构化日期模式：Jun 29, Jul 10
_DATE_PATTERN = re.compile(
    r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})$",
    re.IGNORECASE | re.MULTILINE,
)

# 时间模式：00:00 UTC, 05:30 UTC
_TIME_PATTERN = re.compile(r"^(\d{1,2}):(\d{2})\s*UTC\s*$", re.IGNORECASE)

# 中文日期模式：2026年7月10日
_CN_DATE_PATTERN = re.compile(
    r"(\d{4})年(\d{1,2})月(\d{1,2})日"
)

# 确认重置标记
_CONFIRMED_MARKERS = ["CONFIRMED RESET", "confirmed reset"]

# 非重置标记（回复、下行信号等）
_NON_RESET_MARKERS = [
    "REPLY", "Downward signal", "Archived signal",
    "UPWARD SIGNAL",
]

# 元数据行，不进入 title / notes
_METADATA_KEYS = {"SCOPE", "SOURCE", "COMPENSATION", "VIEW SOURCE", "VIEW REPLY"}


# ──────────────────────────────────────────────
# 数据结构
# ──────────────────────────────────────────────

@dataclass
class ParsedEvent:
    """解析出的事件（未分类）"""
    reset_time: Optional[datetime] = None
    title: str = ""
    description: str = ""
    scope: str = ""
    source: str = ""
    is_confirmed_reset: bool = False
    confidence: float = 0.0
    notes: str = ""


# ──────────────────────────────────────────────
# 源映射
# ──────────────────────────────────────────────

def _map_source(source_text: str) -> str:
    """将源文本映射为标准 SignalSource 值"""
    s = source_text.lower()
    if "openai status" in s or "status page" in s:
        return "openai_status"
    if "tibo" in s or "x" in s or "twitter" in s or "codex radar" in s:
        return "twitter"
    if "reddit" in s:
        return "reddit"
    if "manual" in s:
        return "manual"
    return "other"


# ──────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────

def _is_metadata_line(line_upper: str) -> bool:
    """判断是否为结构化元数据键"""
    return line_upper in _METADATA_KEYS


def _is_noise_line(line: str) -> bool:
    """判断是否为可忽略的社交媒体统计/评分行"""
    line_upper = line.upper()
    if re.match(r"^(Replies|Reposts|Likes|Views):\s", line, re.IGNORECASE):
        return True
    if re.match(r"^\d+[hH]\s*[+-]\d+", line):
        return True
    if line_upper in ("COMPENSATION", "VIEW SOURCE", "VIEW REPLY"):
        return True
    return False


def _infer_source_from_block(block: list[str]) -> str:
    """从整个 block 推断来源"""
    text = "\n".join(block).lower()
    return _map_source(text)


# ──────────────────────────────────────────────
# 结构化解析器
# ──────────────────────────────────────────────

def _find_date_time_pairs(lines: list[str]) -> list[tuple[int, int, datetime]]:
    """找出所有 (日期行索引, 时间行索引, datetime) 三元组"""
    pairs: list[tuple[int, int, datetime]] = []
    i = 0
    while i < len(lines):
        date_match = _DATE_PATTERN.match(lines[i])
        if date_match and i + 1 < len(lines):
            time_match = _TIME_PATTERN.match(lines[i + 1])
            if time_match:
                # 默认年份由调用方传入，这里用占位，后续替换
                pairs.append((i, i + 1, date_match, time_match))
                i += 2
                continue
        i += 1

    result: list[tuple[int, int, datetime]] = []
    for date_idx, time_idx, date_match, time_match in pairs:
        month = _MONTH_MAP[date_match.group(1).lower()]
        day = int(date_match.group(2))
        hour = int(time_match.group(1))
        minute = int(time_match.group(2))
        result.append((date_idx, time_idx, datetime(2000, month, day, hour, minute, tzinfo=timezone.utc)))
    return result


def _is_content_line(line: str) -> bool:
    """判断是否为代表事件实质内容的行（非噪声、非日期时间）"""
    if _is_noise_line(line):
        return False
    if _DATE_PATTERN.match(line) or _TIME_PATTERN.match(line):
        return False
    return True


def _chunk_lines(lines: list[str]) -> list[list[str]]:
    """按 'View source' / 'View reply' 将日志分割为事件块"""
    chunks: list[list[str]] = []
    current: list[str] = []
    for line in lines:
        current.append(line)
        if line.lower() in ("view source", "view reply"):
            chunks.append(current)
            current = []
    if current:
        chunks.append(current)
    return chunks


def _extract_date_time_pair(
    lines: list[str], default_year: int
) -> tuple[Optional[datetime], list[str]]:
    """
    从块开头提取日期+时间，返回 (datetime, 剩余行)。
    如果开头不是日期+时间，返回 (None, 原列表)。
    """
    if len(lines) < 2:
        return None, lines
    date_match = _DATE_PATTERN.match(lines[0])
    time_match = _TIME_PATTERN.match(lines[1])
    if not date_match or not time_match:
        return None, lines
    month = _MONTH_MAP[date_match.group(1).lower()]
    day = int(date_match.group(2))
    hour = int(time_match.group(1))
    minute = int(time_match.group(2))
    try:
        dt = datetime(default_year, month, day, hour, minute, tzinfo=timezone.utc)
    except ValueError:
        return None, lines
    return dt, lines[2:]


def _parse_structured(text: str, default_year: int = 2026) -> list[ParsedEvent]:
    """
    解析半结构化日志格式。

    每个事件以 'View source' / 'View reply' 结尾。
    大部分块以日期+时间开头，后面跟元数据；第一个块没有日期，
    需要使用下一个块的日期+时间。
    """
    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    chunks = _chunk_lines(raw_lines)

    events: list[ParsedEvent] = []

    for idx, chunk in enumerate(chunks):
        # 尝试从当前块开头提取日期+时间
        dt, metadata_lines = _extract_date_time_pair(chunk, default_year)

        if dt is None:
            # 当前块无日期：使用下一个块开头的日期+时间
            if idx + 1 >= len(chunks):
                continue
            next_dt, _ = _extract_date_time_pair(chunks[idx + 1], default_year)
            if next_dt is None:
                continue
            dt = next_dt
            # 当前块整体作为元数据
            metadata_lines = chunk
        else:
            # 当前块以日期+时间开头：元数据是剩余部分
            # 但如果剩余部分为空（文件末尾只有日期），则跳过
            if not metadata_lines:
                continue

        # 剔除噪声行
        clean = [line for line in metadata_lines if not _is_noise_line(line)]
        if not clean:
            continue

        # 确认重置判断
        text_upper = "\n".join(line.upper() for line in clean)
        is_confirmed = any(marker.upper() in text_upper for marker in _CONFIRMED_MARKERS)

        # 提取 title
        title = ""
        if is_confirmed:
            for i, line in enumerate(clean):
                if any(marker.upper() in line.upper() for marker in _CONFIRMED_MARKERS):
                    if i + 1 < len(clean):
                        candidate = clean[i + 1]
                        if not _is_metadata_line(candidate.upper()):
                            title = candidate
                    break
            if not title:
                title = next(
                    (line for line in clean if not _is_metadata_line(line.upper())), ""
                )
        else:
            title = next(
                (line for line in clean if not _is_metadata_line(line.upper())), ""
            )

        # 提取 Scope / Source
        scope = ""
        source_text = ""
        for i, line in enumerate(clean):
            line_upper = line.upper()
            if line_upper == "SCOPE" and i + 1 < len(clean):
                scope = clean[i + 1]
            if line_upper == "SOURCE" and i + 1 < len(clean):
                source_text = clean[i + 1]

        source = _map_source(source_text) if source_text else _infer_source_from_block(clean)

        confidence = 1.0 if is_confirmed else 0.3

        # notes
        note_lines: list[str] = []
        skip_lines = {title, scope, source_text, "CONFIRMED RESET"}
        for line in clean:
            if line in skip_lines or _is_metadata_line(line.upper()):
                continue
            note_lines.append(line)
        notes = " | ".join(note_lines[:5])

        events.append(ParsedEvent(
            reset_time=dt,
            title=title,
            scope=scope,
            source=source,
            is_confirmed_reset=is_confirmed,
            confidence=confidence,
            notes=notes,
        ))

    return events


# ──────────────────────────────────────────────
# 自然语言解析器（简化版）
# ──────────────────────────────────────────────

def _parse_natural_language(text: str, default_year: int = 2026) -> list[ParsedEvent]:
    """
    解析自然语言文本中的重置事件。

    例如："2026年7月10日，Tibo在X表示limit已经reset"
    """
    events: list[ParsedEvent] = []

    # 按行或按句分割
    sentences = re.split(r"[。\n]", text)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # 提取日期
        cn_date_match = _CN_DATE_PATTERN.search(sentence)
        if not cn_date_match:
            continue

        year = int(cn_date_match.group(1))
        month = int(cn_date_match.group(2))
        day = int(cn_date_match.group(3))

        # 尝试提取时间（如果有）
        time_match = re.search(r"(\d{1,2}):(\d{2})", sentence)
        hour = int(time_match.group(1)) if time_match else 0
        minute = int(time_match.group(2)) if time_match else 0

        try:
            reset_time = datetime(
                year, month, day, hour, minute,
                tzinfo=timezone.utc,
            )
        except ValueError:
            continue

        # 判断是否是重置事件
        reset_keywords = ["reset", "重置", "恢复", "额度"]
        is_reset = any(kw in sentence.lower() for kw in reset_keywords)

        # 判断来源
        if "tibo" in sentence.lower():
            source = "twitter"
        elif "openai" in sentence.lower():
            source = "openai_status"
        elif "社区" in sentence or "reddit" in sentence.lower():
            source = "reddit"
        else:
            source = "other"

        # 置信度
        if is_reset:
            confirm_keywords = ["确认", "confirmed", "宣布", "announced", "表示"]
            confidence = 0.8 if any(kw in sentence.lower() for kw in confirm_keywords) else 0.5
        else:
            confidence = 0.2

        events.append(ParsedEvent(
            reset_time=reset_time,
            title=sentence[:200],
            source=source,
            is_confirmed_reset=is_reset and confidence >= 0.5,
            confidence=confidence,
            notes=sentence,
        ))

    return events


# ──────────────────────────────────────────────
# 主接口
# ──────────────────────────────────────────────

@dataclass
class ParseResult:
    """解析结果"""
    confirmed: list[dict] = field(default_factory=list)
    uncertain: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        """返回可序列化的 JSON 结构"""
        return {
            "confirmed": self.confirmed,
            "uncertain": self.uncertain,
        }


def parse_history_text(text: str, default_year: int = 2026) -> ParseResult:
    """
    解析历史重置文本。

    自动检测输入格式（结构化日志 or 自然语言），
    分类为 confirmed / uncertain 两个列表。

    Args:
        text: 原始文本
        default_year: 结构化日期缺省年份

    Returns:
        ParseResult 包含 confirmed 和 uncertain 事件列表
    """
    # 尝试结构化解析
    has_structured = bool(_DATE_PATTERN.search(text, re.MULTILINE))
    events: list[ParsedEvent] = []

    if has_structured:
        events = _parse_structured(text, default_year)
    else:
        events = _parse_natural_language(text, default_year)

    result = ParseResult()

    for ev in events:
        if ev.reset_time is None:
            continue

        entry = {
            "reset_time": ev.reset_time.isoformat(),
            "source": ev.source,
            "confidence": round(ev.confidence, 2),
            "notes": ev.title or ev.notes,
        }

        if ev.is_confirmed_reset and ev.confidence >= 0.5:
            result.confirmed.append(entry)
        else:
            result.uncertain.append(entry)

    return result


def save_parsed_history(
    result: ParseResult,
    history_path: Path,
    uncertain_path: Path,
) -> None:
    """
    将解析结果保存到 JSON 文件。

    Args:
        result: 解析结果
        history_path: reset_history.json 路径
        uncertain_path: uncertain_events.json 路径
    """
    history_path.parent.mkdir(parents=True, exist_ok=True)

    history_path.write_text(
        json.dumps(result.confirmed, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    uncertain_path.write_text(
        json.dumps(result.uncertain, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def parse_and_save(
    text: str,
    history_path: Path,
    uncertain_path: Optional[Path] = None,
    default_year: int = 2026,
) -> ParseResult:
    """
    一步完成：解析文本 + 保存到文件。

    Args:
        text: 原始文本
        history_path: reset_history.json 路径
        uncertain_path: uncertain_events.json 路径（默认为同目录下）
        default_year: 结构化日期缺省年份

    Returns:
        ParseResult
    """
    if uncertain_path is None:
        uncertain_path = history_path.parent / "uncertain_events.json"

    result = parse_history_text(text, default_year)
    save_parsed_history(result, history_path, uncertain_path)
    return result


def main() -> int:
    """CLI 入口：解析 history.txt 并写入 reset_history.json"""
    import argparse

    parser = argparse.ArgumentParser(
        description="将自然语言/半结构化历史记录解析为 reset_history.json",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("history.txt"),
        help="输入历史记录文件路径（默认: history.txt）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reset_history.json"),
        help="输出 reset_history.json 路径（默认: data/reset_history.json）",
    )
    parser.add_argument(
        "--uncertain",
        type=Path,
        default=Path("data/uncertain_events.json"),
        help="输出 uncertain_events.json 路径（默认: data/uncertain_events.json）",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2026,
        help="缺省年份（默认: 2026）",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"错误：输入文件不存在 {args.input}", file=sys.stderr)
        return 1

    text = args.input.read_text(encoding="utf-8")
    result = parse_and_save(
        text,
        history_path=args.output,
        uncertain_path=args.uncertain,
        default_year=args.year,
    )

    print(f"已解析 {args.input}")
    print(f"  确认事件: {len(result.confirmed)} 条 → {args.output}")
    print(f"  不确定事件: {len(result.uncertain)} 条 → {args.uncertain}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "ParsedEvent",
    "ParseResult",
    "parse_history_text",
    "save_parsed_history",
    "parse_and_save",
]
