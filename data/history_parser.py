"""
WillTiboReset - Historical reset data parser

Parses natural-language / semi-structured text into standard reset_history.json.

Supported input formats:
    1. Semi-structured logs (user-provided format):
       Each event contains date, time, CONFIRMED RESET marker, Scope, Source, etc.

    2. Plain natural language:
       "On July 10, 2026, Tibo said on X that the limit has reset"

Parsing strategy:
    - Prefer structured markers (CONFIRMED RESET, etc.)
    - Use date and keyword heuristics for natural language
    - Mark uncertain events with confidence and store them in uncertain_events.json
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
# Month mapping
# ──────────────────────────────────────────────

_MONTH_MAP: dict[str, int] = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Structured date pattern: Jun 29, Jul 10
_DATE_PATTERN = re.compile(
    r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})$",
    re.IGNORECASE | re.MULTILINE,
)

# Time pattern: 00:00 UTC, 05:30 UTC
_TIME_PATTERN = re.compile(r"^(\d{1,2}):(\d{2})\s*UTC\s*$", re.IGNORECASE)

# Confirmed reset markers
_CONFIRMED_MARKERS = ["CONFIRMED RESET", "confirmed reset"]

# Non-reset markers (replies, downward signals, etc.)
_NON_RESET_MARKERS = [
    "REPLY", "Downward signal", "Archived signal",
    "UPWARD SIGNAL",
]

# Metadata lines; not included in title / notes
_METADATA_KEYS = {"SCOPE", "SOURCE", "COMPENSATION", "VIEW SOURCE", "VIEW REPLY"}


# ──────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────

@dataclass
class ParsedEvent:
    """Parsed event (not yet classified)"""
    reset_time: Optional[datetime] = None
    title: str = ""
    description: str = ""
    scope: str = ""
    source: str = ""
    is_confirmed_reset: bool = False
    confidence: float = 0.0
    notes: str = ""


# ──────────────────────────────────────────────
# Source mapping
# ──────────────────────────────────────────────

def _map_source(source_text: str) -> str:
    """Map source text to a standard SignalSource value"""
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
# Utility functions
# ──────────────────────────────────────────────

def _is_metadata_line(line_upper: str) -> bool:
    """Check whether the line is a structured metadata key"""
    return line_upper in _METADATA_KEYS


def _is_noise_line(line: str) -> bool:
    """Check whether the line is ignorable social-media stats/rating text"""
    line_upper = line.upper()
    if re.match(r"^(Replies|Reposts|Likes|Views):\s", line, re.IGNORECASE):
        return True
    if re.match(r"^\d+[hH]\s*[+-]\d+", line):
        return True
    if line_upper in ("COMPENSATION", "VIEW SOURCE", "VIEW REPLY"):
        return True
    return False


def _infer_source_from_block(block: list[str]) -> str:
    """Infer source from the entire block"""
    text = "\n".join(block).lower()
    return _map_source(text)


# ──────────────────────────────────────────────
# Structured parser
# ──────────────────────────────────────────────

def _find_date_time_pairs(lines: list[str]) -> list[tuple[int, int, datetime]]:
    """Find all (date line index, time line index, datetime) triples"""
    pairs: list[tuple[int, int, datetime]] = []
    i = 0
    while i < len(lines):
        date_match = _DATE_PATTERN.match(lines[i])
        if date_match and i + 1 < len(lines):
            time_match = _TIME_PATTERN.match(lines[i + 1])
            if time_match:
                # Default year is provided by the caller; use a placeholder here and replace later
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
    """Check whether the line represents substantive event content (not noise, not date/time)"""
    if _is_noise_line(line):
        return False
    if _DATE_PATTERN.match(line) or _TIME_PATTERN.match(line):
        return False
    return True


def _chunk_lines(lines: list[str]) -> list[list[str]]:
    """Split logs into event blocks by 'View source' / 'View reply'"""
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
    Extract date+time from the start of a block, returning (datetime, remaining lines).
    If the start is not date+time, return (None, original lines).
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
    Parse semi-structured log format.

    Each event ends with 'View source' / 'View reply'.
    Most blocks start with date+time followed by metadata; the first block has no date,
    so use the next block's date+time.
    """
    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    chunks = _chunk_lines(raw_lines)

    events: list[ParsedEvent] = []

    for idx, chunk in enumerate(chunks):
        # Try to extract date+time from the start of the current block
        dt, metadata_lines = _extract_date_time_pair(chunk, default_year)

        if dt is None:
            # Current block has no date: use the date+time from the next block's start
            if idx + 1 >= len(chunks):
                continue
            next_dt, _ = _extract_date_time_pair(chunks[idx + 1], default_year)
            if next_dt is None:
                continue
            dt = next_dt
            # Treat the entire current block as metadata
            metadata_lines = chunk
        else:
            # Current block starts with date+time: metadata is the remaining part
            # Skip if the remaining part is empty (only a date at the end of the file)
            if not metadata_lines:
                continue

        # Remove noise lines
        clean = [line for line in metadata_lines if not _is_noise_line(line)]
        if not clean:
            continue

        # Determine whether this is a confirmed reset
        text_upper = "\n".join(line.upper() for line in clean)
        is_confirmed = any(marker.upper() in text_upper for marker in _CONFIRMED_MARKERS)

        # Extract title
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

        # Extract Scope / Source
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
# Natural-language parser (simplified)
# ──────────────────────────────────────────────

def _parse_natural_language(text: str, default_year: int = 2026) -> list[ParsedEvent]:
    """
    Parse reset events from natural-language text.

    Example: "On July 10, 2026, Tibo said on X that the limit has reset"
    """
    events: list[ParsedEvent] = []

    # English date patterns
    date_patterns = [
        # "July 10, 2026" or "Jul 10, 2026"
        re.compile(
            r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),?\s+(\d{4})",
            re.IGNORECASE,
        ),
        # "2026-07-10" or "2026/07/10"
        re.compile(r"(\d{4})[-/](\d{1,2})[-/](\d{1,2})"),
    ]

    # Split by line or sentence
    sentences = re.split(r"[.!?]\s+|\n", text)

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Extract date
        year: Optional[int] = None
        month: Optional[int] = None
        day: Optional[int] = None
        for pattern in date_patterns:
            match = pattern.search(sentence)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    if groups[0].isdigit():
                        # ISO format: year, month, day
                        year, month, day = int(groups[0]), int(groups[1]), int(groups[2])
                    else:
                        # Month name format: month_name, day, year
                        month = _MONTH_MAP[groups[0].lower()[:3]]
                        day = int(groups[1])
                        year = int(groups[2])
                break

        if year is None or month is None or day is None:
            continue

        # Try to extract time (if any)
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

        # Determine whether this is a reset event
        reset_keywords = ["reset", "limit reset", "usage reset", "quota reset"]
        is_reset = any(kw in sentence.lower() for kw in reset_keywords)

        # Determine source
        if "tibo" in sentence.lower():
            source = "twitter"
        elif "openai" in sentence.lower():
            source = "openai_status"
        elif "reddit" in sentence.lower():
            source = "reddit"
        else:
            source = "other"

        # Confidence
        if is_reset:
            confirm_keywords = ["confirmed", "announced", "said"]
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
# Main interface
# ──────────────────────────────────────────────

@dataclass
class ParseResult:
    """Parse result"""
    confirmed: list[dict] = field(default_factory=list)
    uncertain: list[dict] = field(default_factory=list)

    def to_json(self) -> dict:
        """Return a JSON-serializable structure"""
        return {
            "confirmed": self.confirmed,
            "uncertain": self.uncertain,
        }


def parse_history_text(text: str, default_year: int = 2026) -> ParseResult:
    """
    Parse historical reset text.

    Automatically detects input format (structured log or natural language)
    and classifies events into confirmed / uncertain lists.

    Args:
        text: Raw input text
        default_year: Default year for structured dates

    Returns:
        ParseResult containing confirmed and uncertain event lists
    """
    # Try structured parsing first
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
    Save parse results to JSON files.

    Args:
        result: Parse result
        history_path: Path to reset_history.json
        uncertain_path: Path to uncertain_events.json
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
    One-step parse text and save to files.

    Args:
        text: Raw input text
        history_path: Path to reset_history.json
        uncertain_path: Path to uncertain_events.json (defaults to same directory)
        default_year: Default year for structured dates

    Returns:
        ParseResult
    """
    if uncertain_path is None:
        uncertain_path = history_path.parent / "uncertain_events.json"

    result = parse_history_text(text, default_year)
    save_parsed_history(result, history_path, uncertain_path)
    return result


def main() -> int:
    """CLI entry point: parse history.txt and write to reset_history.json"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse natural-language/semi-structured history into reset_history.json",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("history.txt"),
        help="Input history file path (default: history.txt)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/reset_history.json"),
        help="Output reset_history.json path (default: data/reset_history.json)",
    )
    parser.add_argument(
        "--uncertain",
        type=Path,
        default=Path("data/uncertain_events.json"),
        help="Output uncertain_events.json path (default: data/uncertain_events.json)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2026,
        help="Default year (default: 2026)",
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: input file does not exist {args.input}", file=sys.stderr)
        return 1

    text = args.input.read_text(encoding="utf-8")
    result = parse_and_save(
        text,
        history_path=args.output,
        uncertain_path=args.uncertain,
        default_year=args.year,
    )

    print(f"Parsed {args.input}")
    print(f"  Confirmed events: {len(result.confirmed)} -> {args.output}")
    print(f"  Uncertain events: {len(result.uncertain)} -> {args.uncertain}")
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
