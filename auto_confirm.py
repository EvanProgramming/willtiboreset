"""
WillTiboReset - Auto-confirmation of reset events from Tibo's announcements.

When Tibo explicitly announces on X that usage limits have been reset,
this module automatically:
  1. Records the reset in history.txt and reset_history.json
  2. Lets the model treat the reset as "just occurred", which naturally
     lowers short-term probabilities (a reset right after a reset is rare)
  3. Enables calibration to resolve recent predictions against the
     newly confirmed reset event
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import config
from data.history_parser import parse_and_save
from model.data_models import ResetEvent, Tweet


# Phrases that constitute an explicit reset announcement from Tibo
# (simple substring matching, case-insensitive)
_EXPLICIT_RESET_PHRASES = [
    "i have reset",
    "we have reset",
    "usage limits have been reset",
    "reset usage limits",
    "limits have been reset",
    "codex reset",
    "usage reset",
    "reset for all",
    "resetting",
    "enjoy reset",
    "we reset",
    "another reset",
]

# Regex patterns for reset announcements that may be separated by
# punctuation/whitespace or phrased slightly differently.
_RESET_PATTERNS = [
    re.compile(r"reset\s+(usage\s+)?limits?\b", re.IGNORECASE),
    re.compile(r"(usage\s+)?limits?\s+(have\s+been\s+)?reset\b", re.IGNORECASE),
    re.compile(r"\breset\b.*\b(now|today|tonight|enjoy|again|done)\b", re.IGNORECASE),
]

# Proximity window to avoid duplicate reset records (seconds)
_DUPLICATE_WINDOW_SECONDS = 3600


def _is_tibo_tweet(tweet: Tweet) -> bool:
    """Check whether a tweet originates from Tibo."""
    return (
        "tibo" in tweet.author.lower()
        or "thsottiaux" in tweet.author.lower()
        or "tibo" in tweet.source.lower()
    )


def _text_matches_reset(text: str) -> bool:
    """Check whether a text explicitly announces a reset (phrases or regex)."""
    text_lower = text.lower()
    # 先排除否定语境："no reset"、"never reset"、"won't reset" 等不是重置声明
    if re.search(r"\bno\s+reset\b", text_lower) or re.search(
        r"\b(never|won'?t|not)\s+(reset|be\s+reset)\b", text_lower
    ):
        return False
    if any(phrase in text_lower for phrase in _EXPLICIT_RESET_PHRASES):
        return True
    for pattern in _RESET_PATTERNS:
        if pattern.search(text_lower):
            return True
    return False


def detect_auto_confirm(tweets: list[Tweet]) -> Optional[tuple[datetime, str]]:
    """
    Detect an explicit reset confirmation from Tibo's tweets.

    Returns:
        (reset_time, text) if a confirmation is found, otherwise None.
    """
    for tweet in tweets:
        if not _is_tibo_tweet(tweet):
            continue
        if _text_matches_reset(tweet.text):
            return tweet.timestamp, tweet.text
    return None


def _format_structured_history_entry(reset_time: datetime, text: str) -> str:
    """Format a structured history.txt entry matching the existing log style."""
    month_abbr = reset_time.strftime("%b")
    day = reset_time.day
    hour = reset_time.strftime("%H")
    minute = reset_time.strftime("%M")
    cleaned_text = re.sub(r"\s+", " ", text).strip()
    title = cleaned_text[:200]

    lines = [
        "",
        f"{month_abbr} {day}",
        f"{hour}:{minute} UTC",
        "CONFIRMED RESET",
        "Global Codex quota reset",
        title,
        "Scope",
        "All paid users of Codex and ChatGPT Work",
        "Source",
        "Tibo on X",
        "View source",
    ]
    return "\n".join(lines)


def _already_confirmed(reset_time: datetime, history_path: Path) -> bool:
    """Check whether a reset near the given time is already recorded."""
    if not history_path.exists():
        return False
    try:
        history = json.loads(history_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    for entry in history:
        try:
            entry_time = datetime.fromisoformat(entry["reset_time"])
        except Exception:
            continue
        if abs((entry_time - reset_time).total_seconds()) < _DUPLICATE_WINDOW_SECONDS:
            return True
    return False


def auto_confirm_reset(
    tweets: list[Tweet],
    history_txt_path: Path = Path("history.txt"),
    reset_history_path: Optional[Path] = None,
    uncertain_path: Optional[Path] = None,
) -> Optional[ResetEvent]:
    """
    Auto-confirm a reset when Tibo explicitly announces it.

    Updates history.txt and re-parses it into reset_history.json.
    Returns the confirmed ResetEvent, or None if no confirmation is detected
    or the reset is already recorded.
    """
    detection = detect_auto_confirm(tweets)
    if detection is None:
        return None

    reset_time, text = detection

    if reset_history_path is None:
        reset_history_path = config.reset_history_path
    if uncertain_path is None:
        uncertain_path = reset_history_path.parent / "uncertain_events.json"

    if _already_confirmed(reset_time, reset_history_path):
        return None

    entry_text = _format_structured_history_entry(reset_time, text)
    history_txt_path.parent.mkdir(parents=True, exist_ok=True)

    with history_txt_path.open("a", encoding="utf-8") as f:
        # Ensure the previous block is closed if the file was truncated or
        # the last entry did not end with a standard delimiter.
        if history_txt_path.exists() and history_txt_path.stat().st_size > 0:
            current_text = history_txt_path.read_text(encoding="utf-8")
            stripped = current_text.rstrip()
            if stripped and not stripped.lower().endswith(("view source", "view reply")):
                f.write("\nView source\n")
            elif not current_text.endswith("\n"):
                f.write("\n")
        f.write(entry_text + "\n")

    full_text = history_txt_path.read_text(encoding="utf-8")
    parse_and_save(
        full_text,
        history_path=reset_history_path,
        uncertain_path=uncertain_path,
        default_year=reset_time.year,
    )

    return ResetEvent(
        reset_time=reset_time,
        source="twitter",
        confidence=1.0,
        notes=text[:300],
    )
