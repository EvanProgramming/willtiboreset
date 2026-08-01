"""Tests for auto_confirm.py reset auto-confirmation."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from auto_confirm import (
    _EXPLICIT_RESET_PHRASES,
    auto_confirm_reset,
    detect_auto_confirm,
)
from model.data_models import Tweet


@pytest.fixture
def temp_history(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Provide temporary history.txt and output JSON paths."""
    history_txt = tmp_path / "history.txt"
    reset_history = tmp_path / "reset_history.json"
    uncertain = tmp_path / "uncertain_events.json"
    return history_txt, reset_history, uncertain


def make_tweet(
    text: str,
    author: str = "thsottiaux",
    source: str = "tibo_rss",
    timestamp: datetime | None = None,
) -> Tweet:
    return Tweet(
        timestamp=timestamp or datetime(2026, 8, 1, 3, 32, tzinfo=timezone.utc),
        author=author,
        text=text,
        source=source,
        url="http://example.com/tweet/1",
        authority_score=1.0,
    )


def test_detect_auto_confirm_matches_explicit_phrases() -> None:
    for phrase in _EXPLICIT_RESET_PHRASES:
        tweets = [make_tweet(f"Great news! {phrase} for Codex and ChatGPT Work.")]
        result = detect_auto_confirm(tweets)
        assert result is not None
        assert phrase in result[1].lower()


def test_detect_auto_confirm_ignores_non_tibo() -> None:
    tweets = [
        make_tweet(
            "I have reset usage limits.",
            author="random_user",
            source="community_rss",
        ),
    ]
    assert detect_auto_confirm(tweets) is None


def test_detect_auto_confirm_ignores_weak_language() -> None:
    tweets = [make_tweet("Maybe we should reset the limits soon.")]
    assert detect_auto_confirm(tweets) is None


def test_auto_confirm_appends_history_and_json(temp_history) -> None:
    history_txt, reset_history, uncertain = temp_history
    tweets = [
        make_tweet(
            "To celebrate a week of efficiency, I have reset usage limits "
            "for Codex and ChatGPT Work. Enjoy!",
        ),
    ]

    event = auto_confirm_reset(
        tweets,
        history_txt_path=history_txt,
        reset_history_path=reset_history,
        uncertain_path=uncertain,
    )

    assert event is not None
    assert event.reset_time == datetime(2026, 8, 1, 3, 32, tzinfo=timezone.utc)
    assert event.source.value == "twitter"

    # history.txt should contain a structured entry
    text = history_txt.read_text(encoding="utf-8")
    assert "Aug 1" in text
    assert "03:32 UTC" in text
    assert "CONFIRMED RESET" in text
    assert "I have reset usage limits" in text
    assert "View source" in text

    # reset_history.json should contain the new event
    history = json.loads(reset_history.read_text(encoding="utf-8"))
    assert len(history) == 1
    assert history[0]["reset_time"] == "2026-08-01T03:32:00+00:00"
    assert history[0]["source"] == "twitter"
    assert history[0]["confidence"] == 1.0


def test_auto_confirm_avoids_duplicates(temp_history) -> None:
    history_txt, reset_history, uncertain = temp_history
    tweets = [
        make_tweet("I have reset usage limits for Codex and ChatGPT Work."),
    ]

    first = auto_confirm_reset(
        tweets,
        history_txt_path=history_txt,
        reset_history_path=reset_history,
        uncertain_path=uncertain,
    )
    assert first is not None

    second = auto_confirm_reset(
        tweets,
        history_txt_path=history_txt,
        reset_history_path=reset_history,
        uncertain_path=uncertain,
    )
    assert second is None


def test_auto_confirm_handles_existing_history_without_delimiter(
    temp_history,
) -> None:
    history_txt, reset_history, uncertain = temp_history
    # Simulate a truncated history file that does not end with a delimiter
    history_txt.write_text("Jul 30\n18:24 UTC", encoding="utf-8")

    tweets = [
        make_tweet("I have reset usage limits for Codex and ChatGPT Work."),
    ]
    event = auto_confirm_reset(
        tweets,
        history_txt_path=history_txt,
        reset_history_path=reset_history,
        uncertain_path=uncertain,
    )

    assert event is not None
    text = history_txt.read_text(encoding="utf-8")
    # The previous block should be closed and the new entry parsed correctly
    assert text.count("View source") >= 1

    history = json.loads(reset_history.read_text(encoding="utf-8"))
    assert len(history) == 1
    assert history[0]["reset_time"] == "2026-08-01T03:32:00+00:00"
