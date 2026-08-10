#!/usr/bin/env python3
"""
Manual reset event recording tool.

Usage:
    python add_reset.py "2026-08-08T03:30:00Z" "Global Codex quota reset"
    python add_reset.py "2026-08-09T15:00:00Z" "Tibo manual reset" --source twitter

This tool adds a reset event to data/reset_history.json. Use it when RSS
misses a reset announcement (e.g., Tibo posted it as a reply tweet that
RSSHub cannot collect).
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
HISTORY_FILE = DATA_DIR / "reset_history.json"


def add_reset(
    reset_time_str: str,
    notes: str,
    source: str = "manual",
    confidence: float = 1.0,
) -> None:
    """Add a reset event to the history file."""
    # Parse the timestamp
    dt = datetime.fromisoformat(reset_time_str.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # Load existing history
    if HISTORY_FILE.exists():
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = []

    # Check for duplicates (same timestamp within 1 hour)
    for event in history:
        existing_time = datetime.fromisoformat(
            event["reset_time"].replace("Z", "+00:00")
        )
        if abs((dt - existing_time).total_seconds()) < 3600:
            print(f"WARNING: A reset event already exists near {dt.isoformat()}")
            print(f"  Existing: {event['reset_time']} - {event['notes']}")
            response = input("  Add anyway? (y/N): ")
            if response.lower() != "y":
                print("Aborted.")
                return

    # Add the new event
    new_event = {
        "reset_time": dt.isoformat(),
        "source": source,
        "confidence": confidence,
        "notes": notes,
    }
    history.append(new_event)

    # Sort by time
    history.sort(key=lambda e: e["reset_time"])

    # Save
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Added reset event: {dt.isoformat()} - {notes}")
    print(f"Total events in history: {len(history)}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python add_reset.py <ISO_TIMESTAMP> <NOTES> [--source SOURCE]")
        print('Example: python add_reset.py "2026-08-08T03:30:00Z" "Global Codex quota reset"')
        sys.exit(1)

    reset_time = sys.argv[1]
    notes = sys.argv[2]
    source = "manual"

    if "--source" in sys.argv:
        idx = sys.argv.index("--source")
        if idx + 1 < len(sys.argv):
            source = sys.argv[idx + 1]

    add_reset(reset_time, notes, source)
