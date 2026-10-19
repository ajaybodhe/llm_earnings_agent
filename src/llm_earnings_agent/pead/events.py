"""Load the quarterly_results event dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def event_id(e: dict[str, Any]) -> str:
    """Stable id: SYMBOL|announcement date (as written by the Go builder)."""
    if e.get("event_id"):
        return str(e["event_id"])
    return (
        f"{str(e.get('symbol', '')).upper()}|{e.get('features', {}).get('announcement_date', '')}"
    )


def load_events(path: str | Path) -> list[dict[str, Any]]:
    """Read events JSONL. When an announcement was exported more than once,
    the most recent export wins (its labels are the most complete)."""
    latest: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{n}: {exc}") from exc
            key = event_id(e)
            prev = latest.get(key)
            if prev is None or str(e.get("exported_at", "")) >= str(prev.get("exported_at", "")):
                latest[key] = e
    return sorted(latest.values(), key=lambda e: (e["features"]["announcement_date"], e["symbol"]))
