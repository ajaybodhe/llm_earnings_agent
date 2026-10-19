"""Long-horizon drift labels (60 / 120 trading days after the reaction-day
close, minus SPY) from cached Yahoo daily bars — the PEAD.txt paper reports
drift at 63-day steps, beyond the 5/20-day labels in the event dataset."""

from __future__ import annotations

import json
import time
from bisect import bisect_left
from pathlib import Path
from typing import Any

from .context import fetch_daily

HORIZONS = (60, 120)


def cached_bars(symbol: str, cache: Path) -> dict[str, tuple[float, float]]:
    cache.mkdir(parents=True, exist_ok=True)
    p = cache / f"{symbol.replace('/', '-')}.json"
    if p.exists():
        return {k: tuple(v) for k, v in json.loads(p.read_text()).items()}
    try:
        bars = fetch_daily(symbol)
    except Exception:  # missing history just leaves labels empty
        bars = {}
    p.write_text(json.dumps(bars))
    time.sleep(0.2)
    return bars


def add_long_drift(events: list[dict[str, Any]], cache: Path) -> None:
    """Set labels['drift60_mkt'] / ['drift120_mkt'] in place."""
    spy = cached_bars("SPY", cache)
    by_sym: dict[str, list[dict[str, Any]]] = {}
    for e in events:
        by_sym.setdefault(e["symbol"], []).append(e)
    for sym, evs in by_sym.items():
        bars = cached_bars(sym, cache)
        dates = sorted(bars)
        for e in evs:
            day = e["features"].get("reaction_day")
            if not day or day not in bars:
                continue
            i = bisect_left(dates, day)
            for n in HORIZONS:
                if i + n >= len(dates):
                    continue
                end = dates[i + n]
                if day not in spy or end not in spy:
                    continue
                stock = bars[end][1] / bars[day][1] - 1
                mkt = spy[end][1] / spy[day][1] - 1
                e["labels"][f"drift{n}_mkt"] = (stock - mkt) * 100
