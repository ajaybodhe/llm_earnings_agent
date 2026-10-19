"""Post-release context features derived from the event dataset and two
daily market series (SPY, ^VIX) — all known at the reaction-day open.

* gap_vs_typical: opening gap ÷ the stock's mean |reaction| over its earlier
  reports (strictly earlier events only) — is the open an overreaction?
* surprise_gap_agree: sign(consensus surprise, else SUE) × sign(gap); and
  gap_x_surprise: gap × that surprise sign — the interaction behind the
  open→close effect (gaps that disagree with the surprise fade).
* spy_gap_pct: SPY's overnight gap on the reaction day; vix_prev: VIX close
  the day before.
* intraday_mkt_pct (target): stock open→close minus SPY open→close that day.
"""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from typing import Any
from zoneinfo import ZoneInfo

import httpx

Bars = dict[str, tuple[float, float]]  # date → (open, close)


def fetch_daily(symbol: str, rng: str = "10y", client: httpx.Client | None = None) -> Bars:
    """Daily (open, close) keyed by the exchange-local trading date."""
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range={rng}"
    own = client is None
    c = client or httpx.Client(timeout=30.0, headers={"User-Agent": "Mozilla/5.0"})
    try:
        r = c.get(url)
        r.raise_for_status()
        res = r.json()["chart"]["result"][0]
    finally:
        if own:
            c.close()
    tz = ZoneInfo(res.get("meta", {}).get("exchangeTimezoneName") or "America/New_York")
    q = res["indicators"]["quote"][0]
    out: Bars = {}
    for ts, o, c_ in zip(res["timestamp"], q["open"], q["close"], strict=False):
        if o and c_:
            out[dt.datetime.fromtimestamp(ts, tz).date().isoformat()] = (float(o), float(c_))
    return out


def _prev_date(bars: Bars, date: str) -> str | None:
    earlier = [d for d in bars if d < date]
    return max(earlier) if earlier else None


def add_context(events: list[dict[str, Any]], spy: Bars, vix: Bars) -> None:
    """Add context features in place to each event's `features` dict."""
    spy_dates = sorted(spy)
    vix_dates = sorted(vix)

    def prev(dates: list[str], date: str) -> str | None:
        import bisect

        i = bisect.bisect_left(dates, date)
        return dates[i - 1] if i > 0 else None

    history: dict[str, list[float]] = defaultdict(list)
    for e in sorted(events, key=lambda e: e["features"]["announcement_date"]):
        f = e["features"]
        gap, rx, rday = f.get("gap_ret_pct"), f.get("reaction_ret_pct"), f.get("reaction_day")
        past = history[e["symbol"]]
        if gap is not None and len(past) >= 2:
            typical = sum(past) / len(past)
            if typical > 0:
                f["gap_vs_typical"] = gap / typical
        surprise = f.get("eps_surprise_pct")
        sign_src = surprise if surprise is not None else f.get("sue")
        if gap is not None and sign_src is not None and sign_src != 0:
            s = 1.0 if sign_src > 0 else -1.0
            f["surprise_gap_agree"] = s * (1.0 if gap > 0 else -1.0 if gap < 0 else 0.0)
            f["gap_x_surprise"] = gap * s
        if rday:
            p = prev(spy_dates, rday)
            if rday in spy and p:
                o, c = spy[rday]
                f["spy_gap_pct"] = (o / spy[p][1] - 1) * 100
                f["spy_oc_pct"] = (c / o - 1) * 100
            pv = prev(vix_dates, rday)
            if pv:
                f["vix_prev"] = vix[pv][1]
        if rx is not None:
            past.append(abs(rx))  # only after this event's features are set
