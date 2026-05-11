"""Daily price history via Yahoo Finance v8 chart endpoint.

Used by the backtest harness to score predicted vs actual one-day returns
around earnings announcements. Mirrors `yahoo_price.go` in `quarterly_results`.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import httpx

CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart"


@dataclass(frozen=True)
class PricePoint:
    date: dt.date
    open: float
    close: float


async def fetch_price_history(
    symbol: str,
    *,
    range: str = "2y",
    client: httpx.AsyncClient | None = None,
    timeout_s: float = 30.0,
) -> list[PricePoint]:
    owns_client = client is None
    client = client or httpx.AsyncClient(
        timeout=timeout_s,
        headers={"User-Agent": "llm-earnings-agent/0.1"},
    )
    try:
        resp = await client.get(
            f"{CHART_URL}/{symbol.upper()}",
            params={"interval": "1d", "range": range},
        )
    finally:
        if owns_client:
            await client.aclose()

    resp.raise_for_status()
    payload = resp.json()
    chart = payload.get("chart", {})
    results = chart.get("result") or []
    if not results:
        return []

    r0 = results[0]
    timestamps = r0.get("timestamp") or []
    quote = (r0.get("indicators", {}).get("quote") or [{}])[0]
    opens = quote.get("open") or []
    closes = quote.get("close") or []

    points: list[PricePoint] = []
    for ts, op, cl in zip(timestamps, opens, closes, strict=False):
        if op is None or cl is None:
            continue
        points.append(
            PricePoint(
                date=dt.datetime.fromtimestamp(int(ts), tz=dt.UTC).date(),
                open=float(op),
                close=float(cl),
            )
        )
    return points


def one_day_return_pct(prices: list[PricePoint], announcement: dt.date) -> float | None:
    """Return the simple percent move from close-before to close-on/after.

    Mirrors how `quarterly_results` measures earnings reactions. Returns None
    if we lack data on either side.
    """
    if not prices:
        return None
    before: PricePoint | None = None
    after: PricePoint | None = None
    for p in prices:
        if p.date < announcement:
            before = p
        elif p.date >= announcement:
            after = p
            break
    if before is None or after is None:
        return None
    if before.close == 0:
        return None
    return (after.close - before.close) / before.close * 100.0
