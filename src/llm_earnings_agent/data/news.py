"""Company news headlines via Finnhub free tier.

Endpoint: `https://finnhub.io/api/v1/company-news?symbol=&from=&to=&token=`

`FINNHUB_TOKEN` env var is required.
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass

import httpx

BASE_URL = "https://finnhub.io/api/v1/company-news"


@dataclass(frozen=True)
class NewsItem:
    headline: str
    summary: str
    url: str
    source: str
    datetime_utc: dt.datetime
    category: str = ""


async def fetch_company_news(
    symbol: str,
    *,
    from_date: dt.date,
    to_date: dt.date,
    client: httpx.AsyncClient | None = None,
    timeout_s: float = 30.0,
    token: str | None = None,
) -> list[NewsItem]:
    token = token or os.environ.get("FINNHUB_TOKEN", "") or "d6kt2lhr01qmopd22780d6kt2lhr01qmopd2278g"
    if not token:
        raise RuntimeError("FINNHUB_TOKEN env var is not set")

    params = {
        "symbol": symbol.upper(),
        "from": from_date.isoformat(),
        "to": to_date.isoformat(),
        "token": token,
    }
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=timeout_s)
    try:
        resp = await client.get(BASE_URL, params=params)
    finally:
        if owns_client:
            await client.aclose()

    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, list):
        return []

    items: list[NewsItem] = []
    for it in payload:
        ts = it.get("datetime")
        if not ts:
            continue
        items.append(
            NewsItem(
                headline=str(it.get("headline") or ""),
                summary=str(it.get("summary") or ""),
                url=str(it.get("url") or ""),
                source=str(it.get("source") or ""),
                datetime_utc=dt.datetime.fromtimestamp(int(ts), tz=dt.UTC),
                category=str(it.get("category") or ""),
            )
        )
    return items
