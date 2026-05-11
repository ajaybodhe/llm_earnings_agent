"""Earnings call transcripts.

Primary source: Financial Modeling Prep (FMP).
    https://financialmodelingprep.com/api/v3/earning_call_transcript/{SYMBOL}?quarter=&year=&apikey=
    Requires `FMP_API_KEY` env var.

Fallback: API Ninjas.
    https://api.api-ninjas.com/v1/earningstranscript?ticker=&year=&quarter=
    Requires `API_NINJAS_KEY` env var (header `X-Api-Key`).

Both APIs accept (quarter, year). When the latest transcript is unknown, walk
backward from the current quarter up to `max_lookback_quarters`.
"""

from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass

import httpx

FMP_BASE_URL = "https://financialmodelingprep.com/api/v3/earning_call_transcript"
NINJAS_BASE_URL = "https://api.api-ninjas.com/v1/earningstranscript"


@dataclass(frozen=True)
class Transcript:
    symbol: str
    quarter: int  # 1..4
    year: int
    content: str

    @property
    def label(self) -> str:
        return f"Q{self.quarter} {self.year}"


async def _fmp_fetch(
    symbol: str, quarter: int, year: int, *, client: httpx.AsyncClient, token: str
) -> Transcript | None:
    url = f"{FMP_BASE_URL}/{symbol.upper()}"
    params = {"quarter": str(quarter), "year": str(year), "apikey": token}
    resp = await client.get(url, params=params)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, list) or not payload:
        return None
    entry = payload[0]
    content = entry.get("content") or ""
    if not content:
        return None
    return Transcript(
        symbol=symbol.upper(),
        quarter=int(entry.get("quarter") or quarter),
        year=int(entry.get("year") or year),
        content=content,
    )


async def _ninjas_fetch(
    symbol: str, quarter: int, year: int, *, client: httpx.AsyncClient, token: str
) -> Transcript | None:
    params = {"ticker": symbol.upper(), "year": str(year), "quarter": str(quarter)}
    resp = await client.get(NINJAS_BASE_URL, params=params, headers={"X-Api-Key": token})
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    payload = resp.json()
    # API Ninjas returns {} when no transcript is available, otherwise
    # {date: ..., transcript: ...}.
    if not isinstance(payload, dict):
        return None
    content = payload.get("transcript") or ""
    if not content:
        return None
    return Transcript(symbol=symbol.upper(), quarter=quarter, year=year, content=content)


async def fetch_transcript(
    symbol: str,
    quarter: int,
    year: int,
    *,
    client: httpx.AsyncClient | None = None,
    timeout_s: float = 30.0,
) -> Transcript | None:
    """Fetch a single quarter's transcript. Returns None if not available.

    Tries FMP first; falls back to API Ninjas only if FMP returns no content
    (or its key is missing). Network errors from FMP are also softened to a
    fallback attempt so a bad day on one provider doesn't kill the pipeline.
    """
    fmp_token = os.environ.get("FMP_API_KEY", "")
    ninjas_token = os.environ.get("API_NINJAS_KEY", "")
    if not fmp_token and not ninjas_token:
        raise RuntimeError("FMP_API_KEY (preferred) or API_NINJAS_KEY env var must be set")

    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=timeout_s)
    try:
        if fmp_token:
            try:
                t = await _fmp_fetch(symbol, quarter, year, client=client, token=fmp_token)
                if t is not None:
                    return t
            except httpx.HTTPError:
                if not ninjas_token:
                    raise
        if ninjas_token:
            return await _ninjas_fetch(symbol, quarter, year, client=client, token=ninjas_token)
        return None
    finally:
        if owns_client:
            await client.aclose()


async def fetch_latest_transcript(
    symbol: str,
    *,
    client: httpx.AsyncClient | None = None,
    max_lookback_quarters: int = 6,
) -> Transcript | None:
    """Walk back from the most recent quarter until we find a transcript."""
    now = _dt.date.today()
    q = (now.month - 1) // 3 + 1
    y = now.year

    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=30.0)
    try:
        for _ in range(max_lookback_quarters):
            t = await fetch_transcript(symbol, q, y, client=client)
            if t is not None:
                return t
            q -= 1
            if q == 0:
                q = 4
                y -= 1
        return None
    finally:
        if owns_client:
            await client.aclose()
