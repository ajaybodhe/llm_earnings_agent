"""Material events via SEC EDGAR 8-K filings.

The SEC requires a User-Agent with contact info per their fair use policy:
https://www.sec.gov/os/accessing-edgar-data
"""

from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass

import httpx

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


def _user_agent() -> str:
    return os.environ.get("SEC_USER_AGENT", "llm-earnings-agent ajaybodhe@gmail.com")


@dataclass(frozen=True)
class MaterialEvent:
    filed_date: dt.date
    form: str
    items: str
    accession: str
    primary_doc: str


async def _resolve_cik(symbol: str, client: httpx.AsyncClient) -> str | None:
    resp = await client.get(TICKER_MAP_URL, headers={"User-Agent": _user_agent()})
    resp.raise_for_status()
    payload = resp.json()
    target = symbol.upper()
    for entry in payload.values():
        if str(entry.get("ticker", "")).upper() == target:
            return f"{int(entry['cik_str']):010d}"
    return None


async def fetch_material_events(
    symbol: str,
    *,
    since: dt.date,
    client: httpx.AsyncClient | None = None,
    timeout_s: float = 30.0,
) -> list[MaterialEvent]:
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=timeout_s)
    try:
        cik = await _resolve_cik(symbol, client)
        if not cik:
            return []
        url = SUBMISSIONS_URL.format(cik=cik)
        resp = await client.get(url, headers={"User-Agent": _user_agent()})
        resp.raise_for_status()
        payload = resp.json()
    finally:
        if owns_client:
            await client.aclose()

    recent = payload.get("filings", {}).get("recent", {})
    forms = recent.get("form") or []
    filed = recent.get("filingDate") or []
    items_col = recent.get("items") or []
    accs = recent.get("accessionNumber") or []
    docs = recent.get("primaryDocument") or []

    out: list[MaterialEvent] = []
    for form, fdate, items, acc, doc in zip(forms, filed, items_col, accs, docs, strict=False):
        if not form or not form.startswith("8-K"):
            continue
        try:
            d = dt.date.fromisoformat(fdate)
        except ValueError:
            continue
        if d < since:
            continue
        out.append(
            MaterialEvent(
                filed_date=d,
                form=form,
                items=items or "",
                accession=acc or "",
                primary_doc=doc or "",
            )
        )
    return out
