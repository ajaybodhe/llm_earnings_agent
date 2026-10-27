"""Web backup for Signal 1: a just-released report's EPS surprise, results and
guidance, read from the web by an LLM with web search.

Signal 1 is calibrated on the Nasdaq calendar's % Surprise, which is Zacks
data, so the prompt asks for the Zacks consensus first (Zacks' "X Tops / Lags
Qn Earnings" notes are syndicated on Yahoo Finance and Nasdaq.com). The surprise
used downstream is always recomputed from the actual and consensus EPS, never
taken from the model's arithmetic. A result is usable for Signal 1 only when
it names a source URL, matches the fiscal quarter asked for, and has both EPS
numbers on the same basis.

Guidance vs consensus is returned for context on the card; it is not an input
to either signal (untested historically).
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

MAX_ABS_SURPRISE = 200.0


class Range(BaseModel):
    low: float | None = None
    high: float | None = None


class Guidance(BaseModel):
    period: str = Field(description="Guided period, e.g. 'Q4 2026' or 'FY 2026'")
    eps: Range | None = Field(None, description="Guided EPS range on the company's adjusted basis")
    revenue_b: Range | None = Field(None, description="Guided revenue range, billions")
    prior_eps: Range | None = Field(None, description="Previous EPS guidance for this period")
    prior_revenue_b: Range | None = None
    consensus_eps: float | None = Field(None, description="Consensus EPS for this period")
    consensus_revenue_b: float | None = None


class WebEarnings(BaseModel):
    symbol: str
    fiscal_quarter: str = Field(description="Fiscal quarter reported, e.g. 'Q3 2026 (Sep/2026)'")
    matches_quarter: bool = Field(description="True if this is the quarter that was asked for")
    report_date: str = Field(description="Date the results were released, YYYY-MM-DD")
    eps_actual: float | None = Field(None, description="Reported adjusted EPS (as Zacks counts it)")
    eps_consensus: float | None = Field(None, description="Consensus EPS before the report")
    consensus_source: str = Field("", description="'Zacks', 'LSEG', 'FactSet', … or ''")
    revenue_actual_b: float | None = None
    revenue_consensus_b: float | None = None
    guidance: list[Guidance] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list, description="URLs the numbers came from")
    notes: str = Field("", description="Caveats, e.g. basis differences; one or two sentences")

    def surprise_pct(self) -> float | None:
        """EPS surprise in %, recomputed: (actual − consensus) ÷ |consensus|."""
        a, c = self.eps_actual, self.eps_consensus
        if a is None or c is None or c == 0:
            return None
        return round((a - c) / abs(c) * 100, 2)

    def usable(self) -> tuple[bool, str]:
        """Whether the surprise may stand in for Nasdaq's in Signal 1."""
        if not self.sources:
            return False, "no source URL"
        if not self.matches_quarter:
            return False, f"found {self.fiscal_quarter}, not the quarter asked for"
        s = self.surprise_pct()
        if s is None:
            return False, "actual or consensus EPS missing"
        if abs(s) > MAX_ABS_SURPRISE:
            return False, f"implausible surprise {s:+.0f}%"
        return True, ""


SYSTEM = (
    "You are a meticulous equity research assistant. Use web search to find facts; "
    "never estimate or recall numbers from memory. Copy figures exactly as published."
)


def prompt(symbol: str, company: str, report_date: str, fiscal_quarter: str) -> str:
    return f"""{company} ({symbol}) released quarterly results on or about {report_date} \
for the fiscal quarter ending {fiscal_quarter or "(most recent)"}.

Find, with web search:
1. Reported adjusted (non-GAAP) EPS and the consensus EPS before the report. Prefer the
   Zacks Consensus Estimate (Zacks "{company} ({symbol}) Tops/Lags Q… Earnings" notes, often
   on finance.yahoo.com or nasdaq.com) and say which consensus you used. Actual and consensus
   must be on the same basis.
2. Reported revenue and consensus revenue (billions).
3. Company guidance issued with these results — next quarter and full year: EPS and revenue
   ranges, the previous guidance for the same periods, and the consensus for those periods.
4. The URLs you took numbers from (the company's press release or SEC 8-K exhibit for
   guidance is best).

If a number is not published, use null. Set matches_quarter=false if you can only find a
different quarter's results."""


def cache_path(cache_dir: Path, symbol: str, report_date: str) -> Path:
    return cache_dir / f"{symbol.upper()}_{report_date}.json"


async def fetch(
    runtime: Any,
    symbol: str,
    company: str,
    report_date: str,
    fiscal_quarter: str,
    *,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> WebEarnings | None:
    """One report's web summary; cached per (symbol, report date) once usable."""
    path = cache_path(cache_dir, symbol, report_date) if cache_dir else None
    if path and path.exists() and not refresh:
        return WebEarnings.model_validate_json(path.read_text(encoding="utf-8"))
    try:
        res = await runtime.complete(
            system=SYSTEM,
            user=prompt(symbol, company, report_date, fiscal_quarter),
            schema=WebEarnings,
            web_search=True,
        )
    except Exception as e:  # a failed backup must not break the signals
        logger.warning("web earnings for %s failed: %s", symbol, e)
        return None
    w = res.value
    w.symbol = symbol.upper()
    if path and w.usable()[0]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(w.model_dump_json(indent=2), encoding="utf-8")
    return w


def needs_backup(r: dict[str, Any], today: str) -> bool:
    """A released report (earnings date already past, or today before the
    open) that the Nasdaq calendar has no surprise for yet."""
    f = r.get("drift_features") or {}
    if f.get("eps_surprise_pct", r.get("eps_surprise_pct")) is not None:
        return False
    d, t = r.get("earnings_date") or "", r.get("earnings_time") or ""
    return bool(d) and (d < today or (d == today and t == "bmo"))


def fetch_all(
    runtime: Any,
    results: list[dict[str, Any]],
    today: str,
    *,
    cache_dir: Path | None,
    concurrency: int = 3,
) -> dict[str, WebEarnings]:
    """Web summaries for every result needing a backup, keyed by symbol."""
    todo = [r for r in results if needs_backup(r, today)]
    if not todo:
        return {}
    sem = asyncio.Semaphore(concurrency)

    async def one(r: dict[str, Any]) -> tuple[str, WebEarnings | None]:
        async with sem:
            w = await fetch(
                runtime,
                r["symbol"],
                r.get("company_name") or r["symbol"],
                r["earnings_date"],
                r.get("fiscal_quarter") or "",
                cache_dir=cache_dir,
            )
            return r["symbol"].upper(), w

    async def run() -> list[tuple[str, WebEarnings | None]]:
        return await asyncio.gather(*(one(r) for r in todo))

    return {s: w for s, w in asyncio.run(run()) if w is not None}


def summary(w: WebEarnings) -> dict[str, Any]:
    """JSON block attached to the post-release output."""
    ok, why = w.usable()
    d = json.loads(w.model_dump_json())
    d["surprise_pct"] = w.surprise_pct()
    d["usable"] = ok
    d["rejected_because"] = why
    return d
