"""Walk-forward backtest harness.

Reconstructs point-in-time inputs for each prior earnings announcement and runs
the agents. Compares the predicted label to the actual signed return (±1% dead-
band, same convention as `quarterly_results/backtest.go`). Aggregates a hit
rate and compares it to the always-guess-most-common-label baseline.

Caveats baked into the design:

- Transcripts: we fetch by (quarter, year), so the inputs are correctly
  point-in-time. ✓
- News: Finnhub takes `from`/`to` dates, so we can clip to strictly before the
  announcement. ✓
- Fundamentals: we call `quarterly_results` at runtime, which produces
  *current* fundamentals — there is no built-in "as-of" mode. So this run uses
  today's fundamentals applied to a past announcement, which is a real
  lookahead inside this signal. Treat with caution; we still emit the result
  but flag it as `lookahead=True` in the metadata.
- LLM training cutoff is its own lookahead source we can't remove.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any

import httpx

from .agents import aggregate, analyze_news, analyze_transcript
from .cache import JsonCache
from .data.events import fetch_material_events
from .data.news import fetch_company_news
from .data.prices import fetch_price_history, one_day_return_pct
from .data.transcripts import fetch_transcript
from .runtime.base import Runtime
from .schemas import (
    BacktestPoint,
    BacktestSummary,
    Label,
    NewsAnalysis,
    Rating,
    TranscriptAnalysis,
)

logger = logging.getLogger(__name__)

DEADBAND_PCT = 1.0


def _label_from_return(ret_pct: float) -> Label:
    if ret_pct > DEADBAND_PCT:
        return "Positive"
    if ret_pct < -DEADBAND_PCT:
        return "Negative"
    return "Neutral"


def _baseline_hit_rate(actuals: list[Label]) -> float:
    if not actuals:
        return 0.0
    counts: dict[Label, int] = {"Positive": 0, "Negative": 0, "Neutral": 0}
    for a in actuals:
        counts[a] += 1
    most_common = max(counts.values())
    return most_common / len(actuals)


async def _historical_announcements(symbol: str, quarters: int) -> list[tuple[int, int, dt.date]]:
    """Return `[(quarter, year, approx_announcement_date)]` for the most recent N quarters.

    We approximate the announcement date as 30 days after quarter end. The
    actual date is refined later by checking which trading day Yahoo has prices
    around that window.
    """
    now = dt.date.today()
    q = (now.month - 1) // 3 + 1
    y = now.year
    # Step back one quarter — the *current* quarter likely hasn't reported yet.
    q -= 1
    if q == 0:
        q = 4
        y -= 1

    out: list[tuple[int, int, dt.date]] = []
    for _ in range(quarters):
        # quarter end = last day of quarter
        end_month = q * 3
        end_day = 30 if end_month in (4, 6, 9, 11) else 31
        if end_month == 2:
            end_day = 28
        q_end = dt.date(y, end_month, end_day)
        approx_announce = q_end + dt.timedelta(days=30)
        out.append((q, y, approx_announce))
        q -= 1
        if q == 0:
            q = 4
            y -= 1
    return out


async def run_backtest(
    symbol: str,
    *,
    quarters: int = 4,
    runtime: Runtime,
    model: str | None = None,
    cache: JsonCache | None = None,
    news_lookback_days: int = 30,
) -> BacktestSummary:
    cache = cache or JsonCache()
    announcements = await _historical_announcements(symbol, quarters)

    async with httpx.AsyncClient(timeout=30.0) as http:
        prices = await fetch_price_history(symbol, range="5y", client=http)

        points: list[BacktestPoint] = []
        for q, y, approx_date in announcements:
            transcript = await fetch_transcript(symbol, q, y, client=http)
            if transcript is None:
                logger.info("no transcript for %s Q%d %d, skipping", symbol, q, y)
                continue

            news_from = approx_date - dt.timedelta(days=news_lookback_days)
            try:
                headlines = await fetch_company_news(symbol, from_date=news_from, to_date=approx_date, client=http)
            except Exception as e:  # noqa: BLE001
                logger.warning("news fetch failed for %s Q%d %d: %s", symbol, q, y, e)
                headlines = []
            try:
                events = await fetch_material_events(symbol, since=news_from, client=http)
            except Exception as e:  # noqa: BLE001
                logger.warning("events fetch failed for %s Q%d %d: %s", symbol, q, y, e)
                events = []
            # Strict prefix: keep only items dated before the approximate announce date.
            events = [e for e in events if e.filed_date < approx_date]

            tx_result: TranscriptAnalysis | None = None
            try:
                r = await analyze_transcript(
                    symbol=symbol, transcript=transcript,
                    runtime=runtime, model=model, cache=cache,
                )
                tx_result = r.value
            except Exception as e:  # noqa: BLE001
                logger.warning("transcript agent failed for %s Q%d %d: %s", symbol, q, y, e)

            news_result: NewsAnalysis | None = None
            if headlines or events:
                try:
                    r = await analyze_news(
                        symbol=symbol, headlines=headlines, events=events,
                        runtime=runtime, model=model, cache=cache,
                    )
                    news_result = r.value
                except Exception as e:  # noqa: BLE001
                    logger.warning("news agent failed for %s Q%d %d: %s", symbol, q, y, e)

            agg = await aggregate(
                symbol=symbol,
                fundamentals=None,  # lookahead-unsafe; excluded
                transcript=tx_result, news=news_result,
            )
            rating: Rating = agg.value

            actual_ret = one_day_return_pct(prices, approx_date)
            if actual_ret is None:
                logger.info("no price reference for %s near %s, skipping", symbol, approx_date)
                continue
            actual_label = _label_from_return(actual_ret)

            points.append(
                BacktestPoint(
                    symbol=symbol.upper(),
                    period=f"Q{q} {y}",
                    announcement_date=approx_date,
                    predicted=rating.label,
                    predicted_score=rating.score,
                    confidence=rating.confidence,
                    actual_label=actual_label,
                    actual_ret_pct=round(actual_ret, 3),
                    hit=(rating.label == actual_label),
                )
            )

    total = len(points)
    directional = sum(1 for p in points if p.predicted != "Neutral")
    hits = sum(1 for p in points if p.hit)
    actuals = [p.actual_label for p in points]
    baseline = _baseline_hit_rate(actuals)

    def _avg_when(label: Label) -> float:
        rets = [p.actual_ret_pct for p in points if p.predicted == label]
        return round(sum(rets) / len(rets), 3) if rets else 0.0

    return BacktestSummary(
        symbol=symbol.upper(),
        total=total,
        directional=directional,
        hits=hits,
        hit_rate=round(hits / total, 4) if total else 0.0,
        baseline=round(baseline, 4),
        avg_ret_when_positive=_avg_when("Positive"),
        avg_ret_when_negative=_avg_when("Negative"),
        avg_ret_when_neutral=_avg_when("Neutral"),
        points=points,
    )
