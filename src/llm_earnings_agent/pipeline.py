"""End-to-end single-ticker analysis pipeline.

Fetches data, runs the selected sub-agents, aggregates. Returns the top-level
`AgentResponse` ready to be serialized to JSON.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from typing import Any

import httpx

from .agents import (
    aggregate,
    analyze_dynamic,
    analyze_fundamentals,
    analyze_macro,
    analyze_news,
    analyze_transcript,
)
from .cache import JsonCache
from .data.events import fetch_material_events
from .data.fundamentals import fetch_fundamentals
from .data.news import fetch_company_news
from .data.transcripts import fetch_latest_transcript
from .runtime.base import CompletionUsage, Runtime
from .schemas import (
    AgentResponse,
    DynamicAnalysis,
    FundamentalsAnalysis,
    MacroAnalysis,
    Metadata,
    NewsAnalysis,
    SubRatings,
    TranscriptAnalysis,
)

ALL_AGENTS: tuple[str, ...] = ("fundamentals", "transcript", "news", "macro", "dynamic")

# Agents that read the quarterly_results JSON payload. The other two
# (transcript, news) fetch their own text inputs and never need it.
PAYLOAD_AGENTS: frozenset[str] = frozenset({"fundamentals", "macro", "dynamic"})


def parse_agents(spec: str | None) -> frozenset[str]:
    """Parse a comma-separated agent list (or "all") into a validated set."""
    if spec is None or spec.strip().lower() in ("", "all"):
        return frozenset(ALL_AGENTS)
    names = {part.strip().lower() for part in spec.split(",") if part.strip()}
    unknown = names - set(ALL_AGENTS)
    if unknown:
        choices = ", ".join(ALL_AGENTS)
        raise ValueError(
            f"unknown agent(s): {', '.join(sorted(unknown))} — choose from {choices} or 'all'"
        )
    return frozenset(names)


# Fields from the quarterly_results JSON payload that feed each specialist agent.
# Splitting the payload keeps prompts focused and cache keys per-agent stable.
_MACRO_FIELDS = (
    "symbol",
    "earnings_date",
    "macro_context",
    "sector_etf",
    "sector_ret_1m",
    "sector_ret_3m",
    "currency",
)
_DYNAMIC_FIELDS = (
    "symbol",
    "current_price",
    "ret_1w",
    "ret_1m",
    "ret_6m",
    "ret_1y",
    "rsi14",
    "hi_52",
    "lo_52",
    "pct_from_52hi",
    "pct_from_52lo",
    "beat_rate",
    "avg_beat_pct",
    "implied_vs_hist_ratio",
    "earnings_reactions",
)


def _models_used(usages: list[CompletionUsage]) -> str:
    """Comma-separated distinct LLM models across calls, in first-use order.

    The deterministic aggregator reports model="deterministic"; it is only
    named when no LLM call ran at all.
    """
    names: list[str] = []
    for u in usages:
        if u.model and u.model != "deterministic" and u.model not in names:
            names.append(u.model)
    return ",".join(names) or "deterministic"


def _slice(payload: dict[str, Any], fields: tuple[str, ...]) -> dict[str, Any]:
    return {k: payload[k] for k in fields if k in payload}


def _has_signal(payload: dict[str, Any]) -> bool:
    """True when at least one value is non-empty/non-null/non-N/A."""
    for v in payload.values():
        if v in (None, "", "N/A"):
            continue
        if isinstance(v, (list, dict)) and not v:
            continue
        return True
    return False

logger = logging.getLogger(__name__)


async def analyze_symbol(
    symbol: str,
    *,
    runtime: Runtime,
    model: str | None = None,
    cache: JsonCache | None = None,
    news_lookback_days: int = 30,
    use_llm_aggregator: bool = False,
    agents: frozenset[str] | None = None,
    fundamentals_payload: dict[str, Any] | None = None,
) -> AgentResponse:
    """Analyze one ticker.

    ``agents`` restricts which specialist agents run (default: all five).
    ``fundamentals_payload`` is the caller's own quarterly_results JSON for
    this symbol; when given, the Go binary is not re-run. When no selected
    agent reads the payload, the Go binary is skipped entirely.
    """
    selected = agents if agents is not None else frozenset(ALL_AGENTS)
    today = dt.date.today()
    news_from = today - dt.timedelta(days=news_lookback_days)

    async with httpx.AsyncClient(timeout=30.0) as http:
        fundamentals_data: dict[str, Any] | None = fundamentals_payload
        transcript_obj = None
        headlines: list = []
        events: list = []

        async def _fundamentals() -> None:
            nonlocal fundamentals_data
            try:
                payload = await fetch_fundamentals(symbol)
                fundamentals_data = payload.data
            except Exception as e:  # noqa: BLE001
                logger.warning("fundamentals fetch failed for %s: %s", symbol, e)

        async def _transcript() -> None:
            nonlocal transcript_obj
            try:
                transcript_obj = await fetch_latest_transcript(symbol, client=http)
            except Exception as e:  # noqa: BLE001
                logger.warning("transcript fetch failed for %s: %s", symbol, e)

        async def _news() -> None:
            nonlocal headlines, events
            try:
                headlines = await fetch_company_news(symbol, from_date=news_from, to_date=today, client=http)
            except Exception as e:  # noqa: BLE001
                logger.warning("news fetch failed for %s: %s", symbol, e)
            try:
                events = await fetch_material_events(symbol, since=news_from, client=http)
            except Exception as e:  # noqa: BLE001
                logger.warning("events fetch failed for %s: %s", symbol, e)

        fetches = []
        if fundamentals_data is None and selected & PAYLOAD_AGENTS:
            fetches.append(_fundamentals())
        if "transcript" in selected:
            fetches.append(_transcript())
        if "news" in selected:
            fetches.append(_news())
        await asyncio.gather(*fetches)

    # Sub-agents (skip any whose inputs are missing). Each call's usage is kept
    # so the metadata reports the models that actually ran and their total cost.
    usages: list[CompletionUsage] = []
    fund_result: FundamentalsAnalysis | None = None
    if fundamentals_data and "fundamentals" in selected:
        try:
            r = await analyze_fundamentals(
                symbol=symbol, fundamentals=fundamentals_data,
                runtime=runtime, model=model, cache=cache,
            )
            fund_result = r.value
            usages.append(r.usage)
        except Exception as e:  # noqa: BLE001
            logger.warning("fundamentals agent failed for %s: %s", symbol, e)

    tx_result: TranscriptAnalysis | None = None
    transcript_label: str | None = None
    if transcript_obj is not None:
        transcript_label = transcript_obj.label
        try:
            r = await analyze_transcript(
                symbol=symbol, transcript=transcript_obj,
                runtime=runtime, model=model, cache=cache,
            )
            tx_result = r.value
            usages.append(r.usage)
        except Exception as e:  # noqa: BLE001
            logger.warning("transcript agent failed for %s: %s", symbol, e)

    news_result: NewsAnalysis | None = None
    if headlines or events:
        try:
            r = await analyze_news(
                symbol=symbol, headlines=headlines, events=events,
                runtime=runtime, model=model, cache=cache,
            )
            news_result = r.value
            usages.append(r.usage)
        except Exception as e:  # noqa: BLE001
            logger.warning("news agent failed for %s: %s", symbol, e)

    macro_result: MacroAnalysis | None = None
    dynamic_result: DynamicAnalysis | None = None
    if fundamentals_data:
        macro_slice = _slice(fundamentals_data, _MACRO_FIELDS)
        if "macro" in selected and _has_signal(macro_slice):
            try:
                r = await analyze_macro(
                    symbol=symbol, macro=macro_slice,
                    runtime=runtime, model=model, cache=cache,
                )
                macro_result = r.value
                usages.append(r.usage)
            except Exception as e:  # noqa: BLE001
                logger.warning("macro agent failed for %s: %s", symbol, e)

        dynamic_slice = _slice(fundamentals_data, _DYNAMIC_FIELDS)
        if "dynamic" in selected and _has_signal(dynamic_slice):
            try:
                r = await analyze_dynamic(
                    symbol=symbol, dynamic=dynamic_slice,
                    runtime=runtime, model=model, cache=cache,
                )
                dynamic_result = r.value
                usages.append(r.usage)
            except Exception as e:  # noqa: BLE001
                logger.warning("dynamic agent failed for %s: %s", symbol, e)

    agg = await aggregate(
        symbol=symbol,
        fundamentals=fund_result,
        transcript=tx_result,
        news=news_result,
        macro=macro_result,
        dynamic=dynamic_result,
        runtime=runtime if use_llm_aggregator else None,
        model=model,
        cache=cache,
        use_llm=use_llm_aggregator,
    )

    return AgentResponse(
        symbol=symbol,
        asof=today,
        rating=agg.value,
        sub_ratings=SubRatings(
            fundamentals=fund_result,
            transcript=tx_result,
            news=news_result,
            macro=macro_result,
            dynamic=dynamic_result,
        ),
        metadata=Metadata(
            model=_models_used([*usages, agg.usage]),
            runtime=runtime.name,
            prompt_version="v1",
            timestamp=dt.datetime.now(tz=dt.UTC),
            cost_estimate_usd=round(sum(u.cost_usd for u in (*usages, agg.usage)), 6),
            transcript_quarter=transcript_label,
            agents=[a for a in ALL_AGENTS if a in selected],
        ),
    )
