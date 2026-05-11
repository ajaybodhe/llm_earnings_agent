"""News agent."""

from __future__ import annotations

from ..cache import JsonCache
from ..data.events import MaterialEvent
from ..data.news import NewsItem
from ..runtime.base import CompletionResult, Runtime
from ..schemas import NewsAnalysis
from ._common import cached_complete

AGENT_NAME = "news"
MAX_HEADLINES = 50


async def analyze_news(
    *,
    symbol: str,
    headlines: list[NewsItem],
    events: list[MaterialEvent],
    runtime: Runtime,
    model: str | None = None,
    cache: JsonCache | None = None,
) -> CompletionResult[NewsAnalysis]:
    trimmed = sorted(headlines, key=lambda h: h.datetime_utc, reverse=True)[:MAX_HEADLINES]

    headline_lines = [
        f"- {h.datetime_utc.date().isoformat()} ({h.source}): {h.headline}" for h in trimmed
    ]
    event_lines = [
        f"- {e.filed_date.isoformat()} 8-K items {e.items or 'n/a'}" for e in events
    ]

    user = (
        f"Symbol: {symbol}\n\n"
        f"Recent headlines ({len(trimmed)} of {len(headlines)} shown):\n"
        + "\n".join(headline_lines or ["(none)"])
        + "\n\n"
        f"Recent 8-K material events:\n"
        + "\n".join(event_lines or ["(none)"])
    )

    return await cached_complete(
        agent_name=AGENT_NAME,
        symbol=symbol,
        payload={
            "headline_count": len(headlines),
            "event_count": len(events),
            "first_headlines": [h.headline for h in trimmed[:5]],
        },
        user_prompt=user,
        schema=NewsAnalysis,
        runtime=runtime,
        model=model,
        cache=cache,
    )
