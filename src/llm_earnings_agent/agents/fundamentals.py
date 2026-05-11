"""Fundamentals agent."""

from __future__ import annotations

import json
from typing import Any

from ..cache import JsonCache
from ..runtime.base import CompletionResult, Runtime
from ..schemas import FundamentalsAnalysis
from ._common import cached_complete

AGENT_NAME = "fundamentals"


async def analyze_fundamentals(
    *,
    symbol: str,
    fundamentals: dict[str, Any],
    runtime: Runtime,
    model: str | None = None,
    cache: JsonCache | None = None,
) -> CompletionResult[FundamentalsAnalysis]:
    user = f"Symbol: {symbol}\n\nFundamentals JSON:\n{json.dumps(fundamentals, indent=2, default=str)}"
    return await cached_complete(
        agent_name=AGENT_NAME,
        symbol=symbol,
        payload={"fundamentals": fundamentals},
        user_prompt=user,
        schema=FundamentalsAnalysis,
        runtime=runtime,
        model=model,
        cache=cache,
    )
