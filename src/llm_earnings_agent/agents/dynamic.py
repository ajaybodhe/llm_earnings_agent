"""Dynamic / price-action agent."""

from __future__ import annotations

import json
from typing import Any

from ..cache import JsonCache
from ..runtime.base import CompletionResult, Runtime
from ..schemas import DynamicAnalysis
from ._common import cached_complete

AGENT_NAME = "dynamic"


async def analyze_dynamic(
    *,
    symbol: str,
    dynamic: dict[str, Any],
    runtime: Runtime,
    model: str | None = None,
    cache: JsonCache | None = None,
) -> CompletionResult[DynamicAnalysis]:
    user = f"Symbol: {symbol}\n\nDynamic / price-action JSON:\n{json.dumps(dynamic, indent=2, default=str)}"
    return await cached_complete(
        agent_name=AGENT_NAME,
        symbol=symbol,
        payload={"dynamic": dynamic},
        user_prompt=user,
        schema=DynamicAnalysis,
        runtime=runtime,
        model=model,
        cache=cache,
    )
