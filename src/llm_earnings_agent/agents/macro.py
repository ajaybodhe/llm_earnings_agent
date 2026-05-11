"""Macroeconomic agent."""

from __future__ import annotations

import json
from typing import Any

from ..cache import JsonCache
from ..runtime.base import CompletionResult, Runtime
from ..schemas import MacroAnalysis
from ._common import cached_complete

AGENT_NAME = "macro"


async def analyze_macro(
    *,
    symbol: str,
    macro: dict[str, Any],
    runtime: Runtime,
    model: str | None = None,
    cache: JsonCache | None = None,
) -> CompletionResult[MacroAnalysis]:
    user = f"Symbol: {symbol}\n\nMacro / sector JSON:\n{json.dumps(macro, indent=2, default=str)}"
    return await cached_complete(
        agent_name=AGENT_NAME,
        symbol=symbol,
        payload={"macro": macro},
        user_prompt=user,
        schema=MacroAnalysis,
        runtime=runtime,
        model=model,
        cache=cache,
    )
