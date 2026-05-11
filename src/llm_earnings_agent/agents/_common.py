"""Shared helpers for agent modules."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from ..cache import JsonCache, cache_key
from ..runtime.base import CompletionResult, CompletionUsage, Runtime, T

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"
_VERSION_RE = re.compile(r"<!--\s*prompt_version:\s*([A-Za-z0-9._-]+)\s*-->")


def load_prompt(name: str) -> tuple[str, str]:
    """Return (prompt_body_without_metadata_comment, prompt_version)."""
    path = PROMPTS_DIR / f"{name}.md"
    text = path.read_text(encoding="utf-8")
    m = _VERSION_RE.search(text)
    version = m.group(1) if m else "unknown"
    body = _VERSION_RE.sub("", text).strip()
    return body, version


async def cached_complete(
    *,
    agent_name: str,
    symbol: str,
    payload: dict[str, Any],
    user_prompt: str,
    schema: type[T],
    runtime: Runtime,
    model: str | None,
    cache: JsonCache | None,
) -> CompletionResult[T]:
    """Run an agent against the runtime, caching the validated result by inputs."""
    system, version = load_prompt(agent_name)
    chosen_model = model or "default"

    key: str | None = None
    if cache is not None:
        key = cache_key(
            ticker=symbol,
            agent=agent_name,
            prompt_version=version,
            model=chosen_model,
            payload=payload,
        )
        cached = cache.get(key)
        if cached is not None:
            usage_block = cached.get("usage") or {}
            return CompletionResult(
                value=schema.model_validate(cached["value"]),
                usage=CompletionUsage(
                    model=str(usage_block.get("model", chosen_model)),
                    input_tokens=int(usage_block.get("input_tokens", 0)),
                    output_tokens=int(usage_block.get("output_tokens", 0)),
                    cost_usd=float(usage_block.get("cost_usd", 0.0)),
                ),
            )

    result = await runtime.complete(
        system=system,
        user=user_prompt,
        schema=schema,
        model=model,
    )

    if cache is not None and key is not None:
        cache.put(
            key,
            {
                "value": _serialize_value(result.value),
                "usage": {
                    "model": result.usage.model,
                    "input_tokens": result.usage.input_tokens,
                    "output_tokens": result.usage.output_tokens,
                    "cost_usd": result.usage.cost_usd,
                },
                "prompt_version": version,
            },
        )
    return result


def _serialize_value(value: BaseModel) -> dict:
    return value.model_dump(mode="json")
