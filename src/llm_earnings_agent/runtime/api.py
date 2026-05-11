"""Anthropic Messages API runtime.

Uses the official `anthropic` Python SDK. Requires `ANTHROPIC_API_KEY`.
Structured outputs are obtained by asking the model to emit JSON matching the
schema and validating the response — the SDK's tool-use mode would be more
robust, but tool-result parsing is overkill for this single-shot pattern and
mirrors how `claude_code.py` works.
"""

from __future__ import annotations

import json
import os

from .base import CompletionResult, CompletionUsage, T
from .claude_code import _build_prompt, _strip_code_fences

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"

# Public prices per 1M tokens. Update if Anthropic changes them.
_PRICING_PER_M = {
    "claude-sonnet-4-5-20250929": (3.0, 15.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-7": (15.0, 75.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}


def _estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    in_rate, out_rate = _PRICING_PER_M.get(model, (3.0, 15.0))
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


class AnthropicAPIRuntime:
    name = "anthropic_api"

    def __init__(
        self,
        *,
        default_model: str = DEFAULT_MODEL,
        max_tokens: int = 2048,
    ) -> None:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        # Imported lazily so the dependency is only required when this runtime is used.
        from anthropic import AsyncAnthropic

        self._client = AsyncAnthropic()
        self.default_model = default_model
        self.max_tokens = max_tokens

    async def complete(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        model: str | None = None,
    ) -> CompletionResult[T]:
        chosen_model = model or self.default_model
        schema_json = schema.model_json_schema()
        user_prompt = _build_prompt("", user, schema_json)

        msg = await self._client.messages.create(
            model=chosen_model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = "".join(block.text for block in msg.content if block.type == "text")
        if not text:
            raise RuntimeError("empty response from Anthropic API")

        payload = json.loads(_strip_code_fences(text))
        value = schema.model_validate(payload)

        usage = CompletionUsage(
            model=chosen_model,
            input_tokens=msg.usage.input_tokens,
            output_tokens=msg.usage.output_tokens,
            cost_usd=_estimate_cost(chosen_model, msg.usage.input_tokens, msg.usage.output_tokens),
        )
        return CompletionResult(value=value, usage=usage)
