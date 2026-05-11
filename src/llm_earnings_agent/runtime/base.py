"""Runtime protocol for executing structured-output LLM calls.

Implementations are swappable behind this interface. `claude_code.py` shells
out to `claude -p` (free under the Claude Code subscription, dev-friendly);
`api.py` uses the Anthropic Messages API (per-token billing, parallel-friendly).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


@dataclass(frozen=True)
class CompletionUsage:
    """Bookkeeping returned alongside the structured payload."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass(frozen=True)
class CompletionResult(Generic[T]):
    value: T
    usage: CompletionUsage


class Runtime(Protocol):
    """A runtime that produces a typed `BaseModel` from a system+user prompt."""

    name: str

    async def complete(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        model: str | None = None,
    ) -> CompletionResult[T]: ...


def auto_select_runtime() -> Runtime:
    """Pick a runtime based on environment.

    `ANTHROPIC_API_KEY` set → `AnthropicAPIRuntime`. Otherwise `ClaudeCodeRuntime`.
    `LLM_RUNTIME` env var (`api` | `claude-code`) overrides.
    """
    import os

    forced = os.environ.get("LLM_RUNTIME", "").lower()
    if forced == "api" or (forced == "" and os.environ.get("ANTHROPIC_API_KEY")):
        from .api import AnthropicAPIRuntime

        return AnthropicAPIRuntime()
    from .claude_code import ClaudeCodeRuntime

    return ClaudeCodeRuntime()
