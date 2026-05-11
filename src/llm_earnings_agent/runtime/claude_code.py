"""Claude Code headless runtime.

Shells out to `claude -p` with `--output-format json` and validates the model's
text response against the requested schema. Uses the user's existing Claude
Code subscription quota — no API key, no per-token charge — at the cost of
session rate limits.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from .base import CompletionResult, CompletionUsage, T

DEFAULT_MODEL = "claude-sonnet-4-6"
SUBPROCESS_TIMEOUT_S = 120


def _strip_code_fences(text: str) -> str:
    """Pull JSON out of a ```json fenced block if the model wrapped it."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    return m.group(1) if m else text.strip()


def _build_prompt(system: str, user: str, schema_json: dict[str, Any]) -> str:
    """Combine system+user into a single -p argument with strict JSON instructions."""
    return (
        f"{system}\n\n"
        f"User request:\n{user}\n\n"
        f"Respond with a single JSON object that matches this JSON Schema "
        f"exactly. No prose, no code fences, just the JSON object.\n\n"
        f"{json.dumps(schema_json)}"
    )


class ClaudeCodeRuntime:
    name = "claude_code_headless"

    def __init__(
        self,
        *,
        binary: str = "claude",
        default_model: str = DEFAULT_MODEL,
        timeout_s: int = SUBPROCESS_TIMEOUT_S,
    ) -> None:
        self.binary = binary
        self.default_model = default_model
        self.timeout_s = timeout_s

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
        prompt = _build_prompt(system, user, schema_json)

        args = [
            self.binary,
            "-p",
            prompt,
            "--output-format",
            "json",
            "--model",
            chosen_model,
        ]
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self.timeout_s)
        except TimeoutError:
            proc.kill()
            raise RuntimeError(f"claude -p timed out after {self.timeout_s}s") from None

        if proc.returncode != 0:
            raise RuntimeError(
                f"claude -p exited {proc.returncode}: {stderr.decode(errors='replace')[:500]}"
            )

        envelope = json.loads(stdout.decode())
        text = envelope.get("result") or envelope.get("response") or ""
        if not text:
            raise RuntimeError(f"empty result from claude -p; envelope keys={list(envelope)}")

        payload = json.loads(_strip_code_fences(text))
        value = schema.model_validate(payload)

        # Claude Code's JSON envelope sometimes includes token counts in
        # `usage`; fall back to zeros if not present.
        usage_block = envelope.get("usage") or {}
        usage = CompletionUsage(
            model=chosen_model,
            input_tokens=int(usage_block.get("input_tokens", 0)),
            output_tokens=int(usage_block.get("output_tokens", 0)),
            cost_usd=0.0,  # subscription mode
        )
        return CompletionResult(value=value, usage=usage)
