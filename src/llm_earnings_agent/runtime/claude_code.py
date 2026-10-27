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
WEB_TIMEOUT_S = 420  # searching and reading pages takes several turns


def _strip_code_fences(text: str) -> str:
    """Pull JSON out of a ```json fenced block if the model wrapped it."""
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    return m.group(1) if m else text.strip()


def parse_json_object(text: str) -> Any:
    """The JSON object in a model reply: the whole reply, a fenced block, or
    else the last top-level {...} (web-search replies can add prose first)."""
    text = _strip_code_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    dec = json.JSONDecoder()
    last = None
    i = text.find("{")
    while i != -1:
        try:
            obj, end = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            i = text.find("{", i + 1)
            continue
        if isinstance(obj, dict):
            last = obj
        i = text.find("{", end)
    if last is None:
        raise ValueError(f"no JSON object in model reply: {text[:200]!r}")
    return last


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
        web_search: bool = False,
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
        timeout = self.timeout_s
        if web_search:
            args += ["--allowedTools", "WebSearch,WebFetch"]
            timeout = max(timeout, WEB_TIMEOUT_S)
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except TimeoutError:
            proc.kill()
            raise RuntimeError(f"claude -p timed out after {timeout}s") from None

        if proc.returncode != 0:
            raise RuntimeError(
                f"claude -p exited {proc.returncode}: {stderr.decode(errors='replace')[:500]}"
            )

        envelope = json.loads(stdout.decode())
        text = envelope.get("result") or envelope.get("response") or ""
        if not text:
            raise RuntimeError(f"empty result from claude -p; envelope keys={list(envelope)}")

        payload = parse_json_object(text)
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
