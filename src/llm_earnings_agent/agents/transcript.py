"""Transcript agent."""

from __future__ import annotations

from ..cache import JsonCache
from ..data.transcripts import Transcript
from ..runtime.base import CompletionResult, Runtime
from ..schemas import TranscriptAnalysis
from ._common import cached_complete

AGENT_NAME = "transcript"
MAX_TRANSCRIPT_CHARS = 60_000  # ≈15k tokens; leaves room for prompt + response


async def analyze_transcript(
    *,
    symbol: str,
    transcript: Transcript,
    runtime: Runtime,
    model: str | None = None,
    cache: JsonCache | None = None,
) -> CompletionResult[TranscriptAnalysis]:
    body = transcript.content[:MAX_TRANSCRIPT_CHARS]
    user = (
        f"Symbol: {symbol}\nQuarter: {transcript.label}\n\n"
        f"Transcript (truncated to {MAX_TRANSCRIPT_CHARS} chars):\n{body}"
    )
    return await cached_complete(
        agent_name=AGENT_NAME,
        symbol=symbol,
        payload={
            "quarter": transcript.quarter,
            "year": transcript.year,
            "content_len": len(transcript.content),
            "content_hash_prefix": body[:200],  # plus content_len makes a stable key
        },
        user_prompt=user,
        schema=TranscriptAnalysis,
        runtime=runtime,
        model=model,
        cache=cache,
    )
