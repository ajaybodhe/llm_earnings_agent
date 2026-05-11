"""Aggregator — combines three sub-analyses into a final rating.

We do the weight × score × confidence math deterministically in Python rather
than delegating to the LLM. The aggregator agent (if you flip ``use_llm=True``)
becomes a sanity pass that may rewrite ``top_reasons`` and override the label
on tight calls, but the score arithmetic remains fixed.
"""

from __future__ import annotations

from ..cache import JsonCache
from ..runtime.base import CompletionResult, CompletionUsage, Runtime
from ..schemas import (
    DynamicAnalysis,
    FundamentalsAnalysis,
    Label,
    MacroAnalysis,
    NewsAnalysis,
    Rating,
    TranscriptAnalysis,
)
from ._common import cached_complete

AGENT_NAME = "aggregator"

# Base weights (must sum to 1.0). Effective weight per sub = base × sub.confidence.
# Five-agent split mirrors MarketSenseAI 2.0: fundamentals + transcript dominate,
# news/macro/dynamic complete the picture.
WEIGHT_FUNDAMENTALS = 0.30
WEIGHT_TRANSCRIPT = 0.25
WEIGHT_NEWS = 0.15
WEIGHT_MACRO = 0.10
WEIGHT_DYNAMIC = 0.20

LABEL_SCORE_THRESHOLD = 15.0
LABEL_CONF_THRESHOLD = 0.4


def _label_for(score: float, confidence: float) -> Label:
    if confidence < LABEL_CONF_THRESHOLD:
        return "Neutral"
    if score >= LABEL_SCORE_THRESHOLD:
        return "Positive"
    if score <= -LABEL_SCORE_THRESHOLD:
        return "Negative"
    return "Neutral"


def _compute_rating(
    fundamentals: FundamentalsAnalysis | None,
    transcript: TranscriptAnalysis | None,
    news: NewsAnalysis | None,
    macro: MacroAnalysis | None = None,
    dynamic: DynamicAnalysis | None = None,
) -> Rating:
    parts: list[tuple[str, float, float, float, str]] = []
    if fundamentals is not None:
        parts.append(("Fundamentals", WEIGHT_FUNDAMENTALS, fundamentals.score, fundamentals.confidence, fundamentals.reasoning))
    if transcript is not None:
        parts.append(("Transcript", WEIGHT_TRANSCRIPT, transcript.score, transcript.confidence, transcript.reasoning))
    if news is not None:
        parts.append(("News", WEIGHT_NEWS, news.score, news.confidence, news.reasoning))
    if macro is not None:
        parts.append(("Macro", WEIGHT_MACRO, macro.score, macro.confidence, macro.reasoning))
    if dynamic is not None:
        parts.append(("Dynamic", WEIGHT_DYNAMIC, dynamic.score, dynamic.confidence, dynamic.reasoning))

    if not parts:
        return Rating(label="Neutral", score=0.0, confidence=0.0, top_reasons=[])

    used = sum(w for _, w, _, _, _ in parts)  # weights of agents that returned data
    weighted = sum(w * s * c for _, w, s, c, _ in parts)
    confidence_blend = sum(w * c for _, w, _, c, _ in parts) / used if used > 0 else 0.0
    score_100 = (weighted / used) * 100.0 if used > 0 else 0.0

    ranked = sorted(parts, key=lambda p: abs(p[1] * p[2] * p[3]), reverse=True)
    top_reasons: list[str] = []
    for name, _w, s, c, reason in ranked[:3]:
        if not reason or c == 0:
            continue
        prefix = "[+]" if s > 0 else "[−]" if s < 0 else "[·]"
        short = reason.strip().splitlines()[0][:100]
        top_reasons.append(f"{prefix} {name}: {short}")

    return Rating(
        label=_label_for(score_100, confidence_blend),
        score=round(score_100, 2),
        confidence=round(confidence_blend, 4),
        top_reasons=top_reasons,
    )


async def aggregate(
    *,
    symbol: str,
    fundamentals: FundamentalsAnalysis | None,
    transcript: TranscriptAnalysis | None,
    news: NewsAnalysis | None,
    macro: MacroAnalysis | None = None,
    dynamic: DynamicAnalysis | None = None,
    runtime: Runtime | None = None,
    model: str | None = None,
    cache: JsonCache | None = None,
    use_llm: bool = False,
) -> CompletionResult[Rating]:
    """Compute the final rating.

    ``use_llm=False`` (default): deterministic math, no LLM call.
    ``use_llm=True``: ask the aggregator prompt to re-score (slower, costs tokens, may surface qualitative reasons the math misses). Requires runtime.
    """
    deterministic = _compute_rating(fundamentals, transcript, news, macro, dynamic)
    if not use_llm:
        return CompletionResult(
            value=deterministic,
            usage=CompletionUsage(model="deterministic", input_tokens=0, output_tokens=0, cost_usd=0.0),
        )

    if runtime is None:
        raise ValueError("use_llm=True requires a runtime")

    user_lines = [f"Symbol: {symbol}", f"Deterministic blend: {deterministic.model_dump_json()}"]
    if fundamentals:
        user_lines.append(f"Fundamentals sub-analysis: {fundamentals.model_dump_json()}")
    if transcript:
        user_lines.append(f"Transcript sub-analysis: {transcript.model_dump_json()}")
    if news:
        user_lines.append(f"News sub-analysis: {news.model_dump_json()}")
    if macro:
        user_lines.append(f"Macro sub-analysis: {macro.model_dump_json()}")
    if dynamic:
        user_lines.append(f"Dynamic sub-analysis: {dynamic.model_dump_json()}")
    user_prompt = "\n\n".join(user_lines)

    return await cached_complete(
        agent_name=AGENT_NAME,
        symbol=symbol,
        payload={
            "fundamentals": fundamentals.model_dump(mode="json") if fundamentals else None,
            "transcript": transcript.model_dump(mode="json") if transcript else None,
            "news": news.model_dump(mode="json") if news else None,
            "macro": macro.model_dump(mode="json") if macro else None,
            "dynamic": dynamic.model_dump(mode="json") if dynamic else None,
        },
        user_prompt=user_prompt,
        schema=Rating,
        runtime=runtime,
        model=model,
        cache=cache,
    )
