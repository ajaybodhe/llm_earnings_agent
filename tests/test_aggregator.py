import pytest

from llm_earnings_agent.agents.aggregator import _compute_rating, aggregate
from llm_earnings_agent.schemas import (
    DynamicAnalysis,
    FundamentalsAnalysis,
    MacroAnalysis,
    NewsAnalysis,
    TranscriptAnalysis,
)


def _f(score, conf=0.8):
    return FundamentalsAnalysis(score=score, confidence=conf, themes=[], reasoning="fund reason")


def _t(score, conf=0.8):
    return TranscriptAnalysis(
        score=score, confidence=conf,
        sentiment="neutral", guidance_direction="none",
        themes=[], management_tone="neutral", reasoning="tx reason",
    )


def _n(score, conf=0.8):
    return NewsAnalysis(
        score=score, confidence=conf, material_dev_count=0,
        polarity="neutral", reasoning="news reason",
    )


def _m(score, conf=0.8):
    return MacroAnalysis(
        score=score, confidence=conf,
        regime="neutral", sector_trend="neutral", reasoning="macro reason",
    )


def _d(score, conf=0.8):
    return DynamicAnalysis(
        score=score, confidence=conf,
        momentum="neutral", overbought_oversold="neutral", reasoning="dynamic reason",
    )


def test_compute_rating_all_none():
    r = _compute_rating(None, None, None)
    assert r.label == "Neutral"
    assert r.score == 0.0
    assert r.confidence == 0.0


def test_compute_rating_strong_positive():
    r = _compute_rating(_f(0.8), _t(0.7), _n(0.5))
    assert r.label == "Positive"
    assert r.score > 15


def test_compute_rating_strong_negative():
    r = _compute_rating(_f(-0.8), _t(-0.7), _n(-0.5))
    assert r.label == "Negative"
    assert r.score < -15


def test_compute_rating_low_confidence_forced_neutral():
    r = _compute_rating(_f(0.9, conf=0.2), _t(0.9, conf=0.2), _n(0.9, conf=0.2))
    assert r.label == "Neutral"  # confidence below 0.4 → neutral regardless


def test_top_reasons_ranks_by_impact():
    r = _compute_rating(_f(0.9, conf=0.9), _t(0.1, conf=0.1), _n(-0.3, conf=0.5))
    assert r.top_reasons, "expected at least one reason"
    assert "Fundamentals" in r.top_reasons[0]
    assert r.top_reasons[0].startswith("[+]")


@pytest.mark.asyncio
async def test_aggregate_default_skips_llm():
    res = await aggregate(symbol="X", fundamentals=_f(0.5), transcript=_t(0.5), news=_n(0.5))
    assert res.usage.model == "deterministic"
    assert res.usage.cost_usd == 0.0


def test_compute_rating_with_macro_and_dynamic():
    # Macro/dynamic both bearish must drag a mildly-positive blend toward neutral/negative.
    r_without = _compute_rating(_f(0.4), _t(0.3), _n(0.2))
    r_with = _compute_rating(_f(0.4), _t(0.3), _n(0.2), _m(-0.8), _d(-0.7))
    assert r_with.score < r_without.score


def test_compute_rating_dynamic_only():
    # With only the dynamic agent present, score must still respect its sign.
    r = _compute_rating(None, None, None, None, _d(0.9, conf=0.9))
    assert r.score > 0
    assert "Dynamic" in r.top_reasons[0]
