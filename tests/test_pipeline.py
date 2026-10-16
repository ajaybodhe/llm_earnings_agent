import json

import pytest
from typer.testing import CliRunner

from llm_earnings_agent import cli, pipeline
from llm_earnings_agent.data.fundamentals import parse_payload
from llm_earnings_agent.pipeline import ALL_AGENTS, _models_used, analyze_symbol, parse_agents
from llm_earnings_agent.runtime.base import CompletionResult, CompletionUsage
from llm_earnings_agent.schemas import (
    DynamicAnalysis,
    FundamentalsAnalysis,
    MacroAnalysis,
    NewsAnalysis,
    TranscriptAnalysis,
)

# ── parse_agents ─────────────────────────────────────────────────────────────


def test_parse_agents_all_and_default():
    assert parse_agents(None) == frozenset(ALL_AGENTS)
    assert parse_agents("all") == frozenset(ALL_AGENTS)
    assert parse_agents(" ALL ") == frozenset(ALL_AGENTS)


def test_parse_agents_subset_normalised():
    assert parse_agents("Transcript, news") == frozenset({"transcript", "news"})


def test_parse_agents_rejects_unknown():
    with pytest.raises(ValueError, match="unknown agent"):
        parse_agents("transcript,sentiment")


# ── parse_payload ────────────────────────────────────────────────────────────


def test_parse_payload_object():
    p = parse_payload('{"symbol": "AAPL", "rsi14": "55.0"}', "aapl")
    assert p.symbol == "AAPL"
    assert p.data["rsi14"] == "55.0"


def test_parse_payload_array_picks_matching_symbol():
    text = json.dumps([{"symbol": "MSFT"}, {"symbol": "AAPL", "pe_ttm": "30x"}])
    assert parse_payload(text, "AAPL").data["pe_ttm"] == "30x"


def test_parse_payload_array_falls_back_to_first():
    text = json.dumps([{"symbol": "MSFT"}])
    assert parse_payload(text, "AAPL").data["symbol"] == "MSFT"


def test_parse_payload_skips_log_noise():
    p = parse_payload('Fetching ...\nDone.\n{"symbol": "AAPL"}', "AAPL")
    assert p.data == {"symbol": "AAPL"}


def test_parse_payload_empty_raises():
    with pytest.raises(RuntimeError):
        parse_payload("   ", "AAPL")


# ── analyze_symbol agent selection ───────────────────────────────────────────

_PAYLOAD = {
    "symbol": "AAPL",
    "earnings_date": "2026-11-02",
    "sector_etf": "XLK",
    "sector_ret_1m": 2.0,
    "current_price": "$342.00",
    "rsi14": "55.0",
}


def _usage():
    return CompletionUsage(model="test-model", input_tokens=0, output_tokens=0, cost_usd=0.01)


@pytest.fixture
def stub_pipeline(monkeypatch):
    """Replace every data fetch and agent call with recorders."""
    calls: list[str] = []

    async def fake_fetch_fundamentals(symbol, **_):
        calls.append("go")
        raise AssertionError("Go binary must not be re-run in these tests")

    async def fake_transcript(symbol, **_):
        calls.append("fetch_transcript")
        return None

    async def fake_news(symbol, **_):
        calls.append("fetch_news")
        return [object()]

    async def fake_events(symbol, **_):
        calls.append("fetch_events")
        return []

    def agent(name, value):
        async def _run(**_):
            calls.append(name)
            return CompletionResult(value=value, usage=_usage())

        return _run

    monkeypatch.setattr(pipeline, "fetch_fundamentals", fake_fetch_fundamentals)
    monkeypatch.setattr(pipeline, "fetch_latest_transcript", fake_transcript)
    monkeypatch.setattr(pipeline, "fetch_company_news", fake_news)
    monkeypatch.setattr(pipeline, "fetch_material_events", fake_events)
    monkeypatch.setattr(
        pipeline,
        "analyze_fundamentals",
        agent("fundamentals", FundamentalsAnalysis(score=0.5, confidence=0.8, reasoning="f")),
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_transcript",
        agent(
            "transcript",
            TranscriptAnalysis(
                score=0.5, confidence=0.5, sentiment="positive",
                guidance_direction="raised", management_tone="scripted", reasoning="t",
            ),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_news",
        agent(
            "news",
            NewsAnalysis(
                score=0.4, confidence=0.6, material_dev_count=1,
                polarity="positive", reasoning="n",
            ),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_macro",
        agent(
            "macro",
            MacroAnalysis(
                score=0.1, confidence=0.5, regime="neutral",
                sector_trend="positive", reasoning="m",
            ),
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "analyze_dynamic",
        agent(
            "dynamic",
            DynamicAnalysis(
                score=0.2, confidence=0.5, momentum="positive",
                overbought_oversold="neutral", reasoning="d",
            ),
        ),
    )
    return calls


class _Runtime:
    name = "test"


async def test_text_only_agents_skip_go_and_payload_agents(stub_pipeline):
    resp = await analyze_symbol(
        "AAPL", runtime=_Runtime(), agents=frozenset({"transcript", "news"})
    )
    assert "go" not in stub_pipeline
    assert not {"fundamentals", "macro", "dynamic"} & set(stub_pipeline)
    assert "news" in stub_pipeline
    assert resp.metadata.agents == ["transcript", "news"]
    assert resp.sub_ratings.fundamentals is None
    # Metadata names the specialists' model, not the deterministic aggregator,
    # and sums the cost of the calls that ran (news only: no transcript found).
    assert resp.metadata.model == "test-model"
    assert resp.metadata.cost_estimate_usd == pytest.approx(0.01)


async def test_payload_replaces_go_subprocess(stub_pipeline):
    resp = await analyze_symbol(
        "AAPL", runtime=_Runtime(), fundamentals_payload=_PAYLOAD
    )
    assert "go" not in stub_pipeline
    assert {"fundamentals", "macro", "dynamic"} <= set(stub_pipeline)
    assert resp.metadata.agents == list(ALL_AGENTS)


async def test_unselected_text_agents_skip_their_fetches(stub_pipeline):
    await analyze_symbol(
        "AAPL", runtime=_Runtime(), agents=frozenset({"dynamic"}),
        fundamentals_payload=_PAYLOAD,
    )
    assert stub_pipeline == ["dynamic"]


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_cli_reads_payload_from_stdin(monkeypatch):
    seen = {}

    async def fake_analyze(symbol, **kwargs):
        seen.update(kwargs)
        raise SystemExit(0)

    monkeypatch.setattr(cli, "analyze_symbol", fake_analyze)
    monkeypatch.setattr(cli, "_runtime_for", lambda name: _Runtime())
    result = CliRunner().invoke(
        cli.app,
        ["analyze", "--symbol", "AAPL", "--agents", "all", "--payload", "-", "--no-cache"],
        input=json.dumps(_PAYLOAD),
    )
    assert result.exit_code == 0, result.output
    assert seen["fundamentals_payload"] == _PAYLOAD
    assert seen["agents"] == frozenset(ALL_AGENTS)


def test_cli_rejects_unknown_agent():
    result = CliRunner().invoke(cli.app, ["analyze", "--symbol", "AAPL", "--agents", "bogus"])
    assert result.exit_code != 0


# ── metadata ─────────────────────────────────────────────────────────────────


def test_models_used_skips_deterministic_and_dedupes():
    def u(model):
        return CompletionUsage(model=model, input_tokens=0, output_tokens=0, cost_usd=0.0)

    usages = [u("sonnet"), u("sonnet"), u("haiku"), u("deterministic")]
    assert _models_used(usages) == "sonnet,haiku"
    assert _models_used([u("deterministic")]) == "deterministic"
    assert _models_used([]) == "deterministic"
