import asyncio

from llm_earnings_agent.pead import live
from llm_earnings_agent.pead import webearnings as we
from llm_earnings_agent.runtime.base import CompletionResult, CompletionUsage
from llm_earnings_agent.runtime.claude_code import parse_json_object

CDNS = {
    "symbol": "CDNS",
    "fiscal_quarter": "Q3 2026 (Sep/2026)",
    "matches_quarter": True,
    "report_date": "2026-10-26",
    "eps_actual": 2.11,
    "eps_consensus": 2.04,
    "consensus_source": "Zacks",
    "sources": ["https://finance.yahoo.com/x"],
}


def test_surprise_is_recomputed_and_checked():
    w = we.WebEarnings.model_validate(CDNS)
    assert w.surprise_pct() == 3.43
    assert w.usable() == (True, "")
    assert not w.model_copy(update={"sources": []}).usable()[0]
    assert not w.model_copy(update={"matches_quarter": False}).usable()[0]
    assert not w.model_copy(update={"eps_consensus": None}).usable()[0]
    assert not w.model_copy(update={"eps_consensus": 0.01}).usable()[0]  # implausible


def test_needs_backup_only_for_released_reports_without_surprise():
    today = "2026-10-27"
    assert we.needs_backup({"earnings_date": "2026-10-26", "earnings_time": "amc"}, today)
    assert we.needs_backup({"earnings_date": "2026-10-27", "earnings_time": "bmo"}, today)
    assert not we.needs_backup({"earnings_date": "2026-10-27", "earnings_time": "amc"}, today)
    assert not we.needs_backup({"earnings_date": "2026-10-26", "eps_surprise_pct": 1.0}, today)


def test_parse_json_object_after_prose():
    assert parse_json_object('Found it.\n{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 2}\n```') == {"a": 2}
    assert parse_json_object('{"a": 1} then {"b": {"c": 3}}') == {"b": {"c": 3}}


class FakeRuntime:
    def __init__(self, payload):
        self.payload, self.calls = payload, 0

    async def complete(self, *, system, user, schema, model=None, web_search=False):
        assert web_search
        self.calls += 1
        return CompletionResult(schema.model_validate(self.payload), CompletionUsage("m", 0, 0, 0))


def test_fetch_caches_usable_results(tmp_path):
    rt = FakeRuntime(CDNS)
    args = (rt, "CDNS", "Cadence", "2026-10-26", "Sep/2026")
    w1 = asyncio.run(we.fetch(*args, cache_dir=tmp_path))
    w2 = asyncio.run(we.fetch(*args, cache_dir=tmp_path))
    assert rt.calls == 1 and w1 == w2
    bad = FakeRuntime({**CDNS, "sources": []})
    asyncio.run(we.fetch(bad, "X", "X", "2026-10-26", "", cache_dir=tmp_path))
    assert not we.cache_path(tmp_path, "X", "2026-10-26").exists()


def test_events_from_results_uses_web_backup_only_when_needed():
    w = we.WebEarnings.model_validate(CDNS)
    results = [
        {"symbol": "CDNS", "earnings_date": "2026-10-26"},
        {"symbol": "NOW", "earnings_date": "2026-10-26", "eps_surprise_pct": 5.0},
    ]
    evs = {e["symbol"]: e for e in live.events_from_results(results, {"CDNS": w})}
    assert evs["CDNS"]["features"]["eps_surprise_pct"] == 3.43
    assert evs["CDNS"]["surprise_source"] == "web (Zacks)"
    assert evs["CDNS"]["web"]["usable"] is True
    assert evs["NOW"]["surprise_source"] == "nasdaq"
    assert live.events_from_results(results[:1]) == []  # no backup → skipped
