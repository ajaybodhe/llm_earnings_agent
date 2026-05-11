"""Tests for the transcripts module — focused on the negative cache + walk-back
control flow that consumes Alpha Vantage's free-tier quota."""

from __future__ import annotations

import datetime as dt

import pytest

from llm_earnings_agent.data import transcripts as tx


@pytest.fixture
def neg_cache_file(tmp_path, monkeypatch):
    """Redirect NEGATIVE_CACHE_PATH to a tmp file so tests don't touch the
    real data/cache directory."""
    path = tmp_path / "transcript_negative.json"
    monkeypatch.setattr(tx, "NEGATIVE_CACHE_PATH", path)
    return path


def test_is_cached_empty_within_ttl():
    today = dt.date(2026, 5, 11)
    recorded = today - dt.timedelta(days=tx.NEGATIVE_CACHE_TTL_DAYS - 1)
    cache = {tx._neg_cache_key("MNDY", 1, 2026): recorded.isoformat()}
    assert tx._is_cached_empty(cache, "MNDY", 1, 2026, today) is True


def test_is_cached_empty_expired():
    today = dt.date(2026, 5, 11)
    recorded = today - dt.timedelta(days=tx.NEGATIVE_CACHE_TTL_DAYS)
    cache = {tx._neg_cache_key("MNDY", 1, 2026): recorded.isoformat()}
    assert tx._is_cached_empty(cache, "MNDY", 1, 2026, today) is False


def test_is_cached_empty_missing():
    today = dt.date(2026, 5, 11)
    assert tx._is_cached_empty({}, "MNDY", 1, 2026, today) is False


def test_record_and_persist(neg_cache_file):
    today = dt.date(2026, 5, 11)
    cache = {}
    tx._record_empty(cache, "amd", 4, 2024, today)
    tx._save_negative_cache(cache)
    reloaded = tx._load_negative_cache()
    assert reloaded == {"AMD:Q4:2024": "2026-05-11"}


@pytest.fixture
def mock_sec_none(monkeypatch):
    """Stub the SEC fallback so unit tests don't make real network calls."""
    async def fake(symbol, *, client, today):
        return None
    monkeypatch.setattr(tx, "_sec_extract_transcript", fake)


@pytest.mark.asyncio
async def test_walkback_uses_cache_and_bounds_total_quarters(
    monkeypatch, neg_cache_file, mock_sec_none
):
    """A polluted negative cache should never cause the walk-back to exceed
    `max_lookback_quarters` total quarter-slots — cached skips consume a slot
    too. Otherwise a stale cache would silently extend the walk arbitrarily."""
    today = dt.date(2026, 5, 11)
    # Pre-populate the cache with the first 3 quarters AV would try.
    cache = {}
    for q, y in [(1, 2026), (4, 2025), (3, 2025)]:
        tx._record_empty(cache, "AMD", q, y, today)
    tx._save_negative_cache(cache)

    calls: list[tuple[str, int, int]] = []

    async def fake_av_fetch(symbol, quarter, year, *, client, token):
        calls.append((symbol, quarter, year))
        return None  # always empty

    monkeypatch.setattr(tx, "_alpha_vantage_fetch", fake_av_fetch)
    monkeypatch.setattr(tx, "WALKBACK_REQUEST_DELAY_S", 0)

    result = await tx.fetch_latest_transcript(
        "AMD", max_lookback_quarters=6, today=today
    )

    assert result is None
    # 3 cached skips + 3 fresh API calls = 6 quarter-slots consumed.
    assert calls == [("AMD", 2, 2025), ("AMD", 1, 2025), ("AMD", 4, 2024)]


@pytest.mark.asyncio
async def test_walkback_stops_on_quota_advisory_and_tries_sec(monkeypatch, neg_cache_file):
    """A quota advisory aborts the AV walk-back, then SEC is tried once."""
    today = dt.date(2026, 5, 11)
    av_calls: list[tuple[str, int, int]] = []
    sec_calls: list[str] = []

    async def fake_av_fetch(symbol, quarter, year, *, client, token):
        av_calls.append((symbol, quarter, year))
        raise tx.AlphaVantageQuotaError("daily cap")

    async def fake_sec(symbol, *, client, today):
        sec_calls.append(symbol)
        return None

    monkeypatch.setattr(tx, "_alpha_vantage_fetch", fake_av_fetch)
    monkeypatch.setattr(tx, "_sec_extract_transcript", fake_sec)
    monkeypatch.setattr(tx, "WALKBACK_REQUEST_DELAY_S", 0)

    result = await tx.fetch_latest_transcript(
        "AMD", max_lookback_quarters=6, today=today
    )
    assert result is None
    assert len(av_calls) == 1  # exited after first advisory, didn't try Q4 etc.
    assert sec_calls == ["AMD"]  # SEC fallback ran once


@pytest.mark.asyncio
async def test_sec_fallback_can_return_transcript(monkeypatch, neg_cache_file):
    """When AV is exhausted, a transcript-shaped SEC exhibit is returned."""
    today = dt.date(2026, 5, 11)

    async def fake_av_fetch(symbol, quarter, year, *, client, token):
        return None  # AV consistently empty

    expected = tx.Transcript(symbol="AMD", quarter=1, year=2026, content="...")

    async def fake_sec(symbol, *, client, today):
        return expected

    monkeypatch.setattr(tx, "_alpha_vantage_fetch", fake_av_fetch)
    monkeypatch.setattr(tx, "_sec_extract_transcript", fake_sec)
    monkeypatch.setattr(tx, "WALKBACK_REQUEST_DELAY_S", 0)

    result = await tx.fetch_latest_transcript(
        "AMD", max_lookback_quarters=2, today=today
    )
    assert result is expected


def test_looks_like_transcript_press_release_rejected():
    press_release = (
        "Acme Corp Reports First Quarter 2026 Results. "
        "Acme Corp announced today financial results for its first quarter. "
        "Revenue grew 22% year over year. Net income was $1.2 billion. "
        "John Smith, CEO, said: 'We are pleased with our performance.'"
    ) * 50  # length comfortably above threshold but missing Q&A structure
    assert tx._looks_like_transcript(press_release) is False


def test_looks_like_transcript_with_operator_accepted():
    transcript = (
        "Operator: Good afternoon and welcome to the Acme Corp earnings call. "
        "I will now hand the call over to your host. "
    ) + ("Some content here. " * 800)  # plenty of length
    assert tx._looks_like_transcript(transcript) is True


def test_looks_like_transcript_with_speaker_cues_accepted():
    transcript = (
        "Tim Cook - Chief Executive Officer\n"
        "Thank you. Revenue grew this quarter.\n"
        "Luca Maestri - Chief Financial Officer\n"
        "Margins expanded.\n"
        "Analyst Q - Bank A\n"
        "What about guidance?\n"
    ) + ("Some content here. " * 800)
    assert tx._looks_like_transcript(transcript) is True


def test_strip_html_removes_tags_and_entities():
    html = "<p>Hello&nbsp;<b>world</b> &amp; friends</p>"
    assert tx._strip_html(html) == "Hello world & friends"


def test_is_candidate_exhibit_filters_junk():
    # XBRL rendering files
    assert tx._is_candidate_exhibit("R1.htm") is False
    # Index / header pages
    assert tx._is_candidate_exhibit("0000320193-26-000011-index.html") is False
    assert tx._is_candidate_exhibit("0000320193-26-000011-index-headers.html") is False
    # XBRL Inline financial summaries
    assert tx._is_candidate_exhibit("Financial_Report.xlsx") is False
    # The form itself (Apple-style filename)
    assert tx._is_candidate_exhibit("aapl-20260430.htm") is False
    # Wrong extension
    assert tx._is_candidate_exhibit("transcript.pdf") is False


def test_is_candidate_exhibit_accepts_real_exhibits():
    assert tx._is_candidate_exhibit("a8-kex991q2202603282026.htm") is True  # AAPL pattern
    assert tx._is_candidate_exhibit("ex99-1.htm") is True
    assert tx._is_candidate_exhibit("ex-99.2.htm") is True
    assert tx._is_candidate_exhibit("q1-2026-call-transcript.htm") is True
    assert tx._is_candidate_exhibit("prepared_remarks.txt") is True


def test_exhibit_sort_key_orders_transcripts_first():
    names = ["ex-99.1.htm", "ex-99.2.htm", "q1_transcript.htm"]
    names.sort(key=tx._exhibit_sort_key)
    # named-transcript first, then ex-99.2 (likely transcript), then ex-99.1 (press release)
    assert names == ["q1_transcript.htm", "ex-99.2.htm", "ex-99.1.htm"]


def test_quarter_from_report_date():
    assert tx._quarter_from_report_date(dt.date(2026, 3, 31)) == (1, 2026)
    assert tx._quarter_from_report_date(dt.date(2026, 6, 30)) == (2, 2026)
    assert tx._quarter_from_report_date(dt.date(2025, 12, 31)) == (4, 2025)
