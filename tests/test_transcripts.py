"""Tests for the transcripts module — focused on the negative cache + walk-back
control flow that drives FMP's transcript endpoint, plus SEC EDGAR fallback."""

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
    # Pre-populate the cache with the first 3 quarters FMP would try.
    cache = {}
    for q, y in [(1, 2026), (4, 2025), (3, 2025)]:
        tx._record_empty(cache, "AMD", q, y, today)
    tx._save_negative_cache(cache)

    calls: list[tuple[str, int, int]] = []

    async def fake_fmp_fetch(symbol, quarter, year, *, client, token):
        calls.append((symbol, quarter, year))
        return None  # always empty

    monkeypatch.setattr(tx, "_fmp_fetch", fake_fmp_fetch)
    monkeypatch.setattr(tx, "WALKBACK_REQUEST_DELAY_S", 0)

    result = await tx.fetch_latest_transcript(
        "AMD", max_lookback_quarters=6, today=today
    )

    assert result is None
    # 3 cached skips + 3 fresh API calls = 6 quarter-slots consumed.
    assert calls == [("AMD", 2, 2025), ("AMD", 1, 2025), ("AMD", 4, 2024)]


@pytest.mark.asyncio
async def test_walkback_stops_on_plan_error_and_tries_sec(monkeypatch, neg_cache_file):
    """A plan/auth error aborts the FMP walk-back, then SEC is tried once."""
    today = dt.date(2026, 5, 11)
    fmp_calls: list[tuple[str, int, int]] = []
    sec_calls: list[str] = []

    async def fake_fmp_fetch(symbol, quarter, year, *, client, token):
        fmp_calls.append((symbol, quarter, year))
        raise tx.FMPPlanError("HTTP 402: Restricted Endpoint")

    async def fake_sec(symbol, *, client, today):
        sec_calls.append(symbol)
        return None

    monkeypatch.setattr(tx, "_fmp_fetch", fake_fmp_fetch)
    monkeypatch.setattr(tx, "_sec_extract_transcript", fake_sec)
    monkeypatch.setattr(tx, "WALKBACK_REQUEST_DELAY_S", 0)

    result = await tx.fetch_latest_transcript(
        "AMD", max_lookback_quarters=6, today=today
    )
    assert result is None
    assert len(fmp_calls) == 1  # exited after first plan error, didn't try Q4 etc.
    assert sec_calls == ["AMD"]  # SEC fallback ran once


@pytest.mark.asyncio
async def test_sec_fallback_can_return_transcript(monkeypatch, neg_cache_file):
    """When FMP is exhausted, a transcript-shaped SEC exhibit is returned."""
    today = dt.date(2026, 5, 11)

    async def fake_fmp_fetch(symbol, quarter, year, *, client, token):
        return None  # FMP consistently empty

    expected = tx.Transcript(symbol="AMD", quarter=1, year=2026, content="...")

    async def fake_sec(symbol, *, client, today):
        return expected

    monkeypatch.setattr(tx, "_fmp_fetch", fake_fmp_fetch)
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
    # AMD-style: exhibit 99.1 without the "ex" prefix.
    assert tx._is_candidate_exhibit("q12026991.htm") is True


def test_exhibit_sort_key_orders_transcripts_first():
    names = ["ex-99.1.htm", "ex-99.2.htm", "q1_transcript.htm"]
    names.sort(key=tx._exhibit_sort_key)
    # named-transcript first, then ex-99.2 (likely transcript), then ex-99.1 (press release)
    assert names == ["q1_transcript.htm", "ex-99.2.htm", "ex-99.1.htm"]


def test_exhibit_sort_key_biases_amd_style_press_release_last():
    """AMD names its press release `q1<year>991.htm` (no "ex" prefix). The
    sort key must still bias it after ex-99.2+ so a real transcript exhibit
    in the same filing wins."""
    names = ["q12026991.htm", "ex-99.2.htm"]
    names.sort(key=tx._exhibit_sort_key)
    assert names == ["ex-99.2.htm", "q12026991.htm"]


def test_quarter_from_report_date():
    assert tx._quarter_from_report_date(dt.date(2026, 3, 31)) == (1, 2026)
    assert tx._quarter_from_report_date(dt.date(2026, 6, 30)) == (2, 2026)
    assert tx._quarter_from_report_date(dt.date(2025, 12, 31)) == (4, 2025)


def test_earnings_quarter_8k_uses_report_date():
    """8-K Item 2.02's reportDate is the fiscal period-end and should win
    over the filing date even when the two are weeks apart."""
    # Q1 2026 earnings filed 2026-05-01 with reportDate=2026-03-31 → Q1 2026.
    assert tx._earnings_quarter_for_filing(
        "8-K", dt.date(2026, 5, 1), "2026-03-31"
    ) == (1, 2026)


def test_earnings_indicator_accepts_real_earnings_headers():
    """The earnings-content filter must recognise the common header patterns
    real earnings press releases use, across filer styles."""
    assert tx._looks_like_earnings_press_release(
        "Nebius reports fourth quarter and full-year 2025 financial results. " * 50
    ) is True
    assert tx._looks_like_earnings_press_release(
        "AMD Reports First Quarter 2026 Financial Results. " * 50
    ) is True
    assert tx._looks_like_earnings_press_release(
        "Acme Corp announces Q3 2026 financial results today. " * 50
    ) is True


def test_earnings_indicator_rejects_non_earnings_6k_releases():
    """Acquisitions, partnerships, board changes — all show up as 6-Ks with
    EX-99.1 exhibits but must not be mistaken for earnings press releases."""
    # NBIS Tavily acquisition press release (2026-02-10 6-K).
    assert tx._looks_like_earnings_press_release(
        "Nebius announces agreement to acquire Tavily to add agentic search "
        "to its AI cloud platform. The deal closes subject to regulatory approval. " * 50
    ) is False
    # NBIS NVIDIA partnership press release (2026-03-11 6-K).
    assert tx._looks_like_earnings_press_release(
        "NVIDIA and Nebius Partner to Scale Full-Stack AI Cloud. " * 50
    ) is False
    # Generic governance announcement.
    assert tx._looks_like_earnings_press_release(
        "Company appoints new chief operating officer effective immediately. " * 50
    ) is False


def test_earnings_indicator_rejects_short_text():
    """Even with earnings keywords, anything under 2KB is a fragment."""
    short = "Reports first quarter 2026 financial results."
    assert tx._looks_like_earnings_press_release(short) is False


def test_earnings_quarter_6k_back_dates_filing_for_foreign_filers():
    """6-K reportDate equals the filing date — back-date 45d to land in the
    quarter the earnings are actually reporting."""
    # NBIS files Q1 2026 results on 2026-05-01 (reportDate=2026-05-01 same as filed).
    # 2026-05-01 minus 45d = 2026-03-17 → Q1 2026.
    assert tx._earnings_quarter_for_filing(
        "6-K", dt.date(2026, 5, 1), "2026-05-01"
    ) == (1, 2026)
    # Q4 2025 results filed early Feb 2026 → back-dated to mid-Dec 2025 → Q4 2025.
    assert tx._earnings_quarter_for_filing(
        "6-K", dt.date(2026, 2, 5), "2026-02-05"
    ) == (4, 2025)
    # Q2 2026 results filed Aug 1 → back-dated to mid-June → Q2 2026.
    assert tx._earnings_quarter_for_filing(
        "6-K", dt.date(2026, 8, 1), "2026-08-01"
    ) == (2, 2026)


@pytest.mark.asyncio
async def test_sec_scanner_skips_non_earnings_6k_for_real_one(monkeypatch):
    """The NBIS scenario: most recent 6-K is an acquisition announcement,
    older 6-K is the actual earnings report. Scanner must skip the acquisition
    (no earnings indicator) and stash the earnings press release."""
    today = dt.date(2026, 5, 12)
    acquisition_body = (
        "Nebius announces agreement to acquire Eigen AI for $643M cash and stock. "
        "The deal expands our inference platform and US footprint. " * 80
    )
    earnings_body = (
        "Nebius reports fourth quarter and full-year 2025 financial results. "
        "Revenue grew 466% year-over-year to a record level. " * 80
    )

    async def fake_resolve_cik(symbol, client): return "0001513845"

    async def fake_get(url, *, headers=None):
        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                return {
                    "filings": {
                        "recent": {
                            # Most-recent 6-K = acquisition; older 6-K = earnings.
                            "form": ["6-K", "6-K"],
                            "filingDate": ["2026-05-01", "2026-02-12"],
                            "items": ["", ""],
                            "accessionNumber": ["acq-acc", "earn-acc"],
                            "primaryDocument": ["a.htm", "b.htm"],
                            "reportDate": ["2026-05-01", "2026-02-12"],
                        }
                    }
                }
        return R()

    async def fake_index_fetch(cik, accession, *, client):
        return [{"name": "ex99-1.htm"}]  # both have a standard EX-99.1

    async def fake_exhibit_text(cik, accession, filename, *, client):
        return acquisition_body if accession == "acq-acc" else earnings_body

    monkeypatch.setattr(tx, "_resolve_cik", fake_resolve_cik)
    monkeypatch.setattr(tx, "_sec_fetch_8k_index", fake_index_fetch)
    monkeypatch.setattr(tx, "_sec_fetch_exhibit_text", fake_exhibit_text)

    class FakeClient:
        async def get(self, url, *, headers=None):
            return await fake_get(url, headers=headers)

    result = await tx._sec_extract_transcript("NBIS", client=FakeClient(), today=today)
    assert result is not None
    assert result.kind == "press_release"
    # Earnings release filed 2026-02-12, 6-K → back-dated to late Dec 2025 → Q4 2025.
    assert (result.quarter, result.year) == (4, 2025)
    assert "Nebius reports fourth quarter" in result.content


@pytest.mark.asyncio
async def test_sec_scanner_picks_up_6k_for_foreign_filer(monkeypatch):
    """A foreign private issuer (NBIS-style) has no 8-Ks. The scanner must
    pick up their 6-K instead and label the quarter correctly via the
    filing-date back-dating heuristic."""
    today = dt.date(2026, 5, 12)
    press_release_body = (
        "Nebius reports fourth quarter and full-year 2025 financial results. "
        "Revenue grew to record levels driven by AI infrastructure demand. "
    ) * 100  # ~14KB with the earnings header pattern

    async def fake_resolve_cik(symbol, client): return "0001513845"

    async def fake_get(url, *, headers=None):
        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                # Mix: an unrelated old 8-K (won't match), and a real 6-K from May 1.
                return {
                    "filings": {
                        "recent": {
                            "form": ["6-K", "6-K"],
                            "filingDate": ["2026-05-01", "2026-03-20"],
                            "items": ["", ""],
                            "accessionNumber": ["0001104659-26-053464", "0001104659-26-032735"],
                            "primaryDocument": ["tm2613296d1_6k.htm", "other.htm"],
                            "reportDate": ["2026-05-01", "2026-03-20"],
                        }
                    }
                }
        return R()

    async def fake_index_fetch(cik, accession, *, client):
        if accession == "0001104659-26-053464":  # the earnings 6-K
            return [{"name": "tm2613296d1_ex99-1.htm"}]
        return [{"name": "shareholder_vote.htm"}]  # non-earnings 6-K

    async def fake_exhibit_text(cik, accession, filename, *, client):
        if "ex99-1" in filename:
            return press_release_body
        return "Annual general meeting notice. Vote on board members."  # too short

    monkeypatch.setattr(tx, "_resolve_cik", fake_resolve_cik)
    monkeypatch.setattr(tx, "_sec_fetch_8k_index", fake_index_fetch)
    monkeypatch.setattr(tx, "_sec_fetch_exhibit_text", fake_exhibit_text)

    class FakeClient:
        async def get(self, url, *, headers=None):
            return await fake_get(url, headers=headers)

    result = await tx._sec_extract_transcript("NBIS", client=FakeClient(), today=today)
    assert result is not None
    assert result.kind == "press_release"
    # Filing dated 2026-05-01, 6-K → back-dated to mid-March → Q1 2026.
    assert result.quarter == 1
    assert result.year == 2026
    assert "Nebius" in result.content


def test_transcript_kind_default_is_transcript():
    """Existing call sites that construct Transcript without `kind` should
    continue to default to a full transcript — kind is opt-in for press releases."""
    t = tx.Transcript(symbol="AMD", quarter=1, year=2026, content="x")
    assert t.kind == "transcript"


@pytest.mark.asyncio
async def test_sec_returns_press_release_when_no_transcript_found(monkeypatch):
    """When the only candidate exhibit in a recent 8-K is press-release-shaped
    (long enough but fails the transcript heuristic), return it with
    kind="press_release" rather than None."""
    today = dt.date(2026, 5, 11)
    press_release_body = (
        "Acme Corp Reports First Quarter 2026 Results. "
        "Acme Corp announced today financial results for its first quarter. "
        "Revenue grew 22% year over year. Net income was $1.2 billion. "
    ) * 100  # ~13KB — comfortably above 2KB press-release floor

    async def fake_resolve_cik(symbol, client): return "0000320193"

    async def fake_submissions_get(url, *, headers):
        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                return {
                    "filings": {
                        "recent": {
                            "form": ["8-K"],
                            "filingDate": ["2026-04-30"],
                            "items": ["2.02,9.01"],
                            "accessionNumber": ["0000320193-26-000011"],
                            "primaryDocument": ["aapl-20260430.htm"],
                            "reportDate": ["2026-04-29"],
                        }
                    }
                }
        return R()

    async def fake_index_fetch(cik, accession, *, client):
        return [{"name": "ex-99.1.htm"}]  # only the press release exhibit

    async def fake_exhibit_text(cik, accession, filename, *, client):
        return press_release_body

    monkeypatch.setattr(tx, "_resolve_cik", fake_resolve_cik)
    monkeypatch.setattr(tx, "_sec_fetch_8k_index", fake_index_fetch)
    monkeypatch.setattr(tx, "_sec_fetch_exhibit_text", fake_exhibit_text)

    class FakeClient:
        async def get(self, url, *, headers=None):
            return await fake_submissions_get(url, headers=headers)

    result = await tx._sec_extract_transcript("AAPL", client=FakeClient(), today=today)
    assert result is not None
    assert result.kind == "press_release"
    assert result.quarter == 2  # report_date 2026-04-29 → Q2 2026
    assert result.year == 2026
    assert "Acme Corp Reports" in result.content


@pytest.mark.asyncio
async def test_sec_prefers_transcript_over_press_release_in_same_filing(monkeypatch):
    """When a single filing has both a real transcript exhibit AND a press
    release, the transcript should win — press_release_fallback only fires
    when no transcript-shaped exhibit is found across all candidates."""
    today = dt.date(2026, 5, 11)
    transcript_body = (
        "Operator: Good afternoon and welcome to Acme Corp's Q1 2026 call. "
    ) + ("Lisa Su, CEO: Revenue grew this quarter. " * 600)  # large + Operator cue
    press_release_body = "Acme Reports Q1 Results. Revenue grew. " * 200

    async def fake_resolve_cik(symbol, client): return "0000320193"

    async def fake_get(url, *, headers=None):
        class R:
            status_code = 200
            def raise_for_status(self): pass
            def json(self):
                return {
                    "filings": {
                        "recent": {
                            "form": ["8-K"],
                            "filingDate": ["2026-04-30"],
                            "items": ["2.02"],
                            "accessionNumber": ["0000320193-26-000011"],
                            "primaryDocument": ["x.htm"],
                            "reportDate": ["2026-03-31"],
                        }
                    }
                }
        return R()

    async def fake_index_fetch(cik, accession, *, client):
        return [{"name": "ex-99.1.htm"}, {"name": "q1_transcript.htm"}]

    async def fake_exhibit_text(cik, accession, filename, *, client):
        if "transcript" in filename:
            return transcript_body
        return press_release_body

    monkeypatch.setattr(tx, "_resolve_cik", fake_resolve_cik)
    monkeypatch.setattr(tx, "_sec_fetch_8k_index", fake_index_fetch)
    monkeypatch.setattr(tx, "_sec_fetch_exhibit_text", fake_exhibit_text)

    class FakeClient:
        async def get(self, url, *, headers=None):
            return await fake_get(url, headers=headers)

    result = await tx._sec_extract_transcript("AAPL", client=FakeClient(), today=today)
    assert result is not None
    assert result.kind == "transcript"
    assert "Operator:" in result.content
