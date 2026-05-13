"""Earnings call transcripts.

Single source: **SEC EDGAR earnings filings**. Free, no rate limit, ~100%
coverage for any US-listed or ADR-listed company. We pull the most recent
earnings filing — 8-K Item 2.02 for domestic filers, 6-K for foreign private
issuers (NBIS, ASML, NVO, etc.) — and extract either:

  • A transcript-shaped exhibit (Operator/speaker turns + Q&A markers — rare,
    <10% of S&P 500 attach prepared remarks). Returned as `kind="transcript"`.
  • The press release EX-99.1 otherwise. Returned as `kind="press_release"`.
    The transcript agent's prompt recognises this kind and caps confidence
    at 0.5 since Q&A and unscripted tone aren't observable.

History: Alpha Vantage (free 25/day, too tight), API Ninjas (400s on most US
tickers), Financial Modeling Prep (legacy endpoint deprecated 2025-08-31, new
endpoint requires $59/mo Ultimate plan) — all tried and removed. The SEC
press-release path covers 100% of filers at $0 and degrades gracefully via
the confidence cap.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import re
from dataclasses import dataclass
from typing import Literal

import httpx

from .events import SUBMISSIONS_URL, _resolve_cik, _user_agent

TranscriptKind = Literal["transcript", "press_release"]

logger = logging.getLogger(__name__)

SEC_ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"
# Maximum bytes to download for any single exhibit. SEC HTML exhibits are
# normally <500KB but some PDFs / pictures can balloon.
SEC_EXHIBIT_MAX_BYTES = 600_000
# Look back this many days for an earnings filing. Beyond a quarter, the
# transcript is too stale to be a useful signal anyway.
SEC_LOOKBACK_DAYS = 120
# Heuristic thresholds for "this exhibit looks like a transcript, not just
# the press release."
SEC_TRANSCRIPT_MIN_CHARS = 8_000  # press releases are typically 3–6KB
SEC_TRANSCRIPT_MIN_SPEAKERS = 3


@dataclass(frozen=True)
class Transcript:
    symbol: str
    quarter: int  # 1..4
    year: int
    content: str
    # `transcript` = full prepared remarks + Q&A (SEC named-transcript exhibit
    # when a company attaches one — rare).
    # `press_release` = SEC EX-99.1 — the default kind we return from the
    # press-release-only path. Downstream prompt caps confidence on this kind
    # since Q&A tone and analyst pushback aren't observable from a press release.
    kind: TranscriptKind = "transcript"

    @property
    def label(self) -> str:
        return f"Q{self.quarter} {self.year}"


def _quarter_end(quarter: int, year: int) -> _dt.date:
    """Last calendar day of the given quarter (1..4)."""
    end_month = quarter * 3
    if end_month == 12:
        return _dt.date(year, 12, 31)
    return _dt.date(year, end_month + 1, 1) - _dt.timedelta(days=1)


def _is_completed_quarter(quarter: int, year: int, today: _dt.date) -> bool:
    """True when the calendar quarter has fully ended on or before `today`."""
    return _quarter_end(quarter, year) <= today


_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
# Common transcript speaker patterns. Each match counts as one "speaker turn".
# We're looking for evidence that this is a multi-party Q&A, not a press release.
_SPEAKER_PATTERNS = (
    re.compile(r"\bOperator\b\s*[:\-—]?", re.IGNORECASE),
    # "Tim Cook - Chief Executive Officer" / "Tim Cook – CEO"
    re.compile(r"^[A-Z][A-Za-z.\- ]{2,40}\s*[-–—]\s*[A-Z]", re.MULTILINE),
    # "John Smith:" at line start
    re.compile(r"^[A-Z][A-Za-z.\- ]{2,40}:\s*$", re.MULTILINE),
)


def _strip_html(html: str) -> str:
    """Naive HTML → text. Adequate for SEC filings, which use plain tags."""
    no_tags = _HTML_TAG_RE.sub(" ", html)
    # Unescape the handful of entities that show up in SEC filings — anything
    # rarer just gets left literal, which doesn't hurt the LLM downstream.
    no_tags = (
        no_tags.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&#8217;", "’")
        .replace("&#8220;", "“")
        .replace("&#8221;", "”")
        .replace("&#39;", "'")
        .replace("&quot;", '"')
    )
    return _WHITESPACE_RE.sub(" ", no_tags).strip()


def _looks_like_transcript(text: str) -> bool:
    """True when `text` is plausibly a call transcript rather than a press
    release. A press release is shorter and lacks Q&A speaker turns."""
    if len(text) < SEC_TRANSCRIPT_MIN_CHARS:
        return False
    # "Operator" alone is a strong signal — virtually every transcript has it,
    # and press releases never do.
    if _SPEAKER_PATTERNS[0].search(text):
        return True
    # Otherwise require multiple distinct dash- or colon-separated speaker
    # cues. Press releases occasionally have one ("John Smith, CEO, said:")
    # but rarely several.
    cues = 0
    for pat in _SPEAKER_PATTERNS[1:]:
        cues += len(pat.findall(text))
    return cues >= SEC_TRANSCRIPT_MIN_SPEAKERS


def _quarter_from_report_date(report_date: _dt.date) -> tuple[int, int]:
    """Map a fiscal period-end date to (quarter, year). Companies on calendar
    fiscal years all line up to standard quarter ends; off-cycle fiscals get
    bucketed into whichever calendar quarter their period-end falls in, which
    is good enough for tagging purposes."""
    q = (report_date.month - 1) // 3 + 1
    return q, report_date.year


def _earnings_quarter_for_filing(
    form_type: str, filed_date: _dt.date, report_date: str
) -> tuple[int, int]:
    """Resolve (quarter, year) for an earnings filing.

    For 8-K Item 2.02, `reportDate` is the fiscal period-end (e.g. 2026-03-31
    for a Q1 2026 release filed 2026-05-01) — use it directly.

    For 6-K (foreign filers), SEC's `reportDate` is almost always equal to
    the filing date rather than the period-end. Earnings releases typically
    land 30-60 days after the quarter closes, so we subtract ~45 days from
    the filing date to land back inside the reporting quarter. Without this,
    a Q1 2026 NBIS release filed on 2026-05-01 would be labeled Q2 2026."""
    if form_type == "6-K":
        # Filing date back-dated 45 days to land in the reported quarter.
        rep = filed_date - _dt.timedelta(days=45)
        return _quarter_from_report_date(rep)
    try:
        rep = _dt.date.fromisoformat(report_date) if report_date else filed_date
    except ValueError:
        rep = filed_date
    return _quarter_from_report_date(rep)


# EDGAR filings ship with a lot of boilerplate alongside the actual exhibits:
# rendering files generated for the Inline XBRL viewer, the auto-generated
# index pages, and the form itself. Only Ex-99.* and named "transcript"/
# "remarks"/"call" files are realistic transcript candidates.
#
# Exhibit-99 naming varies wildly across filers:
#   • `ex-99.1.htm`, `ex991.htm`              ← standard
#   • `a8-kex991q2202603282026.htm`           ← Apple-style, "ex99" glued mid-name
#   • `exhibit991pressrelease-q2f.htm`        ← Cisco-style, spelled-out "exhibit"
#   • `q12026991.htm`                         ← AMD-style, no "ex" prefix at all
# So the regex matches EITHER (a) an `ex` or `exhibit` token immediately
# followed by `99`, OR (b) standalone `99N` preceded by a non-letter and
# followed by a separator/extension. The lookbehind on (b) avoids matching
# "99" embedded in words like "1999thAnnual".
_EX99_PAT = re.compile(
    r"(?:ex(?:hibit)?[_\-]?99|(?<![a-z])99[12349](?:\.|[_\-]|$))",
    re.IGNORECASE,
)
_NAMED_TRANSCRIPT_PAT = re.compile(r"transcript|remarks|earningscall|call[_\-]?script", re.IGNORECASE)
_JUNK_FILE_PAT = re.compile(
    r"(?:"
    r"^R\d+\.htm$"             # XBRL rendering
    r"|index[_\-]?headers"      # SEC-generated index pages
    r"|-index\."                # filing index page
    r"|^Financial_Report"       # XBRL summary
    r"|FilingSummary"           # FilingSummary.xml/.htm
    r")",
    re.IGNORECASE,
)


def _is_candidate_exhibit(name: str) -> bool:
    if not name.lower().endswith((".htm", ".html", ".txt")):
        return False
    if _JUNK_FILE_PAT.search(name):
        return False
    return bool(_EX99_PAT.search(name) or _NAMED_TRANSCRIPT_PAT.search(name))


_EX991_PAT = re.compile(
    r"(?:ex(?:hibit)?[_\-]?99[_\-.]?1\b|(?<![a-z])991(?:\.|[_\-]|$))",
    re.IGNORECASE,
)


def _exhibit_sort_key(name: str) -> tuple[int, str]:
    """Order: named-transcript files first, then ex-99.2/3/... before ex-99.1
    (since ex-99.1 is almost always the press release). Lower tuple sorts first.

    Matches both `ex-99.1` and `<period>991` (AMD-style) for the EX-99.1 bias."""
    lower = name.lower()
    if _NAMED_TRANSCRIPT_PAT.search(lower):
        return (0, lower)
    if _EX991_PAT.search(lower):
        return (2, lower)
    return (1, lower)


async def _sec_fetch_8k_index(
    cik: str, accession: str, *, client: httpx.AsyncClient
) -> list[dict] | None:
    """Return the list of files in a single 8-K filing."""
    cik_int = int(cik)
    acc_nodash = accession.replace("-", "")
    url = f"{SEC_ARCHIVE_BASE}/{cik_int}/{acc_nodash}/index.json"
    try:
        resp = await client.get(url, headers={"User-Agent": _user_agent()})
        resp.raise_for_status()
        payload = resp.json()
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        logger.warning("SEC filing index fetch failed for %s: %s", accession, e)
        return None
    directory = payload.get("directory") if isinstance(payload, dict) else None
    if not isinstance(directory, dict):
        return None
    items = directory.get("item")
    return items if isinstance(items, list) else None


async def _sec_fetch_exhibit_text(
    cik: str, accession: str, filename: str, *, client: httpx.AsyncClient
) -> str | None:
    """Download a single exhibit and return its plain-text body."""
    cik_int = int(cik)
    acc_nodash = accession.replace("-", "")
    url = f"{SEC_ARCHIVE_BASE}/{cik_int}/{acc_nodash}/{filename}"
    try:
        resp = await client.get(url, headers={"User-Agent": _user_agent()})
        resp.raise_for_status()
    except httpx.HTTPError as e:
        logger.warning("SEC exhibit fetch failed %s/%s: %s", accession, filename, e)
        return None
    body = resp.content[:SEC_EXHIBIT_MAX_BYTES]
    # We can't easily extract text from PDFs without a heavy dependency,
    # so skip them. HTML, HTM, and TXT are the common transcript formats.
    lower = filename.lower()
    if lower.endswith(".pdf"):
        return None
    try:
        decoded = body.decode("utf-8", errors="replace")
    except UnicodeDecodeError:
        return None
    if lower.endswith((".htm", ".html")):
        return _strip_html(decoded)
    return _WHITESPACE_RE.sub(" ", decoded).strip()


# Minimum length for an exhibit to be considered a real press release rather
# than a stub or fragment. Typical earnings press releases are 4-15 KB of
# text after HTML stripping; below 2 KB it's almost certainly boilerplate or
# a navigation artifact.
SEC_PRESS_RELEASE_MIN_CHARS = 2_000


# 6-K filings cover all foreign-filer material events — earnings, acquisitions,
# board changes, regulatory filings, partnership announcements, etc. We accept
# the 6-K as a candidate filing but only stash its exhibit as a press release
# if the content actually looks like earnings. 8-K Item 2.02 is already gated
# upstream so this filter is skipped for 8-K.
_EARNINGS_INDICATOR_PAT = re.compile(
    r"(?:"
    r"reports?\s+(?:first|second|third|fourth)\s+quarter"
    r"|(?:first|second|third|fourth)\s+quarter\s+(?:and\s+full[\s\-]year|\d{4}\s+(?:financial\s+)?results)"
    r"|reports?\s+(?:fiscal|fy)\s+\d{4}"
    r"|quarterly\s+earnings\s+(?:release|results)"
    r"|q[1-4]\s+\d{4}\s+(?:financial\s+)?results"
    r")",
    re.IGNORECASE,
)


def _looks_like_earnings_press_release(text: str) -> bool:
    """True when the exhibit text contains an unambiguous earnings header
    pattern (e.g. "reports fourth quarter", "Q1 2026 financial results"),
    used to filter 6-K filings that are acquisitions, partnerships, or other
    non-earnings material events."""
    return (
        len(text) >= SEC_PRESS_RELEASE_MIN_CHARS
        and _EARNINGS_INDICATOR_PAT.search(text) is not None
    )


async def _sec_extract_transcript(
    symbol: str, *, client: httpx.AsyncClient, today: _dt.date
) -> Transcript | None:
    """Scan recent Item 2.02 8-K filings for a transcript-shaped exhibit.

    Walks the symbol's recent filings list, filters to earnings filings
    within `SEC_LOOKBACK_DAYS`. Accepts both **8-K with Item 2.02** (domestic
    filers) AND **6-K** (foreign private issuers like NBIS / Nebius / ASML
    which never file 8-Ks). For each candidate, downloads each non-trivial
    exhibit and returns the first one that passes `_looks_like_transcript`
    as a full transcript. If no exhibit clears that bar, returns the first
    press-release-shaped exhibit encountered (most recent filing's EX-99.1)
    as `kind="press_release"`. Returns `None` only if the symbol has no
    qualifying filing in the window at all."""
    cik = await _resolve_cik(symbol, client)
    if not cik:
        return None
    try:
        resp = await client.get(
            SUBMISSIONS_URL.format(cik=cik), headers={"User-Agent": _user_agent()}
        )
        resp.raise_for_status()
        payload = resp.json()
    except (httpx.HTTPError, json.JSONDecodeError) as e:
        logger.warning("SEC submissions fetch failed for %s: %s", symbol, e)
        return None

    recent = payload.get("filings", {}).get("recent", {}) or {}
    forms = recent.get("form") or []
    filed = recent.get("filingDate") or []
    items_col = recent.get("items") or []
    accs = recent.get("accessionNumber") or []
    primary = recent.get("primaryDocument") or []
    report_dates = recent.get("reportDate") or []

    cutoff = today - _dt.timedelta(days=SEC_LOOKBACK_DAYS)
    # (filed_date, accession, primary, reportDate, form_type)
    candidates: list[tuple[_dt.date, str, str, str, str]] = []
    for i, form in enumerate(forms):
        if not form:
            continue
        items_str = items_col[i] if i < len(items_col) else ""
        # 8-K with Item 2.02 = domestic-filer earnings release.
        # 6-K = foreign-private-issuer earnings (no Items field is reliably
        # populated for 6-K; the press-release content filter naturally
        # rejects non-earnings 6-Ks like shareholder votes).
        is_earnings_8k = form.startswith("8-K") and "2.02" in items_str
        is_6k = form == "6-K"
        if not (is_earnings_8k or is_6k):
            continue
        try:
            d = _dt.date.fromisoformat(filed[i])
        except (ValueError, IndexError):
            continue
        if d < cutoff:
            continue
        candidates.append(
            (
                d,
                accs[i] if i < len(accs) else "",
                primary[i] if i < len(primary) else "",
                report_dates[i] if i < len(report_dates) else "",
                form,
            )
        )
    # Most recent first.
    candidates.sort(key=lambda c: c[0], reverse=True)

    # Captured during the scan: the first press-release-shaped exhibit we
    # encounter. Falls back to this when no transcript is found.
    press_release_fallback: Transcript | None = None

    for filed_date, accession, _primary, report_date, form_type in candidates:
        if not accession:
            continue
        items = await _sec_fetch_8k_index(cik, accession, client=client)
        if not items:
            continue
        exhibits = [
            it["name"] for it in items
            if isinstance(it, dict)
            and isinstance(it.get("name"), str)
            and _is_candidate_exhibit(it["name"])
        ]
        # Within the candidate pool, prefer files that name themselves as
        # transcript / remarks / call — those are the ones most likely to be
        # the actual transcript when a company attaches one.
        exhibits.sort(key=_exhibit_sort_key)
        q, y = _earnings_quarter_for_filing(form_type, filed_date, report_date)
        for name in exhibits:
            text = await _sec_fetch_exhibit_text(cik, accession, name, client=client)
            if not text:
                continue
            if _looks_like_transcript(text):
                logger.info(
                    "SEC transcript hit: %s %s exhibit %s (filed %s)",
                    symbol, accession, name, filed_date,
                )
                return Transcript(
                    symbol=symbol.upper(), quarter=q, year=y,
                    content=text, kind="transcript",
                )
            # Only stash the FIRST press-release candidate (from the most
            # recent qualifying filing). For 8-K Item 2.02, the item tag
            # already guarantees this is earnings content. For 6-K, we
            # additionally require the text itself to look like earnings
            # (so we skip acquisition / partnership / governance 6-Ks).
            if press_release_fallback is not None:
                continue
            if form_type.startswith("8-K"):
                qualifies = len(text) >= SEC_PRESS_RELEASE_MIN_CHARS
            else:
                qualifies = _looks_like_earnings_press_release(text)
            if qualifies:
                press_release_fallback = Transcript(
                    symbol=symbol.upper(), quarter=q, year=y,
                    content=text, kind="press_release",
                )
                logger.info(
                    "SEC press-release stash: %s %s exhibit %s (filed %s)",
                    symbol, accession, name, filed_date,
                )
    if press_release_fallback is not None:
        logger.info(
            "SEC press-release fallback: %s Q%d %d (no full transcript found)",
            symbol, press_release_fallback.quarter, press_release_fallback.year,
        )
    return press_release_fallback


async def fetch_latest_transcript(
    symbol: str,
    *,
    client: httpx.AsyncClient | None = None,
    today: _dt.date | None = None,
) -> Transcript | None:
    """Fetch the most recent earnings filing for `symbol` from SEC EDGAR.

    Returns a `Transcript` with `kind="press_release"` from the latest 8-K
    Item 2.02 (domestic filers) or earnings 6-K (foreign private issuers),
    or `kind="transcript"` on the rare occasion a company attaches prepared
    remarks as a separate exhibit. Returns `None` only when the symbol has
    no qualifying filing in the last `SEC_LOOKBACK_DAYS` days."""
    today = today or _dt.date.today()
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=30.0)
    try:
        return await _sec_extract_transcript(symbol, client=client, today=today)
    finally:
        if owns_client:
            await client.aclose()


async def fetch_transcript(
    symbol: str,
    quarter: int,
    year: int,
    *,
    client: httpx.AsyncClient | None = None,
    timeout_s: float = 30.0,
    today: _dt.date | None = None,
) -> Transcript | None:
    """Fetch the earnings filing for a specific quarter (used by backtest).

    Returns the SEC press release for `(quarter, year)` if it falls inside
    the standard lookback window. The `_sec_extract_transcript` scanner
    always returns the *most recent* filing, so this function only yields
    a hit when the latest filing's quarter happens to match the requested
    one. For backtests that need older quarters, this returns `None`."""
    today = today or _dt.date.today()
    if not _is_completed_quarter(quarter, year, today):
        return None
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=timeout_s)
    try:
        result = await _sec_extract_transcript(symbol, client=client, today=today)
        if result is None:
            return None
        return result if (result.quarter, result.year) == (quarter, year) else None
    finally:
        if owns_client:
            await client.aclose()
