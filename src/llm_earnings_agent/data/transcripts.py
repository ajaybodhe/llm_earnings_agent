"""Earnings call transcripts.

Two-stage fallback chain:

1. **Financial Modeling Prep** — primary. Paid endpoint.
       https://financialmodelingprep.com/stable/earning-call-transcript?symbol=&quarter=&year=&apikey=
       (The legacy /api/v3 endpoint was deprecated Aug 31, 2025 and now
       returns a "Legacy Endpoint" error to non-grandfathered keys.)
       Walk-back from the most recent completed calendar quarter up to
       `max_lookback_quarters`, throttled to a polite ~0.25s between calls.
       Negative results (HTTP 200 with empty array) are persisted to a disk
       cache (`NEGATIVE_CACHE_PATH`) with a 7-day TTL so repeat runs don't
       re-walk known-empty quarters.

       Plan/auth errors (401/402/403 or "Error Message" body indicating
       subscription gating) raise `FMPPlanError`, abort the walk-back, and
       fall through to SEC EDGAR. The same key works once the FMP plan is
       upgraded to include the transcript endpoint — no code change needed.

2. **SEC EDGAR earnings filings** — fallback. Free, no rate limit. Accepts
   both 8-K Item 2.02 (domestic filers) and 6-K (foreign private issuers).
       Two-tier extraction within a single scan:
         a. Look for transcript-shaped exhibits (long + multi-speaker + Q&A
            markers — EX-99.2 / named-"transcript" / etc.). Empirically <10%
            of S&P 500 attach prepared remarks; when present, returned as
            `Transcript(kind="transcript")`.
         b. If no transcript exhibit is found in the lookback window, return
            the most recent earnings press release (EX-99.1 from the most
            recent Item 2.02 8-K) as `Transcript(kind="press_release")`. The
            transcript agent recognises this kind and caps confidence
            accordingly — it has revenue/EPS/guidance text but no Q&A tone.
       100% of S&P 500 attach a press release, so this gives the transcript
       agent *some* signal for every ticker rather than null.

History: Alpha Vantage (free 25/day) and API Ninjas / FMP-free were all
tried and removed — quota too tight or endpoints behind paid gates anyway.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

TranscriptKind = Literal["transcript", "press_release"]

import httpx

from .events import SUBMISSIONS_URL, _resolve_cik, _user_agent

logger = logging.getLogger(__name__)

FMP_BASE_URL = "https://financialmodelingprep.com/stable/earning-call-transcript"
FMP_DEFAULT_KEY = "vNrb9krkhGhldJ8Z1Ij21G5239wwsAUU"

# SEC EDGAR archives use the bare integer CIK and the dashless accession in
# the path. `index.json` lists files in the filing.
SEC_FILING_INDEX_URL = (
    "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"  # unused, kept for reference
)
SEC_ARCHIVE_BASE = "https://www.sec.gov/Archives/edgar/data"
# Maximum bytes to download for any single exhibit. SEC HTML exhibits are
# normally <500KB but some PDFs / pictures can balloon.
SEC_EXHIBIT_MAX_BYTES = 600_000
# Look back this many days for an Item 2.02 8-K. Beyond a quarter, the
# transcript is too stale to be a useful fallback.
SEC_LOOKBACK_DAYS = 120
# Heuristic thresholds for "this exhibit looks like a transcript, not just
# the press release."
SEC_TRANSCRIPT_MIN_CHARS = 8_000  # press releases are typically 3–6KB
SEC_TRANSCRIPT_MIN_SPEAKERS = 3

# FMP's paid tiers allow hundreds of req/min, but a small inter-call gap is
# good citizenship and keeps the walk-back from hammering when a ticker has
# many empty quarters in a row.
WALKBACK_REQUEST_DELAY_S = 0.25

# Disk-backed cache of (symbol, quarter, year) tuples that FMP has confirmed
# empty. Walk-back skips any quarter in here, which avoids paying for repeat
# negative lookups across runs. Entries expire after NEGATIVE_CACHE_TTL_DAYS
# so a delayed-posting transcript can still be found.
NEGATIVE_CACHE_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "cache" / "transcript_negative.json"
)
NEGATIVE_CACHE_TTL_DAYS = 7


@dataclass(frozen=True)
class Transcript:
    symbol: str
    quarter: int  # 1..4
    year: int
    content: str
    # `transcript` = full prepared remarks + Q&A (AV, FMP, SEC named-transcript exhibit).
    # `press_release` = SEC EX-99.1 fallback when no full transcript is available.
    # Downstream prompt caps confidence on `press_release` since it lacks Q&A
    # tone and analyst pushback that drive most of the agent's signal.
    kind: TranscriptKind = "transcript"

    @property
    def label(self) -> str:
        return f"Q{self.quarter} {self.year}"


class FMPPlanError(RuntimeError):
    """Raised when FMP returns a plan/auth error (401/402/403, or HTTP 200
    with an "Error Message" body indicating subscription gating). Subsequent
    calls in the same run will hit the same response, so callers should stop
    walking back and surface the condition once."""


def _quarter_end(quarter: int, year: int) -> _dt.date:
    """Last calendar day of the given quarter (1..4)."""
    end_month = quarter * 3
    if end_month == 12:
        return _dt.date(year, 12, 31)
    return _dt.date(year, end_month + 1, 1) - _dt.timedelta(days=1)


def _is_completed_quarter(quarter: int, year: int, today: _dt.date) -> bool:
    """True when the calendar quarter has fully ended on or before `today`."""
    return _quarter_end(quarter, year) <= today


_FMP_PLAN_MSG_PAT = re.compile(
    r"legacy|subscription|upgrade|restricted endpoint|premium|not available under",
    re.IGNORECASE,
)


async def _fmp_fetch(
    symbol: str, quarter: int, year: int, *, client: httpx.AsyncClient, token: str
) -> Transcript | None:
    """Fetch from FMP's `/stable/earning-call-transcript`. Response shape:
        [{"symbol":"AMD","quarter":1,"year":2026,"date":"2026-05-05 17:00:00","content":"Operator..."}]

    Raises `FMPPlanError` when the key/plan can't access the endpoint
    (401/402/403, or HTTP 200 with an "Error Message" body) — the walk-back
    should bail. Returns None for any other miss (no data for this quarter,
    network error) so the walk-back can try an older quarter."""
    params = {
        "symbol": symbol.upper(),
        "quarter": quarter,
        "year": year,
        "apikey": token,
    }
    try:
        resp = await client.get(FMP_BASE_URL, params=params)
        if resp.status_code in (401, 402, 403):
            raise FMPPlanError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        payload = resp.json()
    except FMPPlanError:
        raise
    except httpx.HTTPStatusError as e:
        logger.warning(
            "FMP transcript %s Q%d %d returned %d",
            symbol, quarter, year, e.response.status_code,
        )
        return None
    except httpx.HTTPError as e:
        logger.warning(
            "FMP transcript %s Q%d %d network error: %s", symbol, quarter, year, e
        )
        return None
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(
            "FMP transcript %s Q%d %d response was not valid JSON: %s", symbol, quarter, year, e
        )
        return None

    # FMP sometimes returns 200 with a plan/auth error in the body rather than a 4xx.
    if isinstance(payload, dict):
        err = payload.get("Error Message") or payload.get("error") or ""
        if isinstance(err, str) and _FMP_PLAN_MSG_PAT.search(err):
            raise FMPPlanError(err)
        return None
    if not isinstance(payload, list) or not payload:
        return None
    entry = payload[0]
    if not isinstance(entry, dict):
        return None
    content = (entry.get("content") or "").strip()
    if not content:
        return None
    # Trust the entry's own quarter/year when present; fall back to the request params.
    try:
        q = int(entry.get("quarter", quarter))
        y = int(entry.get("year", year))
    except (TypeError, ValueError):
        q, y = quarter, year
    return Transcript(symbol=symbol.upper(), quarter=q, year=y, content=content)


def _neg_cache_key(symbol: str, quarter: int, year: int) -> str:
    return f"{symbol.upper()}:Q{quarter}:{year}"


def _load_negative_cache() -> dict[str, str]:
    if not NEGATIVE_CACHE_PATH.exists():
        return {}
    try:
        data = json.loads(NEGATIVE_CACHE_PATH.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _save_negative_cache(cache: dict[str, str]) -> None:
    NEGATIVE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = NEGATIVE_CACHE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, indent=0, separators=(",", ":"), sort_keys=True))
    tmp.replace(NEGATIVE_CACHE_PATH)


def _is_cached_empty(
    cache: dict[str, str], symbol: str, quarter: int, year: int, today: _dt.date
) -> bool:
    key = _neg_cache_key(symbol, quarter, year)
    recorded_str = cache.get(key)
    if not recorded_str:
        return False
    try:
        recorded = _dt.date.fromisoformat(recorded_str)
    except ValueError:
        return False
    return (today - recorded).days < NEGATIVE_CACHE_TTL_DAYS


def _record_empty(
    cache: dict[str, str], symbol: str, quarter: int, year: int, today: _dt.date
) -> None:
    cache[_neg_cache_key(symbol, quarter, year)] = today.isoformat()


def _resolve_token() -> str:
    token = os.environ.get("FMP_API_KEY", "") or FMP_DEFAULT_KEY
    if not token:
        raise RuntimeError("FMP_API_KEY env var must be set")
    return token


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
#   • `ex-99.1.htm`, `ex991.htm`            ← standard
#   • `a8-kex991q2202603282026.htm`         ← Apple-style, "ex99" glued mid-name
#   • `q12026991.htm`                       ← AMD-style, no "ex" prefix at all
# So the regex matches EITHER an explicit `ex99` token OR `99N` immediately
# followed by a separator/extension (.htm, _, -). The trailing class avoids
# spurious matches on years like 1999 inside random text.
_EX99_PAT = re.compile(r"(?:ex[_\-]?99|99[12349](?:\.|[_\-]|$))", re.IGNORECASE)
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
    r"(?:ex[_\-]?99[_\-.]?1\b|991(?:\.|[_\-]|$))", re.IGNORECASE
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


async def fetch_transcript(
    symbol: str,
    quarter: int,
    year: int,
    *,
    client: httpx.AsyncClient | None = None,
    timeout_s: float = 30.0,
    today: _dt.date | None = None,
) -> Transcript | None:
    """Fetch a single quarter's transcript via FMP. Returns None if not available.

    Future / in-progress quarters short-circuit to None — FMP has no
    transcript for a quarter that hasn't ended yet."""
    today = today or _dt.date.today()
    if not _is_completed_quarter(quarter, year, today):
        return None

    token = _resolve_token()
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=timeout_s)
    try:
        try:
            return await _fmp_fetch(symbol, quarter, year, client=client, token=token)
        except FMPPlanError as e:
            logger.warning(
                "FMP transcript %s Q%d %d: plan/auth error — %s", symbol, quarter, year, e
            )
            return None
    finally:
        if owns_client:
            await client.aclose()


async def fetch_latest_transcript(
    symbol: str,
    *,
    client: httpx.AsyncClient | None = None,
    max_lookback_quarters: int = 6,
    today: _dt.date | None = None,
) -> Transcript | None:
    """Walk back from the most recent completed calendar quarter until we find
    a transcript. Returns None when nothing is available in the window or when
    FMP's plan/auth gating fires (logged once, falls through to SEC EDGAR)."""
    today = today or _dt.date.today()
    q = (today.month - 1) // 3 + 1
    y = today.year
    while not _is_completed_quarter(q, y, today):
        q -= 1
        if q == 0:
            q = 4
            y -= 1

    token = _resolve_token()
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=30.0)
    neg_cache = _load_negative_cache()
    neg_cache_dirty = False
    api_calls_made = 0
    fmp_plan_hit = False
    try:
        # `max_lookback_quarters` bounds the total quarters we walk back. Cached
        # empties consume a slot too — otherwise a polluted cache could make us
        # walk back arbitrarily far looking for an uncached quarter.
        for _ in range(max_lookback_quarters):
            if _is_cached_empty(neg_cache, symbol, q, y, today):
                logger.debug(
                    "FMP transcript %s Q%d %d: skipping (cached empty)", symbol, q, y
                )
                q -= 1
                if q == 0:
                    q = 4
                    y -= 1
                continue
            if api_calls_made > 0:
                await asyncio.sleep(WALKBACK_REQUEST_DELAY_S)
            api_calls_made += 1
            try:
                t = await _fmp_fetch(symbol, q, y, client=client, token=token)
            except FMPPlanError as e:
                logger.warning(
                    "FMP transcript %s: plan/auth error on Q%d %d — "
                    "falling back to SEC EDGAR. %s",
                    symbol, q, y, e,
                )
                fmp_plan_hit = True
                break
            if t is not None:
                return t
            _record_empty(neg_cache, symbol, q, y, today)
            neg_cache_dirty = True
            q -= 1
            if q == 0:
                q = 4
                y -= 1

        # SEC EDGAR fallback: try once, regardless of whether FMP exhausted via
        # walk-back or via plan-error advisory. Free and rate-limit-free;
        # coverage is partial so this often returns None too, but when it hits
        # it's complementary to FMP.
        logger.info(
            "FMP transcript %s: %s, trying SEC EDGAR fallback",
            symbol, "plan/auth error" if fmp_plan_hit else "no transcript in walk-back window",
        )
        sec_t = await _sec_extract_transcript(symbol, client=client, today=today)
        if sec_t is not None:
            return sec_t
        return None
    finally:
        if neg_cache_dirty:
            _save_negative_cache(neg_cache)
        if owns_client:
            await client.aclose()
