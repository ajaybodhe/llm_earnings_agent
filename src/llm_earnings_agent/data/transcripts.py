"""Earnings call transcripts.

Two-stage fallback chain:

1. **Alpha Vantage** — primary. Free tier, 25 req/day at 1 req/sec.
       https://www.alphavantage.co/query?function=EARNINGS_CALL_TRANSCRIPT&symbol=&quarter=&apikey=
       Quarter format `YYYYQN`. Walk-back from the most recent completed
       calendar quarter up to `max_lookback_quarters`, throttled to ~1 req/sec.
       A quota advisory aborts the walk-back immediately. Negative results are
       persisted to a disk cache (`NEGATIVE_CACHE_PATH`) with a 7-day TTL so
       repeat runs don't burn quota on quarters we already know are empty.

2. **SEC EDGAR 8-K Item 2.02 exhibits** — fallback. Free, no rate limit.
       Companies attach earnings-release exhibits to their results 8-K. Some
       (mostly mid/large caps) attach the analyst-call prepared remarks as a
       separate exhibit (EX-99.2 or similar). We fetch the most recent 8-K
       with Item 2.02, list its exhibits, and accept any HTML/text exhibit
       that looks transcript-shaped (long + multi-speaker + Q&A markers).
       Coverage is partial (~30-40% of large caps) but the data is free and
       complementary to AV — different misses, different hits.

History: FMP's free tier 403's the transcript endpoint and API Ninjas 400's
on most US tickers — both were removed as they contributed only log noise.
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

import httpx

from .events import SUBMISSIONS_URL, _resolve_cik, _user_agent

logger = logging.getLogger(__name__)

ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"
ALPHA_VANTAGE_DEFAULT_KEY = "VSDK9FRT62RNF27R"

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

# AV's free-tier hint is "1 request per second". Wait a hair more between
# walk-back iterations so we don't accidentally trip the burst limit.
WALKBACK_REQUEST_DELAY_S = 1.1

# Disk-backed cache of (symbol, quarter, year) tuples that Alpha Vantage has
# confirmed empty. Walk-back skips any quarter in here, which conserves the
# 25/day free-tier quota across repeated runs. Entries expire after
# NEGATIVE_CACHE_TTL_DAYS so a delayed-posting transcript can still be found.
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

    @property
    def label(self) -> str:
        return f"Q{self.quarter} {self.year}"


class AlphaVantageQuotaError(RuntimeError):
    """Raised when Alpha Vantage returns an advisory ('Information' / 'Note').
    Subsequent calls in the same minute / day will hit the same response, so
    callers should stop walking back and surface the condition once."""


def _quarter_end(quarter: int, year: int) -> _dt.date:
    """Last calendar day of the given quarter (1..4)."""
    end_month = quarter * 3
    if end_month == 12:
        return _dt.date(year, 12, 31)
    return _dt.date(year, end_month + 1, 1) - _dt.timedelta(days=1)


def _is_completed_quarter(quarter: int, year: int, today: _dt.date) -> bool:
    """True when the calendar quarter has fully ended on or before `today`."""
    return _quarter_end(quarter, year) <= today


async def _alpha_vantage_fetch(
    symbol: str, quarter: int, year: int, *, client: httpx.AsyncClient, token: str
) -> Transcript | None:
    """Fetch from Alpha Vantage. Response shape:
        {"symbol": "...", "quarter": "2026Q1",
         "transcript": [{"speaker": "...", "title": "...", "content": "...", ...}, ...]}

    Raises `AlphaVantageQuotaError` on advisory responses (rate-limit / daily
    cap) — the walk-back should bail. Returns None for any other failure mode
    (no data for this quarter, 4xx, network error) so the walk-back can try an
    older quarter."""
    params = {
        "function": "EARNINGS_CALL_TRANSCRIPT",
        "symbol": symbol.upper(),
        "quarter": f"{year}Q{quarter}",
        "apikey": token,
    }
    try:
        resp = await client.get(ALPHA_VANTAGE_BASE_URL, params=params)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        payload = resp.json()
    except httpx.HTTPStatusError as e:
        logger.warning(
            "AlphaVantage transcript %s Q%d %d returned %d",
            symbol, quarter, year, e.response.status_code,
        )
        return None
    except httpx.HTTPError as e:
        logger.warning(
            "AlphaVantage transcript %s Q%d %d network error: %s", symbol, quarter, year, e
        )
        return None

    if not isinstance(payload, dict):
        return None
    if "Information" in payload or "Note" in payload:
        raise AlphaVantageQuotaError(payload.get("Information") or payload.get("Note"))
    turns = payload.get("transcript") or []
    if not isinstance(turns, list) or not turns:
        return None
    parts: list[str] = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        speaker = (t.get("speaker") or "").strip()
        title = (t.get("title") or "").strip()
        content = (t.get("content") or "").strip()
        if not content:
            continue
        header = speaker if not title else f"{speaker} ({title})" if speaker else title
        parts.append(f"[{header}]: {content}" if header else content)
    text = "\n\n".join(parts)
    if not text:
        return None
    return Transcript(symbol=symbol.upper(), quarter=quarter, year=year, content=text)


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
    token = os.environ.get("ALPHAVANTAGE_API_KEY", "") or ALPHA_VANTAGE_DEFAULT_KEY
    if not token:
        raise RuntimeError("ALPHAVANTAGE_API_KEY env var must be set")
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


# EDGAR filings ship with a lot of boilerplate alongside the actual exhibits:
# rendering files generated for the Inline XBRL viewer, the auto-generated
# index pages, and the form itself. Only Ex-99.* and named "transcript"/
# "remarks"/"call" files are realistic transcript candidates.
_EX99_PAT = re.compile(r"ex[_\-]?99", re.IGNORECASE)
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


def _exhibit_sort_key(name: str) -> tuple[int, str]:
    """Order: named-transcript files first, then ex-99.2/3/... before ex-99.1
    (since ex-99.1 is almost always the press release). Lower tuple sorts first."""
    lower = name.lower()
    if _NAMED_TRANSCRIPT_PAT.search(lower):
        return (0, lower)
    # Bias against ex-99.1 (typically press release) by sorting after the rest.
    if re.search(r"ex[_\-]?99[_\-.]?1\b", lower):
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


async def _sec_extract_transcript(
    symbol: str, *, client: httpx.AsyncClient, today: _dt.date
) -> Transcript | None:
    """Scan recent Item 2.02 8-K filings for a transcript-shaped exhibit.

    Walks the symbol's recent filings list, filters to earnings 8-Ks within
    `SEC_LOOKBACK_DAYS`, and for each candidate downloads each non-trivial
    exhibit until one passes the `_looks_like_transcript` heuristic.
    Returns `None` when no filing yields a transcript-shaped exhibit."""
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
    candidates: list[tuple[_dt.date, str, str, str]] = []  # (filed, accession, primary, reportDate)
    for i, form in enumerate(forms):
        if not form or not form.startswith("8-K"):
            continue
        items_str = items_col[i] if i < len(items_col) else ""
        if "2.02" not in items_str:
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
            )
        )
    # Most recent first.
    candidates.sort(key=lambda c: c[0], reverse=True)

    for filed_date, accession, _primary, report_date in candidates:
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
        for name in exhibits:
            text = await _sec_fetch_exhibit_text(cik, accession, name, client=client)
            if text and _looks_like_transcript(text):
                try:
                    rep = _dt.date.fromisoformat(report_date) if report_date else filed_date
                except ValueError:
                    rep = filed_date
                q, y = _quarter_from_report_date(rep)
                logger.info(
                    "SEC transcript hit: %s %s exhibit %s (filed %s)",
                    symbol, accession, name, filed_date,
                )
                return Transcript(symbol=symbol.upper(), quarter=q, year=y, content=text)
    return None


async def fetch_transcript(
    symbol: str,
    quarter: int,
    year: int,
    *,
    client: httpx.AsyncClient | None = None,
    timeout_s: float = 30.0,
    today: _dt.date | None = None,
) -> Transcript | None:
    """Fetch a single quarter's transcript. Returns None if not available.

    Future / in-progress quarters short-circuit to None — Alpha Vantage has no
    transcript for a quarter that hasn't ended yet."""
    today = today or _dt.date.today()
    if not _is_completed_quarter(quarter, year, today):
        return None

    token = _resolve_token()
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=timeout_s)
    try:
        return await _alpha_vantage_fetch(symbol, quarter, year, client=client, token=token)
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
    Alpha Vantage's quota is exhausted (logged once)."""
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
    av_quota_hit = False
    try:
        # `max_lookback_quarters` bounds the total quarters we walk back. Cached
        # empties consume a slot too — otherwise a polluted cache could make us
        # walk back arbitrarily far looking for an uncached quarter.
        for _ in range(max_lookback_quarters):
            if _is_cached_empty(neg_cache, symbol, q, y, today):
                logger.debug(
                    "AlphaVantage transcript %s Q%d %d: skipping (cached empty)", symbol, q, y
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
                t = await _alpha_vantage_fetch(symbol, q, y, client=client, token=token)
            except AlphaVantageQuotaError as e:
                logger.warning(
                    "AlphaVantage transcript %s: quota / rate-limit hit on Q%d %d — "
                    "falling back to SEC EDGAR. %s",
                    symbol, q, y, e,
                )
                av_quota_hit = True
                break
            if t is not None:
                return t
            _record_empty(neg_cache, symbol, q, y, today)
            neg_cache_dirty = True
            q -= 1
            if q == 0:
                q = 4
                y -= 1

        # SEC EDGAR fallback: try once, regardless of whether AV exhausted via
        # walk-back or via quota advisory. Free and rate-limit-free; coverage
        # is partial so this often returns None too, but when it hits, it's
        # complementary to AV.
        logger.info(
            "AlphaVantage transcript %s: %s, trying SEC EDGAR fallback",
            symbol, "quota hit" if av_quota_hit else "no transcript in walk-back window",
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
