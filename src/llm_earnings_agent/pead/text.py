"""Point-in-time text features for historical earnings releases.

For each event the earnings press release (EX-99.1 of the Item 2.02 8-K the
event was built from) is fetched from SEC EDGAR, cached on disk, then turned
into features with no LLM — an LLM may remember how past quarters played out,
which would leak the label:

* Loughran-McDonald tone proportions, when the user supplies the LM Master
  Dictionary CSV (free for academic research; commercial use needs a licence
  from the authors — https://sraf.nd.edu/loughranmcdonald-master-dictionary/).
* A local sentence embedding (fastembed / ONNX on CPU): the first chunks of
  the release — headline, highlights, guidance — embedded and mean-pooled.
"""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import re
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import httpx

from ..data.transcripts import fetch_press_release_text

logger = logging.getLogger(__name__)

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_WORDS = 300
MAX_CHUNKS = 8
LM_CATEGORIES = ("negative", "positive", "uncertainty", "litigious", "constraining")
_WORD_RE = re.compile(r"[A-Za-z]+")


# ── fetching ─────────────────────────────────────────────────────────────────


class RateLimitedClient:
    """httpx client wrapper that spaces requests at least 1/rps seconds apart
    (SEC's fair-access limit is 10 requests/second) and retries throttling
    (429), server errors (5xx) and network failures with backoff — the
    downstream helpers read any failed request as "no press release"."""

    def __init__(
        self, client: httpx.AsyncClient, rps: float, backoff: tuple[float, ...] = (2, 5, 15, 30)
    ) -> None:
        self._client = client
        self._interval = 1.0 / rps
        self._lock = asyncio.Lock()
        self._next = 0.0
        self._backoff = backoff

    async def _slot(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._next = max(now, self._next) + self._interval

    async def get(self, url: str, **kwargs: Any) -> httpx.Response:
        for attempt in range(len(self._backoff) + 1):
            await self._slot()
            try:
                resp = await self._client.get(url, **kwargs)
            except httpx.TransportError:
                if attempt == len(self._backoff):
                    raise
            else:
                if resp.status_code != 429 and resp.status_code < 500:
                    return resp
                if attempt == len(self._backoff):
                    return resp
            await asyncio.sleep(self._backoff[attempt])
        raise AssertionError("unreachable")


def text_path(out_dir: Path, accession: str) -> Path:
    return out_dir / "texts" / f"{accession}.txt"


def missing_marker(out_dir: Path, accession: str) -> Path:
    return out_dir / "texts" / f"{accession}.none"


async def fetch_release_texts(
    events: Iterable[dict[str, Any]],
    out_dir: Path,
    *,
    rps: float = 8.0,
    concurrency: int = 4,
    client: Any = None,
    fetch: Callable[..., Any] = fetch_press_release_text,
) -> dict[str, int]:
    """Download every event's press release into out_dir/texts/ (resumable:
    cached texts and known misses are skipped). Returns counts."""
    (out_dir / "texts").mkdir(parents=True, exist_ok=True)
    todo = []
    stats = {"cached": 0, "fetched": 0, "missing": 0, "no_accession": 0, "errors": 0}
    seen: set[str] = set()
    for e in events:
        acc, cik = e.get("accession"), e.get("cik")
        if not acc or not cik:
            stats["no_accession"] += 1
            continue
        if acc in seen:
            continue
        seen.add(acc)
        if text_path(out_dir, acc).exists() or missing_marker(out_dir, acc).exists():
            stats["cached"] += 1
            continue
        todo.append((cik, acc, e.get("primary_doc", "")))

    owns = client is None
    raw = client or httpx.AsyncClient(timeout=30.0)
    limited = RateLimitedClient(raw, rps) if owns else raw
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def one(cik: Any, acc: str, primary: str) -> None:
        nonlocal done
        async with sem:
            try:
                text = await fetch(cik, acc, primary, client=limited)
            except Exception as exc:  # keep going; retried next run
                stats["errors"] += 1
                logger.warning("press release %s: %s", acc, exc)
                return
            if text:
                text_path(out_dir, acc).write_text(text, encoding="utf-8")
                stats["fetched"] += 1
            else:
                missing_marker(out_dir, acc).touch()
                stats["missing"] += 1
            done += 1
            if done % 250 == 0:
                logger.info("press releases: %d/%d", done, len(todo))

    try:
        await asyncio.gather(*(one(*t) for t in todo))
    finally:
        if owns:
            await raw.aclose()
    return stats


# ── features ─────────────────────────────────────────────────────────────────


def load_lm_dictionary(path: str | Path) -> dict[str, set[str]]:
    """Read the Loughran-McDonald Master Dictionary CSV into category → words.
    A word belongs to a category when that column is non-zero (the CSV stores
    the year the word was added)."""
    cats: dict[str, set[str]] = {c: set() for c in LM_CATEGORIES}
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            word = (row.get("Word") or "").strip().lower()
            if not word:
                continue
            for c in LM_CATEGORIES:
                v = (row.get(c.capitalize()) or "0").strip()
                if v not in ("", "0"):
                    cats[c].add(word)
    return cats


def tone_features(text: str, lm: dict[str, set[str]]) -> dict[str, float]:
    """Share of words in each LM category, plus net tone and word count."""
    words = [w.lower() for w in _WORD_RE.findall(text)]
    n = len(words)
    out: dict[str, float] = {"words": float(n)}
    for c in LM_CATEGORIES:
        out[f"lm_{c}"] = sum(1 for w in words if w in lm[c]) / n if n else 0.0
    out["lm_net_tone"] = out["lm_positive"] - out["lm_negative"]
    return out


def chunks(
    text: str, words_per_chunk: int = CHUNK_WORDS, max_chunks: int = MAX_CHUNKS
) -> list[str]:
    """Split text into up to max_chunks chunks of words_per_chunk words — the
    top of a press release carries the headline, highlights and guidance."""
    words = text.split()
    return [
        " ".join(words[i : i + words_per_chunk])
        for i in range(0, min(len(words), words_per_chunk * max_chunks), words_per_chunk)
    ]


def fastembed_embedder(model_name: str = DEFAULT_EMBED_MODEL) -> Callable[[list[str]], Any]:
    """Return a function mapping texts → (n, d) float32 array via fastembed."""
    try:
        import numpy as np
        from fastembed import TextEmbedding
    except ImportError as exc:  # pragma: no cover — optional dependency
        raise RuntimeError('embeddings need the "pead" extra: uv pip install -e ".[pead]"') from exc
    model = TextEmbedding(model_name)

    def embed(texts: list[str]) -> Any:
        return np.asarray(list(model.embed(texts, batch_size=32)), dtype=np.float32)

    return embed


def document_embedding(
    text: str, embed: Callable[[list[str]], Any], max_chunks: int = MAX_CHUNKS
) -> Any:
    """Mean-pool the chunk embeddings and L2-normalise."""
    import numpy as np

    parts = chunks(text, max_chunks=max_chunks)
    if not parts:
        return None
    v = np.asarray(embed(parts), dtype=np.float32).mean(axis=0)
    norm = float(np.linalg.norm(v))
    return v / norm if norm > 0 else v


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)


def build_text_features(
    events: list[dict[str, Any]],
    out_dir: Path,
    *,
    lm: dict[str, set[str]] | None = None,
    embed: Callable[[list[str]], Any] | None = None,
    model_name: str = DEFAULT_EMBED_MODEL,
    max_chunks: int = MAX_CHUNKS,
    since: str = "",
    newest_first: bool = False,
) -> dict[str, int]:
    """Write text features for every event whose press release is cached:
    out_dir/text_features.jsonl (event_id, accession, tone) and, when embed is
    given, out_dir/embeddings.npy with rows aligned to embeddings_index.json."""
    import numpy as np

    rows, vectors, index = [], [], []
    # Cache per (model, chunk count): embeddings of different configs never mix.
    emb_dir = out_dir / "emb" / f"{_safe(model_name)}_c{max_chunks}"
    emb_dir.mkdir(parents=True, exist_ok=True)
    events = [e for e in events if e["features"]["announcement_date"] >= since]
    if newest_first:
        events = sorted(events, key=lambda e: e["features"]["announcement_date"], reverse=True)
    for e in events:
        acc = e.get("accession")
        if not acc or not text_path(out_dir, acc).exists():
            continue
        eid = e.get("event_id") or f"{e['symbol'].upper()}|{e['features']['announcement_date']}"
        row: dict[str, Any] = {"event_id": eid, "accession": acc}
        text: str | None = None
        if lm is not None:
            text = text_path(out_dir, acc).read_text(encoding="utf-8")
            row.update(tone_features(text, lm))
        rows.append(row)
        if embed is not None:
            # Per-release cache: re-runs (e.g. while texts are still being
            # fetched) only embed releases that are new since the last run.
            cache = emb_dir / f"{acc}.npy"
            if cache.exists():
                v = np.load(cache)
            else:
                if text is None:
                    text = text_path(out_dir, acc).read_text(encoding="utf-8")
                v = document_embedding(text, embed, max_chunks)
                if v is None:
                    continue
                np.save(cache, v)
            vectors.append(v)
            index.append(eid)

    with open(out_dir / "text_features.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    if embed is not None and vectors:
        np.save(out_dir / "embeddings.npy", np.vstack(vectors))
        meta = {
            "model": model_name,
            "chunk_words": CHUNK_WORDS,
            "max_chunks": max_chunks,
            "event_ids": index,
        }
        (out_dir / "embeddings_index.json").write_text(json.dumps(meta), encoding="utf-8")
    return {"events_with_text": len(rows), "embedded": len(vectors)}
