import json

import numpy as np
import pytest

from llm_earnings_agent.data import transcripts as tx
from llm_earnings_agent.pead import text as pt
from llm_earnings_agent.pead.events import load_events

# ── events ───────────────────────────────────────────────────────────────────


def test_load_events_latest_export_wins(tmp_path):
    p = tmp_path / "e.jsonl"
    base = {"symbol": "AAA", "features": {"announcement_date": "2026-01-30"}}
    p.write_text(
        "\n".join(
            json.dumps({**base, "exported_at": t, "labels": lab})
            for t, lab in [("2026-02-01T00:00:00Z", {}), ("2026-03-01T00:00:00Z", {"drift5": 1.0})]
        )
        + "\n"
    )
    evs = load_events(p)
    assert len(evs) == 1 and evs[0]["labels"] == {"drift5": 1.0}


# ── press-release selection ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_press_release_prefers_ex991_and_skips_cover(monkeypatch):
    fetched = []

    async def fake_index(cik, accession, *, client):
        return [
            {"name": n}
            for n in [
                "x-index.html",
                "aapl-20260730.htm",
                "transcript.htm",
                "a8-kex991q3.htm",
                "R1.htm",
            ]
        ]

    async def fake_text(cik, accession, name, *, client):
        fetched.append(name)
        return "Apple reports third quarter results. " * 100

    monkeypatch.setattr(tx, "_sec_fetch_8k_index", fake_index)
    monkeypatch.setattr(tx, "_sec_fetch_exhibit_text", fake_text)
    text = await tx.fetch_press_release_text(320193, "acc", "aapl-20260730.htm", client=None)
    assert text and fetched == ["a8-kex991q3.htm"]


# ── fetching (resumable) ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_release_texts_caches_and_marks_misses(tmp_path):
    calls = []

    async def fake_fetch(cik, acc, primary, *, client):
        calls.append(acc)
        return None if acc == "miss" else f"release text {acc} " * 300

    evs = [
        {
            "symbol": "A",
            "cik": 1,
            "accession": "hit",
            "features": {"announcement_date": "2026-01-01"},
        },
        {
            "symbol": "B",
            "cik": 2,
            "accession": "miss",
            "features": {"announcement_date": "2026-01-02"},
        },
        {
            "symbol": "C",
            "features": {"announcement_date": "2026-01-03"},
        },  # enriched export: no accession
    ]
    stats = await pt.fetch_release_texts(evs, tmp_path, client=object(), fetch=fake_fetch)
    assert stats["fetched"] == 1 and stats["missing"] == 1 and stats["no_accession"] == 1
    assert pt.text_path(tmp_path, "hit").exists() and pt.missing_marker(tmp_path, "miss").exists()

    calls.clear()
    stats = await pt.fetch_release_texts(evs, tmp_path, client=object(), fetch=fake_fetch)
    assert calls == [] and stats["cached"] == 2  # second run re-fetches nothing


# ── features ─────────────────────────────────────────────────────────────────


def test_lm_dictionary_and_tone(tmp_path):
    p = tmp_path / "lm.csv"
    p.write_text(
        "Word,Negative,Positive,Uncertainty,Litigious,Constraining\n"
        "LOSS,2009,0,0,0,0\nSTRONG,0,2009,0,0,0\nMAY,0,0,2009,0,0\n"
    )
    lm = pt.load_lm_dictionary(p)
    assert lm["negative"] == {"loss"} and lm["positive"] == {"strong"}
    tone = pt.tone_features("Strong quarter, strong demand; a loss may follow.", lm)
    assert tone["words"] == 8
    assert tone["lm_positive"] == pytest.approx(2 / 8)
    assert tone["lm_net_tone"] == pytest.approx(1 / 8)


def test_chunks_cap_length():
    text = " ".join(f"w{i}" for i in range(5000))
    parts = pt.chunks(text, words_per_chunk=300, max_chunks=8)
    assert len(parts) == 8 and all(len(p.split()) == 300 for p in parts)


def test_build_text_features_with_fake_embedder(tmp_path):
    (tmp_path / "texts").mkdir()
    pt.text_path(tmp_path, "acc1").write_text("guidance raised record revenue " * 200)
    evs = [
        {
            "event_id": "AAA|2026-01-30",
            "symbol": "AAA",
            "accession": "acc1",
            "features": {"announcement_date": "2026-01-30"},
        },
        {
            "event_id": "BBB|2026-01-31",
            "symbol": "BBB",
            "accession": "nope",
            "features": {"announcement_date": "2026-01-31"},
        },
    ]

    def fake_embed(texts):
        return np.ones((len(texts), 4), dtype=np.float32)

    res = pt.build_text_features(evs, tmp_path, embed=fake_embed, model_name="fake")
    assert res == {"events_with_text": 1, "embedded": 1}
    emb = np.load(tmp_path / "embeddings.npy")
    idx = json.loads((tmp_path / "embeddings_index.json").read_text())
    assert emb.shape == (1, 4) and idx["event_ids"] == ["AAA|2026-01-30"] and idx["model"] == "fake"
    assert np.linalg.norm(emb[0]) == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_rate_limited_client_retries_server_errors():
    import httpx

    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503 if len(calls) < 3 else 200, text="ok")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as raw:
        client = pt.RateLimitedClient(raw, rps=1000, backoff=(0, 0, 0))
        resp = await client.get("https://www.sec.gov/x")
    assert resp.status_code == 200 and len(calls) == 3
