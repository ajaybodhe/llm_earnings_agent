"""Post-release signals for recent earnings releases.

Signal 1 — at the release: direction of the reaction-day move, read from the
consensus EPS surprise via its historical calibration (how often reactions
were positive for surprises of that size, 2021+).

Signal 2 — at the reaction-day open: open→close return minus SPY, from a GBDT
trained on events before CALIB_START and calibrated by prediction decile on
the events after it (which the model never saw). Only decile 10 is a trade
(Long; hedging with SPY is optional): since CALIB_START decile 9 has shown no
edge and the short deciles have averaged about zero, so deciles 1–9 are
"no trade".

Inputs for the new releases come from quarterly_results' JSON output
(`go run . --from D --to D --output json`), whose drift_features carry the
consensus surprise, gap, reaction and momentum for reports released within
the last few sessions. Only features available live are used (no SUE: the new
quarter isn't in SEC XBRL until the 10-Q).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from .model import _intraday_mkt, _open_to_close, make_model
from .webearnings import WebEarnings
from .webearnings import summary as web_summary

CONSENSUS_START = "2021-01-01"
CALIB_START = "2025-07-01"

SIGNAL2_FEATURES = (
    "pre7_ret",
    "ret_1m_pre",
    "ret_6m_pre",
    "gap_ret_pct",
    "eps_surprise_pct",
    "gap_vs_typical",
    "surprise_gap_agree",
    "gap_x_surprise",
    "spy_gap_pct",
    "vix_prev",
)
_CLIP = {"eps_surprise_pct": 100.0, "gap_vs_typical": 10.0, "gap_x_surprise": 30.0}

SURPRISE_BINS = (-math.inf, -10.0, -2.0, 0.0, 2.0, 5.0, 10.0, 20.0, math.inf)


def events_from_results(
    results: list[dict[str, Any]], web: dict[str, WebEarnings] | None = None
) -> list[dict[str, Any]]:
    """Turn quarterly_results JSON results into event dicts for released reports.

    A result qualifies once its consensus surprise is known (top-level
    eps_surprise_pct, from the Nasdaq calendar), or — when `web` has a usable
    entry for the symbol — from the web backup. drift_features — present once
    the reaction-day open exists — add the gap and momentum Signal 2 needs."""
    out = []
    for r in results:
        f = r.get("drift_features") or {}
        surprise = f.get("eps_surprise_pct", r.get("eps_surprise_pct"))
        source = "nasdaq"
        w = (web or {}).get(r["symbol"].upper())
        if surprise is None and w is not None and w.usable()[0]:
            surprise = w.surprise_pct()
            source = f"web ({w.consensus_source or 'consensus'})"
        if surprise is None:
            continue
        feats = dict(f) if f else {"announcement_date": r.get("earnings_date", "")}
        feats["eps_surprise_pct"] = surprise
        e = {
            "event_id": f"{r['symbol'].upper()}|{feats['announcement_date']}|live",
            "symbol": r["symbol"],
            "live": True,
            "has_open": bool(f),
            "earnings_time": r.get("earnings_time") or "",
            "surprise_source": source,
            "features": feats,
            "labels": {},
        }
        if w is not None:
            e["web"] = web_summary(w)
        out.append(e)
    return out


def _row(e: dict[str, Any]) -> list[float]:
    f = e["features"]
    vals = []
    for name in SIGNAL2_FEATURES:
        v = f.get(name)
        if v is None:
            vals.append(np.nan)
        else:
            lim = _CLIP.get(name)
            vals.append(max(-lim, min(lim, float(v))) if lim else float(v))
    return vals


@dataclass
class Bucket:
    lo: float
    hi: float
    n: int
    p_up: float
    mean_rx: float


def surprise_calibration(history: list[dict[str, Any]]) -> list[Bucket]:
    """Reaction-day outcomes by consensus-surprise bucket (2021+)."""
    pairs = [
        (e["features"]["eps_surprise_pct"], e["features"]["reaction_ret_pct"])
        for e in history
        if e["features"].get("eps_surprise_pct") is not None
        and e["features"].get("reaction_ret_pct") is not None
        and e["features"]["announcement_date"] >= CONSENSUS_START
    ]
    out = []
    for lo, hi in pairwise(SURPRISE_BINS):
        rx = [r for s, r in pairs if lo <= s < hi]
        if rx:
            out.append(
                Bucket(lo, hi, len(rx), sum(1 for r in rx if r > 0) / len(rx), float(np.mean(rx)))
            )
    return out


def bucket_for(cal: list[Bucket], surprise: float) -> Bucket | None:
    return next((b for b in cal if b.lo <= surprise < b.hi), None)


@dataclass
class Decile:
    edges: np.ndarray  # 9 interior cut points
    mean: list[float]  # realised open→close minus SPY per decile (calibration set)
    hit: list[float]  # share with open→close minus SPY > 0
    n: list[int]


def train_signal2(history: list[dict[str, Any]]) -> tuple[Any, Decile, str]:
    """Train on [CONSENSUS_START, CALIB_START), calibrate deciles after it."""

    def rows(lo: str, hi: str) -> tuple[np.ndarray, np.ndarray]:
        X, y = [], []
        for e in history:
            d = e["features"]["announcement_date"]
            t = _intraday_mkt(e)
            if lo <= d < hi and t is not None and e["features"].get("eps_surprise_pct") is not None:
                X.append(_row(e))
                y.append(t)
        return np.asarray(X, dtype=float), np.asarray(y)

    Xtr, ytr = rows(CONSENSUS_START, CALIB_START)
    Xca, yca = rows(CALIB_START, "9999")
    lo, hi = np.quantile(ytr, [0.01, 0.99])
    model, backend = make_model([])
    model.fit(Xtr, np.clip(ytr, lo, hi))
    p = model.predict(Xca)
    edges = np.quantile(p, np.linspace(0.1, 0.9, 9))
    dec = np.searchsorted(edges, p, side="right")  # 0..9
    mean, hit, n = [], [], []
    for d in range(10):
        ys = yca[dec == d]
        n.append(len(ys))
        mean.append(float(ys.mean()) if len(ys) else 0.0)
        hit.append(float(np.mean(ys > 0)) if len(ys) else float("nan"))
    return (
        model,
        Decile(edges, mean, hit, n),
        f"{backend}: train {len(ytr)} events, calibrate {len(yca)}",
    )


def lean(decile: int) -> str:
    """Trade lean for a 0-based decile: only the top decile is a trade."""
    return "LONG" if decile == 9 else "no trade"


def signal1_label(p_up: float) -> str:
    return "Positive" if p_up >= 0.55 else "Negative" if p_up <= 0.45 else "Neutral"


def signals(history: list[dict[str, Any]], new: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Signal 1 for every released report; Signal 2 when the open is known."""
    cal = surprise_calibration(history)
    model, dec, info = train_signal2(history)
    out = []
    for e in sorted(new, key=lambda e: (e["features"]["announcement_date"], e["symbol"])):
        f = e["features"]
        s = f["eps_surprise_pct"]
        b = bucket_for(cal, s)
        item: dict[str, Any] = {
            "symbol": e["symbol"],
            "announcement_date": f["announcement_date"],
            "reaction_day": f.get("reaction_day", ""),
            "model": info,
            "signal1": None,
            "signal2": None,
        }
        if b:
            item["signal1"] = {
                "surprise_pct": s,
                "p_up": b.p_up,
                "n": b.n,
                "mean_reaction_pct": b.mean_rx,
                "label": signal1_label(b.p_up),
                "surprise_source": e.get("surprise_source", "nasdaq"),
            }
        if e.get("web"):
            item["web"] = e["web"]
        if e.get("has_open", True) and f.get("gap_ret_pct") is not None:
            p = float(model.predict(np.asarray([_row(e)], dtype=float))[0])
            d = int(np.searchsorted(dec.edges, p, side="right"))
            item["signal2"] = {
                "predicted_pct": p,
                "decile": d + 1,
                "lean": lean(d),
                "hist_mean_pct": dec.mean[d],
                "hist_hit": None if math.isnan(dec.hit[d]) else dec.hit[d],
                "gap_pct": f["gap_ret_pct"],
            }
        out.append(item)
    return out


def report(history: list[dict[str, Any]], new: list[dict[str, Any]]) -> str:
    cal = surprise_calibration(history)
    model, dec, info = train_signal2(history)
    lines = [
        "Signal 1 — at the release: reaction direction from the consensus EPS surprise",
        "Signal 2 — at the open: open→close minus SPY, GBDT decile (10 = long, SPY hedge optional)",
        f"[signal 2 model: {info}]",
        "",
        f"{'symbol':<6} {'released':<10} {'surprise':>8}  {'S1: P(up) hist  mean rx':<26} "
        f"{'gap':>6}  {'S2: decile  lean          hist mean  hit':<42}  "
        f"{'actual: reaction  open→close−SPY':<32}",
    ]
    for e in sorted(new, key=lambda e: (e["features"]["announcement_date"], e["symbol"])):
        f = e["features"]
        s = f["eps_surprise_pct"]
        b = bucket_for(cal, s)
        s1 = f"{b.p_up:>5.0%} (n={b.n:<5}) {b.mean_rx:+5.2f}%" if b else "n/a"
        p = float(model.predict(np.asarray([_row(e)], dtype=float))[0])
        d = int(np.searchsorted(dec.edges, p, side="right"))
        hit = "" if math.isnan(dec.hit[d]) else f"{dec.hit[d]:.0%}"
        s2 = f"D{d + 1:<2}  {lean(d):<13}  {dec.mean[d]:+6.2f}%  {hit:>4}"
        rx = f.get("reaction_ret_pct")
        oc = _intraday_mkt(e)
        act_rx = "n/a" if rx is None else f"{rx:+6.2f}%"
        act_oc = "n/a (no SPY bar)" if oc is None else f"{oc:+6.2f}%"
        if oc is None and _open_to_close(e) is not None:
            act_oc = f"{_open_to_close(e):+6.2f}% (raw)"
        lines.append(
            f"{e['symbol']:<6} {f['announcement_date']:<10} {s:>+7.1f}%  {s1:<26} "
            f"{f.get('gap_ret_pct', 0):>+5.1f}%  {s2:<42}  {act_rx:>8}  {act_oc:>12}"
        )
    lines += [
        "",
        "S1 P(up): share of 2021+ reactions that were positive for surprises in this range.",
        "S2 hist mean / hit: realised open→close−SPY and share of those > 0 for that "
        f"decile on {CALIB_START}+ events the model never saw. Before trading costs.",
    ]
    return "\n".join(lines)
