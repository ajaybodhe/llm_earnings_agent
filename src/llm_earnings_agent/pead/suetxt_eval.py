"""Does SUE.txt help? (1) post-earnings drift, (2) earnings-day signals."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from . import model as pm
from .gate import _t, quarter_key, spearman

DRIFT_LABELS = ("drift5_mkt", "drift20_mkt", "drift60_mkt", "drift120_mkt")


def quintile_spread(
    events: list[dict[str, Any]], signal: str, label: str, since: str
) -> tuple[float, float, int, int]:
    """Mean top-minus-bottom quintile label per quarter (within-quarter ranks),
    averaged over quarters with its t-stat. Returns (spread, t, quarters, events)."""
    by_q: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for e in events:
        s, y = e["features"].get(signal), e["labels"].get(label)
        if s is None or y is None or e["features"]["announcement_date"] < since:
            continue
        by_q[quarter_key(e["features"]["announcement_date"])].append((float(s), float(y)))
    spreads, n = [], 0
    for rows in by_q.values():
        if len(rows) < 50:
            continue
        x = np.array([r[0] for r in rows])
        y = np.array([r[1] for r in rows])
        lo, hi = np.quantile(x, [0.2, 0.8])
        if hi > lo:
            spreads.append(float(y[x >= hi].mean() - y[x <= lo].mean()))
            n += len(rows)
    return (float(np.mean(spreads)) if spreads else 0.0, _t(spreads), len(spreads), n)


def ic(events: list[dict[str, Any]], signal: str, target, since: str) -> tuple[float, float]:
    by_q: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for e in events:
        s, y = e["features"].get(signal), target(e)
        if s is None or y is None or e["features"]["announcement_date"] < since:
            continue
        by_q[quarter_key(e["features"]["announcement_date"])].append((float(s), float(y)))
    ics = [
        spearman(np.array([a for a, _ in v]), np.array([b for _, b in v]))
        for v in by_q.values()
        if len(v) >= 30
    ]
    return (float(np.mean(ics)) if ics else 0.0, _t(ics))


def report(events: list[dict[str, Any]]) -> str:
    rx = pm._f("reaction_ret_pct")
    lines = []
    since_txt = "2018-10-01"  # first SUE.txt test quarter
    lines.append(
        "1) POST-EARNINGS DRIFT — top-minus-bottom quintile, market-adjusted, avg per quarter (t)"
    )
    lines.append(f"   {'signal':<22}" + "".join(f"{lab:>18}" for lab in DRIFT_LABELS))
    for sig, since in (
        ("sue_txt", since_txt),
        ("sue", since_txt),
        ("reaction_ret_pct", since_txt),
        ("eps_surprise_pct", "2021-01-01"),
    ):
        cells = []
        for lab in DRIFT_LABELS:
            s, t, q, _ = quintile_spread(events, sig, lab, since)
            cells.append(f"{s:+7.2f}% ({t:+4.1f})" if q else "—")
        lines.append(f"   {sig:<22}" + "".join(f"{c:>18}" for c in cells))
    lines.append("")
    lines.append("2) EARNINGS-DAY — rank correlation (IC) with the reaction-day return, 2022+ (t)")
    for sig in ("sue_txt", "eps_surprise_pct", "sue"):
        m, t = ic(events, sig, rx, "2022-01-01")
        lines.append(f"   {sig:<22} IC {m:+.3f} ({t:+.1f})")
    return "\n".join(lines)


def combined_models(events: list[dict[str, Any]]) -> str:
    """Walk-forward 2022+: does SUE.txt add to consensus surprise (Signal 1)
    and to the open→close model (Signal 2)?"""
    F = {**pm.OPEN_FEATURES, "sue_txt": pm._clip(pm._f("sue_txt"), 10.0)}
    s1 = pm.TASKS["reaction"]
    s2 = pm.TASKS["intraday_mkt"]
    base2 = [
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
    ]
    rows = []
    for label, task, cols, kind in (
        ("S1 consensus only", s1, ["eps_surprise_pct"], "ridge"),
        ("S1 SUE.txt only", s1, ["sue_txt"], "ridge"),
        ("S1 consensus + SUE.txt", s1, ["eps_surprise_pct", "sue_txt"], "ridge"),
        ("S2 base (live features)", s2, base2, "gbdt"),
        ("S2 base + SUE.txt", s2, [*base2, "sue_txt"], "gbdt"),
    ):
        t = pm.Task(
            task.name, task.target, {c: F[c] for c in cols}, task.embargo_days, task.baselines
        )
        r, _ = pm.walk_forward(events, t, first_test="2022Q1", kind=kind, min_train=1500)
        lo, hi = pm.wilson(r.sign_hits, r.sign_n)
        rows.append(
            f"   {label:<26} {kind:<5} IC {r.mean_ic:+.3f} (t {r.ic_t:+.1f})  "
            f"D10-D1 {r.spread:+.2f}% (t {r.spread_t:+.1f})  "
            f"sign {r.sign_acc:.1%} [{lo:.1%}–{hi:.1%}]"
        )
    return "3) WALK-FORWARD MODELS, 2022+ (does SUE.txt add to what we have?)\n" + "\n".join(rows)
