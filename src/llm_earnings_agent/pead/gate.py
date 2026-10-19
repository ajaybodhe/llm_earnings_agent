"""Does any simple signal predict post-earnings drift? (the Phase 2a gate)

Hit rates with a ±1% deadband are a blunt instrument. This module uses the two
standard PEAD tests, each computed cross-sectionally per calendar quarter (so
decile breakpoints never use future events) and then averaged across quarters
(Fama-MacBeth style), which also handles the strong correlation between stocks
reporting in the same quarter:

* IC: Spearman rank correlation between signal and subsequent drift.
* Decile spread: mean drift of the top-decile minus the bottom-decile events.

t-statistics are mean / (sd / sqrt(quarters)). |t| < 2 means no evidence.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

# Signals known at the reaction-day close (see quarterly_results internal/events).
SIGNALS: dict[str, Callable[[dict[str, Any]], float | None]] = {
    "sue (winsorised ±5)": lambda e: _clip(e["features"].get("sue"), 5.0),
    "reaction_ret_pct": lambda e: e["features"].get("reaction_ret_pct"),
    "gap_ret_pct": lambda e: e["features"].get("gap_ret_pct"),
    "rules_score": lambda e: e.get("rules", {}).get("score"),
    "ret_6m_pre (momentum)": lambda e: e["features"].get("ret_6m_pre"),
    "abnormal_volume": lambda e: e["features"].get("abnormal_volume"),
}

TARGETS = ("drift5_adj", "drift20_adj", "drift20_mkt")

MIN_EVENTS_PER_QUARTER = 30


def _clip(v: float | None, limit: float) -> float | None:
    if v is None:
        return None
    return max(-limit, min(limit, float(v)))


def quarter_key(announcement_date: str) -> str:
    y, m = announcement_date[:4], int(announcement_date[5:7])
    return f"{y}Q{(m - 1) // 3 + 1}"


def _rank(x: np.ndarray) -> np.ndarray:
    order = x.argsort(kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # average ties
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    return (sums / counts)[inv]


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = _rank(x), _rank(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / den) if den > 0 else 0.0


@dataclass
class GateRow:
    signal: str
    target: str
    events: int
    quarters: int
    mean_ic: float
    ic_t: float
    spread: float  # top-decile minus bottom-decile mean drift, % points
    spread_t: float


def _t(values: list[float]) -> float:
    a = np.asarray(values, dtype=float)
    if len(a) < 2 or a.std(ddof=1) == 0:
        return 0.0
    return float(a.mean() / (a.std(ddof=1) / np.sqrt(len(a))))


def evaluate_signal(
    events: list[dict[str, Any]],
    signal: Callable[[dict[str, Any]], float | None],
    target: str,
    name: str = "",
) -> GateRow:
    by_q: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for e in events:
        s, y = signal(e), e.get("labels", {}).get(target)
        if s is None or y is None:
            continue
        by_q[quarter_key(e["features"]["announcement_date"])].append((float(s), float(y)))
    ics, spreads, n = [], [], 0
    for pairs in by_q.values():
        if len(pairs) < MIN_EVENTS_PER_QUARTER:
            continue
        x = np.array([p[0] for p in pairs])
        y = np.array([p[1] for p in pairs])
        n += len(pairs)
        ics.append(spearman(x, y))
        lo, hi = np.quantile(x, [0.1, 0.9])
        if hi > lo:
            spreads.append(float(y[x >= hi].mean() - y[x <= lo].mean()))
    return GateRow(
        signal=name,
        target=target,
        events=n,
        quarters=len(ics),
        mean_ic=float(np.mean(ics)) if ics else 0.0,
        ic_t=_t(ics),
        spread=float(np.mean(spreads)) if spreads else 0.0,
        spread_t=_t(spreads),
    )


def run_gate(events: list[dict[str, Any]]) -> list[GateRow]:
    return [
        evaluate_signal(events, fn, target, name)
        for target in TARGETS
        for name, fn in SIGNALS.items()
    ]


def by_year(
    events: list[dict[str, Any]], signal: Callable[[dict[str, Any]], float | None], target: str
) -> list[tuple[str, GateRow]]:
    years = sorted({e["features"]["announcement_date"][:4] for e in events})
    out = []
    for y in years:
        sub = [e for e in events if e["features"]["announcement_date"].startswith(y)]
        out.append((y, evaluate_signal(sub, signal, target)))
    return out


def format_gate(rows: list[GateRow]) -> str:
    lines = [
        f"{'target':<12} {'signal':<24} {'events':>7} {'qtrs':>5} "
        f"{'mean IC':>8} {'IC t':>6} {'D10-D1 %':>9} {'t':>6}"
    ]
    for r in rows:
        flag = "  *" if abs(r.spread_t) >= 2 or abs(r.ic_t) >= 2 else ""
        lines.append(
            f"{r.target:<12} {r.signal:<24} {r.events:>7} {r.quarters:>5} "
            f"{r.mean_ic:>+8.3f} {r.ic_t:>+6.1f} {r.spread:>+9.2f} {r.spread_t:>+6.1f}{flag}"
        )
    lines.append(
        "* |t| ≥ 2. D10-D1 = top-minus-bottom decile mean drift, averaged across quarters."
    )
    return "\n".join(lines)
