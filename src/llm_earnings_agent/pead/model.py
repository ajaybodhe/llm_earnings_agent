"""Walk-forward gradient-boosted models for post-earnings direction.

Three tasks, each restricted to what is known at its decision time:

* drift20 / drift5 — after the reaction-day close: sector-adjusted drift over
  the next 20 / 5 trading days (Phase 2b-a, the Zhu, Liu & Sheng target).
* reaction — at the release, before the reaction-day open: the reaction-day
  return. SUE is allowed (it is in the release itself); anything measured on
  the reaction day (gap, volume, sector/market to the close) is not.
* intraday — at the reaction-day open: the open→close return of the reaction
  day, given the release and the opening gap. Unlike the full reaction, this
  part is still tradable after an after-close release.

Evaluation is out-of-sample only: for each test quarter from `first_test` on,
the model is trained on events announced before the quarter start minus an
embargo (longer than the label window, so no training label overlaps the
test period), with fixed hyperparameters (nothing tuned on test data).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np

from .gate import _t, quarter_key, spearman

Event = dict[str, Any]
Getter = Callable[[Event], float | None]


def _f(name: str) -> Getter:
    return lambda e: e["features"].get(name)


def _clip(get: Getter, limit: float) -> Getter:
    def g(e: Event) -> float | None:
        v = get(e)
        return None if v is None else max(-limit, min(limit, float(v)))

    return g


def _timing(e: Event) -> float | None:
    t = e["features"].get("release_timing")
    return {"bmo": 0.0, "amc": 1.0}.get(t) if t else None


def _sic2(e: Event) -> float | None:
    sic = e.get("sic")
    return float(int(sic) // 100) if sic else None


def _open_to_close(e: Event) -> float | None:
    r, g = e["features"].get("reaction_ret_pct"), e["features"].get("gap_ret_pct")
    if r is None or g is None or g <= -100:
        return None
    return ((1 + r / 100) / (1 + g / 100) - 1) * 100


def _intraday_mkt(e: Event) -> float | None:
    oc, spy = _open_to_close(e), e["features"].get("spy_oc_pct")
    return None if oc is None or spy is None else oc - spy


# Known at the release (before the reaction-day open).
RELEASE_FEATURES: dict[str, Getter] = {
    "sue": _clip(_f("sue"), 5.0),
    "eps_surprise_pct": _clip(_f("eps_surprise_pct"), 100.0),  # Nasdaq consensus (2021+)
    "pre7_ret": _f("pre7_ret"),
    "ret_1m_pre": _f("ret_1m_pre"),
    "ret_6m_pre": _f("ret_6m_pre"),
    "release_amc": _timing,
    "sic2": _sic2,
}
# Known at the reaction-day open (context.add_context fills the derived ones).
OPEN_FEATURES: dict[str, Getter] = {
    **RELEASE_FEATURES,
    "gap_ret_pct": _f("gap_ret_pct"),
    "gap_vs_typical": _clip(_f("gap_vs_typical"), 10.0),
    "surprise_gap_agree": _f("surprise_gap_agree"),
    "gap_x_surprise": _clip(_f("gap_x_surprise"), 30.0),
    "spy_gap_pct": _f("spy_gap_pct"),
    "vix_prev": _f("vix_prev"),
}
# Known at the reaction-day close.
CLOSE_FEATURES: dict[str, Getter] = {
    **OPEN_FEATURES,
    "reaction_ret_pct": _f("reaction_ret_pct"),
    "abnormal_volume": _f("abnormal_volume"),
    "sector_ret_1m": _f("sector_ret_1m"),
    "market_ret_1m": _f("market_ret_1m"),
}
CATEGORICAL = ("sic2",)


@dataclass
class Task:
    name: str
    target: Getter
    features: dict[str, Getter]
    embargo_days: int  # > label window, in calendar days
    baselines: dict[str, Getter] = field(default_factory=dict)  # single-feature comparisons


TASKS: dict[str, Task] = {
    "drift20": Task(
        "drift20 (post-reaction, sector-adj.)",
        lambda e: e["labels"].get("drift20_adj"),
        CLOSE_FEATURES,
        40,
        {"reaction_ret_pct": _f("reaction_ret_pct"), "sue": _clip(_f("sue"), 5.0)},
    ),
    "drift5": Task(
        "drift5 (post-reaction, sector-adj.)",
        lambda e: e["labels"].get("drift5_adj"),
        CLOSE_FEATURES,
        14,
        {"reaction_ret_pct": _f("reaction_ret_pct"), "sue": _clip(_f("sue"), 5.0)},
    ),
    "reaction": Task(
        "reaction day (at the release)",
        _f("reaction_ret_pct"),
        RELEASE_FEATURES,
        5,
        {"sue": _clip(_f("sue"), 5.0)},
    ),
    "intraday": Task(
        "reaction-day open→close (at the open)",
        _open_to_close,
        OPEN_FEATURES,
        5,
        {"gap_ret_pct": _f("gap_ret_pct"), "sue": _clip(_f("sue"), 5.0)},
    ),
    "intraday_mkt": Task(
        "open→close minus SPY (at the open)",
        _intraday_mkt,
        OPEN_FEATURES,
        5,
        {
            "gap_x_surprise": _f("gap_x_surprise"),
            "eps_surprise_pct": _clip(_f("eps_surprise_pct"), 100.0),
        },
    ),
}


# ── model backend ────────────────────────────────────────────────────────────


def make_ridge() -> tuple[Any, str]:
    """Median-impute, standardise, ridge. A linear model keeps the full ranking
    of a monotonic signal (SUE), which trees coarsen into ties."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=10.0)), (
        "ridge"
    )


def make_model(categorical_idx: list[int]) -> tuple[Any, str]:
    """LightGBM when its native library loads (needs libomp on macOS), else
    scikit-learn's HistGradientBoosting — the same histogram GBDT algorithm."""
    params = {"learning_rate": 0.05, "num_leaves": 31, "min_leaf": 200, "iters": 300, "l2": 1.0}
    try:
        import lightgbm as lgb

        return (
            lgb.LGBMRegressor(
                n_estimators=params["iters"],
                learning_rate=params["learning_rate"],
                num_leaves=params["num_leaves"],
                min_child_samples=params["min_leaf"],
                reg_lambda=params["l2"],
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                verbose=-1,
            ),
            "lightgbm",
        )
    except (ImportError, OSError):
        from sklearn.ensemble import HistGradientBoostingRegressor

        return (
            HistGradientBoostingRegressor(
                max_iter=params["iters"],
                learning_rate=params["learning_rate"],
                max_leaf_nodes=params["num_leaves"],
                min_samples_leaf=params["min_leaf"],
                l2_regularization=params["l2"],
                categorical_features=categorical_idx or None,
                early_stopping=False,
                random_state=0,
            ),
            "sklearn-hgb",
        )


# ── data ─────────────────────────────────────────────────────────────────────


def matrix(
    events: list[Event], task: Task, extra: dict[str, np.ndarray] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str], list[int]]:
    """Rows with a target. Missing features stay NaN (handled by the GBDT).
    extra maps event_id → additional feature vector (e.g. text PCA)."""
    names = list(task.features)
    rows, ys, idx = [], [], []
    for i, e in enumerate(events):
        y = task.target(e)
        if y is None or not math.isfinite(y):
            continue
        row = [np.nan if (v := g(e)) is None else float(v) for g in task.features.values()]
        rows.append(row)
        ys.append(float(y))
        idx.append(i)
    X = np.asarray(rows, dtype=float)
    if extra is not None:
        dim = len(next(iter(extra.values()))) if extra else 0
        ext = np.full((len(idx), dim), np.nan)
        for r, i in enumerate(idx):
            eid = events[i].get("event_id")
            if eid in extra:
                ext[r] = extra[eid]
        X = np.hstack([X, ext])
        names += [f"text_{k}" for k in range(dim)]
    return X, np.asarray(ys), names, idx


def _quarter_start(q: str) -> date:
    return date(int(q[:4]), 3 * (int(q[-1]) - 1) + 1, 1)


@dataclass
class TaskResult:
    task: str
    backend: str
    events: int
    quarters: int
    mean_ic: float
    ic_t: float
    spread: float
    spread_t: float
    sign_hits: int
    sign_n: int
    baselines: dict[str, tuple[float, float]]  # name → (mean IC, t) on the same test events

    @property
    def sign_acc(self) -> float:
        return self.sign_hits / self.sign_n if self.sign_n else 0.0


def walk_forward(
    events: list[Event],
    task: Task,
    *,
    first_test: str = "2019Q1",
    min_train: int = 3000,
    deadband: float = 1.0,
    extra_fn: Callable[[list[Event], list[int], list[int]], dict[str, np.ndarray]] | None = None,
    kind: str = "gbdt",
) -> tuple[TaskResult, list[tuple[int, float]]]:
    """Expanding-window walk-forward. Returns aggregate metrics and the
    out-of-sample (event index, prediction) pairs.

    extra_fn(events, train_idx, test_idx) → {event_id: vector} lets callers add
    features fitted on the training fold only (e.g. a PCA of text embeddings).
    """
    dates = [e["features"]["announcement_date"] for e in events]
    quarters = sorted({quarter_key(d) for d in dates if quarter_key(d) >= first_test})
    preds: list[tuple[int, float]] = []
    backend = ""
    for q in quarters:
        start = _quarter_start(q)
        cutoff = (start - timedelta(days=task.embargo_days)).isoformat()
        train_ids = [i for i, d in enumerate(dates) if d < cutoff]
        test_ids = [i for i, d in enumerate(dates) if quarter_key(d) == q]
        extra = extra_fn(events, train_ids, test_ids) if extra_fn else None
        tr = [events[i] for i in train_ids]
        te = [events[i] for i in test_ids]
        Xtr, ytr, names, _ = matrix(tr, task, extra)
        Xte, _, _, te_rows = matrix(te, task, extra)
        if len(ytr) < min_train or len(te_rows) == 0:
            continue
        keep = ~np.all(np.isnan(Xtr), axis=0)  # drop features absent from the whole fold
        Xtr, Xte = Xtr[:, keep], Xte[:, keep]
        names = [n for n, k in zip(names, keep, strict=True) if k]
        if kind == "ridge":  # linear: drop categorical codes (no ordinal meaning)
            lin = [i for i, n in enumerate(names) if n not in CATEGORICAL]
            Xtr, Xte, names = Xtr[:, lin], Xte[:, lin], [names[i] for i in lin]
        lo, hi = np.quantile(ytr, [0.01, 0.99])  # fit on winsorised targets
        # Categorical only with ≥2 distinct values in the fold (HGB rejects one).
        cat = [
            i
            for i, n in enumerate(names)
            if n in CATEGORICAL and len(np.unique(Xtr[:, i][~np.isnan(Xtr[:, i])])) > 1
        ]
        model, backend = make_ridge() if kind == "ridge" else make_model(cat)
        model.fit(Xtr, np.clip(ytr, lo, hi))
        p = model.predict(Xte)
        preds += [(test_ids[r], float(v)) for r, v in zip(te_rows, p, strict=True)]
    return evaluate(events, task, preds, backend, deadband), preds


def evaluate(
    events: list[Event], task: Task, preds: list[tuple[int, float]], backend: str, deadband: float
) -> TaskResult:
    by_q: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
    hits = n = 0
    for i, p in preds:
        y = task.target(events[i])
        if y is None:
            continue
        by_q[quarter_key(events[i]["features"]["announcement_date"])].append((p, y, i))
        if abs(y) > deadband:
            n += 1
            hits += (p > 0) == (y > 0)
    ics, spreads = [], []
    base_ics: dict[str, list[float]] = {k: [] for k in task.baselines}
    for rows in by_q.values():
        if len(rows) < 30:
            continue
        p = np.array([r[0] for r in rows])
        y = np.array([r[1] for r in rows])
        ics.append(spearman(p, y))
        lo, hi = np.quantile(p, [0.1, 0.9])
        if hi > lo:
            spreads.append(float(y[p >= hi].mean() - y[p <= lo].mean()))
        for name, get in task.baselines.items():
            pairs = [(get(events[r[2]]), r[1]) for r in rows]
            pairs = [(a, b) for a, b in pairs if a is not None]
            if len(pairs) >= 30:
                base_ics[name].append(
                    spearman(np.array([a for a, _ in pairs]), np.array([b for _, b in pairs]))
                )
    return TaskResult(
        task=task.name,
        backend=backend,
        events=len(preds),
        quarters=len(ics),
        mean_ic=float(np.mean(ics)) if ics else 0.0,
        ic_t=_t(ics),
        spread=float(np.mean(spreads)) if spreads else 0.0,
        spread_t=_t(spreads),
        sign_hits=hits,
        sign_n=n,
        baselines={k: (float(np.mean(v)) if v else 0.0, _t(v)) for k, v in base_ics.items()},
    )


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    z, p = 1.96, k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def format_results(results: list[TaskResult]) -> str:
    out = [
        f"{'task':<40} {'OOS ev':>7} {'qtrs':>5} {'IC':>7} {'t':>5} "
        f"{'D10-D1%':>8} {'t':>5} {'sign acc (95% CI)':>20}  baselines: IC (t)"
    ]
    for r in results:
        lo, hi = wilson(r.sign_hits, r.sign_n)
        base = "  ".join(f"{k} {ic:+.3f} ({t:+.1f})" for k, (ic, t) in r.baselines.items())
        out.append(
            f"{r.task:<40} {r.events:>7} {r.quarters:>5} {r.mean_ic:>+7.3f} {r.ic_t:>+5.1f} "
            f"{r.spread:>+8.2f} {r.spread_t:>+5.1f} {r.sign_acc:>8.1%} ({lo:.1%}–{hi:.1%})  {base}"
        )
    backend = results[0].backend if results else ""
    out.append(
        f"backend: {backend}. OOS = walk-forward from 2019, expanding window, "
        "embargo > label window."
    )
    out.append(
        "sign acc: share of |target| > 1% events whose predicted sign was right (50% = coin)."
    )
    return "\n".join(out)
