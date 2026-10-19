"""SUE.txt — a replication of Meursault, Liang, Routledge & Scanlon,
"PEAD.txt: Post-Earnings-Announcement Drift Using Text" (JFQA 2022 / Philadelphia
Fed WP 21-07), adapted to earnings press releases.

Method (as in the paper, section II.B and Online Appendix B):
* Bag of words: the 1,000 most frequent unigrams and 1,000 most frequent
  bigrams of the training window, x = log(1 + count); lowercase, stop words
  removed, numbers replaced by a placeholder ("$1,000.00" → "$#") so the
  reported figures themselves are not used.
* Target: the one-day abnormal return split into Flat (the 33% of training
  observations closest to zero), High (other positive) and Low (other
  negative); cutoffs come from the training window only.
* Model: multinomial logistic regression with elastic-net penalty (α = 0.5),
  re-estimated every quarter on the previous eight quarters; only
  out-of-sample predictions are used.
* SUE.txt = log-odds(High) − log-odds(Low).

Adaptations (documented, not hidden):
* Text: the earnings press release (EX-99.1), not the call transcript — there
  is no Q&A section, so features are not split by section.
* Abnormal return: reaction-day return minus SPY's same-day return (the paper
  uses size/book-to-market benchmark portfolios).
* λ: chosen once by cross-validation on the first training window (the paper
  re-runs 10-fold CV every quarter).
"""

from __future__ import annotations

import json
import logging
import re
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

N_UNIGRAMS = 1000
N_BIGRAMS = 1000
TRAIN_QUARTERS = 8
MAX_CHARS = 20000  # headline, narrative and guidance come first; tables follow
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def preprocess(text: str) -> str:
    """Lowercase and replace numbers with '#' (kept as part of the token)."""
    return _NUM.sub("#", text[:MAX_CHARS].lower())


def quarter_of(date: str) -> str:
    return f"{date[:4]}Q{(int(date[5:7]) - 1) // 3 + 1}"


def _prev_quarters(q: str, n: int) -> list[str]:
    y, k = int(q[:4]), int(q[-1])
    out = []
    for _ in range(n):
        k -= 1
        if k == 0:
            y, k = y - 1, 4
        out.append(f"{y}Q{k}")
    return out


def abnormal_reaction(
    e: dict[str, Any], spy_close: dict[str, float], spy_dates: list[str]
) -> float | None:
    """Reaction-day return minus SPY's close-to-close return that day."""
    f = e["features"]
    rx, day = f.get("reaction_ret_pct"), f.get("reaction_day")
    if rx is None or not day or day not in spy_close:
        return None
    i = bisect_left(spy_dates, day)
    if i == 0:
        return None
    prev = spy_close[spy_dates[i - 1]]
    return rx - (spy_close[day] / prev - 1) * 100


def build_counts(docs: list[str]) -> tuple[Any, Any, np.ndarray, np.ndarray]:
    """Corpus-wide unigram and bigram count matrices (vocabularies trimmed per
    window later). min_df keeps the matrices manageable."""
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer

    token = r"(?u)[\w#$%]*[a-z#$%][\w#$%]*"  # words incl. '#', '$#', '#%'
    uni = CountVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS), token_pattern=token, min_df=30, dtype=np.int32
    )
    bi = CountVectorizer(
        stop_words=list(ENGLISH_STOP_WORDS),
        token_pattern=token,
        ngram_range=(2, 2),
        min_df=30,
        dtype=np.int32,
    )
    Xu = uni.fit_transform(docs)
    Xb = bi.fit_transform(docs)
    return Xu, Xb, uni.get_feature_names_out(), bi.get_feature_names_out()


def _top_columns(X: Any, rows: np.ndarray, k: int) -> np.ndarray:
    totals = np.asarray(X[rows].sum(axis=0)).ravel()
    return np.argsort(-totals)[:k]


def _design(Xu: Any, Xb: Any, rows: np.ndarray, cu: np.ndarray, cb: np.ndarray) -> Any:
    from scipy.sparse import hstack

    X = hstack([Xu[rows][:, cu], Xb[rows][:, cb]]).tocsr().astype(np.float64)
    X.data = np.log1p(X.data)
    return X


def _labels(ar: np.ndarray, flat_cut: float) -> np.ndarray:
    return np.where(np.abs(ar) <= flat_cut, 1, np.where(ar > 0, 2, 0))  # 0=L 1=F 2=H


def rolling_sue_txt(
    events: list[dict[str, Any]],
    texts: dict[str, str],
    ar: dict[str, float],
    *,
    first_test: str = "2018Q4",
    c_grid: tuple[float, ...] = (0.01, 0.03, 0.1, 0.3),
    progress: Callable[[str], None] = logger.info,
) -> dict[str, float]:
    """Out-of-sample SUE.txt per event_id, re-estimated every quarter on the
    previous TRAIN_QUARTERS quarters (events with text and an abnormal return)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import MaxAbsScaler

    ids = [e["event_id"] for e in events if e["event_id"] in texts]
    quarter = {e["event_id"]: quarter_of(e["features"]["announcement_date"]) for e in events}
    progress(f"tokenising {len(ids)} press releases …")
    Xu, Xb, _, _ = build_counts([preprocess(texts[i]) for i in ids])
    q_rows: dict[str, list[int]] = defaultdict(list)
    for r, i in enumerate(ids):
        q_rows[quarter[i]].append(r)
    ar_vec = np.array([ar.get(i, np.nan) for i in ids])

    out: dict[str, float] = {}
    best_c: float | None = None
    for q in sorted(q for q in q_rows if q >= first_test):
        train = np.array(
            [r for p in _prev_quarters(q, TRAIN_QUARTERS) for r in q_rows.get(p, [])], dtype=int
        )
        train = train[~np.isnan(ar_vec[train])] if len(train) else train
        test = np.array(q_rows[q], dtype=int)
        if len(train) < 2000 or len(test) == 0:
            continue
        cu = _top_columns(Xu, train, N_UNIGRAMS)
        cb = _top_columns(Xb, train, N_BIGRAMS)
        Xtr, Xte = _design(Xu, Xb, train, cu, cb), _design(Xu, Xb, test, cu, cb)
        scaler = MaxAbsScaler().fit(Xtr)
        Xtr, Xte = scaler.transform(Xtr), scaler.transform(Xte)
        flat_cut = float(np.quantile(np.abs(ar_vec[train]), 1 / 3))
        ytr = _labels(ar_vec[train], flat_cut)

        def model(c: float) -> LogisticRegression:
            return LogisticRegression(
                penalty="elasticnet", l1_ratio=0.5, C=c, solver="saga", max_iter=300, tol=1e-3
            )

        if best_c is None:  # λ (= 1/C) by 5-fold CV on the first window
            scores = {
                c: cross_val_score(model(c), Xtr, ytr, cv=5, scoring="neg_log_loss").mean()
                for c in c_grid
            }
            best_c = max(scores, key=scores.get)
            progress(
                f"CV log-loss by C: { {c: round(s, 4) for c, s in scores.items()} } → C={best_c}"
            )
        m = model(best_c).fit(Xtr, ytr)
        logp = np.log(np.clip(m.predict_proba(Xte), 1e-12, 1))
        p_h, p_l = np.exp(logp[:, 2]), np.exp(logp[:, 0])
        sue = (np.log(p_h) - np.log1p(-p_h)) - (np.log(p_l) - np.log1p(-p_l))
        for r, s in zip(test, sue, strict=True):
            out[ids[r]] = float(s)
        progress(f"{q}: trained on {len(train)}, scored {len(test)}")
    return out


def save(path: Path, sue: dict[str, float]) -> None:
    path.write_text(
        "\n".join(json.dumps({"event_id": k, "sue_txt": v}) for k, v in sue.items()) + "\n"
    )


def load(path: Path) -> dict[str, float]:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["event_id"]] = d["sue_txt"]
    return out
