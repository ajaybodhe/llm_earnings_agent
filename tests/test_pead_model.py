from datetime import date, timedelta

import numpy as np

from llm_earnings_agent.pead import model as pm


def _events(beta=1.0, seed=0, quarters=28, per_q=150):
    """Synthetic events 2017Q1… with reaction = beta·sue + noise."""
    rng = np.random.default_rng(seed)
    evs = []
    for q in range(quarters):
        y, m = 2017 + q // 4, 1 + 3 * (q % 4)
        for i in range(per_q):
            d = date(y, m, 1) + timedelta(days=int(rng.integers(0, 85)))
            sue = rng.normal()
            evs.append(
                {
                    "event_id": f"S{i}|{d}",
                    "symbol": f"S{i}",
                    "sic": 3571,
                    "features": {
                        "announcement_date": d.isoformat(),
                        "sue": sue,
                        "pre7_ret": float((d - date(2017, 1, 1)).days),  # encodes the date
                        "reaction_ret_pct": beta * sue + rng.normal() * 2,
                        "gap_ret_pct": 0.0,
                    },
                    "labels": {"drift20_adj": rng.normal()},
                }
            )
    return evs


def test_walk_forward_never_trains_on_the_future(monkeypatch):
    seen = []

    class Spy:
        def fit(self, X, y):
            self.max_day = np.nanmax(X[:, 1])  # pre7_ret column = days since 2017-01-01
            return self

        def predict(self, X):
            seen.append((self.max_day, np.nanmin(X[:, 1])))
            return np.zeros(len(X))

    monkeypatch.setattr(pm, "make_model", lambda cat: (Spy(), "spy"))
    task = pm.TASKS["reaction"]
    pm.walk_forward(_events(), task, first_test="2019Q1", min_train=100)
    assert seen, "no folds ran"
    for train_max, test_min in seen:
        # Every training event precedes every test event by more than the embargo.
        assert test_min - train_max > task.embargo_days


def test_walk_forward_finds_real_signal_out_of_sample():
    res, _ = pm.walk_forward(_events(beta=1.5), pm.TASKS["reaction"], min_train=500)
    assert res.quarters >= 15 and res.mean_ic > 0.3 and res.ic_t > 5 and res.sign_acc > 0.6
    assert res.baselines["sue"][0] > 0.3


def test_walk_forward_finds_nothing_in_noise():
    res, _ = pm.walk_forward(_events(beta=0.0, seed=3), pm.TASKS["reaction"], min_train=500)
    assert abs(res.ic_t) < 3


def test_open_to_close():
    e = {"features": {"reaction_ret_pct": 10.0, "gap_ret_pct": 5.0}}
    assert abs(pm._open_to_close(e) - (1.10 / 1.05 - 1) * 100) < 1e-9


def test_release_features_exclude_reaction_day_information():
    leaky = {"reaction_ret_pct", "gap_ret_pct", "abnormal_volume", "sector_ret_1m", "market_ret_1m"}
    assert not leaky & set(pm.RELEASE_FEATURES)
    assert not (leaky - {"gap_ret_pct"}) & set(pm.OPEN_FEATURES)


def test_ridge_keeps_monotonic_signal_ranking():
    res, _ = pm.walk_forward(_events(beta=1.5), pm.TASKS["reaction"], min_train=500, kind="ridge")
    assert res.backend == "ridge" and res.mean_ic > 0.3
