import numpy as np

from llm_earnings_agent.pead import suetxt


def test_preprocess_hides_numbers():
    assert (
        suetxt.preprocess("Revenue rose to $1,234.5 million in Q3")
        == "revenue rose to $# million in q#"
    )


def test_prev_quarters():
    assert suetxt._prev_quarters("2019Q2", 3) == ["2019Q1", "2018Q4", "2018Q3"]


def test_labels_split_flat_by_absolute_tercile():
    ar = np.array([-5.0, -0.2, 0.1, 0.3, 4.0, -3.0])
    y = suetxt._labels(ar, flat_cut=0.3)
    assert list(y) == [0, 1, 1, 1, 2, 0]  # L, F, F, F, H, L


def test_abnormal_reaction_subtracts_spy_same_day():
    e = {"features": {"reaction_ret_pct": 5.0, "reaction_day": "2026-10-14"}}
    spy = {"2026-10-13": 100.0, "2026-10-14": 101.0}
    assert abs(suetxt.abnormal_reaction(e, spy, sorted(spy)) - 4.0) < 1e-9


def _events_and_texts(n_q=12, per_q=250, seed=0):
    rng = np.random.default_rng(seed)
    good = "record revenue raised guidance strong demand momentum"
    bad = "impairment charge lowered outlook weak demand headwinds"
    evs, texts, ar = [], {}, {}
    for q in range(n_q):
        y, m = 2016 + q // 4, 1 + 3 * (q % 4)
        for i in range(per_q):
            eid = f"S{i}|{y}-{m:02d}-10|{q}"
            tone = rng.choice([-1, 1])
            filler = " ".join(
                rng.choice(["company", "quarter", "results", "segment", "operations"], 80)
            )
            texts[eid] = f"{filler} {(good if tone > 0 else bad)} {filler}"
            ar[eid] = tone * 3 + rng.normal()
            evs.append({"event_id": eid, "features": {"announcement_date": f"{y}-{m:02d}-10"}})
    return evs, texts, ar


def test_rolling_sue_txt_learns_tone_out_of_sample(monkeypatch):
    monkeypatch.setattr(suetxt, "N_UNIGRAMS", 50)
    monkeypatch.setattr(suetxt, "N_BIGRAMS", 50)
    evs, texts, ar = _events_and_texts()
    sue = suetxt.rolling_sue_txt(
        evs, texts, ar, first_test="2018Q1", c_grid=(1.0,), progress=lambda m: None
    )
    assert sue, "no out-of-sample scores"
    assert all(q >= "2018" for q in (k.split("|")[1] for k in sue))  # only test quarters scored
    s = np.array([sue[k] for k in sue])
    a = np.array([ar[k] for k in sue])
    assert np.corrcoef(s, a)[0, 1] > 0.5
