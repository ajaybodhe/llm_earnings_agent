from llm_earnings_agent.pead import context as cx


def _ev(sym, date, rday, gap, rx, surprise=None, sue=None):
    return {
        "symbol": sym,
        "features": {
            "announcement_date": date,
            "reaction_day": rday,
            "gap_ret_pct": gap,
            "reaction_ret_pct": rx,
            "eps_surprise_pct": surprise,
            "sue": sue,
        },
    }


def test_add_context_is_point_in_time():
    evs = [
        _ev("A", "2024-01-10", "2024-01-11", 1.0, 4.0, sue=1.0),
        _ev("A", "2024-04-10", "2024-04-11", -2.0, -6.0, sue=-1.0),
        _ev("A", "2024-07-10", "2024-07-11", 3.0, 2.0, surprise=5.0, sue=-2.0),
    ]
    spy = {"2024-07-10": (100.0, 101.0), "2024-07-11": (102.0, 102.51)}
    vix = {"2024-07-10": (14.0, 15.5)}
    cx.add_context(evs, spy, vix)
    f0, f2 = evs[0]["features"], evs[2]["features"]
    assert "gap_vs_typical" not in f0  # needs ≥2 earlier reactions
    assert f2["gap_vs_typical"] == 3.0 / ((4.0 + 6.0) / 2)  # only the two earlier events
    # Consensus surprise (+5) takes precedence over SUE (−2) for the sign.
    assert f2["surprise_gap_agree"] == 1.0 and f2["gap_x_surprise"] == 3.0
    assert abs(f2["spy_gap_pct"] - (102 / 101 - 1) * 100) < 1e-9
    assert abs(f2["spy_oc_pct"] - 0.5) < 1e-9
    assert f2["vix_prev"] == 15.5


def test_gap_disagreeing_with_surprise():
    evs = [_ev("B", "2024-01-10", "2024-01-11", 4.0, 1.0, surprise=-3.0)]
    cx.add_context(evs, {}, {})
    assert evs[0]["features"]["surprise_gap_agree"] == -1.0
    assert evs[0]["features"]["gap_x_surprise"] == -4.0
