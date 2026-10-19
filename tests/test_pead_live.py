from llm_earnings_agent.pead import live


def test_events_from_results_keeps_released_reports_with_consensus():
    results = [
        {
            "symbol": "GS",
            "earnings_time": "bmo",
            "drift_features": {
                "announcement_date": "2026-10-13",
                "eps_surprise_pct": 31.9,
                "gap_ret_pct": 3.2,
            },
        },
        {"symbol": "OLD"},  # not reported / no consensus surprise
        {"symbol": "NOCON", "drift_features": {"announcement_date": "2026-10-13"}},
        # Released after the close; the reaction-day open isn't known yet: Signal 1 only.
        {"symbol": "AMC", "earnings_date": "2026-10-20", "eps_surprise_pct": -4.0},
    ]
    evs = live.events_from_results(results)
    assert [e["symbol"] for e in evs] == ["GS", "AMC"]
    assert evs[0]["has_open"] and not evs[1]["has_open"]
    assert evs[1]["features"] == {"announcement_date": "2026-10-20", "eps_surprise_pct": -4.0}


def test_surprise_calibration_buckets():
    hist = [
        {
            "features": {
                "announcement_date": "2022-01-10",
                "eps_surprise_pct": s,
                "reaction_ret_pct": r,
            }
        }
        for s, r in [(15, 3), (12, -1), (14, 2), (-5, -2), (-4, 1), (1, 0.5)]
    ] + [
        {
            "features": {
                "announcement_date": "2019-01-10",
                "eps_surprise_pct": 15,
                "reaction_ret_pct": -9,
            }
        }
    ]
    cal = live.surprise_calibration(hist)
    b = live.bucket_for(cal, 13.0)
    assert b.n == 3 and abs(b.p_up - 2 / 3) < 1e-9  # pre-2021 event excluded
    assert live.bucket_for(cal, -4.5).p_up == 0.5


def test_lean_by_decile():
    assert live.lean(9) == "LONG"
    assert live.lean(8) == live.lean(5) == live.lean(0) == "no trade"


def test_signal1_label():
    assert live.signal1_label(0.63) == "Positive"
    assert live.signal1_label(0.34) == "Negative"
    assert live.signal1_label(0.50) == "Neutral"
