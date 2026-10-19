import numpy as np

from llm_earnings_agent.pead import gate


def _events(n_quarters=12, per_q=200, beta=1.0, seed=0):
    rng = np.random.default_rng(seed)
    evs = []
    for q in range(n_quarters):
        year, month = 2018 + q // 4, 1 + 3 * (q % 4)
        for i in range(per_q):
            s = rng.normal()
            y = beta * s + rng.normal() * 3
            evs.append(
                {
                    "symbol": f"S{i}",
                    "features": {"announcement_date": f"{year}-{month:02d}-15", "sue": s},
                    "labels": {"drift20_adj": y},
                }
            )
    return evs


def test_spearman_matches_known_values():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert gate.spearman(x, x * 10) == 1.0
    assert gate.spearman(x, -x) == -1.0
    assert abs(gate.spearman(np.array([1.0, 1.0, 2.0]), np.array([1.0, 2.0, 3.0])) - 0.866) < 1e-3


def test_gate_detects_a_real_signal():
    row = gate.evaluate_signal(
        _events(beta=1.0), gate.SIGNALS["sue (winsorised ±5)"], "drift20_adj"
    )
    assert row.quarters == 12 and row.events == 2400
    assert row.mean_ic > 0.2 and row.ic_t > 5 and row.spread > 0 and row.spread_t > 5


def test_gate_finds_nothing_in_noise():
    row = gate.evaluate_signal(
        _events(beta=0.0, seed=1), gate.SIGNALS["sue (winsorised ±5)"], "drift20_adj"
    )
    assert abs(row.ic_t) < 3 and abs(row.spread_t) < 3


def test_small_quarters_are_skipped():
    row = gate.evaluate_signal(
        _events(per_q=10), gate.SIGNALS["sue (winsorised ±5)"], "drift20_adj"
    )
    assert row.quarters == 0 and row.events == 0


def test_format_gate_flags_significant_rows():
    rows = [
        gate.evaluate_signal(_events(), gate.SIGNALS["sue (winsorised ±5)"], "drift20_adj", "sue")
    ]
    assert "*" in gate.format_gate(rows).splitlines()[1]
