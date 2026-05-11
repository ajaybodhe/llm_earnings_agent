import datetime as dt

from llm_earnings_agent.backtest import _baseline_hit_rate, _label_from_return
from llm_earnings_agent.data.prices import PricePoint, one_day_return_pct


def test_label_from_return_deadband():
    assert _label_from_return(2.0) == "Positive"
    assert _label_from_return(-2.0) == "Negative"
    assert _label_from_return(0.5) == "Neutral"
    assert _label_from_return(-0.5) == "Neutral"
    assert _label_from_return(1.0001) == "Positive"


def test_baseline_hit_rate():
    assert _baseline_hit_rate([]) == 0.0
    assert _baseline_hit_rate(["Positive", "Positive", "Negative", "Neutral"]) == 0.5
    assert _baseline_hit_rate(["Neutral"] * 4) == 1.0


def test_one_day_return_pct_basic():
    base = dt.date(2024, 1, 1)
    prices = [
        PricePoint(base, 100, 100),
        PricePoint(base + dt.timedelta(days=1), 102, 105),  # announcement-day close
    ]
    ret = one_day_return_pct(prices, base + dt.timedelta(days=1))
    assert ret is not None
    assert abs(ret - 5.0) < 1e-9


def test_one_day_return_pct_missing_data():
    base = dt.date(2024, 1, 1)
    prices = [PricePoint(base, 100, 100)]  # only one bar
    assert one_day_return_pct(prices, base + dt.timedelta(days=1)) is None
    assert one_day_return_pct([], base) is None
