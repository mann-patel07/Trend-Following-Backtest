"""Tests for backtest.py on small made-up price series where the answer is known by hand."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import backtest as bt  # noqa: E402

COST = 100.0  # basis points, 1% so the cost is easy to see
COST_FRAC = COST / 1e4


def series(values):
    idx = pd.bdate_range("2020-01-01", periods=len(values))
    return pd.Series(values, index=idx, dtype=float)


def rising(n=30):
    return series(100 * 1.01 ** np.arange(n))


def zero_rf(prices):
    return pd.Series(0.0, index=prices.index)


def test_position_uses_only_the_previous_days_signal():
    rng = np.random.default_rng(0)
    prices = series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 80))))
    base = bt.backtest(prices, 3, 8)
    for k in (30, 50, 79):
        changed = prices.copy()
        changed.iloc[k] *= 1.5
        alt = bt.backtest(changed, 3, 8)
        # a price on day k reaches the signal on day k and the position on day k + 1,
        # so every position up to and including day k stays the same
        day = prices.index[k]
        assert base["position"].loc[:day].equals(alt["position"].loc[:day])
    # and the position is the signal one day later
    later = base["position"].iloc[1:].to_numpy()
    earlier_signal = base["signal"].iloc[:-1].to_numpy()
    assert (later == earlier_signal).all()


def test_long_from_the_first_possible_day_pays_the_entry_cost_once():
    prices = rising()
    res = bt.backtest(prices, 2, 3, cost_bps=COST, rf_daily=zero_rf(prices))
    assert res["signal"].iloc[0] == 1
    assert res["position"].iloc[0] == 0
    assert res["position"].iloc[1] == 1
    # day 0 is flat and earns the cash rate (0 here)
    assert res["strat_ret"].iloc[0] == pytest.approx(0.0)
    # the entry cost is charged on day 1, once
    assert res["strat_ret"].iloc[1] == pytest.approx(res["ret"].iloc[1] - COST_FRAC)
    assert res["strat_ret"].iloc[2:].values == pytest.approx(res["ret"].iloc[2:].values)


def test_an_exit_pays_the_cost_once():
    prices = series(list(100 * 1.01 ** np.arange(20)) + list(120 * 0.98 ** np.arange(1, 21)))
    res = bt.backtest(prices, 2, 3, cost_bps=COST, rf_daily=zero_rf(prices))
    gross = res["ret"] * res["position"]
    cost = gross - res["strat_ret"]
    change_days = res["position"].diff().abs().fillna(0) > 0
    assert change_days.sum() == 2  # one entry, one exit
    assert cost[change_days].values == pytest.approx([COST_FRAC, COST_FRAC])
    assert cost[~change_days].abs().max() == pytest.approx(0.0)


def test_a_flat_day_earns_exactly_the_cash_rate():
    prices = series(100 * 0.99 ** np.arange(30))  # falling, so the signal is always 0
    rf = pd.Series(0.0002, index=prices.index)
    res = bt.backtest(prices, 2, 3, cost_bps=COST, rf_daily=rf)
    assert (res["position"] == 0).all()
    assert res["strat_ret"].values == pytest.approx(np.full(len(res), 0.0002))
    assert res["equity_strategy"].iloc[-1] == pytest.approx(1.0002 ** len(res))


def test_strategy_and_buy_and_hold_cover_the_same_dates():
    rng = np.random.default_rng(1)
    prices = series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 120))))
    res = bt.backtest(prices, 5, 20)
    assert res["strat_ret"].notna().all()
    assert res["ret"].notna().all()
    assert res["equity_bh"].count() == res["equity_strategy"].count() == len(res)
    assert res["equity_bh"].index.equals(res["equity_strategy"].index)


def test_cagr_of_a_known_growth_rate():
    equity = pd.Series(1.1 ** (np.arange(1, 253) / 252))  # 252 days, ends at 1.1
    assert bt.cagr(equity) == pytest.approx(0.10)


def test_sharpe_is_mean_excess_over_std_times_sqrt_252():
    rng = np.random.default_rng(2)
    r = pd.Series(rng.normal(0.001, 0.01, 500))
    rf = pd.Series(0.0001, index=r.index)
    excess = r - rf
    assert bt.sharpe(r, rf) == pytest.approx(excess.mean() / excess.std() * np.sqrt(252))


def test_max_drawdown_counts_the_starting_capital():
    equity = series([0.9, 0.95, 1.0])
    dd_series, dd = bt.max_drawdown(equity)
    assert dd == pytest.approx(-0.10)
    assert dd_series.index.equals(equity.index)


def test_max_drawdown_after_a_peak_is_unchanged():
    equity = series([1.1, 1.2, 0.9, 1.0])
    _, dd = bt.max_drawdown(equity)
    assert dd == pytest.approx(0.9 / 1.2 - 1)


def test_trade_stats_counts_a_first_day_long_as_an_entry():
    prices = rising()
    res = bt.backtest(prices, 2, 3, cost_bps=COST, rf_daily=zero_rf(prices))
    stats = bt.trade_stats(res["position"])
    assert stats["AvgHoldingDays"] == pytest.approx(res["position"].sum())  # one entry
    assert stats["RoundTripTrades"] == pytest.approx(0.5)  # entered, never exited


def test_lag_two_position_is_the_signal_two_days_earlier():
    rng = np.random.default_rng(1)
    prices = series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 80))))
    base = bt.backtest(prices, 3, 8, lag=2)
    assert (base["position"].iloc[:2] == 0).all()
    assert (base["position"].iloc[2:].to_numpy() == base["signal"].iloc[:-2].to_numpy()).all()
    for k in (30, 50, 79):
        changed = prices.copy()
        changed.iloc[k] *= 1.5
        alt = bt.backtest(changed, 3, 8, lag=2)
        # a price on day k reaches the signal on day k and the position on day k + 2,
        # so every position up to and including day k + 1 stays the same
        day = prices.index[min(k + 1, 79)]
        assert base["position"].loc[:day].equals(alt["position"].loc[:day])


def test_default_lag_is_one():
    rng = np.random.default_rng(2)
    prices = series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 60))))
    pd.testing.assert_frame_equal(bt.backtest(prices, 3, 8), bt.backtest(prices, 3, 8, lag=1))


def test_lag_below_one_raises():
    with pytest.raises(ValueError):
        bt.backtest(rising(), 2, 3, lag=0)
