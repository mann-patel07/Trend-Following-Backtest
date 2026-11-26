"""Tests for walkforward.py on synthetic data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import walkforward as wf  # noqa: E402

PAIRS = [(5, 50), (10, 50), (20, 100), (30, 100)]


def synthetic_prices(seed=3):
    idx = pd.bdate_range("2019-01-01", "2025-12-31")
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(idx)))), index=idx)


def flat_rf(index):
    return pd.Series(0.04 / 252, index=index)


def test_choice_for_a_year_does_not_use_that_year_or_later():
    prices = synthetic_prices()
    rf = flat_rf(prices.index)
    base = wf.run_pairs(prices, rf, PAIRS)
    changed = prices.copy()
    later = changed.index.year >= 2023
    changed[later] = changed[later] * np.random.default_rng(9).uniform(0.5, 2.0, later.sum())
    alt = wf.run_pairs(changed, flat_rf(changed.index), PAIRS)
    assert wf.choose_pair(base, rf, 2023) == wf.choose_pair(alt, rf, 2023)
    # the check has teeth: the choice for 2025 does use 2023 and 2024
    assert wf.choose_pair(base, rf, 2025)[2] != wf.choose_pair(alt, rf, 2025)[2]


def test_stitching_one_pair_equals_the_backtest():
    prices = synthetic_prices()
    rf_daily = flat_rf(prices.index)
    results = wf.run_pairs(prices, rf_daily, [(20, 50)])
    ret = prices.pct_change()
    days = prices.index[(prices.index.year >= 2021) & (prices.index.year <= 2025)]
    choice = {y: (20, 50) for y in wf.TEST_YEARS}
    _, _, r = wf.stitch(ret, rf_daily, results, choice, days)
    assert np.max(np.abs(r.to_numpy() - results[(20, 50)]["strat_ret"][days].to_numpy())) < 1e-12


def fake_result(index, position):
    return pd.DataFrame({"position": position}, index=index)


def test_cost_at_a_year_boundary_only_when_the_position_changes():
    idx = pd.bdate_range("2020-12-01", "2022-12-31")
    days = idx[(idx.year >= 2021) & (idx.year <= 2022)]
    ret = pd.Series(0.0, index=idx)
    rf = pd.Series(0.0, index=idx)
    always_in = np.ones(len(idx))
    out_in_2022 = np.where(idx.year == 2022, 0.0, 1.0)
    cost_bps = 5.0
    # same position at the boundary: no cost at all
    results = {"A": fake_result(idx, always_in), "B": fake_result(idx, always_in)}
    _, _, r = wf.stitch(ret, rf, results, {2021: "A", 2022: "B"}, days, cost_bps)
    assert np.allclose(r.to_numpy(), 0.0, atol=0, rtol=0)
    # the new pair is flat at the boundary: exactly one cost, on the first day of 2022
    results = {"A": fake_result(idx, always_in), "B": fake_result(idx, out_in_2022)}
    _, _, r = wf.stitch(ret, rf, results, {2021: "A", 2022: "B"}, days, cost_bps)
    first_2022 = days[days.year == 2022][0]
    assert r[first_2022] == pytest.approx(-cost_bps / 1e4, abs=1e-15)
    assert (r.drop(first_2022) == 0).all()


def test_ties_go_to_the_smaller_long_window_then_the_smaller_short_window():
    prices = synthetic_prices()
    rf = flat_rf(prices.index)
    base = wf.run_pairs(prices, rf, [(20, 50)])[(20, 50)]
    results = {(10, 100): base, (30, 50): base, (20, 50): base, (5, 150): base}
    assert wf.choose_pair(results, rf, 2022)[:2] == (20, 50)
    results = {(10, 100): base, (30, 50): base, (5, 150): base}
    assert wf.choose_pair(results, rf, 2022)[:2] == (30, 50)


def test_sign_test():
    assert wf.sign_test_p(14, 28) == 1.0
    assert wf.sign_test_p(0, 10) == pytest.approx(2 / 1024)
    assert wf.sign_test_p(3, 29) == pytest.approx(1.5236437320709229e-05)
