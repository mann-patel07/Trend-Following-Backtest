"""Tests for random_benchmark.py on synthetic data."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import random_benchmark as rb  # noqa: E402


def test_random_paths_match_the_time_in_market_and_holding_period():
    rng = np.random.default_rng(5)
    x = rb.markov_paths(0.6, 20.0, 20000, 200, rng)
    share = x.mean()
    entries = ((x[1:]) & (~x[:-1])).sum(axis=0)
    holding = x.sum(axis=0) / entries
    assert abs(share - 0.6) < 0.01
    assert abs(holding.mean() - 20.0) < 1.0


def test_percentile_at_the_extremes():
    random_values = np.array([0.01, 0.02, 0.03, 0.04])
    assert rb.percentile(0.5, random_values) == 100.0
    assert rb.percentile(-0.5, random_values) == 0.0
    assert rb.percentile(0.025, random_values) == 50.0


def test_same_seed_gives_identical_paths_and_a_different_seed_does_not():
    a = rb.markov_paths(0.5, 10.0, 500, 50, np.random.default_rng([1, 2, 3]))
    b = rb.markov_paths(0.5, 10.0, 500, 50, np.random.default_rng([1, 2, 3]))
    c = rb.markov_paths(0.5, 10.0, 500, 50, np.random.default_rng([1, 2, 4]))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_a_cell_that_is_always_in_the_market_is_skipped_with_a_reason():
    assert rb.skip_reason(1.0, 3) == "always in the market"
    assert rb.skip_reason(0.5, 0) == "no entries"
    assert rb.skip_reason(0.0, 2) == "never in the market"
    idx = pd.bdate_range("2020-01-01", periods=50)
    res = pd.DataFrame({"position": np.ones(50), "ret": 0.001, "strat_ret": 0.001,
                        "equity_strategy": (1.001) ** np.arange(1, 51)}, index=idx)
    row = rb.run_cell(res, pd.Series(0.0001, index=idx), 0, 5, 50, "TEST")
    assert row["skipped"] and row["skip_reason"] != ""


def test_entry_probability_above_one_returns_none():
    assert rb.markov_paths(0.9, 1.0, 10, 5, np.random.default_rng(0)) is None


def test_random_cagr_of_a_path_that_is_always_in_equals_buy_and_hold():
    ret = np.full(252, 0.001)
    rf = np.zeros(252)
    x = np.ones((252, 1), dtype=bool)
    cagr, _ = rb.random_metrics(ret, rf, x, 5.0)
    expected = np.prod(1 + ret - np.r_[5e-4, np.zeros(251)]) - 1
    assert abs(cagr[0] - expected) < 1e-12
