"""Walk-forward test of the moving average crossover. Added after the first version (see UPGRADE_PLAN.md).

Each year Y from 2021 to 2025, the window pair with the highest Sharpe ratio (excess over T-bills) over
2020-01-01 to the end of year Y-1 is chosen and traded during year Y. Costs and fills are as in sweep.py.
"""
import json
import math
import os
import time

import numpy as np
import pandas as pd

import backtest as bt
import sweep

FIRST_TRAIN_YEAR = 2020
TEST_YEARS = [2021, 2022, 2023, 2024, 2025]


def pair_list():
    return [(s, l) for s in sweep.SHORT_WINDOWS for l in sweep.LONG_WINDOWS if s < l]


def run_pairs(prices, rf_daily, pairs, cost_bps=sweep.COST_BPS):
    """backtest() result for every pair, lag 1, exactly as in the sweep."""
    return {p: bt.backtest(prices, p[0], p[1], cost_bps=cost_bps, rf_daily=rf_daily, lag=1) for p in pairs}


def choose_pair(results, rf, year, first_train_year=FIRST_TRAIN_YEAR):
    """Pair with the highest training Sharpe. Ties go to the smaller long window, then the smaller short window.

    Returns (short, long, train_sharpe). Only days from first_train_year to year - 1 are used.
    """
    best = None
    for (s, l), res in results.items():
        days = res.index[(res.index.year >= first_train_year) & (res.index.year <= year - 1)]
        score = round(float(bt.sharpe(res["strat_ret"][days], rf[days])), 12)
        # A pair that is flat on every training day has zero excess return and an undefined Sharpe ratio.
        # It ranks below every pair with a defined score.
        key = (math.inf if math.isnan(score) else -score, l, s)
        if best is None or key < best[0]:
            best = (key, s, l, score)
    return best[1], best[2], best[3]


def stitch(ret, rf, results, choice_by_year, test_days, cost_bps=sweep.COST_BPS):
    """Stitched position and daily return over test_days.

    pos[t] is the chosen pair's position on day t. The position before the first test day is the pair chosen
    for the first test year, taken on the last trading day before it.
    """
    years = test_days.year
    pos = pd.Series(0.0, index=test_days)
    for y in sorted(set(years)):
        res = results[choice_by_year[y]]
        m = years == y
        pos[m] = res["position"].reindex(test_days[m]).to_numpy()
    first = min(set(years))
    res0 = results[choice_by_year[first]]
    before = res0["position"][res0.index < test_days[0]].iloc[-1]
    prev = pos.shift(1)
    prev.iloc[0] = before
    r = ret[test_days] * pos + rf[test_days] * (1 - pos) - (pos - prev).abs() * cost_bps / 1e4
    return pos, prev, r


def metrics(r, rf_days):
    equity = (1 + r).cumprod()
    return bt.cagr(equity), bt.sharpe(r, rf_days), bt.max_drawdown(equity)[1]


def sign_test_p(k, n):
    """Two-sided exact binomial p-value against p = 0.5."""
    lower = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    upper = sum(math.comb(n, i) for i in range(k, n + 1)) / 2 ** n
    return min(1.0, 2 * min(lower, upper))


def walkforward_ticker(prices, rf_daily, pairs=None, test_years=TEST_YEARS):
    """Full walk-forward for one ticker. Returns (row dict, choices list)."""
    pairs = pairs or pair_list()
    results = run_pairs(prices, rf_daily, pairs)
    ret = prices.pct_change()
    rf = rf_daily.reindex(prices.index).ffill()
    idx = prices.index
    for res in results.values():
        need = idx[(idx.year >= FIRST_TRAIN_YEAR) & (idx.year <= test_years[-1])]
        assert res.index.isin(need).sum() == len(need), "backtest result misses trading days"
    test_days = idx[(idx.year >= test_years[0]) & (idx.year <= test_years[-1])]
    choice_by_year, choices = {}, []
    for y in test_years:
        s, l, sc = choose_pair(results, rf, y)
        choice_by_year[y] = (s, l)
        choices.append({"year": y, "short": s, "long": l, "train_sharpe": sc})
    pos, prev, r = stitch(ret, rf, results, choice_by_year, test_days)
    wf = metrics(r, rf[test_days])
    bh = metrics(ret[test_days], rf[test_days])
    fixed = metrics(results[(20, 50)]["strat_ret"][test_days], rf[test_days]) if (20, 50) in results else (np.nan,) * 3
    hind = max(results, key=lambda p: metrics(results[p]["strat_ret"][test_days], rf[test_days])[0])
    hind_cagr = metrics(results[hind]["strat_ret"][test_days], rf[test_days])[0]
    row = {
        "wf_cagr": wf[0], "wf_sharpe": wf[1], "wf_max_drawdown": wf[2],
        "bh_cagr": bh[0], "bh_sharpe": bh[1], "bh_max_drawdown": bh[2],
        "fixed_20_50_cagr": fixed[0], "fixed_20_50_sharpe": fixed[1], "fixed_20_50_max_drawdown": fixed[2],
        "hindsight_best_pair": f"{hind[0]}/{hind[1]}", "hindsight_best_cagr": hind_cagr,
        "wf_pct_days_in_market": float(pos.mean() * 100),
        "wf_round_trips": float((pos - prev).abs().sum() / 2),
        "wf_pair_changes": sum(choice_by_year[y] != choice_by_year[y - 1] for y in test_years[1:]),
    }
    return row, choices


def plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = (df.wf_cagr - df.bh_cagr) * 100
    d.index = df.ticker
    d = d.sort_values()
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(d.index, d.values, color="steelblue")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Walk-forward CAGR minus buy-and-hold CAGR (percentage points)")
    ax.set_ylabel("Ticker")
    ax.set_title("Walk-forward CAGR minus buy-and-hold, 2021 to 2025")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    t0 = time.time()
    for t in sweep.TICKERS:
        f = f"data/{t}_{sweep.START}_{sweep.END}.csv"
        if not os.path.exists(f):
            raise SystemExit(f"missing price file {f}; nothing is downloaded")
    rows, choices = [], []
    for t in sweep.TICKERS:
        prices = bt.load_prices(t, sweep.START, sweep.END)
        rf_daily = bt.load_risk_free(prices.index)
        row, ch = walkforward_ticker(prices, rf_daily)
        rows.append({"ticker": t, **row})
        choices += [{"ticker": t, **c} for c in ch]
    df = pd.DataFrame(rows)
    cols = ["ticker", "wf_cagr", "wf_sharpe", "wf_max_drawdown", "bh_cagr", "bh_sharpe", "bh_max_drawdown",
            "fixed_20_50_cagr", "fixed_20_50_sharpe", "fixed_20_50_max_drawdown", "hindsight_best_pair",
            "hindsight_best_cagr", "wf_pct_days_in_market", "wf_round_trips", "wf_pair_changes"]
    df[cols].to_csv("results/walkforward_by_ticker.csv", index=False)
    pd.DataFrame(choices, columns=["ticker", "year", "short", "long", "train_sharpe"]).to_csv(
        "results/walkforward_choices.csv", index=False)
    n = len(df)
    k = int((df.wf_sharpe > df.bh_sharpe).sum())
    summary = {
        "n_tickers": n,
        "wf_beats_bh_sharpe": k,
        "wf_beats_bh_cagr": int((df.wf_cagr > df.bh_cagr).sum()),
        "wf_beats_bh_drawdown": int((df.wf_max_drawdown > df.bh_max_drawdown).sum()),
        "sign_test_p_sharpe": sign_test_p(k, n),
        "median_wf_minus_bh_cagr": float((df.wf_cagr - df.bh_cagr).median()),
        "median_wf_minus_bh_sharpe": float((df.wf_sharpe - df.bh_sharpe).median()),
        "median_wf_minus_hindsight_cagr": float((df.wf_cagr - df.hindsight_best_cagr).median()),
        "wf_beats_fixed_cagr": int((df.wf_cagr > df.fixed_20_50_cagr).sum()),
        "wf_beats_fixed_sharpe": int((df.wf_sharpe > df.fixed_20_50_sharpe).sum()),
        "aapl": json.loads(df[df.ticker == "AAPL"].iloc[0].to_json(double_precision=15)),
    }
    with open("results/walkforward_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    plot(df, "Images/walkforward.png")
    print(json.dumps({k_: v for k_, v in summary.items() if k_ != "aapl"}, indent=1))
    print("run time %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
