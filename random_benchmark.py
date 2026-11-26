"""Random-timing benchmark for the moving average crossover. Added after the first version (see UPGRADE_PLAN.md).

For every sweep cell, 500 random long-or-flat paths (two-state Markov chain) have the same share of days in the
market and the same average holding period as the strategy, pay the same costs and earn the same cash rate.
"""
import json
import os
import time

import numpy as np
import pandas as pd

import backtest as bt
import sweep

N_PATHS = 500
SEED_BASE = 20261126


def markov_paths(p, H, n_days, n_paths, rng):
    """Boolean array (n_days, n_paths) of in-market states.

    q = 1 / H is the chance of leaving on a day when in. a = q p / (1 - p) is the chance of entering on a day
    when out. Day 0 is in with probability p. One uniform draw per day and path is used for both cases.
    Returns None if a > 1.
    """
    q = 1.0 / H
    a = q * p / (1.0 - p)
    if a > 1.0:
        return None
    x = np.zeros((n_days, n_paths), dtype=bool)
    x[0] = rng.random(n_paths) < p
    u = rng.random((n_days - 1, n_paths))
    for t in range(1, n_days):
        prev = x[t - 1]
        x[t] = np.where(prev, u[t - 1] >= q, u[t - 1] < a)
    return x


def random_metrics(ret, rf, x, cost_bps):
    """CAGR and Sharpe of every random path. ret and rf are arrays of length n_days, x is (n_days, n_paths)."""
    xf = x.astype(float)
    prev = np.vstack([np.zeros((1, xf.shape[1])), xf[:-1]])
    r = ret[:, None] * xf + rf[:, None] * (1 - xf) - np.abs(xf - prev) * cost_bps / 1e4
    n = len(ret)
    cagr = np.prod(1 + r, axis=0) ** (252 / n) - 1
    ex = r - rf[:, None]
    sharpe = ex.mean(axis=0) / ex.std(axis=0, ddof=1) * np.sqrt(252)
    return cagr, sharpe


def percentile(strategy_value, random_values):
    """Percentage of random paths with a value below the strategy's."""
    return float(100 * np.mean(np.asarray(random_values) < strategy_value))


def cell_stats(pos):
    p = float(pos.mean())
    entries = int(((pos[1:] == 1) & (pos[:-1] == 0)).sum())
    H = float(pos.sum() / entries) if entries else float("nan")
    return p, entries, H


def skip_reason(p, entries):
    if entries == 0:
        return "no entries"
    if p <= 0:
        return "never in the market"
    if p >= 1:
        return "always in the market"
    return ""


def run_cell(res, rf_daily, i, short, long, ticker, sweep_cagr=None):
    pos = res["position"].to_numpy()
    strat_cagr = bt.cagr(res["equity_strategy"])
    if sweep_cagr is not None:
        assert abs(strat_cagr - sweep_cagr) < 1e-12, f"strategy CAGR differs from the sweep at {ticker} {short}/{long}"
    summ = bt.summarize(res["equity_strategy"], res["strat_ret"].dropna(), rf_daily)
    p, entries, H = cell_stats(pos)
    row = {"ticker": ticker, "short": short, "long": long, "strategy_cagr": strat_cagr,
           "strategy_sharpe": summ["Sharpe"], "p_in_market": p, "avg_holding_days": H,
           "random_median_cagr": np.nan, "random_p05_cagr": np.nan, "random_p95_cagr": np.nan,
           "cagr_percentile": np.nan, "sharpe_percentile": np.nan, "skipped": False, "skip_reason": ""}
    reason = skip_reason(p, entries)
    x = None
    if not reason:
        rng = np.random.default_rng([SEED_BASE, i, short, long])
        x = markov_paths(p, H, len(pos), N_PATHS, rng)
        if x is None:
            reason = "entry probability above 1"
    if reason:
        row["skipped"], row["skip_reason"] = True, reason
        return row
    rf = rf_daily.reindex(res.index).ffill().to_numpy()
    cagr, sharpe = random_metrics(res["ret"].to_numpy(), rf, x, sweep.COST_BPS)
    row.update({"random_median_cagr": float(np.median(cagr)), "random_p05_cagr": float(np.percentile(cagr, 5)),
                "random_p95_cagr": float(np.percentile(cagr, 95)),
                "cagr_percentile": percentile(strat_cagr, cagr),
                "sharpe_percentile": percentile(summ["Sharpe"], sharpe)})
    return row


def plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    used = df[~df.skipped]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(used.cagr_percentile, bins=np.arange(0, 100.1, 5), color="steelblue", edgecolor="white")
    ax.axhline(len(used) / 20, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Percentile of the strategy CAGR among 500 random paths")
    ax.set_ylabel("Number of cells")
    ax.set_title("Strategy CAGR percentile among 500 random-timing paths")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    t0 = time.time()
    sw = pd.read_csv("results/sweep_results.csv").set_index(["ticker", "short", "long"])
    rows = []
    for i, ticker in enumerate(sweep.TICKERS):
        f = f"data/{ticker}_{sweep.START}_{sweep.END}.csv"
        if not os.path.exists(f):
            raise SystemExit(f"missing price file {f}; nothing is downloaded")
        prices = bt.load_prices(ticker, sweep.START, sweep.END)
        rf_daily = bt.load_risk_free(prices.index)
        for short in sweep.SHORT_WINDOWS:
            for long in sweep.LONG_WINDOWS:
                if short >= long:
                    continue
                res = bt.backtest(prices, short, long, cost_bps=sweep.COST_BPS, rf_daily=rf_daily, lag=1)
                rows.append(run_cell(res, rf_daily, i, short, long, ticker,
                                     sweep_cagr=sw.loc[(ticker, short, long), "strategy_cagr"]))
    df = pd.DataFrame(rows)
    cols = ["ticker", "short", "long", "strategy_cagr", "strategy_sharpe", "p_in_market", "avg_holding_days",
            "random_median_cagr", "random_p05_cagr", "random_p95_cagr", "cagr_percentile", "sharpe_percentile",
            "skipped", "skip_reason"]
    df[cols].to_csv("results/random_benchmark.csv", index=False)
    used = df[~df.skipped]
    aapl = df[(df.ticker == "AAPL") & (df.short == 20) & (df.long == 50)].iloc[0]
    summary = {
        "cells_used": int(len(used)), "cells_skipped": int(df.skipped.sum()),
        "median_cagr_percentile": float(used.cagr_percentile.median()),
        "share_cagr_percentile_above_50": float((used.cagr_percentile > 50).mean()),
        "share_cagr_percentile_above_95": float((used.cagr_percentile > 95).mean()),
        "share_cagr_percentile_below_5": float((used.cagr_percentile < 5).mean()),
        "median_sharpe_percentile": float(used.sharpe_percentile.median()),
        "median_cagr_percentile_by_ticker": {k: float(v) for k, v in used.groupby("ticker").cagr_percentile.median().items()},
        "aapl_20_50": json.loads(aapl.to_json(double_precision=15)),
    }
    with open("results/random_benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    plot(df, "Images/random_percentiles.png")
    print({k: v for k, v in summary.items() if k not in ("median_cagr_percentile_by_ticker", "aapl_20_50")})
    print("run time %.1f s" % (time.time() - t0))


if __name__ == "__main__":
    main()
