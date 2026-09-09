"""Parameter and universe sweep for the SMA crossover strategy.

Runs the same backtest as backtest.py across a broad ticker universe and a
grid of short/long SMA windows, to check whether v1's AAPL 20/50 result
(strategy underperforms buy-and-hold) holds generally or is specific to
AAPL's price path. Run as `python sweep.py`; writes results/sweep_results.csv.
"""

import os
import warnings

import pandas as pd

import backtest as bt

START = "2019-01-01"
END = "2026-01-01"
COST_BPS = 5.0

# 29 liquid, large-cap US tickers spanning consumer staples, energy,
# financials, healthcare, industrials, utilities, materials, and tech, plus
# SPY as a broad-market benchmark. Deliberately not tech-heavy: AAPL is
# included so the v1 20/50 result can be located inside this distribution,
# but tech is otherwise capped at two names (AAPL, MSFT) so the sweep isn't
# dominated by one sector's growth profile the way the AAPL-only v1 test was.
#
# NOTE ON SURVIVORSHIP BIAS: this is today's list of liquid, still-listed
# large caps. Companies that were delisted, went bankrupt, or were acquired
# during 2019-2026 aren't in this universe, so the sweep skews toward
# "the market's winners" rather than a neutral cross-section of everything
# that existed in 2019. Any edge found here is an upper bound, not a clean
# estimate -- a survivorship-bias-free universe would need point-in-time
# constituent data this project doesn't have.
TICKERS = [
    # Consumer staples
    "KO", "PG", "WMT", "PEP", "COST",
    # Energy
    "XOM", "CVX", "COP",
    # Financials
    "JPM", "BAC", "WFC", "GS",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV",
    # Industrials
    "BA", "CAT", "HON", "UPS",
    # Utilities
    "NEE", "DUK", "SO",
    # Materials
    "LIN", "APD", "FCX",
    # Tech
    "AAPL", "MSFT",
    # Broad market benchmark
    "SPY",
]

SHORT_WINDOWS = [5, 10, 20, 30, 50]
LONG_WINDOWS = [50, 100, 150, 200]


def run_sweep():
    """Backtest every (ticker, short, long) cell in the grid and return a results DataFrame."""
    rows = []

    for ticker in TICKERS:
        try:
            prices = bt.load_prices(ticker, START, END)
        except Exception as e:
            warnings.warn(f"Skipping {ticker}: price download failed ({e})")
            continue

        if prices.empty:
            warnings.warn(f"Skipping {ticker}: no price data returned")
            continue

        rf_daily = bt.load_risk_free(prices.index)

        for short in SHORT_WINDOWS:
            for long in LONG_WINDOWS:
                if short >= long:
                    continue

                result = bt.backtest(prices, short, long, cost_bps=COST_BPS, rf_daily=rf_daily)

                strat_summary = bt.summarize(
                    result["equity_strategy"], result["strat_ret"].dropna(), rf_daily
                )
                bh_summary = bt.summarize(
                    result["equity_bh"], result["ret"].dropna(), rf_daily
                )
                trade = bt.trade_stats(result["position"])

                rows.append({
                    "ticker": ticker,
                    "short": short,
                    "long": long,
                    "strategy_cagr": strat_summary["CAGR"],
                    "bh_cagr": bh_summary["CAGR"],
                    "edge": strat_summary["CAGR"] - bh_summary["CAGR"],
                    "strategy_sharpe": strat_summary["Sharpe"],
                    "bh_sharpe": bh_summary["Sharpe"],
                    "sharpe_edge": strat_summary["Sharpe"] - bh_summary["Sharpe"],
                    "strategy_max_drawdown": strat_summary["MaxDrawdown"],
                    "bh_max_drawdown": bh_summary["MaxDrawdown"],
                    "round_trip_trades": trade["RoundTripTrades"],
                    "pct_days_in_market": trade["PctDaysInMarket"],
                })

        print(f"{ticker}: done ({len(SHORT_WINDOWS) * len(LONG_WINDOWS)} grid cells, minus short>=long skips)")

    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    df = run_sweep()
    df.to_csv("results/sweep_results.csv", index=False)
    print(f"\nWrote {len(df)} rows to results/sweep_results.csv")
