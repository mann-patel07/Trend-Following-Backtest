"""Moving-average crossover backtest, extracted from archive/archive_original_v1.ipynb.

Same logic as v1's notebook cells, moved into reusable functions so v2's
parameter sweep and analysis notebooks don't have to copy-paste it.
"""

import os
import warnings

import pandas as pd
import yfinance as yf


def load_prices(ticker, start, end, cache_dir="data", refresh=False):
    """Download adjusted daily closes for `ticker` between `start` and `end`.

    Uses yfinance with auto_adjust=True (split/dividend adjusted), flattens
    the MultiIndex columns yfinance returns, and caches the result to
    `{cache_dir}/{ticker}_{start}_{end}.csv`. Later calls read from that
    cache instead of re-downloading, unless `refresh=True`. Returns a Series
    of closing prices indexed by date.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{ticker}_{start}_{end}.csv")

    if os.path.exists(cache_path) and not refresh:
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return cached["Close"]

    raw = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.droplevel(1)

    prices = raw["Close"].rename("Close")
    prices.to_csv(cache_path)
    return prices


def load_risk_free(index, cache_dir="data", refresh=False):
    """Load the ^IRX (13-week T-bill) daily rate aligned to `index`.

    Downloads once for the date span covered by `index` and caches to CSV
    the same way `load_prices` does. If the download fails for any reason,
    or returns an empty frame or no usable `Close` values, or the cached file
    is empty or has no usable values, falls back to a flat 4% annual rate,
    matching v1's fallback behavior. An empty download is not cached.
    Returns a daily rate series (annual rate / 252) reindexed and
    forward-filled onto `index`. If the first dates of `index` come before the
    first available rate, they get that first rate (back-filled) and a warning
    gives the number of days filled.
    """
    os.makedirs(cache_dir, exist_ok=True)
    start = index.min().strftime("%Y-%m-%d")
    end = index.max().strftime("%Y-%m-%d")
    cache_path = os.path.join(cache_dir, f"IRX_{start}_{end}.csv")
    flat = pd.Series(0.04 / 252, index=index)

    if os.path.exists(cache_path) and not refresh:
        try:
            irx_close = pd.read_csv(cache_path, index_col=0, parse_dates=True)["Close"]
        except Exception as e:
            warnings.warn(f"cached T-bill file could not be read ({e}), falling back to flat 4% annual rate")
            return flat
        if irx_close.dropna().empty:
            warnings.warn("cached T-bill file has no usable values, falling back to flat 4% annual rate")
            return flat
    else:
        try:
            irx = yf.download("^IRX", start=start, end=end, auto_adjust=True, progress=False)
            if isinstance(irx.columns, pd.MultiIndex):
                irx.columns = irx.columns.droplevel(1)
            if irx.empty or "Close" not in irx.columns or irx["Close"].dropna().empty:
                warnings.warn("T-bill download returned no usable values, falling back to flat 4% annual rate")
                return flat
            irx_close = irx["Close"].rename("Close")
            irx_close.to_csv(cache_path)
        except Exception as e:
            warnings.warn(f"T-bill download failed ({e}), falling back to flat 4% annual rate")
            return flat

    rf_annual = irx_close.reindex(index).ffill() / 100
    n_missing = int(rf_annual.isna().sum())
    if n_missing:
        rf_annual = rf_annual.bfill()
        if rf_annual.isna().any():
            warnings.warn("T-bill rates do not overlap the dates, falling back to flat 4% annual rate")
            return flat
        warnings.warn(f"{n_missing} days at the start had no T-bill rate and were filled with the first available rate")
    return rf_annual / 252


def backtest(prices, short, long, cost_bps=5.0, rf_daily=None, lag=1):
    """Run a moving-average crossover backtest on a price series.

    Signal is 1 when the `short`-day SMA is above the `long`-day SMA, 0
    otherwise. Position is the signal lagged by one day, so a day's return
    is only earned using information known at the start of that day (no
    look-ahead). Cash earns `rf_daily` on days the position is flat. A
    transaction cost of `cost_bps` basis points (one-way) is charged on
    `position.diff().abs()` every time the position changes.

    Warm-up rows (before the `long`-day SMA is available) are dropped
    before the position is computed. The first remaining row has no prior-day
    signal to shift from, so the position starts flat (0) on that row. That
    day earns the cash rate, and if the first signal is long, the entry cost
    is charged on the next row, when the position becomes 1. Buy-and-hold and
    the strategy therefore cover the same dates.

    `lag` is the number of days between the signal and the position.
    `lag=1` fills at the close of the signal day, which is the same close the
    signal is computed from. `lag=2` fills at the next day's close. `lag`
    must be an integer of 1 or more.

    Returns a DataFrame indexed like `prices` with columns: close,
    sma_short, sma_long, signal, position, ret, strat_ret, equity_bh
    (buy-and-hold cumulative return) and equity_strategy (strategy
    cumulative return).
    """
    if short >= long:
        raise ValueError(f"short window ({short}) must be < long window ({long})")
    if not isinstance(lag, int) or isinstance(lag, bool) or lag < 1:
        raise ValueError(f"lag ({lag!r}) must be an integer of 1 or more")

    df = pd.DataFrame(index=prices.index)
    df["close"] = prices
    df["sma_short"] = prices.rolling(window=short).mean()
    df["sma_long"] = prices.rolling(window=long).mean()
    df["signal"] = (df["sma_short"] > df["sma_long"]).astype(int)
    df["ret"] = prices.pct_change()

    df = df.dropna(subset=["sma_long"]).copy()

    if rf_daily is None:
        rf_daily = pd.Series(0.04 / 252, index=df.index)
    rf_daily = rf_daily.reindex(df.index).ffill()

    df["position"] = df["signal"].shift(lag).fillna(0)
    turnover = df["position"].diff().abs().fillna(0)

    gross_strat_ret = df["ret"] * df["position"] + rf_daily * (1 - df["position"])
    cost = turnover * (cost_bps / 1e4)
    df["strat_ret"] = gross_strat_ret - cost

    df["equity_bh"] = (1 + df["ret"]).cumprod()
    df["equity_strategy"] = (1 + df["strat_ret"]).cumprod()

    return df


def max_drawdown(cumulative_returns):
    """Drawdown series and its minimum (most negative) value for an equity curve.

    The running peak starts at the starting capital of 1.0, so a loss in the
    first days counts as a drawdown. The returned series keeps the index of
    `cumulative_returns`.
    """
    peak = cumulative_returns.cummax().clip(lower=1.0)
    drawdown = (cumulative_returns - peak) / peak
    return drawdown, drawdown.min()


def cagr(cumulative_returns, periods_per_year=252):
    """Compound annual growth rate from a cumulative-return equity series.

    Uses the count of actual (non-NaN) return periods, not raw row count.
    Both curves from `backtest` now have a value on every row, so this equals
    the number of rows.
    """
    n_returns = cumulative_returns.count()
    n_years = n_returns / periods_per_year
    return cumulative_returns.iloc[-1] ** (1 / n_years) - 1


def sharpe(daily_returns, rf_daily, periods_per_year=252):
    """Annualized Sharpe ratio of `daily_returns` in excess of `rf_daily`."""
    excess = daily_returns - rf_daily
    return (excess.mean() / excess.std()) * (periods_per_year ** 0.5)


def summarize(equity, returns, rf_daily_series):
    """CAGR, Sharpe, max drawdown, final multiple, and drawdown series for one equity curve."""
    dd_series, dd = max_drawdown(equity)
    return {
        "CAGR": cagr(equity),
        "Sharpe": sharpe(returns, rf_daily_series),
        "MaxDrawdown": dd,
        "FinalMultiple": equity.iloc[-1],
        "DrawdownSeries": dd_series,
    }


def trade_stats(position):
    """Round-trip trade count, average holding period, and % of days in market for a position series."""
    turnover = position.diff().abs().fillna(0)
    round_trip_trades = turnover.sum() / 2
    pct_days_in_market = position.mean() * 100
    entries = ((position == 1) & (position.shift(1) == 0)).sum()
    avg_holding_days = position.sum() / entries if entries > 0 else float("nan")
    return {
        "RoundTripTrades": round_trip_trades,
        "AvgHoldingDays": avg_holding_days,
        "PctDaysInMarket": pct_days_in_market,
    }
