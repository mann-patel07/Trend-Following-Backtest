"""Check that every number in README.md comes from the data or the results files.

Numbers checked (each is formatted exactly as the README writes it, and the string must appear in the README):
  AAPL table (recomputed with backtest.bt, 20/50, 5 bps, lag 1): buy-and-hold and strategy CAGR, max drawdown, Sharpe.
  AAPL trade stats: 65% of time in the market, 18 round trips.
  Sweep (results/sweep_results.csv): share of cells beating buy-and-hold on CAGR and Sharpe, AAPL percentile and
    the share of cells it is worse than, ticker bootstrap interval, Boeing, XOM, WFC win rates, the 16% bound,
    the 20 tickers that never win, drawdown share, share with Sharpe above zero.
  Next-day fill (results/sweep_results_next_close.csv): AAPL CAGR, Sharpe, drawdown, the two shares, tickers that never win.
  Walk-forward (results/walkforward_summary.json, results/walkforward_by_ticker.csv): table medians, counts, sign
    test p-value, median differences, AAPL line, pair changes.
  Random timing (results/random_benchmark_summary.json): cells used and skipped, median percentiles, shares, AAPL.
  Test counts (pytest collection) and the files, figures and links named in the README.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt
import sweep

ROOT = Path(__file__).resolve().parent
README = (ROOT / "README.md").read_text()
failures = []
checked = 0


def need(text, label):
    global checked
    checked += 1
    if text not in README:
        failures.append(f"README is missing {label}: {text!r}")


def pct(x, d=2):
    return f"{x * 100:.{d}f}%"


# ---- AAPL
prices = bt.load_prices("AAPL", sweep.START, sweep.END)
rf = bt.load_risk_free(prices.index)
res = bt.backtest(prices, 20, 50, cost_bps=5.0, rf_daily=rf, lag=1)
bh = bt.summarize(res["equity_bh"], res["ret"].dropna(), rf)
st = bt.summarize(res["equity_strategy"], res["strat_ret"].dropna(), rf)
need(f"| Buy-and-hold | {pct(bh['CAGR'])} | {pct(bh['MaxDrawdown'])} | {bh['Sharpe']:.2f} |", "AAPL buy-and-hold row")
need(f"| Strategy | {pct(st['CAGR'])} | {pct(st['MaxDrawdown'])} | {st['Sharpe']:.2f} |", "AAPL strategy row")
ts = bt.trade_stats(res["position"])
need(f"It spent about {ts['PctDaysInMarket']:.0f}% of the time in the market", "time in market")
need(f"The strategy made {int(ts['RoundTripTrades'])} round-trip trades", "round trips")

# ---- sweep
sw = pd.read_csv(ROOT / "results/sweep_results.csv")
nx = pd.read_csv(ROOT / "results/sweep_results_next_close.csv")
need(f"{len(sw)} cells", "cell count")
need(f"The strategy beat buy-and-hold on CAGR in {pct((sw.edge > 0).mean(), 1)} of cells, and on Sharpe in {pct((sw.sharpe_edge > 0).mean(), 1)}", "sweep shares")
a = sw[(sw.ticker == "AAPL") & (sw.short == 20) & (sw.long == 50)].iloc[0]
need(f"about the {round(100 * (sw.edge < a.edge).mean())}th percentile", "AAPL percentile")
need(f"worse than about {round(100 * (sw.edge > a.edge).mean())}% of the cells", "AAPL worse-than share")
pt = sw.groupby("ticker").edge.mean()
rng = np.random.default_rng(42)
boots = [rng.choice(pt.to_numpy(), size=len(pt), replace=True).mean() for _ in range(10000)]
lo, hi = np.percentile(boots, [2.5, 97.5])
need(f"is {lo * 100:.1f}% to {hi * 100:.1f}%", "bootstrap interval")
win = sw.groupby("ticker").edge.apply(lambda s: (s > 0).mean())
for t, label in (("BA", "Boeing (BA)"), ("XOM", "Exxon (XOM)"), ("WFC", "Wells Fargo (WFC)")):
    need(f"{label}", "ticker name")
need(f"beating buy-and-hold in {round(100 * win['BA'])}% of its parameter combinations", "Boeing win rate")
need(f"Exxon (XOM) wins in {round(100 * win['XOM'])}% and Wells Fargo (WFC) in {round(100 * win['WFC'])}%", "XOM and WFC")
rest = win.drop(["BA", "XOM", "WFC"])
need(f"wins in {int(np.ceil(rest.max() * 100))}% of its combinations or fewer", "other tickers bound")
need(f"{int((win == 0).sum())} of the 29 tickers never beat buy-and-hold in a single one", "tickers that never win")
need(f"smaller than buy-and-hold's in {pct((sw.strategy_max_drawdown > sw.bh_max_drawdown).mean(), 1)} of the 551 cells", "drawdown share")
need(f"its Sharpe ratio is above zero in {pct((sw.strategy_sharpe > 0).mean(), 1)}", "Sharpe above zero")
# next-day fill
n = nx[(nx.ticker == "AAPL") & (nx.short == 20) & (nx.long == 50)].iloc[0]
need(f"the strategy CAGR goes from {pct(a.strategy_cagr)} to {pct(n.strategy_cagr)}, the Sharpe ratio from {a.strategy_sharpe:.2f} to {n.strategy_sharpe:.2f} and the max drawdown from {pct(a.strategy_max_drawdown)} to {pct(n.strategy_max_drawdown)}", "next-day AAPL")
need(f"at {pct(a.bh_cagr)} CAGR", "buy-and-hold CAGR in the next-day paragraph")
nwin = nx.groupby("ticker").edge.apply(lambda s: (s > 0).mean())
need(f"on CAGR in {pct((nx.edge > 0).mean(), 1)} of cells ({pct((sw.edge > 0).mean(), 1)} in the main run) and on Sharpe in {pct((nx.sharpe_edge > 0).mean(), 1)} ({pct((sw.sharpe_edge > 0).mean(), 1)})", "next-day shares")
need(f"{int((nwin == 0).sum())} of the 29 tickers never beat buy-and-hold on CAGR ({int((win == 0).sum())} in the main run)", "next-day never-win count")

# ---- walk-forward
wf = pd.read_csv(ROOT / "results/walkforward_by_ticker.csv")
ws = json.load(open(ROOT / "results/walkforward_summary.json"))
need(f"| Walk-forward | {pct(wf.wf_cagr.median())} | {wf.wf_sharpe.median():.2f} | {pct(wf.wf_max_drawdown.median())} |", "walk-forward row")
need(f"| Buy-and-hold | {pct(wf.bh_cagr.median())} | {wf.bh_sharpe.median():.2f} | {pct(wf.bh_max_drawdown.median())} |", "buy-and-hold row")
need(f"| Fixed 20/50 | {pct(wf.fixed_20_50_cagr.median())} | {wf.fixed_20_50_sharpe.median():.2f} | {pct(wf.fixed_20_50_max_drawdown.median())} |", "fixed row")
need(f"| Hindsight best pair | {pct(wf.hindsight_best_cagr.median())} | not recorded | not recorded |", "hindsight row")
need(f"beats buy-and-hold on Sharpe ratio for {ws['wf_beats_bh_sharpe']} of {ws['n_tickers']} tickers (two-sided sign test, p = {ws['sign_test_p_sharpe']:.1e}), on CAGR for {ws['wf_beats_bh_cagr']} of {ws['n_tickers']}, and on maximum drawdown for {ws['wf_beats_bh_drawdown']} of {ws['n_tickers']}", "walk-forward counts")
need(f"on CAGR for {ws['wf_beats_fixed_cagr']} tickers and on Sharpe for {ws['wf_beats_fixed_sharpe']}", "walk-forward against fixed")
need(f"in CAGR against buy-and-hold is {ws['median_wf_minus_bh_cagr'] * 100:.2f} percentage points, and in Sharpe ratio {ws['median_wf_minus_bh_sharpe']:.2f}", "median differences")
need(f"hindsight best pair is {ws['median_wf_minus_hindsight_cagr'] * 100:.2f} percentage points", "median difference to hindsight")
need(f"median CAGR ({pct(wf.hindsight_best_cagr.median())}) below buy-and-hold ({pct(wf.bh_cagr.median())})", "hindsight against buy-and-hold")
need(f"{ws['n_tickers'] - ws['wf_beats_bh_cagr']} of the {ws['n_tickers']} bars are below zero", "bars below zero")
k = ws["aapl"]
need(f"has a CAGR of {pct(k['wf_cagr'])}, a Sharpe ratio of {k['wf_sharpe']:.2f} and a maximum drawdown of {pct(k['wf_max_drawdown'])}", "AAPL walk-forward")
need(f"Buy-and-hold over the same days has {pct(k['bh_cagr'])}, {k['bh_sharpe']:.2f} and {pct(k['bh_max_drawdown'])}", "AAPL buy-and-hold 2021 to 2025")
need(f"The fixed 20/50 pair has {pct(k['fixed_20_50_cagr'])}, {k['fixed_20_50_sharpe']:.2f} and {pct(k['fixed_20_50_max_drawdown'])}", "AAPL fixed")
need(f"The chosen pair changed {k['wf_pair_changes']} times", "AAPL pair changes")
need(f"sign test p-value of {ws['sign_test_p_sharpe']:.1e}", "p-value in the first paragraph")
need(f"on Sharpe ratio for {ws['wf_beats_bh_sharpe']} of {ws['n_tickers']} tickers, with", "first paragraph count")

# ---- random timing
rs = json.load(open(ROOT / "results/random_benchmark_summary.json"))
need(f"All {rs['cells_used']} cells were used and none was skipped", "cells used")
assert rs["cells_skipped"] == 0
need(f"The median CAGR percentile is {rs['median_cagr_percentile']:.1f}", "median CAGR percentile")
need(f"In {rs['share_cagr_percentile_above_50'] * 100:.1f}% of cells the percentile is above 50, in {rs['share_cagr_percentile_above_95'] * 100:.1f}% above 95 and in {rs['share_cagr_percentile_below_5'] * 100:.1f}% below 5", "percentile shares")
need(f"The median Sharpe percentile is {rs['median_sharpe_percentile']:.1f}", "median Sharpe percentile")
need(f"the CAGR percentile is {rs['aapl_20_50']['cagr_percentile']:.1f}", "AAPL percentile in random timing")
need(f"is {rs['median_cagr_percentile']:.1f}, where 50 would mean no difference", "first paragraph median percentile")

# ---- tests, files, figures
out = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"], capture_output=True, text=True, cwd=ROOT).stdout
counts = {}
for line in out.splitlines():
    m = re.match(r"tests/(\w+)\.py::", line)
    if m:
        counts[m.group(1)] = counts.get(m.group(1), 0) + 1
need(f"runs the {sum(counts.values())} tests", "total test count")
for name in ("test_backtest", "test_walkforward", "test_random_benchmark"):
    need(f"`tests/{name}.py`: {counts[name]} tests", f"count for {name}")

for p in set(re.findall(r"\]\(([^)]+)\)", README)) | set(re.findall(r"`((?:[\w./-]+/)?[\w.-]+\.(?:py|md|csv|json|png|ipynb|txt))`", README)):
    checked += 1
    if p.startswith("http"):
        continue
    if not (ROOT / p).exists():
        failures.append(f"README names a path that does not exist: {p}")

# every table row has the same number of pipes as its table header
block = []
for line in README.splitlines() + [""]:
    if line.startswith("|"):
        block.append(line)
    else:
        if block:
            for row in block:
                checked += 1
                if row.count("|") != block[0].count("|"):
                    failures.append(f"table row has a different number of cells: {row}")
        block = []

if failures:
    print("CONSISTENCY CHECK FAILED")
    for f in failures:
        print(" -", f)
    sys.exit(1)
print(f"Consistency check passed: {checked} checks.")
