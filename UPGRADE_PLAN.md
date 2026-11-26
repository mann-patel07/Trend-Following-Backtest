# Plan for the walk-forward and random-timing tests

Date: 2026-11-26

Written before any of these results existed. The first version of this project (the AAPL backtest and the 551-cell sweep) is unchanged.

## Walk-forward test
- Tickers: the 29 in sweep.py. Window pairs: the 19 in sweep.py. Costs: 5 bps one-way. Fills: lag 1, as in the sweep.
- Each year Y from 2021 to 2025, I choose the window pair with the highest Sharpe ratio (excess over T-bills) over all trading days from 2020-01-01 to the end of year Y-1. Ties go to the smaller long window, then the smaller short window. I trade that pair during year Y.
- I compare the walk-forward strategy with buy-and-hold and with the fixed 20/50 pair over the same days (2021 to 2025), on CAGR, Sharpe ratio and maximum drawdown.
- Main number: on how many of the 29 tickers the walk-forward Sharpe ratio beats buy-and-hold, with a two-sided sign test.

## Random-timing benchmark
- For each of the 551 sweep cells, I simulate 500 random long-or-flat paths with the same share of days in the market and the same average holding period as the strategy, using a two-state Markov chain. Each path pays the same costs and earns the same cash rate.
- For each cell I record the percentile of the strategy's CAGR among the 500 random CAGRs.
- If the crossover signal adds nothing beyond its time in the market, these percentiles should be spread evenly from 0 to 100, with a median near 50.
- Main numbers: the median percentile over the 551 cells, and the share of cells above the 95th and below the 5th percentile.

## Rules
- Seeds are fixed. Nothing here changes the first version's numbers.
- I report the results whatever they are.
