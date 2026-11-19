"""Sweep with each trade filled one day later.

Runs the same grid as sweep.py with lag=2, so a signal from the close of day
t is filled at the close of day t+1. Run as `python sweep_next_close.py`;
writes results/sweep_results_next_close.csv.
"""

import os

from sweep import run_sweep

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    df = run_sweep(lag=2)
    df.to_csv("results/sweep_results_next_close.csv", index=False)
    print(f"\nWrote {len(df)} rows to results/sweep_results_next_close.csv")
