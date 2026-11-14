"""Find the days on which a stock moved unusually: the candidate events to investigate.

Examples:
    python scripts/find_events.py NVDA            # last 12 months, 10 strongest moves
    python scripts/find_events.py NVDA --months 6 --top 5

Finnhub's free news coverage goes back about a year, so recent events work best.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketdetective.config import load_config  # noqa: E402
from marketdetective.data import PriceStore, load_event_prices  # noqa: E402
from marketdetective.stats import compute_returns, scan_anomalies  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Find the days a stock moved unusually")
    ap.add_argument("ticker")
    ap.add_argument("--months", type=int, default=12, help="how many months back to search (default 12)")
    ap.add_argument("--top", type=int, default=10, help="how many events to show (default 10)")
    ap.add_argument("--quiet", action="store_true",
                    help="show the QUIETEST days instead, to use as a negative control")
    args = ap.parse_args()

    cfg = load_config()
    info = cfg.ticker(args.ticker)
    store = PriceStore(cfg.cache_dir)
    prices = load_event_prices(store, info.symbol, info.market, info.sector, start=cfg.data_start)
    rets = compute_returns(prices)

    start = (rets.index[-1] - pd.DateOffset(months=args.months)).strftime("%Y-%m-%d")
    found = scan_anomalies(rets, cfg.event_study, start=start, only_anomalous=not args.quiet)
    if found.empty:
        print(f"No data for {info.symbol} over the last {args.months} months.")
        return

    order = found["t"].abs().sort_values(ascending=bool(args.quiet))
    found = found.reindex(order.index).head(args.top)
    kind = "quietest days" if args.quiet else "anomalous moves"
    print(f"{info.name} ({info.symbol}): {kind} over the last {args.months} months\n")
    print(f"{'date':<12}{'return':>12}{'abnormal':>10}{'t':>8}   main driver")
    for _, r in found.iterrows():
        print(f"{r['date']:<12}{r['return']:>+12.2%}{r['abnormal']:>+10.2%}{r['t']:>8.2f}   {r['driver']}")

    if args.quiet:
        print("\nUse one of these as a negative control: the system must NOT invent a cause.")
    else:
        print("\nPick the dates, preferably with a company-specific driver, then run everything with:")
        print(f"  python scripts/run_all.py {info.symbol} DATE1 DATE2 DATE3")


if __name__ == "__main__":
    main()
