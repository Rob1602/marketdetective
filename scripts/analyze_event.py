"""Run an event study on one stock and one date, from the command line.

Examples:
    python scripts/analyze_event.py NVDA 2026-11-09
    python scripts/analyze_event.py NVDA 2026-11-09 --model market
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketdetective.config import load_config  # noqa: E402
from marketdetective.data import PriceStore, load_event_prices  # noqa: E402
from marketdetective.stats import compute_returns, event_study  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Event study on one stock and one date")
    ap.add_argument("ticker")
    ap.add_argument("date", help="event date, format YYYY-MM-DD")
    ap.add_argument("--model", default="market_sector", choices=["market", "market_sector"])
    args = ap.parse_args()

    cfg = load_config()
    info = cfg.ticker(args.ticker)
    store = PriceStore(cfg.cache_dir)
    prices = load_event_prices(store, info.symbol, info.market, info.sector, start=cfg.data_start)
    rets = compute_returns(prices)

    result = event_study(rets, args.date, cfg.event_study, model=args.model)
    print(f"{info.name} ({info.symbol}) vs {info.market} / {info.sector}")
    print(result.summary())


if __name__ == "__main__":
    main()
