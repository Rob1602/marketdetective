"""Download Finnhub news around one or more events and store it in the local archive.

Examples:
    python scripts/download_news.py NVDA 2026-11-09
    python scripts/download_news.py NVDA 2026-11-09 2026-04-30 2026-08-27

For each event it also writes a spreadsheet file:
    data/review/NVDA_2026-11-09.csv
listing every news item and whether the model is allowed to see it. Reading it is how the ground
truth in data/ground_truth.yaml is decided.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketdetective.config import PROJECT_ROOT, load_config  # noqa: E402
from marketdetective.news import FinnhubClient, NewsArchive, download_event_news, event_news_range  # noqa: E402
from marketdetective.rag import market_close_utc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Download Finnhub news for one or more events")
    ap.add_argument("ticker")
    ap.add_argument("dates", nargs="+", help="event dates, format YYYY-MM-DD")
    args = ap.parse_args()

    cfg = load_config()
    info = cfg.ticker(args.ticker)
    ticker = info.symbol
    client = FinnhubClient()
    archive = NewsArchive(cfg.news_dir)
    review_dir = PROJECT_ROOT / "data" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    for date in args.dates:
        print(f"\n=== {ticker} {date}: downloading one day at a time...")
        rep = download_event_news(client, archive, ticker, date, cfg.news_days_before, cfg.news_days_after)
        for day, n in rep.per_day.items():
            print(f"    {day}: {n} news items")
        print(f"    total {rep.downloaded}, new in the archive {rep.added}")
        if rep.saturated_days:
            print(f"    Warning: {', '.join(rep.saturated_days)} hit the Finnhub per-request cap, "
                  "so some items from those days may be missing.")

        start, end = event_news_range(date, cfg.news_days_before, cfg.news_days_after)
        close = market_close_utc(date)
        items = [it for it in archive.load(ticker) if start <= it.published[:10] <= end]
        visible = [it for it in items if it.published_ts <= close]

        out = review_dir / f"{ticker}_{date}.csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["id", "published_utc", "visible_to_model", "source", "headline", "summary", "url"])
            for it in items:
                w.writerow([it.id, it.published[:16].replace("T", " "),
                            "yes" if it.published_ts <= close else "no",
                            it.source, it.headline, it.summary, it.url])

        name_words = {ticker.lower(), info.name.lower()}
        about = [it for it in visible if any(w in it.headline.lower() for w in name_words)]
        print(f"\n    The model will see {len(visible)} items, published before the close on {date}.")
        print(f"    Last 15 with '{info.name}' in the headline:")
        for it in about[-15:]:
            print(f"      {it.published[:16].replace('T', ' ')}  {it.source:<12.12} {it.headline[:95]}")
        print(f"    Every item, including later ones: {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
