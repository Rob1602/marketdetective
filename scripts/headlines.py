"""Print the headlines used to decide the ground truth cause of an event.

Shows only news that mentions the company and that the model is allowed to see, meaning published
before the market close of the event day.

Examples:
    python scripts/headlines.py NVDA 2026-06-01
    python scripts/headlines.py NVDA 2026-06-01 2026-04-27 --top 20

The output is also saved to data/review/headlines.txt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketdetective.config import PROJECT_ROOT, load_config  # noqa: E402
from marketdetective.news import NewsArchive  # noqa: E402
from marketdetective.rag import market_close_utc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Headlines used to decide the cause of an event")
    ap.add_argument("ticker")
    ap.add_argument("dates", nargs="+")
    ap.add_argument("--top", type=int, default=25, help="how many headlines per event (default 25)")
    ap.add_argument("--hours", type=int, default=30,
                    help="how many hours before the close to cover (default 30)")
    args = ap.parse_args()

    cfg = load_config()
    info = cfg.ticker(args.ticker)
    items = NewsArchive(cfg.news_dir).load(info.symbol)
    names = {info.symbol.lower(), info.name.lower()}
    out: list[str] = []

    for date in args.dates:
        close = market_close_utc(date)
        since = close - __import__("pandas").Timedelta(hours=args.hours)
        window = [it for it in items if since <= it.published_ts <= close
                  and any(n in it.headline.lower() for n in names)]
        out.append(f"\n=== {info.symbol} {date} · {len(window)} headlines mentioning {info.name} "
                   f"in the {args.hours}h before the close")
        if not window:
            out.append("    (none: download the news first with download_news.py or run_all.py)")
            continue
        for it in window[-args.top:]:
            out.append(f"  {it.published[:16].replace('T', ' ')}  {it.source:<12.12} {it.headline}")

    text = "\n".join(out)
    print(text)
    path = PROJECT_ROOT / "data" / "review" / "headlines.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"\nAlso saved to {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
