"""Try the RAG: search the archive for the news most relevant to an event.

Example:
    python scripts/search_news.py NVDA 2026-11-09 "why did the stock fall?"

Only news published before the market close of the event day is returned.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketdetective.config import load_config  # noqa: E402
from marketdetective.news import NewsArchive  # noqa: E402
from marketdetective.rag import NewsIndex, OllamaEmbedder, market_close_utc  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Semantic search over the news archive")
    ap.add_argument("ticker")
    ap.add_argument("date")
    ap.add_argument("query")
    args = ap.parse_args()

    cfg = load_config()
    index = NewsIndex(NewsArchive(cfg.news_dir), OllamaEmbedder(cfg.embedding_model))
    n = index.build()
    print(f"Archive: {len(index.items)} news items ({n} new embeddings computed)")
    print(f"Only news published before {market_close_utc(args.date)} (market close)\n")

    hits = index.search_event(args.query, cfg.ticker(args.ticker).symbol, args.date, k=cfg.top_k)
    if not hits:
        print("No news in the archive for this ticker and period.")
    for h in hits:
        print(f"{h.score:.3f}  {h.item.citation()}")
        if h.item.summary:
            print(f"       {h.item.summary[:160]}")


if __name__ == "__main__":
    main()
