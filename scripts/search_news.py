"""Prova il RAG: cerca nell'archivio le notizie più pertinenti per un evento.

Esempio:
    python scripts/search_news.py NVDA 2025-04-03 "perché il titolo è sceso?"

Mostra solo notizie pubblicate prima della chiusura di borsa del giorno dell'evento.
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
    ap = argparse.ArgumentParser(description="Ricerca semantica nell'archivio notizie")
    ap.add_argument("ticker")
    ap.add_argument("date")
    ap.add_argument("query")
    args = ap.parse_args()

    cfg = load_config()
    index = NewsIndex(NewsArchive(cfg.news_dir), OllamaEmbedder(cfg.embedding_model))
    n = index.build()
    print(f"Archivio: {len(index.items)} notizie ({n} nuovi embedding calcolati)")
    print(f"Solo notizie fino a {market_close_utc(args.date)} (chiusura di borsa)\n")

    hits = index.search_event(args.query, cfg.ticker(args.ticker).symbol, args.date, k=cfg.top_k)
    if not hits:
        print("Nessuna notizia nell'archivio per questo titolo e periodo.")
    for h in hits:
        print(f"{h.score:.3f}  {h.item.citation()}")
        if h.item.summary:
            print(f"       {h.item.summary[:160]}")


if __name__ == "__main__":
    main()
