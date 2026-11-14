"""Scarica da Finnhub le notizie intorno a uno o più eventi e le salva nell'archivio locale.

Esempi:
    python scripts/download_news.py NVDA 2026-11-09
    python scripts/download_news.py NVDA 2026-11-09 2026-04-30 2026-08-27

Per ogni evento crea anche un file da aprire con Excel o Numbers:
    data/review/NVDA_2026-11-09.csv
con tutte le notizie e una colonna che dice se il modello potrà vederle.
Leggendolo decidi la causa vera dell'evento e la scrivi in data/ground_truth.yaml.
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
    ap = argparse.ArgumentParser(description="Scarica notizie Finnhub per uno o più eventi")
    ap.add_argument("ticker")
    ap.add_argument("dates", nargs="+", help="date degli eventi, formato AAAA-MM-GG")
    args = ap.parse_args()

    cfg = load_config()
    info = cfg.ticker(args.ticker)
    ticker = info.symbol
    client = FinnhubClient()
    archive = NewsArchive(cfg.news_dir)
    review_dir = PROJECT_ROOT / "data" / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    for date in args.dates:
        print(f"\n=== {ticker} {date}: scarico un giorno alla volta...")
        rep = download_event_news(client, archive, ticker, date, cfg.news_days_before, cfg.news_days_after)
        for day, n in rep.per_day.items():
            print(f"    {day}: {n} notizie")
        print(f"    totale {rep.downloaded}, nuove nell'archivio {rep.added}")
        if rep.saturated_days:
            print(f"    Attenzione: {', '.join(rep.saturated_days)} hanno raggiunto il limite di Finnhub, "
                  "alcune notizie di quei giorni potrebbero mancare.")

        start, end = event_news_range(date, cfg.news_days_before, cfg.news_days_after)
        close = market_close_utc(date)
        items = [it for it in archive.load(ticker) if start <= it.published[:10] <= end]
        visible = [it for it in items if it.published_ts <= close]

        out = review_dir / f"{ticker}_{date}.csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            w.writerow(["id", "pubblicata_utc", "visibile_al_modello", "fonte", "titolo", "riassunto", "url"])
            for it in items:
                w.writerow([it.id, it.published[:16].replace("T", " "),
                            "si" if it.published_ts <= close else "no",
                            it.source, it.headline, it.summary, it.url])

        name_words = {ticker.lower(), info.name.lower()}
        about = [it for it in visible if any(w in it.headline.lower() for w in name_words)]
        print(f"\n    Il modello potrà vedere {len(visible)} notizie (uscite prima della chiusura del {date}).")
        print(f"    Ultime 15 con '{info.name}' nel titolo:")
        for it in about[-15:]:
            print(f"      {it.published[:16].replace('T', ' ')}  {it.source:<12.12} {it.headline[:95]}")
        print(f"    Tutte le notizie, anche quelle successive: {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
