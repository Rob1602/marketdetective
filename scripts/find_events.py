"""Trova i giorni in cui un titolo si è mosso in modo anomalo: i candidati eventi da investigare.

Esempio:
    python scripts/find_events.py NVDA            # ultimi 12 mesi, 10 eventi più forti
    python scripts/find_events.py NVDA --months 6 --top 5

Le notizie gratuite di Finnhub coprono un periodo limitato (circa un anno),
quindi conviene scegliere eventi recenti.
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
    ap = argparse.ArgumentParser(description="Trova i giorni di movimento anomalo di un titolo")
    ap.add_argument("ticker")
    ap.add_argument("--months", type=int, default=12, help="quanti mesi indietro cercare (default 12)")
    ap.add_argument("--top", type=int, default=10, help="quanti eventi mostrare (default 10)")
    ap.add_argument("--quiet", action="store_true",
                    help="mostra invece i giorni PIU' tranquilli, da usare come controllo negativo")
    args = ap.parse_args()

    cfg = load_config()
    info = cfg.ticker(args.ticker)
    store = PriceStore(cfg.cache_dir)
    prices = load_event_prices(store, info.symbol, info.market, info.sector, start=cfg.data_start)
    rets = compute_returns(prices)

    start = (rets.index[-1] - pd.DateOffset(months=args.months)).strftime("%Y-%m-%d")
    found = scan_anomalies(rets, cfg.event_study, start=start, only_anomalous=not args.quiet)
    if found.empty:
        print(f"Nessun dato per {info.symbol} negli ultimi {args.months} mesi.")
        return

    order = found["t"].abs().sort_values(ascending=bool(args.quiet))
    found = found.reindex(order.index).head(args.top)
    kind = "giorni più tranquilli" if args.quiet else "movimenti anomali"
    print(f"{info.name} ({info.symbol}): {kind} negli ultimi {args.months} mesi\n")
    print(f"{'data':<12}{'rendimento':>12}{'anomalo':>10}{'t':>8}   driver principale")
    for _, r in found.iterrows():
        print(f"{r['date']:<12}{r['return']:>+12.2%}{r['abnormal']:>+10.2%}{r['t']:>8.2f}   {r['driver']}")

    if args.quiet:
        print("\nUsa uno di questi come controllo negativo: il sistema NON deve inventare una causa.")
    else:
        print("\nScegli le date, preferibilmente con driver 'specifico dell'azienda', e lancia tutto con:")
        print(f"  python scripts/run_all.py {info.symbol} DATA1 DATA2 DATA3")


if __name__ == "__main__":
    main()
