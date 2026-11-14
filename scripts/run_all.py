"""Esegue l'intero esperimento su più eventi, senza bisogno di lanciare un comando per volta.

Per ogni data: scarica le notizie (se mancano), poi esegue l'agente e la baseline RAG.
Si può lasciar girare e tornare dopo.

Esempi:
    python scripts/run_all.py NVDA 2026-11-09 2026-04-30 2026-08-27
    python scripts/run_all.py NVDA 2026-02-02 2026-06-01 --no-download
    python scripts/run_all.py AAPL 2026-05-15 --modes agent

Le esecuzioni già presenti in runs/ vengono saltate (usa --force per rifarle).
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ask import build_agent, save  # noqa: E402
from marketdetective.config import PROJECT_ROOT, load_config  # noqa: E402
from marketdetective.news import FinnhubClient, NewsArchive, download_event_news  # noqa: E402

RUNS = PROJECT_ROOT / "runs"


def already_done(ticker: str, date: str, mode: str) -> bool:
    return any(RUNS.glob(f"{ticker}_{date}_{mode}_*.json"))


def hms(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s:02d}s"


def main() -> None:
    ap = argparse.ArgumentParser(description="Esegue agente e baseline su più eventi")
    ap.add_argument("ticker")
    ap.add_argument("dates", nargs="+", help="date degli eventi, formato AAAA-MM-GG")
    ap.add_argument("--modes", default="agent,rag", help="quali modalità eseguire (default: agent,rag)")
    ap.add_argument("--no-download", action="store_true", help="non scaricare notizie, usa l'archivio esistente")
    ap.add_argument("--force", action="store_true", help="riesegui anche i casi già presenti in runs/")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    cfg = load_config()
    ticker = cfg.ticker(args.ticker).symbol
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    # 1. Notizie
    if not args.no_download:
        print("=== Notizie ===")
        client = FinnhubClient()
        archive = NewsArchive(cfg.news_dir)
        for date in args.dates:
            try:
                rep = download_event_news(client, archive, ticker, date,
                                          cfg.news_days_before, cfg.news_days_after)
                warn = "  ATTENZIONE: limite Finnhub raggiunto" if rep.saturated_days else ""
                print(f"  {date}: {rep.downloaded} scaricate, {rep.added} nuove{warn}")
            except Exception as e:  # noqa: BLE001
                print(f"  {date}: errore nel download ({e}). Proseguo con l'archivio esistente.")

    # 2. Indice e agente (gli embedding mancanti si calcolano una volta sola)
    print("\n=== Indice RAG ===")
    t0 = time.perf_counter()
    agent = build_agent(cfg, args.model)
    print(f"  {len(agent.tb.index.items)} notizie indicizzate in {hms(time.perf_counter() - t0)}")

    # 3. Esecuzioni
    jobs = [(d, m) for d in args.dates for m in modes]
    print(f"\n=== {len(jobs)} esecuzioni · modello {agent.model} ===")
    done, skipped, failed = 0, 0, []
    t_start = time.perf_counter()

    for i, (date, mode) in enumerate(jobs, 1):
        if not args.force and already_done(ticker, date, mode):
            print(f"\n[{i}/{len(jobs)}] {date} · {mode}: già fatto, salto (--force per rifarlo)")
            skipped += 1
            continue
        print(f"\n[{i}/{len(jobs)}] {date} · {mode}")
        t_run = time.perf_counter()
        try:
            run = agent.investigate if mode == "agent" else agent.rag_baseline
            result = run(ticker, date)
            base = save(result, RUNS)
            flag = "  CITAZIONI NON VALIDE" if result.citations_invalid else ""
            err = f"  ERRORE: {result.error}" if result.error else ""
            causes = len(result.report.get("supported_causes", []))
            print(f"  {hms(time.perf_counter() - t_run)} · {causes} cause · "
                  f"{len(result.citations_valid)} citazioni valide{flag}{err}")
            print(f"  → {base.name}.md")
            done += 1
        except Exception as e:  # noqa: BLE001
            failed.append((date, mode, str(e)))
            print(f"  FALLITA: {type(e).__name__}: {e}")
            traceback.print_exc(limit=1)

    print(f"\n=== Fatto in {hms(time.perf_counter() - t_start)}: "
          f"{done} eseguite, {skipped} saltate, {len(failed)} fallite ===")
    for date, mode, err in failed:
        print(f"  {date} · {mode}: {err}")
    print("\nAggiungi le cause vere dei nuovi eventi in data/ground_truth.yaml,")
    print("poi lancia: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
