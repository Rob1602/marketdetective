"""Run the whole experiment over several events, without launching one command at a time.

For each date it downloads the news if missing, then runs the agent and the RAG baseline. It can
be left running unattended.

Examples:
    python scripts/run_all.py NVDA 2026-11-09 2026-04-30 2026-08-27
    python scripts/run_all.py NVDA 2026-02-02 2026-06-01 --no-download
    python scripts/run_all.py AAPL 2026-05-15 --modes agent

Runs already present in runs/ are skipped; use --force to repeat them.
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
    ap = argparse.ArgumentParser(description="Run the agent and the baseline over several events")
    ap.add_argument("ticker")
    ap.add_argument("dates", nargs="+", help="event dates, format YYYY-MM-DD")
    ap.add_argument("--modes", default="agent,rag", help="which modes to run (default: agent,rag)")
    ap.add_argument("--no-download", action="store_true", help="skip the download and use the existing archive")
    ap.add_argument("--force", action="store_true", help="re-run cases already present in runs/")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    cfg = load_config()
    ticker = cfg.ticker(args.ticker).symbol
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    # 1. News
    if not args.no_download:
        print("=== News ===")
        client = FinnhubClient()
        archive = NewsArchive(cfg.news_dir)
        for date in args.dates:
            try:
                rep = download_event_news(client, archive, ticker, date,
                                          cfg.news_days_before, cfg.news_days_after)
                warn = "  WARNING: Finnhub per-request cap reached" if rep.saturated_days else ""
                print(f"  {date}: {rep.downloaded} downloaded, {rep.added} new{warn}")
            except Exception as e:  # noqa: BLE001
                print(f"  {date}: download failed ({e}). Carrying on with the existing archive.")

    # 2. Index and agent (missing embeddings are computed once)
    print("\n=== RAG index ===")
    t0 = time.perf_counter()
    agent = build_agent(cfg, args.model)
    print(f"  {len(agent.tb.index.items)} news items indexed in {hms(time.perf_counter() - t0)}")

    # 3. Runs
    jobs = [(d, m) for d in args.dates for m in modes]
    print(f"\n=== {len(jobs)} runs · model {agent.model} ===")
    done, skipped, failed = 0, 0, []
    t_start = time.perf_counter()

    for i, (date, mode) in enumerate(jobs, 1):
        if not args.force and already_done(ticker, date, mode):
            print(f"\n[{i}/{len(jobs)}] {date} · {mode}: already done, skipping (--force to repeat)")
            skipped += 1
            continue
        print(f"\n[{i}/{len(jobs)}] {date} · {mode}")
        t_run = time.perf_counter()
        try:
            run = agent.investigate if mode == "agent" else agent.rag_baseline
            result = run(ticker, date)
            base = save(result, RUNS)
            flag = "  INVALID CITATIONS" if result.citations_invalid else ""
            err = f"  ERROR: {result.error}" if result.error else ""
            causes = len(result.report.get("supported_causes", []))
            print(f"  {hms(time.perf_counter() - t_run)} · {causes} causes · "
                  f"{len(result.citations_valid)} valid citations{flag}{err}")
            print(f"  → {base.name}.md")
            done += 1
        except Exception as e:  # noqa: BLE001
            failed.append((date, mode, str(e)))
            print(f"  FAILED: {type(e).__name__}: {e}")
            traceback.print_exc(limit=1)

    print(f"\n=== Done in {hms(time.perf_counter() - t_start)}: "
          f"{done} run, {skipped} skipped, {len(failed)} failed ===")
    for date, mode, err in failed:
        print(f"  {date} · {mode}: {err}")
    print("\nAdd the ground truth cause of any new event to data/ground_truth.yaml,")
    print("then run: python scripts/evaluate.py")


if __name__ == "__main__":
    main()
