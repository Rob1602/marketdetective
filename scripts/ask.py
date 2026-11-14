"""Chiedi all'agente di spiegare un movimento.

Esempi:
    python scripts/ask.py NVDA 2026-11-09                 # agente con strumenti
    python scripts/ask.py NVDA 2026-11-09 --mode rag      # baseline: solo RAG, niente strumenti
    python scripts/ask.py NVDA 2026-11-09 --question "Was the drop caused by the CEO scandal?"

Il report viene stampato e salvato in runs/ (Markdown da leggere, JSON per la valutazione).
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from marketdetective.agent import MarketDetective  # noqa: E402
from marketdetective.config import PROJECT_ROOT, load_config  # noqa: E402
from marketdetective.news import NewsArchive  # noqa: E402
from marketdetective.rag import NewsIndex, OllamaEmbedder  # noqa: E402
from marketdetective.tools import Toolbox  # noqa: E402


def build_agent(cfg, model: str | None = None) -> MarketDetective:
    index = NewsIndex(NewsArchive(cfg.news_dir), OllamaEmbedder(cfg.embedding_model))
    index.build()
    return MarketDetective(Toolbox(cfg, index), model=model or cfg.llm.get("model", "qwen3:8b"),
                           temperature=float(cfg.llm.get("temperature", 0.0)))


def save(result, runs_dir: Path) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = runs_dir / f"{result.ticker}_{result.date}_{result.mode}_{stamp}"
    base.with_suffix(".json").write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    base.with_suffix(".md").write_text(result.to_markdown(), encoding="utf-8")
    return base


def main() -> None:
    ap = argparse.ArgumentParser(description="Investiga un movimento di borsa")
    ap.add_argument("ticker")
    ap.add_argument("date")
    ap.add_argument("--mode", choices=["agent", "rag"], default="agent")
    ap.add_argument("--question", default=None)
    ap.add_argument("--model", default=None, help="es. qwen3:14b (default: quello in config.yaml)")
    args = ap.parse_args()

    cfg = load_config()
    agent = build_agent(cfg, args.model)
    print(f"Modalità {args.mode}, modello {agent.model}\n")
    run = agent.investigate if args.mode == "agent" else agent.rag_baseline
    result = run(cfg.ticker(args.ticker).symbol, args.date, args.question)

    print("\n" + result.to_markdown())
    base = save(result, PROJECT_ROOT / "runs")
    print(f"\nSalvato in {base.relative_to(PROJECT_ROOT)}.md e .json")


if __name__ == "__main__":
    main()
