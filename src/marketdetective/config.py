"""Configuration loading from config.yaml."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


@dataclass(frozen=True)
class TickerInfo:
    symbol: str
    name: str
    market: str
    sector: str


@dataclass(frozen=True)
class EventStudyParams:
    estimation_window: int = 120
    gap: int = 10
    event_window: tuple[int, int] = (-1, 1)
    vol_window: int = 20
    z_threshold: float = 2.0


@dataclass(frozen=True)
class Config:
    tickers: dict[str, TickerInfo]
    event_study: EventStudyParams
    data_start: str
    cache_dir: Path
    llm: dict = field(default_factory=dict)
    news_dir: Path = Path("data/news")
    news_days_before: int = 3
    news_days_after: int = 2
    embedding_model: str = "nomic-embed-text"
    top_k: int = 5

    def ticker(self, symbol: str) -> TickerInfo:
        try:
            return self.tickers[symbol.upper()]
        except KeyError:
            known = ", ".join(self.tickers) or "none"
            raise KeyError(f"Ticker '{symbol}' is not in config.yaml (configured: {known})") from None


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    tickers = {
        sym.upper(): TickerInfo(symbol=sym.upper(), name=v.get("name", sym), market=v["market"], sector=v["sector"])
        for sym, v in (raw.get("tickers") or {}).items()
    }
    es = raw.get("event_study", {})
    params = EventStudyParams(
        estimation_window=int(es.get("estimation_window", 120)),
        gap=int(es.get("gap", 10)),
        event_window=tuple(es.get("event_window", [-1, 1])),
        vol_window=int(es.get("vol_window", 20)),
        z_threshold=float(es.get("z_threshold", 2.0)),
    )
    data = raw.get("data", {})
    news = raw.get("news", {})
    rag = raw.get("rag", {})

    def resolve(p: str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else path.parent / p

    return Config(
        tickers=tickers,
        event_study=params,
        data_start=str(data.get("start", "2018-01-01")),
        cache_dir=resolve(data.get("cache_dir", "data/prices")),
        llm=raw.get("llm", {}),
        news_dir=resolve(news.get("archive_dir", "data/news")),
        news_days_before=int(news.get("days_before", 3)),
        news_days_after=int(news.get("days_after", 2)),
        embedding_model=str(rag.get("embedding_model", "nomic-embed-text")),
        top_k=int(rag.get("top_k", 5)),
    )
