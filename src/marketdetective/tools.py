"""The tools the LLM can call (function calling).

Principle: numbers and documents always come from code. The model decides which tool to call and
with which arguments, then interprets the result.

Each tool has a Python function that runs it and a JSON schema that describes it to the model.
"""

from __future__ import annotations

import pandas as pd

from .config import Config
from .data import PriceStore, load_event_prices
from .news import NewsItem
from .rag import NewsIndex, market_close_utc
from .stats import compute_returns, event_study

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_event",
            "description": (
                "Event study of a stock on one trading day. Returns the day's return, how much of it "
                "is explained by the overall market and by the sector, the company-specific (abnormal) "
                "part, and whether the move is statistically anomalous. Call this FIRST."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker, e.g. NVDA"},
                    "date": {"type": "string", "description": "Event date, format YYYY-MM-DD"},
                },
                "required": ["ticker", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_news",
            "description": (
                "Semantic search in the news archive about a stock. Only returns news published before "
                "the market close of the event date. Use specific queries about possible causes "
                "(e.g. 'earnings results guidance', 'CEO investigation', 'competition custom chips', "
                "'export restrictions China'). You can call it several times with different queries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Stock ticker, e.g. NVDA"},
                    "date": {"type": "string", "description": "Event date, format YYYY-MM-DD"},
                    "query": {"type": "string", "description": "What to look for, in English"},
                    "days_before": {
                        "type": "integer",
                        "description": "How many days before the event to search (1 to 7, default 2)",
                    },
                },
                "required": ["ticker", "date", "query"],
            },
        },
    },
]


def _pct(x: float) -> str:
    return f"{x * 100:+.2f}%"


class Toolbox:
    """Runs the tools and tracks which documents the model was shown, so citations can be checked."""

    def __init__(self, cfg: Config, index: NewsIndex, store: PriceStore | None = None):
        self.cfg = cfg
        self.index = index
        self.store = store or PriceStore(cfg.cache_dir)
        self.seen_news: dict[str, NewsItem] = {}
        self.event_results: dict[tuple[str, str], object] = {}

    def run(self, name: str, args: dict) -> dict:
        try:
            if name == "analyze_event":
                return self.analyze_event(**args)
            if name == "search_news":
                return self.search_news(**args)
            return {"error": f"Unknown tool '{name}'. Available: analyze_event, search_news."}
        except TypeError as e:
            return {"error": f"Wrong arguments for {name}: {e}"}
        except (KeyError, ValueError) as e:
            return {"error": str(e).strip("'\"")}

    def analyze_event(self, ticker: str, date: str) -> dict:
        info = self.cfg.ticker(ticker)
        prices = load_event_prices(self.store, info.symbol, info.market, info.sector, start=self.cfg.data_start)
        res = event_study(compute_returns(prices), date, self.cfg.event_study)
        self.event_results[(info.symbol, res.event_date)] = res
        d = res.decomposition
        return {
            "ticker": info.symbol,
            "company": info.name,
            "date": res.event_date,
            "benchmarks": {"market": info.market, "sector": info.sector},
            "day_return": _pct(res.day0_return),
            "explained_by_market": _pct(d.market),
            "explained_by_sector": _pct(d.sector),
            "company_specific_abnormal_return": _pct(d.idiosyncratic),
            "t_stat": round(res.day0_t, 2),
            "p_value": round(res.day0_p, 4),
            "is_anomalous": bool(res.is_anomalous),
            "main_driver": d.dominant_driver,
            "cumulative_abnormal_return_days_-1_to_+1": _pct(res.car),
            "notes": res.notes,
        }

    def search_news(self, ticker: str, date: str, query: str, days_before: int = 2) -> dict:
        ticker = self.cfg.ticker(ticker).symbol
        days_before = max(1, min(int(days_before), 7))
        hits = self.index.search_event(query, ticker, date, lookback_days=days_before, k=self.cfg.top_k)
        for h in hits:
            self.seen_news[h.item.id] = h.item
        return {
            "query": query,
            "only_published_before": str(market_close_utc(date)),
            "results": [
                {"id": h.item.id, "published": h.item.published[:16], "source": h.item.source,
                 "headline": h.item.headline, "summary": h.item.summary[:300]}
                for h in hits
            ],
            "note": "No news found in the archive for this period." if not hits else "",
        }

    def is_valid_citation(self, news_id: str, ticker: str, date: str) -> bool:
        """A citation is valid if the model actually received it and it was published before the close."""
        it = self.seen_news.get(news_id)
        return bool(it and it.ticker == ticker.upper() and it.published_ts <= market_close_utc(date))
