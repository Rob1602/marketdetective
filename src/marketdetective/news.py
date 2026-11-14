"""News: download from Finnhub and local archive.

Downloaded news is stored as JSONL, one file per ticker:
    data/news/NVDA.jsonl

Why store it instead of querying the API every time:
- reproducibility: the evaluation always uses the same archive, even if the web changes;
- free tier limits: everything is downloaded once;
- the RAG then works offline.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

FINNHUB_URL = "https://finnhub.io/api/v1/company-news"


@dataclass
class NewsItem:
    id: str
    ticker: str
    published: str      # ISO 8601 in UTC, e.g. "2024-04-19T13:05:00+00:00"
    source: str
    headline: str
    summary: str
    url: str

    @property
    def published_ts(self) -> pd.Timestamp:
        return pd.Timestamp(self.published)

    def text(self) -> str:
        """The text used for embeddings and shown to the model."""
        return f"{self.headline}. {self.summary}".strip()

    def citation(self) -> str:
        return f"[{self.id}] {self.source}, {self.published[:10]}: {self.headline}"


class FinnhubClient:
    """Minimal client for Finnhub's company-news endpoint."""

    def __init__(self, api_key: str | None = None, session=None, timeout: float = 20):
        self.api_key = api_key or os.environ.get("FINNHUB_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Finnhub API key missing. Set it with:\n"
                "  conda env config vars set FINNHUB_API_KEY=your_key\n"
                "then reactivate the environment: conda deactivate && conda activate marketdetective"
            )
        if session is None:
            import requests
            session = requests.Session()
        self.session = session
        self.timeout = timeout

    def company_news(self, ticker: str, start: str, end: str) -> list[NewsItem]:
        params = {"symbol": ticker.upper(), "from": start, "to": end, "token": self.api_key}
        resp = self.session.get(FINNHUB_URL, params=params, timeout=self.timeout)
        if resp.status_code == 401:
            raise RuntimeError("Finnhub rejected the key (401). Check FINNHUB_API_KEY.")
        if resp.status_code == 429:
            raise RuntimeError("Too many requests to Finnhub (429). Wait a minute and retry.")
        resp.raise_for_status()
        return [parse_finnhub_item(raw, ticker) for raw in resp.json() if raw.get("headline")]


def parse_finnhub_item(raw: dict, ticker: str) -> NewsItem:
    published = datetime.fromtimestamp(int(raw["datetime"]), tz=timezone.utc).isoformat()
    return NewsItem(
        id=f"fh-{raw.get('id', '')}",
        ticker=ticker.upper(),
        published=published,
        source=raw.get("source", "") or "",
        headline=(raw.get("headline", "") or "").strip(),
        summary=(raw.get("summary", "") or "").strip(),
        url=raw.get("url", "") or "",
    )


class NewsArchive:
    """Local news archive, one JSONL file per ticker. Items already stored are not duplicated."""

    def __init__(self, archive_dir: str | Path):
        self.dir = Path(archive_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, ticker: str) -> Path:
        return self.dir / f"{ticker.upper()}.jsonl"

    def load(self, ticker: str) -> list[NewsItem]:
        p = self._path(ticker)
        if not p.exists():
            return []
        with open(p, encoding="utf-8") as f:
            return [NewsItem(**json.loads(line)) for line in f if line.strip()]

    def load_all(self) -> list[NewsItem]:
        items = []
        for p in sorted(self.dir.glob("*.jsonl")):
            items += self.load(p.stem)
        return items

    def add(self, items: Iterable[NewsItem]) -> int:
        """Add the new items and return how many were added."""
        by_ticker: dict[str, list[NewsItem]] = {}
        for it in items:
            by_ticker.setdefault(it.ticker, []).append(it)
        added = 0
        for ticker, new in by_ticker.items():
            existing = self.load(ticker)
            seen = {it.id for it in existing} | {(it.headline.lower(), it.published[:10]) for it in existing}
            fresh = []
            for it in new:
                key = (it.headline.lower(), it.published[:10])
                if it.id in seen or key in seen:
                    continue
                seen |= {it.id, key}
                fresh.append(it)
            if fresh:
                all_items = sorted(existing + fresh, key=lambda x: x.published)
                with open(self._path(ticker), "w", encoding="utf-8") as f:
                    for it in all_items:
                        f.write(json.dumps(asdict(it), ensure_ascii=False) + "\n")
            added += len(fresh)
        return added


def event_news_range(event_date: str, days_before: int, days_after: int) -> tuple[str, str]:
    d = pd.Timestamp(event_date)
    return ((d - timedelta(days=days_before)).strftime("%Y-%m-%d"),
            (d + timedelta(days=days_after)).strftime("%Y-%m-%d"))


# Finnhub returns at most about 250 items per request, the most recent ones.
# For a widely covered stock such as NVIDIA a 6-day window exceeds that, and the items lost
# are exactly those published BEFORE the event. Hence one request per day.
FINNHUB_PAGE_CAP = 240


@dataclass
class DownloadReport:
    downloaded: int
    added: int
    per_day: dict[str, int]
    saturated_days: list[str]   # days where Finnhub probably truncated the results


def download_event_news(client: FinnhubClient, archive: NewsArchive, ticker: str, event_date: str,
                        days_before: int = 3, days_after: int = 2, pause: float = 1.1) -> DownloadReport:
    """Download the news around an event, one day at a time, and store it in the archive.

    `pause` keeps requests under Finnhub's free limit of 60 per minute.
    """
    import time

    start, end = event_news_range(event_date, days_before, days_after)
    per_day: dict[str, int] = {}
    items: list[NewsItem] = []
    for i, day in enumerate(pd.date_range(start, end, freq="D")):
        if i and pause:
            time.sleep(pause)
        d = day.strftime("%Y-%m-%d")
        got = client.company_news(ticker, d, d)
        per_day[d] = len(got)
        items += got
    return DownloadReport(
        downloaded=len(items),
        added=archive.add(items),
        per_day=per_day,
        saturated_days=[d for d, n in per_day.items() if n >= FINNHUB_PAGE_CAP],
    )
