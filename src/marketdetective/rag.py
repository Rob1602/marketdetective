"""RAG: semantic search over the news archive, filtered by ticker and date.

How it works:
1. Every news item (headline + summary) becomes a vector (embedding), computed by a local model
   served by Ollama (nomic-embed-text).
2. The question becomes a vector too.
3. News is filtered to the right ticker and to items published BEFORE the close of the event day
   (no look-ahead), then ranked by cosine similarity.

Why a NumPy index and not a vector database: with a few thousand news items, an exact search over
every vector takes milliseconds, is transparent and adds no dependency. A much larger archive
would call for Chroma or FAISS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import numpy as np
import pandas as pd

from .news import NewsArchive, NewsItem

MARKET_TZ = "America/New_York"
MARKET_CLOSE = "16:00"


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class OllamaEmbedder:
    """Local embeddings through Ollama. Requires: ollama pull nomic-embed-text"""

    # nomic-embed-text is trained with these prefixes to tell documents from queries.
    DOC_PREFIX = "search_document: "
    QUERY_PREFIX = "search_query: "

    def __init__(self, model: str = "nomic-embed-text", batch_size: int = 32):
        self.name = model
        self.batch_size = batch_size

    def _embed(self, texts: Sequence[str]) -> np.ndarray:
        import ollama

        out = []
        for i in range(0, len(texts), self.batch_size):
            resp = ollama.embed(model=self.name, input=list(texts[i:i + self.batch_size]))
            out += resp["embeddings"]
        return np.asarray(out, dtype=np.float32)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        return self._embed([self.DOC_PREFIX + t for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([self.QUERY_PREFIX + text])[0]


def market_close_utc(date) -> pd.Timestamp:
    """US market close (16:00 New York) on that day, in UTC.

    An item published after the close cannot explain that day's move.
    """
    local = pd.Timestamp(f"{pd.Timestamp(date).date()} {MARKET_CLOSE}").tz_localize(MARKET_TZ)
    return local.tz_convert("UTC")


@dataclass
class SearchHit:
    item: NewsItem
    score: float

    def to_dict(self) -> dict:
        return {
            "id": self.item.id,
            "published": self.item.published,
            "source": self.item.source,
            "headline": self.item.headline,
            "summary": self.item.summary,
            "url": self.item.url,
            "score": round(self.score, 3),
        }


class NewsIndex:
    """Vector index over the news, with the embeddings cached on disk."""

    def __init__(self, archive: NewsArchive, embedder: Embedder, cache_path: str | Path | None = None):
        self.archive = archive
        self.embedder = embedder
        safe = embedder.name.replace("/", "_").replace(":", "_")
        self.cache_path = Path(cache_path) if cache_path else archive.dir / f"embeddings_{safe}.npz"
        self.items: list[NewsItem] = []
        self.vectors = np.zeros((0, 0), dtype=np.float32)

    def build(self) -> int:
        """Load the archive and compute the missing embeddings. Returns how many were computed."""
        self.items = self.archive.load_all()
        cache: dict[str, np.ndarray] = {}
        if self.cache_path.exists():
            data = np.load(self.cache_path, allow_pickle=False)
            cache = dict(zip(data["ids"].tolist(), data["vectors"]))

        missing = [it for it in self.items if it.id not in cache]
        if missing:
            vecs = self.embedder.embed_documents([it.text() for it in missing])
            cache.update({it.id: v for it, v in zip(missing, vecs)})
            ids = list(cache)
            np.savez(self.cache_path, ids=np.array(ids), vectors=np.stack([cache[i] for i in ids]))

        if self.items:
            m = np.stack([cache[it.id] for it in self.items]).astype(np.float32)
            self.vectors = m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)
        return len(missing)

    def search(self, query: str, ticker: str, until: pd.Timestamp, lookback_days: int = 7,
               k: int = 5) -> list[SearchHit]:
        """The k most relevant items about `ticker`, published between `until - lookback_days` and `until`."""
        if not self.items:
            self.build()
        until = pd.Timestamp(until)
        until = until.tz_localize("UTC") if until.tzinfo is None else until.tz_convert("UTC")
        since = until - pd.Timedelta(days=lookback_days)

        idx = [i for i, it in enumerate(self.items)
               if it.ticker == ticker.upper() and since <= it.published_ts <= until]
        if not idx:
            return []
        q = self.embedder.embed_query(query).astype(np.float32)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        scores = self.vectors[idx] @ q
        order = np.argsort(-scores)[:k]
        return [SearchHit(self.items[idx[j]], float(scores[j])) for j in order]

    def search_event(self, query: str, ticker: str, event_date, lookback_days: int = 7,
                     k: int = 5) -> list[SearchHit]:
        """Search for an event: only items published before the close of the event day."""
        return self.search(query, ticker, market_close_utc(event_date), lookback_days, k)
