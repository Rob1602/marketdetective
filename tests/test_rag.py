import numpy as np
import pandas as pd

from marketdetective.news import NewsArchive, NewsItem
from marketdetective.rag import NewsIndex, market_close_utc

VOCAB = ["export", "china", "ban", "earnings", "beat", "revenue", "fed", "rates", "ceo"]


class FakeEmbedder:
    """Fake bag-of-words embeddings: enough to check the filters and the ranking."""
    name = "fake"

    def __init__(self):
        self.doc_calls = 0

    def _vec(self, text):
        t = text.lower()
        return np.array([t.count(w) for w in VOCAB], dtype=np.float32) + 1e-3

    def embed_documents(self, texts):
        self.doc_calls += len(texts)
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


def item(i, ticker, when, headline):
    return NewsItem(id=f"n{i}", ticker=ticker, published=pd.Timestamp(when, tz="UTC").isoformat(),
                    source="Test", headline=headline, summary="", url="")


def make_index(tmp_path):
    archive = NewsArchive(tmp_path)
    archive.add([
        item(1, "NVDA", "2024-04-18 14:00", "US weighs export ban on chips to China"),
        item(2, "NVDA", "2024-04-19 15:00", "Analysts expect earnings beat and record revenue"),
        item(3, "NVDA", "2024-04-19 22:30", "China export ban confirmed after the close"),   # after the close
        item(4, "AAPL", "2024-04-19 14:00", "Apple faces China export ban"),                 # other ticker
        item(5, "NVDA", "2024-03-01 14:00", "Old news on China export ban"),                 # too old
    ])
    return NewsIndex(archive, FakeEmbedder())


def test_market_close_is_new_york_four_pm():
    assert market_close_utc("2024-04-19") == pd.Timestamp("2024-04-19 20:00", tz="UTC")  # summer time
    assert market_close_utc("2024-01-19") == pd.Timestamp("2024-01-19 21:00", tz="UTC")  # winter time


def test_search_filters_ticker_date_and_lookahead(tmp_path):
    idx = make_index(tmp_path)
    hits = idx.search_event("china export ban", "NVDA", "2024-04-19")
    ids = [h.item.id for h in hits]
    assert ids[0] == "n1"                 # the most relevant
    assert "n3" not in ids                # after the close: no look-ahead
    assert "n4" not in ids                # other ticker
    assert "n5" not in ids                # outside the window
    assert set(ids) == {"n1", "n2"}


def test_ranking_follows_query(tmp_path):
    idx = make_index(tmp_path)
    hits = idx.search_event("earnings revenue", "NVDA", "2024-04-19")
    assert hits[0].item.id == "n2"


def test_embeddings_are_cached(tmp_path):
    idx = make_index(tmp_path)
    assert idx.build() == 5
    again = NewsIndex(idx.archive, idx.embedder)
    assert again.build() == 0             # everything already cached
    assert idx.embedder.doc_calls == 5


def test_no_news_returns_empty(tmp_path):
    idx = make_index(tmp_path)
    assert idx.search_event("anything", "TSLA", "2024-04-19") == []
