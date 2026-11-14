import pandas as pd

from marketdetective.data import PriceStore, load_event_prices


class FakeDownloader:
    """Simula yfinance: restituisce prezzi finti e conta le chiamate."""

    def __init__(self):
        self.calls = []

    def __call__(self, symbol, start, end=None):
        self.calls.append((symbol, start, end))
        idx = pd.bdate_range(start, pd.Timestamp(end) - pd.Timedelta(days=1))
        base = {"AAA": 100.0, "MKT": 400.0, "SEC": 50.0}.get(symbol, 10.0)
        return pd.Series([base + i for i in range(len(idx))], index=idx, name=symbol, dtype=float)


def test_cache_avoids_second_download(tmp_path):
    dl = FakeDownloader()
    store = PriceStore(tmp_path, downloader=dl)
    a = store.get("AAA", "2024-01-01", "2024-03-29")
    b = store.get("AAA", "2024-01-01", "2024-03-29")
    assert len(dl.calls) == 1
    pd.testing.assert_series_equal(a, b, check_freq=False)
    assert (tmp_path / "AAA.csv").exists()


def test_cache_extends_when_range_grows(tmp_path):
    dl = FakeDownloader()
    store = PriceStore(tmp_path, downloader=dl)
    store.get("AAA", "2024-01-01", "2024-02-29")
    s = store.get("AAA", "2024-01-01", "2024-04-30")
    assert len(dl.calls) == 2
    assert s.index.max() >= pd.Timestamp("2024-04-26")
    assert not s.index.duplicated().any()


def test_load_event_prices_aligns_and_renames(tmp_path):
    store = PriceStore(tmp_path, downloader=FakeDownloader())
    df = load_event_prices(store, "AAA", "MKT", "SEC", "2024-01-01", "2024-03-29")
    assert list(df.columns) == ["stock", "market", "sector"]
    assert not df.isna().any().any()
