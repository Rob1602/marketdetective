"""Market data: prices from yfinance, cached locally on disk.

Prices are adjusted closes (dividends and splits). Without the adjustment a split would look like
a crash and be flagged as an anomaly.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd

# A downloader takes (ticker, start, end) and returns a price Series indexed by date.
Downloader = Callable[[str, str, str | None], pd.Series]


def yfinance_downloader(symbol: str, start: str, end: str | None = None) -> pd.Series:
    """Download daily adjusted closes from Yahoo Finance."""
    import yfinance as yf  # local import: the tests do not need yfinance

    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise ValueError(f"No data downloaded for {symbol}. Check the ticker and the connection.")
    close = df["Close"]
    # Recent yfinance versions return MultiIndex columns even for a single ticker.
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close.index = pd.to_datetime(close.index).tz_localize(None).normalize()
    close.name = symbol
    return close.dropna()


class PriceStore:
    """Fetch prices and store them as CSV, one file per ticker, to avoid re-downloading."""

    def __init__(self, cache_dir: str | Path, downloader: Downloader = yfinance_downloader, stale_days: int = 4):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.downloader = downloader
        self.stale_days = stale_days

    def _path(self, symbol: str) -> Path:
        return self.cache_dir / f"{symbol.upper()}.csv"

    def _read_cache(self, symbol: str) -> pd.Series | None:
        p = self._path(symbol)
        if not p.exists():
            return None
        s = pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]
        s.name = symbol.upper()
        return s

    def get(self, symbol: str, start: str, end: str | None = None, refresh: bool = False) -> pd.Series:
        """Prices for `symbol` between `start` and `end`, inclusive. Downloads only if the cache falls short."""
        symbol = symbol.upper()
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end) if end else pd.Timestamp(date.today())

        cached = None if refresh else self._read_cache(symbol)
        need_download = (
            cached is None
            or cached.empty
            or cached.index.min() > start_ts + timedelta(days=self.stale_days)
            or cached.index.max() < end_ts - timedelta(days=self.stale_days)
        )
        if need_download:
            # yfinance treats `end` as exclusive, so add a day.
            dl_end = (end_ts + timedelta(days=1)).strftime("%Y-%m-%d")
            fresh = self.downloader(symbol, start_ts.strftime("%Y-%m-%d"), dl_end)
            fresh = fresh.sort_index()
            if cached is not None and not cached.empty:
                fresh = pd.concat([cached, fresh])
                fresh = fresh[~fresh.index.duplicated(keep="last")].sort_index()
            fresh.to_frame().to_csv(self._path(symbol))
            cached = fresh

        return cached.loc[start_ts:end_ts]

    def get_many(self, symbols: Iterable[str], start: str, end: str | None = None) -> pd.DataFrame:
        """Prices for several tickers aligned on common dates, keeping only days where all have a price."""
        frames = {s.upper(): self.get(s, start, end) for s in symbols}
        return pd.DataFrame(frames).dropna(how="any")


def load_event_prices(store: PriceStore, ticker: str, market: str, sector: str,
                      start: str, end: str | None = None) -> pd.DataFrame:
    """Aligned prices for the stock, the market index and the sector ETF, as stock/market/sector columns."""
    df = store.get_many([ticker, market, sector], start, end)
    return df.rename(columns={ticker.upper(): "stock", market.upper(): "market", sector.upper(): "sector"})
