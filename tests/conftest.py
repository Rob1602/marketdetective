"""Synthetic data with known parameters, so the estimates can be checked against the truth."""

import numpy as np
import pandas as pd
import pytest

TRUE_ALPHA = 0.0003
TRUE_BETA_M = 1.4
TRUE_BETA_S = 0.8
NOISE = 0.012
EVENT_DATE = "2024-03-15"
SHOCK = -0.07  # company-specific shock on the event day


def make_returns(seed: int = 0, n: int = 400, shock: float = SHOCK, market_shock: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end="2024-06-28", periods=n)
    market = rng.normal(0.0004, 0.010, n)
    sector_excess = rng.normal(0.0, 0.008, n)
    sector = market + sector_excess
    eps = rng.normal(0, NOISE, n)
    df = pd.DataFrame({"market": market, "sector": sector}, index=idx)
    pos = idx.get_loc(pd.Timestamp(EVENT_DATE))
    df.iloc[pos, df.columns.get_loc("market")] += market_shock
    df.iloc[pos, df.columns.get_loc("sector")] += market_shock
    df["stock"] = TRUE_ALPHA + TRUE_BETA_M * df["market"] + TRUE_BETA_S * (df["sector"] - df["market"]) + eps
    df.iloc[pos, df.columns.get_loc("stock")] += shock
    return df


@pytest.fixture
def returns() -> pd.DataFrame:
    return make_returns()
