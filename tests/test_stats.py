import numpy as np
import pandas as pd
import pytest

from conftest import EVENT_DATE, SHOCK, TRUE_BETA_M, TRUE_BETA_S, make_returns
from marketdetective.config import EventStudyParams
from marketdetective.stats import (
    compute_returns,
    event_study,
    locate_event,
    rolling_volatility,
    scan_anomalies,
)

PARAMS = EventStudyParams(estimation_window=200, gap=10, event_window=(-1, 1), z_threshold=2.0)


def test_log_returns_sum_to_total_change():
    prices = pd.Series([100.0, 110.0, 99.0, 120.0], index=pd.bdate_range("2024-01-01", periods=4))
    r = compute_returns(prices)
    assert np.isclose(r.sum(), np.log(120 / 100))


def test_simple_returns():
    prices = pd.Series([100.0, 110.0], index=pd.bdate_range("2024-01-01", periods=2))
    assert np.isclose(compute_returns(prices, "simple").iloc[0], 0.10)


def test_rolling_volatility_annualized():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 2000))
    vol = rolling_volatility(r, window=500).dropna()
    assert abs(vol.mean() - 0.01 * np.sqrt(252)) < 0.01


def test_betas_recovered(returns):
    res = event_study(returns, EVENT_DATE, PARAMS, model="market_sector")
    assert res.model.betas["market"] == pytest.approx(TRUE_BETA_M, abs=0.15)
    assert res.model.betas["sector_excess"] == pytest.approx(TRUE_BETA_S, abs=0.2)
    assert res.model.r2 > 0.5


def test_firm_specific_shock_detected(returns):
    res = event_study(returns, EVENT_DATE, PARAMS)
    assert res.is_anomalous
    assert res.day0_abnormal == pytest.approx(SHOCK, abs=0.03)
    assert res.day0_p < 0.01
    assert res.decomposition.dominant_driver == "company-specific"


def test_market_driven_move_is_not_anomalous():
    # The market drops 5%: the stock falls a lot, but beta explains it.
    df = make_returns(shock=0.0, market_shock=-0.05)
    res = event_study(df, EVENT_DATE, PARAMS)
    assert res.day0_return < -0.03
    assert not res.is_anomalous
    assert res.decomposition.dominant_driver == "market"


def test_decomposition_adds_up(returns):
    d = event_study(returns, EVENT_DATE, PARAMS).decomposition
    assert d.alpha + d.market + d.sector + d.idiosyncratic == pytest.approx(d.actual, abs=1e-12)


def test_no_look_ahead(returns):
    """Changing the data from the event day onward must not change the model estimates."""
    base = event_study(returns, EVENT_DATE, PARAMS)
    altered = returns.copy()
    pos = returns.index.get_loc(pd.Timestamp(EVENT_DATE))
    altered.iloc[pos - 1:, :] *= 5
    after = event_study(altered, EVENT_DATE, PARAMS)
    assert after.model.alpha == base.model.alpha
    assert after.model.betas == base.model.betas
    assert pd.Timestamp(base.model.estimation_end) < pd.Timestamp(EVENT_DATE)


def test_car_is_sum_of_abnormal_returns(returns):
    res = event_study(returns, EVENT_DATE, PARAMS)
    assert res.car == pytest.approx(sum(r["abnormal"] for r in res.window))
    assert [r["rel_day"] for r in res.window] == [-1, 0, 1]


def test_single_factor_model_notes_sector(returns):
    res = event_study(returns, EVENT_DATE, PARAMS, model="market")
    assert "sector_excess" not in res.model.betas
    assert res.decomposition.sector == 0.0
    assert any("Single-factor" in n for n in res.notes)


def test_non_trading_day_raises(returns):
    with pytest.raises(ValueError, match="not a trading day"):
        locate_event(returns.index, "2024-03-16")  # a Saturday


def test_insufficient_history_raises(returns):
    with pytest.raises(ValueError, match="Not enough history"):
        event_study(returns, returns.index[50], PARAMS)


def test_truncated_window_near_end(returns):
    res = event_study(returns, returns.index[-1], PARAMS)
    assert [r["rel_day"] for r in res.window] == [-1, 0]
    assert any("truncated" in n for n in res.notes)


def test_to_dict_is_json_serializable(returns):
    import json
    json.dumps(event_study(returns, EVENT_DATE, PARAMS).to_dict())


def test_scan_finds_the_event(returns):
    found = scan_anomalies(returns, PARAMS, start="2024-03-01", end="2024-03-29")
    assert EVENT_DATE in set(found["date"])
    assert found["is_anomalous"].all()


def test_scan_can_also_return_quiet_days(returns):
    everything = scan_anomalies(returns, PARAMS, start="2024-03-01", end="2024-03-29", only_anomalous=False)
    quiet = everything[~everything["is_anomalous"]]
    assert len(everything) > len(quiet) > 0            # both anomalous and quiet days are present
    assert quiet["t"].abs().max() <= PARAMS.z_threshold
    assert EVENT_DATE not in set(quiet["date"])
