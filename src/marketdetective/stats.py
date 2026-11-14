"""Statistics: returns, volatility and an event study based on a market model.

The question it answers: is the stock's move on day X unusual, and how much of it comes from the
market, from the sector, or from something specific to the company?

Timeline (trading days, 0 = event day):

    |------ estimation window ------|-- gap --|--- event window ---|
    t0                            t1          -1        0        +1

Alpha and beta are estimated ONLY on the estimation window, which ends before the event window, so
no data from the event or after it enters the estimate.

Two models:
- "market":        R_stock = a + b_m * R_market + e
- "market_sector": R_stock = a + b_m * R_market + b_s * (R_sector - R_market) + e
  The sector factor is expressed in excess of the market, so the two factors are less correlated
  and their contributions can be read separately: b_s * (R_sector - R_market) is the part due to
  the sector beyond the market.

The abnormal return (AR) is the residual e: what market and sector do not explain.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps

from .config import EventStudyParams

TRADING_DAYS = 252


# ------------------------------------------------------------------- returns

def compute_returns(prices: pd.Series | pd.DataFrame, kind: str = "log") -> pd.Series | pd.DataFrame:
    """Daily returns. 'log' (the default) adds up over time; 'simple' is the percentage change."""
    if kind == "log":
        out = np.log(prices / prices.shift(1))
    elif kind == "simple":
        out = prices.pct_change()
    else:
        raise ValueError("kind must be 'log' or 'simple'")
    return out.dropna(how="all")


def rolling_volatility(returns: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """Rolling standard deviation of returns, annualised with sqrt(252) when asked."""
    vol = returns.rolling(window).std()
    return vol * np.sqrt(TRADING_DAYS) if annualize else vol


# ---------------------------------------------------------------- positioning

def locate_event(index: pd.DatetimeIndex, event_date) -> int:
    """Position of the event day in the index. Raises a clear error if it is not a trading day."""
    ts = pd.Timestamp(event_date).normalize()
    if ts in index:
        return int(index.get_loc(ts))
    after = index[index > ts]
    hint = f" The next trading day is {after[0].date()}." if len(after) else ""
    raise ValueError(f"{ts.date()} is not a trading day present in the data.{hint}")


def _estimation_bounds(event_pos: int, params: EventStudyParams) -> tuple[int, int]:
    """Index bounds [start, end) of the estimation window."""
    est_end = event_pos + params.event_window[0] - params.gap
    est_start = est_end - params.estimation_window
    if est_start < 0:
        raise ValueError(
            f"Not enough history before the event: {params.estimation_window + params.gap - params.event_window[0]} "
            f"trading days are needed, {event_pos} are available. Move data.start earlier in config.yaml."
        )
    return est_start, est_end


# ------------------------------------------------------------- market model

@dataclass
class MarketModelFit:
    model: str
    alpha: float
    betas: dict[str, float]
    sigma: float          # standard deviation of the residuals over the estimation window
    r2: float
    n_obs: int
    dof: int
    estimation_start: str
    estimation_end: str


def _factors(returns: pd.DataFrame, model: str) -> pd.DataFrame:
    if model == "market":
        return returns[["market"]]
    if model == "market_sector":
        return pd.DataFrame({"market": returns["market"], "sector_excess": returns["sector"] - returns["market"]})
    raise ValueError("model must be 'market' or 'market_sector'")


def fit_market_model(y: pd.Series, X: pd.DataFrame, model: str) -> MarketModelFit:
    """OLS regression of y on the factors X, with an intercept."""
    A = np.column_stack([np.ones(len(X)), X.to_numpy()])
    coef, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
    resid = y.to_numpy() - A @ coef
    n, k = A.shape
    dof = n - k
    sigma = float(np.sqrt(resid @ resid / dof))
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - float(resid @ resid) / ss_tot if ss_tot > 0 else float("nan")
    return MarketModelFit(
        model=model,
        alpha=float(coef[0]),
        betas={c: float(b) for c, b in zip(X.columns, coef[1:])},
        sigma=sigma,
        r2=r2,
        n_obs=n,
        dof=dof,
        estimation_start=str(X.index[0].date()),
        estimation_end=str(X.index[-1].date()),
    )


# --------------------------------------------------------------- event study

@dataclass
class Decomposition:
    """Decomposition of the day 0 return: the parts add up to the actual return."""
    actual: float
    alpha: float
    market: float
    sector: float
    idiosyncratic: float
    dominant_driver: str


@dataclass
class EventStudyResult:
    event_date: str
    model: MarketModelFit
    window: list[dict]              # one row per day of the event window
    day0_return: float
    day0_abnormal: float
    day0_t: float
    day0_p: float
    car: float
    car_t: float
    car_p: float
    raw_zscore: float               # day 0 return against the mean and sd of the estimation window
    pre_event_volatility: float     # annualised volatility over the days before the event
    is_anomalous: bool
    decomposition: Decomposition
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-serialisable version, used by the agent tools."""
        return asdict(self)

    def summary(self) -> str:
        d = self.decomposition
        verdict = "ANOMALOUS" if self.is_anomalous else "not anomalous"
        lines = [
            f"Event {self.event_date} - model {self.model.model} (R2 {self.model.r2:.2f}, estimated {self.model.estimation_start} -> {self.model.estimation_end})",
            f"Day 0 return: {self.day0_return:+.2%} | abnormal: {self.day0_abnormal:+.2%} (t={self.day0_t:.2f}, p={self.day0_p:.3f}) -> {verdict}",
            f"CAR over the window: {self.car:+.2%} (t={self.car_t:.2f}, p={self.car_p:.3f}) | raw z-score: {self.raw_zscore:.2f}",
            f"Decomposition: market {d.market:+.2%}, sector {d.sector:+.2%}, company-specific {d.idiosyncratic:+.2%}, alpha {d.alpha:+.3%} -> main driver: {d.dominant_driver}",
        ]
        lines += [f"Note: {n}" for n in self.notes]
        return "\n".join(lines)


def event_study(returns: pd.DataFrame, event_date, params: EventStudyParams | None = None,
                model: str = "market_sector") -> EventStudyResult:
    """Event study on a returns DataFrame with 'stock', 'market' and 'sector' columns.

    Returns must be logarithmic (the compute_returns default) so that CAR is a plain sum.
    """
    params = params or EventStudyParams()
    returns = returns.dropna()
    idx = returns.index
    pos = locate_event(idx, event_date)
    est_start, est_end = _estimation_bounds(pos, params)
    notes: list[str] = []

    X_all = _factors(returns, model)
    est = slice(est_start, est_end)
    fit = fit_market_model(returns["stock"].iloc[est], X_all.iloc[est], model)

    # Event window, truncated if the event is too close to the end of the data.
    w_lo, w_hi = params.event_window
    first, last = pos + w_lo, min(pos + w_hi, len(idx) - 1)
    if pos + w_hi > len(idx) - 1:
        notes.append(f"Event window truncated: no data after {idx[-1].date()}.")

    beta_vec = np.array([fit.betas[c] for c in X_all.columns])
    rows, car = [], 0.0
    for i in range(first, last + 1):
        actual = float(returns["stock"].iloc[i])
        expected = fit.alpha + float(X_all.iloc[i].to_numpy() @ beta_vec)
        ar = actual - expected
        car += ar
        L = i - first + 1
        rows.append({
            "rel_day": i - pos,
            "date": str(idx[i].date()),
            "actual": actual,
            "expected": expected,
            "abnormal": ar,
            "t": ar / fit.sigma,
            "car": car,
            "car_t": car / (fit.sigma * np.sqrt(L)),
        })

    day0 = next(r for r in rows if r["rel_day"] == 0)
    L = len(rows)
    car_t = car / (fit.sigma * np.sqrt(L))
    pval = lambda t: float(2 * sps.t.sf(abs(t), fit.dof))

    est_stock = returns["stock"].iloc[est]
    raw_z = (day0["actual"] - est_stock.mean()) / est_stock.std(ddof=1)
    vol_slice = returns["stock"].iloc[max(0, pos - params.vol_window):pos]
    pre_vol = float(vol_slice.std(ddof=1) * np.sqrt(TRADING_DAYS))

    # Day 0 contributions.
    x0 = X_all.iloc[pos]
    mkt = fit.betas["market"] * float(x0["market"])
    sec = fit.betas.get("sector_excess", 0.0) * float(x0.get("sector_excess", 0.0))
    parts = {"market": mkt, "sector": sec, "company-specific": day0["abnormal"]}
    dominant = max(parts, key=lambda k: abs(parts[k]))
    if model == "market":
        notes.append("Single-factor model: the sector effect is absorbed into the company-specific part.")

    return EventStudyResult(
        event_date=str(idx[pos].date()),
        model=fit,
        window=rows,
        day0_return=day0["actual"],
        day0_abnormal=day0["abnormal"],
        day0_t=day0["t"],
        day0_p=pval(day0["t"]),
        car=car,
        car_t=float(car_t),
        car_p=pval(car_t),
        raw_zscore=float(raw_z),
        pre_event_volatility=pre_vol,
        is_anomalous=abs(day0["t"]) > params.z_threshold,
        decomposition=Decomposition(
            actual=day0["actual"], alpha=fit.alpha, market=mkt, sector=sec,
            idiosyncratic=day0["abnormal"], dominant_driver=dominant,
        ),
        notes=notes,
    )


def scan_anomalies(returns: pd.DataFrame, params: EventStudyParams | None = None,
                   model: str = "market_sector", start=None, end=None,
                   only_anomalous: bool = True) -> pd.DataFrame:
    """Run the event study on every day between start and end.

    With only_anomalous=True (the default) it returns only the anomalous days, the candidate
    events. With False it returns every day, which is useful for picking a quiet day as a
    negative control: the system should not invent a cause where there is nothing to explain.
    """
    params = params or EventStudyParams()
    idx = returns.dropna().index
    days = idx[(idx >= pd.Timestamp(start)) if start else slice(None)]
    if end:
        days = days[days <= pd.Timestamp(end)]
    out = []
    for d in days:
        try:
            r = event_study(returns, d, params, model)
        except ValueError:
            continue
        if r.is_anomalous or not only_anomalous:
            out.append({"date": r.event_date, "return": r.day0_return, "abnormal": r.day0_abnormal,
                        "t": r.day0_t, "driver": r.decomposition.dominant_driver,
                        "is_anomalous": r.is_anomalous})
    return pd.DataFrame(out)
