"""Layer statistico: rendimenti, volatilità e event study con market model.

Domanda a cui risponde: "il movimento del titolo nel giorno X è anomalo, e quanto
dipende dal mercato, dal settore o da qualcosa di specifico dell'azienda?"

Schema temporale (in giorni di borsa, 0 = giorno dell'evento):

    |------ finestra di stima ------|-- gap --|--- finestra evento ---|
    t0                            t1          -1         0         +1

Alfa e beta si stimano SOLO sulla finestra di stima, che termina prima della
finestra evento: nessun dato dell'evento (o successivo) entra nella stima.

Modelli disponibili:
- "market":        R_stock = a + b_m * R_market + e
- "market_sector": R_stock = a + b_m * R_market + b_s * (R_sector - R_market) + e
  Il fattore di settore è espresso in eccesso rispetto al mercato, così i due
  fattori sono meno correlati e i contributi si interpretano separatamente:
  b_s * (R_sector - R_market) è la parte dovuta al settore *oltre* il mercato.

Il rendimento anomalo (AR) è il residuo e: ciò che mercato e settore non spiegano.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as sps

from .config import EventStudyParams

TRADING_DAYS = 252


# ---------------------------------------------------------------- rendimenti

def compute_returns(prices: pd.Series | pd.DataFrame, kind: str = "log") -> pd.Series | pd.DataFrame:
    """Rendimenti giornalieri. 'log' (default) si somma nel tempo; 'simple' è la variazione percentuale."""
    if kind == "log":
        out = np.log(prices / prices.shift(1))
    elif kind == "simple":
        out = prices.pct_change()
    else:
        raise ValueError("kind deve essere 'log' o 'simple'")
    return out.dropna(how="all")


def rolling_volatility(returns: pd.Series, window: int = 20, annualize: bool = True) -> pd.Series:
    """Deviazione standard mobile dei rendimenti, annualizzata con sqrt(252) se richiesto."""
    vol = returns.rolling(window).std()
    return vol * np.sqrt(TRADING_DAYS) if annualize else vol


# ------------------------------------------------------------ posizionamento

def locate_event(index: pd.DatetimeIndex, event_date) -> int:
    """Posizione del giorno dell'evento nell'indice. Errore chiaro se non è un giorno di borsa."""
    ts = pd.Timestamp(event_date).normalize()
    if ts in index:
        return int(index.get_loc(ts))
    after = index[index > ts]
    hint = f" Il primo giorno di borsa successivo è {after[0].date()}." if len(after) else ""
    raise ValueError(f"{ts.date()} non è un giorno di borsa presente nei dati.{hint}")


def _estimation_bounds(event_pos: int, params: EventStudyParams) -> tuple[int, int]:
    """Indici [inizio, fine) della finestra di stima."""
    est_end = event_pos + params.event_window[0] - params.gap
    est_start = est_end - params.estimation_window
    if est_start < 0:
        raise ValueError(
            f"Storico insufficiente prima dell'evento: servono {params.estimation_window + params.gap - params.event_window[0]} "
            f"giorni di borsa, ne sono disponibili {event_pos}. Anticipa data.start in config.yaml."
        )
    return est_start, est_end


# ------------------------------------------------------------- market model

@dataclass
class MarketModelFit:
    model: str
    alpha: float
    betas: dict[str, float]
    sigma: float          # deviazione standard dei residui nella finestra di stima
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
    raise ValueError("model deve essere 'market' o 'market_sector'")


def fit_market_model(y: pd.Series, X: pd.DataFrame, model: str) -> MarketModelFit:
    """Regressione OLS di y sui fattori X con intercetta."""
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
    """Scomposizione del rendimento del giorno 0: la somma delle parti è il rendimento effettivo."""
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
    window: list[dict]              # una riga per giorno della finestra evento
    day0_return: float
    day0_abnormal: float
    day0_t: float
    day0_p: float
    car: float
    car_t: float
    car_p: float
    raw_zscore: float               # rendimento del giorno 0 rispetto a media/dev. std. della finestra di stima
    pre_event_volatility: float     # volatilità annualizzata nei giorni prima dell'evento
    is_anomalous: bool
    decomposition: Decomposition
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Versione serializzabile in JSON (servirà ai tool dell'agente)."""
        return asdict(self)

    def summary(self) -> str:
        d = self.decomposition
        verdict = "ANOMALO" if self.is_anomalous else "non anomalo"
        lines = [
            f"Evento {self.event_date} - modello {self.model.model} (R2 {self.model.r2:.2f}, stima {self.model.estimation_start} -> {self.model.estimation_end})",
            f"Rendimento giorno 0: {self.day0_return:+.2%} | anomalo: {self.day0_abnormal:+.2%} (t={self.day0_t:.2f}, p={self.day0_p:.3f}) -> {verdict}",
            f"CAR finestra: {self.car:+.2%} (t={self.car_t:.2f}, p={self.car_p:.3f}) | z-score grezzo: {self.raw_zscore:.2f}",
            f"Scomposizione: mercato {d.market:+.2%}, settore {d.sector:+.2%}, specifico {d.idiosyncratic:+.2%}, alfa {d.alpha:+.3%} -> driver principale: {d.dominant_driver}",
        ]
        lines += [f"Nota: {n}" for n in self.notes]
        return "\n".join(lines)


def event_study(returns: pd.DataFrame, event_date, params: EventStudyParams | None = None,
                model: str = "market_sector") -> EventStudyResult:
    """Event study su un DataFrame di rendimenti con colonne 'stock', 'market', 'sector'.

    I rendimenti devono essere logaritmici (compute_returns default), così CAR è una somma.
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

    # Finestra evento, troncata se l'evento è troppo vicino alla fine dei dati.
    w_lo, w_hi = params.event_window
    first, last = pos + w_lo, min(pos + w_hi, len(idx) - 1)
    if pos + w_hi > len(idx) - 1:
        notes.append(f"Finestra evento troncata: mancano dati dopo {idx[-1].date()}.")

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

    # Contributi del giorno 0.
    x0 = X_all.iloc[pos]
    mkt = fit.betas["market"] * float(x0["market"])
    sec = fit.betas.get("sector_excess", 0.0) * float(x0.get("sector_excess", 0.0))
    parts = {"mercato": mkt, "settore": sec, "specifico dell'azienda": day0["abnormal"]}
    dominant = max(parts, key=lambda k: abs(parts[k]))
    if model == "market":
        notes.append("Modello a un fattore: l'effetto settore è incluso nella componente specifica.")

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
    """Applica l'event study a ogni giorno tra start ed end.

    Con only_anomalous=True (default) restituisce solo i giorni anomali: i candidati eventi.
    Con False restituisce tutti i giorni, utile per scegliere un giorno tranquillo come
    controllo negativo (il sistema non deve inventare una causa dove non c'è nulla da spiegare).
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
