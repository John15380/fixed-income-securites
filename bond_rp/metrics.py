"""Forecast diagnostics and annual, non-overlapping portfolio statistics."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from .config import NW_LAGS


def nw_ols(y, X, lags=NW_LAGS):
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": min(lags, len(y)-1),
                                                     "use_correction": True})


def oos_r2(actual, pred, bench):
    denominator = np.mean((np.asarray(actual)-np.asarray(bench))**2)
    return 1 - np.mean((np.asarray(actual)-np.asarray(pred))**2)/denominator if denominator > 0 else np.nan


def hit_rate(actual, pred):
    return np.mean(np.sign(actual) == np.sign(pred))


def loss_test(actual, pred, bench, *, nested=False, lags=NW_LAGS):
    """One-sided HAC loss comparison; CW adjustment only for nested OLS models.

    Return statistic, asymptotic p-value. No automatic significance claims or
    star annotations; model search and finite samples remain limitations.
    """
    actual, pred, bench = map(np.asarray, (actual, pred, bench))
    difference = (actual-bench)**2 - (actual-pred)**2
    if nested:
        difference += (bench-pred)**2
    if np.allclose(difference, 0):
        return 0.0, 1.0
    fit = nw_ols(difference, np.ones((len(difference), 1)), lags)
    statistic = float(fit.tvalues[0])
    return statistic, float(norm.sf(statistic))


def clark_west(actual, pred, bench, lags=NW_LAGS):
    return loss_test(actual, pred, bench, nested=True, lags=lags)


def annual_performance(returns, risk_free, gamma=5):
    """Simple annual returns; include initial wealth in endpoint drawdowns."""
    r, rf = np.asarray(returns, dtype=float), np.asarray(risk_free, dtype=float)
    if len(r) < 2 or len(r) != len(rf) or not np.isfinite(r).all():
        raise ValueError("At least two finite annual outcomes are required")
    if (r <= -1).any():
        raise ValueError("Portfolio insolvency: an annual return is <= -100%")
    wealth = np.r_[1.0, np.cumprod(1+r)]
    drawdown = wealth/np.maximum.accumulate(wealth)-1
    excess = r-rf
    volatility = np.std(excess, ddof=1)
    return {"years": len(r), "cagr_pct": 100*(wealth[-1]**(1/len(r))-1),
            "annual_mean_pct": 100*r.mean(), "annual_vol_pct": 100*np.std(r, ddof=1),
            "sharpe": excess.mean()/volatility if volatility > 0 else np.nan,
            "max_dd_pct": 100*drawdown.min(), "terminal_wealth": wealth[-1],
            "cer_pct": 100*(r.mean()-gamma*np.var(r, ddof=1)/2)}
