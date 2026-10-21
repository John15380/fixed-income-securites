"""Statistical evaluation metrics: OOS R², Clark-West test, hit rate."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm as scipy_norm

from .config import NW_LAGS


def nw_ols(y, X, lags: int = NW_LAGS):
    """OLS with Newey-West HAC standard errors."""
    return sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})


def oos_r2(actual: np.ndarray, pred: np.ndarray, bench: np.ndarray) -> float:
    """Campbell-Thompson (2008) OOS R²."""
    mse_m = np.mean((actual - pred) ** 2)
    mse_b = np.mean((actual - bench) ** 2)
    return 1.0 - mse_m / mse_b


def clark_west(actual: np.ndarray, pred: np.ndarray, bench: np.ndarray):
    """
    Clark-West (2007) MSFE-adjusted statistic for nested model comparison.
    Returns (cw_stat, p_value) — one-sided test, large stat favours the model.
    """
    d    = (actual - bench) ** 2 - ((actual - pred) ** 2 - (bench - pred) ** 2)
    dbar = d.mean()
    se   = d.std(ddof=1) / np.sqrt(len(d))
    cw   = dbar / se
    pval = 1.0 - scipy_norm.cdf(cw)
    return cw, pval


def hit_rate(actual: np.ndarray, pred: np.ndarray) -> float:
    """Fraction of periods where predicted and realised signs agree."""
    return np.mean(np.sign(actual) == np.sign(pred))
