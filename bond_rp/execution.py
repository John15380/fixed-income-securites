"""Cost-aware signal adjustment for nonnegative equal-basket allocations."""

import numpy as np
import pandas as pd


def cost_aware_forecasts(bonds, predictions, *, horizon=1, cost_bps=.5):
    """Optimize on expected net excess return; deduct actual fees once in P&L.

    For a one-NAV long position, net excess return is
      (1-c)*x - 2*c*(1+rf).
    Its conditional variance is approximately (1-c)^2*Var(x). The transformed
    mean below therefore gives the optimal weight when passed to the existing
    gross-variance mean/variance allocator. This adjustment itself is not P&L.

    Annual rf is known at entry. Monthly funding uses the known one-year yield
    as an explicit proxy, never next month's realized French RF. Cash-rate
    uncertainty/covariance is not estimated by this simple sizing approximation.
    """
    if horizon not in (1, 12) or not 0 <= cost_bps < 10000:
        raise ValueError("Invalid horizon or proportional cost")
    out = predictions.copy()
    dates = pd.PeriodIndex(out.ym, freq="M")
    c = cost_bps/10000
    rf = (np.expm1(bonds.loc[dates, "y1"].to_numpy()/1200) if horizon == 1 else
          bonds.loc[dates, "rf_simple"].to_numpy()/100)
    cols = [f"mu{n}" for n in range(2, 6)]
    mu = out[cols].to_numpy()/100
    out[cols] = 100*((1-c)*mu-2*c*(1+rf[:, None]))/(1-c)**2
    return out
