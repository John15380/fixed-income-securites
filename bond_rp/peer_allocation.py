"""Per-maturity annual-signal allocation following reference slides 16--19.

Annual conditional means and annual variances set the allocation. Actual wealth
is marked using one-month prices of the same aging bonds, without refitting a
one-month return model or compounding overlapping annual outcome labels.
The slides leave the precise three-parameter brake unspecified; this module's
explicit convex exposure penalty and causal drawdown budget are assumptions.
"""
import numpy as np
import pandas as pd
from .trading import monthly_performance


def diagonal_allocation(mu, variance, budget=1., concentration=0., gamma=5.):
    """Solve max mu'w - .5 sum((gamma*v+lambda*mean(v))*w²), w>=0,sum<=B."""
    mu, variance = np.asarray(mu), np.asarray(variance)
    d = gamma*np.maximum(variance, 1e-10)+concentration*np.mean(variance)
    w = np.maximum(mu, 0)/d
    if w.sum() > budget:
        lo, hi = 0., max(float(mu.max()), 0.)
        for _ in range(60):
            eta = (lo+hi)/2
            w = np.maximum(mu-eta, 0)/d
            if w.sum() > budget:
                lo = eta
            else:
                hi = eta
        w = np.maximum(mu-hi, 0)/d
    return w


def risk_panel(bonds, dates):
    """Same 60 strictly mature annual returns for mean and maturity variances."""
    mean, var, cutoffs = [], [], []
    for date in dates:
        past = bonds.loc[bonds.index+12 < date, [f"rx{n}" for n in range(2, 6)]].dropna().tail(60)/100
        if len(past) != 60:
            raise ValueError("A full 60-observation matured annual window is required")
        mean.append(past.mean().to_numpy()); var.append(past.var(ddof=1).to_numpy())
        cutoffs.append(str(past.index.max()+12))
    return np.asarray(mean), np.asarray(var), cutoffs


def allocation_path(bonds, forecasts, *, budget=1., drawdown=0., concentration=0., gamma=5., cost_bps=0.):
    f = forecasts.copy()
    f.index = pd.PeriodIndex(f.index, freq="M")
    dates = f.index
    if dates.has_duplicates or not (np.diff(dates.asi8) == 1).all():
        raise ValueError("The allocation account needs a complete monthly calendar")
    if not 0 <= budget <= 1 or min(drawdown, concentration, cost_bps) < 0:
        raise ValueError("Invalid bounded long-only parameters")
    _, variance, cutoff = risk_panel(bonds, dates)
    actual = bonds.loc[dates, [f"mr{n}" for n in range(2, 6)]].to_numpy()/100
    rf = bonds.rf_month.reindex(dates).to_numpy()/100
    outcome = bonds.loc[dates, [f"rx{n}" for n in range(2, 6)]].to_numpy()/100
    wealth, peak, c, records = 1., 1., cost_bps/10000, []
    for i, date in enumerate(dates):
        prior_dd = wealth/peak-1
        effective = budget*np.exp(drawdown*prior_dd)
        q = diagonal_allocation(f.iloc[i].to_numpy()/100, variance[i], effective,
                                concentration, gamma)
        q *= min(1., 1/((1+c)*max(q.sum(), 1e-14)))
        cash = 1-q.sum()*(1+c)
        sale = np.sum(q*(1+actual[i]))
        net = cash*(1+rf[i])+sale*(1-c)-1
        gross = rf[i]+q@(actual[i]-rf[i])
        cost = c*q.sum()*(1+rf[i])+c*sale
        np.testing.assert_allclose(net, gross-cost, atol=1e-14)
        wealth *= 1+net; peak = max(peak, wealth)
        records.append(dict(purchase=str(date), sale=str(date+1), return_net=net, return_gross=gross,
            risk_free=rf[i], cost=cost, cash=cash, exposure=q.sum(), wealth=wealth,
            previous_drawdown=prior_dd, effective_budget=effective, risk_last_label_end=cutoff[i],
            annual_label_payoff=float(q@outcome[i]), **{f"q{n}": q[j] for j, n in enumerate(range(2, 6))}))
    return pd.DataFrame(records)


def summarize(path):
    out = monthly_performance(path.return_net, path.risk_free)
    annual = path.annual_label_payoff.to_numpy()
    out.update(cumulative_total_pct=100*(np.prod(1+path.return_net)-1),
               cumulative_cash_excess_sum_pct=100*np.sum(path.return_net-path.risk_free),
               mean_exposure=path.exposure.mean(),
               overlapping_annual_payoff_mean_pct=100*annual.mean(),
               overlapping_payoff_sum_div12_pct=100*annual.sum()/12,
               overlapping_annual_payoff_sharpe=float(annual.mean()/annual.std(ddof=1)))
    return out
