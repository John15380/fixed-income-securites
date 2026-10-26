"""A long-only, fully funded bond/cash ledger with complete monthly entry/exit fees."""
import numpy as np
import pandas as pd


def performance(returns, risk_free):
    r, rf = np.asarray(returns, float), np.asarray(risk_free, float)
    if len(r) < 12 or len(r) != len(rf) or not np.isfinite([r,rf]).all() or (r <= -1).any():
        raise ValueError("Invalid portfolio account")
    wealth = np.r_[1., np.cumprod(1+r)]
    excess = r-rf
    return dict(months=len(r), cagr_pct=100*(wealth[-1]**(12/len(r))-1),
                sharpe=np.sqrt(12)*excess.mean()/excess.std(ddof=1) if excess.std(ddof=1)>1e-12 else np.nan,
                vol_pct=100*r.std(ddof=1)*np.sqrt(12),
                max_dd_pct=100*np.min(wealth/np.maximum.accumulate(wealth)-1),
                mean_excess_pct=1200*excess.mean(), terminal_wealth=wealth[-1])


def simulate(market, forecasts, *, start="2000-01", end="2026-01", gamma=5., cap=1.,
             shrink=.5, risk_window=60, cost_bps=.5, delay=0):
    if not (gamma>0 and 0<cap<=1 and 0<=shrink<=1 and cost_bps>=0 and delay>=0):
        raise ValueError("Invalid fully funded strategy parameters")
    forecasts = forecasts.copy()
    if set(forecasts.horizon) != {1} or forecasts.window.nunique() != 1:
        raise ValueError("One monthly forecast window per account")
    forecasts["ym"] = pd.PeriodIndex(forecasts.ym, freq="M")
    pred = {m:g.set_index("ym")[[f"mu{n}" for n in range(2,6)]]/100 for m,g in forecasts.groupby("model")}
    dates = market.dropna(subset=[f"mr{n}" for n in range(2,6)]+["rf_month"]).loc[start:end].index
    for p in pred.values():
        dates = dates.intersection(p.index+delay)
    if not len(dates) or not (np.diff(dates.asi8)==1).all():
        raise ValueError("Account needs a contiguous common monthly calendar")
    variances, cutoffs = [], []
    for t in dates:
        past = market.loc[(market.index+1<t)&(market.index>=t-1-risk_window), [f"mx{n}" for n in range(2,6)]].dropna()/100
        if len(past)<36:
            raise ValueError("Insufficient risk history")
        variances.append(max(past.mean(axis=1).var(ddof=1),1e-10))
        cutoffs.append(str(past.index.max()+1))
    variance = np.array(variances)
    decision = dates-delay
    actual = market.loc[dates, [f"mr{n}" for n in range(2,6)]].to_numpy()/100
    rf = market.rf_month.reindex(dates).to_numpy()/100
    c = cost_bps/10000
    records = []
    for name in [*pred, "BuyHold", "Cash"]:
        if name == "BuyHold":
            w = np.ones(len(dates))
        elif name == "Cash":
            w = np.zeros(len(dates))
        else:
            mu = pred["EH"].loc[decision]+shrink*(pred[name].loc[decision]-pred["EH"].loc[decision])
            w = np.clip(mu.mean(axis=1).to_numpy()/(gamma*variance),0,cap)
        # q is the actual fraction of opening wealth spent on each maturity.
        # Reserve entry fees in cash; after one month liquidate every bond.
        w = w*np.minimum(1.,1/np.maximum(w*(1+c),1e-15))
        q = np.repeat((w/4)[:,None],4,axis=1)
        cash = 1-w-c*w
        ending = cash*(1+rf)+(q*(1+actual)*(1-c)).sum(axis=1)
        gross = rf+(q*(actual-rf[:,None])).sum(axis=1)
        net = ending-1
        if (cash < -1e-12).any():
            raise AssertionError("Unfunded entry")
        records.append(pd.DataFrame(dict(model=name,purchase=dates.astype(str),sale=(dates+1).astype(str),
                         decision=decision.astype(str),return_net=net,return_gross=gross,risk_free=rf,
                         cost=gross-net,gross_exposure=w,cash_after_entry=cash,risk_last_label_end=cutoffs,
                         **{f"q{n}":q[:,n-2] for n in range(2,6)})))
    trades = pd.concat(records,ignore_index=True)
    return summarize(trades), trades


def summarize(trades):
    return pd.DataFrame([dict(model=name,start=g.purchase.min(),last_purchase=g.purchase.max(),
                             last_sale=g.sale.max(),mean_exposure=g.gross_exposure.mean(),
                             **performance(g.return_net,g.risk_free)) for name,g in trades.groupby("model",sort=False)])


def annual_accounts(trades, gamma=5):
    """Disjoint 12-month account blocks, with compounded strategy and cash paths."""
    rows, stats = [], []
    for name,g in trades.groupby("model"):
        g = g.sort_values("purchase")
        n = len(g)//12
        r = np.prod(1+g.return_net.to_numpy()[:12*n].reshape(n,12),axis=1)-1
        rf = np.prod(1+g.risk_free.to_numpy()[:12*n].reshape(n,12),axis=1)-1
        rows.extend(dict(model=name,year=i,return_net=a,risk_free=b) for i,(a,b) in enumerate(zip(r,rf)))
        stats.append(dict(model=name,years=n,cer_pct=100*(r.mean()-gamma*r.var(ddof=1)/2),
                          annual_excess_sharpe=(r-rf).mean()/(r-rf).std(ddof=1) if np.std(r-rf)>1e-12 else np.nan))
    return pd.DataFrame(stats),pd.DataFrame(rows)
