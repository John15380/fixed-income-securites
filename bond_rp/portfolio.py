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


def risk_information(market, dates, risk_window=60, embargo=1):
    if embargo < 0:
        raise ValueError("Risk embargo must be nonnegative")
    variances, cutoffs = [], []
    for t in dates:
        past = market.loc[(market.index+1<=t-embargo)&(market.index>=t-embargo-risk_window),
                          [f"mx{n}" for n in range(2,6)]].dropna()/100
        if len(past)<36:
            raise ValueError("Insufficient risk history")
        variances.append(max(past.mean(axis=1).var(ddof=1),1e-10))
        cutoffs.append(str(past.index.max()+1))
    return np.array(variances), cutoffs


def basket_account(weights, basket_returns, risk_free, cost_bps=.5, borrow_spread_bps=50.,
                   short_borrow_bps=25.):
    """Common ledger for long-only or long-short policies.

    A fully invested, unlevered long position reserves its entry fee. Negative
    cash pays the realized bill rate plus the stated spread. Short sale proceeds
    remain in cash and the absolute short notional pays an explicit borrow fee.
    These are financing sensitivities, not observed security-level repo quotes.
    """
    w=np.array(weights,dtype=float,copy=True)
    if not np.isfinite(w).all() or min(cost_bps,borrow_spread_bps,short_borrow_bps)<0:
        raise ValueError("Invalid ledger inputs")
    c=cost_bps/10000
    # Only 0<w<=1 needs rescaling to reserve its entry fee without borrowing.
    w*=np.where((w>0)&(w<=1),np.minimum(1.,1/np.maximum(w*(1+c),1e-15)),1.)
    cash=1-w-c*np.abs(w)
    financing=np.maximum(-cash,0)*borrow_spread_bps/120000
    short_fee=np.maximum(-w,0)*short_borrow_bps/120000
    net=(cash*(1+risk_free)+w*(1+basket_returns)
         -c*np.abs(w)*(1+basket_returns)-financing-short_fee-1)
    gross=risk_free+w*(basket_returns-risk_free)
    return dict(return_net=net,return_gross=gross,cost=gross-net-financing-short_fee,
                financing=financing,short_fee=short_fee,gross_exposure=w,
                absolute_exposure=np.abs(w),cash_after_entry=cash)


def multi_asset_account(weights, asset_returns, risk_free, cost_bps=.5,
                        borrow_spread_bps=50., short_borrow_bps=25.):
    """Monthly entry/exit ledger for maturity-specific long or short weights."""
    q=np.asarray(weights,float).copy();r=np.asarray(asset_returns,float);rf=np.asarray(risk_free,float)
    if (q.ndim!=2 or q.shape!=r.shape or len(q)!=len(rf) or
            not all(np.isfinite(x).all() for x in (q,r,rf)) or
            min(cost_bps,borrow_spread_bps,short_borrow_bps)<0):
        raise ValueError("Invalid multi-asset ledger inputs")
    c=cost_bps/10000
    total=q.sum(axis=1);absolute=np.abs(q).sum(axis=1)
    # A nonlevered all-long position reserves its entry fee from initial cash.
    funded=(q.min(axis=1)>=0)&(total>0)&(total<=1)
    scale=np.ones(len(q))
    scale[funded]=np.minimum(1.,1/np.maximum(total[funded]+c*absolute[funded],1e-15))
    q*=scale[:,None]
    total=q.sum(axis=1);absolute=np.abs(q).sum(axis=1)
    cash=1-total-c*absolute
    financing=np.maximum(-cash,0)*borrow_spread_bps/120000
    short_fee=np.maximum(-q,0).sum(axis=1)*short_borrow_bps/120000
    net=(cash*(1+rf)+(q*(1+r)).sum(axis=1)
         -c*(np.abs(q)*(1+r)).sum(axis=1)-financing-short_fee-1)
    gross=rf+(q*(r-rf[:,None])).sum(axis=1)
    return dict(return_net=net,return_gross=gross,
                cost=gross-net-financing-short_fee,financing=financing,
                short_fee=short_fee,gross_exposure=total,
                absolute_exposure=absolute,cash_after_entry=cash,weights=q)


def simulate(market, forecasts, *, start="2000-01", end="2026-01", gamma=5., cap=1., floor=0.,
             shrink=.5, risk_window=60, cost_bps=.5, delay=0, risk_embargo=1,
             borrow_spread_bps=50., short_borrow_bps=25.):
    if not (gamma>0 and 0<cap<=3 and -cap<=floor<=0 and 0<=shrink<=1 and
            min(cost_bps,borrow_spread_bps,short_borrow_bps)>=0 and delay>=0):
        raise ValueError("Invalid strategy parameters")
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
    variance,cutoffs = risk_information(market,dates,risk_window,risk_embargo)
    decision = dates-delay
    actual = market.loc[dates, [f"mr{n}" for n in range(2,6)]].to_numpy()/100
    rf = market.rf_month.reindex(dates).to_numpy()/100
    records = []
    for name in [*pred, "BuyHold", "Cash"]:
        if name == "BuyHold":
            w = np.ones(len(dates))
        elif name == "Cash":
            w = np.zeros(len(dates))
        else:
            mu = pred["EH"].loc[decision]+shrink*(pred[name].loc[decision]-pred["EH"].loc[decision])
            w = np.clip(mu.mean(axis=1).to_numpy()/(gamma*variance),floor,cap)
        account=basket_account(w,actual.mean(axis=1),rf,cost_bps,borrow_spread_bps,short_borrow_bps)
        w=account["gross_exposure"]
        q = np.repeat((w/4)[:,None],4,axis=1)
        if floor>=0 and cap<=1 and (account["cash_after_entry"] < -1e-12).any():
            raise AssertionError("Unfunded entry")
        records.append(pd.DataFrame(dict(model=name,purchase=dates.astype(str),sale=(dates+1).astype(str),
                         decision=decision.astype(str),risk_free=rf,**account,risk_last_label_end=cutoffs,
                         **{f"q{n}":q[:,n-2] for n in range(2,6)})))
    trades = pd.concat(records,ignore_index=True)
    return summarize(trades), trades


def summarize(trades):
    return pd.DataFrame([dict(model=name,start=g.purchase.min(),last_purchase=g.purchase.max(),
                             last_sale=g.sale.max(),mean_exposure=g.gross_exposure.mean(),
                             mean_absolute_exposure=g.absolute_exposure.mean(),
                             short_months=int((g.gross_exposure<0).sum()),
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
