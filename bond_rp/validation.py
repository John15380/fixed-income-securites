"""Executable research-choice audits: selection, information timing and R2 accounting.

All comparisons are retrospective. A causal selection rule can be verified, but
cannot turn previously inspected history into a genuinely untouched holdout.
"""
from itertools import product
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from .portfolio import performance, risk_information, basket_account, simulate
from .forecasting import target_columns
from .inference import joint_block_bootstrap_t
from .vintages import VintageArchive
from .config import DATA_DIR


def r2_components(actual,prediction,benchmark):
    """Same dates and target: isolate the benchmark denominator exactly."""
    y,p,b=map(lambda a:np.asarray(a,float),(actual,prediction,benchmark))
    if y.ndim!=1 or not (y.shape==p.shape==b.shape) or not np.isfinite([y,p,b]).all():
        raise ValueError("Expected aligned finite vectors")
    tss=np.sum((y-y.mean())**2)
    sse=np.sum((y-p)**2)
    base=np.sum((y-b)**2)
    if min(tss,base)<=0:
        raise ValueError("R2 denominators must be positive")
    calibrated=LinearRegression().fit(p[:,None],y).predict(p[:,None])
    calibration_sse=np.sum((y-calibrated)**2)
    if calibration_sse>sse+1e-8*max(1.,sse):
        raise AssertionError("Full-sample calibration cannot have higher SSE")
    return dict(sse_forecast=sse,sse_historical_mean=base,test_centered_tss=tss,
                eh_denominator_ratio=base/tss,r2_oos=1-sse/base,r2_centered=1-sse/tss,
                calibrated_signal_r2=1-calibration_sse/tss,
                denominator_effect=(1-sse/base)-(1-sse/tss))


def r2_accounting(market,forecasts,source,shrink=.5):
    rows=[]
    p=forecasts.loc[forecasts.window.astype(str).eq("120")&forecasts.ym.ge("2000-01")].copy()
    p["ym"]=pd.PeriodIndex(p.ym,freq="M")
    for h,g in p.groupby("horizon"):
        base=g.loc[g.model.eq("EH")].set_index("ym")[[f"mu{n}" for n in range(2,6)]]
        dates=base.index.intersection(market.dropna(subset=target_columns(h)).index)
        y=market.loc[dates,target_columns(h)].mean(axis=1).to_numpy()
        b=base.loc[dates].mean(axis=1).to_numpy()
        for name,a in g.groupby("model"):
            raw=a.set_index("ym")[[f"mu{n}" for n in range(2,6)]].loc[dates].mean(axis=1).to_numpy()
            for k in (1.,shrink):
                f=b+k*(raw-b)
                rows.append(dict(source=source,horizon=h,model=name,shrink=k,months=len(dates),
                                 start=str(dates.min()),last_purchase=str(dates.max()),
                                 **r2_components(y,f,b)))
    return pd.DataFrame(rows)


def choose_policy(returns,dates,cutoff,*,lookback=None,utility_gamma=5.):
    """Rank net monthly CER using only accounts with sale strictly before cutoff."""
    dates=pd.PeriodIndex(dates,freq="M")
    cutoff=pd.Period(cutoff,"M")
    known=dates+1<cutoff
    if lookback is not None:
        known&=dates>=cutoff-1-lookback
    if known.sum()<60:
        raise ValueError("Need at least 60 matured validation accounts")
    past=np.asarray(returns)[known]
    scores=12*(past.mean(axis=0)-utility_gamma*past.var(axis=0,ddof=1)/2)
    winner=int(np.argmax(scores))  # Stable tie order puts the base design first.
    return winner,dict(validation_months=int(known.sum()),last_validation_sale=str(dates[known].max()+1),
                       criterion_cer_pct=100*scores[winner])


def parameter_selection(market,forecasts,out,replications=1999):
    p=forecasts.loc[forecasts.horizon.eq(1)].copy()
    p["ym"]=pd.PeriodIndex(p.ym,freq="M")
    p["window"]=p.window.astype(str)
    dates=pd.period_range("2000-01","2026-01",freq="M")
    variance,_=risk_information(market,dates)
    realized=market.loc[dates,[f"mr{n}" for n in range(2,6)]].mean(axis=1).to_numpy()/100
    rf=market.rf_month.reindex(dates).to_numpy()/100
    series={(w,m):g.set_index("ym")[[f"mu{n}" for n in range(2,6)]].mean(axis=1).reindex(dates).to_numpy()/100
            for (w,m),g in p.groupby(["window","model"])}
    base=("120",12,5.,.5,1.)
    designs=[base]+[d for d in product(("120","240","exp"),(3,6,9,12,18,24),
                                     (2.,5.,10.),(0.,.5,1.),(.5,1.)) if d!=base]
    mu=np.column_stack([series[w,"EH"]+s*(series[w,f"EW3_MA{k}"]-series[w,"EH"]) for w,k,ga,s,c in designs])
    gamma=np.array([d[2] for d in designs]);caps=np.array([d[4] for d in designs])
    weights=np.clip(mu/(variance[:,None]*gamma),0,caps)
    ledger=basket_account(weights,realized[:,None],rf[:,None])
    returns=ledger["return_net"]
    names=[f"cfg{i:03d}" for i in range(len(designs))]
    settings=pd.DataFrame([dict(config=n,window=w,ma=k,gamma=g,shrink=s,cap=c)
                           for n,(w,k,g,s,c) in zip(names,designs)])
    rows=[]
    for label,mask in [("full",np.ones(len(dates),bool)),("validation",dates+1<pd.Period("2013-01","M")),
                       ("later",dates>=pd.Period("2013-01","M"))]:
        for j,n in enumerate(names):
            r=returns[mask,j]
            rows.append(dict(config=n,period=label,**performance(r,rf[mask]),
                             monthly_cer_pct=1200*(r.mean()-2.5*r.var(ddof=1))))
    grid=pd.DataFrame(rows).merge(settings,on="config",validate="many_to_one")
    schedules=[];selected={"validation_frozen":[],"yearly_walkforward":[]}
    frozen,metadata=choose_policy(returns,dates,"2013-01")
    yearly=frozen;yearly_metadata=metadata
    test=dates>=pd.Period("2013-01","M")
    for t in dates[test]:
        if t.month==1:
            yearly,yearly_metadata=choose_policy(returns,dates,t,lookback=120)
        for rule,j,info,selected_at in [("validation_frozen",frozen,metadata,"2013-01"),
                                       ("yearly_walkforward",yearly,yearly_metadata,f"{t.year}-01")]:
            selected[rule].append(j)
            schedules.append(dict(rule=rule,purchase=str(t),selected_at=selected_at,config=names[j],**info))
    test_returns=returns[test]
    summaries=[]
    for rule,indices in selected.items():
        r=test_returns[np.arange(test.sum()),indices]
        summaries.append(dict(rule=rule,start=str(dates[test].min()),last_purchase=str(dates[test].max()),
                              **performance(r,rf[test])))
    summaries.append(dict(rule="fixed_base",start=str(dates[test].min()),last_purchase=str(dates[test].max()),
                          **performance(test_returns[:,0],rf[test])))
    eh=basket_account(np.clip(series["120","EH"]/(5*variance),0,1),realized,rf)["return_net"]
    family=[]
    for period,mask in [("full",np.ones(len(dates),bool)),("later",test)]:
        tests=joint_block_bootstrap_t(returns[mask]-eh[mask,None],lags=6,block=12,replications=replications)
        for j,n in enumerate(names):
            family.append(dict(config=n,period=period,hac_t=tests["statistic"][j],p=tests["bootstrap_p"][j],
                               family_p=tests["max_t_p"][j],family_configs=len(designs),replications=replications))
    out.mkdir(exist_ok=True,parents=True)
    pd.DataFrame(returns,index=dates.astype(str),columns=names).rename_axis("purchase").to_csv(out/"parameter_accounts.csv")
    return grid,pd.DataFrame(schedules).merge(settings,on="config",validate="many_to_one"),pd.DataFrame(summaries),pd.DataFrame(family)


def factor_selection(market,forecasts,cutoff="2013-01"):
    """Freeze a factor-family choice using only accounts matured before cutoff."""
    candidates=["CP","LN6","Macro8","Trend12","CP_LN6","CP_Macro8",
                "CP_Trend12","Macro8_Trend12","EW3"]
    core=forecasts.loc[forecasts.horizon.eq(1)&forecasts.window.astype(str).eq("120")&
                       forecasts.model.isin(["EH",*candidates])]
    _,trades=simulate(market,core,start="2000-01",end="2026-01")
    paths=trades.pivot(index="purchase",columns="model",values="return_net")
    dates=pd.PeriodIndex(paths.index,freq="M")
    rf=trades.loc[trades.model.eq("EH")].set_index("purchase").loc[paths.index,"risk_free"].to_numpy()
    winner,info=choose_policy(paths[candidates].to_numpy(),dates,cutoff)
    rows=[]
    for period,mask in [("validation",dates+1<pd.Period(cutoff,"M")),
                        ("later",dates>=pd.Period(cutoff,"M"))]:
        for name in candidates:
            rows.append(dict(model=name,period=period,selected_before_cutoff=name==candidates[winner],
                             selection_cutoff=cutoff,**info,
                             **performance(paths.loc[mask,name],rf[mask])))
    return pd.DataFrame(rows)


def macro_coverage(start="2000-01",end="2025-12"):
    rows=[]
    with VintageArchive(DATA_DIR) as archive:
        for t in pd.period_range(start,end,freq="M"):
            vintage,panel=archive.at(t)
            counts={lag:int(panel.reindex([t-lag]).notna().sum(axis=1).iloc[0]) for lag in (1,2,3)}
            observed=panel.notna().any(axis=1)
            rows.append(dict(purchase=str(t),vintage=str(vintage),last_reference=str(panel.index[observed].max()),
                             series=len(panel.columns),reference_lag1_available=counts[1],
                             reference_lag2_available=counts[2],reference_lag3_available=counts[3]))
    return pd.DataFrame(rows)


def assumption_checks(market,macro,archival,revised,out,loader):
    """Change information timing only in labeled, matched-calendar diagnostics."""
    variants={"archival_strict":(archival,1),"revised_lag2":(revised,1)}
    settings=[("archival_lag3",1,3,1),("current_vintage_lag2",0,2,1),
              ("current_vintage_lag1",0,1,1),("same_close_labels",1,2,0),
              ("same_close_all",0,1,0)]
    for name,delay,lag,embargo in settings:
        with VintageArchive(DATA_DIR,vintage_delay=delay) as archive:
            p,_=loader(out/name,market,macro,archive,start="1999-12",end="2025-12",horizons=(1,),
                       windows=(120,),lookbacks=(12,),lag=lag,embargo=embargo)
        variants[name]=(p,embargo)
    rows=[]
    for name,(p,embargo) in variants.items():
        keep=p.loc[p.horizon.eq(1)&p.window.astype(str).eq("120")&p.model.isin(["EH","EW3"])]
        stats,_=simulate(market,keep,start="2000-01",end="2025-12",risk_embargo=embargo)
        stats["case"]=name
        rows.append(stats)
    core=archival.loc[archival.horizon.eq(1)&archival.window.astype(str).eq("120")&archival.model.isin(["EH","EW3"])]
    for name,kw in [("zero_cost",dict(cost_bps=0)),("no_shrink",dict(shrink=1)),("gamma10",dict(gamma=10)),
                    ("cap2_spread50",dict(cap=2)),("cap3_spread50",dict(cap=3)),
                    ("cap3_spread200",dict(cap=3,borrow_spread_bps=200)),("execution_delay1",dict(delay=1))]:
        stat,_=simulate(market,core,start="2000-01",end="2025-12",**kw)
        stat["case"]=name
        rows.append(stat)
    return pd.concat(rows,ignore_index=True)


def financing_grid(market,forecasts):
    """Disclosed long/short and borrowing grid; no row silently replaces the base."""
    core=forecasts.loc[forecasts.horizon.eq(1)&forecasts.window.astype(str).eq("120")&
                       forecasts.model.isin(["EH","EW3"])]
    rows=[]
    for gamma,cap,borrow in product((2.,5.,10.),(1.,2.,3.),(0.,50.,200.)):
        settings=[("long_only",0.,0.)]
        settings += [("long_short",-cap,short) for short in (0.,25.,100.)]
        for direction,floor,short in settings:
            stat,trades=simulate(market,core,start="2000-01",end="2026-01",gamma=gamma,cap=cap,
                                 floor=floor,borrow_spread_bps=borrow,short_borrow_bps=short)
            row=stat.loc[stat.model.eq("EW3")].iloc[0].to_dict()
            ledger=trades.loc[trades.model.eq("EW3")]
            row.update(direction=direction,gamma=gamma,lower_bound=floor,upper_bound=cap,
                       borrowing_spread_bps=borrow,short_borrow_bps=short,
                       borrowing_months=int((ledger.cash_after_entry<0).sum()),
                       total_financing_cost_pct=100*ledger.financing.sum(),
                       total_short_fees_pct=100*ledger.short_fee.sum())
            rows.append(row)
    grid=pd.DataFrame(rows)
    grid["rank_cagr"]=grid.cagr_pct.rank(method="min",ascending=False).astype(int)
    grid["rank_sharpe"]=grid.sharpe.rank(method="min",ascending=False).astype(int)
    grid["rank_drawdown"]=grid.max_dd_pct.rank(method="min",ascending=False).astype(int)
    chosen=[]
    def add(label,row,selection):
        item=row.to_dict();item.update(label=label,selection_basis=selection);chosen.append(item)
    base=grid.loc[(grid.direction=="long_only")&(grid.gamma==5)&(grid.upper_bound==1)&
                  (grid.borrowing_spread_bps==50)].iloc[0]
    moderate=grid.loc[(grid.direction=="long_short")&(grid.gamma==5)&(grid.upper_bound==2)&
                      (grid.borrowing_spread_bps==50)&(grid.short_borrow_bps==25)].iloc[0]
    aggressive=grid.loc[(grid.direction=="long_short")&(grid.gamma==5)&(grid.upper_bound==3)&
                        (grid.borrowing_spread_bps==50)&(grid.short_borrow_bps==25)].iloc[0]
    add("unlevered_long_only",base,"fixed reference")
    add("financed_long_short_cap2",moderate,"fixed economic comparison")
    add("financed_long_short_cap3",aggressive,"fixed economic comparison")
    cost_grid=grid.loc[(grid.borrowing_spread_bps==50)&(
        ((grid.direction=="long_only")&(grid.short_borrow_bps==0))|
        ((grid.direction=="long_short")&(grid.short_borrow_bps==25)))]
    add("cost_grid_best_cagr",cost_grid.sort_values(
        ["cagr_pct","sharpe"],ascending=False).iloc[0],
        "ex post over 18-policy cost-controlled grid")
    add("cost_grid_best_sharpe",cost_grid.sort_values(
        ["sharpe","cagr_pct"],ascending=False).iloc[0],
        "ex post over 18-policy cost-controlled grid")
    add("grid_best_cagr",grid.sort_values(["cagr_pct","sharpe"],ascending=False).iloc[0],"ex post over disclosed grid")
    add("grid_best_sharpe",grid.sort_values(["sharpe","cagr_pct"],ascending=False).iloc[0],"ex post over disclosed grid")
    return grid,pd.DataFrame(chosen)


def run_validation(market,macro,archival,horizons,revised,out,loader):
    path=out/"validation"
    path.mkdir(exist_ok=True)
    grid,schedule,selection,family=parameter_selection(market,archival,path)
    grid.to_csv(out/"parameter_grid.csv",index=False)
    schedule.to_csv(out/"selection_schedule.csv",index=False)
    selection.to_csv(out/"selection_performance.csv",index=False)
    family.to_csv(out/"parameter_family_tests.csv",index=False)
    factor_selection(market,archival).to_csv(out/"factor_validation.csv",index=False)
    finance,finance_summary=financing_grid(market,archival)
    finance.to_csv(out/"financing_grid.csv",index=False)
    finance_summary.to_csv(out/"financing_summary.csv",index=False)
    macro_coverage().to_csv(out/"macro_information.csv",index=False)
    assumption_checks(market,macro,archival,revised,path,loader).to_csv(out/"assumption_performance.csv",index=False)
    pd.concat([r2_accounting(market,pd.concat([archival,horizons]),"archival"),
               r2_accounting(market,revised,"revised")],ignore_index=True).to_csv(out/"r2_accounting.csv",index=False)
