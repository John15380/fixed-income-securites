"""Common-calendar forecast tests, economic value and disclosed sensitivity checks."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from .forecasting import target_columns, features_at, fit_forecasts
from .models import _numeric_pipeline
from .factors import ln6_terms
from .inference import mean_hac_t, joint_block_bootstrap_t
from .portfolio import simulate, summarize, annual_accounts

PRIMARY = ("EH", "FB", "CP", "LN6", "Macro8", "Trend12", "CP_Macro8", "EW3")


def forecast_scores(market, forecasts, shrink=.5):
    rows = []
    p = forecasts.copy()
    p["ym"] = pd.PeriodIndex(p.ym, freq="M")
    for (h,window),g in p.groupby(["horizon","window"]):
        panels = {name:a.set_index("ym")[[f"mu{n}" for n in range(2,6)]] for name,a in g.groupby("model")}
        index = panels["EH"].index.intersection(market.dropna(subset=target_columns(h)).index)
        actual4 = market.loc[index,target_columns(h)].to_numpy()
        base4 = panels["EH"].loc[index].to_numpy()
        for name,a in panels.items():
            pred4 = base4+shrink*(a.loc[index].to_numpy()-base4)
            for j,label in [(None,"basket"), *[(n-2,str(n)) for n in range(2,6)]]:
                y,b,f = [(v.mean(axis=1) if j is None else v[:,j]) for v in (actual4,base4,pred4)]
                mse0,mse = np.mean((y-b)**2),np.mean((y-f)**2)
                loss = (y-b)**2-(y-f)**2
                t = float(mean_hac_t(loss,lags=max(6,18 if h==12 else h)))
                # CW applies to raw nested OLS vs its constant model, not a
                # shrunk ensemble. Only the basket CP/Macro8/LN6/Trend is OLS.
                cw = np.nan
                if shrink==1 and j is None and name in ("CP","Macro8","LN6","Trend12"):
                    cw = float(mean_hac_t(loss+(b-f)**2,lags=max(6,18 if h==12 else h)))
                rows.append(dict(horizon=h,window=window,model=name,maturity=label,months=len(index),
                                 start=str(index.min()),last_purchase=str(index.max()),shrink=shrink,
                                 mse=mse,r2_oos=1-mse/mse0,hac_loss_t=t,hac_loss_p=float(norm.sf(t)),
                                 cw_t=cw,cw_p=float(norm.sf(cw)) if np.isfinite(cw) else np.nan))
    return pd.DataFrame(rows)


def in_sample_scores(market, macro, start="1985-01", end="2026-01"):
    """Descriptive full-sample fits only; never consumed by any OOS fit."""
    rows = []
    features = features_at(market,macro,lookbacks=(12,))
    mc = [c for c in features if c.startswith("macro::")]
    for h in (1,3,12):
        targets = target_columns(h)
        data = features.join(market[targets]).loc[start:end].dropna(subset=targets+["trend12"])
        values,_ = fit_forecasts(data,data,mc,targets,(12,))
        y = data[targets].mean(axis=1).to_numpy()
        tss = ((y-y.mean())**2).sum()
        for name in PRIMARY:
            f = np.asarray(values[name])
            r2 = 1-((y-f.mean(axis=1))**2).sum()/tss
            k={"EH":1,"CP":6,"LN6":7,"Macro8":9,"Trend12":2}.get(name)
            rows.append(dict(horizon=h,model=name,months=len(data),r2_is=r2,
                             adjusted_r2_is=1-(1-r2)*(len(data)-1)/(len(data)-k) if k else np.nan,
                             start=str(data.index.min()),last_purchase=str(data.index.max())))
    return pd.DataFrame(rows)


def in_sample_coefficients(market,macro,start="1985-01",end="2026-01"):
    """HAC conditional inference for descriptive regressions, not trading inputs.

    CP/Macro8/LN6 are first-stage basket regressions. These are not two-stage
    GMM standard errors, and condition on estimated PCs and preprocessing.
    """
    rows=[]
    features=features_at(market,macro,lookbacks=(12,))
    mc=[c for c in features if c.startswith("macro::")]
    for h in (1,3,12):
        targets=target_columns(h)
        d=features.join(market[targets]).loc[start:end].dropna(subset=targets+["trend12"])
        Z=_numeric_pipeline(pca=True).fit_transform(d[mc])
        blocks={"CP":d[[f"f{n}" for n in range(1,6)]],
                "Macro8":pd.DataFrame(Z,index=d.index,columns=[f"F{k}" for k in range(1,9)]),
                "LN6":pd.DataFrame(ln6_terms(Z),index=d.index,columns=["F1","F1_cube","F2","F3","F4","F8"]),
                "Trend12":d[["trend12"]],"EH":pd.DataFrame(index=d.index)}
        for name,X in blocks.items():
            fit=sm.OLS(d[targets].mean(axis=1),sm.add_constant(X,has_constant="add")).fit(
                cov_type="HAC",cov_kwds={"maxlags":18 if h==12 else 6,"use_correction":True})
            for term in fit.params.index:
                rows.append(dict(horizon=h,model=name,target="basket",term=term,coefficient=fit.params[term],
                                 hac_se=fit.bse[term],hac_t=fit.tvalues[term],p_two_sided=fit.pvalues[term]))
        for n in range(2,6):
            fit=sm.OLS(d[targets[n-2]],sm.add_constant(d[[f"fs{n}"]])).fit(
                cov_type="HAC",cov_kwds={"maxlags":18 if h==12 else 6,"use_correction":True})
            for term in fit.params.index:
                rows.append(dict(horizon=h,model="FB",target=str(n),term=term,coefficient=fit.params[term],
                                 hac_se=fit.bse[term],hac_t=fit.tvalues[term],p_two_sided=fit.pvalues[term]))
    return pd.DataFrame(rows)


def sensitivity(market, forecasts, start, end):
    rows = []
    core = forecasts.loc[(forecasts.horizon==1)&(forecasts.window.astype(str)=="120")]
    # One dimension at a time; no row is promoted to the main strategy.
    cases = [("base",{})]
    for name, key, values in [("cost","cost_bps",(0,2,5)),("gamma","gamma",(2,10)),
                              ("cap","cap",(.5,.75)),("risk","risk_window",(36,120)),
                              ("shrink","shrink",(0,1)),("delay","delay",(1,))]:
        cases += [(f"{name}={v}",{key:v}) for v in values]
    for label,kw in cases:
        keep = core[core.model.isin(["EH","EW3"])]
        stat,_ = simulate(market,keep,start=start,end=end,**kw)
        stat["case"] = label
        rows.append(stat)
    for window in ("240","exp"):
        keep = forecasts.loc[(forecasts.horizon==1)&(forecasts.window.astype(str)==window)&forecasts.model.isin(["EH","EW3"])]
        if len(keep):
            stat,_ = simulate(market,keep,start=start,end=end)
            stat["case"] = "training="+window
            rows.append(stat)
    for k in (3,6,9,12,18,24):
        name=f"EW3_MA{k}"
        stat,_ = simulate(market,core[core.model.isin(["EH",name])],start=start,end=end)
        stat["model"] = stat.model.replace({name:"EW3"})
        stat["case"] = f"MA={k}"
        rows.append(stat)
    return pd.concat(rows,ignore_index=True)


def economic_inference(trades, replications=1999):
    annual, accounts = annual_accounts(trades)
    paths = trades.pivot(index="purchase",columns="model",values="return_net")
    names = [name for name in PRIMARY if name not in ("EH",) and name in paths]
    rows=[]
    for reference in ("EH","BuyHold"):
        dif = paths[names].subtract(paths[reference],axis=0)
        tests = joint_block_bootstrap_t(dif.to_numpy(),lags=6,block=12,replications=replications)
        for j,name in enumerate(names):
            rows.append(dict(model=name,reference=reference,hac_active_t=tests["statistic"][j],
                             active_p=tests["bootstrap_p"][j],family_p=tests["max_t_p"][j],
                             family_models=len(names),replications=replications,block_months=12))
    # Paired circular-block percentile CI for annual CER difference; disjoint
    # annual observations, blocks of 3 years preserve some serial dependence.
    ap = accounts.pivot(index="year",columns="model",values="return_net")
    rng=np.random.default_rng(42)
    n=len(ap)
    starts=rng.integers(0,n,(replications,int(np.ceil(n/3))))
    ix=((starts[:,:,None]+np.arange(3))%n).reshape(replications,-1)[:,:n]
    intervals=[]
    for reference in ("EH","BuyHold"):
        a,b=ap.EW3.to_numpy()[ix],ap[reference].to_numpy()[ix]
        d=10000*((a.mean(1)-2.5*a.var(1,ddof=1))-(b.mean(1)-2.5*b.var(1,ddof=1)))
        value=100*(annual.set_index("model").loc["EW3","cer_pct"]-annual.set_index("model").loc[reference,"cer_pct"])
        intervals.append(dict(model="EW3",reference=reference,years=n,cer_gain_bps=value,
                              ci_low_bps=np.quantile(d,.025),ci_high_bps=np.quantile(d,.975)))
    return annual,accounts,pd.DataFrame(rows),pd.DataFrame(intervals)


def subperiods(trades):
    rows=[]
    for label,start,end in [("2000-2012","2000-01","2012-12"),("2013-2019","2013-01","2019-12"),
                            ("2020-2026","2020-01","2026-01"),("2013-2026","2013-01","2026-01")]:
        part=trades.loc[trades.purchase.between(start,end)]
        if len(part):
            s=summarize(part)
            s["period"]=label
            rows.append(s)
    return pd.concat(rows,ignore_index=True)
