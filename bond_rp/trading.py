"""Alternative trading designs using the same curve, macro, trend and sentiment inputs.

Monthly returns reprice the *same bond* at n-1/12 years using Svensson curves.
No overlapping annual labels are compounded as monthly investment returns.
"""

from pathlib import Path
import re
import zipfile
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LinearRegression, Ridge

from .config import DATA_DIR, TRAIN_START
from .data import load_gsw_bond_data
from .models import _numeric_pipeline, available_features
from .robustness import technical_features

WINDOWS = (None, 120, 240)
RF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"


def load_cash_returns(path=None):
    """RF for return month t+1, stored at its purchase month t. Percent units.

    Use only RF, never the equity factors. It is a realized bill/cash benchmark,
    not a future input into the forecast or its trading signal.
    """
    path = Path(path or Path(DATA_DIR)/"F-F_Research_Data_Factors_CSV.zip")
    with zipfile.ZipFile(path) as archive:
        member = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        lines = archive.read(member).decode("utf-8-sig").splitlines()
    values = {}
    for line in lines:
        if re.match(r"^\s*\d{6},", line):
            parts = line.strip().split(",")
            date = pd.Period(pd.to_datetime(parts[0], format="%Y%m"), "M")
            values[date-1] = float(parts[-1])
    result = pd.Series(values, name="rf_month").sort_index()
    if not len(result) or not (np.diff(result.index.asi8) == 1).all():
        raise ValueError("RF file must contain a contiguous monthly section")
    return result


def svensson_yields(parameters, maturities):
    """Continuously compounded zero yields in percent, maturities in years."""
    p = np.asarray(parameters, dtype=float)
    n = np.asarray(maturities, dtype=float).reshape(1, -1)
    if p.ndim != 2 or p.shape[1] != 6 or (n <= 0).any():
        raise ValueError("Expected BETA0..3, TAU1..2 and positive maturities")
    # Early GSW observations are Nelson-Siegel: BETA3=0 and TAU2=-999.99.
    # The inactive second curvature term is exactly zero, not missing history.
    inactive = p[:, 3] == 0
    if (p[:, 4] <= 0).any() or ((p[:, 5] <= 0) & ~inactive).any():
        raise ValueError("Active Svensson decay constants must be positive")
    a, b = n/p[:, [4]], n/np.where(inactive, 1., p[:, 5])[:, None]
    f, g = -np.expm1(-a)/a, -np.expm1(-b)/b
    return p[:, [0]]+p[:, [1]]*f+p[:, [2]]*(f-np.exp(-a))+p[:, [3]]*(g-np.exp(-b))


def monthly_curve_returns(parameters):
    """Calendar-month repricing, cash from fitted one-month zero yield.

    GSW short-end yields are model extrapolations, not observed bill quotes.
    rf_y1 is retained as an alternative funding proxy for sensitivity checks.
    """
    index = pd.period_range(parameters.index.min(), parameters.index.max(), freq="M")
    parameters = parameters.reindex(index)
    p = parameters.to_numpy(dtype=float)
    maturity = np.arange(2, 6, dtype=float)
    current_y = svensson_yields(p, maturity)
    aged_y = svensson_yields(p, maturity-1/12)
    lp = -current_y*maturity
    aged_lp = -aged_y*(maturity-1/12)
    sale = pd.DataFrame(aged_lp, index=index).reindex(index+1).to_numpy()
    monthly = 100*np.expm1((sale-lp)/100)
    carry = 100*np.expm1((aged_lp-lp)/100)
    rf = 100*np.expm1(svensson_yields(p, [1/12]).ravel()/1200)
    frame = pd.DataFrame({"rf_month": rf,
                          "rf_y1": 100*np.expm1(svensson_yields(p, [1]).ravel()/1200)}, index=index)
    for i, n in enumerate(range(2, 6)):
        frame[f"mr{n}"] = monthly[:, i]
        frame[f"mx{n}"] = monthly[:, i]-rf
        frame[f"carry{n}"] = carry[:, i]-rf
    frame["mx_bar"] = frame[[f"mx{n}" for n in range(2, 6)]].mean(axis=1, skipna=False)
    frame.index.name = "ym"
    return frame


def load_monthly_market():
    raw = pd.read_csv(Path(DATA_DIR)/"GSW_yields.csv", skiprows=9, parse_dates=["Date"])
    cols = [f"BETA{i}" for i in range(4)]+["TAU1", "TAU2"]
    raw[cols] = raw[cols].apply(pd.to_numeric, errors="coerce")
    raw = raw.dropna(subset=cols).sort_values("Date")
    raw["ym"] = raw.Date.dt.to_period("M")
    monthly = raw.groupby("ym").tail(1).set_index("ym")
    check = svensson_yields(monthly[cols], np.arange(1, 6))
    quoted = monthly[[f"SVENY0{n}" for n in range(1, 6)]].to_numpy()
    # Published SVENY quotes are rounded to four decimal percentage points.
    np.testing.assert_allclose(check, quoted, atol=5.1e-5, rtol=0)
    bonds = load_gsw_bond_data().set_index("ym")
    market = monthly_curve_returns(monthly[cols])
    market["rf_nss"] = market.rf_month
    market["rf_month"] = load_cash_returns().reindex(market.index)
    # Trading targets and realized cash use bills, avoiding short-end extrapolation.
    for n in range(2, 6):
        market[f"mx{n}"] = market[f"mr{n}"]-market.rf_month
    market["mx_bar"] = market[[f"mx{n}" for n in range(2, 6)]].mean(axis=1, skipna=False)
    return bonds.join(market)


def forecast_designs(bonds, macro, sentiment=None, *, horizon=1, start="1985-01", end=None,
                     vintage_archive=None, lag=2, verbose=True):
    """Expanding/10y/20y multi-output fits. All labels end strictly before t.

    OLS/Ridge100 and 50% shrink-to-EH are disclosed, fixed research choices.
    Forecast individual maturities as well as their average. No OOS tuning.
    """
    if horizon not in {1, 12}:
        raise ValueError("Only monthly and annual targets are implemented")
    target_cols = [f"{'mx' if horizon == 1 else 'sx'}{n}" for n in range(2, 6)]
    static = available_features(bonds, macro, sentiment, publication_lag=lag)
    static = static.join(technical_features(bonds))
    if end is None:
        end = bonds.dropna(subset=target_cols).index.max()
    records, audit = [], []
    for date in static.loc[start:end].index:
        features = static
        vintage = None
        if vintage_archive is not None:
            vintage, panel = vintage_archive.at(date)
            features = available_features(bonds, panel, publication_lag=lag).join(technical_features(bonds))
        mc = [c for c in features if c.startswith("macro::")]
        market = [f"f{n}" for n in range(1, 6)]+["fs_avg", "TI_avg"]
        if sentiment is not None:
            market += ["SENT", "SENT_SQ"]
        if not features.loc[date, market].notna().all() or not features.loc[date, mc].notna().any():
            continue
        data = features.join(bonds[target_cols])
        mask = (data.index >= TRAIN_START) & (data.index+horizon < date)
        eligible = data[market+target_cols].notna().all(axis=1) & data[mc].notna().any(axis=1)
        for window in WINDOWS:
            chosen = mask & eligible
            if window is not None:
                chosen &= data.index >= date-horizon-window
            train = data.loc[chosen]
            if len(train) < 120:
                continue
            label = "exp" if window is None else f"w{window}"
            y = train[target_cols].to_numpy()
            macros = _numeric_pipeline(pca=True)
            Z, z = macros.fit_transform(train[mc]), None
            z = macros.transform(features.loc[[date], mc])
            block, now = {"LN": Z}, {"LN": z}
            groups = {"FB": ["fs_avg"], "CP": [f"f{n}" for n in range(1, 6)], "TI": ["TI_avg"]}
            if sentiment is not None:
                groups["BW"] = ["SENT", "SENT_SQ"]
            for name, cols in groups.items():
                pipe = _numeric_pipeline()
                block[name] = pipe.fit_transform(train[cols])
                now[name] = pipe.transform(features.loc[[date], cols])
            specs = {name: [name] for name in block}
            specs["CPLN"] = ["CP", "LN"]
            specs["CurveTrend"] = ["CP", "TI"]
            if sentiment is not None:
                specs["LNBW"] = ["LN", "BW"]
            specs["ALL"] = ["CP", "LN", "TI"]+(["BW"] if sentiment is not None else [])
            values = {"EH": y.mean(axis=0)}
            for name, components in specs.items():
                X, x = (np.hstack([source[c] for c in components]) for source in (block, now))
                for alpha in (0, 100):
                    estimator = Ridge(alpha=alpha) if alpha else LinearRegression()
                    name_out = name+("_R100" if alpha else "")
                    values[name_out] = estimator.fit(X, y).predict(x).ravel()
            for suffix in ("", "_R100"):
                components = [c+suffix for c in (["CP", "LN", "BW", "TI"] if sentiment is not None else ["CP", "LN", "TI"])]
                values["EW"+suffix] = np.mean([values[c] for c in components], axis=0)
            for name, prediction in values.items():
                records.append({"ym": str(date), "model": f"{name}/{label}",
                                **{f"mu{n}": prediction[i] for i, n in enumerate(range(2, 6))}})
            audit.append({"ym": str(date), "window": label, "horizon": horizon, "train_n": len(train),
                          "last_label_end": str(train.index.max()+horizon),
                          "latest_macro_reference": str(date-lag),
                          "vintage": str(vintage) if vintage else "revised"})
        if verbose and date.month == 1 and date.year % 10 == 0:
            print(f"  horizon={horizon}, through {date}", flush=True)
    return pd.DataFrame(records), pd.DataFrame(audit)


def monthly_performance(returns, risk_free):
    r, rf = np.asarray(returns), np.asarray(risk_free)
    if len(r) < 24 or len(r) != len(rf) or not np.isfinite(r).all() or (r <= -1).any():
        raise ValueError("Invalid monthly portfolio path")
    wealth = np.r_[1., np.cumprod(1+r)]
    excess = r-rf
    return {"months": len(r), "cagr_pct": 100*(wealth[-1]**(12/len(r))-1),
            "sharpe": np.sqrt(12)*excess.mean()/excess.std(ddof=1) if excess.std(ddof=1) > 1e-12 else np.nan,
            "vol_pct": 100*np.std(r, ddof=1)*np.sqrt(12),
            "max_dd_pct": 100*np.min(wealth/np.maximum.accumulate(wealth)-1),
            "mean_excess_pct": 1200*excess.mean(), "terminal_wealth": wealth[-1]}


def combine_windows(bonds, forecasts, min_errors=36, lookback=60):
    """Avoid selecting the ten-year window from its evaluation-period ranking.

    Mix the six EW OLS/Ridge x expanding/10y/20y experts. Adaptive weights or a
    selected expert use only last 60 purchase months with s+1<t, after 36 errors.
    Score the same 50% shrinkage used in the primary trading experiment.
    """
    candidates = [f"EW{suffix}/{window}" for suffix in ("", "_R100") for window in ("exp", "w120", "w240")]
    p = forecasts.copy()
    p["ym"] = pd.PeriodIndex(p.ym, freq="M")
    cols = [f"mu{n}" for n in range(2, 6)]
    indexed = {m: g.set_index("ym")[cols].sort_index() for m, g in p.groupby("model")}
    dates = indexed[candidates[0]].index
    experts = np.stack([indexed[c].loc[dates].to_numpy() for c in candidates], axis=1)
    baselines = np.stack([indexed["EH/"+c.split("/")[-1]].loc[dates].to_numpy() for c in candidates], axis=1)
    used = (baselines+.5*(experts-baselines)).mean(axis=2)
    actual = bonds.mx_bar.reindex(dates).to_numpy()
    records, audits = [], []
    for i, date in enumerate(dates):
        history = (dates+1 < date) & (dates >= date-1-lookback) & np.isfinite(actual)
        if history.sum() < min_errors:
            weights = np.full(6, 1/6)
            selection = weights
        else:
            mse = np.mean((used[history]-actual[history, None])**2, axis=0)
            weights = 1/np.maximum(mse, 1e-10)
            weights /= weights.sum()
            selection = np.eye(6)[np.argmin(mse)]
        for label, w in [("mix", np.full(6, 1/6)), ("online", weights), ("select", selection)]:
            for name, block in [("EW", experts), ("EH", baselines)]:
                pred = w@block[i]
                records.append({"ym": str(date), "model": f"{name}/{label}",
                                **{c: pred[j] for j, c in enumerate(cols)}})
        audits.append({"ym": str(date), "matured_errors": int(history.sum()),
                       "last_error_end": str(dates[history].max()+1) if history.any() else "",
                       "selected": candidates[int(np.argmax(selection))] if history.sum() >= min_errors else "equal_mix",
                       **{f"weight::{c}": weights[j] for j, c in enumerate(candidates)}})
    return pd.concat([forecasts, pd.DataFrame(records)], ignore_index=True), pd.DataFrame(audits)


def monthly_backtest(bonds, forecasts, *, start="1990-01", end=None, policy="long_only",
                     cap=1., gamma=5., risk_window=60, shrink=.5, cost_bps=.5,
                     funding="rf_month", borrow_spread_bps=50., delay=0):
    """Self-financing monthly rebalancing; complete entry/exit costs on each leg.

    shrink=.5 blends each model with the same-window EH. Risk history uses only
    monthly outcomes ending before the decision month. delay=1 executes the
    previous month's prediction, isolating the idealized same-close assumption.
    """
    forecasts = forecasts.copy()
    forecasts["ym"] = pd.PeriodIndex(forecasts.ym, freq="M")
    pred = {m: g.set_index("ym")[[f"mu{n}" for n in range(2, 6)]]/100 for m, g in forecasts.groupby("model")}
    if end is None:
        # Delayed execution must not quietly add a new return month at the end.
        end = min(p.index.max() for p in pred.values())
    common = bonds.dropna(subset=[f"mr{n}" for n in range(2, 6)]+[funding]).loc[start:end].index
    for p in pred.values():
        common = common.intersection(p.index+delay)
    if not len(common) or not (np.diff(common.asi8) == 1).all():
        raise ValueError("Monthly evaluation must have a complete common calendar")
    variances, covariances, cutoffs = [], [], []
    for date in common:
        past = bonds.loc[(bonds.index+1 < date) & (bonds.index >= date-1-risk_window),
                         [f"mx{n}" for n in range(2, 6)]].dropna()/100
        if len(past) < 36:
            raise ValueError("Insufficient matured monthly risk history")
        variances.append(max(past.mean(axis=1).var(ddof=1), 1e-10))
        covariances.append(LedoitWolf().fit(past).covariance_)
        cutoffs.append(str(past.index.max()+1))
    variance, cov = np.array(variances), np.array(covariances)
    inverse = np.linalg.inv(cov+np.eye(4)*1e-10)
    duration = np.arange(2, 6, dtype=float)
    iduration = inverse@duration
    projected = inverse-np.einsum("ti,tj->tij", iduration, iduration)/(iduration@duration)[:, None, None]
    decisions = common-delay
    actual = bonds.loc[common, [f"mr{n}" for n in range(2, 6)]].to_numpy(dtype=float)/100
    rf = bonds.loc[common, funding].to_numpy()/100
    carry = bonds.loc[decisions, [f"carry{n}" for n in range(2, 6)]].to_numpy(dtype=float)/100
    technical = technical_features(bonds).loc[decisions]
    trend12 = (bonds.loc[decisions, "TI_avg"].to_numpy() > 0).astype(float)
    trend_vote = (technical > 0).mean(axis=1).to_numpy()
    rows = []
    names = [*pred, "BuyHold", "Cash", "Carry", "Trend12", "TrendVote", "CarryTrend"]
    for name in names:
        if name in pred:
            mu = pred[name].loc[decisions].to_numpy()
            reference = pred["EH/"+name.split("/")[-1]].loc[decisions].to_numpy()
            mu = reference+shrink*(mu-reference)
        else:
            mu = carry if name == "Carry" else np.zeros((len(common), 4))
        if name == "BuyHold":
            q = np.full_like(actual, .25)
        elif name == "Cash":
            q = np.zeros_like(actual)
        elif name.startswith("Trend") or name == "CarryTrend":
            vote = trend_vote if name == "TrendVote" else trend12
            if name == "CarryTrend":
                vote = vote*(carry.mean(axis=1) > 0)
            q = np.repeat((vote/4)[:, None], 4, axis=1)
        elif policy in {"long_only", "mv", "vol_target"}:
            if policy == "vol_target":
                w = np.where(mu.mean(axis=1) > 0, np.minimum(cap, .04/np.sqrt(12*variance)), 0.)
            else:
                w = np.clip(mu.mean(axis=1)/(gamma*variance), 0 if policy == "long_only" else -cap, cap)
            q = np.repeat((w/4)[:, None], 4, axis=1)
        elif policy == "diagonal":
            q = np.maximum(mu, 0)/(gamma*np.maximum(np.diagonal(cov, axis1=1, axis2=2), 1e-10))
            q *= np.minimum(1., cap/np.maximum(q.sum(axis=1), 1e-15))[:, None]
        elif policy == "duration_neutral":
            q = np.einsum("tij,tj->ti", projected, mu)/gamma
            q *= np.minimum(1., cap/np.maximum(np.abs(q).sum(axis=1), 1e-15))[:, None]
        else:
            raise ValueError(policy)
        # A fully invested long-only book must reserve cash for its entry fees.
        # Otherwise "100% assets" plus transaction fees borrows a small amount.
        total = q.sum(axis=1)
        cash_only = (q >= -1e-15).all(axis=1) & (total <= 1+1e-12)
        reserve = np.minimum(1., 1/np.maximum(total*(1+cost_bps/10000), 1e-15))
        q *= np.where(cash_only, reserve, 1.)[:, None]
        exposure = np.abs(q).sum(axis=1)
        gross = rf+(q*(actual-rf[:, None])).sum(axis=1)
        cost = cost_bps/10000*(exposure*(1+rf)+(np.abs(q)*(1+actual)).sum(axis=1))
        financing = np.maximum(q.sum(axis=1)-1, 0)*borrow_spread_bps/120000
        rows.append(pd.DataFrame({"model": name, "purchase": common.astype(str), "sale": (common+1).astype(str),
                     "return_net": gross-cost-financing, "return_gross": gross, "risk_free": rf,
                     "cost": cost, "financing": financing, "gross_exposure": exposure,
                     "cash_after_entry": 1-q.sum(axis=1)-cost_bps/10000*exposure,
                     "risk_last_label_end": cutoffs,
                     **{f"q{n}": q[:, i] for i, n in enumerate(range(2, 6))}}))
    trades = pd.concat(rows, ignore_index=True)
    summaries = []
    for name, group in trades.groupby("model", sort=False):
        stat = monthly_performance(group.return_net, group.risk_free)
        stat.update(model=name, start=str(common.min()), end=str(common.max()+1),
                    mean_gross_exposure=group.gross_exposure.mean(), policy=policy,
                    cap=cap, shrink=shrink, risk_window=risk_window, cost_bps=cost_bps, funding=funding, delay=delay)
        summaries.append(stat)
    return pd.DataFrame(summaries), trades
