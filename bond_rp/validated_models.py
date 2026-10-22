"""Literature-based predictors and validation-only nonlinear model selection.

This is a retrospective, bounded experiment. No author full-sample factors enter
these forecasts; every supervised transformation is rebuilt on matured labels.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from .config import DATA_DIR
from .coursework import common_index
from .data import apply_transform
from .models import _numeric_pipeline
from .trading import load_monthly_market, svensson_yields

REFERENCE = ("EH", "FB", "CP05", "LN6", "CP15", "CP_EMA3", "CP_EMA6", "CP_EMA12", "Full_OLS", "PCA5_OLS", "FullPCA_OLS")
LEARNERS = ("EN_real", "EN_GC", "XGB_real", "XGB_GC")
PRIMARY = "XGB_GC_cal"


def inflation_trend(levels, *, lag=2, decay=.987, window=120):
    """CP15 Eq.19: trailing 120-month DMA of year-over-year log CPI (%).

    Original finite-sum weights (1-v)*v**j are not silently normalized.
    Lag is in calendar months; incomplete windows remain unavailable.
    """
    s = levels.reindex(pd.period_range(levels.index.min(), levels.index.max(), freq="M"))
    annual = 100*(np.log(s)-np.log(s.shift(12)))
    weights = (1-decay)*decay**np.arange(window-1, -1, -1)
    result = annual.rolling(window).apply(lambda x: np.dot(x, weights), raw=True)
    result.index += lag
    return result


def load_long_market():
    """Keep the existing tradable 2-5y panel; add 1-20y zero curves for CP15.

    CP15's original duration-standardized 2-20y average constructs the index.
    Curve source here is GSW (an explicit adaptation from the paper's CMT data).
    """
    bonds = load_monthly_market()
    raw = pd.read_csv(Path(DATA_DIR)/"GSW_yields.csv", skiprows=9, parse_dates=["Date"])
    cols = [f"BETA{i}" for i in range(4)]+["TAU1", "TAU2"]
    raw[cols] = raw[cols].apply(pd.to_numeric, errors="coerce")
    raw = raw.dropna(subset=cols).sort_values("Date")
    raw["ym"] = raw.Date.dt.to_period("M")
    p = raw.groupby("ym").tail(1).set_index("ym")[cols].reindex(bonds.index)
    maturity = np.arange(1, 21)
    zero = pd.DataFrame(svensson_yields(p, maturity), index=bonds.index)
    lp = -zero.to_numpy()*maturity
    extra = {f"long_y{n}": zero.iloc[:, n-1] for n in maturity}
    aged = pd.DataFrame(-svensson_yields(p, maturity[1:]-1/12)*(maturity[1:]-1/12), index=bonds.index)
    for j, n in enumerate(maturity[1:]):
        extra[f"long_rx{n}"] = pd.Series(lp[:, n-2], index=bonds.index).reindex(bonds.index+12).to_numpy()-lp[:, n-1]-zero.iloc[:, 0]
        sale = aged.iloc[:, j].reindex(bonds.index+1).to_numpy()
        extra[f"long_mx{n}"] = 100*np.expm1((sale-lp[:, n-1])/100)-bonds.rf_month
    return bonds.join(pd.DataFrame(extra, index=bonds.index))


def fred_snapshot(archive, date):
    """Transform macro inputs AND extract CPI from exactly the same t-1 file."""
    vintage, macro = archive.at(date)
    z, member = archive.members[vintage]
    with z.open(member) as source:
        raw = pd.read_csv(source).iloc[1:]
    raw.index = pd.PeriodIndex(pd.to_datetime(raw.sasdate, format="mixed"), freq="M")
    level = pd.to_numeric(raw.CPIAUCSL, errors="coerce")
    return vintage, macro, level


def make_features(bonds, macro, cpi):
    f = bonds[[f"f{n}" for n in range(1, 6)]+[f"fs{n}" for n in range(2, 6)]+[f"long_y{n}" for n in range(1, 21)]].copy()
    lagged = macro.copy()
    lagged.index += 2
    lagged.columns = ["macro::"+str(c) for c in macro]
    f = f.join(lagged)
    f["tau"] = inflation_trend(cpi).reindex(f.index)
    for half in (3, 6, 12):
        for n in range(1, 6):
            f[f"ema{half}_f{n}"] = bonds[f"f{n}"].ewm(halflife=half, adjust=False, min_periods=24).mean()
    return f


def cp15_cycles(train, current, history):
    """CP15 Eqs.20/21: project each yield on trend inflation, using known history."""
    columns = [f"long_y{n}" for n in range(1, 21)]
    trend = LinearRegression().fit(history[["tau"]], history[columns])
    cycles = train[columns].to_numpy()-trend.predict(train[["tau"]])
    now = current[columns].to_numpy()-trend.predict(current[["tau"]])
    return (np.column_stack([cycles[:, 0], cycles[:, 1:].mean(axis=1)]),
            np.column_stack([now[:, 0], now[:, 1:].mean(axis=1)]))


def prepare_fold(bonds, features, date, horizon, *, predict_dates=None, train_start="1994-01"):
    """All fitted objects use only date-known history or strictly matured labels.

    A multi-date forecast batch freezes the fitted pipeline at its first date;
    subsequent predictor values are applied as they become known. Main test
    forecasts use single-date batches and monthly refits.
    """
    date = pd.Period(date, "M")
    dates = pd.PeriodIndex(predict_dates if predict_dates is not None else [date], freq="M")
    if dates.min() < date:
        raise ValueError("Prediction batch precedes fitting date")
    prefix = "rx" if horizon == 12 else "mx"
    targets = [f"{prefix}{n}" for n in range(2, 6)]
    long_targets = [f"long_{prefix}{n}" for n in range(2, 21)]
    mc = [c for c in features if c.startswith("macro::")]
    ycols = [f"long_y{n}" for n in range(1, 21)]
    history = features.loc[train_start:str(date)]
    history = history[history.tau.notna() & history[mc].notna().any(axis=1)]
    mask = (history.index+horizon < date)
    train = history.loc[mask]
    valid = bonds.loc[train.index, targets+long_targets].notna().all(axis=1)
    train = train.loc[valid]
    if len(train) < 120:
        raise ValueError(f"Insufficient matured history at {date}")
    now = features.loc[dates]
    if not now.tau.notna().all():
        raise ValueError("Missing contemporaneously available inflation trend")
    y = bonds.loc[train.index, targets].to_numpy()
    cp = [f"f{n}" for n in range(1, 6)]
    pca = _numeric_pipeline(pca=True).fit(history[mc])
    Z, z = pca.transform(train[mc]), pca.transform(now[mc])
    ln = lambda a: np.column_stack([a[:, 0], a[:, 0]**3, a[:, 1:4], a[:, 7]])
    reference = {"EH": np.tile(y.mean(axis=0), (len(now), 1))}
    indices, current = [], []
    for name, X, x in [("CP05", train[cp].to_numpy(), now[cp].to_numpy()), ("LN6", ln(Z), ln(z))]:
        pred, first, _ = common_index(X, y, x, second_intercept=name != "CP05")
        reference[name] = pred
        indices.append(first.predict(X)); current.append(first.predict(x))
    # CP15 Eqs.20/21/25/27: detrend each yield, then one-year cycle + average
    # 2-20y cycle predicts a duration-standardized 2-20y return index.
    X, x = cp15_cycles(train, now, history)
    y_long = bonds.loc[train.index, long_targets].to_numpy()
    weighted = (y_long/np.arange(2, 21)).mean(axis=1)
    first = LinearRegression().fit(X, weighted)
    signal, signal_now = first.predict(X), first.predict(x)
    reference["CP15"] = LinearRegression().fit(signal[:, None], y).predict(signal_now[:, None])
    indices.append(signal); current.append(signal_now)
    for half in (3, 6, 12):
        cols = [f"ema{half}_f{n}" for n in range(1, 6)]
        pred, first, _ = common_index(train[cols], y, now[cols], second_intercept=False)
        reference[f"CP_EMA{half}"] = pred
        indices.append(first.predict(train[cols])); current.append(first.predict(now[cols]))
    reference["FB"] = np.column_stack([LinearRegression().fit(train[[f"fs{n}"]], y[:, n-2]).predict(now[[f"fs{n}"]]) for n in range(2, 6)])
    base, bnow = np.column_stack(indices), np.column_stack(current)
    reference["Full_OLS"] = LinearRegression().fit(base, y).predict(bnow)
    reference["PCA5_OLS"] = LinearRegression().fit(Z[:, :5], y).predict(z[:, :5])
    full, fnow = np.column_stack([base, Z[:, :5]]), np.column_stack([bnow, z[:, :5]])
    reference["FullPCA_OLS"] = LinearRegression().fit(full, y).predict(fnow)
    scaler = StandardScaler().fit(full)
    audit = dict(ym=str(date), horizon=horizon, n_train=len(train), train_start=str(train.index.min()),
                 last_label_end=str(train.index.max()+horizon), macro_reference=str(date-2),
                 history_end=str(history.index.max()), n_features=full.shape[1], n_macro=len(mc))
    return dict(X=scaler.transform(full), x=scaler.transform(fnow), y=y, reference=reference,
                dates=dates, audit=audit)


def gaussian_augment(X, y, *, seed=42, ratio=1.):
    """Empirical-marginal Gaussian copula; training rows only, incl. their labels.

    Artificial rows are a distributional regularizer, not independent evidence.
    No random train/test split and no synthetic observations in evaluation.
    """
    values = np.column_stack([X, y])
    if not np.isfinite(values).all() or len(values) < 10:
        raise ValueError("Copula requires finite training observations")
    ranks = np.column_stack([rankdata(values[:, j], method="average") for j in range(values.shape[1])])
    latent = ndtri((ranks-.5)/len(values))
    # Copula marginals must be standard normal before applying Phi. Normalize
    # the rank-score covariance, including finite-sample and tied-rank effects.
    covariance = np.cov(latent, rowvar=False)
    scale = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    covariance /= np.outer(scale, scale)
    np.fill_diagonal(covariance, 1.)
    covariance = covariance*(1-1e-8)+np.eye(values.shape[1])*1e-8
    draws = np.random.default_rng(seed).multivariate_normal(np.zeros(values.shape[1]), covariance,
                                                           size=int(len(values)*ratio))
    probability = ndtr(draws)
    synthetic = np.column_stack([np.quantile(values[:, j], probability[:, j]) for j in range(values.shape[1])])
    return np.vstack([X, synthetic[:, :X.shape[1]]]), np.vstack([y, synthetic[:, X.shape[1]:]])


def learner_predict(fold, name, params):
    X, y, x = fold["X"], fold["y"], fold["x"]
    # Standardizing targets makes EN's penalty comparable across maturities.
    location, scale = y.mean(axis=0), np.maximum(y.std(axis=0), 1e-8)
    normalized = (y-location)/scale
    if name.endswith("_GC"):
        X, normalized = gaussian_augment(X, normalized)
    if name.startswith("XGB"):
        from xgboost import XGBRegressor
        make = lambda: XGBRegressor(objective="reg:squarederror", tree_method="hist", n_jobs=1,
                                    random_state=42, verbosity=0, **params)
    else:
        make = lambda: ElasticNet(max_iter=100000, tol=1e-5, **params)
    result = np.column_stack([make().fit(X, normalized[:, j]).predict(x) for j in range(4)])
    return result*scale+location


def calibration(actual, prediction, benchmark):
    """One frozen validation-only shrinkage coefficient, bounded to [0,1]."""
    delta = np.asarray(prediction)-np.asarray(benchmark)
    residual = np.asarray(actual)-np.asarray(benchmark)
    denominator = np.sum(delta**2)
    return float(np.clip(np.sum(delta*residual)/denominator, 0, 1)) if denominator > 0 else 0.
