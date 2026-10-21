"""Retrospective specification audit; every forecast and selector is causal.

The grids are disclosed research choices, not a newly untouched holdout.
"""

from itertools import combinations
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline

from .config import MIN_TRAIN_OBS
from .models import (ForecastResult, _numeric_pipeline, available_features,
                     matured_training, model_pipeline)

LOOKBACKS = (3, 6, 9, 12, 18, 24)
FAMILIES = ("CP", "LN", "BW", "TI")
RIDGES = (0, 10, 100)


def technical_features(bonds):
    """Price minus trailing calendar-month mean, including the current month."""
    prices = bonds[[f"lp{n}" for n in range(2, 6)]]
    return pd.DataFrame({f"MA{k}": (prices-prices.rolling(k, min_periods=k).mean())
                         .mean(axis=1, skipna=False) for k in LOOKBACKS})


def family_forecasts(bonds, macro, sentiment, start="1985-01", end="2024-02",
                     publication_lag=2, verbose=True):
    """All 15 subsets of four families, with OLS and two ridge penalties.

    Share the *training-only* PCA across models at each date. Ridge penalties
    apply after block scaling; neither penalties nor families use evaluation
    labels for selection. An online combination is added separately.
    """
    if "ym" in bonds:
        bonds = bonds.set_index("ym")
    features = available_features(bonds, macro, sentiment, publication_lag)
    features = features.join(technical_features(bonds))
    macro_cols = [c for c in features if c.startswith("macro::")]
    market_cols = [f"f{n}" for n in range(1, 6)]+["SENT", "SENT_SQ"]+list(technical_features(bonds))
    # Same training observations for all subset comparisons.
    eligible = features[market_cols].notna().all(axis=1) & features[macro_cols].notna().any(axis=1)
    data = features.loc[eligible].join(bonds.sx_bar)
    subsets = [c for size in range(1, 5) for c in combinations(FAMILIES, size)]
    output, audit = {}, []
    for date in data.loc[start:end].index:
        training = matured_training(data, date).dropna(subset=["sx_bar"])
        if len(training) < MIN_TRAIN_OBS:
            continue
        y = training.sx_bar.to_numpy()
        macro_pipe = _numeric_pipeline(pca=True)
        z_macro = macro_pipe.fit_transform(training[macro_cols])
        x_macro = macro_pipe.transform(data.loc[[date], macro_cols])
        blocks, now = {}, {}
        for name, cols in {"CP": [f"f{n}" for n in range(1, 6)],
                           "BW": ["SENT", "SENT_SQ"], "TI": ["MA12"]}.items():
            pipe = _numeric_pipeline()
            blocks[name] = pipe.fit_transform(training[cols])
            now[name] = pipe.transform(data.loc[[date], cols])
        blocks["LN"], now["LN"] = z_macro, x_macro
        row = {"EH": y.mean()}
        for subset in subsets:
            X, x = (np.hstack([source[f] for f in subset]) for source in (blocks, now))
            for alpha in RIDGES:
                label = "+".join(subset)+(f"_R{alpha}" if alpha else "")
                model = Ridge(alpha=alpha) if alpha else LinearRegression()
                row[label] = float(model.fit(X, y).predict(x)[0])
        for k in LOOKBACKS:
            pipe = Pipeline([("prepare", _numeric_pipeline()), ("ols", LinearRegression())])
            row[f"MA{k}"] = float(pipe.fit(training[[f"MA{k}"]], y)
                                   .predict(data.loc[[date], [f"MA{k}"]])[0])
        multi = Pipeline([("prepare", _numeric_pipeline()), ("ridge", Ridge(alpha=100))])
        cols = [f"MA{k}" for k in LOOKBACKS]
        row["MA_multi_R100"] = float(multi.fit(training[cols], y).predict(data.loc[[date], cols])[0])
        # No regression or outcome-based window selection for this direct signal.
        row["TrendVote"] = 100*float((data.loc[date, cols] > 0).mean())
        row["EW4"] = np.mean([row[f] for f in FAMILIES])
        output[date] = row
        audit.append({"forecast_month": str(date), "train_n": len(y),
                      "train_last_label_end": str(training.index.max()+12),
                      "macro_latest_reference": str(date-publication_lag)})
        if verbose and date.month == 1 and date.year % 5 == 0:
            print(f"  full family grid through {date}", flush=True)
    result = pd.DataFrame.from_dict(output, orient="index").sort_index()
    result.index.name = "ym"
    return ForecastResult(result, pd.DataFrame(audit))


def online_combination(predictions, actual, candidates, *, min_history=60, lookback=120):
    """Mix experts using only losses whose one-year outcomes have matured.

    EW_mature: inverse recent MSE, no fitted learning rate. Select_mature:
    minimum recent MSE. Both default to EH until 60 matured forecast errors.
    Recompute monthly and retain a complete weight/selection audit trail.
    """
    if "EH" not in candidates:
        raise ValueError("The historical mean must be a candidate")
    frame = predictions[candidates].join(actual.rename("actual"))
    values, rows = {}, []
    for date in predictions.index:
        past = frame.loc[(frame.index+12 < date) & (frame.index >= date-12-lookback)].dropna()
        if len(past) < min_history:
            weights = pd.Series(0., index=candidates)
            weights["EH"] = 1.
            selected = "EH"
        else:
            mse = past[candidates].sub(past.actual, axis=0).pow(2).mean()
            weights = (1/mse.clip(lower=1e-10))
            weights /= weights.sum()
            selected = mse.idxmin()
        values[date] = {"EW_mature": predictions.loc[date, candidates] @ weights,
                        "Select_mature": predictions.loc[date, selected]}
        rows.append({"forecast_month": str(date), "matured_errors": len(past),
                     "last_label_end": str(past.index.max()+12) if len(past) else "",
                     "selected": selected, **{f"weight::{c}": weights[c] for c in candidates}})
    return pd.DataFrame.from_dict(values, orient="index"), pd.DataFrame(rows)


def r2_decomposition(bonds, macro, sentiment, predictions, start="1990-01", end=None):
    """Compare explicitly different R2 denominators and sample periods.

    Ex-post fits on the evaluation period are diagnostic upper bounds for a
    fixed OLS design on that period, never tradable forecasts.
    """
    if "ym" in bonds:
        bonds = bonds.set_index("ym")
    features = available_features(bonds, macro, sentiment)
    macro_cols = [c for c in features if c.startswith("macro::")]
    cp = [f"f{n}" for n in range(1, 6)]
    specs = {"CP": (cp, False), "LN": ([], True),
             "BW": (["SENT", "SENT_SQ"], False), "TI": (["TI_avg"], False),
             "CP+LN": (cp, True), "LN+BW": (["SENT", "SENT_SQ"], True)}
    evaluation = predictions.loc[start:end, ["EH", *specs]].join(bonds.sx_bar).dropna()
    rows = []
    common_market = [f"f{n}" for n in range(1, 6)]+["SENT", "SENT_SQ", "TI_avg"]
    initial = matured_training(features.join(bonds.sx_bar), pd.Period(start, "M"))
    initial = initial.dropna(subset=["sx_bar", *common_market])
    initial = initial.loc[initial[macro_cols].notna().any(axis=1)]
    for name, (market_cols, pca) in specs.items():
        train = initial
        cols = market_cols+(macro_cols if pca else [])
        y0, y = train.sx_bar, evaluation.sx_bar
        def fitted_r2(X, target):
            pipe = model_pipeline("ols", macro_cols if pca else [], market_cols, pca=pca)
            return pipe.fit(X, target).score(X, target)
        model_mse = np.mean((y-evaluation[name])**2)
        eh_mse = np.mean((y-evaluation.EH)**2)
        const_mse = np.mean((y-y.mean())**2)
        rows.append({"model": name, "train_n": len(train), "oos_n": len(y),
                     "initial_train_r2_pct": 100*fitted_r2(train[cols], y0),
                     "oos_r2_eh_pct": 100*(1-model_mse/eh_mse),
                     "oos_r2_test_mean_pct": 100*(1-model_mse/const_mse),
                     "expost_oos_fit_r2_pct": 100*fitted_r2(features.loc[y.index, cols], y),
                     "model_mse": model_mse, "eh_mse": eh_mse,
                     "test_mean_mse": const_mse, "initial_train_var": np.var(y0)})
    return pd.DataFrame(rows)


def paired_sharpe_bootstrap(trades, model, benchmark="BuyHold", n_boot=5000, seed=42):
    """Paired circular block bootstrap, retaining shared years and cash rates.

    Five-year blocks reflect persistence and very limited independent years.
    Percentile intervals quantify uncertainty, not correction for model search.
    """
    groups = {m: g.set_index("purchase") for m, g in trades.groupby("model")}
    a, b = groups[model].align(groups[benchmark], join="inner", axis=0)
    x = (a.return_net-a.risk_free).to_numpy()
    z = (b.return_net-b.risk_free).to_numpy()
    n = len(x)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, int(np.ceil(n/5))))
    indices = ((starts[:, :, None]+np.arange(5)) % n).reshape(n_boot, -1)[:, :n]
    def sr(values):
        mean, std = values.mean(axis=-1), values.std(axis=-1, ddof=1)
        return np.divide(mean, std, out=np.full_like(mean, np.nan), where=std > 1e-12)
    delta = sr(x[indices])-sr(z[indices])
    return {"model": model, "years": n, "delta_sharpe": sr(x)-sr(z),
            "ci_low": np.nanquantile(delta, .025), "ci_high": np.nanquantile(delta, .975),
            "valid_bootstraps": int(np.isfinite(delta).sum())}
