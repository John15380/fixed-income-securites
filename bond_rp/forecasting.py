"""Monthly recursive forecasts with strictly matured labels and vintage-local PCA."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from .config import TRAIN_START
from .models import _numeric_pipeline
from .factors import common_index, ln6_terms


def target_columns(horizon):
    if horizon not in (1, 3, 12):
        raise ValueError("Supported holding periods: 1, 3, 12 months")
    return [f"mx{n}" if horizon == 1 else f"ex{horizon}_{n}" for n in range(2, 6)]


def features_at(market, panel, *, lag=2, lookbacks=(12,)):
    if lag < 1:
        raise ValueError("Macro reference-month lag must be positive")
    features = market[[f"f{n}" for n in range(1,6)]+[f"fs{n}" for n in range(2,6)]].copy()
    macro = panel.copy()
    macro.index = macro.index+lag
    macro.columns = ["macro::"+str(c) for c in macro]
    features = features.join(macro)
    lp = market[[f"lp{n}" for n in range(2,6)]]
    for k in lookbacks:
        features[f"trend{k}"] = (lp-lp.rolling(k).mean()).mean(axis=1, skipna=False)
    return features


def training_sample(data, month, horizon, window=120, embargo=1):
    if embargo < 0:
        raise ValueError("Label embargo must be nonnegative")
    mask = (data.index >= TRAIN_START) & (data.index+horizon <= month-embargo)
    if window is not None:
        mask &= data.index >= month-horizon-embargo-window+1
    return data.loc[mask]


def fit_forecasts(train, current, macro_cols, targets, lookbacks=(12,)):
    y = train[targets].to_numpy()
    cp_cols = [f"f{n}" for n in range(1,6)]
    # For a basket-average forecast, common_index and multivariate OLS have
    # identical fitted means. CP's second stage has no additional intercept.
    values = {"EH": np.tile(y.mean(axis=0),(len(current),1))}
    values["FB"] = np.column_stack([LinearRegression().fit(train[[f"fs{n}"]], y[:,n-2])
                                    .predict(current[[f"fs{n}"]]) for n in range(2,6)])
    X, x = train[cp_cols].to_numpy(), current[cp_cols].to_numpy()
    values["CP"] = common_index(X, y, x, second_intercept=False)[0]
    pipe = _numeric_pipeline(pca=True)
    Z = pipe.fit_transform(train[macro_cols])
    z = pipe.transform(current[macro_cols])
    values["LN6"] = common_index(ln6_terms(Z), y, ln6_terms(z))[0]
    # The strategy extension retains all eight PCs linearly. It is not LN6.
    values["Macro8"] = common_index(Z, y, z)[0]
    for k in lookbacks:
        values[f"Trend{k}"] = LinearRegression().fit(train[[f"trend{k}"]], y).predict(current[[f"trend{k}"]])
        values[f"EW3_MA{k}"] = np.mean([values["CP"], values["Macro8"], values[f"Trend{k}"]], axis=0)
    values["CP_Macro8"] = (values["CP"]+values["Macro8"])/2
    values["CP_Trend12"] = (values["CP"]+values["Trend12"])/2
    values["Macro8_Trend12"] = (values["Macro8"]+values["Trend12"])/2
    values["CP_LN6"] = (values["CP"]+values["LN6"])/2
    values["EW3"] = values["EW3_MA12"]
    return values, int(pipe.named_steps["columns"].keep_.sum())


def generate_forecasts(market, macro, *, archive=None, start="2000-01", end="2026-01",
                       horizons=(1,3,12), windows=(120,), lookbacks=(3,6,9,12,18,24), lag=2, embargo=1):
    records, audit = [], []
    static = features_at(market, macro, lag=lag, lookbacks=lookbacks)
    dates = market.loc[start:end].index
    for date in dates:
        vintage = None
        features = static
        if archive is not None:
            vintage, panel = archive.at(date)
            features = features_at(market, panel, lag=lag, lookbacks=lookbacks)
        mc = [c for c in features if c.startswith("macro::")]
        required = [f"f{n}" for n in range(1,6)]+[f"fs{n}" for n in range(2,6)]+[f"trend{k}" for k in lookbacks]
        if not features.loc[date, required].notna().all() or not features.loc[date, mc].notna().any():
            raise ValueError(f"Unavailable current features: {date}")
        for h in horizons:
            targets = target_columns(h)
            data = features.join(market[targets])
            eligible = data[required+targets].notna().all(axis=1) & data[mc].notna().any(axis=1)
            for window in windows:
                train = training_sample(data.loc[eligible], date, h, window, embargo)
                if len(train) < 120:
                    raise ValueError(f"Insufficient training rows: {date}, h={h}, window={window}")
                values, columns = fit_forecasts(train, features.loc[[date]], mc, targets, lookbacks)
                window_name = "exp" if window is None else str(window)
                for model, prediction in values.items():
                    records.append(dict(ym=str(date), horizon=h, window=window_name, model=model,
                                        **{f"mu{n}":prediction[0,n-2] for n in range(2,6)}))
                audit.append(dict(ym=str(date), horizon=h, window=window_name, train_n=len(train),
                                  train_start=str(train.index.min()), last_label_end=str(train.index.max()+h),
                                  latest_macro_reference=str(date-lag), vintage=str(vintage) if vintage else "revised",
                                  macro_columns=columns, embargo=embargo, macro_lag=lag))
        if date.month == 1 and date.year % 5 == 0:
            print(f"Forecasts through {date}: {'archival' if archive else 'revised'}", flush=True)
    return pd.DataFrame(records), pd.DataFrame(audit)
