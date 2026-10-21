"""Expanding-window forecasts with matured labels and fold-local preprocessing."""

from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import (CV_SPLITS, HORIZON_MONTHS, LASSO_ALPHAS, MIN_TRAIN_OBS,
                     OOS_START, PUB_LAG_MONTHS, TRAIN_START)


class MissingRateFilter(TransformerMixin, BaseEstimator):
    """Select columns using only this fit's observations, including inside CV."""
    def __init__(self, max_missing=0.20):
        self.max_missing = max_missing

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.keep_ = np.mean(~np.isfinite(values), axis=0) <= self.max_missing
        if not self.keep_.any():
            raise ValueError("No usable features in the training window")
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)[:, self.keep_].copy()
        values[~np.isfinite(values)] = np.nan
        return values


def matured_training(frame, forecast_month, window=None):
    """A label bought at s is usable only if s+12 < forecast_month."""
    mask = (frame.index >= TRAIN_START) & (frame.index + HORIZON_MONTHS < forecast_month)
    if window is not None:
        mask &= frame.index >= forecast_month - HORIZON_MONTHS - window
    return frame.loc[mask]


def purged_time_splits(index, n_splits=CV_SPLITS):
    """Forward validation, with the gap measured in calendar months, not rows."""
    dates = pd.PeriodIndex(index, freq="M")
    if dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("CV dates must be unique and ordered")
    folds = []
    test_size = min(len(dates)//(n_splits+1), (len(dates)-49)//n_splits)
    if test_size < 1:
        raise ValueError("Insufficient observations for purged CV")
    for training, validation in TimeSeriesSplit(n_splits=n_splits, test_size=test_size).split(dates):
        training = training[dates[training] + HORIZON_MONTHS < dates[validation[0]]]
        if len(training) < 24:
            raise ValueError("Too little training data after the 12-month purge")
        folds.append((training, validation))
    return folds


def available_features(bonds, macro, sentiment=None, publication_lag=PUB_LAG_MONTHS):
    if publication_lag < 1:
        raise ValueError("A positive publication lag is required")
    market = ["fs_avg", "f1", "f2", "f3", "f4", "f5", "TI_avg"]
    features = bonds[market].copy()
    lagged = macro.copy()
    lagged.index = lagged.index + publication_lag
    lagged.columns = ["macro::" + str(c) for c in lagged]
    features = features.join(lagged, how="left")
    if sentiment is not None:
        lagged_sentiment = sentiment.copy()
        lagged_sentiment.index = lagged_sentiment.index + publication_lag
        features = features.join(lagged_sentiment, how="left")
    return features


def _numeric_pipeline(pca=False):
    steps = [("columns", MissingRateFilter()), ("impute", SimpleImputer(strategy="mean")),
             ("scale", StandardScaler())]
    if pca:
        steps += [("pca", PCA(n_components=8, svd_solver="full")), ("pc_scale", StandardScaler())]
    return Pipeline(steps)


def model_pipeline(kind, macro_columns=(), market_columns=(), pca=False):
    blocks = []
    if macro_columns:
        blocks.append(("macro", _numeric_pipeline(pca), list(macro_columns)))
    if market_columns:
        blocks.append(("market", _numeric_pipeline(False), list(market_columns)))
    if kind == "ols":
        estimator = LinearRegression()
    elif kind == "lasso":
        estimator = Lasso(alpha=0.1, max_iter=100000, tol=1e-5, selection="cyclic")
    elif kind == "rf":
        estimator = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=42, n_jobs=1)
    elif kind == "gbm":
        estimator = GradientBoostingRegressor(n_estimators=100, max_depth=2,
                                             learning_rate=0.05, subsample=0.5, random_state=42)
    else:
        raise ValueError(kind)
    return Pipeline([("features", ColumnTransformer(blocks, remainder="drop")), ("model", estimator)])


@dataclass
class ForecastResult:
    predictions: pd.DataFrame
    audit: pd.DataFrame


def forecast_models(bonds, macro, sentiment=None, *, include_ml=True, oos_start=OOS_START,
                    publication_lag=PUB_LAG_MONTHS, window=None, target="sx_bar", names=None,
                    verbose=True):
    """Refit monthly; tune LASSO each January using only purged historical folds.

    Current-vintage macro/sentiment values are lagged, not genuine real-time
    vintages. Statistical evaluation must retain that distinction.
    """
    if "ym" in bonds:
        bonds = bonds.set_index("ym")
    features = available_features(bonds, macro, sentiment, publication_lag)
    macro_cols = [c for c in features if c.startswith("macro::")]
    cp = [f"f{n}" for n in range(1, 6)]
    specs = {"EH": ("mean", [], False, False),
             "FB": ("ols", ["fs_avg"], False, False),
             "CP": ("ols", cp, False, False),
             "TI": ("ols", ["TI_avg"], False, False),
             "LN": ("ols", [], True, True),
             "LN_LASSO": ("lasso", [], True, True),
             "CPLN": ("ols", cp, True, True)}
    if sentiment is not None:
        specs["BW"] = ("ols", ["SENT", "SENT_SQ"], False, False)
        specs["LNBW"] = ("ols", ["SENT", "SENT_SQ"], True, True)
    if include_ml:
        specs.update({"LASSO_raw": ("lasso", [], True, False),
                      "RF_raw": ("rf", [], True, False), "GBM_raw": ("gbm", [], True, False),
                      "RF_pca": ("rf", [], True, True), "GBM_pca": ("gbm", [], True, True)})
    if names is not None:
        specs = {name: specs[name] for name in names}
    output, audit = {}, []
    for name, (kind, market_cols, use_macro, pca) in specs.items():
        used = market_cols + (macro_cols if use_macro else [])
        data = features[used].join(bonds[target])
        # Missing market/sentiment observations are unavailable signals, not imputable history.
        if market_cols:
            data = data.loc[data[market_cols].notna().all(axis=1)]
        if use_macro:
            data = data.loc[data[macro_cols].notna().any(axis=1)]
        dates = data.index[data.index >= oos_start]
        values, selected_alpha, selected_at = {}, None, None
        if verbose:
            print(f"  {name}: {len(dates)} forecast dates", flush=True)
        for count, date in enumerate(dates):
            training = matured_training(data, date, window).dropna(subset=[target])
            if len(training) < MIN_TRAIN_OBS:
                continue
            if kind == "mean":
                prediction = training[target].mean()
            else:
                pipeline = model_pipeline(kind, macro_cols if use_macro else [], market_cols, pca)
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    if kind == "lasso" and (selected_alpha is None or date.month == 1):
                        folds = purged_time_splits(training.index)
                        search = GridSearchCV(pipeline, {"model__alpha": LASSO_ALPHAS}, cv=folds,
                                              scoring="neg_mean_squared_error", n_jobs=1, error_score="raise")
                        search.fit(training[used], training[target])
                        selected_alpha = search.best_params_["model__alpha"]
                        selected_at = date
                        fitted = search.best_estimator_
                    else:
                        if kind == "lasso":
                            pipeline.set_params(model__alpha=selected_alpha)
                        fitted = pipeline.fit(training[used], training[target])
                prediction = float(fitted.predict(data.loc[[date], used])[0])
            values[date] = float(prediction)
            audit.append({"model": name, "forecast_month": str(date), "train_n": len(training),
                          "train_last_purchase": str(training.index.max()),
                          "train_last_label_end": str(training.index.max() + HORIZON_MONTHS),
                          "macro_latest_reference": str(date - publication_lag) if use_macro else None,
                          "sentiment_latest_reference": str(date - publication_lag) if "SENT" in market_cols else None,
                          "alpha": selected_alpha if kind == "lasso" else None,
                          "alpha_selected_at": str(selected_at) if kind == "lasso" else None})
            if verbose and count and count % 120 == 0:
                print(f"    through {date}", flush=True)
        output[name] = pd.Series(values, dtype=float)
    predictions = pd.DataFrame(output).sort_index()
    # Predeclared combination; never choose weights using OOS scores.
    components = ["CP", "LN", "BW", "GBM_raw"] if include_ml else ["CP", "LN", "BW"]
    if all(c in predictions for c in components):
        predictions["Ensemble"] = predictions[components].mean(axis=1, skipna=False)
    predictions.index.name = "ym"
    return ForecastResult(predictions, pd.DataFrame(audit))
