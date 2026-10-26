"""Causal nonlinear model comparison for the project's Treasury strategy.

Hyperparameters are tuned on sparse 2000--2006 walk-forward folds.  Model
family and forecast shrinkage are frozen with 2007--2012 outcomes.  Purchases
from 2013 onward are used only for evaluation.  Every fold fits missing-value
filters, macro PCA, scaling and the learner from its own matured training rows.
"""
from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri
from scipy.stats import rankdata
from sklearn.linear_model import ElasticNet
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

from .factors import common_index
from .forecasting import features_at, training_sample
from .models import _numeric_pipeline
from .portfolio import performance, simulate, summarize

TUNING_END = pd.Period("2007-01", "M")
SELECTION_END = pd.Period("2013-01", "M")
FAMILIES = ("ElasticNet", "ElasticNet_GC", "XGBoost", "XGBoost_GC")


def gaussian_augment(X, y, *, ratio=1.0, seed=42):
    """Draw training-only rows from an empirical-marginal Gaussian copula."""
    values = np.column_stack([X, y])
    if len(values) < 36 or not np.isfinite(values).all() or ratio <= 0:
        raise ValueError("Copula augmentation needs finite training data and a positive ratio")
    ranks = np.column_stack([rankdata(values[:, j], method="average")
                             for j in range(values.shape[1])])
    latent = ndtri((ranks-.5)/len(values))
    covariance = np.cov(latent, rowvar=False)
    scale = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
    correlation = covariance/np.outer(scale, scale)
    correlation = .99999999*correlation+np.eye(values.shape[1])*1e-8
    draws = np.random.default_rng(seed).multivariate_normal(
        np.zeros(values.shape[1]), correlation, size=int(round(len(values)*ratio)))
    probabilities = ndtr(draws)
    synthetic = np.column_stack([
        np.quantile(values[:, j], probabilities[:, j]) for j in range(values.shape[1])])
    return (np.vstack([X, synthetic[:, :X.shape[1]]]),
            np.vstack([y, synthetic[:, X.shape[1]:]]))


def _fold(market, panel, date, window=120):
    lookbacks = (3, 6, 9, 12, 18, 24)
    features = features_at(market, panel, lag=2, lookbacks=lookbacks)
    macro = [c for c in features if c.startswith("macro::")]
    curve = ([f"f{n}" for n in range(1, 6)]+[f"fs{n}" for n in range(2, 6)]
             +[f"trend{k}" for k in lookbacks])
    targets = [f"mx{n}" for n in range(2, 6)]
    data = features.join(market[targets])
    eligible = data[curve+targets].notna().all(axis=1) & data[macro].notna().any(axis=1)
    train = training_sample(data.loc[eligible], date, 1, window, embargo=1)
    if len(train) != window:
        raise ValueError(f"Expected {window} matured rows at {date}, found {len(train)}")
    now = features.loc[[date]]
    macro_pipe = _numeric_pipeline(pca=True).fit(train[macro])
    Z, z = macro_pipe.transform(train[macro]), macro_pipe.transform(now[macro])
    scaler = StandardScaler().fit(train[curve])
    C, c = scaler.transform(train[curve]), scaler.transform(now[curve])
    X = {"curve": C, "curve_macro": np.column_stack([C, Z])}
    x = {"curve": c, "curve_macro": np.column_stack([c, z])}
    y = train[targets].to_numpy()
    return dict(date=date, X=X, x=x, y=y, actual=market.loc[date, targets].to_numpy(),
                eh=y.mean(axis=0), last_label_end=train.index.max()+1,
                macro_columns=int(macro_pipe.named_steps["columns"].keep_.sum()))


def build_folds(market, archive, dates):
    folds = []
    for date in dates:
        _, panel = archive.at(date)
        folds.append(_fold(market, panel, date))
        if date.month == 1 and date.year % 5 == 0:
            print(f"Nonlinear folds through {date}", flush=True)
    return folds


def _config(trial, family):
    cfg = {"representation": trial.suggest_categorical("representation", ["curve", "curve_macro"])}
    if family.endswith("_GC"):
        cfg["gc_ratio"] = trial.suggest_categorical("gc_ratio", [.5, 1.0])
    else:
        cfg["gc_ratio"] = 0.0
    if family.startswith("ElasticNet"):
        cfg.update(alpha=trial.suggest_float("alpha", 1e-3, 10., log=True),
                   l1_ratio=trial.suggest_float("l1_ratio", .05, 1.0))
    else:
        cfg.update(n_estimators=trial.suggest_int("n_estimators", 25, 400, step=25),
                   max_depth=trial.suggest_int("max_depth", 1, 4),
                   learning_rate=trial.suggest_float("learning_rate", .01, .2, log=True),
                   min_child_weight=trial.suggest_float("min_child_weight", 1., 50., log=True),
                   reg_lambda=trial.suggest_float("reg_lambda", .1, 100., log=True),
                   reg_alpha=trial.suggest_float("reg_alpha", 1e-6, 1., log=True),
                   subsample=trial.suggest_float("subsample", .6, 1.),
                   colsample_bytree=trial.suggest_float("colsample_bytree", .6, 1.))
    return cfg


def _baseline(family):
    base = dict(representation="curve_macro")
    if family.endswith("_GC"):
        base["gc_ratio"] = 1.
    if family.startswith("ElasticNet"):
        base.update(alpha=.1, l1_ratio=.5)
    else:
        base.update(n_estimators=100, max_depth=2, learning_rate=.05,
                    min_child_weight=10., reg_lambda=10., reg_alpha=1e-3,
                    subsample=1., colsample_bytree=1.)
    return base


def _predict(fold, family, cfg):
    X, x, y = fold["X"][cfg["representation"]], fold["x"][cfg["representation"]], fold["y"]
    location, scale = y.mean(axis=0), np.maximum(y.std(axis=0), 1e-8)
    normalized = (y-location)/scale
    if cfg.get("gc_ratio", 0) > 0:
        X, normalized = gaussian_augment(X, normalized, ratio=cfg["gc_ratio"])
    if family.startswith("ElasticNet"):
        columns=[]
        for j in range(4):
            model=ElasticNet(alpha=cfg["alpha"],l1_ratio=cfg["l1_ratio"],max_iter=5000,
                             tol=1e-4,random_state=42)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always",ConvergenceWarning)
                model.fit(X,normalized[:,j])
            if any(issubclass(w.category,ConvergenceWarning) for w in caught):
                raise ArithmeticError("ElasticNet candidate did not converge")
            columns.append(model.predict(x))
        pred=np.column_stack(columns)
    else:
        from xgboost import XGBRegressor
        keys = ("n_estimators", "max_depth", "learning_rate", "min_child_weight",
                "reg_lambda", "reg_alpha", "subsample", "colsample_bytree")
        params = {k: cfg[k] for k in keys}
        pred = XGBRegressor(objective="reg:squarederror", tree_method="hist", n_jobs=1,
                            random_state=42, verbosity=0,
                            multi_strategy="one_output_per_tree", **params).fit(X, normalized).predict(x)
    return np.asarray(pred).reshape(-1, 4)*scale+location


def _loss(prediction, folds):
    y = np.vstack([f["actual"] for f in folds])
    eh = np.vstack([f["eh"] for f in folds])
    denominator = np.sum((y-eh)**2, axis=0)
    return float(np.mean(np.sum((y-prediction)**2, axis=0)/denominator))


def tune(folds, output, *, en_trials=32, xgb_trials=64):
    """Tune four learner families without using 2007 or later outcomes."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tuning = [f for f in folds if f["date"]+1 < TUNING_END and f["date"].month in (1, 4, 7, 10)]
    if len(tuning) < 24:
        raise ValueError("Insufficient pre-2007 tuning folds")
    selected, rows = {}, []
    for family in FAMILIES:
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.enqueue_trial(_baseline(family))
        budget = en_trials if family.startswith("ElasticNet") else xgb_trials
        def objective(trial):
            cfg = _config(trial, family)
            try:
                prediction = np.vstack([_predict(fold, family, cfg) for fold in tuning])
                return _loss(prediction, tuning)
            except (ArithmeticError,FloatingPointError):
                return 1e6
        study.optimize(objective, n_trials=budget, n_jobs=1, show_progress_bar=False)
        selected[family] = _config(study.best_trial, family)
        for trial in study.trials:
            rows.append(dict(family=family, trial=trial.number, loss=trial.value,
                             selected=trial.number == study.best_trial.number,
                             params=json.dumps(trial.params, sort_keys=True)))
        print(f"{family}: {budget} Optuna trials, tuning loss={study.best_value:.6f}", flush=True)
    table = pd.DataFrame(rows)
    table.to_csv(Path(output)/"nonlinear_search.csv", index=False)
    return selected, table


def _r2(y, prediction, benchmark):
    return 1-np.sum((y-prediction)**2)/np.sum((y-benchmark)**2)


def run_nonlinear_study(market, archive, linear_forecasts, output):
    """Run tuned EN/XGBoost/GC comparisons and a frozen 2013+ evaluation."""
    output = Path(output)
    dates = market.dropna(subset=[f"mx{n}" for n in range(2, 6)]).loc["2000-01":"2026-01"].index
    folds = build_folds(market, archive, dates)
    configs, search = tune(folds, output)
    raw = {family: np.vstack([_predict(fold, family, cfg) for fold in folds])
           for family, cfg in configs.items()}
    y = np.vstack([f["actual"] for f in folds])
    eh = np.vstack([f["eh"] for f in folds])
    selection = np.array([(f["date"] >= TUNING_END) and (f["date"]+1 < SELECTION_END)
                          for f in folds])
    holdout = np.array([f["date"] >= SELECTION_END for f in folds])
    rows, calibrated = [], {}
    for family in FAMILIES:
        delta = raw[family][selection]-eh[selection]
        residual = y[selection]-eh[selection]
        shrink = float(np.clip(np.sum(delta*residual)/np.sum(delta**2), 0, 1))
        calibrated[family] = eh+shrink*(raw[family]-eh)
        rows.append(dict(family=family, selection_loss=_loss(raw[family][selection],
                                                             [f for f, keep in zip(folds, selection) if keep]),
                         shrinkage=shrink, selected=False,
                         configuration=json.dumps(configs[family], sort_keys=True)))
    family_winner = min(rows, key=lambda r: r["selection_loss"])["family"]
    accepted = (next(r for r in rows if r["family"] == family_winner)["selection_loss"] < 1-1e-8
                and next(r for r in rows if r["family"] == family_winner)["shrinkage"] > 0)
    winner = family_winner if accepted else "HistoricalMean"
    for row in rows:
        row["family_winner"] = row["family"] == family_winner
        row["accepted"] = accepted and row["family"] == family_winner
    selection_table = pd.DataFrame(rows)
    selection_table.to_csv(output/"nonlinear_selection.csv", index=False)

    records = []
    for i, date in enumerate(dates):
        records.append(dict(ym=str(date), horizon=1, window="120", model="EH",
                            **{f"mu{n}": eh[i, n-2] for n in range(2, 6)}))
        for family in FAMILIES:
            records.append(dict(ym=str(date), horizon=1, window="120", model=family,
                                **{f"mu{n}": calibrated[family][i, n-2] for n in range(2, 6)}))
    prediction_frame = pd.DataFrame(records)
    prediction_frame.to_csv(output/"nonlinear_forecasts.csv", index=False)
    _, nonlinear_trades = simulate(market, prediction_frame, start=str(dates.min()),
                                   end=str(dates.max()), shrink=1.)
    linear = linear_forecasts.loc[(linear_forecasts.horizon == 1)
                                  & (linear_forecasts.window.astype(str) == "120")
                                  & linear_forecasts.model.isin(["EH", "EW3"])]
    _, linear_trades = simulate(market, linear, start=str(dates.min()), end=str(dates.max()))
    account = pd.concat([linear_trades.loc[linear_trades.model.eq("EW3")],
                         nonlinear_trades.loc[nonlinear_trades.model.isin(FAMILIES)]], ignore_index=True)
    period_masks = {"selection_2007_2012": lambda x: (x >= "2007-01") & (x <= "2012-11"),
                    "holdout_2013_2026": lambda x: x >= "2013-01",
                    "full_2000_2026": lambda x: np.ones(len(x), dtype=bool)}
    performances = []
    for model, group in account.groupby("model", sort=False):
        purchase = group.purchase.astype(str).to_numpy()
        for period, make_mask in period_masks.items():
            mask = make_mask(purchase)
            stats = performance(group.return_net.to_numpy()[mask], group.risk_free.to_numpy()[mask])
            performances.append(dict(model=model, period=period, **stats,
                                     mean_exposure=group.gross_exposure.to_numpy()[mask].mean()))
    # Forecast R2 uses each model's frozen calibration and the same EH denominator.
    ew = linear.loc[linear.model.eq("EW3")].copy()
    ew["ym"] = pd.PeriodIndex(ew.ym, freq="M")
    ew = ew.set_index("ym")[[f"mu{n}" for n in range(2, 6)]].reindex(dates).to_numpy()
    ew = eh+.5*(ew-eh)
    forecast_models = {"EW3": ew, **calibrated}
    scores = []
    for model, prediction in forecast_models.items():
        for period, mask in (("selection_2007_2012", selection), ("holdout_2013_2026", holdout)):
            for j, maturity in enumerate((2, 3, 4, 5, "basket")):
                if j < 4:
                    yy, pp, bb = y[mask, j], prediction[mask, j], eh[mask, j]
                else:
                    yy, pp, bb = y[mask].mean(1), prediction[mask].mean(1), eh[mask].mean(1)
                scores.append(dict(model=model, period=period, maturity=maturity,
                                   r2_oos_pct=100*_r2(yy, pp, bb)))
    performance_table, score_table = pd.DataFrame(performances), pd.DataFrame(scores)
    performance_table.to_csv(output/"model_performance.csv", index=False)
    score_table.to_csv(output/"model_forecast_scores.csv", index=False)
    protocol = dict(tuning="2000-01..2006-11 quarterly purchase folds",
                    family_selection_and_shrinkage="2007-01..2012-11, outcomes known before 2013-01",
                    evaluation="2013-01..2026-01 purchases; no model or parameter selection",
                    training_window=120, macro_lag=2, label_rule="purchase+1 < decision",
                    en_trials_per_family=32, xgb_trials_per_family=64, seed=42,
                    selected_family=winner,
                    family_winner=family_winner,
                    selected_configuration=configs[family_winner] if accepted else None,
                    selected_shrinkage=(float(selection_table.loc[selection_table.accepted,
                                                                  "shrinkage"].iloc[0])
                                        if accepted else 0.))
    (output/"nonlinear_protocol.json").write_text(json.dumps(protocol, indent=2)+"\n")
    return performance_table, score_table, selection_table, winner
