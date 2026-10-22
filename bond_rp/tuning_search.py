"""Deterministic, validation-only search over forecast model configurations.

The caller supplies already purged folds and their early validation outcomes.
This module has no market-data loader, test-period interface, or portfolio
objective. Model selection is conditional on these fixed, historical folds.
Gaussian-copula rows regularize the training distribution; evaluation always
uses observed validation outcomes.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

from .validated_models import gaussian_augment

REPRESENTATIONS = ("index11", "primitive32", "bounded32")
WINDOWS = ("expanding", "120", "240")
GC_RATIOS = (.25, .5, 1., 2.)
SEEDS = (42, 73)
ENGINE = "xgboost-native-multioutput-one-output-per-tree-v1"
VALIDATION_START = pd.Period("2011-01", "M")
VALIDATION_LABEL_CUTOFF = pd.Period("2014-01", "M")


def predict_config(fold, config):
    """Predict one or more rows using only supplied fold training observations.

    XGBoost builds an independent scalar-leaf tree for each maturity. Native
    multioutput and separate fits agree under full sampling; with row/column
    subsampling their random-number streams differ. All searches here use the
    same native engine, so such fits are a newly declared model configuration,
    rather than a claimed numerical reproduction of the old separate engine.
    """
    representation = config["representation"]
    X = np.asarray(fold["X"][representation], dtype=float)
    x = np.asarray(fold["x"][representation], dtype=float)
    y = np.asarray(fold["y"], dtype=float)
    if X.ndim != 2 or x.ndim != 2 or y.ndim != 2 or y.shape[1] != 4:
        raise ValueError("Expected matrices and four maturity-specific targets")
    if len(X) != len(y) or X.shape[1] != x.shape[1] or not len(y):
        raise ValueError("Training and prediction dimensions do not match")
    if not all(np.isfinite(a).all() for a in (X, x, y)):
        raise ValueError("Fold inputs must be finite after causal preprocessing")
    location = y.mean(axis=0)
    scale = np.maximum(y.std(axis=0), 1e-8)
    normalized = (y-location)/scale
    ratio = float(config.get("gc_ratio", 0.))
    if not np.isfinite(ratio) or ratio < 0:
        raise ValueError("Copula augmentation ratio must be finite and nonnegative")
    if ratio > 0:
        X, normalized = gaussian_augment(X, normalized, seed=42, ratio=ratio)
    params = dict(config["params"])
    if config["family"] == "EN":
        result = np.column_stack([
            ElasticNet(max_iter=100000, tol=1e-5, **params)
            .fit(X, normalized[:, j]).predict(x) for j in range(4)
        ])
    elif config["family"] == "XGB":
        from xgboost import XGBRegressor
        result = XGBRegressor(
            objective="reg:squarederror", tree_method="hist", n_jobs=1,
            random_state=42, verbosity=0, multi_strategy="one_output_per_tree",
            **params,
        ).fit(X, normalized).predict(x)
    else:
        raise ValueError(f"Unknown learner family: {config['family']}")
    return np.asarray(result)*scale+location


def _configuration(trial, family, augmented, windows):
    representation = trial.suggest_categorical("representation", list(REPRESENTATIONS))
    window = trial.suggest_categorical("window", list(windows))
    ratio = trial.suggest_categorical("gc_ratio", list(GC_RATIOS)) if augmented else 0.
    if family == "EN":
        params = dict(alpha=trial.suggest_float("alpha", 1e-4, 10., log=True),
                      l1_ratio=trial.suggest_float("l1_ratio", .02, 1.))
    else:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 25, 600, step=25),
            max_depth=trial.suggest_int("max_depth", 1, 5),
            learning_rate=trial.suggest_float("learning_rate", .005, .2, log=True),
            min_child_weight=trial.suggest_float("min_child_weight", 1., 80., log=True),
            reg_lambda=trial.suggest_float("reg_lambda", .01, 1000., log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-6, 10., log=True),
            subsample=trial.suggest_float("subsample", .6, 1.),
            colsample_bytree=trial.suggest_float("colsample_bytree", .6, 1.),
        )
    return dict(family=family, representation=representation, window=str(window),
                gc_ratio=float(ratio), params=params)


def _boundary_notes(trial):
    """Report near-boundary continuous/integer parameters, without convergence claims."""
    result = {}
    for key, distribution in trial.distributions.items():
        if not hasattr(distribution, "low"):
            continue
        lo, hi = float(distribution.low), float(distribution.high)
        value = float(trial.params[key])
        if getattr(distribution, "log", False):
            position = (np.log(value)-np.log(lo))/(np.log(hi)-np.log(lo))
        else:
            position = (value-lo)/(hi-lo)
        if position <= .05 or position >= .95:
            result[key] = dict(value=value, low=lo, high=hi,
                               edge="lower" if position <= .05 else "upper")
    return result


def _baseline_trial(family, augmented, windows):
    candidate = dict(representation="index11", window="expanding" if "expanding" in windows else windows[0])
    if augmented:
        candidate["gc_ratio"] = 1.
    if family == "EN":
        candidate.update(alpha=.1, l1_ratio=.5)
    else:
        candidate.update(n_estimators=100, max_depth=2, learning_rate=.05,
                         min_child_weight=10., reg_lambda=10., reg_alpha=1e-3,
                         subsample=1., colsample_bytree=1.)
    return candidate


def search(folds_by_window, actual, eh_expanding, folder, *, en_trials=64, xgb_trials=128):
    """Select four forecast configurations using fixed early-validation outcomes.

    EN's total budget is split equally between real and augmented fits, as is
    XGBoost's. Each resulting study budget is split between independent seed-42
    and seed-73 TPE studies. Two candidates are asked before either is told;
    their results are told in trial-number order. This makes parallel scheduling
    immaterial to model selection. No pruning or test-driven stopping is used.

    Returns ``(configurations, predictions)``. The configuration dictionary has
    EN_real, EN_GC, XGB_real, XGB_GC and selected_family; prediction values are
    the corresponding stacked, uncalibrated validation predictions.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if en_trials < 4 or xgb_trials < 4:
        raise ValueError("Each learner requires at least four trials to cover two groups and two seeds")
    actual = np.asarray(actual, dtype=float)
    eh_expanding = np.asarray(eh_expanding, dtype=float)
    if actual.ndim != 2 or actual.shape[1] != 4 or actual.shape != eh_expanding.shape:
        raise ValueError("Validation outcomes and common expanding-EH predictions must be N by 4")
    if not np.isfinite(actual).all() or not np.isfinite(eh_expanding).all():
        raise ValueError("Validation outcomes and benchmark must be finite")
    denominator = np.sum((actual-eh_expanding)**2, axis=0)
    if np.any(denominator <= 0):
        raise ValueError("Every maturity needs positive common benchmark prediction error")
    windows = tuple(w for w in WINDOWS if w in folds_by_window)
    if not windows:
        raise ValueError("No declared estimation windows were supplied")
    validation_calendar = None
    for window in windows:
        folds = folds_by_window[window]
        if not folds or sum(len(f["y"]) == 0 for f in folds):
            raise ValueError("Empty training fold")
        rows = sum(len(f["x"][REPRESENTATIONS[0]]) for f in folds)
        if rows != len(actual):
            raise ValueError("Every window must cover exactly the same validation outcomes")
        if all("dates" in fold for fold in folds):
            calendar = []
            for fold in folds:
                dates = pd.PeriodIndex(fold["dates"], freq="M")
                horizon = int(fold["audit"]["horizon"])
                if (dates < VALIDATION_START).any() or (dates+horizon >= VALIDATION_LABEL_CUTOFF).any():
                    raise ValueError("Search folds must be early validation purchases with labels strictly before 2014-01")
                if "last_label_end" in fold["audit"] and pd.Period(fold["audit"]["last_label_end"], "M") >= dates.min():
                    raise ValueError("Search training fold contains an immature return label")
                calendar.extend(str(date) for date in dates)
            if validation_calendar is not None and calendar != validation_calendar:
                raise ValueError("Estimation windows use different validation calendars")
            validation_calendar = calendar
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    configurations, predictions, diagnostics, rows = {}, {}, {}, []
    start = time.perf_counter()

    def evaluate(config):
        prediction = np.vstack([predict_config(fold, config)
                                for fold in folds_by_window[config["window"]]])
        if prediction.shape != actual.shape or not np.isfinite(prediction).all():
            raise ValueError("Model generated invalid validation predictions")
        loss = float(np.mean(np.sum((actual-prediction)**2, axis=0)/denominator))
        return loss, prediction

    with ThreadPoolExecutor(max_workers=2) as executor:
        for family, total in (("EN", int(en_trials)), ("XGB", int(xgb_trials))):
            for augmented, group_budget in ((False, total//2), (True, total-total//2)):
                name = family+("_GC" if augmented else "_real")
                group = dict(budget=group_budget, seeds=[], engine=ENGINE if family == "XGB" else "independent-elasticnet-v1")
                best_loss, best_config, best_prediction = np.inf, None, None
                finished = 0
                for seed, seed_budget in zip(SEEDS, (group_budget//2, group_budget-group_budget//2)):
                    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=8))
                    study.enqueue_trial(_baseline_trial(family, augmented, windows))
                    for offset in range(0, seed_budget, 2):
                        pending = []
                        for _ in range(min(2, seed_budget-offset)):
                            trial = study.ask()
                            config = _configuration(trial, family, augmented, windows)
                            pending.append((trial, config, executor.submit(evaluate, config)))
                        for trial, config, future in pending:
                            loss, prediction = future.result()
                            study.tell(trial, loss)
                            finished += 1
                            record = dict(model=name, seed=seed, number=trial.number,
                                          group_trial=finished, state="COMPLETE", value=loss,
                                          config=json.dumps(config, sort_keys=True))
                            record.update({"params_"+key: value for key, value in trial.params.items()})
                            rows.append(record)
                            if loss < best_loss:
                                best_loss, best_config, best_prediction = loss, config, prediction
                                best_config = dict(best_config, selection=dict(
                                    objective=loss, seed=seed, trial=trial.number,
                                    group_trial=finished, common_benchmark="expanding EH"))
                            if finished % 16 == 0 or finished == group_budget:
                                print(f"    {name}: {finished}/{group_budget} trials; best validation loss={best_loss:.6f}", flush=True)
                    winner = study.best_trial
                    group["seeds"].append(dict(seed=seed, budget=seed_budget, best_value=winner.value,
                        best_trial=winner.number, last_improvement_trial=winner.number,
                        boundary_parameters=_boundary_notes(winner), best_params=winner.params))
                configurations[name] = best_config
                predictions[name] = best_prediction
                group["selected"] = best_config
                group["selection_is_last_group_trial"] = best_config["selection"]["group_trial"] == group_budget
                diagnostics[name] = group
                pd.DataFrame(rows).to_csv(folder/"trials.csv", index=False)
                pd.DataFrame([r for r in rows if r["model"] == name]).to_csv(folder/f"{name}_validation_trials.csv", index=False)
    configurations["selected_family"] = min(predictions, key=lambda name: configurations[name]["selection"]["objective"])
    diagnostics.update(protocol=dict(
        objective="mean across four maturities of validation SSE / common expanding-EH SSE",
        seeds=list(SEEDS), deterministic_batch_size=2, workers=2, pruning=False,
        en_total_budget=int(en_trials), xgb_total_budget=int(xgb_trials),
        representations=list(REPRESENTATIONS), windows=list(windows), gc_ratios=[0., *GC_RATIOS],
        engine=ENGINE, validation_rows=len(actual),
        validation_purchase_start=str(VALIDATION_START),
        validation_labels_strictly_before=str(VALIDATION_LABEL_CUTOFF),
        audited_validation_calendar=validation_calendar,
        limitation="Finite search; boundary flags and seed variation do not establish convergence."),
        selected_family=configurations["selected_family"], elapsed_seconds=time.perf_counter()-start)
    (folder/"search_diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    (folder/"frozen_configurations.json").write_text(json.dumps(configurations, indent=2))
    np.savez_compressed(folder/"best_validation_predictions.npz", **predictions)
    return configurations, predictions
