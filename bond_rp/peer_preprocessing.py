"""Fold-local macro cleaning, leaving the archived peer reconstruction intact.

The 10-IQR missing-value rule comes from McCracken--Ng (2016), footnote 5.
Mean imputation here is an explicit approximation to their EM procedure. The
previous-value alternative combines that threshold with Huang et al.'s causal
replacement principle. Neither is asserted to be the peer group's exact rule.
"""
import copy
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from .models import _numeric_pipeline

METHODS = ("raw", "fred10_mean", "fred10_previous")


def clean_panel(history, training_dates, method):
    """Fit thresholds on training X, then transform ordered, date-known X.

    With zero training IQR the threshold is unbounded, as in the existing
    diagnostic. Previous-value replacement never backfills from a future row.
    Original missing observations also remain missing until a previous value
    exists; any leading gaps are filled by the subsequent training imputer.
    """
    if method not in METHODS:
        raise ValueError(method)
    if not history.index.is_monotonic_increasing or history.index.has_duplicates:
        raise ValueError("Cleaning needs unique, ordered observations")
    if method == "raw":
        return history.copy(), dict(flagged_train=0, flagged_total=0)
    q = history.loc[training_dates].quantile([.25, .5, .75])
    width = 10*(q.loc[.75]-q.loc[.25])
    width = width.where(width > 1e-12, np.inf)
    flags = (history-q.loc[.5]).abs() > width
    cleaned = history.mask(flags)
    if method == "fred10_previous":
        cleaned = cleaned.ffill()
    return cleaned, dict(flagged_train=int(flags.loc[training_dates].to_numpy().sum()),
                         flagged_total=int(flags.to_numpy().sum()))


def _evaluation_inputs(fold):
    """Recover t-1 in the original current-factor basis from saved lag blocks."""
    training = pd.DataFrame(fold["X"], index=fold["train_dates"])
    old_lags = training.reindex(fold["lagged"]["train_dates"]-1).to_numpy()
    sd = old_lags.std(0)
    sd = np.where(sd > 0, sd, 1.)
    previous = fold["lagged"]["x"][:, 15:]*sd+old_lags.mean(0)
    now = pd.DataFrame(fold["x"], index=fold["dates"])
    lag = pd.DataFrame(previous, index=fold["dates"]-1)
    overlap = now.index.intersection(lag.index)
    np.testing.assert_allclose(now.loc[overlap], lag.loc[overlap], atol=1e-9, rtol=1e-9)
    return pd.concat([now, lag.loc[~lag.index.isin(now.index)]]).sort_index()


def rebuild_macro_fold(fold, features, candidates, selected, method):
    """Refit LN and template PCs; keep CP/MS/CP15/ME and mature labels fixed.

    Candidate membership is the same pre-2010 frozen set in every branch.
    Cleaning is applied to transformed macro series and generated candidate
    features before their respective PCA, with a separate fit in each fold.
    Both current and lagged values use the same fitted outer factor basis.
    """
    if method == "raw":
        return copy.deepcopy(fold)
    train_dates, dates = fold["train_dates"], fold["dates"]
    evaluated = _evaluation_inputs(fold)
    mc = [c for c in features if c.startswith("macro::")]
    history = features.loc["1994-01":str(dates.max()), mc]
    cleaned, macro_audit = clean_panel(history, train_dates, method)
    pipe = _numeric_pipeline(pca=True).fit(cleaned.loc[train_dates])
    Z, z = pipe.transform(cleaned.loc[train_dates]), pipe.transform(cleaned.loc[evaluated.index])
    six = lambda a: np.column_stack([a[:, 0], a[:, 0]**3, a[:, 1:4], a[:, 7]])
    ln, point = six(Z), six(z)
    X, x = fold["X"].copy(), evaluated.to_numpy().copy()
    for col, subset in ((7, [0, 1, 3, 4, 5]), (8, list(range(6)))):
        first = LinearRegression().fit(ln[:, subset], fold["y"].mean(1))
        X[:, col], x[:, col] = first.predict(ln[:, subset]), first.predict(point[:, subset])
    generated = candidates.loc["1994-01":str(dates.max()), selected]
    cleaned_c, candidate_audit = clean_panel(generated, train_dates, method)
    imputer, scale = SimpleImputer(strategy="mean"), StandardScaler()
    C = scale.fit_transform(imputer.fit_transform(cleaned_c.loc[train_dates]))
    c = scale.transform(imputer.transform(cleaned_c.loc[evaluated.index]))
    pca = PCA(n_components=5, svd_solver="full").fit(C)
    X[:, 10:], x[:, 10:] = pca.transform(C), pca.transform(c)
    scaler = StandardScaler().fit(X)
    X, x = scaler.transform(X), scaler.transform(x)
    requested = evaluated.index.get_indexer(dates)
    result = {k:copy.deepcopy(v) for k,v in fold.items() if k != "inner_folds"}
    result.update(X=X, x=x[requested])
    blocks = {"L1_Macro":[4, 6, 7, 8, 9], "L1_Full":list(range(10)),
              "L2_PCA":list(range(10, 15)), "L2_Incr":list(range(15))}
    for name, cols in blocks.items():
        result["reference"][name] = LinearRegression().fit(X[:, cols], fold["y"]).predict(x[requested][:, cols])
    lagged = pd.DataFrame(X, index=train_dates).reindex(train_dates-1).to_numpy()
    keep = np.isfinite(lagged).all(1)
    lag_X = np.column_stack([X[keep], lagged[keep]])
    lag_x = np.column_stack([x[requested], x[evaluated.index.get_indexer(dates-1)]])
    lag_scaler = StandardScaler().fit(lag_X)
    result["lagged"].update(X=lag_scaler.transform(lag_X), x=lag_scaler.transform(lag_x),
                            y=fold["y"][keep], train_dates=train_dates[keep])
    result["audit"].update(preprocessing=method, macro_outliers=macro_audit,
        candidate_outliers=candidate_audit, agent_explained5=float(pca.explained_variance_ratio_.sum()),
        threshold_fit_last=str(train_dates.max()), threshold_rule="training median +/- 10 IQR")
    return result
