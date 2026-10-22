"""Required FB/CP/LN reference specifications, separate from strategy extensions."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from .config import TRAIN_START
from .models import _numeric_pipeline, available_features


REFERENCE_MODELS = ("EH", "FB_n", "CP_2stage", "LN6_2stage", "LN5_2stage", "LN8_2stage",
                    "CP_LN5_2stage", "CP_LN5_direct", "CP_LN6_2stage", "CP_LN_raw")
NESTED_AVERAGE_OLS = {"CP_2stage", "LN6_2stage", "LN5_2stage", "LN8_2stage", "CP_LN_raw"}


def common_index(X, y, current, *, second_intercept=True):
    """OLS average-return index followed by maturity loadings.

    CP (2005), eq. (2), restricts the second stage to b_n * index: the
    first-stage index already contains a constant. LN Table 2 includes a
    separate intercept. This function is shared by replication and forecasts.
    """
    y = np.asarray(y, dtype=float)
    first = LinearRegression().fit(X, y.mean(axis=1))
    index = first.predict(X).reshape(-1, 1)
    now = first.predict(current).reshape(-1, 1)
    second = LinearRegression(fit_intercept=second_intercept).fit(index, y)
    prediction = second.predict(now)
    np.testing.assert_allclose(prediction.mean(axis=1), now.ravel(), atol=1e-7, rtol=1e-7)
    return prediction, first, second


def reference_features(bonds, macro, lag=2):
    features = available_features(bonds, macro, publication_lag=lag)
    return features.join(bonds[[f"fs{n}" for n in range(2, 6)]])


def fit_reference(train, current, macro_columns, targets, *, macro_history=None):
    """Fit first-stage indices and then four maturity-specific second stages.

    Supervised stages use matured labels only. Unsupervised PCA may additionally
    use macro_history known at forecast formation, even without matured labels.
    LN6 and CP's companion LN5 follow LN (2009), equations (7) and (8).
    """
    y = train[targets].to_numpy()
    average = y.mean(axis=1)
    cp = [f"f{n}" for n in range(1, 6)]
    pipeline = _numeric_pipeline(pca=True)
    pipeline.fit((train if macro_history is None else macro_history)[macro_columns])
    Z = pipeline.transform(train[macro_columns])
    z = pipeline.transform(current[macro_columns])
    def six(scores):
        return np.column_stack([scores[:, 0], scores[:, 0]**3, scores[:, 1:4], scores[:, 7]])
    designs = {"CP": (train[cp].to_numpy(), current[cp].to_numpy(), cp),
               "LN6": (six(Z), six(z), ["F1", "F1_cubed", "F2", "F3", "F4", "F8"]),
               "LN8": (Z, z, [f"F{k}" for k in range(1, 9)])}
    # LN (2009), p.5038: F2 is omitted when conditioning on CP.
    keep_five = [0, 1, 3, 4, 5]
    designs["LN5"] = (designs["LN6"][0][:, keep_five], designs["LN6"][1][:, keep_five],
                       [designs["LN6"][2][j] for j in keep_five])
    designs["CP_LN_raw"] = (np.hstack([designs["CP"][0], designs["LN6"][0]]),
                             np.hstack([designs["CP"][1], designs["LN6"][1]]), cp+designs["LN6"][2])
    # Each maturity has its own constant-only benchmark, not one basket mean.
    predictions = {"EH": np.tile(y.mean(axis=0), (len(current), 1))}
    fb = []
    for j, n in enumerate(range(2, 6)):
        fit = LinearRegression().fit(train[[f"fs{n}"]], y[:, j])
        fb.append(fit.predict(current[[f"fs{n}"]]))
    predictions["FB_n"] = np.column_stack(fb)
    indices, current_indices, first_stages = {}, {}, {}
    for name, (X, x, columns) in designs.items():
        prediction, fit, second = common_index(X, y, x, second_intercept=name != "CP")
        first_stages[name] = {"X": X, "names": columns, "fit": fit}
        if name == "CP_LN_raw":
            predictions[name] = LinearRegression().fit(X, y).predict(x)
            continue
        indices[name] = fit.predict(X).reshape(-1, 1)
        current_indices[name] = fit.predict(x).reshape(-1, 1)
        predictions[name+"_2stage"] = prediction
        # Averaging the second-stage fits must reproduce the first-stage forecast.
        np.testing.assert_allclose(predictions[name+"_2stage"].mean(axis=1),
                                   fit.predict(x), rtol=1e-7, atol=1e-7)
    for macro_name in ("LN5", "LN6"):
        X = np.hstack([indices["CP"], indices[macro_name]])
        x = np.hstack([current_indices["CP"], current_indices[macro_name]])
        predictions[f"CP_{macro_name}_2stage"] = LinearRegression().fit(X, y).predict(x)
    # LN Table 3 also tests the five macro terms + CP directly, without compressing
    # the macro terms to a common index across maturities.
    X = np.hstack([indices["CP"], designs["LN5"][0]])
    x = np.hstack([current_indices["CP"], designs["LN5"][1]])
    predictions["CP_LN5_direct"] = LinearRegression().fit(X, y).predict(x)
    return predictions, first_stages, pipeline


def reference_forecasts(bonds, macro, *, start="1985-01", end="2024-02", windows=(None, 240), verbose=True):
    features = reference_features(bonds, macro)
    mc = [c for c in features if c.startswith("macro::")]
    market = [f"f{n}" for n in range(1, 6)]+[f"fs{n}" for n in range(2, 6)]
    target_cols = [f"{prefix}{n}" for prefix in ("rx", "sx") for n in range(2, 6)]
    data = features.join(bonds[target_cols])
    eligible = data[market+target_cols].notna().all(axis=1) & data[mc].notna().any(axis=1)
    data = data.loc[eligible]
    predictions, audit = [], []
    for date in data.loc[start:end].index:
        for window in windows:
            mask = (data.index >= TRAIN_START) & (data.index+12 < date)
            if window:
                mask &= data.index >= date-12-window
            train = data.loc[mask]
            if len(train) < 120:
                continue
            label = "exp" if window is None else f"w{window}"
            # No return labels enter PCA. Features already contain t-2 macro data;
            # all such available observations, including the forecast row, are legal.
            history_mask = (features.index >= TRAIN_START) & (features.index <= date)
            if window:
                history_mask &= features.index > date-window
            history = features.loc[history_mask]
            history = history.loc[history[mc].notna().any(axis=1)]
            for kind, prefix in [("log", "rx"), ("simple", "sx")]:
                targets = [f"{prefix}{n}" for n in range(2, 6)]
                fits, stages, pipe = fit_reference(train, data.loc[[date]], mc, targets,
                                                   macro_history=history)
                for name in REFERENCE_MODELS:
                    predictions.append({"ym": str(date), "model": name, "window": label, "target": kind,
                                        **{f"pred{n}": fits[name][0, j] for j, n in enumerate(range(2, 6))}})
            z_train = pipe.transform(train[mc])[:, 0]
            z_now = pipe.transform(data.loc[[date], mc])[0, 0]
            audit.append({"ym": str(date), "window": label, "train_n": len(train),
                          "train_start": str(train.index.min()), "last_label_end": str(train.index.max()+12),
                          "macro_reference": str(date-2), "pca_last_reference": str(history.index.max()-2),
                          "pca_observations": len(history), "F1_current": z_now,
                          "F1_training_min": z_train.min(), "F1_training_max": z_train.max()})
        if verbose and date.month == 1 and date.year % 10 == 0:
            print(f"  required two-stage references through {date}", flush=True)
    return pd.DataFrame(predictions), pd.DataFrame(audit)


def reference_vintage_forecasts(bonds, archive, *, start="2000-01", end="2024-02", verbose=True):
    """Rebuild each required specification using the prior-month macro snapshot.

    Same t-2 reference lag and t-1 vintage convention as the existing vintage
    experiment. Latest/revised values never fill a historical snapshot.
    """
    predictions, audit = [], []
    for date in pd.period_range(start, end, freq="M"):
        vintage, panel = archive.at(date)
        if vintage >= date:
            raise ValueError("Historical macro snapshot must predate the forecast month")
        current = panel.loc[panel.index == date-2]
        if current.empty or not current.notna().any(axis=None):
            raise ValueError(f"No published macro input at {date-2}")
        p, a = reference_forecasts(bonds, panel, start=str(date), end=str(date), windows=(None,), verbose=False)
        if p.empty:
            raise ValueError(f"Insufficient reference history at {date}")
        a["vintage"] = str(vintage)
        predictions.append(p); audit.append(a)
        if verbose and date.month == 1 and date.year % 5 == 0:
            print(f"  vintage LN5/LN6 reference specifications through {date}", flush=True)
    return pd.concat(predictions, ignore_index=True), pd.concat(audit, ignore_index=True)
