"""Bounded, literature-led refinements; historical exploration, not preregistration.

Keep the original specifications. Add the Goh forward-spread signals, supervised
scaled PCA, and separately labelled protection against polynomial extrapolation.
"""

from itertools import product
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .config import DATA_DIR, TRAIN_START
from .coursework import reference_features
from .models import _numeric_pipeline

WINDOWS = (None, 120, 240)
BASE_MODELS = ("EH", "FB", "CP", "LN6_raw", "LN8_raw", "LN6_safe", "LN6_safe_R100",
               "LN6_macro_clip", "LN6_cube_clip", "SPCA", "SPCA_raw", "Goh48", "Market", "Core")
PRIMARY = "Core/online"
IQR_BOUND = 6.0


def goh_signals(bonds):
    """Goh et al. (2013), II.A (1)-(2): 4 maturities x 3 short x 4 long MAs.

    Uses forward spreads, strict >, and complete trailing calendar-month windows.
    This is the 48 forward-spread signals, not the unavailable volume component.
    """
    if not isinstance(bonds.index, pd.PeriodIndex) or bonds.index.has_duplicates:
        raise ValueError("Expected unique monthly bond observations")
    bonds = bonds.reindex(pd.period_range(bonds.index.min(), bonds.index.max(), freq="M"))
    rows = {}
    for n, short, long in product(range(2, 6), (3, 6, 9), (18, 24, 30, 36)):
        a = bonds[f"fs{n}"].rolling(short, min_periods=short).mean()
        b = bonds[f"fs{n}"].rolling(long, min_periods=long).mean()
        rows[f"goh::{n}:{short}:{long}"] = (a > b).astype(float).where(a.notna() & b.notna())
    return pd.DataFrame(rows)


def literal_sentiment():
    """Laborda-Olmo eq.(14) inputs, using the downloaded revised ORTH series.

    It cannot certify historical orthogonalization or release vintages. Keep it
    out of the macro-vintage primary strategy and report this sensitivity alone.
    """
    raw = pd.read_excel(Path(DATA_DIR)/"SENTIMENT.xlsx", sheet_name="DATA")
    date = pd.PeriodIndex(pd.to_datetime(raw.yearmo.astype(int).astype(str), format="%Y%m"), freq="M")
    s = pd.Series(pd.to_numeric(raw.SENT_ORTH, errors="coerce").to_numpy(), index=date).sort_index()
    s = s.reindex(pd.period_range(s.index.min(), s.index.max(), freq="M"))
    return pd.DataFrame({"paper_bw": s, "paper_bw_squared": s*s, "paper_bw_change": s.diff()})


def robust_bounds(history, multiple=IQR_BOUND):
    """Explicit project protection: median +/- 6 IQR, fitted on known past X.

    Not claimed to be an exact original-paper outlier rule. Keep zero-IQR columns
    unbounded, since a rare change in a near-constant series is not automatically
    a data error. Realized returns are never clipped.
    """
    q = history.quantile([.25, .5, .75])
    width = multiple*(q.loc[.75]-q.loc[.25])
    width = width.where(width > 1e-12, np.inf)
    return q.loc[.5]-width, q.loc[.5]+width


def ln_design(scores, cube_range=None):
    f1 = scores[:, 0]
    cube_input = f1 if cube_range is None else np.clip(f1, *cube_range)
    return np.column_stack([f1, cube_input**3, scores[:, 1:4], scores[:, 7]])


def two_stage(X, x, y, penalty=0):
    average = y.mean(axis=1)
    if penalty:
        scale = StandardScaler().fit(X)
        X, x = scale.transform(X), scale.transform(x)
        # This labelled extension uses direct multi-output Ridge; a second-stage
        # slope would otherwise undo part of the intended forecast shrinkage.
        return Ridge(alpha=penalty).fit(X, y).predict(x)
    first = LinearRegression().fit(X, average)
    index = first.predict(X).reshape(-1, 1)
    return LinearRegression().fit(index, y).predict(first.predict(x).reshape(-1, 1))


def spca_predict(X, x, y):
    """Huang et al. (2020), section 3.2: slope scaling, then ONE PC.

    X is standardized; slopes and weighted PCA use matured training labels only.
    Do not standardize the weighted columns again: that would erase the weights.
    """
    centered = y.mean(axis=1)-y.mean()
    centered_x = X-X.mean(axis=0)
    denominator = np.sum(centered_x**2, axis=0)
    slopes = np.divide(centered_x.T@centered, denominator,
                       out=np.zeros(X.shape[1]), where=denominator > 1e-12)
    weighted = X*slopes
    if np.linalg.norm(weighted-weighted.mean(axis=0)) < 1e-12:
        return np.tile(y.mean(axis=0), (len(x), 1))
    pc = PCA(n_components=1, svd_solver="full").fit(weighted)
    return LinearRegression().fit(pc.transform(weighted), y).predict(pc.transform(x*slopes))


def goh_predict(X, x, y, history=None):
    """Bai-Ng ICp2 factor count (ceiling eight), then per-maturity subset BIC.

    Goh's two selection steps are distinct: explain the signal panel first, then
    choose predictive factors. All available X may inform unsupervised PCA;
    only matured Y may inform BIC. No volume factor is fabricated.
    """
    history = X if history is None else history
    history = np.asarray(history, dtype=float)
    history = history[np.isfinite(history).all(axis=1)]
    scale = StandardScaler().fit(history)
    a, b = scale.transform(np.asarray(X)), scale.transform(np.asarray(x))
    H = scale.transform(history)
    T, N = H.shape
    pc = PCA(n_components=min(8, N-1, T-1), svd_solver="full").fit(H)
    total = np.sum((H-H.mean(axis=0))**2)/(T*N)
    residual = np.maximum(total-np.r_[0, np.cumsum(pc.singular_values_**2)/(T*N)], 1e-12)
    count = int(np.argmin(np.log(residual)+np.arange(len(residual))*(N+T)/(N*T)*np.log(min(N, T))))
    if count == 0:
        return np.tile(y.mean(axis=0), (len(x), 1)), []
    Z, z = pc.transform(a)[:, :count], pc.transform(b)[:, :count]
    mean_Z, mean_y = Z.mean(axis=0), y.mean(axis=0)
    C, Y = Z-mean_Z, y-mean_y
    gram, cross, sum_y2 = C.T@C, C.T@Y, np.sum(Y**2, axis=0)
    best, prediction, selected = np.full(y.shape[1], np.inf), np.empty((len(x), y.shape[1])), [None]*y.shape[1]
    for mask in product((False, True), repeat=count):
        selected_columns = np.flatnonzero(mask)
        G = gram[np.ix_(selected_columns, selected_columns)]
        xy = cross[selected_columns]
        coefficient, _, rank, _ = np.linalg.lstsq(G, xy, rcond=1e-12)
        sse = np.maximum(sum_y2-2*np.sum(coefficient*xy, axis=0)
                         +np.sum(coefficient*(G@coefficient), axis=0), 1e-12)
        bic = len(y)*np.log(sse/len(y))+(rank+1)*np.log(len(y))
        improves = bic < best
        forecast = mean_y+(z[:, mask]-mean_Z[list(mask)])@coefficient
        prediction[:, improves] = forecast[:, improves]
        for j in np.flatnonzero(improves):
            selected[j] = (np.flatnonzero(mask)+1).tolist()
        best = np.minimum(best, bic)
    return prediction, selected


def fit_refinements(train, current, history, macro_columns, targets, *, iqr_bound=IQR_BOUND):
    y = train[targets].to_numpy()
    mc = macro_columns
    # Bound estimation excludes the forecast row itself. Only known predictors,
    # not future target magnitudes, determine the transformation.
    before = history.loc[history.index < history.index.max(), mc]
    low, high = robust_bounds(before, iqr_bound)
    clean_history = history[mc].clip(low, high, axis=1)
    clean_train = train[mc].clip(low, high, axis=1)
    clean_current = current[mc].clip(low, high, axis=1)
    raw_pc = _numeric_pipeline(pca=True).fit(history[mc])
    safe_pc = _numeric_pipeline(pca=True).fit(clean_history)
    Z, z = raw_pc.transform(train[mc]), raw_pc.transform(current[mc])
    Q, q = safe_pc.transform(clean_train), safe_pc.transform(clean_current)
    safe_range = (Q[:, 0].min(), Q[:, 0].max())
    raw_range = (Z[:, 0].min(), Z[:, 0].max())
    predictions = {"EH": np.tile(y.mean(axis=0), (len(current), 1))}
    predictions["FB"] = np.column_stack([
        LinearRegression().fit(train[[f"fs{n}"]], y[:, n-2]).predict(current[[f"fs{n}"]])
        for n in range(2, 6)])
    cols = [f"f{n}" for n in range(1, 6)]
    predictions["CP"] = two_stage(train[cols].to_numpy(), current[cols].to_numpy(), y)
    for name, A, a, bounds, penalty in [
        ("LN6_raw", Z, z, None, 0), ("LN6_safe", Q, q, safe_range, 0),
        ("LN6_safe_R100", Q, q, safe_range, 100),
        ("LN6_macro_clip", Q, q, None, 0), ("LN6_cube_clip", Z, z, raw_range, 0)]:
        predictions[name] = two_stage(ln_design(A, bounds), ln_design(a, bounds), y, penalty)
    predictions["LN8_raw"] = two_stage(Z, z, y)
    for name, A, a, H in [("SPCA", clean_train, clean_current, clean_history),
                          ("SPCA_raw", train[mc], current[mc], history[mc])]:
        pipe = _numeric_pipeline().fit(H)
        predictions[name] = spca_predict(pipe.transform(A), pipe.transform(a), y)
    tech = [c for c in train if c.startswith("goh::")]
    predictions["Goh48"], selected = goh_predict(train[tech], current[tech], y, history[tech])
    predictions["Market"] = (predictions["CP"]+predictions["Goh48"])/2
    macro = (predictions["LN6_safe"]+predictions["SPCA"])/2
    predictions["Core"] = (predictions["CP"]+macro+predictions["Goh48"])/3
    bw = ["paper_bw", "paper_bw_squared", "paper_bw_change"]
    bw_train = train.dropna(subset=bw) if set(bw) <= set(train) else train.iloc[:0]
    if len(bw_train) >= 120 and current[bw].notna().all(axis=None):
        # EH_BW uses exactly the BW regression's shorter eligible history.
        Y = bw_train[targets].to_numpy()
        predictions["EH_BW"] = np.tile(Y.mean(axis=0), (len(current), 1))
        predictions["BW3"] = two_stage(bw_train[bw].to_numpy(), current[bw].to_numpy(), Y)
        predictions["Core_BW"] = (3*predictions["Core"]+predictions["BW3"])/4
    info = {"goh_selected_pcs": str(selected),
            "macro_current_clipped": int(((current[mc] < low) | (current[mc] > high)).sum(axis=1).iloc[0]),
            "F1_raw": z[0, 0], "F1_safe": q[0, 0],
            "safe_cube_lower": safe_range[0], "safe_cube_upper": safe_range[1]}
    return predictions, info


def refinement_forecasts(bonds, macro, *, horizon=1, kind="simple", sentiment=None,
                         archive=None, start="1980-01", end="2026-01", lag=2,
                         windows=WINDOWS, iqr_bound=IQR_BOUND, verbose=True):
    prefix = "mx" if horizon == 1 else ("sx" if kind == "simple" else "rx")
    targets = [f"{prefix}{n}" for n in range(2, 6)]
    tech = goh_signals(bonds)
    static = reference_features(bonds, macro, lag).join(tech)
    if sentiment is not None:
        static = static.join(sentiment.reindex(bonds.index).shift(lag))
    values, audits = [], []
    for date in static.loc[start:end].index:
        features, vintage = static, "revised"
        if archive is not None:
            vintage, panel = archive.at(date)
            if vintage >= date:
                raise ValueError("Macro vintage must predate the forecast month")
            features = reference_features(bonds, panel, lag).join(tech)
        mc = [c for c in features if c.startswith("macro::")]
        required = [f"f{n}" for n in range(1, 6)]+list(tech)
        if not features.loc[date, required].notna().all() or not features.loc[date, mc].notna().any():
            continue
        data = features.join(bonds[targets])
        eligible = data[required+targets].notna().all(axis=1) & data[mc].notna().any(axis=1)
        for window in windows:
            mask = (data.index >= TRAIN_START) & (data.index+horizon < date) & eligible
            history_mask = (features.index >= TRAIN_START) & (features.index <= date)
            if window:
                mask &= data.index >= date-horizon-window
                history_mask &= features.index > date-window
            train, history = data.loc[mask], features.loc[history_mask]
            history = history.loc[history[mc].notna().any(axis=1)]
            if len(train) < 120:
                continue
            predictions, info = fit_refinements(train, data.loc[[date]], history, mc, targets, iqr_bound=iqr_bound)
            label = "exp" if window is None else f"w{window}"
            for name, prediction in predictions.items():
                values.append({"ym": str(date), "model": f"{name}/{label}",
                               **{f"mu{n}": prediction[0, j] for j, n in enumerate(range(2, 6))}})
            audits.append({"ym": str(date), "window": label, "horizon": horizon, "target": kind,
                           "train_n": len(train), "last_label_end": str(train.index.max()+horizon),
                           "macro_reference": str(date-lag), "vintage": str(vintage), **info})
        if verbose and date.month == 1 and date.year % 10 == 0:
            print(f"  {kind}, horizon {horizon}: {date}", flush=True)
    return pd.DataFrame(values), pd.DataFrame(audits)


def causal_shrinkage(predictions, actual, *, horizon, min_errors=60, lookback=120):
    """Equal windows, then forecast-level least-squares shrinkage towards EH.

    Lambda = clip(sum((y-EH)*deviation)/sum(deviation²),0,1), using only past
    stored forecasts with s+h<t. No best-window or best-model OOS ranking.
    """
    p = predictions.copy()
    p["ym"] = pd.PeriodIndex(p.ym, freq="M")
    cols = [f"mu{n}" for n in range(2, 6)]
    indexed = {m: g.set_index("ym")[cols].sort_index() for m, g in p.groupby("model")}
    names = [name for name in BASE_MODELS if all(f"{name}/{w}" in indexed for w in ("exp", "w120", "w240"))]
    dates = indexed["EH/exp"].index
    records, audits = [], []
    eh = np.mean([indexed[f"EH/{w}"].loc[dates].to_numpy() for w in ("exp", "w120", "w240")], axis=0)
    for name in names:
        blend = np.mean([indexed[f"{name}/{w}"].loc[dates].to_numpy() for w in ("exp", "w120", "w240")], axis=0)
        deviation = (blend-eh).mean(axis=1)
        residual = actual.reindex(dates).to_numpy()-eh.mean(axis=1)
        for i, date in enumerate(dates):
            mask = (dates+horizon < date) & (dates >= date-horizon-lookback) & np.isfinite(residual)
            denominator = deviation[mask]@deviation[mask]
            coefficient = (float(np.clip(deviation[mask]@residual[mask]/denominator, 0, 1))
                           if mask.sum() >= min_errors and denominator > 1e-12 else 0.)
            for suffix, forecast in [("blend", blend[i]), ("half", eh[i]+.5*(blend[i]-eh[i])),
                                     ("online", eh[i]+coefficient*(blend[i]-eh[i]))]:
                records.append({"ym": str(date), "model": f"{name}/{suffix}", **dict(zip(cols, forecast))})
            audits.append({"ym": str(date), "model": name, "lambda": coefficient,
                           "matured_errors": int(mask.sum()),
                           "last_error_end": str(dates[mask].max()+horizon) if mask.any() else ""})
    return pd.concat([predictions, pd.DataFrame(records)], ignore_index=True), pd.DataFrame(audits)
