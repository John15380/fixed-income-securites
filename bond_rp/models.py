"""
OOS prediction functions — all return ym-indexed pd.Series.

Training rows precede each forecast date, but their forward 12-month labels
are not embargoed until maturity. These are historical research estimates,
not strictly real-time forecasts. See README.md for additional limitations.
LN uses PCA loadings fixed on 1964-1989 and recursive regression coefficients.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

from .config import MATS, OOS_START


# ── helpers ────────────────────────────────────────────────────────────────────

def _lstsq_predict(X_train, y_train, x_pred):
    coef, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    return float(x_pred @ coef), coef


def _train_slice(data, ym, window):
    """Select preceding rows; forward labels are NOT embargoed until maturity."""
    if window is None:
        return data[data.index < ym]
    return data[(data.index < ym) & (data.index >= ym - window)]


def _impute_standardize(X_train: np.ndarray, x_pred: np.ndarray):
    """Mean-impute NaN then standardize; apply same transform to the prediction row."""
    col_means = np.nanmean(X_train, axis=0)
    col_stds  = np.nanstd(X_train, axis=0, ddof=1)
    col_stds[col_stds == 0] = 1.0

    X_imp = X_train.copy()
    for j in range(X_imp.shape[1]):
        mask = np.isnan(X_imp[:, j])
        if mask.any():
            X_imp[mask, j] = col_means[j]

    x_imp = x_pred.copy()
    for j in range(len(x_imp)):
        if np.isnan(x_imp[j]):
            x_imp[j] = col_means[j]

    return (X_imp - col_means) / col_stds, (x_imp - col_means) / col_stds


# ── benchmark ──────────────────────────────────────────────────────────────────

def oos_eh(fb_ym: pd.DataFrame, y_col: str = "rx_bar",
           oos_start=OOS_START, min_obs: int = 120) -> pd.Series:
    """Expanding historical mean (EH benchmark)."""
    s = fb_ym[y_col].dropna()
    preds = {}
    for ym in s[s.index >= oos_start].index:
        hist = s[s.index < ym]
        if len(hist) >= min_obs:
            preds[ym] = hist.mean()
    return pd.Series(preds, name="EH")


# ── single-factor models ───────────────────────────────────────────────────────

def oos_fb(fb_ym: pd.DataFrame, y_col: str = "rx_bar",
           oos_start=OOS_START, min_obs: int = 120,
           window=None) -> pd.Series:
    """Fama-Bliss: forward spread → rx_bar (recursive/rolling OLS)."""
    data = fb_ym[[y_col, "fs_avg"]].dropna()
    preds = {}
    for ym in data[data.index >= oos_start].index:
        train = _train_slice(data, ym, window)
        if len(train) < min_obs:
            continue
        X_tr = np.column_stack([np.ones(len(train)), train["fs_avg"].values])
        x_pr = np.array([1.0, data.loc[ym, "fs_avg"]])
        preds[ym], _ = _lstsq_predict(X_tr, train[y_col].values, x_pr)
    return pd.Series(preds, name="FB")


def oos_cp(fb_ym: pd.DataFrame, y_col: str = "rx_bar",
           oos_start=OOS_START, min_obs: int = 120,
           window=None) -> pd.Series:
    """Cochrane-Piazzesi two-step: tent factor from forward rates."""
    F_cols = ["f1", "f2", "f3", "f4", "f5"]
    all_rx = [f"rx{n}" for n in MATS]
    data   = fb_ym[[y_col] + all_rx + F_cols].dropna().copy()
    data["_rxbar"] = data[all_rx].mean(axis=1)

    preds = {}
    for ym in data[data.index >= oos_start].index:
        train = _train_slice(data, ym, window)
        if len(train) < min_obs:
            continue
        X1 = np.column_stack([np.ones(len(train))] + [train[c].values for c in F_cols])
        coef_g, *_ = np.linalg.lstsq(X1, train["_rxbar"].values, rcond=None)

        f_t  = np.array([1.0] + [data.loc[ym, c] for c in F_cols])
        cp_t = float(f_t @ coef_g)
        cp_tr = X1 @ coef_g

        X2 = np.column_stack([np.ones(len(cp_tr)), cp_tr])
        x2_pr = np.array([1.0, cp_t])
        preds[ym], _ = _lstsq_predict(X2, train[y_col].values, x2_pr)
    return pd.Series(preds, name="CP")


def oos_ln(fb_ym: pd.DataFrame, df_factors: pd.DataFrame,
           y_col: str = "rx_bar", oos_start=OOS_START,
           min_obs: int = 120, window=None) -> pd.Series:
    """
    Ludvigson-Ng two-step: macro PCA composite factor.
    PCA loadings are fixed (pre-OOS window); only δ is updated recursively.
    """
    F6_cols = ["F1", "F1_cube", "F2", "F3", "F4", "F8"]
    all_rx  = [f"rx{n}" for n in MATS]
    fb_clean = fb_ym[[y_col] + all_rx].dropna().copy()
    fb_clean["_rxbar"] = fb_clean[all_rx].mean(axis=1)
    data = fb_clean.join(df_factors[F6_cols], how="inner").dropna()

    preds = {}
    for ym in data[data.index >= oos_start].index:
        train = _train_slice(data, ym, window)
        if len(train) < min_obs:
            continue
        X1 = np.column_stack([np.ones(len(train))] + [train[c].values for c in F6_cols])
        coef_d, *_ = np.linalg.lstsq(X1, train["_rxbar"].values, rcond=None)

        f6_t  = np.array([1.0] + [data.loc[ym, c] for c in F6_cols])
        f6t_t = float(f6_t @ coef_d)
        f6t_tr = X1 @ coef_d

        X2 = np.column_stack([np.ones(len(f6t_tr)), f6t_tr])
        x2_pr = np.array([1.0, f6t_t])
        preds[ym], _ = _lstsq_predict(X2, train[y_col].values, x2_pr)
    return pd.Series(preds, name="LN")


def oos_ln_lasso(fb_ym: pd.DataFrame, df_factors: pd.DataFrame,
                 y_col: str = "rx_bar", oos_start=OOS_START,
                 min_obs: int = 120, window=None,
                 return_selection: bool = False):
    """
    LN with recursive LASSO feature selection instead of a fixed subset.
    This addresses feature selection only; forward labels still need an embargo.

    At each OOS step, LassoCV selects from {F1,...,F8, F1³} using only
    training data strictly before the forecast date. Selected features are
    then used in the same two-step OLS composite as oos_ln.

    Compared to oos_ln (which hardcodes F6_cols chosen by full-sample BIC):
      - Feature selection is repeated within each historical training sample.
        Forward-label availability and time-ordered CV remain unresolved.
      - Candidate pool is expanded to all 9 features (vs. hardcoded 6).

    When return_selection=True, returns (Series, DataFrame) where the
    DataFrame records which features were selected at each OOS date.
    """
    F_all  = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F1_cube"]
    all_rx = [f"rx{n}" for n in MATS]
    fb_clean = fb_ym[[y_col] + all_rx].dropna().copy()
    fb_clean["_rxbar"] = fb_clean[all_rx].mean(axis=1)
    data = fb_clean.join(df_factors[F_all], how="inner").dropna()

    preds       = {}
    sel_records = []

    for ym in data[data.index >= oos_start].index:
        train = _train_slice(data, ym, window)
        if len(train) < min_obs:
            continue

        X1 = train[F_all].values
        y1 = train["_rxbar"].values

        scaler = StandardScaler()
        X1s    = scaler.fit_transform(X1)

        lasso = LassoCV(cv=5, max_iter=5000, random_state=42, n_jobs=1)
        lasso.fit(X1s, y1)

        mask = np.abs(lasso.coef_) > 1e-10
        if not mask.any():
            mask[:] = True  # fallback: use all features

        sel_cols = [F_all[i] for i, m in enumerate(mask) if m]
        if return_selection:
            sel_records.append({"ym": ym, **{c: int(c in sel_cols) for c in F_all}})

        X1_sel = np.column_stack([np.ones(len(train))] + [train[c].values for c in sel_cols])
        coef_d, *_ = np.linalg.lstsq(X1_sel, y1, rcond=None)

        f_t    = np.array([1.0] + [data.loc[ym, c] for c in sel_cols])
        f6t_t  = float(f_t @ coef_d)
        f6t_tr = X1_sel @ coef_d

        X2     = np.column_stack([np.ones(len(f6t_tr)), f6t_tr])
        x2_pr  = np.array([1.0, f6t_t])
        preds[ym], _ = _lstsq_predict(X2, train[y_col].values, x2_pr)

    series = pd.Series(preds, name="LN_LASSO")
    if return_selection:
        sel_df = (pd.DataFrame(sel_records).set_index("ym")
                  if sel_records else pd.DataFrame(columns=F_all))
        return series, sel_df
    return series


def oos_ti(fb_ym: pd.DataFrame, y_col: str = "rx_bar",
           oos_start=OOS_START, min_obs: int = 120,
           window=None, ti_col: str = "TI_avg") -> pd.Series:
    """Goh et al. (2013): bond log-price MA momentum."""
    data = fb_ym[[y_col, ti_col]].dropna()
    preds = {}
    for ym in data[data.index >= oos_start].index:
        train = _train_slice(data, ym, window)
        if len(train) < min_obs:
            continue
        X_tr = np.column_stack([np.ones(len(train)), train[ti_col].values])
        x_pr = np.array([1.0, data.loc[ym, ti_col]])
        preds[ym], _ = _lstsq_predict(X_tr, train[y_col].values, x_pr)
    return pd.Series(preds, name="TI")


def oos_bw(fb_ym: pd.DataFrame, df_sent: pd.DataFrame,
           y_col: str = "rx_bar", oos_start=OOS_START,
           min_obs: int = 120, window=None) -> pd.Series:
    """Laborda-Olmo (2014) Eq. 14: sentiment composite factor."""
    S_cols  = ["SENT_ORTH", "SENT_SQ", "dSENT"]
    all_rx  = [f"rx{n}" for n in MATS]
    fb_clean = fb_ym[[y_col] + all_rx].dropna().copy()
    fb_clean["_rxbar"] = fb_clean[all_rx].mean(axis=1)
    data = fb_clean.join(df_sent[S_cols], how="inner").dropna()

    preds = {}
    for ym in data[data.index >= oos_start].index:
        train = _train_slice(data, ym, window)
        if len(train) < min_obs:
            continue
        X1 = np.column_stack([np.ones(len(train))] + [train[c].values for c in S_cols])
        coef_phi, *_ = np.linalg.lstsq(X1, train["_rxbar"].values, rcond=None)

        s_t  = np.array([1.0] + [data.loc[ym, c] for c in S_cols])
        bw_t = float(s_t @ coef_phi)
        bw_tr = X1 @ coef_phi

        X2 = np.column_stack([np.ones(len(bw_tr)), bw_tr])
        x2_pr = np.array([1.0, bw_t])
        preds[ym], _ = _lstsq_predict(X2, train[y_col].values, x2_pr)
    return pd.Series(preds, name="BW")


# ── combined models ────────────────────────────────────────────────────────────

def oos_cpln(fb_ym: pd.DataFrame, df_factors: pd.DataFrame,
             y_col: str = "rx_bar", oos_start=OOS_START,
             min_obs: int = 120, window=None) -> pd.Series:
    """CP + LN joint second stage."""
    F_cols  = ["f1", "f2", "f3", "f4", "f5"]
    F6_cols = ["F1", "F1_cube", "F2", "F3", "F4", "F8"]
    all_rx  = [f"rx{n}" for n in MATS]
    fb_clean = fb_ym[[y_col] + all_rx + F_cols].dropna().copy()
    fb_clean["_rxbar"] = fb_clean[all_rx].mean(axis=1)
    data = fb_clean.join(df_factors[F6_cols], how="inner").dropna()

    preds = {}
    for ym in data[data.index >= oos_start].index:
        train = _train_slice(data, ym, window)
        if len(train) < min_obs:
            continue
        X1cp = np.column_stack([np.ones(len(train))] + [train[c].values for c in F_cols])
        coef_g, *_ = np.linalg.lstsq(X1cp, train["_rxbar"].values, rcond=None)
        cp_t  = float(np.array([1.0] + [data.loc[ym, c] for c in F_cols]) @ coef_g)
        cp_tr = X1cp @ coef_g

        X1ln = np.column_stack([np.ones(len(train))] + [train[c].values for c in F6_cols])
        coef_d, *_ = np.linalg.lstsq(X1ln, train["_rxbar"].values, rcond=None)
        f6t_t  = float(np.array([1.0] + [data.loc[ym, c] for c in F6_cols]) @ coef_d)
        f6t_tr = X1ln @ coef_d

        X2 = np.column_stack([np.ones(len(train)), cp_tr, f6t_tr])
        x2_pr = np.array([1.0, cp_t, f6t_t])
        preds[ym], _ = _lstsq_predict(X2, train[y_col].values, x2_pr)
    return pd.Series(preds, name="CPLN")


def oos_lnbw(fb_ym: pd.DataFrame, df_factors: pd.DataFrame,
             df_sent: pd.DataFrame, y_col: str = "rx_bar",
             oos_start=OOS_START, min_obs: int = 120,
             window=None) -> pd.Series:
    """LN + BW joint second stage."""
    F6_cols = ["F1", "F1_cube", "F2", "F3", "F4", "F8"]
    S_cols  = ["SENT_ORTH", "SENT_SQ", "dSENT"]
    all_rx  = [f"rx{n}" for n in MATS]
    fb_clean = fb_ym[[y_col] + all_rx].dropna().copy()
    fb_clean["_rxbar"] = fb_clean[all_rx].mean(axis=1)
    data = (fb_clean
            .join(df_factors[F6_cols], how="inner")
            .join(df_sent[S_cols],     how="inner")
            .dropna())

    preds = {}
    for ym in data[data.index >= oos_start].index:
        train = _train_slice(data, ym, window)
        if len(train) < min_obs:
            continue
        X1ln = np.column_stack([np.ones(len(train))] + [train[c].values for c in F6_cols])
        coef_d, *_ = np.linalg.lstsq(X1ln, train["_rxbar"].values, rcond=None)
        f6t_t  = float(np.array([1.0] + [data.loc[ym, c] for c in F6_cols]) @ coef_d)
        f6t_tr = X1ln @ coef_d

        X1bw = np.column_stack([np.ones(len(train))] + [train[c].values for c in S_cols])
        coef_phi, *_ = np.linalg.lstsq(X1bw, train["_rxbar"].values, rcond=None)
        bw_t  = float(np.array([1.0] + [data.loc[ym, c] for c in S_cols]) @ coef_phi)
        bw_tr = X1bw @ coef_phi

        X2 = np.column_stack([np.ones(len(train)), f6t_tr, bw_tr])
        x2_pr = np.array([1.0, f6t_t, bw_t])
        preds[ym], _ = _lstsq_predict(X2, train[y_col].values, x2_pr)
    return pd.Series(preds, name="LNBW")


# ── additional combinations ────────────────────────────────────────────────────

def oos_cpti(fb_ym: pd.DataFrame, y_col: str = "rx_bar",
             oos_start=OOS_START, min_obs: int = 120,
             window=None, ti_col: str = "TI_avg") -> pd.Series:
    """CP + TI joint second stage (new combination)."""
    F_cols  = ["f1", "f2", "f3", "f4", "f5"]
    all_rx  = [f"rx{n}" for n in MATS]
    fb_clean = fb_ym[[y_col] + all_rx + F_cols + [ti_col]].dropna().copy()
    fb_clean["_rxbar"] = fb_clean[all_rx].mean(axis=1)
    data = fb_clean.dropna()

    preds = {}
    for ym in data[data.index >= oos_start].index:
        train = _train_slice(data, ym, window)
        if len(train) < min_obs:
            continue
        X1 = np.column_stack([np.ones(len(train))] + [train[c].values for c in F_cols])
        coef_g, *_ = np.linalg.lstsq(X1, train["_rxbar"].values, rcond=None)
        cp_t  = float(np.array([1.0] + [data.loc[ym, c] for c in F_cols]) @ coef_g)
        cp_tr = X1 @ coef_g
        X2    = np.column_stack([np.ones(len(train)), cp_tr, train[ti_col].values])
        x2_pr = np.array([1.0, cp_t, data.loc[ym, ti_col]])
        preds[ym], _ = _lstsq_predict(X2, train[y_col].values, x2_pr)
    return pd.Series(preds, name="CPTI")


def oos_cpbw(fb_ym: pd.DataFrame, df_sent: pd.DataFrame,
             y_col: str = "rx_bar", oos_start=OOS_START,
             min_obs: int = 120, window=None) -> pd.Series:
    """CP + BW joint second stage (new combination)."""
    F_cols  = ["f1", "f2", "f3", "f4", "f5"]
    S_cols  = ["SENT_ORTH", "SENT_SQ", "dSENT"]
    all_rx  = [f"rx{n}" for n in MATS]
    fb_clean = fb_ym[[y_col] + all_rx + F_cols].dropna().copy()
    fb_clean["_rxbar"] = fb_clean[all_rx].mean(axis=1)
    data = fb_clean.join(df_sent[S_cols], how="inner").dropna()

    preds = {}
    for ym in data[data.index >= oos_start].index:
        train = _train_slice(data, ym, window)
        if len(train) < min_obs:
            continue
        X1cp = np.column_stack([np.ones(len(train))] + [train[c].values for c in F_cols])
        coef_g, *_ = np.linalg.lstsq(X1cp, train["_rxbar"].values, rcond=None)
        cp_t  = float(np.array([1.0] + [data.loc[ym, c] for c in F_cols]) @ coef_g)
        cp_tr = X1cp @ coef_g
        X1bw  = np.column_stack([np.ones(len(train))] + [train[c].values for c in S_cols])
        coef_phi, *_ = np.linalg.lstsq(X1bw, train["_rxbar"].values, rcond=None)
        bw_t  = float(np.array([1.0] + [data.loc[ym, c] for c in S_cols]) @ coef_phi)
        bw_tr = X1bw @ coef_phi
        X2    = np.column_stack([np.ones(len(train)), cp_tr, bw_tr])
        x2_pr = np.array([1.0, cp_t, bw_t])
        preds[ym], _ = _lstsq_predict(X2, train[y_col].values, x2_pr)
    return pd.Series(preds, name="CPBW")


def oos_lnti(fb_ym: pd.DataFrame, df_factors: pd.DataFrame,
             y_col: str = "rx_bar", oos_start=OOS_START,
             min_obs: int = 120, window=None,
             ti_col: str = "TI_avg") -> pd.Series:
    """LN + TI joint second stage (new combination)."""
    F6_cols = ["F1", "F1_cube", "F2", "F3", "F4", "F8"]
    all_rx  = [f"rx{n}" for n in MATS]
    fb_clean = fb_ym[[y_col] + all_rx + [ti_col]].dropna().copy()
    fb_clean["_rxbar"] = fb_clean[all_rx].mean(axis=1)
    data = fb_clean.join(df_factors[F6_cols], how="inner").dropna()

    preds = {}
    for ym in data[data.index >= oos_start].index:
        train = _train_slice(data, ym, window)
        if len(train) < min_obs:
            continue
        X1    = np.column_stack([np.ones(len(train))] + [train[c].values for c in F6_cols])
        coef_d, *_ = np.linalg.lstsq(X1, train["_rxbar"].values, rcond=None)
        f6t_t  = float(np.array([1.0] + [data.loc[ym, c] for c in F6_cols]) @ coef_d)
        f6t_tr = X1 @ coef_d
        X2     = np.column_stack([np.ones(len(train)), f6t_tr, train[ti_col].values])
        x2_pr  = np.array([1.0, f6t_t, data.loc[ym, ti_col]])
        preds[ym], _ = _lstsq_predict(X2, train[y_col].values, x2_pr)
    return pd.Series(preds, name="LNTI")


def oos_all3(fb_ym: pd.DataFrame, df_factors: pd.DataFrame,
             df_sent: pd.DataFrame, y_col: str = "rx_bar",
             oos_start=OOS_START, min_obs: int = 120,
             window=None) -> pd.Series:
    """LN + CP + BW triple combination."""
    F_cols  = ["f1", "f2", "f3", "f4", "f5"]
    F6_cols = ["F1", "F1_cube", "F2", "F3", "F4", "F8"]
    S_cols  = ["SENT_ORTH", "SENT_SQ", "dSENT"]
    all_rx  = [f"rx{n}" for n in MATS]
    fb_clean = fb_ym[[y_col] + all_rx + F_cols].dropna().copy()
    fb_clean["_rxbar"] = fb_clean[all_rx].mean(axis=1)
    data = (fb_clean
            .join(df_factors[F6_cols], how="inner")
            .join(df_sent[S_cols],     how="inner")
            .dropna())

    preds = {}
    for ym in data[data.index >= oos_start].index:
        train = _train_slice(data, ym, window)
        if len(train) < min_obs:
            continue
        X1cp  = np.column_stack([np.ones(len(train))] + [train[c].values for c in F_cols])
        coef_g, *_ = np.linalg.lstsq(X1cp, train["_rxbar"].values, rcond=None)
        cp_t   = float(np.array([1.0] + [data.loc[ym, c] for c in F_cols]) @ coef_g)
        cp_tr  = X1cp @ coef_g
        X1ln   = np.column_stack([np.ones(len(train))] + [train[c].values for c in F6_cols])
        coef_d, *_ = np.linalg.lstsq(X1ln, train["_rxbar"].values, rcond=None)
        f6t_t  = float(np.array([1.0] + [data.loc[ym, c] for c in F6_cols]) @ coef_d)
        f6t_tr = X1ln @ coef_d
        X1bw   = np.column_stack([np.ones(len(train))] + [train[c].values for c in S_cols])
        coef_phi, *_ = np.linalg.lstsq(X1bw, train["_rxbar"].values, rcond=None)
        bw_t   = float(np.array([1.0] + [data.loc[ym, c] for c in S_cols]) @ coef_phi)
        bw_tr  = X1bw @ coef_phi
        X2     = np.column_stack([np.ones(len(train)), cp_tr, f6t_tr, bw_tr])
        x2_pr  = np.array([1.0, cp_t, f6t_t, bw_t])
        preds[ym], _ = _lstsq_predict(X2, train[y_col].values, x2_pr)
    return pd.Series(preds, name="ALL3")


# ── per-maturity models ────────────────────────────────────────────────────────

def oos_fb_per_mat(fb_ym: pd.DataFrame, oos_start=OOS_START,
                   min_obs: int = 120, window=None) -> dict:
    """
    Per-maturity Fama-Bliss: separate OLS for each n in {2,3,4,5}.
    Returns dict {n: Series}. Each series has name 'FB{n}'.

    This extends the scalar FB model: instead of one regression on
    fs_avg, runs rx_n ~ 1 + fs_n for each maturity individually.
    Enables maturity-specific weights in the MV strategy.
    """
    result = {}
    for n in MATS:
        data  = fb_ym[[f"rx{n}", f"fs{n}"]].dropna()
        preds = {}
        for ym in data[data.index >= oos_start].index:
            train = _train_slice(data, ym, window)
            if len(train) < min_obs:
                continue
            X_tr = np.column_stack([np.ones(len(train)), train[f"fs{n}"].values])
            x_pr = np.array([1.0, data.loc[ym, f"fs{n}"]])
            preds[ym], _ = _lstsq_predict(X_tr, train[f"rx{n}"].values, x_pr)
        result[n] = pd.Series(preds, name=f"FB{n}")
    return result


def oos_ln_per_mat(fb_ym: pd.DataFrame, df_factors: pd.DataFrame,
                   oos_start=OOS_START, min_obs: int = 120,
                   window=None) -> dict:
    """
    LN two-step with per-maturity stage 2.
    Stage 1: same as oos_ln — predict rx_bar composite → scalar f6t_t.
    Stage 2: separate regression rx_n ~ 1 + f6t_t for each maturity n.
    Returns dict {n: Series}.

    Rationale: the composite signal is formed on the average bond (as in
    LN2009) but applied maturity-specifically, allowing the strategy to
    overweight long vs short maturities when f6t implies differential carry.
    """
    F6_cols = ["F1", "F1_cube", "F2", "F3", "F4", "F8"]
    all_rx  = [f"rx{n}" for n in MATS]
    fb_clean = fb_ym[["rx_bar"] + all_rx].dropna().copy()
    fb_clean["_rxbar"] = fb_clean[all_rx].mean(axis=1)
    data = fb_clean.join(df_factors[F6_cols], how="inner").dropna()

    result = {n: {} for n in MATS}
    for ym in data[data.index >= oos_start].index:
        train = _train_slice(data, ym, window)
        if len(train) < min_obs:
            continue

        X1 = np.column_stack([np.ones(len(train))] + [train[c].values for c in F6_cols])
        coef_d, *_ = np.linalg.lstsq(X1, train["_rxbar"].values, rcond=None)
        f6_t   = np.array([1.0] + [data.loc[ym, c] for c in F6_cols])
        f6t_t  = float(f6_t @ coef_d)
        f6t_tr = X1 @ coef_d

        X2   = np.column_stack([np.ones(len(f6t_tr)), f6t_tr])
        x2pr = np.array([1.0, f6t_t])
        for n in MATS:
            pred, _ = _lstsq_predict(X2, train[f"rx{n}"].values, x2pr)
            result[n][ym] = pred

    return {n: pd.Series(result[n], name=f"LN{n}") for n in MATS}


def oos_rf(fb_ym: pd.DataFrame, df_factors: pd.DataFrame,
           y_col: str = "rx_bar", oos_start=OOS_START,
           min_obs: int = 120, window=None) -> pd.Series:
    """
    Random Forest direct prediction — Bianchi, Büchner & Tamoni (RFS 2020).
    Maps {F1,...,F8} → rx directly without a two-step composite stage.
    Project hyperparameters: n_estimators=100, max_depth=3.

    Unlike LN (unsupervised PCA + supervised OLS), RF is fully supervised
    and can capture non-linear interactions between macro factors.
    No standardization needed (RF is scale-invariant).
    """
    from sklearn.ensemble import RandomForestRegressor

    F_all    = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
    fb_clean = fb_ym[[y_col]].dropna().copy()
    data     = fb_clean.join(df_factors[F_all], how="inner").dropna()

    preds = {}
    for ym in data[data.index >= oos_start].index:
        train = _train_slice(data, ym, window)
        if len(train) < min_obs:
            continue
        rf = RandomForestRegressor(n_estimators=100, max_depth=3,
                                   random_state=42, n_jobs=1)
        rf.fit(train[F_all].values, train[y_col].values)
        preds[ym] = float(rf.predict(data.loc[[ym], F_all].values)[0])

    return pd.Series(preds, name="RF")


def oos_gbm(fb_ym: pd.DataFrame, df_factors: pd.DataFrame,
            y_col: str = "rx_bar", oos_start=OOS_START,
            min_obs: int = 120, window=None) -> pd.Series:
    """
    Gradient Boosting direct prediction — Bianchi, Büchner & Tamoni (RFS 2020).
    Maps {F1,...,F8} → rx directly. Sequential ensemble of shallow trees.

    Hyperparameters follow Gu, Kelly & Xiu (RFS 2020) convention:
      n_estimators=100, max_depth=2, learning_rate=0.05, subsample=0.5.
    Shallow trees (max_depth=2) prevent overfitting on the ~300-obs training window.
    GBM captures non-linear interactions; subsample=0.5 adds stochastic regularisation.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    F_all    = ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]
    fb_clean = fb_ym[[y_col]].dropna().copy()
    data     = fb_clean.join(df_factors[F_all], how="inner").dropna()

    preds = {}
    for ym in data[data.index >= oos_start].index:
        train = _train_slice(data, ym, window)
        if len(train) < min_obs:
            continue
        gbm = GradientBoostingRegressor(
            n_estimators=100, max_depth=2,
            learning_rate=0.05, subsample=0.5,
            random_state=42,
        )
        gbm.fit(train[F_all].values, train[y_col].values)
        preds[ym] = float(gbm.predict(data.loc[[ym], F_all].values)[0])

    return pd.Series(preds, name="GBM")


# ── BBT2020 raw-variable ML (no PCA compression) ──────────────────────────────

def oos_lasso_raw(fb_ym: pd.DataFrame, df_macro_raw: pd.DataFrame,
                  y_col: str = "rx_bar", oos_start=OOS_START,
                  min_obs: int = 120, window=None,
                  nan_thresh: float = 0.20) -> pd.Series:
    """
    Direct LASSO on raw ~130 macro variables — BBT2020 linear baseline.
    At each OOS step: drop columns with >nan_thresh NaN in training window,
    mean-impute remaining NaN, standardize, fit LassoCV(cv=5).
    Direct prediction (no two-step composite).
    """
    data = fb_ym[[y_col]].dropna().join(df_macro_raw, how="inner")
    macro_cols = [c for c in data.columns if c != y_col]

    preds = {}
    for ym in data[data.index >= oos_start].index:
        train = _train_slice(data, ym, window)
        if len(train) < min_obs:
            continue
        nan_rates = train[macro_cols].isna().mean()
        use_cols  = [c for c in macro_cols if nan_rates[c] <= nan_thresh]
        if len(use_cols) < 5:
            continue
        X_tr_raw = train[use_cols].values.astype(float)
        y_tr     = train[y_col].values
        x_pr_raw = data.loc[ym, use_cols].values.astype(float)
        X_tr, x_pr = _impute_standardize(X_tr_raw, x_pr_raw)
        lasso = LassoCV(cv=5, max_iter=5000, random_state=42, n_jobs=1)
        lasso.fit(X_tr, y_tr)
        preds[ym] = float(lasso.predict(x_pr.reshape(1, -1))[0])
    return pd.Series(preds, name="LASSO_raw")


def oos_rf_raw(fb_ym: pd.DataFrame, df_macro_raw: pd.DataFrame,
               y_col: str = "rx_bar", oos_start=OOS_START,
               min_obs: int = 120, window=None,
               nan_thresh: float = 0.20) -> pd.Series:
    """
    Random Forest on raw ~130 macro variables — BBT2020 proper RF replication.
    n_estimators=100, max_depth=3 (BBT2020 / Gu-Kelly-Xiu 2020).
    Mean-impute NaN; RF is scale-invariant but we standardize for consistency.
    """
    from sklearn.ensemble import RandomForestRegressor

    data = fb_ym[[y_col]].dropna().join(df_macro_raw, how="inner")
    macro_cols = [c for c in data.columns if c != y_col]

    preds = {}
    for ym in data[data.index >= oos_start].index:
        train = _train_slice(data, ym, window)
        if len(train) < min_obs:
            continue
        nan_rates = train[macro_cols].isna().mean()
        use_cols  = [c for c in macro_cols if nan_rates[c] <= nan_thresh]
        if len(use_cols) < 5:
            continue
        X_tr_raw = train[use_cols].values.astype(float)
        y_tr     = train[y_col].values
        x_pr_raw = data.loc[ym, use_cols].values.astype(float)
        X_tr, x_pr = _impute_standardize(X_tr_raw, x_pr_raw)
        rf = RandomForestRegressor(n_estimators=100, max_depth=3,
                                   random_state=42, n_jobs=1)
        rf.fit(X_tr, y_tr)
        preds[ym] = float(rf.predict(x_pr.reshape(1, -1))[0])
    return pd.Series(preds, name="RF_raw")


def oos_gbm_raw(fb_ym: pd.DataFrame, df_macro_raw: pd.DataFrame,
                y_col: str = "rx_bar", oos_start=OOS_START,
                min_obs: int = 120, window=None,
                nan_thresh: float = 0.20) -> pd.Series:
    """
    GBM on raw ~130 macro variables — BBT2020 proper GBM replication.
    Hyperparameters follow Gu, Kelly & Xiu (RFS 2020): n_estimators=100,
    max_depth=2, learning_rate=0.05, subsample=0.5.
    Mean-impute NaN; standardize before fitting.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    data = fb_ym[[y_col]].dropna().join(df_macro_raw, how="inner")
    macro_cols = [c for c in data.columns if c != y_col]

    preds = {}
    for ym in data[data.index >= oos_start].index:
        train = _train_slice(data, ym, window)
        if len(train) < min_obs:
            continue
        nan_rates = train[macro_cols].isna().mean()
        use_cols  = [c for c in macro_cols if nan_rates[c] <= nan_thresh]
        if len(use_cols) < 5:
            continue
        X_tr_raw = train[use_cols].values.astype(float)
        y_tr     = train[y_col].values
        x_pr_raw = data.loc[ym, use_cols].values.astype(float)
        X_tr, x_pr = _impute_standardize(X_tr_raw, x_pr_raw)
        gbm = GradientBoostingRegressor(
            n_estimators=100, max_depth=2,
            learning_rate=0.05, subsample=0.5,
            random_state=42,
        )
        gbm.fit(X_tr, y_tr)
        preds[ym] = float(gbm.predict(x_pr.reshape(1, -1))[0])
    return pd.Series(preds, name="GBM_raw")


# ── Cieslak-Povala (2015) cyclical / trend ──────────────────────────────────

def oos_cp15_cyclical(fb_ym: pd.DataFrame, df_cp15: pd.DataFrame,
                      y_col: str = "rx_bar", oos_start=OOS_START,
                      min_obs: int = 120, window=None) -> pd.Series:
    """
    Cieslak-Povala (2015) cyclical component — predicts 6-12m returns.
    Regression: rx_bar ~ 1 + CYC  (expanding/rolling OLS).
    """
    data = fb_ym[[y_col]].dropna().join(df_cp15[["CYC"]], how="inner").dropna()
    preds = {}
    for ym in data[data.index >= oos_start].index:
        train = _train_slice(data, ym, window)
        if len(train) < min_obs:
            continue
        X_tr = np.column_stack([np.ones(len(train)), train["CYC"].values])
        x_pr = np.array([1.0, data.loc[ym, "CYC"]])
        preds[ym], _ = _lstsq_predict(X_tr, train[y_col].values, x_pr)
    return pd.Series(preds, name="CYC")


def oos_cp15_trend(fb_ym: pd.DataFrame, df_cp15: pd.DataFrame,
                   y_col: str = "rx_bar", oos_start=OOS_START,
                   min_obs: int = 120, window=None) -> pd.Series:
    """
    Cieslak-Povala (2015) trend component — predicts 2-5y returns.
    Regression: rx_bar ~ 1 + TREND  (expanding/rolling OLS).
    """
    data = fb_ym[[y_col]].dropna().join(df_cp15[["TREND"]], how="inner").dropna()
    preds = {}
    for ym in data[data.index >= oos_start].index:
        train = _train_slice(data, ym, window)
        if len(train) < min_obs:
            continue
        X_tr = np.column_stack([np.ones(len(train)), train["TREND"].values])
        x_pr = np.array([1.0, data.loc[ym, "TREND"]])
        preds[ym], _ = _lstsq_predict(X_tr, train[y_col].values, x_pr)
    return pd.Series(preds, name="TREND")


def oos_cp15_both(fb_ym: pd.DataFrame, df_cp15: pd.DataFrame,
                  y_col: str = "rx_bar", oos_start=OOS_START,
                  min_obs: int = 120, window=None) -> pd.Series:
    """
    Cieslak-Povala (2015) combined: rx_bar ~ 1 + CYC + TREND.
    Both cyclical and secular components in a single predictive regression.
    """
    data = fb_ym[[y_col]].dropna().join(df_cp15[["CYC", "TREND"]], how="inner").dropna()
    preds = {}
    for ym in data[data.index >= oos_start].index:
        train = _train_slice(data, ym, window)
        if len(train) < min_obs:
            continue
        X_tr = np.column_stack([np.ones(len(train)),
                                train["CYC"].values, train["TREND"].values])
        x_pr = np.array([1.0, data.loc[ym, "CYC"], data.loc[ym, "TREND"]])
        preds[ym], _ = _lstsq_predict(X_tr, train[y_col].values, x_pr)
    return pd.Series(preds, name="CYC+TREND")


# ── dispatcher ─────────────────────────────────────────────────────────────────

def build_all_oos(fb, df_factors, macro_last_ym,
                  oos_start=OOS_START, window=None,
                  df_sent=None) -> tuple[dict, pd.DataFrame]:
    """
    Run all OOS models. Returns (preds_dict, fb_ym).
    All prediction Series are ym-indexed.

    Note: LN/CPLN/LNBW OOS window ends at macro_last_ym (2011-12),
    shorter than FB/CP/TI/BW (2012-12). This is a data constraint, not
    a modelling choice — comparisons across these groups are approximate.
    """
    fb_ym = fb.set_index("ym")
    tag   = "recursive" if window is None else f"rolling-{window}m"
    print(f"\n  OOS predictions  [{tag}, start={oos_start}]")

    def _run(name, fn):
        print(f"    {name:<6} ...", end=" ", flush=True)
        s = fn()
        print(f"{len(s)} obs")
        return s

    out = {
        "EH":   _run("EH",   lambda: oos_eh(fb_ym, oos_start=oos_start)),
        "FB":   _run("FB",   lambda: oos_fb(fb_ym, oos_start=oos_start, window=window)),
        "CP":   _run("CP",   lambda: oos_cp(fb_ym, oos_start=oos_start, window=window)),
        "LN":   _run("LN",   lambda: oos_ln(fb_ym, df_factors, oos_start=oos_start, window=window)),
        "CPLN": _run("CPLN", lambda: oos_cpln(fb_ym, df_factors, oos_start=oos_start, window=window)),
        "TI":   _run("TI",   lambda: oos_ti(fb_ym, oos_start=oos_start, window=window)),
    }
    if df_sent is not None:
        out["BW"]   = _run("BW",   lambda: oos_bw(fb_ym, df_sent, oos_start=oos_start, window=window))
        out["LNBW"] = _run("LNBW", lambda: oos_lnbw(fb_ym, df_factors, df_sent, oos_start=oos_start, window=window))

    return out, fb_ym
