"""Data loading: Fama-Bliss bonds, Ludvigson-Ng macro factors, BW sentiment."""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA

from .config import DATA_DIR, MATS, TI_LOOKBACK, TRANSFORM, IS_END


def apply_transform(series: pd.Series, code: int) -> pd.Series:
    if code == 1:
        return series.copy()
    elif code == 2:
        return series.diff()
    elif code == 3:
        return series.diff().diff()
    elif code == 4:
        s = series.copy().astype(float); s[s <= 0] = np.nan; return np.log(s)
    elif code == 5:
        s = series.copy().astype(float); s[s <= 0] = np.nan
        return 100 * np.log(s).diff()
    elif code == 6:
        s = series.copy().astype(float); s[s <= 0] = np.nan
        return 100 * np.log(s).diff().diff()
    elif code == 7:
        return 100 * series.astype(float).pct_change().diff()
    else:
        raise ValueError(f"Unknown transform code {code}")


def load_bond_data() -> pd.DataFrame:
    """
    Fama-Bliss zero-coupon bond data.
    Returns DataFrame indexed 0..N with a 'ym' Period column.
    All lp/y/f/fs/rx columns are in percentage-point annual units.
    """
    fb = pd.read_csv(
        f"{DATA_DIR}/Fama_Bliss_price.txt", sep=r"\s+", comment="%",
        names=["date", "p1", "p2", "p3", "p4", "p5"],
    )
    fb["date"] = pd.to_datetime(fb["date"], format="%Y%m%d")
    fb = fb.sort_values("date").reset_index(drop=True)
    fb["ym"] = fb["date"].dt.to_period("M")

    for n in range(1, 6):
        fb[f"lp{n}"] = 100 * np.log(fb[f"p{n}"] / 100)
        fb[f"y{n}"]  = -fb[f"lp{n}"] / n

    fb["f1"] = fb["y1"]
    for n in range(2, 6):
        fb[f"f{n}"]  = fb[f"lp{n-1}"] - fb[f"lp{n}"]
        fb[f"fs{n}"] = fb[f"f{n}"] - fb["y1"]
    fb["fs_avg"] = fb[[f"fs{n}" for n in MATS]].mean(axis=1)

    for n in range(2, 6):
        fb[f"rx{n}"] = fb[f"lp{n-1}"].shift(-12) - fb[f"lp{n}"] - fb["y1"]
    fb["rx_bar"] = fb[[f"rx{n}" for n in MATS]].mean(axis=1)

    for n in MATS:
        ma = fb[f"lp{n}"].rolling(TI_LOOKBACK, min_periods=TI_LOOKBACK).mean()
        fb[f"TI{n}"] = fb[f"lp{n}"] - ma
    fb["TI_avg"] = fb[[f"TI{n}" for n in MATS]].mean(axis=1)

    return fb


def _build_macro_factors(pca_start: str, pca_end: str | None = None):
    """
    Fit PCA on [pca_start, pca_end], project all available data.
    Returns (df indexed by ym Period, last_ym Period).
    """
    macro_raw = pd.read_excel(f"{DATA_DIR}/macro_raw_data.xlsx", sheet_name="macrodata")
    macro_raw["date"] = pd.to_datetime(
        macro_raw["year"].astype(int).astype(str) + "-"
        + macro_raw["month"].astype(int).astype(str).str.zfill(2)
    )
    macro_raw = macro_raw.sort_values("date").reset_index(drop=True)

    all_cols     = [c for c in macro_raw.columns if c not in ("year", "month", "date")]
    include_cols = [c for c in all_cols if c in TRANSFORM]

    transformed = {"date": macro_raw["date"]}
    for col in include_cols:
        transformed[col] = apply_transform(macro_raw[col], TRANSFORM[col])
    macro_tf = pd.DataFrame(transformed)

    fit_mask = macro_tf["date"] >= pca_start
    if pca_end:
        fit_mask &= macro_tf["date"] <= pca_end
    pca_fit = macro_tf[fit_mask].dropna(subset=include_cols).copy()

    X_fit    = pca_fit[include_cols].values
    fit_mean = X_fit.mean(axis=0)
    fit_std  = X_fit.std(axis=0, ddof=1)
    pca_obj  = PCA(n_components=8)
    pca_obj.fit((X_fit - fit_mean) / fit_std)

    macro_valid = macro_tf.dropna(subset=include_cols).copy()
    X_all       = macro_valid[include_cols].values
    factors     = pca_obj.transform((X_all - fit_mean) / fit_std)

    df            = pd.DataFrame(factors, columns=[f"F{k+1}" for k in range(8)])
    df["ym"]      = macro_valid["date"].dt.to_period("M").values
    df["F1_cube"] = df["F1"] ** 3
    df            = df.set_index("ym")
    return df, df.index.max()


def load_macro_factors_oos():
    """
    PCA fitted on 1964-01 to IS_END (1989-12) — used for OOS predictions.
    Fixed loadings prevent look-ahead bias in recursive evaluation.
    """
    df, last = _build_macro_factors("1964-01-01", IS_END.to_timestamp(how="end").strftime("%Y-%m-%d"))
    print(f"  [OOS-macro] PCA fit: 1964-01 – {IS_END}  ({sum((df.index >= '1964-01') & (df.index <= str(IS_END)))} obs in window)")
    print(f"  Macro range: {df.index.min()} – {last}  ({len(df)} obs)")
    return df, last


def load_macro_factors_insample():
    """
    PCA fitted on 1964-01 to 2003-12 — matches Ludvigson-Ng (2009) for IS table.
    """
    df, last = _build_macro_factors("1964-01-01", "2003-12-31")
    print(f"  [IS-macro]  PCA fit: 1964-01 – 2003-12  ({sum((df.index >= '1964-01') & (df.index <= '2003-12'))} obs in window)")
    return df


def load_macro_raw_transformed() -> pd.DataFrame:
    """
    Full transformed macro panel without PCA — all ~130 Ludvigson-Ng (2009) series.
    Same pipeline as _build_macro_factors but stops before PCA compression.
    Returns ym-indexed DataFrame; NaN preserved where transforms require them.
    Used for BBT2020 raw-variable ML (LASSO/RF/GBM applied to full high-dim panel).
    """
    macro_raw = pd.read_excel(f"{DATA_DIR}/macro_raw_data.xlsx", sheet_name="macrodata")
    macro_raw["date"] = pd.to_datetime(
        macro_raw["year"].astype(int).astype(str) + "-"
        + macro_raw["month"].astype(int).astype(str).str.zfill(2)
    )
    macro_raw = macro_raw.sort_values("date").reset_index(drop=True)

    all_cols     = [c for c in macro_raw.columns if c not in ("year", "month", "date")]
    include_cols = [c for c in all_cols if c in TRANSFORM]

    transformed = {"date": macro_raw["date"]}
    for col in include_cols:
        transformed[col] = apply_transform(macro_raw[col], TRANSFORM[col])
    macro_tf = pd.DataFrame(transformed)

    macro_tf["ym"] = macro_tf["date"].dt.to_period("M")
    df = macro_tf.set_index("ym").drop(columns=["date"])

    print(f"  Macro raw panel: {len(include_cols)} series, "
          f"{df.index.min()} – {df.index.max()}  ({len(df)} months)")
    return df


def load_sentiment() -> pd.DataFrame:
    """
    Baker-Wurgler (2006) investor sentiment — Laborda & Olmo (2014) Eq. 14.
    Returns ym-indexed DataFrame with SENT_ORTH, SENT_SQ, dSENT.

    Note: current BW data vintage (updated to 2024) differs from the ~2012
    vintage used in Laborda-Olmo (2014); IS R² reflects this data revision.
    """
    raw = pd.read_excel(f"{DATA_DIR}/SENTIMENT.xlsx", sheet_name="DATA")
    raw["ym"] = pd.to_datetime(raw["yearmo"].astype(str), format="%Y%m").dt.to_period("M")
    raw = raw.sort_values("ym").reset_index(drop=True)

    df = pd.DataFrame(index=raw["ym"])
    df["SENT_ORTH"] = raw["SENT_ORTH"].values
    df["SENT_SQ"]   = df["SENT_ORTH"] ** 2
    df["dSENT"]     = df["SENT_ORTH"] - df["SENT_ORTH"].shift(12)
    df = df.dropna()

    print(f"  Sentiment range: {df.index.min()} – {df.index.max()}  ({len(df)} obs)")
    return df


# ── Extended data (GSW + FRED-MD) ─────────────────────────────────────────────

def _load_gsw_monthly() -> pd.DataFrame:
    """GSW zero-coupon yields → monthly last-observation DataFrame (lp1-lp5, y1-y5)."""
    gsw = pd.read_csv(f"{DATA_DIR}/GSW_yields.csv", skiprows=9, parse_dates=["Date"])
    gsw = gsw.rename(columns={"Date": "date"})
    gsw["ym"] = gsw["date"].dt.to_period("M")
    gsw_m = gsw.sort_values("date").groupby("ym").last()

    df = pd.DataFrame(index=gsw_m.index)
    for n in range(1, 6):
        y = gsw_m[f"SVENY0{n}"].astype(float)
        df[f"lp{n}"] = -n * y
        df[f"y{n}"]  = y
    return df


def load_bond_data_extended() -> pd.DataFrame:
    """
    Fama-Bliss (1953-2012) spliced with GSW zero-coupon yields (2013-2026).
    Identical format to load_bond_data().  GSW extends OOS through ~2025.
    """
    fb = load_bond_data()
    fb_ym = fb.set_index("ym")

    gsw_m = _load_gsw_monthly()

    fb_end    = pd.Period("2012-12", "M")
    gsw_start = pd.Period("2013-01", "M")

    all_idx = fb_ym.index.union(gsw_m.index)
    uni = pd.DataFrame(index=all_idx)
    uni.index.name = "ym"

    for n in range(1, 6):
        lp = pd.concat([
            fb_ym.loc[fb_ym.index <= fb_end, f"lp{n}"],
            gsw_m.loc[gsw_m.index >= gsw_start, f"lp{n}"],
        ])
        y = pd.concat([
            fb_ym.loc[fb_ym.index <= fb_end, f"y{n}"],
            gsw_m.loc[gsw_m.index >= gsw_start, f"y{n}"],
        ])
        uni[f"lp{n}"] = lp
        uni[f"y{n}"]  = y

    uni["f1"] = uni["y1"]
    for n in range(2, 6):
        uni[f"f{n}"]  = uni[f"lp{n-1}"] - uni[f"lp{n}"]
        uni[f"fs{n}"] = uni[f"f{n}"] - uni["y1"]
    uni["fs_avg"] = uni[[f"fs{n}" for n in MATS]].mean(axis=1)

    for n in range(2, 6):
        uni[f"rx{n}"] = uni[f"lp{n-1}"].shift(-12) - uni[f"lp{n}"] - uni["y1"]
    uni["rx_bar"] = uni[[f"rx{n}" for n in MATS]].mean(axis=1)

    for n in MATS:
        ma = uni[f"lp{n}"].rolling(TI_LOOKBACK, min_periods=TI_LOOKBACK).mean()
        uni[f"TI{n}"] = uni[f"lp{n}"] - ma
    uni["TI_avg"] = uni[[f"TI{n}" for n in MATS]].mean(axis=1)

    return uni.reset_index()


def _build_macro_factors_fredmd(pca_start: str, pca_end: str | None = None):
    """
    PCA on FRED-MD (McCracken & Ng updated panel, 2026-04-MD.csv).
    First data row is the transform-code header; applies FRED-MD codes 1-7.
    """
    raw = pd.read_csv(f"{DATA_DIR}/2026-04-MD.csv")
    transform_row = raw.iloc[0].copy()
    raw = raw.iloc[1:].copy().reset_index(drop=True)
    raw["sasdate"] = pd.to_datetime(raw["sasdate"])
    raw = raw.sort_values("sasdate").reset_index(drop=True)

    macro_cols = [c for c in raw.columns if c != "sasdate"]

    trans_codes = {}
    for col in macro_cols:
        try:
            trans_codes[col] = int(float(transform_row[col]))
        except (ValueError, TypeError):
            pass

    transformed = {"date": raw["sasdate"]}
    for col in macro_cols:
        if col in trans_codes:
            s = pd.to_numeric(raw[col], errors="coerce")
            transformed[col] = apply_transform(s, trans_codes[col])

    macro_tf   = pd.DataFrame(transformed)
    incl_cols  = [c for c in macro_cols if c in trans_codes]

    fit_mask = macro_tf["date"] >= pca_start
    if pca_end:
        fit_mask &= macro_tf["date"] <= pca_end
    pca_window = macro_tf[fit_mask][incl_cols]

    # Drop series with >20% NaN in the fit window (e.g. series not yet published)
    nan_rate   = pca_window.isna().mean()
    incl_cols  = [c for c in incl_cols if nan_rate[c] <= 0.20]

    pca_fit    = macro_tf[fit_mask].dropna(subset=incl_cols).copy()

    X_fit    = pca_fit[incl_cols].values.astype(float)
    fit_mean = X_fit.mean(axis=0)
    fit_std  = X_fit.std(axis=0, ddof=1)
    fit_std[fit_std == 0] = 1.0
    pca_obj  = PCA(n_components=8)
    pca_obj.fit((X_fit - fit_mean) / fit_std)

    macro_valid = macro_tf.dropna(subset=incl_cols).copy()
    X_all       = macro_valid[incl_cols].values.astype(float)
    factors     = pca_obj.transform((X_all - fit_mean) / fit_std)

    df            = pd.DataFrame(factors, columns=[f"F{k+1}" for k in range(8)])
    df["ym"]      = macro_valid["date"].dt.to_period("M").values
    df["F1_cube"] = df["F1"] ** 3
    df            = df.set_index("ym")
    return df, df.index.max()


def load_macro_factors_oos_extended():
    """
    FRED-MD PCA for extended OOS analysis.  PCA fitted on 1964-01 to IS_END
    (1989-12) — same window as baseline — projected through 2026-03.
    """
    pca_end = IS_END.to_timestamp(how="end").strftime("%Y-%m-%d")
    df, last = _build_macro_factors_fredmd("1964-01-01", pca_end)
    print(f"  [EXT-macro] FRED-MD PCA fit: 1964-01 – {IS_END}")
    print(f"  Extended macro range: {df.index.min()} – {last}  ({len(df)} obs)")
    return df, last


# ── Cieslak-Povala (2015) cyclical / trend decomposition ───────────────────

def load_cp15_components(fit_end: str = "1989-12") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cieslak & Povala (JFE 2015): decompose yields into cyclical and trend
    components using CPI inflation and unemployment rate.

    For each maturity n ∈ {1,...,5}:
        y_n(t) = α + β · CPI_YoY(t) + γ · UNRATE(t) + ε_n(t)
    Fitted values = cyclical component (expectations-driven).
    Residuals     = trend component (risk-premium-driven).

    Then extract the first principal component from cyclical components across
    maturities → CYC factor, and from trend residuals → TREND factor.

    PCA fitted on 1964-01 to fit_end (1989-12 for OOS); projected to full
    sample. Returns (df_components ym-indexed, fit_end Period).
    """
    # Read FRED-MD raw (untransformed) data for CPIAUCSL and UNRATE
    raw = pd.read_csv(f"{DATA_DIR}/2026-04-MD.csv")
    raw = raw.iloc[1:].copy().reset_index(drop=True)
    raw["sasdate"] = pd.to_datetime(raw["sasdate"])
    raw = raw.sort_values("sasdate").reset_index(drop=True)

    cpi_raw = pd.to_numeric(raw["CPIAUCSL"], errors="coerce")
    unr_raw = pd.to_numeric(raw["UNRATE"],  errors="coerce")

    # CPI YoY inflation rate (%): 100 * log(CPI_t / CPI_{t-12})
    cpi_yoy = 100 * np.log(cpi_raw / cpi_raw.shift(12))

    macro_cyc = pd.DataFrame({
        "date":  raw["sasdate"],
        "CPI_YoY": cpi_yoy,
        "UNRATE":  unr_raw,
    }).dropna()
    macro_cyc["ym"] = macro_cyc["date"].dt.to_period("M")
    macro_cyc = macro_cyc.set_index("ym").drop(columns=["date"])

    # Load bond yields
    fb = load_bond_data()
    fb_ym = fb.set_index("ym")
    yield_data = fb_ym[[f"y{n}" for n in range(1, 6)]].copy()

    # Join macro and yields
    data = yield_data.join(macro_cyc, how="inner").dropna()
    full_idx = data.index

    fit_end_period = pd.Period(fit_end, "M")
    fit_mask = data.index <= fit_end_period
    fit_data = data[fit_mask]

    # Stage 1: for each maturity, regress y_n on [CPI_YoY, UNRATE]
    X_fit = sm.add_constant(fit_data[["CPI_YoY", "UNRATE"]].values)
    cyclical = {}
    trend    = {}
    for n in range(1, 6):
        y_fit = fit_data[f"y{n}"].values
        res   = sm.OLS(y_fit, X_fit).fit()
        # Full-sample projection with same coefficients
        X_all = sm.add_constant(data[["CPI_YoY", "UNRATE"]].values)
        cyclical[n] = res.predict(X_all)
        trend[n]    = data[f"y{n}"].values - cyclical[n]

    cyc_df = pd.DataFrame(
        {f"CYC_y{n}": cyclical[n] for n in range(1, 6)},
        index=full_idx,
    )
    trd_df = pd.DataFrame(
        {f"TREND_y{n}": trend[n] for n in range(1, 6)},
        index=full_idx,
    )

    # Stage 2: PCA on cyclical components (fitted on pre-OOS window)
    cyc_cols = [f"CYC_y{n}" for n in range(1, 6)]
    trd_cols = [f"TREND_y{n}" for n in range(1, 6)]

    cyc_fit = cyc_df.loc[fit_mask, cyc_cols].dropna()
    trd_fit = trd_df.loc[fit_mask, trd_cols].dropna()

    pca_cyc = PCA(n_components=1).fit(
        (cyc_fit.values - cyc_fit.values.mean(axis=0)) / cyc_fit.values.std(axis=0, ddof=1)
    )
    pca_trd = PCA(n_components=1).fit(
        (trd_fit.values - trd_fit.values.mean(axis=0)) / trd_fit.values.std(axis=0, ddof=1)
    )

    X_cyc_all = (cyc_df[cyc_cols].values - cyc_fit.values.mean(axis=0)) \
                / cyc_fit.values.std(axis=0, ddof=1)
    X_trd_all = (trd_df[trd_cols].values - trd_fit.values.mean(axis=0)) \
                / trd_fit.values.std(axis=0, ddof=1)

    result = pd.DataFrame({
        "CYC":   pca_cyc.transform(X_cyc_all)[:, 0],
        "TREND": pca_trd.transform(X_trd_all)[:, 0],
    }, index=full_idx)

    corr_cyc_y = result.loc[fit_mask, "CYC"].corr(data.loc[fit_mask, "y1"])
    corr_trd_y = result.loc[fit_mask, "TREND"].corr(data.loc[fit_mask, "y1"])

    print(f"  CP2015 components: {result.index.min()} – {result.index.max()}  ({len(result)} obs)")
    print(f"  PCA fit: 1964-01 – {fit_end}")
    print(f"  corr(CYC, y1)={corr_cyc_y:.3f}  corr(TREND, y1)={corr_trd_y:.3f}")
    print(f"  CP2015 finding: CYC counter-cyclical (predicts returns at 6-12m),")
    print(f"                   TREND secular (predicts returns at 2-5y)")

    return result, fit_end_period
