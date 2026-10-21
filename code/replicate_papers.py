"""
Historical in-sample formula checks inspired by three bond risk premium papers.
Uses the available updated macro panel, not an exact original-vintage replication.
For corrected out-of-sample results use scripts/run_analysis.py.

Papers:
  1. Fama & Bliss (1987)    "The Information in Long-Maturity Forward Rates"
  2. Cochrane & Piazzesi (2005)  "Bond Risk Premia"
  3. Ludvigson & Ng (2009)  "Macro Factors in Bond Risk Premia"

Critical convention (all three papers):
  - Holding period = ONE YEAR (not one month)
  - rx(n)_t = lp(n-1)_{t+12} - lp(n)_t - y(1)_t  (annual excess return)
  - Data observed monthly → overlapping 12-month returns
  - S.E. corrected via Newey-West with 18 lags (as CP and LN do)

Data used (raw only):
  - Fama_Bliss_price.txt : raw monthly bond prices (1953–2013), CRSP/WRDS
  - macro_raw_data.xlsx  : 132 raw macro series (1960–2011)

Pre-computed factors/returns are not used in estimation. RFS2009.xls is
used only in an optional final alignment check; 1964to2007.xls is unused.
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
import warnings


from bond_rp.config import DATA_DIR

BASE = DATA_DIR


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1: BOND MARKET DATA (Fama-Bliss)
# ─────────────────────────────────────────────────────────────────────────────

def load_all_bond_data():
    """
    Load all Fama-Bliss price data and compute bond market variables.

    Variable definitions (Cochrane-Piazzesi 2005 notation, all × 100 → percent):
      lp(n)   = 100 × log(p_n / 100)          [log price, percent]
      y(n)    = −lp(n) / n                     [cont.-comp. yield, ann., %]
      f(n)    = lp(n−1) − lp(n)               [1-yr forward rate for yr n]
                  f(1) = y(1) by definition
      rx(n)   = lp(n−1)_{t+12} − lp(n)_t − y(1)_t   [annual excess return, %]
      fs(n)   = f(n)_t − y(1)_t               [forward-spot spread]

    One-year holding period: buy n-year bond at t, sell as (n-1)-year bond
    at t+12 months.  All rx values stored at the DATE OF PURCHASE (time t).
    """
    fb = pd.read_csv(
        f"{BASE}/Fama_Bliss_price.txt",
        sep=r"\s+", comment="%",
        names=["date", "p1", "p2", "p3", "p4", "p5"],
    )
    fb["date"] = pd.to_datetime(fb["date"], format="%Y%m%d")
    fb = fb.sort_values("date").reset_index(drop=True)
    fb["ym"] = fb["date"].dt.to_period("M")

    # log prices (percent)
    for n in range(1, 6):
        fb[f"lp{n}"] = 100 * np.log(fb[f"p{n}"] / 100)

    # yields (annualized, percent)
    for n in range(1, 6):
        fb[f"y{n}"] = -fb[f"lp{n}"] / n

    # forward rates: f(n) = lp(n−1) − lp(n);  f(1) = y(1)
    fb["f1"] = fb["y1"]
    for n in range(2, 6):
        fb[f"f{n}"] = fb[f"lp{n-1}"] - fb[f"lp{n}"]

    # forward-spot spreads
    for n in range(2, 6):
        fb[f"fs{n}"] = fb[f"f{n}"] - fb["y1"]

    # ONE-YEAR holding period excess returns (shift by 12 months)
    # rx(n)_t = lp(n−1)_{t+12} − lp(n)_t − y(1)_t
    for n in range(2, 6):
        fb[f"rx{n}"] = fb[f"lp{n-1}"].shift(-12) - fb[f"lp{n}"] - fb["y1"]

    return fb


def nw_ols(y, X, lags=18):
    """OLS with Newey-West standard errors (lag length = lags)."""
    res = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return res


print("=" * 72)
print("Loading all Fama-Bliss bond data …")
fb_all = load_all_bond_data()
print(f"  Total rows: {len(fb_all)}  ({fb_all['ym'].iloc[0]} – {fb_all['ym'].iloc[-1]})")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2: FAMA & BLISS (1987)
# ─────────────────────────────────────────────────────────────────────────────
#
# Equation (12): rx(n)_{t+1yr} = a + b * [f(n)_t − y(1)_t] + u
#
# Sample (paper): January 1964 – December 1985
# Regression obs: t = 1964:01 – 1984:12  (N = 252; rx uses prices through 1985:12)
#
# Paper Table 1 (expected):
#   n=2: b≈0.91, se≈0.28, R²≈0.14
#   n=3: b≈1.13, se≈0.33, R²≈0.13
#   n=4: b≈1.42, se≈0.45, R²≈0.11
#   n=5: b≈0.93, se≈0.53, R²≈0.05

print("\n" + "=" * 72)
print("PAPER 1: FAMA & BLISS (1987)")
print("  rx(n) = a + b*(f(n) − y(1)) + u")
print("  Regression sample: 1964:01 – 1984:12  (N should be 252)")
print("-" * 72)

# Filter to regressor sample: 1964:01 – 1984:12
# rx is valid here because lp_{t+12} uses prices through 1985:12
fb87 = fb_all[
    (fb_all["ym"] >= "1964-01") & (fb_all["ym"] <= "1984-12")
].dropna(subset=[f"rx{n}" for n in range(2, 6)]).copy()

print(f"  N = {len(fb87)}")
print(f"\n{'n':>4} {'a':>9} {'b':>9} {'NW-se(b)':>10} {'t(b)':>7} {'R²':>7}"
      f"   Paper: b / se / R²")
print("-" * 72)

paper_fb87 = {2: (0.91, 0.28, 0.14), 3: (1.13, 0.33, 0.13),
              4: (1.42, 0.45, 0.11), 5: (0.93, 0.53, 0.05)}
fb87_results = {}
for n in range(2, 6):
    y = fb87[f"rx{n}"]
    X = sm.add_constant(fb87[f"fs{n}"])
    res = nw_ols(y, X)
    a  = res.params.iloc[0]
    b  = res.params.iloc[1]
    se = res.bse.iloc[1]
    t  = res.tvalues.iloc[1]
    r2 = res.rsquared
    fb87_results[n] = res
    pb, pse, pr2 = paper_fb87[n]
    print(f"  {n}   {a:9.4f} {b:9.4f} {se:10.4f} {t:7.2f} {r2:7.4f}"
          f"   [{pb} / {pse} / {pr2}]")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3: COCHRANE & PIAZZESI (2005)
# ─────────────────────────────────────────────────────────────────────────────
#
# Step 1 – average return regression (identifies the single factor γ):
#   rx̄_{t+1} = γ₀ + γ₁f(1)_t + γ₂f(2)_t + γ₃f(3)_t + γ₄f(4)_t + γ₅f(5)_t + ε
#   where rx̄ = (1/4) Σ_{n=2}^{5} rx(n)
#
# Step 2 – CP factor:
#   CP_t = γ̂₀ + γ̂₁f(1)_t + … + γ̂₅f(5)_t  (fitted value from step 1)
#
# Step 3 – individual bond regressions:
#   rx(n)_{t+1} = α_n + β_n CP_t + ε_{t+1}
#
# Sample: t = 1964:01 – 2003:12  (rx uses prices through 2004:12)
# Paper N = 480 (uses all months 1964:01–2003:12 as regression observations)
#
# Expected (Table 1B, single-factor model):
#   β_n ≈ 0.47, 0.87, 1.24, 1.43   (increasing in maturity)
#   R²  ≈ 0.32, 0.34, 0.37, 0.35
#   Average return regression R² ≈ 0.35
#
# γ tent-shape (Table 1A, approximate):
#   γ₁ ≈ −2.14, γ₂ ≈ 0.81, γ₃ ≈ 3.00, γ₄ ≈ −2.08, γ₅ ≈ 0.80

print("\n" + "=" * 72)
print("PAPER 2: COCHRANE & PIAZZESI (2005)")
print("  Sample: 1963:01 – 2003:12 (purchase dates; sale dates 1964:01–2004:12)")
print("  RFS2009 convention: yr2 labeled at sale date → our 1963-01 = RFS2009 1964-01")
print("-" * 72)

cp_data = fb_all[
    (fb_all["ym"] >= "1963-01") & (fb_all["ym"] <= "2002-12")
].dropna(subset=[f"rx{n}" for n in range(2, 6)]).copy()

print(f"  N = {len(cp_data)}")

# average excess return
cp_data["rx_bar"] = cp_data[[f"rx{n}" for n in range(2, 6)]].mean(axis=1)

# Step 1
F_cols = ["f1", "f2", "f3", "f4", "f5"]
X1 = sm.add_constant(cp_data[F_cols])
res_avg = nw_ols(cp_data["rx_bar"], X1)

print(f"\nStep 1 – Average return regression")
print(f"  R² = {res_avg.rsquared:.4f}   (paper: ~0.35)")
print(f"  γ coefficients (tent-shaped pattern expected):")
for name, coef, se in zip(["const"] + F_cols, res_avg.params, res_avg.bse):
    print(f"    {name:6s}: {coef:8.4f}  (NW-se={se:.4f})")

# Step 2 – CP factor
cp_data["CP"] = res_avg.fittedvalues

# Step 3
print(f"\nStep 3 – Individual bond regressions on CP factor")
print(f"  [Expected: β ≈ 0.47, 0.87, 1.24, 1.43; R² ≈ 0.32, 0.34, 0.37, 0.35]")
print(f"{'n':>4} {'α':>9} {'β':>9} {'NW-se':>8} {'t(β)':>7} {'R²':>7}")
print("-" * 55)

paper_cp = {2: (0.47, 0.32), 3: (0.87, 0.34), 4: (1.24, 0.37), 5: (1.43, 0.35)}
cp_betas = {}
for n in range(2, 6):
    y = cp_data[f"rx{n}"]
    X = sm.add_constant(cp_data["CP"])
    res = nw_ols(y, X)
    a, b = res.params.iloc[0], res.params.iloc[1]
    se   = res.bse.iloc[1]
    t    = res.tvalues.iloc[1]
    r2   = res.rsquared
    cp_betas[n] = b
    pb, pr2 = paper_cp[n]
    print(f"  {n}   {a:9.4f} {b:9.4f} {se:8.4f} {t:7.2f} {r2:7.4f}"
          f"   [β={pb}, R²={pr2}]")

# Fama-Bliss univariate benchmark on 1964-2003 sample (CP Table 2)
print(f"\nFama-Bliss univariate on same 1964-2003 sample (CP Table 2 comparison):")
print(f"  [Expected R²: 0.16, 0.17, 0.18, 0.09]")
print(f"{'n':>4} {'R²':>8}")
paper_fb_cp = {2: 0.16, 3: 0.17, 4: 0.18, 5: 0.09}
for n in range(2, 6):
    y  = cp_data[f"rx{n}"]
    X  = sm.add_constant(cp_data[f"fs{n}"])
    r2 = nw_ols(y, X).rsquared
    print(f"  {n}   {r2:8.4f}   [paper: {paper_fb_cp[n]}]")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4: MACRO FACTOR CONSTRUCTION (Ludvigson & Ng 2009)
# ─────────────────────────────────────────────────────────────────────────────
#
# LN (2009) extract principal components from 132 macro series.
# Financial variables (interest rates, spreads, equities, FX) ARE included.
# Transformation codes follow Stock-Watson (2002) / McCracken-Ng (2016):
#   1 = level
#   2 = Δ           (first difference)
#   4 = ln          (log level)
#   5 = Δln × 100  (log first difference → growth rate %)
#   6 = Δ²ln × 100 (log second difference → change in inflation rate)

print("\n" + "=" * 72)
print("PAPER 3: LUDVIGSON & NG (2009) – Macro Factor Construction")
print("-" * 72)

macro_raw = pd.read_excel(f"{BASE}/macro_raw_data.xlsx", sheet_name="macrodata")
macro_raw["date"] = pd.to_datetime(
    macro_raw["year"].astype(int).astype(str) + "-"
    + macro_raw["month"].astype(int).astype(str).str.zfill(2)
)
macro_raw = macro_raw.sort_values("date").reset_index(drop=True)

# ── Transformation codes for all 132 macro series ────────────────────────────
# LN (2009) include ALL series in macrodata: real variables + prices +
# interest rates + stock market + exchange rates (from their Table A.1).
# Codes follow Stock-Watson / McCracken-Ng:
#   5 = 100 × Δlog  (log first difference → growth rate %)
#   2 = Δ           (first difference)
#   1 = level       (already stationary)
#
# Note: column names 'PPI: int mat’ls' and 'PPI: crude mat’ls'
# use right curly apostrophe (’), not straight apostrophe.
# Note: 'DC&I loans' has many zero/negative values → use code 2, not 5.

TRANSFORM = {
    # Income & consumption
    "PI": 5, "PI less transfers": 5, "Real Consumption": 5,
    "M&T sales": 5, "Retail sales": 5,
    # Industrial production
    "IP: total": 5, "IP: products": 5, "IP: final prod": 5,
    "IP: cons gds": 5, "IP: cons dble": 5, "IP: cons nondble": 5,
    "IP: bus eqpt": 5, "IP: matls": 5, "IP: dble matls": 5,
    "IP: nondble matls": 5, "IP: mfg": 5, "IP: res util": 5, "IP: fuels": 5,
    # Surveys / capacity
    "NAPM prodn ": 1, "Cap util": 2,
    "Help wanted indx": 5, "Help wanted/unemp": 2,
    # Employment (CPS and payroll)
    "Emp CPS total": 5, "Emp CPS nonag": 5,
    "U: all": 2, "U: mean duration": 2, "U < 5 wks": 2, "U 5-14 wks": 2,
    "U 15+ wks ": 2, "U 15-26 wks": 2, "U 27+ wks": 2,
    "UI claims": 5,
    "Emp: total": 5, "Emp: gds prod": 5, "Emp: mining": 5, "Emp: const": 5,
    "Emp: mfg": 5, "Emp: dble gds": 5, "Emp: nondbles": 5, "Emp: services": 5,
    "Emp: TTU": 5, "Emp: wholesale": 5, "Emp: retail": 5, "Emp: FIRE": 5,
    "Emp: Govt": 5,
    # Hours
    "Agg wkly hours": 5, "Avg hrs ": 2, "Overtime: mfg": 2, "Avg hrs: mfg": 2,
    "NAPM empl": 1,
    # Housing (Table A1 codes 4 = ln level, NOT Δln)
    "Starts: nonfarm": 4, "Starts: NE": 4, "Starts: MW": 4,
    "Starts: South": 4, "Starts: West": 4,
    "BP: total": 4, "BP: NE": 4, "BP: MW": 4, "BP: South": 4, "BP: West": 4,
    # Orders / inventories
    "PMI": 1, "NAPM new ordrs": 1, "NAPM vendor del": 1, "NAPM Invent": 1,
    "Orders: cons gds": 5, "Orders: dble gds": 5, "Orders: cap gds": 5,
    "Unf orders: dble": 5, "M&T invent": 5, "M&T invent/sales": 2,
    # Money & credit (Table A1: money stocks = code 6 Δ²ln)
    "M1": 6, "M2": 6, "Currency": 6, "M2 (real)": 6, "MB": 6,
    "Reserves tot": 6, "Reserves nonbor": 6,
    "C&I loan plus": 5,
    "DC&I loans": 1,       # already "ΔC&I loans" (a change series) → level as-is
    "Cons credit": 6, "Inst cred/PI": 2,
    # Stock market (LN include these as macro variables)
    "S&P 500": 5, "S&P: indust": 5, "S&P div yield": 2, "S&P PE ratio": 2,
    # Interest rates → first difference (already stationary as levels)
    "Fed Funds": 2, "Comm paper": 2,
    "3 mo T-bill": 2, "6 mo T-bill": 2, "1 yr T-bond": 2,
    "5 yr T-bond": 2, "10 yr T-bond": 2, "Aaa bond": 2, "Baa bond": 2,
    # Spreads → level (already stationary)
    "CP-FF spread": 1, "3 mo-FF spread": 1, "6 mo-FF spread": 1,
    "1 yr-FF spread": 1, "5 yr-FF spread": 1, "10 yr-FF spread": 1,
    "Aaa-FF spread": 1, "Baa-FF spread": 1,
    # Exchange rates → log diff
    "Ex rate: avg": 5, "Ex rate: Switz": 5, "Ex rate: Japan": 5,
    "Ex rate: UK": 5, "EX rate: Canada": 5,
    # Prices (Table A1: price indices = code 6 Δ²ln, NOT Δln)
    # Curly apostrophe (‘) in two PPI column names
    "PPI: fin gds": 6, "PPI: cons gds": 6,
    "PPI: int mat’ls": 6, "PPI: crude mat’ls": 6,
    "Spot market price": 6, "PPI: nonferrous": 6, "NAPM com price": 1,
    "CPI-U: all": 6, "CPI-U: apparel": 6, "CPI-U: transp": 6,
    "CPI-U: medical": 6, "CPI-U: comm.": 6, "CPI-U: dbles": 6,
    "CPI-U: services": 6, "CPI-U: ex food": 6, "CPI-U: ex shelter": 6,
    "CPI-U: ex med": 6,
    "PCE defl": 6, "PCE defl: dlbes": 6, "PCE defl: nondble": 6,
    "PCE defl: service": 6,
    "AHE: goods": 6, "AHE: const": 6, "AHE: mfg": 6,
    "Consumer expect": 2,
}

def apply_transform(series, code):
    if code == 1:
        return series.copy()
    elif code == 2:
        return series.diff()
    elif code == 4:
        s = series.copy().astype(float)
        s[s <= 0] = np.nan
        return np.log(s)                    # log level (no differencing)
    elif code == 5:
        s = series.copy().astype(float)
        s[s <= 0] = np.nan
        return 100 * np.log(s).diff()       # Δln × 100
    elif code == 6:
        s = series.copy().astype(float)
        s[s <= 0] = np.nan
        return 100 * np.log(s).diff().diff()  # Δ²ln × 100
    else:
        raise ValueError(f"Unknown code {code}")

all_cols = [c for c in macro_raw.columns if c not in ("year", "month", "date")]
include_cols = [c for c in all_cols if c in TRANSFORM]
missing = [c for c in all_cols if c not in TRANSFORM]
print(f"  Macro series for PCA: {len(include_cols)}")
print(f"  (LN 2009 paper uses 132 series)")
if missing:
    print(f"  Columns not in TRANSFORM (skipped): {missing}")

# Apply transformations to full raw series (keeps all dates)
transformed = {"date": macro_raw["date"]}
for col in include_cols:
    transformed[col] = apply_transform(macro_raw[col], TRANSFORM[col])
macro_tf_full = pd.DataFrame(transformed)

# ── PCA fitted on 1964-2003 (paper's standard in-sample period) ──────────────
macro_fit = macro_tf_full[
    (macro_tf_full["date"] >= "1964-01-01") &
    (macro_tf_full["date"] <= "2003-12-31")
].dropna().copy()
print(f"  Macro obs for PCA fit (1964–2003, after transform+dropna): {len(macro_fit)}")

X_fit = macro_fit[include_cols].values
fit_mean = X_fit.mean(axis=0)
fit_std  = X_fit.std(axis=0, ddof=1)
X_fit_std = (X_fit - fit_mean) / fit_std

pca = PCA(n_components=8)
pca.fit(X_fit_std)
var_exp = pca.explained_variance_ratio_

print(f"\n  Variance explained by first 8 PCs (fitted on 1964–2003):")
cumvar = 0
for k in range(8):
    cumvar += var_exp[k]
    print(f"    F̃{k+1}: {100*var_exp[k]:5.1f}%  (cumulative: {100*cumvar:.1f}%)")

# ── Apply same PCA loadings to 1963-2003 (to get factors for 1963-01) ────────
# Standardize using the 1964-2003 mean/std (not re-fitted on 1963-2003)
macro_tf = macro_tf_full[
    (macro_tf_full["date"] >= "1963-01-01") &
    (macro_tf_full["date"] <= "2003-12-31")
].dropna().copy()
print(f"  Macro obs projected (1963–2003): {len(macro_tf)}")

X_all     = macro_tf[include_cols].values
X_all_std = (X_all - fit_mean) / fit_std
factors   = pca.transform(X_all_std)

macro_tf["ym"] = macro_tf["date"].dt.to_period("M")
df_factors = pd.DataFrame(factors, columns=[f"F{k+1}" for k in range(8)])
df_factors["ym"] = macro_tf["ym"].values


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5: LUDVIGSON & NG (2009) – Return Forecasting Regressions
# ─────────────────────────────────────────────────────────────────────────────
#
# BIC selects:  F̄6_t = {F̂₁, F̂³₁, F̂₂, F̂₃, F̂₄, F̂₈}  (without CP)
#               F̄5_t = {F̂₁, F̂³₁, F̂₃, F̂₄, F̂₈}        (with CP, F2 subsumed)
#
# Table 2 rows (for rx^(2)):
#   (a) CP only              R²=0.31
#   (b) F1,F1³,F2,F3,F4,F8  R²=0.26
#   (c) same + CP            R²=0.45
#   (f) F5_t single factor   R²=0.22
#   (g) F6_t single factor   R²=0.26
#   (h) F5_t + CP            R²=0.44
#
# Composite single factors (analogous to CP):
#   F6_t = γ̂'F̄6_t  (fitted value from avg-return reg on 6 factors)
#   F5_t = δ̂'F̄5_t  (fitted value from avg-return reg on 5 factors)

print("\n" + "=" * 72)
print("PAPER 3: LUDVIGSON & NG (2009) – Return Forecasting Regressions")
print("  Sample: 1964:01 – 2003:12")
print("-" * 72)

# Merge bond data with macro factors
cp_ym = cp_data[["ym", "CP"] + [f"rx{n}" for n in range(2,6)] +
                [f"f{n}" for n in range(1,6)] + [f"fs{n}" for n in range(2,6)]].copy()
ln_data = cp_ym.merge(df_factors, on="ym", how="inner")
ln_data = ln_data.dropna()
print(f"  Joint obs (bond + macro): {len(ln_data)}")

# Add cubic of F1
ln_data["F1_cube"] = ln_data["F1"] ** 3
ln_data["F1_sq"]   = ln_data["F1"] ** 2   # for reference

# BIC-selected factor sets
F6_BIC = ["F1", "F1_cube", "F2", "F3", "F4", "F8"]   # without CP
F5_BIC = ["F1", "F1_cube",        "F3", "F4", "F8"]   # with CP (F2 subsumed)

# ── Composite factor F6_t (Eq. 7 in paper) ───────────────────────────────────
# F6_t = fitted value from regressing average excess return on F̄6 factors
rx_bar = ln_data[[f"rx{n}" for n in range(2, 6)]].mean(axis=1)

res_f6 = nw_ols(rx_bar, sm.add_constant(ln_data[F6_BIC]))
ln_data["F6t"] = res_f6.fittedvalues

print(f"\n  Avg-return reg on F̄6 factors (F1,F1³,F2,F3,F4,F8):")
print(f"  R̄² = {res_f6.rsquared:.4f}   [Paper eq.(7): R̄²=0.224]")
print(f"  t-stats: " + "  ".join(
    f"{nm}:{tv:.2f}" for nm, tv in zip(
        ["const"]+F6_BIC, res_f6.tvalues)))

# ── Composite factor F5_t (Eq. 8 in paper) ───────────────────────────────────
res_f5 = nw_ols(rx_bar, sm.add_constant(ln_data[F5_BIC]))
ln_data["F5t"] = res_f5.fittedvalues

print(f"\n  Avg-return reg on F̄5 factors (F1,F1³,F3,F4,F8):")
print(f"  R̄² = {res_f5.rsquared:.4f}   [Paper eq.(8)]")

# ── Table 2 replications ──────────────────────────────────────────────────────
print(f"\n  TABLE 2 Replication (Paper values in brackets):")
print(f"  {'Row':<5} {'Model':<35}  {'n=2':>6} {'n=3':>6} {'n=4':>6} {'n=5':>6}")
print("-" * 70)

specs_table2 = [
    ("(a)", "CP only",                         ["CP"]),
    ("(b)", "F1,F1³,F2,F3,F4,F8 (F̄6)",       F6_BIC),
    ("(c)", "F1,F1³,F2,F3,F4,F8 + CP",        F6_BIC + ["CP"]),
    ("(d)", "F1,F1³,F3,F4,F8 (F̄5)",           F5_BIC),
    ("(e)", "F1,F1³,F3,F4,F8 + CP",           F5_BIC + ["CP"]),
    ("(f)", "F5t (composite 5-factor)",        ["F5t"]),
    ("(g)", "F6t (composite 6-factor)",        ["F6t"]),
    ("(h)", "F5t + CP",                        ["F5t", "CP"]),
]
paper_t2 = {
    "(a)": (0.31, 0.34, 0.37, 0.34),
    "(b)": (0.26, 0.18, 0.16, 0.14),
    "(c)": (0.45, 0.44, 0.45, 0.42),
    "(d)": (0.22, 0.19, 0.17, 0.14),
    "(e)": (0.45, 0.24, 0.23, 0.21),
    "(f)": (0.22, 0.19, 0.17, 0.14),
    "(g)": (0.26, 0.24, 0.23, 0.21),
    "(h)": (0.44, 0.44, 0.45, 0.42),
}
for row, label, xcols in specs_table2:
    r2s = []
    for n in range(2, 6):
        y   = ln_data[f"rx{n}"]
        X   = sm.add_constant(ln_data[xcols])
        r2s.append(sm.OLS(y, X).fit().rsquared)
    vals = "".join(f" {r:6.4f}" for r in r2s)
    pv   = paper_t2[row]
    pstr = "".join(f" {v:6.2f}" for v in pv)
    print(f"  {row:<5} {label:<35} {vals}")
    print(f"  {'':5} {'[paper]':<35} {pstr}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6: GRAND SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 72)
print("GRAND SUMMARY – In-sample R² by maturity")
print("=" * 72)
print(f"{'Model':<45} {'n=2':>6} {'n=3':>6} {'n=4':>6} {'n=5':>6}")
print("-" * 70)

def get_r2s(data, xcols_by_n):
    r2s = []
    for n in range(2, 6):
        y  = data[f"rx{n}"]
        X  = sm.add_constant(data[xcols_by_n[n]])
        r2s.append(sm.OLS(y, X).fit().rsquared)
    return r2s

specs = [
    ("FB87: fs(n)~rx(n), 1964–1984",
     fb87, {n: [f"fs{n}"] for n in range(2,6)}),
    ("CP05: CP~rx(n), 1964–2003",
     cp_data, {n: ["CP"] for n in range(2,6)}),
    ("LN09 row(b): F̄6~rx(n), 1964–2003",
     ln_data, {n: F6_BIC for n in range(2,6)}),
    ("LN09 row(c): F̄6+CP~rx(n), 1964–2003",
     ln_data, {n: F6_BIC+["CP"] for n in range(2,6)}),
    ("LN09 row(g): F6t~rx(n), 1964–2003",
     ln_data, {n: ["F6t"] for n in range(2,6)}),
    ("LN09 row(h): F5t+CP~rx(n), 1964–2003",
     ln_data, {n: ["F5t","CP"] for n in range(2,6)}),
]
for label, data, xcols in specs:
    r2s = get_r2s(data, xcols)
    vals = "".join(f" {r:6.4f}" for r in r2s)
    print(f"{label:<45}{vals}")

print()
print("PAPER REPORTED VALUES:")
print(f"{'FB87 Table 1':<45}  0.14   0.13   0.11   0.05")
print(f"{'CP05 Table 1B (CP single factor)':<45}  0.32   0.34   0.37   0.35")
print(f"{'CP05 Table 2 (FB benchmark)':<45}  0.16   0.17   0.18   0.09")
print(f"{'LN09 Table 2 row(b) F̄6 macro only':<45}  0.26   0.18   0.16   0.14")
print(f"{'LN09 Table 2 row(c) F̄6 + CP':<45}  0.45   0.44   0.45   0.42")
print(f"{'LN09 Table 2 row(g) F6t single':<45}  0.26   0.24   0.23   0.21")
print(f"{'LN09 Table 2 row(h) F5t + CP':<45}  0.44   0.44   0.45   0.42")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7: VERIFY ALIGNMENT WITH RFS2009
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("ALIGNMENT CHECK: our rx vs RFS2009 yr (our 1963-01 = RFS2009 1964-01)")
print("-" * 72)

try:
    import os
    rfs_path = os.path.join(BASE, "RFS2009.xls")
    df_rfs = pd.read_excel(rfs_path, sheet_name="Data Table2", header=4, skiprows=[5])
    df_rfs.columns = ["date","CP_rfs","rf1","rf2","rf3","rf4","rf5","rf6",
                      "rf7","rf8","RF5","RF6","_","yr2","yr3","yr4","yr5"]
    df_rfs["date"] = pd.to_datetime(df_rfs["date"], errors="coerce")
    for c in ["yr2","yr3","yr4","yr5","RF5","RF6"]:
        df_rfs[c] = pd.to_numeric(df_rfs[c], errors="coerce")
    df_rfs["ym_rfs"] = df_rfs["date"].dt.to_period("M")
    df_rfs = df_rfs.dropna(subset=["yr2"]).reset_index(drop=True)

    # our cp_data has ym (purchase date); RFS2009 labels 12 months later
    our = cp_data[["ym"] + [f"rx{n}" for n in range(2,6)]].copy()
    our["ym_rfs"] = our["ym"] + 12          # our 1963-01 → rfs 1964-01

    merged = our.merge(df_rfs[["ym_rfs","yr2","yr3","yr4","yr5"]], on="ym_rfs")
    print(f"  Matched rows: {len(merged)}")

    for n in range(2, 6):
        corr = merged[f"rx{n}"].corr(merged[f"yr{n}"])
        max_diff = (merged[f"rx{n}"] - merged[f"yr{n}"]).abs().max()
        print(f"  rx{n} vs yr{n}: corr={corr:.8f},  max|diff|={max_diff:.2e}")

    print(f"\n  First 5 rows (our rx2 vs RFS2009 yr2):")
    print(f"  {'our ym':>10}  {'rfs ym':>10}  {'our rx2':>10}  {'rfs yr2':>10}  {'diff':>10}")
    for _, row in merged.head(5).iterrows():
        diff = row["rx2"] - row["yr2"]
        print(f"  {str(row['ym']):>10}  {str(row['ym_rfs']):>10}  "
              f"{row['rx2']:>10.6f}  {row['yr2']:>10.6f}  {diff:>10.2e}")
except Exception as e:
    print(f"  [RFS2009 check skipped: {e}]")

print("\nDone.")
