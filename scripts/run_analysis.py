"""
Research tables and robustness checks for the bond risk premium project.
See README.md for the limits of the historical evaluation protocol.

Usage:
    uv run scripts/run_analysis.py

Output order:
    Table 1  : in-sample R² and NW-t statistics
    Table 2  : OOS R², Clark-West, hit rate  (+OOS>IS explanation)
    Table 3  : economic evaluation (γ=5, W_CAP=3)  (+transaction costs)
    Table 4a : subperiod robustness
    Table 4b : γ sensitivity — expanded grid + empirical calibration
    Table 4b-ext : W_CAP sensitivity
    Table 4c : rolling-window (120m) OOS R²
    Table 4c-ext : OOS start-date sensitivity
    Table 4d : quarterly holding period
    Table 4e : time-varying σ² (Gargano et al. 2019)
    Table 4f : TI lookback sensitivity  [FIXED: was hardcoded at 12]
    Table 4g : model combination comparison  [FIXED: justifies LN+CP, LN+BW]
    Table 4h : publication lag robustness  [FIXED: temporal alignment]
    Table 4i : transaction cost sensitivity  [FIXED: was ignored]
    Table 4j : recursive LASSO vs. BIC pseudo-OOS  [NEW: look-ahead bias fix]
"""

from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from bond_rp.config import OOS_START, GAMMA_BASE
from bond_rp.data   import (
    load_bond_data, load_macro_factors_oos, load_macro_factors_insample,
    load_sentiment, load_bond_data_extended, load_macro_factors_oos_extended,
    load_macro_raw_transformed, load_cp15_components,
)
from bond_rp.models import build_all_oos
from bond_rp.tables import (
    run_insample_table,
    print_oos_table,
    print_strategy_table,
    robustness_subperiods,
    robustness_gamma,
    robustness_wcap,
    robustness_oos_start,
    robustness_rolling,
    robustness_quarterly,
    robustness_rolling_sigma,
    robustness_ti_lookback,
    robustness_model_combos,
    robustness_publag,
    robustness_transaction_costs,
    robustness_lasso,
    print_carry_rolldown_table,
    print_maturity_oos_table,
    print_maturity_strategy_table,
    print_rf_table,
    print_extended_oos_table,
    print_cp15_table,
)

print("=" * 72)
print("BOND RISK PREMIUM – STRATEGY ANALYSIS")
print("=" * 72)
print("Research specification: overlapping 12-month labels; see README limitations.")

print("\nLoading Fama-Bliss bond data ...")
fb = load_bond_data()
print(f"  Bond data: {fb['ym'].iloc[0]} – {fb['ym'].iloc[-1]}  ({len(fb)} rows)")

print("\nLoading macro factor data (OOS: PCA 1964-1989) ...")
df_factors, macro_last_ym = load_macro_factors_oos()

print("\nLoading macro factor data (in-sample: PCA 1964-2003) ...")
df_factors_is = load_macro_factors_insample()

print("\nLoading investor sentiment data (Baker-Wurgler / Laborda-Olmo) ...")
df_sent = load_sentiment()

# ── Table 1 ───────────────────────────────────────────────────────────────────
win_m = run_insample_table(fb, df_factors_oos=df_factors,
                           df_factors_is=df_factors_is, df_sent=df_sent)

# ── Table 2 ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("OUT-OF-SAMPLE RECURSIVE PREDICTIONS")
print("=" * 72)
preds, fb_ym = build_all_oos(fb, df_factors, macro_last_ym,
                              oos_start=OOS_START, window=None, df_sent=df_sent)
aligned = print_oos_table(preds, fb_ym, tag="Recursive expanding window")

# ── Table 3 ───────────────────────────────────────────────────────────────────
strat_res = print_strategy_table(preds, aligned, fb_ym, gamma=GAMMA_BASE)

# ── Table 4a ──────────────────────────────────────────────────────────────────
robustness_subperiods(preds, aligned, fb_ym, gamma=GAMMA_BASE)

# ── Table 4b : expanded γ grid + empirical calibration ───────────────────────
robustness_gamma(preds, aligned, fb_ym)

# ── Table 4b-ext : W_CAP sensitivity ─────────────────────────────────────────
robustness_wcap(preds, aligned, fb_ym, gamma=GAMMA_BASE)

# ── Table 4c : rolling window ─────────────────────────────────────────────────
robustness_rolling(fb, df_factors, macro_last_ym,
                   oos_start=OOS_START, roll_window=120, df_sent=df_sent)

# ── Table 4c-ext : OOS start-date sensitivity ────────────────────────────────
robustness_oos_start(fb, df_factors, macro_last_ym, df_sent=df_sent, gamma=GAMMA_BASE)

# ── Table 4d : quarterly holding period ──────────────────────────────────────
robustness_quarterly(fb)

# ── Table 4e : time-varying σ² ───────────────────────────────────────────────
robustness_rolling_sigma(preds, aligned, fb_ym, gamma=GAMMA_BASE, roll_vol=36)

# ── Table 4f : TI lookback sensitivity ───────────────────────────────────────
robustness_ti_lookback(fb, oos_start=OOS_START)

# ── Table 4g : model combination comparison ───────────────────────────────────
robustness_model_combos(fb, df_factors, macro_last_ym,
                        df_sent=df_sent, oos_start=OOS_START)

# ── Table 4h : publication lag robustness ─────────────────────────────────────
robustness_publag(fb, df_factors, macro_last_ym,
                  df_sent=df_sent, oos_start=OOS_START)

# ── Table 4i : transaction cost sensitivity ───────────────────────────────────
robustness_transaction_costs(preds, aligned, fb_ym, gamma=GAMMA_BASE)

# ── Table 4j : recursive LASSO vs. BIC pseudo-OOS ────────────────────────────
robustness_lasso(fb, df_factors, df_sent=df_sent,
                 oos_start=OOS_START, gamma=GAMMA_BASE)

# ── Table 5 : carry + rolldown decomposition (A2) ────────────────────────────
print_carry_rolldown_table(fb)

# ── Table 6 : per-maturity OOS R² (A1) ───────────────────────────────────────
print_maturity_oos_table(fb, df_factors, oos_start=OOS_START)

# ── Table 7 : per-maturity MV strategy (A1) ──────────────────────────────────
print_maturity_strategy_table(fb, df_factors, oos_start=OOS_START, gamma=GAMMA_BASE)

# ── Table 8 : Random Forest vs LN (B1) ────────────────────────────────────────
print("\nLoading raw macro panel (no PCA) for the machine-learning comparison ...")
df_macro_raw = load_macro_raw_transformed()

print_rf_table(fb, df_factors, df_macro_raw=df_macro_raw,
               oos_start=OOS_START, gamma=GAMMA_BASE)

# ── Table 9 : Extended OOS (GSW + FRED-MD) ────────────────────────────────────
print("\nLoading extended bond data (Fama-Bliss + GSW yields 2013-2026) ...")
fb_ext = load_bond_data_extended()
print(f"  Extended bond data: {fb_ext['ym'].iloc[0]} – {fb_ext['ym'].iloc[-1]}  "
      f"({len(fb_ext)} rows)")

print("\nLoading extended macro factors (FRED-MD PCA 1964-1989 → 2026) ...")
df_factors_ext, macro_last_ext = load_macro_factors_oos_extended()

print_extended_oos_table(fb_ext, df_factors_ext, macro_last_ext,
                         oos_start=OOS_START, gamma=GAMMA_BASE)

# ── Table 10 : Cieslak-Povala (2015) cyclical / trend decomposition ──────────
print("\nLoading Cieslak-Povala (2015) cyclical/trend decomposition ...")
df_cp15, cp15_last = load_cp15_components(fit_end="1989-12")

print_cp15_table(fb, df_cp15, df_factors, oos_start=OOS_START, gamma=GAMMA_BASE)

print("\n" + "=" * 72)
print("Done.")
print("=" * 72)
