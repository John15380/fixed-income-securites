"""Disclosed sensitivity grids and further strategy research (retrospective).

Run with OPENBLAS_NUM_THREADS=1 for fast repeated small matrix fits.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".cache"/"matplotlib"))

import numpy as np
import pandas as pd
from bond_rp.config import DATA_DIR
from bond_rp.data import (load_bond_data, load_gsw_bond_data,
                          load_macro_raw_transformed, load_sentiment)
from bond_rp.robustness import (family_forecasts, online_combination,
                                paired_sharpe_bootstrap, r2_decomposition)
from bond_rp.strategy import annual_backtest
from bond_rp.tables import aligned_forecasts, prediction_table

STARTS = ("1985-01", "1990-01", "1995-01", "2000-01")
RISK_MODELS = ["EH", "CP", "LN", "BW", "TI", "EW4", "EW_mature", "Select_mature"]


def cache_key(paths, settings):
    h = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for path in [*sorted((ROOT/"bond_rp").glob("*.py")),
                 ROOT/"scripts"/"audit_research_choices.py",
                 ROOT/"scripts"/"audit_macro_vintages.py", *paths]:
        h.update(path.name.encode())
        h.update(path.read_bytes())
    import sklearn
    h.update(f"{np.__version__}|{pd.__version__}|{sklearn.__version__}".encode())
    return h.hexdigest()


def read_predictions(path):
    frame = pd.read_csv(path, index_col=0)
    frame.index = pd.PeriodIndex(frame.index, freq="M")
    return frame


def run_dataset(tag, bonds, macro, sentiment, args):
    folder = args.output_dir/tag
    folder.mkdir(parents=True, exist_ok=True)
    raw_files = ["SENTIMENT.xlsx"]+(["Fama_Bliss_price.txt", "macro_raw_data.xlsx"]
                 if tag == "baseline" else ["GSW_yields.csv", "2026-04-MD.csv"])
    settings = {"start": "1985-01", "end": "2012-02" if tag == "baseline" else "2024-02",
                "lag": 2, "protocol": "family-audit-v1"}
    key = cache_key([Path(DATA_DIR)/p for p in raw_files], settings)
    if not args.refresh and (folder/"cache_key.txt").exists() and (folder/"cache_key.txt").read_text() == key:
        print(f"Using checked grid cache: {tag}", flush=True)
        forecasts = read_predictions(folder/"predictions.csv")
    else:
        print(f"Computing complete factor grid: {tag}", flush=True)
        result = family_forecasts(bonds, macro, sentiment, start=settings["start"], end=settings["end"])
        forecasts = result.predictions
        forecasts.to_csv(folder/"predictions.csv")
        result.audit.to_csv(folder/"forecast_audit.csv", index=False)
        (folder/"cache_key.txt").write_text(key)
    indexed = bonds.set_index("ym")
    # 45 subset/penalty specifications + EH + six MAs + multiscale model.
    candidates = [c for c in forecasts if c not in {"TrendVote", "EW4"}]
    forecasts, selections = forecasts.copy(), []
    starts, grids, offsets, regimes, bootstraps, vol_grid, subperiods = [], [], [], [], [], [], []
    for start in STARTS:
        subset = forecasts.loc[start:].copy()
        adaptive, audit = online_combination(subset, indexed.sx_bar, candidates)
        subset = subset.join(adaptive)
        audit["origin"] = start
        selections.append(audit)
        aligned = aligned_forecasts(subset.drop(columns="TrendVote"), indexed)
        scores = prediction_table(aligned).reset_index()
        scores["origin"] = start
        starts.append(scores)
        all_summary, all_trades = annual_backtest(indexed, aligned.drop(columns="actual"))
        all_summary.to_csv(folder/f"annual_all_{start[:4]}.csv")
        if start == "1990-01":
            subset.to_csv(folder/"research_predictions_1990.csv")
            scores.to_csv(folder/"forecast_summary.csv", index=False)
            all_trades.to_csv(folder/"annual_trades_default.csv", index=False)
            r2_decomposition(indexed, macro, sentiment, subset, start).to_csv(folder/"r2_decomposition.csv", index=False)
            risk_pred = aligned[RISK_MODELS]
            for target_vol in (.02, .04, .06, .08):
                summary, _ = annual_backtest(indexed, risk_pred, method="vol_target", cap=1,
                                              target_vol=target_vol)
                summary["target_vol"] = target_vol
                vol_grid.append(summary.reset_index())
            for gamma in (1, 2, 3, 5, 10, 20):
                for cap in (1, 1.5, 2, 3):
                    summary, _ = annual_backtest(indexed, risk_pred, gamma=gamma, cap=cap)
                    summary["gamma"], summary["cap"] = gamma, cap
                    grids.append(summary.reset_index())
            for method, cap in [("scalar", 3), ("long_only", 1), ("vol_target", 1), ("direct", 1)]:
                p = risk_pred if method != "direct" else subset[["EH", "TrendVote"]]
                p = p.reindex(aligned.index).copy()
                if method == "direct":
                    p["EH"] = 100.  # Direct-weight comparator = a fully invested basket.
                summary, trades = annual_backtest(indexed, p, method=method, cap=cap)
                summary["policy"] = method
                regimes.append(summary.reset_index())
                trades.to_csv(folder/f"annual_trades_{method}.csv", index=False)
                if method in {"long_only", "vol_target"}:
                    for period, start_year, end_year in [("1990–2012", 1990, 2012), ("2013+", 2013, 2100)]:
                        from bond_rp.metrics import annual_performance
                        for model, group in trades.groupby("model"):
                            dates = pd.PeriodIndex(group.purchase, freq="M")
                            g = group.loc[(dates.year >= start_year) & (dates.year <= end_year)]
                            if len(g) >= 2:
                                stat = annual_performance(g.return_net, g.risk_free)
                                stat.update(model=model, policy=method, period=period)
                                subperiods.append(stat)
                for model in summary.index.difference(["Cash", "BuyHold"]):
                    bootstrap = paired_sharpe_bootstrap(trades, model)
                    bootstrap["policy"] = method
                    bootstraps.append(bootstrap)
                if method in {"long_only", "vol_target"}:
                    for month in range(1, 13):
                        s, _ = annual_backtest(indexed, p, method=method, cap=cap, rebalance_month=month)
                        s["month"], s["policy"] = month, method
                        offsets.append(s.reset_index())
                    for cost in (0, .5, 2, 5, 10):
                        s, _ = annual_backtest(indexed, p, method=method, cap=cap, cost_bps=cost)
                        s["cost_bps"], s["policy"] = cost, method
                        s.to_csv(folder/f"cost_{method}_{cost:g}.csv")
        # Reset each online learner at each proposed origin; no 1985 errors in a 2000 start.
        summary, _ = annual_backtest(indexed, aligned[RISK_MODELS], method="long_only", cap=1)
        summary["origin"] = start
        summary.to_csv(folder/f"annual_long_only_{start[:4]}.csv")
    for name, frames in [("start_sensitivity", starts), ("risk_grid", grids),
                          ("policy_summary", regimes), ("offset_sensitivity", offsets),
                          ("selection_audit", selections)]:
        pd.concat(frames, ignore_index=True).to_csv(folder/f"{name}.csv", index=False)
    pd.DataFrame(bootstraps).to_csv(folder/"sharpe_intervals.csv", index=False)
    pd.concat(vol_grid, ignore_index=True).to_csv(folder/"vol_target_grid.csv", index=False)
    pd.DataFrame(subperiods).to_csv(folder/"subperiods.csv", index=False)
    print(f"Finished {tag}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"output"/"research_audit")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--dataset", choices=["baseline", "gsw", "both"], default="both")
    args = parser.parse_args()
    sentiment = load_sentiment()
    if args.dataset in {"baseline", "both"}:
        run_dataset("baseline", load_bond_data(), load_macro_raw_transformed("jln"), sentiment, args)
    if args.dataset in {"gsw", "both"}:
        run_dataset("gsw", load_gsw_bond_data(), load_macro_raw_transformed("fredmd"), sentiment, args)


if __name__ == "__main__":
    main()
