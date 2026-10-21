"""Compare revised macro predictors against prior-month FRED-MD snapshots."""

import argparse
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from bond_rp.config import DATA_DIR
from bond_rp.data import load_gsw_bond_data, load_macro_raw_transformed
from bond_rp.models import _numeric_pipeline, available_features, matured_training
from bond_rp.robustness import online_combination, paired_sharpe_bootstrap
from bond_rp.strategy import annual_backtest
from bond_rp.tables import aligned_forecasts, prediction_table
from bond_rp.vintages import VINTAGE_URLS, VintageArchive
from audit_research_choices import cache_key, read_predictions


def vintage_forecasts(bonds, latest, archive, lag):
    values, audits = {"vintage": {}, "revised_matched": {}, "revised_full": {}}, []
    # No sentiment file is needed here: use every fully realized bond label.
    end = min(bonds.sx_bar.dropna().index.max(), max(archive.members)+1)
    for date in pd.period_range("2000-01", end, freq="M"):
        vintage, macro = archive.at(date)
        # Matched comparator uses the same named universe known in this vintage.
        # Vintage-specific transformation codes remain in the historical panel;
        # the latest panel has its own codes, so residual definition changes are
        # included in this sensitivity, not attributed solely to value revisions.
        common = macro.columns.intersection(latest.columns)
        for source, panel in [("vintage", macro), ("revised_matched", latest[common]),
                              ("revised_full", latest)]:
            features = available_features(bonds, panel, publication_lag=lag)
            cols = [c for c in features if c.startswith("macro::")]
            data = features.join(bonds.sx_bar)
            train = matured_training(data, date).dropna(subset=["sx_bar", "TI_avg", "f1", "f2", "f3", "f4", "f5"])
            train = train.loc[train[cols].notna().any(axis=1)]
            y = train.sx_bar.to_numpy()
            macro_pipe = _numeric_pipeline(pca=True)
            Z = macro_pipe.fit_transform(train[cols])
            z = macro_pipe.transform(data.loc[[date], cols])
            market_cols = [f"f{n}" for n in range(1, 6)]+["TI_avg"]
            pipe = _numeric_pipeline()
            M = pipe.fit_transform(train[market_cols])
            m = pipe.transform(data.loc[[date], market_cols])
            row = {"EH": y.mean()}
            for name, X, x in [("CP", M[:, :5], m[:, :5]), ("TI", M[:, [5]], m[:, [5]]),
                                ("LN", Z, z), ("CP+LN", np.hstack([M[:, :5], Z]), np.hstack([m[:, :5], z]))]:
                for penalty in ([0, 100] if "LN" in name else [0]):
                    model = Ridge(alpha=penalty) if penalty else LinearRegression()
                    label = name+("_R100" if penalty else "")
                    # Entirely absent current macro row is an unavailable signal,
                    # not a valid "real-time" forecast equal to its intercept.
                    row[label] = (float(model.fit(X, y).predict(x)[0])
                                  if "LN" not in name or data.loc[date, cols].notna().any() else np.nan)
            row["EW3"] = np.mean([row[n] for n in ["CP", "TI", "LN"]])
            values[source][date] = row
            audits.append({"forecast_month": str(date), "source": source, "lag": lag,
                           "vintage": str(vintage) if source == "vintage" else "2026-04",
                           "latest_reference": str(date-lag), "train_n": len(train),
                           "train_last_label_end": str(train.index.max()+12),
                           "panel_columns": len(cols),
                           "current_nonmissing": int(data.loc[date, cols].notna().sum())})
        if date.month == 1 and date.year % 5 == 0:
            print(f"  vintage lag {lag}: {date}", flush=True)
    return {s: pd.DataFrame.from_dict(v, orient="index") for s, v in values.items()}, pd.DataFrame(audits)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Download the official ZIP archives if missing")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"output"/"research_audit"/"vintages")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, url in VINTAGE_URLS.items():
        path = Path(DATA_DIR)/name
        if not path.exists():
            if not args.download:
                raise FileNotFoundError(f"{path}; rerun with --download")
            urllib.request.urlretrieve(url, path)
    bonds = load_gsw_bond_data().set_index("ym")
    latest = load_macro_raw_transformed("fredmd")
    rows, policy_rows, coverage, bootstrap_rows, offsets, costs, risk_grid = [], [], [], [], [], [], []
    files = [Path(DATA_DIR)/name for name in [*VINTAGE_URLS, "GSW_yields.csv", "2026-04-MD.csv"]]
    with VintageArchive(DATA_DIR) as archive:
        for lag in (1, 2, 3):
            key = cache_key(files, {"lag": lag, "protocol": "monthly-vintage-v2-require-current-input"})
            marker = args.output_dir/f"cache_key_lag{lag}.txt"
            if not args.refresh and marker.exists() and marker.read_text() == key:
                predictions = {s: read_predictions(args.output_dir/f"predictions_{s}_lag{lag}.csv")
                               for s in ("vintage", "revised_matched", "revised_full")}
            else:
                predictions, audit = vintage_forecasts(bonds, latest, archive, lag)
                for source, p in predictions.items():
                    p.index.name = "ym"
                    p.to_csv(args.output_dir/f"predictions_{source}_lag{lag}.csv")
                audit.to_csv(args.output_dir/f"audit_lag{lag}.csv", index=False)
                marker.write_text(key)
            for source, p in predictions.items():
                coverage.append({"source": source, "lag": lag, "dates": len(p),
                                 "dates_with_all_signals": len(p.dropna())})
                if len(p.dropna()) < 24:
                    print(f"  {source}, lag {lag}: insufficient published current inputs; no score", flush=True)
                    continue
                adaptive, audit = online_combination(p, bonds.sx_bar, list(p.columns.difference(["EW3"])))
                p = p.join(adaptive)
                audit.to_csv(args.output_dir/f"selector_{source}_lag{lag}.csv", index=False)
                aligned = aligned_forecasts(p, bonds)
                scores = prediction_table(aligned).reset_index()
                scores["source"], scores["lag"] = source, lag
                rows.append(scores)
                for method, cap in [("scalar", 3), ("long_only", 1), ("vol_target", 1)]:
                    summary, trades = annual_backtest(bonds, aligned.drop(columns="actual"), method=method, cap=cap)
                    summary["source"], summary["lag"], summary["policy"] = source, lag, method
                    policy_rows.append(summary.reset_index())
                    trades.to_csv(args.output_dir/f"trades_{source}_lag{lag}_{method}.csv", index=False)
                    if source == "vintage" and lag == 2:
                        for model in ["LN", "LN_R100", "EW3", "EW_mature"]:
                            row = paired_sharpe_bootstrap(trades, model)
                            row["policy"] = method
                            bootstrap_rows.append(row)
                        for month in range(1, 13):
                            stat, _ = annual_backtest(bonds, aligned.drop(columns="actual"), method=method,
                                                       cap=cap, rebalance_month=month)
                            stat["policy"], stat["month"] = method, month
                            offsets.append(stat.reset_index())
                        for cost in (0, .5, 2, 5, 10):
                            stat, _ = annual_backtest(bonds, aligned.drop(columns="actual"), method=method,
                                                       cap=cap, cost_bps=cost)
                            stat["policy"], stat["cost_bps"] = method, cost
                            costs.append(stat.reset_index())
                if source == "vintage" and lag == 2:
                    for gamma in (1, 2, 3, 5, 10, 20):
                        for cap in (1, 1.5, 2, 3):
                            stat, _ = annual_backtest(bonds, aligned[["EH", "LN", "LN_R100", "EW3", "EW_mature"]],
                                                       gamma=gamma, cap=cap)
                            stat["gamma"], stat["cap"] = gamma, cap
                            risk_grid.append(stat.reset_index())
    pd.concat(rows, ignore_index=True).to_csv(args.output_dir/"forecast_summary.csv", index=False)
    pd.concat(policy_rows, ignore_index=True).to_csv(args.output_dir/"policy_summary.csv", index=False)
    pd.DataFrame(coverage).to_csv(args.output_dir/"coverage.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(args.output_dir/"sharpe_intervals.csv", index=False)
    for name, frames in [("offset_sensitivity", offsets), ("cost_sensitivity", costs), ("risk_grid", risk_grid)]:
        pd.concat(frames, ignore_index=True).to_csv(args.output_dir/f"{name}.csv", index=False)


if __name__ == "__main__":
    main()
