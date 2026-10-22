"""Rolling estimates, shrinkage and correctly repriced monthly Treasury trades."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from bond_rp.config import DATA_DIR
from bond_rp.data import load_macro_raw_transformed, load_sentiment
from bond_rp.metrics import loss_test, oos_r2
from bond_rp.strategy import annual_backtest
from bond_rp.trading import RF_URL, combine_windows, forecast_designs, load_monthly_market, monthly_backtest, monthly_performance
from bond_rp.vintages import VINTAGE_URLS, VintageArchive


def fingerprint(files, settings):
    h = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    paths = [*sorted((ROOT/"bond_rp").glob("*.py")), Path(__file__)]
    paths += [Path(DATA_DIR)/f for f in files]
    for p in paths:
        h.update(p.name.encode()); h.update(p.read_bytes())
    import sklearn
    h.update(f"{np.__version__}|{pd.__version__}|{sklearn.__version__}".encode())
    return h.hexdigest()


def forecasts(tag, bonds, macro, sentiment, args, horizon, archive=None):
    folder = args.output_dir/tag
    folder.mkdir(parents=True, exist_ok=True)
    files = ["GSW_yields.csv", "2026-04-MD.csv", "F-F_Research_Data_Factors_CSV.zip"]
    if sentiment is not None:
        files.append("SENTIMENT.xlsx")
    if archive is not None:
        files += list(VINTAGE_URLS)
    end = "2024-02" if sentiment is not None else "2026-01"
    start = "1985-01" if archive is None else "1999-09"
    key = fingerprint(files, {"horizon": horizon, "start": start, "end": end, "lag": 2})
    marker = folder/"cache_key.txt"
    if not args.refresh and marker.exists() and marker.read_text() == key:
        print(f"Verified forecast cache: {tag}", flush=True)
        p = pd.read_csv(folder/"predictions.csv")
    else:
        print(f"Forecasting {tag}", flush=True)
        p, audit = forecast_designs(bonds, macro, sentiment, horizon=horizon,
                                   start=start, end=end, vintage_archive=archive)
        p.to_csv(folder/"predictions.csv", index=False)
        audit.to_csv(folder/"forecast_audit.csv", index=False)
        marker.write_text(key)
    return p


def annual_experiment(bonds, predictions, args):
    folder = args.output_dir/"annual_rolling"
    p = predictions.copy()
    p["prediction"] = p[[f"mu{n}" for n in range(2, 6)]].mean(axis=1)
    wide = p.pivot(index="ym", columns="model", values="prediction")
    wide.index = pd.PeriodIndex(wide.index, freq="M")
    rows, scores = [], []
    # Map EH/exp to the backtest's comparison label, retain the window-specific baselines.
    wide["EH"] = wide["EH/exp"]
    for start in ("1985-01", "1990-01", "2000-01", "2013-01"):
        subset = wide.loc[start:].dropna()
        actual = bonds.loc[subset.index, "sx_bar"]
        for name in subset:
            scores.append({"model": name, "origin": start,
                           "r2_pct": 100*oos_r2(actual, subset[name], subset.EH)})
        for policy, cap in [("scalar", 3), ("long_only", 1), ("vol_target", 1)]:
            summary, trades = annual_backtest(bonds, subset, method=policy, cap=cap)
            summary["origin"], summary["policy"] = start, policy
            rows.append(summary.reset_index())
            if start == "1990-01":
                trades.to_csv(folder/f"trades_{policy}.csv", index=False)
    pd.concat(rows, ignore_index=True).to_csv(folder/"policy_summary.csv", index=False)
    pd.DataFrame(scores).to_csv(folder/"forecast_summary.csv", index=False)


def monthly_experiment(tag, bonds, p, args):
    folder = args.output_dir/tag
    p, window_audit = combine_windows(bonds, p)
    p.to_csv(folder/"combined_predictions.csv", index=False)
    window_audit.to_csv(folder/"window_selection_audit.csv", index=False)
    start = "2000-01" if tag == "monthly_vintage" else "1990-01"
    rows = []
    # All candidate forecasts and policies are retained, including poor performers.
    for policy in ("long_only", "mv", "vol_target", "diagonal", "duration_neutral"):
        cap = 2 if policy == "mv" else 1
        summary, trades = monthly_backtest(bonds, p, start=start, policy=policy, cap=cap)
        rows.append(summary)
        trades.to_csv(folder/f"trades_{policy}.csv", index=False)
        print(f"  {tag}: {policy} evaluated", flush=True)
    pd.concat(rows, ignore_index=True).to_csv(folder/"policy_summary.csv", index=False)
    wide = p.assign(prediction=p[[f"mu{n}" for n in range(2, 6)]].mean(axis=1)).pivot(index="ym", columns="model", values="prediction")
    wide.index = pd.PeriodIndex(wide.index, freq="M")
    aligned = wide.loc[start:].join(bonds.mx_bar.rename("actual")).dropna()
    scores = []
    from statsmodels.stats.multitest import multipletests
    for name in wide:
        reference = aligned["EH/"+name.split("/")[-1]]
        used = reference+.5*(aligned[name]-reference)
        stat, value = loss_test(aligned.actual, used, aligned["EH/exp"], lags=6)
        scores.append({"model": name, "n": len(aligned), "start": str(aligned.index.min()), "end": str(aligned.index.max()),
                       "raw_r2_pct": 100*oos_r2(aligned.actual, aligned[name], aligned["EH/exp"]),
                       "r2_pct": 100*oos_r2(aligned.actual, used, aligned["EH/exp"]), "hac_stat": stat, "hac_p": value})
    scores = pd.DataFrame(scores)
    scores["holm_p"] = multipletests(scores.hac_p, method="holm")[1]
    scores.to_csv(folder/"forecast_summary.csv", index=False)
    sensitivity, periods = [], []
    # Fixed display rules; do not pick the model with the best OOS Sharpe.
    chosen = [f"{m}/{w}" for m in ("EH", "LN_R100", "ALL_R100", "EW_R100") for w in ("exp", "w120", "w240")]
    chosen += [f"{m}/{w}" for m in ("EW", "EH") for w in ("mix", "online", "select")]
    subset = p[p.model.isin(chosen)]
    for origin in ("1985-01", "1990-01", "2000-01", "2013-01"):
        if tag == "monthly_vintage" and origin < "2000-01":
            continue
        summary, _ = monthly_backtest(bonds, subset, start=origin, policy="long_only", cap=1)
        summary["origin"] = origin
        sensitivity.append(summary.assign(check="origin"))
    scenarios = [dict(shrink=s) for s in (0, .25, .5, 1)]
    scenarios += [dict(cost_bps=c) for c in (2, 5, 10)]
    scenarios += [dict(delay=1), dict(funding="rf_y1"), dict(funding="rf_nss"), dict(risk_window=120)]
    for scenario in scenarios:
        summary, _ = monthly_backtest(bonds, subset, start=start, policy="long_only", cap=1, **scenario)
        sensitivity.append(summary.assign(check=json.dumps(scenario, sort_keys=True)))
    pd.concat(sensitivity, ignore_index=True).to_csv(folder/"sensitivity.csv", index=False)
    # Performance by calendar regimes, compounded from actual monthly returns.
    for policy in ("long_only", "mv", "vol_target", "diagonal", "duration_neutral"):
        trades = pd.read_csv(folder/f"trades_{policy}.csv")
        for label, lo, hi in [("1990–2012", "1990-01", "2012-12"), ("2013+", "2013-01", "2100-12")]:
            for model, group in trades.groupby("model"):
                g = group.loc[(group.purchase >= lo) & (group.purchase <= hi)]
                if len(g) >= 24:
                    result = monthly_performance(g.return_net, g.risk_free)
                    result.update(model=model, policy=policy, period=label)
                    periods.append(result)
    pd.DataFrame(periods).to_csv(folder/"subperiods.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"output"/"trading_designs")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--skip-vintages", action="store_true")
    parser.add_argument("--download", action="store_true", help="Download the official RF file if absent")
    args = parser.parse_args()
    cash_path = Path(DATA_DIR)/"F-F_Research_Data_Factors_CSV.zip"
    if not cash_path.exists():
        if not args.download:
            raise FileNotFoundError(f"{cash_path}; rerun with --download")
        urllib.request.urlretrieve(RF_URL, cash_path)
    bonds = load_monthly_market()
    macro, sentiment = load_macro_raw_transformed("fredmd"), load_sentiment()
    p = forecasts("annual_rolling", bonds, macro, sentiment, args, horizon=12)
    annual_experiment(bonds, p, args)
    p = forecasts("monthly_revised", bonds, macro, sentiment, args, horizon=1)
    monthly_experiment("monthly_revised", bonds, p, args)
    if not args.skip_vintages:
        with VintageArchive(DATA_DIR) as archive:
            p = forecasts("monthly_vintage", bonds, macro, None, args, horizon=1, archive=archive)
        monthly_experiment("monthly_vintage", bonds, p, args)


if __name__ == "__main__":
    main()
