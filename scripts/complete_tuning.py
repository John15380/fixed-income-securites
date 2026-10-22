"""Complete the disclosed retrospective search, with separate investment calibration.

This study preserves the earlier limited experiment. Its historical evaluation
periods have already been inspected; they are not a fresh, untouched holdout.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".cache"/"matplotlib"))
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from statsmodels.stats.multitest import multipletests
from bond_rp.config import DATA_DIR
from bond_rp.data import load_macro_raw_transformed
from bond_rp.inference import joint_block_bootstrap_t
from bond_rp.metrics import loss_test, oos_r2
from bond_rp.replication import run_replication
from bond_rp.validated_models import (calibration, fred_snapshot, load_long_market,
                                     make_features)
from bond_rp.vintages import VintageArchive

PROTOCOL = dict(
    train_start="1994-01", model_validation_start="2011-01", model_freeze="2014-01",
    allocation_validation_start="2014-01", allocation_split="2015-07", freeze="2017-01",
    test_sale_end="2022-12", followup_sale_end="2025-12",
    en_trials=64, xgb_trials=128, seeds=[42, 73],
    windows=["expanding", "120", "240"], representations=["index11", "primitive32", "bounded32"],
    gc_ratios=[0., .25, .5, 1., 2.],
    model_objective="mean maturity-specific validation SSE / common expanding EH SSE",
    allocation_objective="0.5*mean+0.5*min of two late-validation net CER gains versus EH",
    gamma=5., cap=1., cost_bps=.5, risk_window=60, macro_lag=2, vintage_lag=1,
    xgb_engine="multioutput one_output_per_tree; hist; n_jobs=1",
    selection="best of four learner families by pre-2014 predictive validation only",
    evaluation="retrospective; all historical evaluation periods previously inspected",
)
TAGS = ("core_revised", "headline_vintage", "headline_revised")
MODELS = ("EN_real", "EN_GC", "XGB_real", "XGB_GC")


def versions():
    import sklearn, xgboost, optuna
    return dict(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                xgboost=xgboost.__version__, optuna=optuna.__version__)


def hashes(*, fitting_only=False):
    modules = ["config", "data", "models", "coursework", "trading", "vintages",
               "validated_models", "tuning_features", "tuning_search"]
    if not fitting_only:
        modules += ["allocation_calibration", "tuning_evaluation", "inference", "metrics"]
    files = [ROOT/"bond_rp"/(n+".py") for n in modules]+[Path(__file__)]
    files += [Path(DATA_DIR)/n for n in ("GSW_yields.csv", "2026-04-MD.csv", "CPILFESL.csv",
              "F-F_Research_Data_Factors_CSV.zip", "fred_md_vintages_1999_2014.zip",
              "fred_md_vintages_2015_2025.zip")]
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def fingerprint(tag, horizon, inherited=None):
    return hashlib.sha256(json.dumps(dict(sources=hashes(fitting_only=True), versions=versions(),
        protocol=PROTOCOL, tag=tag, horizon=horizon, inherited=inherited), sort_keys=True).encode()).hexdigest()


def build_folds(bonds, macro, core, tag, horizon, folder, key):
    from bond_rp.tuning_features import prepare_tuning_fold
    cache = folder/"folds.pkl"
    marker = folder/"fold_key.txt"
    if cache.exists() and marker.exists() and marker.read_text() == key:
        with cache.open("rb") as source:
            return pickle.load(source)
    if tag == "core_revised":
        levels = core
    else:
        raw = pd.read_csv(Path(DATA_DIR)/"2026-04-MD.csv").iloc[1:]
        levels = pd.Series(pd.to_numeric(raw.CPIAUCSL, errors="coerce").to_numpy(),
            index=pd.PeriodIndex(pd.to_datetime(raw.sasdate, format="mixed"), freq="M"))
    static = make_features(bonds, macro, levels)
    target = [f"{'rx' if horizon == 12 else 'mx'}{n}" for n in range(2, 6)]
    end = min(bonds.dropna(subset=target).index.max(), pd.Period(PROTOCOL["followup_sale_end"], "M")-horizon)
    dates = pd.period_range(PROTOCOL["model_validation_start"], end, freq="M")
    folds = {w: {} for w in PROTOCOL["windows"]}
    archive = VintageArchive(DATA_DIR) if tag == "headline_vintage" else None
    try:
        for date in dates:
            features, vintage = static, "revised"
            if archive:
                vintage, snapshot, cpi = fred_snapshot(archive, date)
                features = make_features(bonds, snapshot, cpi)
            for window in PROTOCOL["windows"]:
                fold = prepare_tuning_fold(bonds, features, date, horizon, window=window)
                fold["audit"]["vintage"] = str(vintage)
                folds[window][date] = fold
            if date.month == 1 and date.year % 3 == 2:
                print(f"  {tag}/h{horizon}: folded through {date}", flush=True)
    finally:
        if archive:
            archive.close()
    with cache.open("wb") as destination:
        pickle.dump(folds, destination, protocol=pickle.HIGHEST_PROTOCOL)
    marker.write_text(key)
    return folds


def panel(bonds, macro, core, tag, horizon, output, inherited=None):
    from bond_rp.tuning_search import predict_config, search
    folder = output/tag/f"h{horizon}"
    folder.mkdir(parents=True, exist_ok=True)
    key = fingerprint(tag, horizon, inherited)
    marker = folder/"cache_key.txt"
    needed = ("predictions.csv", "frozen_parameters.json", "forecast_audit.csv", "validation_predictions.csv")
    if marker.exists() and marker.read_text() == key and all((folder/n).exists() for n in needed):
        print(f"Verified completed search cache {tag}/h{horizon}", flush=True)
        return json.loads((folder/"frozen_parameters.json").read_text())
    folds = build_folds(bonds, macro, core, tag, horizon, folder, key)
    dates = list(folds["expanding"])
    target = [f"{'rx' if horizon == 12 else 'mx'}{n}" for n in range(2, 6)]
    cutoff = pd.Period(PROTOCOL["model_freeze"], "M")
    early = [d for d in dates if d+horizon < cutoff]
    actual = bonds.loc[early, target].to_numpy()
    eh = np.vstack([folds["expanding"][d]["reference"]["EH"] for d in early])
    if inherited is None:
        frozen, early_prediction = search({w: [folds[w][d] for d in early] for w in folds},
            actual, eh, folder, en_trials=PROTOCOL["en_trials"], xgb_trials=PROTOCOL["xgb_trials"])
    else:
        frozen = json.loads(json.dumps(inherited))
        early_prediction = {name: np.vstack([predict_config(folds[str(frozen[name]["window"])][d],
            frozen[name]) for d in early]) for name in MODELS}
    selected = frozen["selected_family"]
    validation = []
    for name, values in early_prediction.items():
        for d, row in zip(early, values):
            validation.append(dict(ym=str(d), stage="model_selection", model=name,
                **{f"mu{n}": float(row[n-2]) for n in range(2, 6)}))
    # Later validation forecasts use hyperparameters selected only in the early
    # stage. Neither MSE shrinkage nor allocation calibration sees test labels.
    records = []
    for date in dates:
        if date < cutoff:
            continue
        forecasts = dict(folds["expanding"][date]["reference"])
        for name in MODELS:
            config = frozen[name]
            forecasts[name] = predict_config(folds[str(config["window"])][date], config)
        forecasts["ModelSelected"] = forecasts[selected]
        for name, values in forecasts.items():
            records.append(dict(ym=str(date), model=name,
                **{f"mu{n}": float(values[0, n-2]) for n in range(2, 6)}))
        if date.month == 1 and date.year % 3 == 2:
            print(f"  {tag}/h{horizon}: predictions through {date}", flush=True)
    p = pd.DataFrame(records)
    periods = pd.PeriodIndex(p.ym, freq="M")
    mature = periods+horizon < pd.Period(PROTOCOL["freeze"], "M")
    late_dates = pd.PeriodIndex(p.loc[mature & p.model.eq("EH"), "ym"], freq="M")
    bench = p[p.model.eq("EH")].set_index("ym")
    for name in (*MODELS, "ModelSelected"):
        raw = p[p.model.eq(name)].copy()
        vr = raw[pd.PeriodIndex(raw.ym, freq="M").isin(late_dates)]
        vb = bench.loc[vr.ym, [f"mu{n}" for n in range(2, 6)]].to_numpy()
        if inherited is None:
            shrink = calibration(bonds.loc[late_dates, target].to_numpy(),
                                 vr[[f"mu{n}" for n in range(2, 6)]].to_numpy(), vb)
        else:
            shrink = inherited["mse_shrinkage"][name]
        frozen.setdefault("mse_shrinkage", {})[name] = shrink
        for row in vr.to_dict("records"):
            row["stage"] = "allocation_validation"
            validation.append(row)
        for n in range(2, 6):
            b = bench.loc[raw.ym, f"mu{n}"].to_numpy()
            raw[f"mu{n}"] = b+shrink*(raw[f"mu{n}"].to_numpy()-b)
        raw["model"] = name+"_mse"
        p = pd.concat([p, raw], ignore_index=True)
    p.to_csv(folder/"predictions.csv", index=False)
    pd.DataFrame(validation).to_csv(folder/"validation_predictions.csv", index=False)
    audit = []
    for window, data in folds.items():
        for date, fold in data.items():
            row = dict(fold["audit"])
            # Nested bounds remain in the fold cache; CSV stores JSON safely.
            audit.append({k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                          for k, v in row.items()})
    pd.DataFrame(audit).to_csv(folder/"forecast_audit.csv", index=False)
    frozen["dating"] = dict(model_validation_last_purchase=str(max(early)),
        model_validation_last_label_end=str(max(early)+horizon), model_freeze=str(cutoff),
        allocation_validation_last_purchase=str(max(late_dates)),
        allocation_validation_last_label_end=str(max(late_dates)+horizon), freeze=PROTOCOL["freeze"])
    (folder/"frozen_parameters.json").write_text(json.dumps(frozen, indent=2))
    (folder/"fit_metadata.json").write_text(json.dumps(dict(sources=hashes(fitting_only=True),
        versions=versions(), protocol=PROTOCOL, tag=tag, horizon=horizon), indent=2))
    marker.write_text(key)
    print(f"Finished {tag}/h{horizon}, selected {selected}", flush=True)
    return frozen


def forecast_scores(bonds, p, horizon, tag):
    p = p.copy(); p["ym"] = pd.PeriodIndex(p.ym, freq="M")
    target = [f"{'rx' if horizon == 12 else 'mx'}{n}" for n in range(2, 6)]
    out = []
    for era, start, end in (("test", "2017-01", "2022-12"), ("followup", "2023-01", "2025-12")):
        q = p[(p.ym >= pd.Period("2017-01", "M")) & (p.ym+horizon >= pd.Period(start, "M")) &
              (p.ym+horizon <= pd.Period(end, "M"))]
        benchmark = q[q.model.eq("EH")].set_index("ym")
        rows, differences = [], []
        for name, g in q.groupby("model", sort=True):
            dates = pd.PeriodIndex(g.ym, freq="M")
            actual = bonds.loc[dates, target].to_numpy()
            prediction = g[[f"mu{n}" for n in range(2, 6)]].to_numpy()
            eh = benchmark.loc[dates, [f"mu{n}" for n in range(2, 6)]].to_numpy()
            for j, maturity in enumerate((2, 3, 4, 5, "basket")):
                y, yhat, b = ((a[:, j] for a in (actual, prediction, eh)) if j < 4 else
                              (a.mean(axis=1) for a in (actual, prediction, eh)))
                _, pv = loss_test(y, yhat, b, lags=18 if horizon == 12 else 3)
                rows.append(dict(panel=tag, horizon=horizon, period=era, model=name, maturity=maturity,
                    n=len(dates), purchase_start=str(dates.min()), purchase_end=str(dates.max()),
                    r2_pct=100*oos_r2(y, yhat, b), raw_hac_p=pv))
                if j == 4 and name != "EH":
                    differences.append((name, (y-b)**2-(y-yhat)**2))
        table = pd.DataFrame(rows)
        eligible = table.model.ne("EH")
        table.loc[eligible, "holm_p"] = multipletests(table.loc[eligible, "raw_hac_p"], method="holm")[1]
        joint = joint_block_bootstrap_t(np.column_stack([x for _, x in differences]),
            lags=18 if horizon == 12 else 3, block=12 if horizon == 12 else 6, replications=1999)
        for i, (name, _) in enumerate(differences):
            table.loc[table.model.eq(name) & table.maturity.eq("basket"), "max_t_p"] = joint["max_t_p"][i]
        out.append(table)
    return pd.concat(out, ignore_index=True)


def evaluate(output, bonds):
    from bond_rp.tuning_evaluation import evaluate_panel
    scores, performances, utilities = [], [], []
    for tag in TAGS:
        for horizon in (1, 12):
            folder = output/tag/f"h{horizon}"
            p = pd.read_csv(folder/"predictions.csv")
            s = forecast_scores(bonds, p, horizon, tag)
            s.to_csv(folder/"forecast_scores.csv", index=False); scores.append(s)
            if horizon == 1:
                inherited = None
                if tag == "headline_revised":
                    inherited = json.loads((output/"headline_vintage"/"h1"/
                                            "allocation_parameters.json").read_text())
                result = evaluate_panel(bonds, p, folder, inherited_parameters=inherited)
                perf, utility = result["performance"].copy(), result["utility"].copy()
                perf["panel"], utility["panel"] = tag, tag
                performances.append(perf); utilities.append(utility)
                print(f"Evaluated cash flows and validation-only allocation: {tag}", flush=True)
    pd.concat(scores).to_csv(output/"all_forecast_scores.csv", index=False)
    pd.concat(performances).to_csv(output/"all_performance.csv", index=False)
    pd.concat(utilities).to_csv(output/"all_utility.csv", index=False)
    (output/"metadata.json").write_text(json.dumps(dict(sources=hashes(), versions=versions(),
        protocol=PROTOCOL), indent=2))


def load_inputs():
    bonds, macro = load_long_market(), load_macro_raw_transformed("fredmd")
    raw = pd.read_csv(Path(DATA_DIR)/"CPILFESL.csv")
    core = pd.Series(pd.to_numeric(raw.CPILFESL, errors="coerce").to_numpy(),
                     index=pd.PeriodIndex(pd.to_datetime(raw.iloc[:, 0]), freq="M"))
    return bonds, macro, core


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"output"/"completed_tuning")
    parser.add_argument("--panel", choices=TAGS)
    parser.add_argument("--horizon", choices=(1, 12), type=int)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # No optimization can start until current production code passes the paper
    # numeric checks. Each independent worker executes this prerequisite.
    _, checks, _, _ = run_replication()
    print(f"Prerequisite: {len(checks)} literature numerical checks passed", flush=True)
    (args.output_dir/"protocol.json").write_text(json.dumps(PROTOCOL, indent=2))
    bonds, macro, core = load_inputs()
    if not args.evaluate_only:
        for tag in ([args.panel] if args.panel else TAGS):
            for horizon in ([args.horizon] if args.horizon else (1, 12)):
                inherited = None
                if tag == "headline_revised":
                    parent = args.output_dir/"headline_vintage"/f"h{horizon}"/"frozen_parameters.json"
                    inherited = json.loads(parent.read_text())
                panel(bonds, macro, core, tag, horizon, args.output_dir, inherited)
    if args.evaluate_only or not args.panel:
        evaluate(args.output_dir, bonds)
    print("Completed requested tuning work", flush=True)


if __name__ == "__main__":
    main()
