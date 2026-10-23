"""Reconstruct the disclosed first-22-page reference pipeline, with assumptions marked.

Unlike the earlier independent study this uses the full validation interval,
15 named inputs, the actual MS recursive factor, SPF, fixed depth2/100 trees,
per-maturity annual-signal allocations and the slides' rolling-MV benchmark.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".cache"/"matplotlib"))
import numpy as np
import pandas as pd
from bond_rp.config import DATA_DIR
from bond_rp.data import load_macro_raw_transformed
from bond_rp.peer_allocation import allocation_path, risk_panel, summarize
from bond_rp.peer_learners import elastic_predict, xgb_predict
from bond_rp.peer_reference import candidate_menu, screen_candidates, peer_fold, BASE_NAMES
from bond_rp.replication import run_replication
from bond_rp.survey_expectations import load_spf
from bond_rp.validated_models import load_long_market, make_features

PROTOCOL = dict(train_start="1994-01", screening_end="2009-12", validation_start="2011-01",
    validation_end="2016-12", freeze="2017-01", test_start="2017-01", test_end="2022-12",
    horizon=12, strict_label_rule="purchase+12 < decision", macro_lag=2,
    preprocessing="matured training rows only", n_estimators=100, max_depth=2,
    xgb_trials_per_group=32, seeds=[42, 73], elasticnet="rolling purged inner CV; copula fitted inside folds",
    allocation_trials=128, gamma=5., cap=1., base_cost_bps=0.,
    cases=["dispersion", "expectations"], main_case="dispersion",
    target="annual log excess return, with actual monthly simple-return NAV",
    date_convention="formation dates; annual test outcomes finish 2023-12, monthly NAV 2023-01",
    status="reconstruction of disclosed design; not an exact numerical reproduction; all test history already seen")

ASSUMPTIONS = [
    dict(pages="3,13", item="ME", disclosed="SPF forecaster disagreement; attributed to Eriksen (2017)",
         implementation="Primary: IQR of respondent growth forecasts, PCA3 and supervised index. Separate original-paper median-expectation panel.",
         limitation="Slides and cited paper use different definitions; neither exact peer variable list nor aggregation is disclosed."),
    dict(pages="4,5", item="82 candidates", disclosed="about 20 series, transformation templates, cutoff |t|>2.5",
         implementation="Explicit 20-source x4-template +2-curve menu; standard OLS t-screen ending 2009-12; HAC t also saved.",
         limitation="The original 82 names/formulas and selected set are absent; this menu cannot be called identical."),
    dict(pages="10,13", item="10 literature +5 PCs", disclosed="four spread names, CP15, LN F5/F6, MS recursive, ME, CP05",
         implementation="Four FS +CP05+MS+CP15+LN5+LN6+ME; five PCs from screened generated features.",
         limitation="Full membership inferred from the top-feature chart plus the stated count; PC explained variance is measured, not forced to 93%."),
    dict(pages="11,12", item="learners", disclosed="dynamic ElasticNetCV; rolling copula; XGB depth2/100trees and strong L1/L2",
         implementation="Purged inner CV, training-only copula; XGB32 validation trials/group, two seeds; disclosed tree count and depth fixed.",
         limitation="Remaining penalties, GC size/window, seeds, and search range are not disclosed. Explicit validation-only ranges saved."),
    dict(pages="16-19", item="investment", disclosed="per-maturity mu/(gamma variance); positive signals; rolling60 MV; B,lambda_dd,lambda_g",
         implementation="Annual means/variance set per-maturity weights; monthly aging-bond NAV. Convex L2 exposure penalty and causal drawdown budget.",
         limitation="Gamma, caps, exact brake equations, transaction costs, holding/rebalancing and Optuna objective/ranges are absent; values are declared assumptions."),
    dict(pages="6,20,21", item="dates/performance", disclosed="1994-2010/2011-2016/2017-2022, 12month embargo,72 test observations",
         implementation="Formation-date test has72 annual forecasts and72 real monthly trades. Only60 validation annual labels and71 monthly trades mature before2017.",
         limitation="Annual test outcomes extend to2023-12. Also save overlapping-annual payoff statistics as diagnostics, never call them actual monthly NAV."),
]


def versions():
    import sklearn, xgboost, optuna
    return dict(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, xgboost=xgboost.__version__, optuna=optuna.__version__)


def source_hashes():
    names = ["data", "models", "coursework", "validated_models", "trading", "ms17",
             "survey_expectations", "peer_reference", "peer_learners", "peer_allocation"]
    files = [ROOT/"bond_rp"/(n+".py") for n in names]+[Path(__file__)]
    files += [Path(DATA_DIR)/n for n in ("GSW_yields.csv", "2026-04-MD.csv", "CPILFESL.csv",
        "SPFmicrodata.xlsx", "spf_release_dates.txt", "US_Policy_Uncertainty_Data.xlsx", "F-F_Research_Data_Factors_CSV.zip")]
    pdf = ROOT/"固定收益证券分析和模型-2026-05-25第3-5节.pdf"
    if pdf.exists():
        files.append(pdf)
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def inputs():
    bonds, macro = load_long_market(), load_macro_raw_transformed("fredmd")
    c = pd.read_csv(Path(DATA_DIR)/"CPILFESL.csv")
    cpi = pd.Series(c.CPILFESL.to_numpy(), index=pd.PeriodIndex(pd.to_datetime(c.iloc[:, 0]), freq="M"))
    features = make_features(bonds, macro, cpi)
    candidates = candidate_menu(bonds, macro)
    selected, screening = screen_candidates(bonds, candidates)
    dates = pd.period_range("1994-01", "2022-12", freq="M")
    surveys = dict(zip(PROTOCOL["cases"], reversed(load_spf(DATA_DIR, dates))))
    return bonds, features, candidates, selected, screening, surveys


def search_xgb(folds, dates, bonds, folder, label):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    actual = bonds.loc[dates, [f"rx{n}" for n in range(2, 6)]].to_numpy()
    benchmark = np.vstack([folds[d]["reference"]["EH"] for d in dates])
    denominator = ((actual-benchmark)**2).sum(0)
    results, best, bestpred = [], (np.inf, None), None
    def evaluate(config):
        prediction = np.vstack([xgb_predict(folds[d], config) for d in dates])
        return float(np.mean(((actual-prediction)**2).sum(0)/denominator)), prediction
    for seed in PROTOCOL["seeds"]:
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=8))
        initial = dict(learning_rate=.05, min_child_weight=10., reg_lambda=10., reg_alpha=1.)
        if label == "L4_GC":
            initial.update(ratio=1., window="120")
        study.enqueue_trial(initial)
        with ThreadPoolExecutor(max_workers=2) as executor:
            for offset in range(0, 16, 2):
                pending = []
                for _ in range(2):
                    trial = study.ask()
                    params = dict(learning_rate=trial.suggest_float("learning_rate", .02, .15, log=True),
                        min_child_weight=trial.suggest_float("min_child_weight", 5, 40, log=True),
                        reg_lambda=trial.suggest_float("reg_lambda", 1, 100, log=True),
                        reg_alpha=trial.suggest_float("reg_alpha", .001, 10, log=True))
                    config = dict(params=params, ratio=0., window=120)
                    if label == "L4_GC":
                        config.update(ratio=trial.suggest_categorical("ratio", [.5, 1., 2.]),
                                      window=int(trial.suggest_categorical("window", ["60", "120", "0"])))
                    pending.append((trial, config, executor.submit(evaluate, config)))
                for trial, config, future in pending:
                    value, prediction = future.result(); study.tell(trial, value)
                    results.append(dict(model=label, seed=seed, number=trial.number, value=value,
                                        config=json.dumps(config, sort_keys=True)))
                    if value < best[0]:
                        best, bestpred = (value, config), prediction
                if offset % 8 == 6:
                    print(f"  {folder.name} {label} seed{seed} {offset+2}/16, best loss {best[0]:.5f}", flush=True)
    pd.DataFrame(results).to_csv(folder/(label+"_trials.csv"), index=False)
    return dict(best[1], validation_loss=best[0]), bestpred


def run_case(case, output, data):
    bonds, features, candidates, selected, screening, surveys = data
    folder = output/case; folder.mkdir(parents=True, exist_ok=True)
    metadata = dict(sources=source_hashes(), versions=versions(), protocol=PROTOCOL, case=case)
    key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    cache = folder/"folds.pkl"
    dates = pd.period_range("2011-01", "2022-12", freq="M")
    if cache.exists() and (folder/"fold_key.txt").read_text() == key:
        folds = pickle.loads(cache.read_bytes())
    else:
        folds = {}
        for date in dates:
            folds[date] = peer_fold(bonds, features, surveys[case], candidates, selected, date)
            if date.month == 1:
                print(f"  {case}: fifteen-feature fold {date}", flush=True)
        cache.write_bytes(pickle.dumps(folds, protocol=pickle.HIGHEST_PROTOCOL)); (folder/"fold_key.txt").write_text(key)
    screening.to_csv(folder/"screening.csv", index=False)
    pd.DataFrame([f["audit"] for f in folds.values()]).to_csv(folder/"forecast_audit.csv", index=False)
    frozen, cached = {}, {}
    validation = dates[dates+12 < pd.Period("2017-01", "M")]
    for label in ("L4_Real", "L4_GC"):
        frozen[label], p = search_xgb(folds, validation, bonds, folder, label)
        cached[label] = dict(zip(validation, p))
    # Dynamic EN cross-validation takes place in each mature outer training set,
    # rather than switching to a fixed model selected on an early subperiod.
    records, en_audit, inner_cache = [], [], {}
    for date, fold in folds.items():
        predictions = dict(fold["reference"])
        fold["inner_folds"] = []
        # Three contiguous annual validation blocks, all contained in mature
        # outer training labels. Every generated regressor is reconstructed
        # at each inner boundary; outer-training labels never fit inner PCs.
        for offset in (36, 24, 12):
            validation_dates = fold["train_dates"][-offset:][:12]
            start, end = validation_dates[0], validation_dates[-1]
            if (start,end) not in inner_cache:
                inner = peer_fold(bonds, features, surveys[case], candidates, selected, start,
                                  predict_dates=validation_dates)
                inner["validation_y"] = bonds.loc[validation_dates,[f"rx{n}" for n in range(2,6)]].to_numpy()
                inner_cache[(start,end)] = inner
            fold["inner_folds"].append(inner_cache[(start,end)])
        for label in ("L3_Real", "L3_GC"):
            prediction, audit = elastic_predict(fold, copula=label.endswith("GC"))
            predictions[label] = prediction
            en_audit.append(dict(ym=str(date), model=label, **audit))
        for label in ("L4_Real", "L4_GC"):
            predictions[label] = cached[label][date][None, :] if date in cached[label] else xgb_predict(fold, frozen[label])
        for label, p in predictions.items():
            records.append(dict(ym=str(date), model=label, **{f"mu{n}": float(p[0,n-2]) for n in range(2,6)}))
        if date.month == 1:
            print(f"  {case}: predicted {date}", flush=True)
    predictions = pd.DataFrame(records)
    predictions.to_csv(folder/"predictions.csv", index=False)
    (folder/"elasticnet_cv_audit.json").write_text(json.dumps(en_audit, indent=2))
    frozen["validation_last_purchase"], frozen["validation_last_outcome"] = str(validation[-1]), str(validation[-1]+12)
    frozen["names"] = BASE_NAMES+[f"PC{k}" for k in range(1,6)]
    (folder/"frozen_parameters.json").write_text(json.dumps(frozen, indent=2))
    (folder/"fit_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Finished all reference predictions: {case}", flush=True)


def economic_evaluation(bonds, predictions, folder):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    p = predictions.copy(); p["ym"] = pd.PeriodIndex(p.ym, freq="M")
    columns = [f"mu{n}" for n in range(2,6)]
    all_dates = pd.period_range("2011-01", "2022-12", freq="M")
    forecasts = {name:g.set_index("ym")[columns].reindex(all_dates) for name,g in p.groupby("model")}
    historical, _, _ = risk_panel(bonds, all_dates)
    forecasts["RollingMV"] = pd.DataFrame(historical*100, index=all_dates, columns=columns)
    val = all_dates[(all_dates <= pd.Period("2016-12", "M")) & (all_dates+1 < pd.Period("2017-01", "M"))]
    selected = forecasts["L4_GC"].loc[val]
    trials = []
    def objective(trial):
        params = dict(budget=trial.suggest_float("budget", .1, 1.),
                      drawdown=trial.suggest_float("drawdown", 0., 30.),
                      concentration=trial.suggest_float("concentration", 0., 20.))
        path = allocation_path(bonds, selected, **params)
        excess = path.return_net.to_numpy()-path.risk_free.to_numpy()
        score = np.sqrt(12)*excess.mean()/excess.std(ddof=1) if excess.std() > 1e-12 else 0.
        trials.append(dict(number=trial.number, score=score, **params))
        return score
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.enqueue_trial(dict(budget=1., drawdown=0., concentration=0.))
    study.optimize(objective, n_trials=PROTOCOL["allocation_trials"])
    params = study.best_params
    pd.DataFrame(trials).to_csv(folder/"execution_trials.csv", index=False)
    (folder/"execution_parameters.json").write_text(json.dumps(dict(params=params,
        validation_last_purchase=str(val[-1]), validation_last_outcome=str(val[-1]+1), objective="monthly validation excess Sharpe",
        disclosure="Three parameter names/roles from slides; equations, objective and ranges are explicit reconstruction assumptions"), indent=2))
    test = pd.period_range("2017-01", "2022-12", freq="M")
    paths, rows = {}, []
    for name in ("RollingMV", "L1_CP15", "L1_Full", "L3_GC", "L4_Real", "L4_GC"):
        paths[name+"/direct"] = allocation_path(bonds, forecasts[name].loc[test])
    paths["L4_GC/optimized"] = allocation_path(bonds, forecasts["L4_GC"].loc[test], **params)
    for name,path in paths.items():
        rows.append(dict(model=name, **summarize(path)))
    pd.DataFrame(rows).to_csv(folder/"performance.csv", index=False)
    pd.concat([v.assign(model=k) for k,v in paths.items()], ignore_index=True).to_csv(folder/"trades.csv", index=False)
    stresses=[]
    for cost in (.5, 2., 5.):
        for name in ("RollingMV", "L4_GC"):
            kwargs = params if name == "L4_GC" else {}
            path = allocation_path(bonds, forecasts[name].loc[test], cost_bps=cost, **kwargs)
            stresses.append(dict(model=name, cost_bps=cost, **summarize(path)))
    pd.DataFrame(stresses).to_csv(folder/"cost_stress.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=PROTOCOL["cases"])
    parser.add_argument("--output-dir", type=Path, default=ROOT/"output"/"peer_reference")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True, exist_ok=True)
    _, checks, _, _ = run_replication(); print(f"Prerequisite {len(checks)} paper checks passed", flush=True)
    (args.output_dir/"protocol.json").write_text(json.dumps(PROTOCOL, indent=2))
    (args.output_dir/"reference_assumptions.json").write_text(json.dumps(ASSUMPTIONS, indent=2))
    data = inputs()
    for case in ([args.case] if args.case else PROTOCOL["cases"]):
        folder = args.output_dir/case
        if not args.evaluate_only:
            run_case(case, args.output_dir, data)
        meta = json.loads((folder/"fit_metadata.json").read_text())
        if meta["sources"] != source_hashes() or meta["versions"] != versions():
            raise ValueError("Stale reference results: source/data/version changed")
        p = pd.read_csv(folder/"predictions.csv")
        economic_evaluation(data[0], p, folder)
        (folder/"evaluation_metadata.json").write_text(json.dumps(meta, indent=2))
    print("Finished requested reference reconstruction", flush=True)


if __name__ == "__main__":
    main()
