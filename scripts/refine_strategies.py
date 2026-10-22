"""Literature-led forecast refinement and a declared, causally shrunk strategy."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from bond_rp.config import DATA_DIR
from bond_rp.data import load_bond_data, load_gsw_bond_data, load_macro_raw_transformed
from bond_rp.refinement import (BASE_MODELS, IQR_BOUND, PRIMARY, causal_shrinkage,
                                literal_sentiment, refinement_forecasts)
from bond_rp.trading import forecast_designs, load_monthly_market
from bond_rp.vintages import VINTAGE_URLS, VintageArchive

PROTOCOL = {
    "status": "retrospective method development; no untouched holdout claimed",
    "primary": PRIMARY, "core_weights": {"CP": 1/3, "LN6_safe": 1/6, "SPCA": 1/6, "Goh48": 1/3},
    "windows_months": [None, 120, 240], "window_weights": [1/3, 1/3, 1/3],
    "shrinkage": "closed-form slope versus EH using stored, matured forecast errors",
    "minimum_errors": 60, "error_lookback_months": 120, "macro_lag": 2, "vintage_lag": 1,
    "iqr_bound": IQR_BOUND, "goh_bic": "Bai-Ng ICp2 (max 8), then each maturity's subset BIC including intercept only",
    "primary_policy": "long_only", "gamma": 5, "cap": 1, "cost_bps": .5,
    "origins": ["1985-01", "1990-01", "1995-01", "2000-01"],
    "stress": {"cost_bps": [2, 5], "delay_months": [1], "gamma": [2, 10], "risk_window": [120]},
    "iqr_sensitivity": [4, 8],
    "cost_aware": "structural sizing sensitivity using entry-date funding proxy; primary rule is preserved",
    "prior_candidates": ["EW3 OLS / 120 months / half-shrink", "EW3 Ridge100 / 120 months / half-shrink"],
    "disjoint_subperiods": [["1985-01", "1999-12"], ["2000-01", "2012-12"],
                           ["2013-01", "2019-12"], ["2020-01", "2026-01"]],
    "forecast_models": BASE_MODELS,
    "inference": "raw forecast loss, HAC and joint centered-null block max-t; conditional on stored forecasts",
    "bootstrap_blocks": [12, 24], "bootstrap_replications": 1999,
    "inputs": "existing local snapshots; no latest-value fill of historical macro vintages",
}


def fingerprint(files, settings):
    h = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for name in ("config.py", "data.py", "models.py", "coursework.py", "refinement.py", "vintages.py", "trading.py"):
        h.update((ROOT/"bond_rp"/name).read_bytes())
    for name in files:
        h.update(name.encode()); h.update((Path(DATA_DIR)/name).read_bytes())
    import sklearn
    h.update(f"{np.__version__}|{pd.__version__}|{sklearn.__version__}".encode())
    return h.hexdigest()


def generate(tag, bonds, macro, horizon, kind, args, archive=None, iqr_bound=IQR_BOUND):
    folder = args.output_dir/tag
    folder.mkdir(parents=True, exist_ok=True)
    files = (["Fama_Bliss_price.txt", "macro_raw_data.xlsx"] if tag.startswith("fb") else
             ["GSW_yields.csv", "2026-04-MD.csv"])
    if horizon == 1:
        files.append("F-F_Research_Data_Factors_CSV.zip")
    sentiment = None
    if archive is None:
        files.append("SENTIMENT.xlsx")
        sentiment = literal_sentiment()
    else:
        files += list(VINTAGE_URLS)
    settings = {"horizon": horizon, "kind": kind, "start": "1980-01" if archive is None else "1999-09",
                "end": "2012-02" if tag.startswith("fb") else "2026-01", "lag": 2, "iqr_bound": float(iqr_bound)}
    key = fingerprint(files, settings)
    marker = folder/"cache_key.txt"
    if not args.refresh and marker.exists() and marker.read_text() == key:
        print(f"Verified refinement cache: {tag}", flush=True)
        p = pd.read_csv(folder/"predictions.csv")
    else:
        print(f"Refining {tag}: {settings['start']} to {settings['end']}", flush=True)
        p, audit = refinement_forecasts(bonds, macro, horizon=horizon, kind=kind, sentiment=sentiment,
                                        archive=archive, start=settings["start"], end=settings["end"], iqr_bound=iqr_bound)
        p.to_csv(folder/"predictions.csv", index=False)
        audit.to_csv(folder/"forecast_audit.csv", index=False)
        marker.write_text(key)
    target = bonds["mx_bar"] if horizon == 1 else bonds["sx_bar" if kind == "simple" else "rx_bar"]
    combined, audit = causal_shrinkage(p, target, horizon=horizon)
    combined.to_csv(folder/"combined_predictions.csv", index=False)
    audit.to_csv(folder/"shrinkage_audit.csv", index=False)
    (folder/"settings.json").write_text(json.dumps(settings, indent=2))


def match_warmup(output, source, target, actual, horizon):
    """Same learner launch date as the historical-vintage experiment."""
    folder = output/target
    folder.mkdir(exist_ok=True)
    original = output/source
    p = pd.read_csv(original/"predictions.csv")
    p = p.loc[p.ym >= "1999-09"]
    p.to_csv(folder/"predictions.csv", index=False)
    combined, audit = causal_shrinkage(p, actual, horizon=horizon)
    combined.to_csv(folder/"combined_predictions.csv", index=False)
    audit.to_csv(folder/"shrinkage_audit.csv", index=False)
    f = pd.read_csv(original/"forecast_audit.csv")
    f[f.ym >= "1999-09"].to_csv(folder/"forecast_audit.csv", index=False)
    settings = json.loads((original/"settings.json").read_text())
    settings.update(start="1999-09", comparison="matched vintage-learner warmup")
    (folder/"settings.json").write_text(json.dumps(settings, indent=2))


def retain_prior_candidates(bonds, macro, args, archive=None):
    """Reproduce the candidates already disclosed before this refinement round.

    CP + LN8 + price-MA12, no sentiment; same vintage convention, fees and targets.
    These are comparison baselines, not replacements picked from new rankings.
    """
    folder = args.output_dir/"prior_candidates"/("vintage" if archive else "revised")
    folder.mkdir(parents=True, exist_ok=True)
    files = ["GSW_yields.csv", "2026-04-MD.csv", "F-F_Research_Data_Factors_CSV.zip"]
    if archive:
        files += list(VINTAGE_URLS)
    settings = {"protocol": "previously disclosed EW3/w120 candidates", "horizon": 1,
                "start": "1999-09" if archive else "1980-01", "end": "2026-01"}
    key = fingerprint(files, settings)
    marker = folder/"cache_key.txt"
    if not args.refresh and marker.exists() and marker.read_text() == key:
        p = pd.read_csv(folder/"predictions.csv")
        print(f"Verified prior candidate cache: {folder.name}", flush=True)
    else:
        p, audit = forecast_designs(bonds, macro, horizon=1, start=settings["start"], end=settings["end"],
                                    vintage_archive=archive)
        p.to_csv(folder/"predictions.csv", index=False)
        audit.to_csv(folder/"forecast_audit.csv", index=False)
        marker.write_text(key)
    cols = [f"mu{n}" for n in range(2, 6)]
    reference = p[p.model == "EH/w120"].set_index("ym")[cols]
    rows = []
    for old, new in [("EW/w120", "PriorEW3/w120"), ("EW_R100/w120", "PriorEW3_R100/w120")]:
        x = p[p.model == old].set_index("ym")[cols]
        x = reference+.5*(x-reference)
        rows.append(x.reset_index().assign(model=new))
    pd.concat(rows, ignore_index=True).to_csv(folder/"comparison_predictions.csv", index=False)
    (folder/"settings.json").write_text(json.dumps(settings, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"output"/"refinement")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--stage", choices=("forecast", "evaluate", "all"), default="all")
    parser.add_argument("--update-readme", action="store_true")
    parser.add_argument("--iqr-stress", action="store_true", help="Also compute 4/8-IQR monthly sensitivity without replacing 6-IQR primary")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"protocol.json").write_text(json.dumps(PROTOCOL, indent=2, ensure_ascii=False))
    if args.stage != "evaluate":
        fb, gsw, monthly = load_bond_data().set_index("ym"), load_gsw_bond_data().set_index("ym"), load_monthly_market()
        jln, fred = load_macro_raw_transformed(), load_macro_raw_transformed("fredmd")
        for kind in ("log", "simple"):
            generate(f"fb_annual_{kind}", fb, jln, 12, kind, args)
            generate(f"gsw_annual_{kind}", gsw, fred, 12, kind, args)
        generate("gsw_monthly", monthly, fred, 1, "simple", args)
        with VintageArchive(DATA_DIR) as archive:
            for kind in ("log", "simple"):
                generate(f"vintage_annual_{kind}", gsw, fred, 12, kind, args, archive)
            generate("vintage_monthly", monthly, fred, 1, "simple", args, archive)
        for kind in ("log", "simple"):
            match_warmup(args.output_dir, f"gsw_annual_{kind}", f"matched_annual_{kind}",
                         gsw["rx_bar" if kind == "log" else "sx_bar"], 12)
        match_warmup(args.output_dir, "gsw_monthly", "matched_monthly", monthly.mx_bar, 1)
        retain_prior_candidates(monthly, fred, args)
        with VintageArchive(DATA_DIR) as archive:
            retain_prior_candidates(monthly, fred, args, archive)
        if args.iqr_stress:
            with VintageArchive(DATA_DIR) as archive:
                for bound in PROTOCOL["iqr_sensitivity"]:
                    stress = copy.copy(args)
                    stress.output_dir = args.output_dir/"iqr_stress"/f"bound{bound}"
                    stress.output_dir.mkdir(parents=True, exist_ok=True)
                    generate("gsw_monthly", monthly, fred, 1, "simple", stress, iqr_bound=bound)
                    generate("vintage_monthly", monthly, fred, 1, "simple", stress, archive, iqr_bound=bound)
    if args.stage != "forecast":
        from evaluate_refinements import evaluate, export
        evaluate(args.output_dir, PROTOCOL)
        if args.update_readme:
            export(args.output_dir, PROTOCOL)


if __name__ == "__main__":
    main()
