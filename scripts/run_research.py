"""Rebuild the final strategy, literature checks, figures and reviewable CSV tables."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache"/"matplotlib"))
from threadpoolctl import threadpool_limits
import numpy as np
import pandas as pd
import sklearn
import scipy
import statsmodels
from bond_rp.config import DATA_DIR, DEFAULT
from bond_rp.data import load_macro_raw_transformed
from bond_rp.market import load_monthly_market
from bond_rp.vintages import VintageArchive
from bond_rp.forecasting import generate_forecasts
from bond_rp.portfolio import simulate, summarize
from bond_rp.evaluation import PRIMARY, forecast_scores, in_sample_scores, in_sample_coefficients, sensitivity, economic_inference, subperiods
from bond_rp.replication import run_replication


def fingerprint():
    names=("CP2005_replication.zip","RFS2009.xls","LN_handbook_replication.zip","GSW_yields.csv",
           "2026-04-MD.csv","F-F_Research_Data_Factors_CSV.zip",
           "fred_md_vintages_1999_2014.zip","fred_md_vintages_2015_2025.zip")
    return {n:hashlib.sha256((DATA_DIR/n).read_bytes()).hexdigest() for n in names}


def cached_forecasts(directory,market,macro,archive,**kwargs):
    directory.mkdir(parents=True,exist_ok=True)
    paths=[ROOT/"bond_rp"/(name+".py") for name in
           ("config","data","market","models","factors","forecasting","vintages")]
    stamp=dict(inputs=fingerprint(),settings=kwargs,
               source={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
               numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__)
    key=json.dumps(stamp,sort_keys=True)
    meta=directory/"cache_key.json"
    if meta.exists() and meta.read_text()==key:
        print(f"Verified forecast cache: {directory.name}",flush=True)
        return pd.read_csv(directory/"forecasts.csv",dtype={"window":str}),pd.read_csv(directory/"forecast_audit.csv")
    forecasts,audit=generate_forecasts(market,macro,archive=archive,**kwargs)
    forecasts.to_csv(directory/"forecasts.csv",index=False)
    audit.to_csv(directory/"forecast_audit.csv",index=False)
    meta.write_text(key)
    return forecasts,audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=ROOT/"output"/"final")
    args=parser.parse_args()
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    replication,checks,coefficients,_=run_replication()
    replication.to_csv(out/"paper_comparison.csv",index=False)
    checks.to_csv(out/"paper_checks.csv",index=False)
    coefficients.to_csv(out/"paper_coefficients.csv",index=False)
    market=load_monthly_market()
    macro=load_macro_raw_transformed()
    with VintageArchive(DATA_DIR) as archive:
        p,a=cached_forecasts(out/"archival_monthly",market,macro,archive,start="1999-12",end=DEFAULT.end,
                             horizons=(1,),windows=(120,240,None))
        hp,ha=cached_forecasts(out/"archival_horizons",market,macro,archive,start=DEFAULT.start,end=DEFAULT.end,
                               horizons=(3,12),windows=(120,))
    revised,ra=cached_forecasts(out/"revised",market,macro,None,start="1985-01",end=DEFAULT.end,
                                horizons=(1,3,12),windows=(120,))
    allp=pd.concat([p,hp],ignore_index=True)
    allp=allp.loc[allp.ym>=DEFAULT.start]
    allp["window"]=allp.window.astype(str)
    core=p.loc[(p.window.astype(str)=="120")&p.model.isin(PRIMARY)]
    summary,trades=simulate(market,core,start=DEFAULT.start,end=DEFAULT.end)
    summary.to_csv(out/"performance.csv",index=False)
    trades.to_csv(out/"monthly_accounts.csv",index=False)
    subperiods(trades).to_csv(out/"subperiods.csv",index=False)
    forecast_scores(market,allp).to_csv(out/"forecast_scores.csv",index=False)
    forecast_scores(market,allp,shrink=1).to_csv(out/"raw_forecast_scores.csv",index=False)
    in_sample_scores(market,macro).to_csv(out/"in_sample.csv",index=False)
    in_sample_coefficients(market,macro).to_csv(out/"in_sample_coefficients.csv",index=False)
    sensitivity(market,p,DEFAULT.start,DEFAULT.end).to_csv(out/"sensitivity.csv",index=False)
    annual,accounts,tests,intervals=economic_inference(trades)
    for name,frame in [("annual_utility",annual),("annual_accounts",accounts),
                       ("active_return_tests",tests),("cer_intervals",intervals)]:
        frame.to_csv(out/(name+".csv"),index=False)
    # Different start dates use exactly the same rolling forecasting rule.
    origins=[]
    rc=revised.loc[(revised.horizon==1)&revised.model.isin(PRIMARY)]
    for start in ("1985-01","1990-01","1995-01","2000-01"):
        s,_=simulate(market,rc,start=start,end=DEFAULT.end)
        s["oos_origin"]=start
        origins.append(s)
    pd.concat(origins,ignore_index=True).to_csv(out/"revised_origins.csv",index=False)
    sources=fingerprint()
    source_paths=(sorted((ROOT/"bond_rp").glob("*.py"))+sorted((ROOT/"scripts").glob("*.py"))
                  +sorted((ROOT/"tests").glob("*.py"))+[ROOT/"pyproject.toml"])
    manifest=dict(strategy=DEFAULT.to_dict(),inputs_sha256=sources,
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths},
                  python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,
                  scipy=scipy.__version__,statsmodels=statsmodels.__version__,
                  paper_checks=int(len(checks)),paper_checks_passed=bool(checks.passed.all()),
                  interpretation="Retrospective research; historical monthly macro vintages, not certified daily real-time execution")
    (out/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(summary.loc[summary.model.isin(["EW3","EH","BuyHold"])].to_string(index=False),flush=True)
    print(intervals.to_string(index=False),flush=True)
    # Plot/export are imported here so paper-only checks need no plotting UI.
    from bond_rp.reporting import export_report
    export_report(out,ROOT)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        main()
