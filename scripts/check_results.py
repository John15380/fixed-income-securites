"""Independent replay of saved holdings, cash, fees, calendar cutoffs and metrics."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bond_rp.config import DATA_DIR
from bond_rp.market import load_monthly_market


def main():
    out=ROOT/"output"/"final"
    manifest=json.loads((out/"manifest.json").read_text())
    for name,digest in manifest["inputs_sha256"].items():
        assert hashlib.sha256((DATA_DIR/name).read_bytes()).hexdigest()==digest,name
    for name,digest in manifest["source_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    market=load_monthly_market()
    trades=pd.read_csv(out/"monthly_accounts.csv")
    performance=pd.read_csv(out/"performance.csv").set_index("model")
    c=manifest["strategy"]["one_way_cost_bps"]/10000
    max_error=0.
    for name,g in trades.groupby("model"):
        g=g.sort_values("purchase")
        dates=pd.PeriodIndex(g.purchase,freq="M")
        assert (np.diff(dates.asi8)==1).all()
        assert (pd.PeriodIndex(g.sale,freq="M")==dates+1).all()
        assert (pd.PeriodIndex(g.risk_last_label_end,freq="M")<dates).all()
        q=g[[f"q{n}" for n in range(2,6)]].to_numpy()
        rf=market.rf_month.reindex(dates).to_numpy()/100
        actual=market.loc[dates,[f"mr{n}" for n in range(2,6)]].to_numpy()/100
        assert (q>=0).all() and (q.sum(axis=1)<=1+1e-12).all()
        entry_fee=c*q.sum(axis=1)
        cash=1-q.sum(axis=1)-entry_fee
        assert (cash>=-1e-12).all()
        sale_value=(q*(1+actual)).sum(axis=1)
        r=cash*(1+rf)+sale_value-c*sale_value-1
        max_error=max(max_error,np.max(abs(r-g.return_net)))
        np.testing.assert_allclose(r,g.return_net,atol=1e-14,rtol=0)
        np.testing.assert_allclose(cash,g.cash_after_entry,atol=1e-14,rtol=0)
        wealth=np.r_[1,np.cumprod(1+r)]
        row=performance.loc[name]
        np.testing.assert_allclose(100*(wealth[-1]**(12/len(r))-1),row.cagr_pct,atol=1e-11)
        np.testing.assert_allclose(100*np.min(wealth/np.maximum.accumulate(wealth)-1),row.max_dd_pct,atol=1e-11)
    n=0
    for label in ("archival_monthly","archival_horizons","revised"):
        a=pd.read_csv(out/label/"forecast_audit.csv")
        dates=pd.PeriodIndex(a.ym,freq="M")
        assert (pd.PeriodIndex(a.last_label_end,freq="M")<dates).all()
        assert (pd.PeriodIndex(a.latest_macro_reference,freq="M")<=dates-2).all()
        if label.startswith("archival"):
            assert (pd.PeriodIndex(a.vintage,freq="M")==dates-1).all()
        n+=len(a)
    checks=pd.read_csv(out/"paper_checks.csv")
    assert checks.passed.all()
    print(f"Verified {len(trades)} monthly account rows; max independent replay error {max_error:.3g}")
    print(f"Verified {n} forecast training/vintage audit rows and {len(checks)} paper checks")
    print("Input and source SHA-256 fingerprints match the completed run")


if __name__=="__main__":
    main()
