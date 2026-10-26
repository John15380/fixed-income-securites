"""Check original paper tables and shared estimators using public author archives."""
from pathlib import Path
import json
import sys
from threadpoolctl import threadpool_limits
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bond_rp.replication import run_replication

if __name__=="__main__":
    with threadpool_limits(limits=1):
        comparison,checks,coefficients,sources=run_replication()
    out=ROOT/"output"/"final"/"literature"
    out.mkdir(parents=True,exist_ok=True)
    comparison.to_csv(out/"comparison.csv",index=False)
    checks.to_csv(out/"checks.csv",index=False)
    coefficients.to_csv(out/"coefficients.csv",index=False)
    (out/"inputs.json").write_text(json.dumps(sources,indent=2)+"\n")
    print(f"{int(checks.passed.sum())}/{len(checks)} numerical and logic checks passed")
    print(comparison.loc[comparison.model.isin(["CP05 restricted","LN09 LN6 released","LN09 CP+LN5"])].to_string(index=False))
    print(f"Full results: {out}")
