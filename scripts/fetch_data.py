"""Fetch public research inputs; preserve existing snapshots and never upload data."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bond_rp.config import DATA_DIR
from bond_rp.replication import CP_URL, AUTHOR_URL, HANDBOOK_URL
from bond_rp.market import RF_URL
from bond_rp.vintages import VINTAGE_URLS

URLS={"CP2005_replication.zip":CP_URL,"RFS2009.xls":AUTHOR_URL,
      "LN_handbook_replication.zip":HANDBOOK_URL,
      "GSW_yields.csv":"https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv",
      "2026-04-MD.csv":"https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/2026-04-md.csv",
      "F-F_Research_Data_Factors_CSV.zip":RF_URL,**VINTAGE_URLS}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify",action="store_true",help="Require every file to match results/data_manifest.json; no downloads")
    args=parser.parse_args()
    DATA_DIR.mkdir(parents=True,exist_ok=True)
    expected_path=ROOT/"results"/"data_manifest.json"
    expected=json.loads(expected_path.read_text())["inputs_sha256"] if expected_path.exists() else {}
    mismatches=[]
    for name,url in URLS.items():
        path=DATA_DIR/name
        if not path.exists() and not args.verify:
            temp=path.with_suffix(path.suffix+".part")
            request=urllib.request.Request(url,headers={"User-Agent":"Treasury-research/1.0"})
            with urllib.request.urlopen(request,timeout=60) as response:
                temp.write_bytes(response.read())
            if name.endswith(".zip"):
                with zipfile.ZipFile(temp) as z:
                    if z.testzip() is not None:
                        raise ValueError(f"Corrupt archive: {name}")
            temp.replace(path)
        if not path.exists():
            mismatches.append(name+": missing")
            continue
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        match=expected.get(name)==digest
        print(f"{name}: {'matches recorded snapshot' if match else 'snapshot differs or not yet recorded'}")
        if not match:
            mismatches.append(name+": checksum mismatch")
    if args.verify and mismatches:
        raise SystemExit("\n".join(mismatches))
    if mismatches:
        print("Live endpoints can revise historical data. New snapshots produce a new, separately hashed research run.")


if __name__=="__main__":
    main()
