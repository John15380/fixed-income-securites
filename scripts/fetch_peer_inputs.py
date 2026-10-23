"""Fetch the public SPF source files required by the reference reconstruction."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bond_rp.config import DATA_DIR

BASE="https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/survey-of-professional-forecasters/"
SOURCES={"SPFmicrodata.xlsx":BASE+"historical-data/SPFmicrodata.xlsx",
         "spf_release_dates.txt":BASE+"spf-release-dates.txt",
         "US_Policy_Uncertainty_Data.xlsx":"https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.xlsx"}


def main():
    directory=Path(DATA_DIR);directory.mkdir(parents=True,exist_ok=True)
    records={}
    for name,url in SOURCES.items():
        path=directory/name
        if not path.exists():
            request=urllib.request.Request(url,headers={"User-Agent":"bond-rp academic replication"})
            with urllib.request.urlopen(request,timeout=90) as source:
                data=source.read()
            if name.endswith("xlsx") and not data.startswith(b"PK"):
                raise ValueError("The downloaded workbook is not an Excel ZIP container")
            path.write_bytes(data)
        if name.endswith("xlsx"):
            with zipfile.ZipFile(path) as archive:
                assert "xl/workbook.xml" in archive.namelist()
        records[name]={"url":url,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
        print("Verified",name,flush=True)
    (directory/"peer_input_sources.json").write_text(json.dumps(records,indent=2))


if __name__=="__main__":
    main()
