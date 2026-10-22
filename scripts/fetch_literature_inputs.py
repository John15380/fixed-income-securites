"""Download the three public inputs needed for literature checks/extensions.

Existing files are preserved. CRSP/course inputs are not public downloads and
must still be supplied locally as described in README.
"""
from pathlib import Path
import sys
import urllib.request
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bond_rp.config import DATA_DIR
from bond_rp.replication import AUTHOR_URL, HANDBOOK_URL

URLS = {
    "RFS2009.xls": AUTHOR_URL,
    "LN_handbook_replication.zip": HANDBOOK_URL,
    "CPILFESL.csv": "https://fred.stlouisfed.org/graph/fredgraph.csv?id=CPILFESL&cosd=1957-01-01&coed=2026-04-01",
}


def main():
    import pandas as pd
    import zipfile
    directory = Path(DATA_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    for name, url in URLS.items():
        path = directory/name
        if path.exists():
            print(f"Preserved existing {name}")
            continue
        temporary = path.with_suffix(path.suffix+".part")
        request = urllib.request.Request(url, headers={"User-Agent": "bond-rp-coursework/0.4"})
        with urllib.request.urlopen(request, timeout=60) as response:
            temporary.write_bytes(response.read())
        if name.endswith(".zip"):
            with zipfile.ZipFile(temporary) as archive:
                if "Data&ReplicationFiles/lndata.mat" not in archive.namelist():
                    raise ValueError("Unexpected MATLAB replication archive")
        elif name.endswith(".csv"):
            if "CPILFESL" not in pd.read_csv(temporary, nrows=2):
                raise ValueError("Unexpected FRED response")
        else:
            pd.read_excel(temporary, sheet_name="Data Table2", nrows=8)
        temporary.rename(path)
        print(f"Downloaded {name}")


if __name__ == "__main__":
    main()
