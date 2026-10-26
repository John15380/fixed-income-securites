"""Monthly FRED-MD vintage snapshots, read without extracting ZIP members."""

from pathlib import Path
import re
import zipfile
import pandas as pd
from .data import apply_transform


VINTAGE_URLS = {
    "fred_md_vintages_1999_2014.zip": "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/historical_fred-md.zip",
    "fred_md_vintages_2015_2025.zip": "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/historical-vintages-of-fred-md-2015-01-to-2025-12.zip",
}


def transform_fred_csv(source):
    raw = pd.read_csv(source)
    codes = {c: int(float(raw.iloc[0][c])) for c in raw if c != "sasdate"}
    raw = raw.iloc[1:].copy()
    raw.index = pd.PeriodIndex(pd.to_datetime(raw.sasdate, format="mixed"), freq="M")
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError("Vintage dates must be unique and ordered")
    raw = raw.reindex(pd.period_range(raw.index.min(), raw.index.max(), freq="M"))
    values = pd.DataFrame({c: apply_transform(raw[c], code) for c, code in codes.items()})
    return values


class VintageArchive:
    """Use t-1's snapshot at decision month t; never fall forward to a new file.

    Monthly file names do not encode exact release timestamps. This conservative
    one-month vintage delay is explicit, in addition to the reference-month lag.
    It is a monthly-vintage experiment, not certification of daily availability.
    """

    def __init__(self, directory, vintage_delay=1):
        if vintage_delay not in (0,1):
            raise ValueError("Vintage delay must be zero or one month")
        self.vintage_delay = vintage_delay
        self.members = {}
        self.archives = []
        for name in VINTAGE_URLS:
            archive = zipfile.ZipFile(Path(directory)/name)
            self.archives.append(archive)
            for member in archive.namelist():
                match = re.fullmatch(r"(?:(\d{4})-(\d{2})|FRED-MD[-_](\d{4})m(\d{2}))\.csv", Path(member).name)
                if match:
                    year, month = (match.group(1), match.group(2)) if match.group(1) else (match.group(3), match.group(4))
                    date = pd.Period(f"{year}-{month}", "M")
                    if date in self.members:
                        raise ValueError(f"Duplicate vintage {date}")
                    self.members[date] = (archive, member)

    def at(self, forecast_month):
        vintage = pd.Period(forecast_month, "M")-self.vintage_delay
        archive, member = self.members[vintage]  # A missing vintage fails loudly.
        with archive.open(member) as source:
            data = transform_fred_csv(source)
        if data.index.max() > vintage:
            raise ValueError("Vintage contains observations later than its label")
        return vintage, data

    def close(self):
        for archive in self.archives:
            archive.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
