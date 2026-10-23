"""Eriksen's 24 SPF expectation series, and a separately labelled dispersion panel.

Eriksen (2017), section III, uses respondent-specific changes relative to that
respondent's own nowcast, median aggregation, PCA3, then a return-fitted index.
The reference slides call ME 'disagreement'; that is not the paper's definition.
Both data panels are retained so this ambiguity is explicit rather than hidden.
"""
import re
from pathlib import Path
import numpy as np
import pandas as pd

SPF_SERIES = ("RGDP", "PGDP", "UNEMP", "CPROF", "INDPROD", "HOUSING")


def release_dates(path):
    result, year = {}, None
    for line in Path(path).read_text().splitlines():
        match = re.match(r"\s*(?:(\d{4})\s+)?Q([1-4])\s+(.+)", line)
        if not match:
            continue
        if match[1]:
            year = int(match[1])
        dates = re.findall(r"\d{1,2}/\d{1,2}/\d{2,4}", match[3])
        if year and len(dates) >= 2:
            result[pd.Period(f"{year}Q{match[2]}", freq="Q")] = pd.Timestamp(dates[-1])
    return result


def respondent_expectations(frame, name):
    """Median of individual log(future/own nowcast), not log(median levels)."""
    f = frame.replace(-999, np.nan).copy()
    q = pd.PeriodIndex(f.YEAR.astype(int).astype(str)+"Q"+f.QUARTER.astype(int).astype(str), freq="Q")
    now = pd.to_numeric(f[name+"2"], errors="coerce")
    transformed = {}
    for h in range(1, 5):
        future = pd.to_numeric(f[name+str(h+2)], errors="coerce")
        if name == "HOUSING":
            values = future-now
        else:
            values = 100*np.log(future.where(future > 0)/now.where(now > 0))
        transformed[f"{name}_h{h}"] = pd.Series(values.to_numpy(), index=q).groupby(level=0).median()
    return pd.DataFrame(transformed)


def load_spf(directory, dates):
    """Align each release to month-end using the Fed's actual release calendar.

    Never backfill a survey into earlier months. Early 1994 decisions may use
    the already released 1993Q4 panel. The latest corrections to historical SPF
    files remain a documented archive-version limitation.
    """
    directory = Path(directory)
    releases = release_dates(directory/"spf_release_dates.txt")
    xls = pd.ExcelFile(directory/"SPFmicrodata.xlsx")
    panels = [respondent_expectations(pd.read_excel(xls, name), name) for name in SPF_SERIES]
    quarter = pd.concat(panels, axis=1).sort_index()
    # Consistent expectation-horizon dispersion extension from micro forecasts:
    # IQR of individual log growth/change, distinct from original median ME.
    dispersion = []
    for name in SPF_SERIES:
        f = pd.read_excel(xls, name).replace(-999, np.nan)
        q = pd.PeriodIndex(f.YEAR.astype(int).astype(str)+"Q"+f.QUARTER.astype(int).astype(str), freq="Q")
        now = pd.to_numeric(f[name+"2"], errors="coerce")
        block = {}
        for h in range(1, 5):
            future = pd.to_numeric(f[name+str(h+2)], errors="coerce")
            value = future-now if name == "HOUSING" else 100*np.log(future.where(future > 0)/now.where(now > 0))
            grouped = pd.Series(value.to_numpy(), index=q).groupby(level=0)
            block[f"{name}_h{h}"] = grouped.quantile(.75)-grouped.quantile(.25)
        dispersion.append(pd.DataFrame(block))
    dispersion = pd.concat(dispersion, axis=1)
    def align(panel):
        available = panel.loc[panel.index.isin(releases)].copy()
        available["release_date"] = [releases[q] for q in available.index]
        # As-of join retains actual publication timestamps as audit metadata.
        available = available.sort_values("release_date").reset_index(drop=True)
        decisions = pd.DataFrame({"decision_date": pd.PeriodIndex(dates, freq="M").to_timestamp(how="end")})
        out = pd.merge_asof(decisions, available, left_on="decision_date", right_on="release_date", direction="backward")
        out.index = pd.PeriodIndex(dates, freq="M")
        return out.drop(columns="decision_date")
    return align(quarter), align(dispersion)
