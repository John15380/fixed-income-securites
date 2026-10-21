"""Calendar-aligned bond outcomes and uncompressed macroeconomic inputs."""

from pathlib import Path
import numpy as np
import pandas as pd

from .config import DATA_DIR, HORIZON_MONTHS, MATS, TI_LOOKBACK, TRANSFORM


def apply_transform(series: pd.Series, code: int) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    log = np.log(s.where(s > 0))
    transforms = {1: lambda: s, 2: lambda: s.diff(), 3: lambda: s.diff().diff(),
                  4: lambda: log, 5: lambda: 100 * log.diff(),
                  6: lambda: 100 * log.diff().diff(),
                  7: lambda: 100 * s.pct_change(fill_method=None).diff()}
    if code not in transforms:
        raise ValueError(f"Unknown transform code: {code}")
    return transforms[code]().replace([np.inf, -np.inf], np.nan)


def bond_variables(log_prices: pd.DataFrame) -> pd.DataFrame:
    """Log prices in percent; outcomes use exact calendar months, never row offsets.

    rx*: academic log excess returns (%); sx*: investable simple excess returns
    (%). Each label is stored at its purchase month and matures 12 months later.
    """
    frame = log_prices.sort_index().copy()
    if not isinstance(frame.index, pd.PeriodIndex) or frame.index.freqstr != "M":
        raise ValueError("Expected a monthly PeriodIndex")
    if frame.index.has_duplicates:
        raise ValueError("Duplicate bond months")
    frame = frame.reindex(pd.period_range(frame.index.min(), frame.index.max(), freq="M"))
    frame.index.name = "ym"
    for n in range(1, 6):
        frame[f"y{n}"] = -frame[f"lp{n}"] / n
    frame["f1"] = frame["y1"]
    frame["rf_simple"] = 100 * np.expm1(frame["y1"] / 100)
    for n in MATS:
        frame[f"f{n}"] = frame[f"lp{n-1}"] - frame[f"lp{n}"]
        frame[f"fs{n}"] = frame[f"f{n}"] - frame["y1"]
        sale_price = frame[f"lp{n-1}"].reindex(frame.index + HORIZON_MONTHS).to_numpy()
        log_return = sale_price - frame[f"lp{n}"].to_numpy()
        frame[f"rx{n}"] = log_return - frame["y1"]
        frame[f"simple{n}"] = 100 * np.expm1(log_return / 100)
        frame[f"sx{n}"] = frame[f"simple{n}"] - frame["rf_simple"]
        frame[f"TI{n}"] = frame[f"lp{n}"] - frame[f"lp{n}"].rolling(TI_LOOKBACK).mean()
    for name, prefix in [("fs_avg", "fs"), ("rx_bar", "rx"), ("sx_bar", "sx"),
                         ("TI_avg", "TI")]:
        frame[name] = frame[[f"{prefix}{n}" for n in MATS]].mean(axis=1, skipna=False)
    return frame


def load_bond_data() -> pd.DataFrame:
    raw = pd.read_csv(Path(DATA_DIR) / "Fama_Bliss_price.txt", sep=r"\s+", comment="%",
                      names=["date", "p1", "p2", "p3", "p4", "p5"])
    dates = pd.to_datetime(raw["date"], format="%Y%m%d").dt.to_period("M")
    prices = pd.DataFrame({f"lp{n}": 100 * np.log(raw[f"p{n}"] / 100)
                           for n in range(1, 6)})
    prices.index = dates
    return bond_variables(prices).reset_index()


def load_gsw_bond_data() -> pd.DataFrame:
    """A continuous GSW curve throughout; no Fama-Bliss/GSW splice."""
    raw = pd.read_csv(Path(DATA_DIR) / "GSW_yields.csv", skiprows=9, parse_dates=["Date"])
    columns = [f"SVENY0{n}" for n in range(1, 6)]
    raw[columns] = raw[columns].apply(pd.to_numeric, errors="coerce")
    raw = raw.dropna(subset=columns).sort_values("Date")
    raw["ym"] = raw["Date"].dt.to_period("M")
    monthly = raw.groupby("ym").tail(1).set_index("ym")
    lp = pd.DataFrame({f"lp{n}": -n * monthly[f"SVENY0{n}"] for n in range(1, 6)})
    return bond_variables(lp).reset_index()


def load_macro_raw_transformed(source: str = "jln") -> pd.DataFrame:
    """Transform each series; leave imputation, scaling and PCA to training folds."""
    if source == "jln":
        raw = pd.read_excel(Path(DATA_DIR) / "macro_raw_data.xlsx", sheet_name="macrodata")
        dates = pd.to_datetime(dict(year=raw["year"], month=raw["month"], day=1))
        codes = {c: TRANSFORM[c] for c in raw if c in TRANSFORM}
    elif source == "fredmd":
        raw = pd.read_csv(Path(DATA_DIR) / "2026-04-MD.csv")
        codes = {c: int(float(raw.iloc[0][c])) for c in raw if c != "sasdate"}
        raw = raw.iloc[1:].reset_index(drop=True)
        dates = pd.to_datetime(raw["sasdate"])
    else:
        raise ValueError(f"Unknown macro source: {source}")
    raw = raw.copy()
    raw.index = pd.PeriodIndex(dates, freq="M")
    if raw.index.has_duplicates or not raw.index.is_monotonic_increasing:
        raise ValueError("Macro months must be unique and ordered")
    raw = raw.reindex(pd.period_range(raw.index.min(), raw.index.max(), freq="M"))
    return pd.DataFrame({c: apply_transform(raw[c], code) for c, code in codes.items()})


def load_sentiment() -> pd.DataFrame:
    """Use SENT levels and their square; no unsupported changes-in-sentiment proxy."""
    raw = pd.read_excel(Path(DATA_DIR) / "SENTIMENT.xlsx", sheet_name="DATA")
    dates = pd.to_datetime(raw["yearmo"].astype(int).astype(str), format="%Y%m").dt.to_period("M")
    values = pd.to_numeric(raw["SENT"], errors="coerce").to_numpy()
    frame = pd.DataFrame({"SENT": values, "SENT_SQ": values ** 2}, index=dates)
    if frame.index.has_duplicates:
        raise ValueError("Duplicate sentiment months")
    return frame.sort_index()
