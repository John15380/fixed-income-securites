"""Calendar-aligned bond outcomes and uncompressed macroeconomic inputs."""

from pathlib import Path
import numpy as np
import pandas as pd

from .config import DATA_DIR, HORIZON_MONTHS, MATS, TI_LOOKBACK


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
    """CP authors' publicly released CRSP-derived original price archive."""
    import zipfile
    with zipfile.ZipFile(Path(DATA_DIR)/"CP2005_replication.zip") as z:
        raw = pd.read_csv(z.open("cp/bondprice.dat"), sep=r"\s+", comment="%",
                          names=["date", "p1", "p2", "p3", "p4", "p5"])
    dates = pd.to_datetime(raw["date"], format="%Y%m%d").dt.to_period("M")
    prices = pd.DataFrame({f"lp{n}": 100*np.log(raw[f"p{n}"]/100) for n in range(1,6)})
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


def load_macro_raw_transformed() -> pd.DataFrame:
    """Pinned revised FRED-MD panel, used only in labeled sensitivity analyses."""
    from .vintages import transform_fred_csv
    return transform_fred_csv(Path(DATA_DIR)/"2026-04-MD.csv")
