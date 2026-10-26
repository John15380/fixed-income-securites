"""Treasury curve pricing, aging-bond returns and observed monthly bill returns."""
from pathlib import Path
import re
import zipfile
import numpy as np
import pandas as pd
from .config import DATA_DIR
from .data import load_gsw_bond_data

RF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"


def load_cash_returns(path=None):
    """RF for return month t+1, stored at its purchase month t. Percent units.

    Use only RF, never the equity factors. It is a realized bill/cash benchmark,
    not a future input into the forecast or its trading signal.
    """
    path = Path(path or Path(DATA_DIR)/"F-F_Research_Data_Factors_CSV.zip")
    with zipfile.ZipFile(path) as archive:
        member = next(n for n in archive.namelist() if n.lower().endswith(".csv"))
        lines = archive.read(member).decode("utf-8-sig").splitlines()
    values = {}
    for line in lines:
        if re.match(r"^\s*\d{6},", line):
            parts = line.strip().split(",")
            date = pd.Period(pd.to_datetime(parts[0], format="%Y%m"), "M")
            values[date-1] = float(parts[-1])
    result = pd.Series(values, name="rf_month").sort_index()
    if not len(result) or not (np.diff(result.index.asi8) == 1).all():
        raise ValueError("RF file must contain a contiguous monthly section")
    return result


def svensson_yields(parameters, maturities):
    """Continuously compounded zero yields in percent, maturities in years."""
    p = np.asarray(parameters, dtype=float)
    n = np.asarray(maturities, dtype=float).reshape(1, -1)
    if p.ndim != 2 or p.shape[1] != 6 or (n <= 0).any():
        raise ValueError("Expected BETA0..3, TAU1..2 and positive maturities")
    # Early GSW observations are Nelson-Siegel: BETA3=0 and TAU2=-999.99.
    # The inactive second curvature term is exactly zero, not missing history.
    inactive = p[:, 3] == 0
    if (p[:, 4] <= 0).any() or ((p[:, 5] <= 0) & ~inactive).any():
        raise ValueError("Active Svensson decay constants must be positive")
    a, b = n/p[:, [4]], n/np.where(inactive, 1., p[:, 5])[:, None]
    f, g = -np.expm1(-a)/a, -np.expm1(-b)/b
    return p[:, [0]]+p[:, [1]]*f+p[:, [2]]*(f-np.exp(-a))+p[:, [3]]*(g-np.exp(-b))


def monthly_curve_returns(parameters):
    """Calendar-month repricing of the same aging bond, without a cash proxy."""
    index = pd.period_range(parameters.index.min(), parameters.index.max(), freq="M")
    parameters = parameters.reindex(index)
    p = parameters.to_numpy(dtype=float)
    maturity = np.arange(2, 6, dtype=float)
    current_y = svensson_yields(p, maturity)
    aged_y = svensson_yields(p, maturity-1/12)
    lp = -current_y*maturity
    aged_lp = -aged_y*(maturity-1/12)
    sale = pd.DataFrame(aged_lp, index=index).reindex(index+1).to_numpy()
    monthly = 100*np.expm1((sale-lp)/100)
    frame = pd.DataFrame(index=index)
    for i, n in enumerate(range(2, 6)):
        frame[f"mr{n}"] = monthly[:, i]
    frame.index.name = "ym"
    return frame


def load_monthly_market():
    raw = pd.read_csv(Path(DATA_DIR)/"GSW_yields.csv", skiprows=9, parse_dates=["Date"])
    cols = [f"BETA{i}" for i in range(4)]+["TAU1", "TAU2"]
    raw[cols] = raw[cols].apply(pd.to_numeric, errors="coerce")
    raw = raw.dropna(subset=cols).sort_values("Date")
    raw["ym"] = raw.Date.dt.to_period("M")
    monthly = raw.groupby("ym").tail(1).set_index("ym")
    check = svensson_yields(monthly[cols], np.arange(1, 6))
    quoted = monthly[[f"SVENY0{n}" for n in range(1, 6)]].to_numpy()
    # Published SVENY quotes are rounded to four decimal percentage points.
    np.testing.assert_allclose(check, quoted, atol=5.1e-5, rtol=0)
    bonds = load_gsw_bond_data().set_index("ym")
    market = monthly_curve_returns(monthly[cols])
    market["rf_month"] = load_cash_returns().reindex(market.index)
    # Trading targets and realized cash use bills, avoiding short-end extrapolation.
    for n in range(2, 6):
        market[f"mx{n}"] = market[f"mr{n}"]-market.rf_month
    market["mx_bar"] = market[[f"mx{n}" for n in range(2, 6)]].mean(axis=1, skipna=False)
    result = bonds.join(market)
    # Quarterly/annual labels use the same original bond aged by h/12 years.
    # Cash is the compounded realized bill account over the identical interval.
    index = result.index
    p = monthly[cols].reindex(index).to_numpy()
    for h in (3, 12):
        current = -svensson_yields(p, np.arange(2, 6))*np.arange(2, 6)
        remaining = np.arange(2, 6)-h/12
        aged = pd.DataFrame(-svensson_yields(p, remaining)*remaining, index=index)
        sold = aged.reindex(index+h).to_numpy()
        returns = 100*np.expm1((sold-current)/100)
        cash = np.ones(len(index))
        for k in range(h):
            cash *= 1+result.rf_month.reindex(index+k).to_numpy()/100
        for j, n in enumerate(range(2, 6)):
            result[f"ex{h}_{n}"] = returns[:, j]-100*(cash-1)
    return result
