"""Global constants and configuration."""

import os
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.environ.get("BOND_RP_DATA_DIR", os.path.join(PROJECT_ROOT, "GroupProjectData"))

# ── Sample design ──────────────────────────────────────────────────────────────
OOS_START  = pd.Period("1990-01", "M")   # baseline OOS start
IS_END     = pd.Period("1989-12", "M")   # in-sample window end (PCA fit for OOS)

# ── Model parameters ───────────────────────────────────────────────────────────
NW_LAGS     = 18
GAMMA_BASE  = 5
W_CAP       = 3.0
MATS        = [2, 3, 4, 5]
TI_LOOKBACK = 12

# ── Robustness grids ───────────────────────────────────────────────────────────
TI_LOOKBACKS         = [3, 6, 9, 12, 24]          # Table 4f: lookback sensitivity
GAMMA_GRID           = [1,2,3,4,5,6,7,8,10,15,20] # Table 4b / γ optimization
PUB_LAG_MONTHS       = 1                           # Table 4h: default publication lag
TRANSACTION_COST_BPS = 0.5                         # Table 4i: bid-ask per unit (bps)

# ── Macro series transformation codes (Stock & Watson 2002) ───────────────────
# 1=level, 2=Δ, 4=log, 5=Δlog×100, 6=Δ²log×100
TRANSFORM = {
    "PI": 5, "PI less transfers": 5, "Real Consumption": 5,
    "M&T sales": 5, "Retail sales": 5,
    "IP: total": 5, "IP: products": 5, "IP: final prod": 5,
    "IP: cons gds": 5, "IP: cons dble": 5, "IP: cons nondble": 5,
    "IP: bus eqpt": 5, "IP: matls": 5, "IP: dble matls": 5,
    "IP: nondble matls": 5, "IP: mfg": 5, "IP: res util": 5, "IP: fuels": 5,
    "NAPM prodn ": 1, "Cap util": 2,
    "Help wanted indx": 5, "Help wanted/unemp": 2,
    "Emp CPS total": 5, "Emp CPS nonag": 5,
    "U: all": 2, "U: mean duration": 2, "U < 5 wks": 2, "U 5-14 wks": 2,
    "U 15+ wks ": 2, "U 15-26 wks": 2, "U 27+ wks": 2, "UI claims": 5,
    "Emp: total": 5, "Emp: gds prod": 5, "Emp: mining": 5, "Emp: const": 5,
    "Emp: mfg": 5, "Emp: dble gds": 5, "Emp: nondbles": 5, "Emp: services": 5,
    "Emp: TTU": 5, "Emp: wholesale": 5, "Emp: retail": 5, "Emp: FIRE": 5,
    "Emp: Govt": 5,
    "Agg wkly hours": 5, "Avg hrs ": 2, "Overtime: mfg": 2, "Avg hrs: mfg": 2,
    "NAPM empl": 1,
    "Starts: nonfarm": 4, "Starts: NE": 4, "Starts: MW": 4,
    "Starts: South": 4, "Starts: West": 4,
    "BP: total": 4, "BP: NE": 4, "BP: MW": 4, "BP: South": 4, "BP: West": 4,
    "PMI": 1, "NAPM new ordrs": 1, "NAPM vendor del": 1, "NAPM Invent": 1,
    "Orders: cons gds": 5, "Orders: dble gds": 5, "Orders: cap gds": 5,
    "Unf orders: dble": 5, "M&T invent": 5, "M&T invent/sales": 2,
    "M1": 6, "M2": 6, "Currency": 6, "M2 (real)": 6, "MB": 6,
    "Reserves tot": 6, "Reserves nonbor": 6,
    "C&I loan plus": 5, "DC&I loans": 1,
    "Cons credit": 6, "Inst cred/PI": 2,
    "S&P 500": 5, "S&P: indust": 5, "S&P div yield": 2, "S&P PE ratio": 2,
    "Fed Funds": 2, "Comm paper": 2,
    "3 mo T-bill": 2, "6 mo T-bill": 2, "1 yr T-bond": 2,
    "5 yr T-bond": 2, "10 yr T-bond": 2, "Aaa bond": 2, "Baa bond": 2,
    "CP-FF spread": 1, "3 mo-FF spread": 1, "6 mo-FF spread": 1,
    "1 yr-FF spread": 1, "5 yr-FF spread": 1, "10 yr-FF spread": 1,
    "Aaa-FF spread": 1, "Baa-FF spread": 1,
    "Ex rate: avg": 5, "Ex rate: Switz": 5, "Ex rate: Japan": 5,
    "Ex rate: UK": 5, "EX rate: Canada": 5,
    "PPI: fin gds": 6, "PPI: cons gds": 6,
    "PPI: int mat’ls": 6, "PPI: crude mat’ls": 6,
    "Spot market price": 6, "PPI: nonferrous": 6, "NAPM com price": 1,
    "CPI-U: all": 6, "CPI-U: apparel": 6, "CPI-U: transp": 6,
    "CPI-U: medical": 6, "CPI-U: comm.": 6, "CPI-U: dbles": 6,
    "CPI-U: services": 6, "CPI-U: ex food": 6, "CPI-U: ex shelter": 6,
    "CPI-U: ex med": 6,
    "PCE defl": 6, "PCE defl: dlbes": 6, "PCE defl: nondble": 6,
    "PCE defl: service": 6,
    "AHE: goods": 6, "AHE: const": 6, "AHE: mfg": 6,
    "Consumer expect": 2,
}
