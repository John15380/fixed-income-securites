"""Reproducible research settings; all return units are explicit at boundaries."""
import os
from pathlib import Path
from dataclasses import dataclass, asdict
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("BOND_RP_DATA_DIR", PROJECT_ROOT/"GroupProjectData"))
TRAIN_START = pd.Period("1964-01", "M")
HORIZON_MONTHS = 12
MATS = [2, 3, 4, 5]
TI_LOOKBACK = 12
NW_LAGS = 18

@dataclass(frozen=True)
class StrategyConfig:
    start: str = "2000-01"
    end: str = "2026-01"
    training_months: int = 120
    macro_lag: int = 2
    ma_months: int = 12
    signal_shrink: float = .5
    risk_months: int = 60
    gamma: float = 5.
    exposure_cap: float = 1.
    one_way_cost_bps: float = .5
    seed: int = 42
    def to_dict(self):
        return asdict(self)

DEFAULT = StrategyConfig()
