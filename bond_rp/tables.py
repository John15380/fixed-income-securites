"""Aligned forecast tables and non-overlapping annual robustness diagnostics."""

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from .metrics import hit_rate, loss_test, oos_r2
from .strategy import annual_backtest

NESTED_OLS = {"FB", "CP", "TI", "LN", "CPLN", "BW", "LNBW"}


def aligned_forecasts(predictions, bonds):
    if "ym" in bonds:
        bonds = bonds.set_index("ym")
    aligned = predictions.join(bonds["sx_bar"].rename("actual"), how="inner").dropna()
    if len(aligned) < 24:
        raise ValueError("Insufficient common evaluation months")
    if not np.all(np.diff(aligned.index.asi8) == 1):
        raise ValueError("HAC evaluation requires consecutive monthly observations")
    return aligned


def prediction_table(aligned):
    actual, benchmark = aligned["actual"].to_numpy(), aligned["EH"].to_numpy()
    rows = []
    for name in aligned.columns.drop("actual"):
        prediction = aligned[name].to_numpy()
        statistic, p = loss_test(actual, prediction, benchmark, nested=name in NESTED_OLS)
        rows.append({"model": name, "n": len(aligned), "start": str(aligned.index.min()),
                     "end": str(aligned.index.max()), "r2_pct": 100*oos_r2(actual, prediction, benchmark),
                     "rmse_pct": np.sqrt(np.mean((actual-prediction)**2)),
                     "hit_pct": 100*hit_rate(actual, prediction), "hac_stat": statistic, "hac_p": p,
                     "test": "CW-HAC" if name in NESTED_OLS else "loss-difference-HAC"})
    table = pd.DataFrame(rows).set_index("model")
    alternatives = table.index != "EH"
    table.loc[alternatives, "holm_p"] = multipletests(table.loc[alternatives, "hac_p"], method="holm")[1]
    return table


def annual_robustness(bonds, predictions):
    offsets, costs, allocations = [], [], []
    for month in range(1, 13):
        summary, _ = annual_backtest(bonds, predictions, rebalance_month=month)
        summary["month"] = month
        offsets.append(summary.reset_index())
    for cost in [0.0, 0.5, 2.0, 5.0, 10.0]:
        summary, _ = annual_backtest(bonds, predictions, cost_bps=cost)
        summary["cost_bps"] = cost
        costs.append(summary.reset_index())
    for method in ["scalar", "diagonal", "shrinkage"]:
        # A scalar forecast is repeated across maturities for the allocation comparison.
        summary, _ = annual_backtest(bonds, predictions[["EH", "CP", "LN"]], method=method)
        summary["allocation"] = method
        allocations.append(summary.reset_index())
    return {"offset": pd.concat(offsets, ignore_index=True),
            "cost": pd.concat(costs, ignore_index=True),
            "allocation": pd.concat(allocations, ignore_index=True)}


def markdown_table(table, columns):
    lines = ["| 模型 | " + " | ".join(label for _, label in columns) + " |",
             "| --- | " + " | ".join("---" for _ in columns) + " |"]
    for name, row in table.iterrows():
        lines.append("| " + str(name) + " | " + " | ".join(("—" if pd.isna(row[key]) else str(int(row[key])) if key in {"n", "years"} else f"{row[key]:.2f}") for key, _ in columns) + " |")
    return "\n".join(lines)
