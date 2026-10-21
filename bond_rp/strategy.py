"""Self-financing, non-overlapping one-year bond allocation experiments."""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .config import (GAMMA_BASE, HORIZON_MONTHS, MATS, MIN_TRAIN_OBS,
                     REBALANCE_MONTH, TRANSACTION_COST_BPS, W_CAP)
from .metrics import annual_performance
from .models import matured_training


def allocation_weights(expected, historical_excess, gamma=GAMMA_BASE, cap=W_CAP, method="scalar",
                       target_vol=.04):
    """Four actual NAV weights; the total absolute exposure is bounded by cap.

    historical_excess MUST consist of labels known at the decision date. The
    date-aware backtest below enforces this. All inputs are simple decimals.
    long_only clips mean-variance weights to [0, cap]; vol_target uses only the
    forecast sign and target_vol / past basket volatility. direct interprets
    expected as a requested basket weight (e.g. an outcome-free trend vote).
    """
    history = np.asarray(historical_excess, dtype=float)
    if history.ndim != 2 or history.shape[1] != len(MATS) or not np.isfinite(history).all():
        raise ValueError("Expected finite historical excess returns with four maturities")
    if gamma <= 0 or cap <= 0 or target_vol <= 0:
        raise ValueError("gamma, cap and target volatility must be positive")
    mu = np.asarray(expected, dtype=float)
    if mu.ndim == 0:
        mu = np.repeat(mu, len(MATS))
    if mu.shape != (len(MATS),) or not np.isfinite(mu).all():
        raise ValueError("Expected one forecast or four maturity forecasts")
    if method in {"scalar", "long_only", "vol_target", "direct"}:
        variance = np.var(history.mean(axis=1), ddof=1)
        if method == "direct":
            weight = np.clip(mu.mean(), 0, cap)
        elif method == "vol_target":
            weight = min(cap, target_vol/np.sqrt(max(variance, 1e-10))) if mu.mean() > 0 else 0.
        else:
            weight = np.clip(mu.mean()/(gamma*max(variance, 1e-10)),
                             0 if method == "long_only" else -cap, cap)
        return np.repeat(weight/len(MATS), len(MATS))
    if method == "diagonal":
        covariance = np.diag(np.maximum(np.var(history, axis=0, ddof=1), 1e-10))
    elif method == "shrinkage":
        covariance = LedoitWolf().fit(history).covariance_
    else:
        raise ValueError(method)
    weights = np.linalg.solve(gamma*(covariance+np.eye(len(MATS))*1e-12), mu)
    gross = np.abs(weights).sum()
    return weights*min(1.0, cap/gross) if gross else weights


def annual_backtest(bonds, predictions, *, gamma=GAMMA_BASE, cap=W_CAP,
                    cost_bps=TRANSACTION_COST_BPS, rebalance_month=REBALANCE_MONTH,
                    method="scalar", target_vol=.04):
    """Select one calendar month, hold exactly 12 months, then liquidate all legs.

    Annual cash return is the observed one-year zero-coupon yield converted to
    simple return. Financing is symmetric at that rate (a stated assumption).
    Each leg pays entry and exit costs on its actual principal/exit market value.
    Entry fees reduce the cash balance, so their foregone interest is included.
    """
    if "ym" in bonds:
        bonds = bonds.set_index("ym")
    if not 1 <= rebalance_month <= 12 or cost_bps < 0:
        raise ValueError("Invalid rebalance month or transaction cost")
    common = predictions.dropna().index.intersection(bonds.dropna(subset=["sx_bar", "rf_simple"]).index)
    schedule = common[common.month == rebalance_month].sort_values()
    if len(schedule) < 2:
        raise ValueError("Too few annual trades")
    if not np.all(np.diff(schedule.asi8) == HORIZON_MONTHS):
        raise ValueError("Missing annual trade: do not compound across unobserved years")
    rows = []
    cost = cost_bps/10000
    for model in [*predictions.columns, "BuyHold", "Cash"]:
        for date in schedule:
            historical = matured_training(bonds, date).dropna(subset=[f"sx{n}" for n in MATS])
            if len(historical) < MIN_TRAIN_OBS:
                raise ValueError(f"Insufficient matured history at {date}")
            if model == "BuyHold":
                weights = np.repeat(1/len(MATS), len(MATS))
            elif model == "Cash":
                weights = np.zeros(len(MATS))
            else:
                weights = allocation_weights(predictions.loc[date, model]/100,
                                              historical[[f"sx{n}" for n in MATS]].to_numpy()/100,
                                              gamma, cap, method, target_vol)
            actual = bonds.loc[date, [f"simple{n}" for n in MATS]].to_numpy(dtype=float)/100
            rf = bonds.loc[date, "rf_simple"]/100
            gross = rf + weights @ (actual-rf)
            entry_fee = cost*np.abs(weights).sum()
            exit_fee = cost*(np.abs(weights)*(1+actual)).sum()
            fee_with_financing = entry_fee*(1+rf)+exit_fee
            rows.append({"model": model, "purchase": date, "sale": date+HORIZON_MONTHS,
                         "return_gross": gross, "return_net": gross-fee_with_financing,
                         "risk_free": rf, "cost": fee_with_financing,
                         "gross_exposure": np.abs(weights).sum(), "net_exposure": weights.sum(),
                         "train_last_label_end": historical.index.max()+HORIZON_MONTHS,
                         **{f"w{n}": weights[i] for i, n in enumerate(MATS)}})
    trades = pd.DataFrame(rows)
    results = []
    for model, group in trades.groupby("model", sort=False):
        result = annual_performance(group["return_net"], group["risk_free"], gamma)
        result.update(model=model, start=str(schedule.min()), end=str(schedule.max()+HORIZON_MONTHS),
                      mean_gross_exposure=group["gross_exposure"].mean(),
                      mean_cost_bps=10000*group["cost"].mean())
        results.append(result)
    summary = pd.DataFrame(results).set_index("model")
    summary["delta_cer_bps"] = 100*(summary["cer_pct"]-summary.loc["EH", "cer_pct"])
    return summary, trades
