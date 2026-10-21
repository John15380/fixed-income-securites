"""Mean-variance timing and historical research statistics.

Inputs are overlapping one-year labels observed monthly. Compounded returns,
drawdowns and turnover costs below are not a realizable portfolio backtest;
see README.md for interpretation and the covariance look-ahead limitation.
"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def mv_strategy(rx_hat_pct: np.ndarray, rx_act_pct: np.ndarray,
                y1_pct: np.ndarray, sigma2_dec: float,
                gamma: float, w_cap: float,
                cost_bps: float = 0.0) -> dict:
    """
    Markowitz (1952) mean-variance timing strategy.

    All return inputs are in percentage-point annual units.
    sigma2_dec is in decimal (e.g. 0.001795 for σ=4.24% pa).
    cost_bps: one-way transaction cost per unit of position change (bid-ask, bps).
              Monthly rebalancing cost = |Δw_t| × cost_bps / 10000.

    Returns metrics dict with keys:
      w, er, rp, mean_er_pct, std_er_pct, sharpe, max_dd, au, cer, mean_w,
      turnover, er_net_pct, sharpe_net, au_net, cer_net
    """
    rhat = rx_hat_pct / 100.0
    rx   = rx_act_pct / 100.0
    y1   = y1_pct     / 100.0

    w  = np.clip(rhat / (gamma * sigma2_dec), -w_cap, w_cap)
    er = w * rx
    rp = er + y1

    # Transaction costs: |Δw_t| × cost per unit per period
    w_prev   = np.concatenate([[0.0], w[:-1]])
    turnover = np.abs(w - w_prev)
    tc       = turnover * cost_bps / 10000          # decimal, per period
    er_net   = er - tc
    rp_net   = er_net + y1

    cum   = np.cumprod(1 + er)
    peak  = np.maximum.accumulate(cum)
    max_dd = ((cum - peak) / peak).min() * 100

    sharpe     = er.mean()     / er.std(ddof=1)     if er.std(ddof=1)     > 0 else np.nan
    sharpe_net = er_net.mean() / er_net.std(ddof=1) if er_net.std(ddof=1) > 0 else np.nan
    au         = rp.mean()     - 0.5 * gamma * rp.var(ddof=1)
    au_net     = rp_net.mean() - 0.5 * gamma * rp_net.var(ddof=1)

    return {
        "w":           w,
        "er":          er,
        "rp":          rp,
        "mean_er_pct": er.mean() * 100,
        "std_er_pct":  er.std(ddof=1) * 100,
        "sharpe":      sharpe,
        "max_dd":      max_dd,
        "au":          au,
        "cer":         au * 100,
        "mean_w":      np.mean(np.abs(w)),
        "turnover":    turnover.mean(),
        "er_net_pct":  er_net.mean() * 100,
        "sharpe_net":  sharpe_net,
        "au_net":      au_net,
        "cer_net":     au_net * 100,
    }


def mv_strategy_per_mat(rx_hat_dict: dict, rx_act_dict: dict,
                        y1_pct: np.ndarray, sigma2_dict: dict,
                        gamma: float, w_cap: float,
                        cost_bps: float = 0.0) -> dict:
    """
    Per-maturity mean-variance strategy.

    rx_hat_dict, rx_act_dict: {n: np.ndarray} in percentage-point annual units.
    sigma2_dict: {n: float} IS variance in decimal units.
    y1_pct: 1-year yield series aligned to the common index (pct-point annual).

    Each maturity n gets an independent weight:
        w_{n,t} = clip(rx_hat_{n,t}/100 / (gamma * sigma2_n), -w_cap, w_cap)

    Portfolio excess return (equally-weighted across maturities):
        er_t = mean_n(w_{n,t} * rx_{n,t})

    Returns same metrics dict as mv_strategy, plus:
        w_dict      : {n: weight series}
        w_curve_std : time-series std of cross-maturity weight dispersion
                      (measures how much the strategy trades the curve
                       rather than pure level/duration)
    """
    mats = sorted(rx_hat_dict.keys())
    T    = len(y1_pct)
    y1   = y1_pct / 100.0

    w_mat  = {}
    er_mat = {}
    for n in mats:
        rhat = rx_hat_dict[n] / 100.0
        rx   = rx_act_dict[n] / 100.0
        w_n  = np.clip(rhat / (gamma * sigma2_dict[n]), -w_cap, w_cap)
        w_mat[n]  = w_n
        er_mat[n] = w_n * rx

    # portfolio = equal-weight average across maturities
    er_stack = np.column_stack([er_mat[n] for n in mats])
    er       = er_stack.mean(axis=1)
    rp       = er + y1

    # transaction costs: per-maturity turnover, averaged
    tc_total = np.zeros(T)
    for n in mats:
        w_prev    = np.concatenate([[0.0], w_mat[n][:-1]])
        turn_n    = np.abs(w_mat[n] - w_prev)
        tc_total += turn_n * cost_bps / 10000
    tc_total /= len(mats)

    er_net = er - tc_total
    rp_net = er_net + y1

    # cross-maturity weight dispersion (curve exposure)
    w_stack = np.column_stack([w_mat[n] for n in mats])
    w_curve_std = np.std(w_stack, axis=1, ddof=0)

    cum  = np.cumprod(1 + er)
    peak = np.maximum.accumulate(cum)
    max_dd = ((cum - peak) / peak).min() * 100

    sharpe     = er.mean()     / er.std(ddof=1)     if er.std(ddof=1)     > 0 else np.nan
    sharpe_net = er_net.mean() / er_net.std(ddof=1) if er_net.std(ddof=1) > 0 else np.nan
    au         = rp.mean()     - 0.5 * gamma * rp.var(ddof=1)
    au_net     = rp_net.mean() - 0.5 * gamma * rp_net.var(ddof=1)

    return {
        "w_dict":        w_mat,
        "er":            er,
        "rp":            rp,
        "mean_er_pct":   er.mean() * 100,
        "std_er_pct":    er.std(ddof=1) * 100,
        "sharpe":        sharpe,
        "max_dd":        max_dd,
        "au":            au,
        "cer":           au * 100,
        "mean_w":        np.mean([np.abs(w_mat[n]).mean() for n in mats]),
        "turnover":      tc_total.mean() / (cost_bps / 10000) if cost_bps > 0 else
                         np.mean([np.abs(w_mat[n] - np.concatenate([[0.], w_mat[n][:-1]])).mean()
                                  for n in mats]),
        "er_net_pct":    er_net.mean() * 100,
        "sharpe_net":    sharpe_net,
        "au_net":        au_net,
        "cer_net":       au_net * 100,
        "w_curve_std":   w_curve_std.mean(),
    }


def mv_strategy_shrinkage(rx_hat_dict: dict, rx_act_dict: dict,
                          y1_pct: np.ndarray, gamma: float, w_cap: float,
                          cost_bps: float = 0.0,
                          shrink_type: str = "lw") -> dict:
    """
    Thornton & Valente (RFS 2012) — full 4×4 MV with shrinkage.

    Instead of independent per-maturity weights, estimates the full 4×4
    covariance matrix of bond excess returns and applies Ledoit-Wolf
    shrinkage before computing MV weights.

    w_t = clip(Σ_shrink⁻¹ * μ̂_t / γ, -w_cap, w_cap)

    where Σ_shrink = (1-λ) * Σ_sample + λ * diag(Σ_sample)
    and μ̂_t = [rx_hat_2(t), rx_hat_3(t), rx_hat_4(t), rx_hat_5(t)]' / 100

    Returns same metrics dict as mv_strategy_per_mat, plus:
        w_mat      : dict of per-maturity weight arrays
        w_curve_std: cross-maturity weight dispersion
    """
    mats = sorted(rx_hat_dict.keys())
    K    = len(mats)
    T    = len(y1_pct)
    y1   = y1_pct / 100.0

    rhat_mat = np.column_stack([rx_hat_dict[n] / 100.0 for n in mats])
    rx_mat   = np.column_stack([rx_act_dict[n]  / 100.0 for n in mats])

    # Estimate full-sample shrinkage covariance on excess returns
    # (TV2012 uses IS data to estimate Σ; same principle as σ²_is)
    lw = LedoitWolf().fit(rx_mat)
    Sigma = lw.covariance_  # K×K shrinkage estimate

    # Clip extreme correlations for robustness
    Sigma = np.clip(Sigma, -0.01, 0.01) * 100 + Sigma * 0.99  # mild regularisation

    w_mat = np.zeros((T, K))
    for t in range(T):
        mu = rhat_mat[t]
        try:
            w_raw = np.linalg.solve(gamma * Sigma, mu)
        except np.linalg.LinAlgError:
            w_raw = mu / (gamma * np.diag(Sigma))
        w_mat[t] = np.clip(w_raw, -w_cap, w_cap)

    er_k    = w_mat * rx_mat
    er      = er_k.mean(axis=1)
    rp      = er + y1

    # Transaction costs
    w_prev  = np.vstack([np.zeros(K), w_mat[:-1]])
    turn_k  = np.abs(w_mat - w_prev)
    tc_total = (turn_k * cost_bps / 10000).mean(axis=1)
    er_net  = er - tc_total
    rp_net  = er_net + y1

    w_curve_std = np.std(w_mat, axis=1, ddof=0).mean()

    cum   = np.cumprod(1 + er)
    peak  = np.maximum.accumulate(cum)
    max_dd = ((cum - peak) / peak).min() * 100

    sharpe     = er.mean()     / er.std(ddof=1)     if er.std(ddof=1)     > 0 else np.nan
    sharpe_net = er_net.mean() / er_net.std(ddof=1) if er_net.std(ddof=1) > 0 else np.nan
    au         = rp.mean()     - 0.5 * gamma * rp.var(ddof=1)
    au_net     = rp_net.mean() - 0.5 * gamma * rp_net.var(ddof=1)

    return {
        "w_dict":  {n: w_mat[:, i] for i, n in enumerate(mats)},
        "er":      er,
        "rp":      rp,
        "mean_er_pct": er.mean() * 100,
        "std_er_pct":  er.std(ddof=1) * 100,
        "sharpe":      sharpe,
        "max_dd":      max_dd,
        "au":          au,
        "cer":         au * 100,
        "mean_w":      np.mean(np.abs(w_mat)),
        "turnover":    turn_k.mean(),
        "er_net_pct":  er_net.mean() * 100,
        "sharpe_net":  sharpe_net,
        "au_net":      au_net,
        "cer_net":     au_net * 100,
        "w_curve_std": w_curve_std,
    }
