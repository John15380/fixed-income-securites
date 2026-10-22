"""HAC-studentized circular-block inference for forecast loss differences.

Conditional on the saved forecast path: this is not a bootstrap that re-estimates
every historical PCA, first-stage factor or model-selection decision.
"""

import numpy as np


def mean_hac_t(values, lags=18):
    """Bartlett HAC t statistic for a mean, with n/(n-1) correction."""
    x = np.asarray(values, dtype=float)
    if x.ndim not in (1, 2) or x.shape[-1] < 3 or not np.isfinite(x).all():
        raise ValueError("Expected finite series or bootstrap rows, at least 3 observations")
    n = x.shape[-1]
    kmax = min(int(lags), n-1)
    if kmax < 0:
        raise ValueError("HAC lag length must be nonnegative")
    mean = x.mean(axis=-1)
    residual = x-mean[..., None]
    lrv = np.sum(residual**2, axis=-1)/n
    for k in range(1, kmax+1):
        lrv += 2*(1-k/(kmax+1))*np.sum(residual[..., k:]*residual[..., :-k], axis=-1)/n
    se = np.sqrt(np.maximum(lrv, 0)/(n-1))
    statistic = np.divide(mean, se, out=np.zeros_like(mean), where=se > 1e-14)
    signed_infinity = np.where(mean >= 0, np.inf, -np.inf)
    statistic = np.where((se <= 1e-14) & (np.abs(mean) > 1e-14), signed_infinity, statistic)
    return statistic


def block_bootstrap_t(values, *, lags=18, block=24, replications=1999, seed=42):
    """One-sided H0 E[d]<=0, centered-null circular bootstrap, re-studentized.

    Positive d means the alternative improves forecast loss or net return.
    Block length and HAC lag are distinct tuning choices, both disclosed.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) < 3 or not np.isfinite(x).all():
        raise ValueError("Expected a finite one-dimensional series")
    if not 1 <= block <= len(x) or replications < 99:
        raise ValueError("Invalid block length or too few bootstrap replications")
    if np.allclose(x, 0):
        return {"statistic": 0., "bootstrap_p": 1., "block": block, "replications": replications}
    observed = float(mean_hac_t(x, lags))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(x), (replications, int(np.ceil(len(x)/block))))
    indices = ((starts[:, :, None]+np.arange(block)) % len(x)).reshape(replications, -1)[:, :len(x)]
    null = (x-x.mean())[indices]
    simulated = mean_hac_t(null, lags)
    p = (1+np.count_nonzero(simulated >= observed))/(replications+1)
    return {"statistic": observed, "bootstrap_p": p, "block": block, "replications": replications}


def joint_block_bootstrap_t(differences, *, lags=6, block=12, replications=1999, seed=42):
    """Paired, centered-null max-t test across a disclosed set of forecast paths.

    Columns are models, rows are common consecutive dates. Every bootstrap draw
    keeps the models together and re-estimates the HAC standard error. This
    controls this experiment's comparison family, conditional on the forecasts;
    it does not cover prior undocumented research or re-estimate latent factors.
    """
    d = np.asarray(differences, dtype=float)
    if d.ndim != 2 or d.shape[0] < 3 or not np.isfinite(d).all():
        raise ValueError("Expected a finite date-by-model matrix")
    n, m = d.shape
    if not m or not 1 <= block <= n or replications < 99:
        raise ValueError("Invalid model count, block length or replication count")
    observed = mean_hac_t(d.T, lags)
    centered = d-d.mean(axis=0)
    rng = np.random.default_rng(seed)
    counts, adjusted = np.zeros(m), np.zeros(m)
    for begin in range(0, replications, 32):
        b = min(32, replications-begin)
        starts = rng.integers(0, n, (b, int(np.ceil(n/block))))
        ix = ((starts[:, :, None]+np.arange(block)) % n).reshape(b, -1)[:, :n]
        draws = centered[ix].transpose(0, 2, 1)
        t = mean_hac_t(draws.reshape(b*m, n), lags).reshape(b, m)
        counts += (t >= observed).sum(axis=0)
        adjusted += (t.max(axis=1)[:, None] >= observed).sum(axis=0)
    return {"statistic": observed, "bootstrap_p": (counts+1)/(replications+1),
            "max_t_p": (adjusted+1)/(replications+1), "models": m,
            "block": block, "replications": replications}
