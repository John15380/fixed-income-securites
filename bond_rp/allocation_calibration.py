"""Bounded investment calibration using mature validation outcomes only.

The calibration changes the strength of a forecast, not the investor's gamma.
It never chooses leverage, risk windows, gamma, or transaction cost assumptions.
All means are excess-return forecasts in percentage points; realized asset and
cash returns and the basket variance are decimal quantities.  Four columns
always refer, in order, to the two-, three-, four-, and five-year zero bonds.
"""

import itertools

import numpy as np
import pandas as pd


STRENGTH_GRID = tuple(np.arange(0., 2.001, .25))
BIAS_GRID = tuple(np.round(np.arange(-.2, .201, .05), 8))


def _monthly_index(index):
    if isinstance(index, pd.PeriodIndex):
        result = index.asfreq("M")
    elif isinstance(index, pd.DatetimeIndex):
        result = index.to_period("M")
    else:
        result = pd.PeriodIndex(index, freq="M")
    if result.has_duplicates or not result.is_monotonic_increasing:
        raise ValueError("Inputs must have unique, increasing monthly dates")
    return result


def _aligned(value, index, name, columns=None):
    if not isinstance(value, (pd.DataFrame, pd.Series)):
        raise TypeError(f"{name} must have an explicit monthly index")
    value = value.copy()
    value.index = _monthly_index(value.index)
    value = value.reindex(index)
    array = value.to_numpy(dtype=float)
    if columns is None:
        if array.ndim != 1:
            raise ValueError(f"{name} must be a Series")
    elif array.shape != (len(index), columns):
        raise ValueError(f"{name} must have exactly {columns} maturity columns")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} has missing or nonfinite evaluation inputs")
    return array


def allocation_path(mu_pct, eh_pct, asset_returns, risk_free, risk_variance,
                    risk_last_label_end, *, signal_strength=1., bias_sigma=0.,
                    gamma=5., cap=1., cost_bps=.5):
    """Return exact self-financing, long-only monthly cash flows.

    The conditional forecast is EH + strength*(model-EH) + bias*monthly_sigma.
    A common bias is added to all maturities; sigma comes from the supplied,
    strictly matured equal-basket variance.  Bias is in standard-deviation
    units, *not* percentage points.  The desired basket allocation is
    clip(mean(forecast)/(gamma*variance), 0, cap).  It is divided by (1+c) if
    necessary to reserve entry fees, where c is the one-way proportional cost.

    Each purchase is liquidated next month.  Entry fees reduce invested cash
    immediately, and exit fees are paid on the actual sale proceeds.  Future
    asset/cash returns affect the accounting only, never the allocation.
    risk_last_label_end must be a Series of actual last observed risk outcome
    months and must be strictly before each corresponding decision month.
    This checks provenance metadata; callers must also estimate that variance
    from the indicated historical outcomes.
    """
    if not isinstance(mu_pct, pd.DataFrame):
        raise TypeError("mu_pct must be a four-column DataFrame")
    index = _monthly_index(mu_pct.index)
    if not len(index) or not (np.diff(index.asi8) == 1).all():
        raise ValueError("Evaluation must have a nonempty, complete monthly calendar")
    scalar_parameters = (signal_strength, bias_sigma, gamma, cap, cost_bps)
    if not np.isfinite(scalar_parameters).all() or gamma <= 0 or not 0 <= cap <= 1:
        raise ValueError("Invalid parameters; only nonlevered long-only allocations are supported")
    if not 0 <= signal_strength <= 2 or not -.2 <= bias_sigma <= .2:
        raise ValueError("Calibration strength/bias exceeds the declared bounded protocol")
    if not 0 <= cost_bps < 10000:
        raise ValueError("Invalid proportional cost")
    mu = _aligned(mu_pct, index, "mu_pct", 4)/100
    eh = _aligned(eh_pct, index, "eh_pct", 4)/100
    actual = _aligned(asset_returns, index, "asset_returns", 4)
    rf = _aligned(risk_free, index, "risk_free")
    variance = _aligned(risk_variance, index, "risk_variance")
    if (variance <= 0).any() or (actual <= -1).any() or (rf <= -1).any():
        raise ValueError("Variance must be positive and gross returns positive")
    if not isinstance(risk_last_label_end, pd.Series):
        raise TypeError("risk_last_label_end must be a Series")
    cutoffs = risk_last_label_end.copy()
    cutoffs.index = _monthly_index(cutoffs.index)
    cutoffs = cutoffs.reindex(index)
    if cutoffs.isna().any():
        raise ValueError("Every risk estimate needs a last observed outcome month")
    last_end = pd.PeriodIndex(cutoffs.astype(str), freq="M")
    if not (last_end < index).all():
        raise ValueError("Risk variance consumes an immature outcome")
    calibrated = eh+signal_strength*(mu-eh)+bias_sigma*np.sqrt(variance)[:, None]
    desired = np.clip(calibrated.mean(axis=1)/(gamma*variance), 0., cap)
    c = cost_bps/10000
    exposure = np.minimum(desired, 1/(1+c))
    q = np.repeat((exposure/4)[:, None], 4, axis=1)
    entry_fee = c*exposure
    cash = 1-exposure-entry_fee
    sale_proceeds = (q*(1+actual)).sum(axis=1)
    exit_fee = c*sale_proceeds
    net = cash*(1+rf)+sale_proceeds-exit_fee-1
    gross = rf+(q*(actual-rf[:, None])).sum(axis=1)
    cost = entry_fee*(1+rf)+exit_fee
    np.testing.assert_allclose(net, gross-cost, atol=5e-16, rtol=1e-12)
    if (cash < -1e-14).any():
        raise ArithmeticError("Entry fees were not fully reserved")
    return pd.DataFrame({
        "purchase": index.astype(str), "sale": (index+1).astype(str),
        "return_net": net, "return_gross": gross, "risk_free": rf,
        "cost": cost, "entry_fee": entry_fee, "exit_fee": exit_fee,
        "financing": 0., "cash_after_entry": cash,
        "gross_exposure": exposure, "desired_exposure": desired,
        "risk_variance": variance, "risk_last_label_end": last_end.astype(str),
        "forecast_excess": calibrated.mean(axis=1),
        **{f"q{n}": q[:, j] for j, n in enumerate(range(2, 6))},
    }, index=index)


def annualized_utility(returns):
    """12*(mean monthly net total return - 5/2*sample monthly variance)."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Utility requires at least two finite monthly returns")
    return float(12*(values.mean()-2.5*values.var(ddof=1)))


def calibrate_allocation(mu_pct, eh_pct, asset_returns, risk_free, risk_variance,
                         risk_last_label_end, *, validation_start="2011-01",
                         validation_end="2016-12", freeze_at="2017-01",
                         split_at="2014-01", strengths=STRENGTH_GRID,
                         biases=BIAS_GRID, cost_bps=.5):
    """Choose bounded signal strength/bias using two historical validation eras.

    Returns ``(parameters, all_candidates, selected_validation_trades)``.
    ``parameters`` contains just signal_strength and bias_sigma and can be
    passed unchanged as **kwargs to allocation_path on any later test panel.

    Only purchases in validation_start..validation_end with purchase+1 strictly
    before freeze_at are ever read.  Thus a 2017-01 freeze excludes the 2016-12
    purchase, whose one-month return is not yet eligible under this project's
    strict maturity convention.  The first era is before split_at; the second
    starts at split_at.  Defaults cover 2011-2013 and 2014-2016; each era must
    contain at least 12 months.  Defaults evaluate 81 fixed settings.  To keep
    model selection and investment calibration temporally separate, callers
    can use validation_start="2014-01", split_at="2015-07", after tuning model
    hyperparameters only on returns available strictly before 2014-01.

    gamma=5 and cap=1 are fixed.  For each era we calculate realized, after-fee
    total-return utility and its gain over the *same* EH allocation.  The score
    is .5*mean(gains)+.5*min(gains), equal to .75*worst+.25*best.  This penalizes
    gains driven by one era.  It is a tuning objective, not a hypothesis test.
    Ties prefer the uncalibrated strength=1,bias=0 setting and then less bias.
    The full search table is retained; test-period ranking is never consulted.
    Passing mu_pct=eh_pct calibrates the EH benchmark under the same bias grid;
    strength is then redundant and the deterministic tie rule prefers one.
    """
    if not isinstance(mu_pct, pd.DataFrame):
        raise TypeError("mu_pct must be a four-column DataFrame")
    index = _monthly_index(mu_pct.index)
    begin, end, freeze, split = (pd.Period(x, "M") for x in
                                 (validation_start, validation_end, freeze_at, split_at))
    if begin < pd.Period("2011-01", "M") or end > pd.Period("2016-12", "M") or begin > end:
        raise ValueError("Declared investment validation is restricted to 2011-2016")
    chosen = index[(index >= begin) & (index <= end) & (index+1 < freeze)]
    # Slice before inspecting values: post-validation outcomes, even missing or
    # arbitrarily changed ones, cannot influence the calibration.
    inputs = []
    for value in (mu_pct, eh_pct, asset_returns, risk_free, risk_variance, risk_last_label_end):
        if not isinstance(value, (pd.DataFrame, pd.Series)):
            raise TypeError("All calibration inputs require an explicit monthly index")
        frame = value.copy()
        frame.index = _monthly_index(frame.index)
        inputs.append(frame.reindex(chosen))
    first = chosen < split
    second = chosen >= split
    if min(first.sum(), second.sum()) < 12:
        raise ValueError("Both validation eras must include at least 12 eligible months")
    strengths, biases = tuple(strengths), tuple(biases)
    if not strengths or not biases:
        raise ValueError("Calibration grids cannot be empty")
    if len(set(strengths)) != len(strengths) or len(set(biases)) != len(biases):
        raise ValueError("Duplicate calibration grid points")
    if (not np.isfinite(strengths).all() or not np.isfinite(biases).all()
            or min(strengths) < 0 or max(strengths) > 2 or min(biases) < -.2 or max(biases) > .2):
        raise ValueError("Candidate grids exceed the declared strength/bias bounds")
    if 0. not in strengths or 0. not in biases:
        raise ValueError("The candidate grid must retain the unchanged EH setting (0, 0)")
    baseline = allocation_path(*inputs, signal_strength=0., bias_sigma=0., cost_bps=cost_bps)
    baseline_utilities = [annualized_utility(baseline.return_net.to_numpy()[mask])
                          for mask in (first, second)]
    records = []
    for strength, bias in itertools.product(strengths, biases):
        path = allocation_path(*inputs, signal_strength=float(strength),
                               bias_sigma=float(bias), cost_bps=cost_bps)
        utilities = np.array([annualized_utility(path.return_net.to_numpy()[mask])
                              for mask in (first, second)])
        gains = utilities-np.asarray(baseline_utilities)
        record = {
            "signal_strength": float(strength), "bias_sigma": float(bias),
            "utility_first": utilities[0], "utility_second": utilities[1],
            "eh_utility_first": baseline_utilities[0],
            "eh_utility_second": baseline_utilities[1],
            "utility_gain_first": gains[0], "utility_gain_second": gains[1],
            "utility_gain_mean": gains.mean(), "utility_gain_worst": gains.min(),
            "score_annual": .5*gains.mean()+.5*gains.min(),
            "utility_full": annualized_utility(path.return_net),
            "mean_exposure": path.gross_exposure.mean(),
            "mean_exposure_first": path.gross_exposure.to_numpy()[first].mean(),
            "mean_exposure_second": path.gross_exposure.to_numpy()[second].mean(),
            "eh_mean_exposure_first": baseline.gross_exposure.to_numpy()[first].mean(),
            "eh_mean_exposure_second": baseline.gross_exposure.to_numpy()[second].mean(),
            "months_first": int(first.sum()), "months_second": int(second.sum()),
            "first_period": f"{chosen[first].min()}..{chosen[first].max()}",
            "second_period": f"{chosen[second].min()}..{chosen[second].max()}",
            "validation_first_purchase": str(chosen.min()),
            "validation_last_purchase": str(chosen.max()),
            "validation_last_label_end": str(chosen.max()+1),
            "freeze_at": str(freeze), "split_at": str(split),
            "gamma": 5., "cap": 1., "cost_bps": cost_bps,
        }
        # Preserve the original era-specific column names only for the default
        # split, where their labels remain accurate.  Generic fields above are
        # the stable API for callers using the later investment-only interval.
        if split == pd.Period("2014-01", "M"):
            for generic, old in (("first", "2011_2013"), ("second", "2014_2016")):
                for prefix in ("utility", "eh_utility", "utility_gain", "mean_exposure",
                               "eh_mean_exposure", "months"):
                    record[f"{prefix}_{old}"] = record[f"{prefix}_{generic}"]
        records.append(record)
    table = pd.DataFrame(records)
    # Treat only floating-point equality as a tie, not a wide "close" band.
    best_score = table.score_annual.max()
    tied = table.loc[np.abs(table.score_annual-best_score) <= 1e-14].copy()
    tied["distance_from_default"] = np.abs(tied.signal_strength-1)+np.abs(tied.bias_sigma)
    tied["absolute_bias"] = np.abs(tied.bias_sigma)
    best_index = tied.sort_values(["distance_from_default", "absolute_bias",
                                   "signal_strength", "bias_sigma"], kind="stable").index[0]
    parameters = {key: float(table.loc[best_index, key])
                  for key in ("signal_strength", "bias_sigma")}
    table["selected"] = table.index == best_index
    path = allocation_path(*inputs, **parameters, cost_bps=cost_bps)
    return parameters, table, path
