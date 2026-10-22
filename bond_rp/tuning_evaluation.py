"""Matched-account economic evaluation after frozen pre-2014 model selection.

Only 2014-01..2016-11 outcomes select investment-calibration parameters.
All test/follow-up forecasts are supplied by the caller; nothing in this module
selects a predictive model or hyperparameters using their test performance.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .allocation_calibration import allocation_path, annualized_utility, calibrate_allocation
from .trading import monthly_performance


PERIODS = {
    "test": ("2017-01", "2022-11"),
    "followup": ("2022-12", "2025-11"),
    "test_2017_2019": ("2017-01", "2019-12"),
    "test_2020_2022": ("2020-01", "2022-11"),
}
LATE_CALIBRATION = dict(validation_start="2014-01", validation_end="2016-12",
                        split_at="2015-07", freeze_at="2017-01")


def _indexed(frame):
    result = frame.copy()
    if isinstance(result.index, pd.DatetimeIndex):
        result.index = result.index.to_period("M")
    else:
        result.index = pd.PeriodIndex(result.index, freq="M")
    if result.index.has_duplicates or not result.index.is_monotonic_increasing:
        raise ValueError("Monthly inputs require unique increasing dates")
    return result


def risk_inputs(bonds, dates, window=60):
    """Return total asset returns, cash, strictly mature basket risk and audit.

    Return tuple: four-column DataFrame with string columns '2'..'5', RF
    Series, equal-basket variance Series, and last outcome-month Series.
    All return/variance inputs use decimal units.  Risk observations are
    purchases s in the last ``window`` eligible calendar months with s+1<t;
    rows missing any maturity are removed, with at least 36 complete rows
    required.  Thus the latest possible purchase is t-2 and outcome is t-1.
    """
    if not isinstance(window, (int, np.integer)) or window < 36:
        raise ValueError("Risk window must contain at least 36 monthly slots")
    market = _indexed(bonds)
    index = pd.PeriodIndex(dates, freq="M")
    if index.has_duplicates or not index.is_monotonic_increasing or not len(index):
        raise ValueError("Evaluation dates must be unique, increasing and nonempty")
    columns = [str(n) for n in range(2, 6)]
    assets = market.reindex(index)[[f"mr{n}" for n in range(2, 6)]]/100
    assets.columns = columns
    cash = market.rf_month.reindex(index)/100
    if not np.isfinite(assets.to_numpy()).all() or not np.isfinite(cash).all():
        raise ValueError("Missing realized asset/cash outcomes on evaluation dates")
    variances, last_end = [], []
    for date in index:
        eligible = (market.index+1 < date) & (market.index >= date-window-1)
        past = market.loc[eligible, [f"mx{n}" for n in range(2, 6)]].dropna()/100
        if len(past) < 36 or not np.isfinite(past.to_numpy()).all():
            raise ValueError(f"Insufficient complete matured risk history at {date}")
        variances.append(max(float(past.mean(axis=1).var(ddof=1)), 1e-10))
        last_end.append(str(past.index.max()+1))
    return (assets, cash, pd.Series(variances, index=index, name="risk_variance"),
            pd.Series(last_end, index=index, name="risk_last_label_end"))


def _period_path(path, begin, end):
    result = path.loc[begin:end]
    expected = pd.period_range(begin, end, freq="M")
    if not result.index.equals(expected):
        raise ValueError(f"Incomplete matched account calendar {begin}..{end}")
    return result


def _performance(path, model, period):
    stats = monthly_performance(path.return_net, path.risk_free)
    excess = path.return_net-path.risk_free
    stats.update(model=model, period=period, purchase_start=str(path.index.min()),
                 purchase_end=str(path.index.max()), sale_end=str(path.index.max()+1),
                 annualized_arithmetic_excess_pct=float(1200*excess.mean()),
                 cumulative_excess_sum_pct=float(100*excess.sum()),
                 cumulative_total_compound_pct=float(100*(np.prod(1+path.return_net)-1)),
                 mean_gross_exposure=float(path.gross_exposure.mean()),
                 utility_annual_pct=100*annualized_utility(path.return_net))
    return stats


def _utility_interval(strategy, reference, *, block=12, replications=999, seed=42):
    """Paired circular-block percentile interval; fixed saved strategy paths.

    Draws preserve both accounts' same-month cash/return outcomes.  This does
    not re-fit models, factor estimates, or allocation calibration, and does
    not correct for the historical strategy-search family.
    """
    x, y = np.asarray(strategy, dtype=float), np.asarray(reference, dtype=float)
    if x.ndim != 1 or x.shape != y.shape or len(x) < block:
        raise ValueError("Utility inference requires aligned paths and a valid block length")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(x), (replications, int(np.ceil(len(x)/block))))
    indices = ((starts[:, :, None]+np.arange(block)) % len(x)).reshape(replications, -1)[:, :len(x)]
    a, b = x[indices], y[indices]
    draws = 12*(a.mean(axis=1)-2.5*a.var(axis=1, ddof=1)
                 -b.mean(axis=1)+2.5*b.var(axis=1, ddof=1))
    low, high = np.quantile(draws, [.025, .975])
    gain = annualized_utility(x)-annualized_utility(y)
    return {"utility_gain_bps": 10000*gain, "ci_low_bps": 10000*low,
            "ci_high_bps": 10000*high, "block": block, "replications": replications,
            "evaluation_gamma": 5., "search_adjusted": False,
            "bootstrap": "paired circular-block percentile, conditional on frozen paths"}


def _load_prior(folder, dates, prior_path):
    if prior_path is False:
        return None, None
    if prior_path is None:
        flavor = "vintage" if "vintage" in str(folder).lower() else "revised"
        prior_path = (Path(__file__).resolve().parents[1]/"output"/"refinement"/
                      "prior_candidates"/flavor/"comparison_predictions.csv")
    source = Path(prior_path)
    if not source.exists():
        return None, None
    frame = pd.read_csv(source)
    frame = frame.loc[frame.model.eq("PriorEW3/w120")].copy()
    frame.index = pd.PeriodIndex(frame.ym, freq="M")
    if frame.index.has_duplicates:
        raise ValueError("Duplicate legacy EW3 purchase months")
    frame = frame.reindex(dates)[[f"mu{n}" for n in range(2, 6)]]
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Legacy EW3 cannot cover the matched evaluation calendar")
    frame.columns = [str(n) for n in range(2, 6)]
    return frame, str(source.resolve())


def evaluate_panel(bonds, predictions, folder, *, prior_path=None, inherited_parameters=None):
    """Save matched costs, utility inference and fixed-parameter sensitivities.

    predictions is a long frame ym,model,mu2..mu5 in percentage points, with an
    'EH' model and complete monthly raw forecasts from 2014-01 to 2025-11.
    Hyperparameters must already be frozen using only pre-2014 outcomes.
    We calibrate each model, including EH, on 2014-01..2016-11.  Test and followup
    periods are fixed in PERIODS; their performance never chooses parameters.

    Return dict with parameters/calibration/trades/performance/utility/stress.
    Six review artifacts are written under folder.  prior_path=False disables
    the optional legacy EW3 comparison; a path explicitly overrides discovery.
    inherited_parameters accepts a previous allocation_parameters.json path or
    the protocol dict returned by an earlier call.  Those parameters, including
    EH, are then copied without re-optimizing investment calibration, allowing a
    matched revised/vintage comparison.  If available, the source's complete
    candidate table is copied for inspection and marked as inherited.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    required = ["ym", "model", *[f"mu{n}" for n in range(2, 6)]]
    if not set(required).issubset(predictions):
        raise ValueError("Predictions need ym, model and four maturity means")
    frame = predictions[required].copy()
    frame["ym"] = pd.PeriodIndex(frame.ym, freq="M")
    if frame.duplicated(["ym", "model"]).any():
        raise ValueError("Duplicate model/purchase forecast")
    dates = pd.period_range("2014-01", "2025-11", freq="M")
    columns = [str(n) for n in range(2, 6)]
    wide = {}
    for model, group in frame.groupby("model", sort=True):
        values = group.set_index("ym").reindex(dates)[[f"mu{n}" for n in range(2, 6)]]
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"Incomplete fixed forecast calendar for {model}")
        values.columns = columns
        wide[model] = values
    if "EH" not in wide:
        raise ValueError("The matched frozen forecast panel must contain EH")
    assets, rf, variance, last_end = risk_inputs(bonds, dates)
    common_inputs = (assets, rf, variance, last_end)
    inherited, inherited_table, inheritance_source = None, None, None
    if inherited_parameters is not None:
        if isinstance(inherited_parameters, (str, Path)):
            source_file = Path(inherited_parameters)
            inherited_protocol = json.loads(source_file.read_text(encoding="utf-8"))
            inheritance_source = str(source_file.resolve())
            candidates_file = source_file.parent/"fullcalibration.csv"
        elif isinstance(inherited_parameters, dict):
            inherited_protocol = inherited_parameters
            inheritance_source = "provided frozen protocol"
            candidate_name = inherited_protocol.get("allocation_calibration_candidates")
            candidates_file = Path(candidate_name) if candidate_name else None
        else:
            raise TypeError("inherited_parameters must be a protocol dict or JSON path")
        inherited = inherited_protocol.get("parameters", inherited_protocol)
        if not set(wide).issubset(inherited):
            raise ValueError("Inherited calibration is missing models in the matched forecast panel")
        if candidates_file is not None and candidates_file.exists():
            inherited_table = pd.read_csv(candidates_file)
            if not set(wide).issubset(set(inherited_table.model)):
                raise ValueError("Inherited candidate table does not cover the matched forecast models")
    parameters, candidate_tables, paths, specs = {}, [], {}, {}
    for model, mu in wide.items():
        if inherited is None:
            selected, candidates, _ = calibrate_allocation(mu, wide["EH"], *common_inputs,
                                                          **LATE_CALIBRATION)
            candidates.insert(0, "model", model)
            choice = candidates.loc[candidates.selected].iloc[0]
            parameters[model] = {**selected, "score_annual": float(choice.score_annual),
                                 "utility_gain_first": float(choice.utility_gain_first),
                                 "utility_gain_second": float(choice.utility_gain_second),
                                 "mean_exposure_first": float(choice.mean_exposure_first),
                                 "mean_exposure_second": float(choice.mean_exposure_second)}
            candidates["inherited"] = False
        else:
            selected = {key: float(inherited[model][key])
                        for key in ("signal_strength", "bias_sigma")}
            parameters[model] = dict(inherited[model])
            if inherited_table is None:
                candidates = pd.DataFrame([{"model": model, **parameters[model], "selected": True}])
            else:
                candidates = inherited_table.loc[inherited_table.model.eq(model)].copy()
            candidates["inherited"] = True
        candidate_tables.append(candidates)
        for suffix, config in (("default", {"signal_strength": 1., "bias_sigma": 0.}),
                               ("cal", selected)):
            label = f"{model}/{suffix}"
            paths[label] = allocation_path(mu, wide["EH"], *common_inputs, **config)
            specs[label] = (mu, config)
    # These fixed signals produce exactly a fee-reserved fully invested basket
    # or all cash under the same account implementation used for the models.
    for label, signal in (("BuyHold", 2*5*variance*100), ("Cash", -5*variance*100)):
        mu = pd.DataFrame(np.repeat(signal.to_numpy()[:, None], 4, axis=1), index=dates, columns=columns)
        config = {"signal_strength": 1., "bias_sigma": 0.}
        paths[label] = allocation_path(mu, wide["EH"], *common_inputs, **config)
        specs[label] = (mu, config)
    prior, source = _load_prior(folder, dates, prior_path)
    if prior is not None:
        config = {"signal_strength": 1., "bias_sigma": 0.}
        paths["PriorEW3"] = allocation_path(prior, wide["EH"], *common_inputs, **config)
        specs["PriorEW3"] = (prior, config)
    summary, inference = [], []
    benchmarks = ["EH/cal", "BuyHold"]+(["PriorEW3"] if prior is not None else [])
    for period, (begin, end) in PERIODS.items():
        subsets = {model: _period_path(path, begin, end) for model, path in paths.items()}
        for model, path in subsets.items():
            summary.append(_performance(path, model, period))
            for benchmark in benchmarks:
                inference.append({"model": model, "benchmark": benchmark, "period": period,
                                  "months": len(path), **_utility_interval(path.return_net,
                                                          subsets[benchmark].return_net)})
    stress_rows = []
    scenarios = (("cost2", {"cost_bps": 2.}), ("cost5", {"cost_bps": 5.}),
                 ("gamma2", {"gamma": 2.}), ("gamma10", {"gamma": 10.}),
                 ("delay1", {"delay": 1}))
    # Portfolio parameters remain the 0.5bps/gamma5 validation selections.
    # Changes in gamma here are sensitivity assumptions, not fitted preferences.
    evaluation_dates = dates[dates >= pd.Period("2017-01", "M")]
    a, f, v, cut = (value.reindex(evaluation_dates) for value in common_inputs)
    for scenario, options in scenarios:
        for model, (mu, config) in specs.items():
            kwargs = dict(options)
            delay = kwargs.pop("delay", 0)
            # Passive/cash holdings do not depend on model signal timing or
            # risk-aversion.  Preserve their fixed allocations in sensitivities.
            if model == "BuyHold":
                used_gamma = kwargs.get("gamma", 5.)
                values = 2*used_gamma*v*100
                signal = pd.DataFrame(np.repeat(values.to_numpy()[:, None], 4, axis=1),
                                      index=evaluation_dates, columns=columns)
            elif model == "Cash":
                signal = pd.DataFrame(-1., index=evaluation_dates, columns=columns)
            else:
                signal = mu.reindex(evaluation_dates-delay).copy()
                signal.index = evaluation_dates
            eh = wide["EH"].reindex(evaluation_dates-delay).copy()
            eh.index = evaluation_dates
            path = allocation_path(signal, eh, a, f, v, cut, **config, **kwargs)
            for period, (begin, end) in PERIODS.items():
                stats = _performance(_period_path(path, begin, end), model, period)
                stats.update(scenario=scenario, allocation_gamma=kwargs.get("gamma", 5.),
                             evaluation_gamma=5., cost_bps=kwargs.get("cost_bps", .5), delay=delay)
                stress_rows.append(stats)
    trades = pd.concat([path.loc["2017-01":].assign(model=model)
                        for model, path in paths.items()], ignore_index=True)
    performance, utility = pd.DataFrame(summary), pd.DataFrame(inference)
    stress, calibration = pd.DataFrame(stress_rows), pd.concat(candidate_tables, ignore_index=True)
    protocol = {"model_parameters_frozen_before": "2014-01",
                "investment_validation": LATE_CALIBRATION,
                "last_validation_purchase": "2016-11", "last_validation_outcome": "2016-12",
                "first_test_purchase": "2017-01", "strict_maturity": "purchase+1 < decision",
                "risk_window": 60, "risk_minimum_rows": 36,
                "gamma": 5., "cap": 1., "cost_bps_each_way": .5,
                "parameters": parameters, "prior_ew3_source": source,
                "allocation_parameters_inherited_from": inheritance_source,
                "allocation_calibration_candidates": str((folder/"fullcalibration.csv").resolve()),
                "periods_purchase_months": PERIODS,
                "utility_inference": "gamma5 total monthly CER; paired 12-month circular blocks; "
                                     "999 percentile draws; conditional forecasts; no search correction"}
    (folder/"allocation_parameters.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    for name, table in (("fullcalibration", calibration), ("economic_trades", trades),
                        ("performance", performance), ("utility", utility), ("stress", stress)):
        table.to_csv(folder/f"{name}.csv", index=False)
    return {"parameters": protocol, "calibration": calibration, "trades": trades,
            "performance": performance, "utility": utility, "stress": stress}
