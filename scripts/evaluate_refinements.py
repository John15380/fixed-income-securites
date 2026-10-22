"""Evaluate the bounded refinement menu; never replace the declared primary."""

import ast
import hashlib
import json
from pathlib import Path
import re
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from bond_rp.config import DATA_DIR
from bond_rp.execution import cost_aware_forecasts
from bond_rp.data import load_bond_data, load_gsw_bond_data, load_macro_raw_transformed
from bond_rp.inference import joint_block_bootstrap_t
from bond_rp.metrics import annual_performance, loss_test, oos_r2
from bond_rp.refinement import (BASE_MODELS, PRIMARY, causal_shrinkage, fit_refinements,
                                goh_signals, literal_sentiment)
from bond_rp.coursework import reference_features
from bond_rp.strategy import annual_backtest
from bond_rp.trading import load_monthly_market, monthly_backtest, monthly_performance
from bond_rp.vintages import VintageArchive
from export_research_audit import md


CHOSEN = [PRIMARY, "Core/blend", "Core/half", "Market/online", "SPCA/online", "Goh48/online", "LN6_safe/online",
          "PriorEW3/w120", "PriorEW3_R100/w120"]


def load_predictions(folder):
    frame = pd.read_csv(folder/"combined_predictions.csv")
    frame = frame[frame.model.str.rsplit("/", n=1).str[0].isin(BASE_MODELS)].copy()
    if folder.name in {"gsw_monthly", "matched_monthly", "vintage_monthly"}:
        source = "vintage" if folder.name == "vintage_monthly" else "revised"
        path = folder.parent/"prior_candidates"/source/"comparison_predictions.csv"
        if path.exists():
            from refine_strategies import fingerprint
            from bond_rp.vintages import VINTAGE_URLS
            files = ["GSW_yields.csv", "2026-04-MD.csv", "F-F_Research_Data_Factors_CSV.zip"]
            if source == "vintage":
                files += list(VINTAGE_URLS)
            settings = json.loads((path.parent/"settings.json").read_text())
            if (path.parent/"cache_key.txt").read_text() != fingerprint(files, settings):
                raise ValueError("Stale previously retained candidate cache")
            previous = pd.read_csv(path)
            previous = previous[previous.ym.isin(frame.ym)]
            if any(set(g.ym) != set(frame.ym) for _, g in previous.groupby("model")):
                raise ValueError("Previously retained candidates need the same complete calendar")
            frame = pd.concat([frame, previous], ignore_index=True)
    return frame


def wide_predictions(p):
    q = p.assign(average=p[[f"mu{n}" for n in range(2, 6)]].mean(axis=1))
    q = q.pivot(index="ym", columns="model", values="average")
    q.index = pd.PeriodIndex(q.index, freq="M")
    return q


def initial_in_sample(folder, bonds, settings, start):
    date = pd.Period(start, freq="M")
    horizon = settings["horizon"]
    source = "jln" if folder.name.startswith("fb") else "fredmd"
    macro = load_macro_raw_transformed(source)
    if folder.name.startswith("vintage"):
        with VintageArchive(DATA_DIR) as archive:
            _, macro = archive.at(date)
    features = reference_features(bonds, macro).join(goh_signals(bonds))
    mc = [c for c in features if c.startswith("macro::")]
    prefix = "mx" if horizon == 1 else ("sx" if settings["kind"] == "simple" else "rx")
    targets = [f"{prefix}{n}" for n in range(2, 6)]
    required = [f"f{n}" for n in range(1, 6)]+[c for c in features if c.startswith("goh::")]
    data = features.join(bonds[targets])
    mask = (data.index >= "1964-01") & (data.index+horizon < date)
    mask &= data[required+targets].notna().all(axis=1) & data[mc].notna().any(axis=1)
    train = data.loc[mask]
    history = features.loc["1964-01":date]
    history = history.loc[history[mc].notna().any(axis=1)]
    fitted, _ = fit_refinements(train, train, history, mc, targets)
    y = train[targets].mean(axis=1)
    rows = [{"model": name, "train_n": len(train), "start": str(train.index.min()), "end": str(train.index.max()),
             "last_label_end": str(train.index.max()+horizon), "decision_month": str(date),
             "r2_pct": 100*oos_r2(y, value.mean(axis=1), y.mean())} for name, value in fitted.items()]
    pd.DataFrame(rows).to_csv(folder/"in_sample.csv", index=False)


def sentiment_diagnostic(folder, bonds, settings):
    p = pd.read_csv(folder/"predictions.csv")
    q = wide_predictions(p)
    if "BW3/exp" not in q:
        return
    actual = bonds.mx_bar if settings["horizon"] == 1 else bonds["sx_bar" if settings["kind"] == "simple" else "rx_bar"]
    rows = []
    for window in ("exp", "w120", "w240"):
        names = [f"{name}/{window}" for name in ("BW3", "EH_BW", "Core", "Core_BW")]
        a = q[names].join(actual.rename("actual")).loc["1990-01":].dropna()
        rows.append({"window": window, "n": len(a), "start": str(a.index.min()), "end": str(a.index.max()),
                     "bw3_r2_vs_same_history_eh_pct": 100*oos_r2(a.actual, a[names[0]], a[names[1]]),
                     "core_bw_incremental_r2_vs_core_pct": 100*oos_r2(a.actual, a[names[3]], a[names[2]])})
    pd.DataFrame(rows).to_csv(folder/"sentiment_diagnostic.csv", index=False)


def forecast_evidence(folder, bonds, settings, protocol):
    p = load_predictions(folder)
    q = wide_predictions(p)
    h, kind = settings["horizon"], settings["kind"]
    actual = bonds.mx_bar if h == 1 else bonds["sx_bar" if kind == "simple" else "rx_bar"]
    start = "2000-01" if folder.name.startswith(("vintage", "matched")) else "1990-01"
    a = q.join(actual.rename("actual")).loc[start:].dropna()
    if not (np.diff(a.index.asi8) == 1).all():
        raise ValueError("Forecast evidence needs a contiguous common calendar")
    reference = a["EH/blend"].to_numpy()
    y = a.actual.to_numpy()
    names = [c for c in q if c not in {"EH/blend", "EH/half", "EH/online"}]
    differences = np.column_stack([(y-reference)**2-(y-a[c].to_numpy())**2 for c in names])
    rows, origins = [], []
    for name in names:
        pred = a[name].to_numpy()
        suffix = name.split("/")[-1]
        own = a["EH/"+suffix].to_numpy() if suffix in {"exp", "w120", "w240"} else reference
        statistic, raw_p = loss_test(y, pred, reference, lags=18 if h == 12 else 6)
        rows.append({"model": name, "n": len(a), "start": str(a.index.min()), "end": str(a.index.max()),
                     "r2_pct": 100*oos_r2(y, pred, reference), "same_window_r2_pct": 100*oos_r2(y, pred, own),
                     "rmse_pp": np.sqrt(np.mean((y-pred)**2)), "hac_stat": statistic, "hac_p": raw_p})
    scores = pd.DataFrame(rows).set_index("model")
    scores["holm_p"] = multipletests(scores.hac_p, method="holm")[1]
    for block in protocol["bootstrap_blocks"]:
        test = joint_block_bootstrap_t(differences, lags=18 if h == 12 else 6,
                                       block=block, replications=protocol["bootstrap_replications"])
        scores[f"bootstrap_p{block}"] = test["bootstrap_p"]
        scores[f"max_t_p{block}"] = test["max_t_p"]
    scores.to_csv(folder/"forecast_summary.csv")
    incremental = []
    for model, benchmark in [("Core/blend", "Market/blend"), (PRIMARY, "Market/online"),
                             ("LN6_safe/blend", "LN8_raw/blend"), ("SPCA/blend", "LN8_raw/blend")]:
        _, value = loss_test(y, a[model], a[benchmark], lags=18 if h == 12 else 6)
        incremental.append({"model": model, "benchmark": benchmark,
                            "r2_pct": 100*oos_r2(y, a[model], a[benchmark]), "hac_p": value})
    incremental = pd.DataFrame(incremental)
    incremental["holm_p"] = multipletests(incremental.hac_p, method="holm")[1]
    incremental.to_csv(folder/"incremental_models.csv", index=False)
    for origin in protocol["origins"]:
        if folder.name.startswith(("vintage", "matched")) and origin < "2000-01":
            continue
        d = q.join(actual.rename("actual")).loc[origin:].dropna()
        for name in names:
            _, value = loss_test(d.actual, d[name], d["EH/blend"], lags=18 if h == 12 else 6)
            origins.append({"model": name, "origin": origin, "n": len(d), "end": str(d.index.max()),
                            "r2_pct": 100*oos_r2(d.actual, d[name], d["EH/blend"]), "hac_p": value})
    origins = pd.DataFrame(origins)
    origins["holm_p"] = multipletests(origins.hac_p, method="holm")[1]
    origins.to_csv(folder/"origin_forecast_summary.csv", index=False)
    return p, q, start


def portfolio_uncertainty(folder, trades, horizon, replications):
    r = trades.pivot(index="purchase", columns="model", values="return_net").sort_index()
    rf = trades.groupby("purchase").risk_free.first().reindex(r.index).to_numpy()
    passive, eh = r.BuyHold.to_numpy(), r["EH/blend"].to_numpy()
    rows = []
    for name in r:
        x, z = r[name].to_numpy()-rf, passive-rf
        fit = sm.OLS(x, sm.add_constant(z)).fit(cov_type="HAC", cov_kwds={"maxlags": 12 if horizon == 1 else 2, "use_correction": True})
        rows.append({"model": name, "alpha_pct": (1200 if horizon == 1 else 100)*fit.params[0],
                     "beta": fit.params[1], "alpha_p": float(norm.sf(fit.tvalues[0])) if name != "BuyHold" else 1.})
    alpha = pd.DataFrame(rows)
    alpha["holm_p"] = multipletests(alpha.alpha_p.fillna(1), method="holm")[1]
    alpha.to_csv(folder/"alpha_diagnostics.csv", index=False)
    # Full realized-return family. Mean gain is not alpha: keep beta above.
    names = [c for c in r if c not in {"BuyHold", "Cash", "EH/blend", "EH/half", "EH/online"}]
    mean_tests = []
    for benchmark, series in [("EH/blend", eh), ("BuyHold", passive)]:
        differences = r[names].to_numpy()-series[:, None]
        test = joint_block_bootstrap_t(differences, lags=6 if horizon == 1 else 2,
                                       block=12 if horizon == 1 else 3, replications=replications)
        for j, name in enumerate(names):
            mean_tests.append({"model": name, "benchmark": benchmark,
                               "mean_gain_bps_annual": (120000 if horizon == 1 else 10000)*differences[:, j].mean(),
                               "bootstrap_p": test["bootstrap_p"][j], "max_t_p": test["max_t_p"][j]})
    pd.DataFrame(mean_tests).to_csv(folder/"return_gain_tests.csv", index=False)
    # Annual CER: aggregate actual one-month returns before computing utility.
    if horizon == 1:
        n = len(r)//12
        returns = np.prod(1+r.to_numpy()[:12*n].reshape(n, 12, len(r.columns)), axis=1)-1
        cash = np.prod(1+rf[:12*n].reshape(n, 12), axis=1)-1
    else:
        returns, cash = r.to_numpy(), rf
    n = len(returns)
    stats, intervals = [], []
    def cer(v):
        return v.mean(axis=1)-2.5*v.var(axis=1, ddof=1)
    rng = np.random.default_rng(42)
    begin = rng.integers(0, n, (replications, int(np.ceil(n/3))))
    ix = ((begin[:, :, None]+np.arange(3)) % n).reshape(replications, -1)[:, :n]
    eh_j, passive_j = r.columns.get_loc("EH/blend"), r.columns.get_loc("BuyHold")
    point = returns.mean(axis=0)-2.5*returns.var(axis=0, ddof=1)
    for j, name in enumerate(r.columns):
        stat = annual_performance(returns[:, j], cash, gamma=5)
        stats.append({"model": name, **stat, "delta_cer_eh_bps": 10000*(point[j]-point[eh_j]),
                      "delta_cer_passive_bps": 10000*(point[j]-point[passive_j])})
        if name in CHOSEN:
            samples = returns[:, [j, eh_j, passive_j]][ix]
            values = cer(samples)
            for k, benchmark in [(1, "EH/blend"), (2, "BuyHold")]:
                gain = 10000*(values[:, 0]-values[:, k])
                intervals.append({"model": name, "benchmark": benchmark, "years": n,
                                  "delta_cer_bps": 10000*(point[j]-point[r.columns.get_loc(benchmark)]),
                                  "ci_low": np.quantile(gain, .025), "ci_high": np.quantile(gain, .975),
                                  "block_years": 3, "replications": replications})
    pd.DataFrame(stats).to_csv(folder/"annual_cer.csv", index=False)
    pd.DataFrame(intervals).to_csv(folder/"cer_intervals.csv", index=False)


def trading_evidence(folder, bonds, p, q, start, horizon, protocol):
    chosen = p[p.model.isin(CHOSEN+[f"EH/{w}" for w in ("exp", "w120", "w240", "blend", "half", "online")])]
    if horizon == 1:
        summary, trades = monthly_backtest(bonds, p, start=start, policy="long_only", cap=1, shrink=1)
    else:
        aq = q.loc[start:].copy();aq["EH"] = aq["EH/blend"]
        summary, trades = annual_backtest(bonds, aq, method="long_only", cap=1)
        summary = summary.reset_index()
        summary, trades = summary[summary.model != "EH"], trades[trades.model != "EH"]
    summary.to_csv(folder/"policy_summary.csv", index=False)
    trades.to_csv(folder/"trades.csv", index=False)
    portfolio_uncertainty(folder, trades, horizon, protocol["bootstrap_replications"])
    sensitivities = []
    for origin in protocol["origins"]:
        if folder.name.startswith(("vintage", "matched")) and origin < "2000-01":
            continue
        if horizon == 1:
            s, _ = monthly_backtest(bonds, chosen, start=origin, shrink=1)
        else:
            a = wide_predictions(chosen).loc[origin:];a["EH"] = a["EH/blend"]
            s, _ = annual_backtest(bonds, a, method="long_only", cap=1);s = s.reset_index()
        sensitivities.append(s.assign(check="origin", value=origin))
    scenarios = [dict(cost_bps=c) for c in (2, 5)]+[dict(gamma=g) for g in (2, 10)]
    if horizon == 1:
        scenarios += [dict(delay=1), dict(risk_window=120)]
    for scenario in scenarios:
        if horizon == 1:
            s, _ = monthly_backtest(bonds, chosen, start=start, shrink=1, **scenario)
        else:
            a = wide_predictions(chosen).loc[start:];a["EH"] = a["EH/blend"]
            s, _ = annual_backtest(bonds, a, method="long_only", cap=1, **scenario);s = s.reset_index()
        sensitivities.append(s.assign(check="scenario", value=json.dumps(scenario, sort_keys=True)))
    # A strategy starting later is different from reporting a later subperiod.
    base = pd.read_csv(folder/"predictions.csv")
    actual = bonds.mx_bar if horizon == 1 else bonds.sx_bar
    for origin in protocol["origins"]:
        if folder.name.startswith(("vintage", "matched")) and origin < "2000-01":
            continue
        cold, audit = causal_shrinkage(base.loc[base.ym >= origin], actual, horizon=horizon)
        cold = cold[cold.model.isin([PRIMARY, "EH/online", "EH/blend"])]
        if horizon == 1:
            s, _ = monthly_backtest(bonds, cold, start=origin, shrink=1)
        else:
            a = wide_predictions(cold).loc[origin:];a["EH"] = a["EH/blend"]
            s, _ = annual_backtest(bonds, a, method="long_only", cap=1);s = s.reset_index()
        sensitivities.append(s.assign(check="cold_start", value=origin))
        audit[audit.model == "Core"].to_csv(folder/f"cold_start_{origin[:4]}.csv", index=False)
    pd.concat(sensitivities, ignore_index=True).to_csv(folder/"sensitivity.csv", index=False)
    cost_aware_rows = []
    for cost in (.5, 2, 5):
        adjusted = cost_aware_forecasts(bonds, chosen, horizon=horizon, cost_bps=cost)
        if horizon == 1:
            stat, _ = monthly_backtest(bonds, adjusted, start=start, shrink=1, cost_bps=cost)
        else:
            a = wide_predictions(adjusted).loc[start:];a["EH"] = a["EH/blend"]
            stat, _ = annual_backtest(bonds, a, method="long_only", cap=1, cost_bps=cost)
            stat = stat.reset_index()
        cost_aware_rows.append(stat.assign(cost_bps=cost))
    pd.concat(cost_aware_rows, ignore_index=True).to_csv(folder/"cost_aware_sensitivity.csv", index=False)
    periods = []
    for lo, hi in protocol["disjoint_subperiods"]:
        for name, g in trades.groupby("model"):
            d = g[(g.purchase.astype(str) >= lo) & (g.purchase.astype(str) <= hi)]
            if len(d) < (24 if horizon == 1 else 3):
                continue
            stat = (monthly_performance(d.return_net, d.risk_free) if horizon == 1 else
                    annual_performance(d.return_net, d.risk_free))
            periods.append({"model": name,
                            "period": f"{d.purchase.astype(str).min()}/{d.purchase.astype(str).max()}",
                            "requested_period": f"{lo}/{hi}", **stat})
    pd.DataFrame(periods).to_csv(folder/"subperiods.csv", index=False)


def evaluate(output, protocol):
    from refine_strategies import fingerprint
    fb, gsw, monthly = load_bond_data().set_index("ym"), load_gsw_bond_data().set_index("ym"), load_monthly_market()
    for folder in sorted(output.iterdir()):
        if not folder.is_dir() or not (folder/"settings.json").exists():
            continue
        settings = json.loads((folder/"settings.json").read_text())
        original = folder
        fit_settings = settings
        if folder.name.startswith("matched"):
            source = "gsw_monthly" if settings["horizon"] == 1 else f"gsw_annual_{settings['kind']}"
            original = output/source
            fit_settings = json.loads((original/"settings.json").read_text())
        files = (["Fama_Bliss_price.txt", "macro_raw_data.xlsx"] if original.name.startswith("fb") else
                 ["GSW_yields.csv", "2026-04-MD.csv"])
        if settings["horizon"] == 1:
            files.append("F-F_Research_Data_Factors_CSV.zip")
        if original.name.startswith("vintage"):
            from bond_rp.vintages import VINTAGE_URLS
            files += list(VINTAGE_URLS)
        else:
            files.append("SENTIMENT.xlsx")
        if (original/"cache_key.txt").read_text() != fingerprint(files, fit_settings):
            raise ValueError(f"Stale forecast cache at {original}; rerun --stage forecast")
        bonds = fb if folder.name.startswith("fb") else (monthly if settings["horizon"] == 1 else gsw)
        print(f"Evaluating refinements: {folder.name}", flush=True)
        p, q, start = forecast_evidence(folder, bonds, settings, protocol)
        initial_in_sample(folder, bonds, settings, start)
        sentiment_diagnostic(folder, bonds, settings)
        if settings["kind"] == "simple":
            trading_evidence(folder, bonds, p, q, start, settings["horizon"], protocol)
        print(f"Finished refinement evidence: {folder.name}", flush=True)
    stress_rows = []
    for folder in sorted((output/"iqr_stress").glob("*/*")):
        if not (folder/"settings.json").exists():
            continue
        settings = json.loads((folder/"settings.json").read_text())
        files = ["GSW_yields.csv", "2026-04-MD.csv", "F-F_Research_Data_Factors_CSV.zip"]
        if folder.name.startswith("vintage"):
            from bond_rp.vintages import VINTAGE_URLS
            files += list(VINTAGE_URLS)
        else:
            files.append("SENTIMENT.xlsx")
        if (folder/"cache_key.txt").read_text() != fingerprint(files, settings):
            raise ValueError(f"Stale IQR stress cache at {folder}")
        p = load_predictions(folder)
        selected = p[p.model.isin([PRIMARY, "EH/online", "EH/blend"])]
        start = "2000-01" if folder.name.startswith("vintage") else "1990-01"
        stat, _ = monthly_backtest(monthly, selected, start=start, shrink=1)
        q = wide_predictions(p).join(monthly.mx_bar.rename("actual")).loc[start:].dropna()
        for _, r in stat.iterrows():
            stress_rows.append({"panel": folder.name, "bound": settings["iqr_bound"], **r.to_dict(),
                                "r2_pct": 100*oos_r2(q.actual, q[r.model], q["EH/blend"]) if r.model in q else np.nan})
    if stress_rows:
        pd.DataFrame(stress_rows).to_csv(output/"iqr_sensitivity.csv", index=False)
    sources = [*sorted((ROOT/"bond_rp").glob("*.py")), Path(__file__), ROOT/"scripts"/"refine_strategies.py"]
    (output/"metadata.json").write_text(json.dumps({
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(), "protocol": protocol,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "forecast_sha256": {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in output.rglob("combined_predictions.csv")},
        "prior_candidate_sha256": {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in (output/"prior_candidates").rglob("comparison_predictions.csv")}
    }, indent=2))


def plot(output):
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".cache"/"matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8), layout="constrained")
    colors = {PRIMARY: "#228833", "EH/blend": "#4477aa", "BuyHold": "#777777", "PriorEW3/w120": "#cc6677"}
    for ax, tag in zip(axes[0], ("gsw_monthly", "vintage_monthly")):
        trades = pd.read_csv(output/tag/"trades.csv")
        for name, color in colors.items():
            g = trades[trades.model == name].sort_values("purchase")
            dates = pd.PeriodIndex([g.purchase.iloc[0], *g.sale], freq="M").to_timestamp()
            ax.plot(dates, np.r_[1, np.cumprod(1+g.return_net)], color=color, label=name)
        ax.set(title="Revised macro" if tag == "gsw_monthly" else "Historical macro vintages", ylabel="Net wealth (initial NAV = 1)")
        ax.legend(fontsize=8)
    f = pd.read_csv(output/"gsw_annual_log"/"forecast_summary.csv").set_index("model")
    labels = ["Raw LN6", "Macro cap only", "Cubic cap only", "Both protections", "sPCA", "Goh48"]
    models = ["LN6_raw/blend", "LN6_macro_clip/blend", "LN6_cube_clip/blend", "LN6_safe/blend", "SPCA/blend", "Goh48/blend"]
    axes[1, 0].barh(labels, f.loc[models, "rmse_pp"], color="#4477aa")
    axes[1, 0].set_xscale("log")
    axes[1, 0].invert_yaxis()
    axes[1, 0].set(title="Annual prediction: extrapolation ablations", xlabel="Log-return RMSE (pp, log axis)")
    for tag, color, label in [("gsw_monthly", "#4477aa", "Revised"), ("vintage_monthly", "#cc6677", "Vintages")]:
        a = pd.read_csv(output/tag/"shrinkage_audit.csv")
        a = a[(a.model == "Core") & (a.ym >= "1990-01")]
        a["year"] = a.ym.str[:4].astype(int)
        mean = a.groupby("year")["lambda"].mean()
        axes[1, 1].plot(mean.index, mean, label=label, color=color)
    axes[1, 1].set(title="Causal shrinkage (annual averages for display)", ylabel="Lambda: 0 = EH, 1 = full Core", ylim=(-.02, 1.02))
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Declared Core and previously retained EW3: comparable information, costs and realized returns", fontsize=12)
    fig.savefig(ROOT/"assets"/"refinement_study.png", dpi=160)
    plt.close(fig)


def export(output, protocol):
    def table(tag, name, index="model"):
        return pd.read_csv(output/tag/(name+".csv")).set_index(index)
    def pv(value):
        return "<0.001" if value < .001 else f"{value:.3f}"
    text = [r"""<a id="refinement-study"></a>

### 5.4 文献驱动的策略改进：外推保护、sPCA 与递归收缩

本轮继续推进策略与结果，不制作 Word 报告。原文基准及其失败结果保留；本节是明确标注的后续方法开发，仍然已经接触历史样本，不能称为预注册或真正封存的测试。

**先限定策略，再检验。** 主规则固定为 `Core/online`，不改用本轮排名最高的模型。曲线、宏观、技术三类信息各占 1/3；宏观内部 LN6_safe 与 sPCA 各占一半。扩展、120 月、240 月三个估计窗口等权，不选事后最优窗口。SENT 缺少完整历史版本，因此不进入主规则，也不再让其 2023 年截止日截短其他因子的全样本。

| 构造 | 原文依据与本轮实现 | 保留的边界 |
| --- | --- | --- |
| CP | 保留平均收益提取指数、再估计期限暴露的两阶段结构 | 仍受曲线测量误差和参数不稳定影响 |
| Goh48 | 按 [Goh 等](https://ssrn.com/abstract=1914227) §II.A 构造 2–5 年远期利差的 48 个短/长均线交叉；[Bai–Ng](https://onlinelibrary.wiley.com/doi/10.1111/1468-0262.00273) ICp2 从 0–8 中决定因子数，再按各期限 BIC 选择子集，允许常数模型。PCA 可使用当时已知的指标，BIC 只用成熟收益 | 已实现远期利差部分，未捏造成交量因子；上限 8、数据版本和滚动窗口均为本项目配置，不宣称原文逐表精确复现 |
| sPCA | 按本地 [Huang 等 (2020)](https://ssrn.com/abstract=3107612) §3.2，以成熟训练样本中各标准化宏观变量对平均收益的单变量预测斜率缩放，再提取第一主成分。缩放后不再次逐列标准化，避免抹掉权重；另保留 SPCA_raw 对照 | 使用现有 FRED-MD/JLN 面板，并非原文 60 个 ALFRED 系列；监督缩放仍有估计误差，OOS 优势要检验 |
| LN6_safe | 保留原六项。宏观输入的保护界限为已知历史中位数 ±6 IQR，零 IQR 列不强制截断；再将三次项的输入限制在成熟训练样本 F1 的范围内，线性项保留。另保留只处理宏观、只处理三次项和完全不处理的消融对照 | 这是本项目的外推保护，不伪称原文精确异常值规则；不删除疫情月份，不截断实现收益。LN6_safe_R100 是直接逐期限 Ridge 扩展，避免第二阶段斜率抵消收缩 |
| BW3 诊断 | 另外按 Laborda–Olmo 式 (14) 使用 SENT_ORTH、平方与差分构建两阶段指数；保留 EH_BW 的同训练历史基准 | 下载文件的正交化和历史发布版本未重建，仅为修订数据诊断；不能用它为主规则的实时性背书 |

设三个窗口混合后的 Core 预测为 m，历史均值混合预测为 b，d=m−b。每个决策时点仅使用过去已兑现的、当时实际生成的预测，估计：

$$\lambda_t=\operatorname{clip}\left(\frac{\sum_s d_s(y_s-b_s)}{\sum_s d_s^2},0,1\right),\quad \widehat y_t=b_t+\lambda_t(m_t-b_t),\quad s+h<t.$$

使用最近 120 个买入月份，至少 60 个成熟误差；不足时 λ=0。权重、成熟误差截止日和特征处理均留档。监督回归与 sPCA 缩放只使用成熟标签，无监督 PCA 可以使用当时已公开的观测；两者的信息集分别检查。三个窗口混合后的固定 λ=0.5 和不收缩版本也保留，不能把学习后的 λ 当成先验最佳参数。

**经济评价固定约束。** 主回测为仅多头、最高 100% 风险资产，γ=5；每笔买卖单边 0.5 bps，入场费用预留现金。每个期限的真实一月老化价格用于月度回测；一年期策略只复利不重叠的一年持仓。主规则相对同窗口混合 EH、被动和现金比较，预测回测不再额外乘一次 50% 收缩。另完整检查 γ=2/10、2/5 bps、一月延迟、风险窗口、四个起点及冷启动，结果不用于改换主规则。
"""]
    rows = []
    for tag, label in [("fb_annual_log", "FB/JLN"), ("gsw_annual_log", "GSW 修订"), ("vintage_annual_log", "GSW 历史版本")]:
        f = table(tag, "forecast_summary")
        for name in ["CP/blend", "LN6_raw/blend", "LN6_safe/blend", "SPCA/blend", "Goh48/blend", "Core/blend", PRIMARY]:
            r = f.loc[name]
            rows.append([label, name, int(r.n), r.r2_pct, pv(r.hac_p), pv(r.max_t_p24)])
    text.append("**一年期预测结果。** 本表为对数篮子超额收益，基准是三个窗口等权的 EH/blend；与此前单一扩展 EH 的分母不同。修订版主评价从 1990 年起，历史版本从 2000 年起，各自使用已兑现标签的完整结束月；这些列不是同期间的纯修订效应估计。")
    text.append(md(["面板", "规则", "月数", "OOS R² (%)", "原始损失 HAC p", "联合 max-t p（24月）"], rows))
    rows = []
    for tag, label in [("gsw_annual_log", "一年对数"), ("gsw_monthly", "一月简单")]:
        initial, forecast = table(tag, "in_sample"), table(tag, "forecast_summary")
        for name in ("CP", "LN6_safe", "SPCA", "Goh48", "Core"):
            a, b = initial.loc[name], forecast.loc[name+"/exp"]
            rows.append([label, name+"/exp", int(a.train_n), a.r2_pct, b.r2_pct, b.same_window_r2_pct])
    text.append("**新方法也补齐样本内证据。** 下表 IS 是 1990 年首次评价时实际可用标签的拟合 R²，PCA 只用当时可见的宏观信息；它是描述性拟合，不参与此后的收缩系数学习。sPCA 的变量权重和 BIC 选择均依赖训练目标，不能把选出的低维因子当作事先给定的一两个自由度来宣称样本内显著。OOS 同时报告相对混合 EH 和同为扩展窗口 EH 的 R²，避免分母造成误读。")
    text.append(md(["目标", "规则", "IS 标签数", "IS R² (%)", "OOS/混合EH (%)", "OOS/扩展EH (%)"], rows))
    rows = []
    for tag, label in [("gsw_annual_log", "一年对数"), ("gsw_monthly", "一月简单")]:
        s = pd.read_csv(output/tag/"sentiment_diagnostic.csv")
        a = s[s.window == "exp"].iloc[0]
        rows.append([label, f"{a.start}–{a.end}", int(a.n), a.bw3_r2_vs_same_history_eh_pct, a.core_bw_incremental_r2_vs_core_pct])
    text.append("原文情绪输入的单独诊断如下，使用扩展窗口和当期可用的修订文件。BW3 相对自身同训练历史 EH；加入情绪相对 Core 的增量 R² 直接检查它是否降低预测误差。该小表不属于主规则的宏观历史版本确认，也不据此选择是否纳入主组合。")
    text.append(md(["目标", "共同买入期", "月数", "BW3/EH R² (%)", "Core+BW 相对 Core R² (%)"], rows))
    rows = []
    for tag, label in [("gsw_monthly", "修订全期"), ("matched_monthly", "修订同起步"), ("vintage_monthly", "历史版本")]:
        s = table(tag, "policy_summary")
        for name in ["EH/blend", "BuyHold", "PriorEW3/w120", "PriorEW3_R100/w120", "Market/online", "SPCA/online", "Goh48/online", "Core/half", PRIMARY]:
            r = s.loc[name]
            rows.append([label, name, int(r.months), r.cagr_pct, r.sharpe, r.max_dd_pct, 100*r.mean_gross_exposure])
    text.append("**一月期策略结果。** 年化夏普由实际非重叠月收益计算，回撤含月末路径。‘修订同起步’与历史版本均从 1999-09 开始积累预测误差、2000-01 开始投资；它消除收缩学习器启动时间不同造成的比较偏差。变量构成、定义与数值版本仍可能不同，差值不能全部解释为数值修订。")
    text.append("同时重新生成本轮之前已披露的两个 **PriorEW3** 候选：CP、LN8、价格 MA12 三类预测等权，120 月滚动，50% 向同窗口 EH 收缩，分别使用 OLS 与 Ridge100。它们不含情绪，因此可与历史版本完整样本比较；PCA 按其既有的成熟训练行定义重估。它们是先前候选的复核，不是从本轮新模型排名中挑出的替代主规则；两者一并进入本轮月度联合比较。")
    text.append(md(["面板", "规则", "月数", "净 CAGR (%)", "超额夏普", "月末回撤 (%)", "平均敞口 (%)"], rows))
    rows = []
    for tag, label in [("gsw_monthly", "修订全期"), ("matched_monthly", "修订同起步"), ("vintage_monthly", "历史版本")]:
        f, a, c = table(tag, "forecast_summary"), table(tag, "alpha_diagnostics"), table(tag, "annual_cer")
        ci = pd.read_csv(output/tag/"cer_intervals.csv")
        for name in [PRIMARY, "PriorEW3/w120"]:
            interval = ci[(ci.model == name) & (ci.benchmark == "EH/blend")].iloc[0]
            rows.append([label, name, f.loc[name, "r2_pct"], pv(f.loc[name, "max_t_p24"]),
                         c.loc[name, "delta_cer_eh_bps"], f"[{interval.ci_low:.1f}, {interval.ci_high:.1f}]",
                         a.loc[name, "alpha_pct"], pv(a.loc[name, "holm_p"])])
    text.append("**主检验规则与先前候选的预测增量、效用和风险暴露。** ΔCER 将实际月收益复利为完整、不重叠的 12 月持仓区块后，以同一 γ=5 计算，相对 EH；末尾不满一年的月收益不进入此表 CER。95% 区间是 3 年循环区块、配对重抽并重新计算均值/方差的百分位区间，尚未对策略搜索校正。α 是对被动篮子超额收益回归的年化截距，仅控制这一个久期风险代理，不是完整资产定价 alpha。")
    text.append(md(["面板", "规则", "一月 R² (%)", "预测联合 p", "ΔCER/EH (bps)", "ΔCER 95% 区间", "被动控制 α (%)", "α Holm p"], rows))
    rows = []
    for tag, label in [("gsw_monthly", "修订"), ("vintage_monthly", "历史版本")]:
        a = pd.read_csv(output/tag/"incremental_models.csv")
        a = a[(a.model == PRIMARY) & (a.benchmark == "Market/online")].iloc[0]
        rows.append([label, a.r2_pct, pv(a.hac_p), pv(a.holm_p)])
    text.append("三类输入存在相关性，宏观面板本身也含金融变量；不能把它们视为三个独立风险来源。因此另将主规则与只含 CP/Goh 的 Market/online 比较，直接检查加入宏观后的增量，而不只选常数基准。")
    text.append(md(["一月目标", "主规则相对 Market R² (%)", "HAC p", "增量比较 Holm p"], rows))
    rows = []
    s = pd.read_csv(output/"vintage_monthly"/"sensitivity.csv")
    for value in ['{"cost_bps": 2}', '{"cost_bps": 5}', '{"delay": 1}', '{"gamma": 2}', '{"gamma": 10}', '{"risk_window": 120}']:
        a = s[(s.check == "scenario") & (s.value == value)].set_index("model")
        r, b = a.loc[PRIMARY], a.loc["BuyHold"]
        rows.append([value, r.cagr_pct, r.sharpe, b.cagr_pct, b.sharpe])
    text.append("**历史版本主规则的压力测试。** 每个情景都重算策略及基准费用/风险，不把延迟后新增月份混入样本。γ 是偏好参数，不能跨 γ 比较 CER 后挑赢家。")
    text.append(md(["情景", "主规则 CAGR (%)", "主规则夏普", "被动 CAGR (%)", "被动夏普"], rows))
    rows = []
    for tag, label in [("gsw_monthly", "修订"), ("vintage_monthly", "历史版本")]:
        s = pd.read_csv(output/tag/"cost_aware_sensitivity.csv")
        for cost in (.5, 2, 5):
            a = s[s.cost_bps == cost].set_index("model")
            r, e = a.loc[PRIMARY], a.loc["EH/blend"]
            rows.append([label, cost, r.cagr_pct, r.sharpe, e.cagr_pct, e.sharpe])
    text.append(r"""**费用也进入决策的结构性对照。** 上面的主规则先按预测配置，再扣实现费用；它没有假装已最大化扣费效用。若每单位非负篮子敞口的超额收益为 x，单边费率为 c，则扣费后的超额收益为 (1−c)x−2c(1+rf)。新增成本感知版本按该期望净收益及近似净方差 (1−c)²Var(x) 决定仓位，再按真实本金/卖出市值扣费一次；EH 也作相同调整。月度 rf 只用当期已知一年收益率折算作费用预测代理，不把下一月实现 RF 当成决策输入。现金利率不确定性尚未单独建模；该表是结构性敏感性，不据其排名替换已声明主规则。""")
    text.append(md(["面板", "单边费 bps", "成本感知 Core CAGR (%)", "Core 夏普", "成本感知 EH CAGR (%)", "EH 夏普"], rows))
    path = output/"iqr_sensitivity.csv"
    if path.exists():
        q = pd.read_csv(path)
        rows = [[r.panel, r.bound, r.r2_pct, r.cagr_pct, r.sharpe] for r in q[q.model == PRIMARY].itertuples()]
        text.append("宏观输入界限的 6 IQR 是工程保护选择，未声称由论文指定或估计为最优。另用 4/8 IQR 完整重建月度预测、sPCA 和收缩状态作为敏感性检查，不在几个阈值中挑赢家；原始标签/实现收益始终保留。")
        text.append(md(["面板", "IQR 倍数", "主规则 R² (%)", "净 CAGR (%)", "夏普"], rows))
    rows = []
    s = pd.read_csv(output/"gsw_monthly"/"sensitivity.csv")
    for origin in protocol["origins"]:
        warm = s[(s.check == "origin") & (s.value == origin)].set_index("model")
        cold = s[(s.check == "cold_start") & (s.value == origin)].set_index("model")
        rows.append([origin, warm.loc[PRIMARY, "sharpe"], cold.loc[PRIMARY, "sharpe"], warm.loc["BuyHold", "sharpe"]])
    text.append("**第③问进一步区分评价起点与学习器起点。** 主规则先在 1980 年开始生成预测并积累成熟误差，后按不同评价起点截取。冷启动则在每个起点重新累计误差，前 60 个成熟误差不足时回退 EH，不沿用更早测试期的学习状态。")
    text.append(md(["评价起点", "已有历史状态的夏普", "冷启动夏普", "同期间被动夏普"], rows))
    rows = []
    for tag, label in [("gsw_monthly", "修订"), ("vintage_monthly", "历史版本")]:
        d = pd.read_csv(output/tag/"subperiods.csv")
        for period, g in d.groupby("period"):
            g = g.set_index("model")
            a, b, e = g.loc[PRIMARY], g.loc["BuyHold"], g.loc["EH/blend"]
            rows.append([label, period, a.cagr_pct, a.sharpe, b.cagr_pct, b.sharpe, e.sharpe])
    text.append("**互不重叠的时期检查。** 不把多起点和多建仓月当成独立重复证据；下面各时期在时间上不重叠。")
    text.append(md(["面板", "买入时期", "主 CAGR (%)", "主夏普", "被动 CAGR (%)", "被动夏普", "EH 夏普"], rows))
    text.append("![固定主规则、宏观版本、外推消融及递归收缩](assets/refinement_study.png)")
    text.append("**推断范围。** 预测误差使用原始损失差，避免对非线性、BIC、窗口混合及递归收缩机械套 CW。对本节主评价期间全部候选/窗口/收缩规格，按共同日期同步重抽、零假设去均值，并在每次抽样重新估计 HAC 标准误，保存 12/24 月区块的联合 max-t 诊断；默认 1,999 次、种子 42。它检验这个已披露候选集合，仍依赖区块/渐近近似，且条件于已保存的预测，未重新抽取宏观面板并完整重估所有历史模型，也未消除之前研究的样本接触。起点全表另做 Holm，不能称为覆盖所有历史参数搜索的统一校正。收益均值改善的联合检验也保存，不能把它称为风险控制后的 α。")
    v = table("vintage_monthly", "policy_summary")
    ci = pd.read_csv(output/"vintage_monthly"/"cer_intervals.csv")
    d = ci[(ci.model == PRIMARY) & (ci.benchmark == "EH/blend")].iloc[0]
    better = v.loc[PRIMARY, "cagr_pct"] >= v.loc["BuyHold", "cagr_pct"] and v.loc[PRIMARY, "sharpe"] >= v.loc["BuyHold", "sharpe"]
    assessment = "主规则的收益和夏普点估计均高于同期间被动" if better else "主规则未同时在收益和夏普上超过同期间被动"
    text.append(f"**本轮结论。** 在宏观历史版本实验中，{assessment}；ΔCER 相对 EH 的估计为 {d.delta_cer_bps:.1f} bps，95% 区间 [{d.ci_low:.1f}, {d.ci_high:.1f}]。这是该数据和交易假设下的结论，不能自动推广为实盘表现或跨时期稳定优势。外推保护、目标导向因子和成熟误差收缩的作用由消融、原始 OOS 损失和实际账户收益分别判断；任何一项不好看都保留，不把新赢家替换成主规则。月度版本标签仍不等于逐系列精确发布时间，GSW 曲线也未完成历史版本重建；拟合零息债券尚未映射为可成交合约。")
    old = v.loc["PriorEW3/w120"]
    old_ci = ci[(ci.model == "PriorEW3/w120") & (ci.benchmark == "EH/blend")].iloc[0]
    text.append(f"先前已保留的 EW3/120 月 OLS 候选在同一历史版本样本中，净 CAGR={old.cagr_pct:.2f}%、超额夏普={old.sharpe:.2f}；相对 EH 的 ΔCER={old_ci.delta_cer_bps:.1f} bps，未校正搜索的 95% 区间为 [{old_ci.ci_low:.1f}, {old_ci.ci_high:.1f}]。它继续作为策略候选保留；本轮更复杂的 Core 并未因此获得优先地位。应将‘修复数值失稳’‘降低预测误差’‘改善投资效用’分别判断，而不是以模型复杂度或单次最好夏普作结论。")
    text.append("复现：`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/refine_strategies.py --iqr-stress --update-readme`。`--stage forecast` 生成可验证缓存，`--stage evaluate` 仅重算统计/策略；不一致缓存拒绝评分。全部菜单、逐期预测、变换/选择日志、收缩参数、不同起点/冷启动、压力测试、CER 区间与联合检验保存在本地 `output/refinement/`，不上传原始数据。")
    body = "\n\n".join(text)
    (output/"report.md").write_text(body)
    plot(output)
    path = ROOT/"README.md"
    content = path.read_text()
    block = "<!-- BEGIN REFINEMENT_STUDY -->\n"+body+"\n<!-- END REFINEMENT_STUDY -->\n\n"
    if "<!-- BEGIN REFINEMENT_STUDY -->" in content:
        content = re.sub(r"<!-- BEGIN REFINEMENT_STUDY -->.*?<!-- END REFINEMENT_STUDY -->\s*", lambda m: block, content, flags=re.S)
    else:
        content = content.replace("## 6. 已修复", block+"## 6. 已修复")
    path.write_text(content)
