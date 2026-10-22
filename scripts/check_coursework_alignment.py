"""Restore required FB/CP/LN references and audit statistical/economic evidence."""

import argparse
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
from sklearn.linear_model import LinearRegression
from statsmodels.stats.multitest import multipletests
from bond_rp.config import DATA_DIR
from bond_rp.coursework import (NESTED_AVERAGE_OLS, REFERENCE_MODELS, fit_reference,
                                reference_features, reference_forecasts, reference_vintage_forecasts)
from bond_rp.data import load_bond_data, load_gsw_bond_data, load_macro_raw_transformed, load_sentiment
from bond_rp.inference import block_bootstrap_t
from bond_rp.metrics import annual_performance, loss_test, nw_ols, oos_r2
from bond_rp.models import _numeric_pipeline, available_features
from bond_rp.strategy import annual_backtest
from bond_rp.vintages import VINTAGE_URLS, VintageArchive
from export_research_audit import md


def cache_key(files):
    digest = hashlib.sha256(b"required-FB-CP-LN-log-simple-exp-240-v1")
    for p in [ROOT/"bond_rp"/n for n in ("config.py", "data.py", "models.py", "coursework.py", "vintages.py")]+[Path(DATA_DIR)/f for f in files]:
        digest.update(p.name.encode()); digest.update(p.read_bytes())
    import sklearn
    digest.update(f"{np.__version__}|{pd.__version__}|{sklearn.__version__}".encode())
    return digest.hexdigest()


def sample_inference(bonds, macro, folder):
    features = reference_features(bonds, macro)
    mc = [c for c in features if c.startswith("macro::")]
    required = [f"f{n}" for n in range(1, 6)]+[f"fs{n}" for n in range(2, 6)]
    cols = [f"{p}{n}" for p in ("rx", "sx") for n in range(2, 6)]
    train = features.join(bonds[cols]).loc["1964-01":"2003-12"].dropna(subset=[*required, *cols])
    train = train.loc[train[mc].notna().any(axis=1)]
    summaries, first_rows, coefficients, loadings = [], [], [], []
    for kind, prefix in [("log", "rx"), ("simple", "sx")]:
        targets = [f"{prefix}{n}" for n in range(2, 6)]
        predictions, first, pipeline = fit_reference(train, train, mc, targets)
        actual = train[targets].to_numpy()
        for name, pred in predictions.items():
            for j, maturity in enumerate([2, 3, 4, 5, "basket"]):
                y, fitted = (actual[:, j], pred[:, j]) if j < 4 else (actual.mean(axis=1), pred.mean(axis=1))
                summaries.append({"target": kind, "model": name, "maturity": maturity,
                                  "n": len(train), "start": str(train.index.min()), "end": str(train.index.max()),
                                  "r2_pct": 100*(1-np.sum((y-fitted)**2)/np.sum((y-y.mean())**2))})
        for name, entry in first.items():
            X = sm.add_constant(entry["X"], has_constant="add")
            fit = nw_ols(actual.mean(axis=1), X)
            restriction = np.eye(X.shape[1])[1:]
            wald = fit.wald_test(restriction, scalar=True, use_f=False)
            row = {"target": kind, "model": name, "n": len(train), "r2_pct": 100*fit.rsquared,
                   "adjusted_r2_pct": 100*fit.rsquared_adj, "hac_wald": float(wald.statistic),
                   "hac_wald_p": float(wald.pvalue)}
            if name == "CP_LN_raw":
                macro_test = fit.wald_test(np.eye(X.shape[1])[6:], scalar=True, use_f=False)
                row["macro_given_cp_p"] = float(macro_test.pvalue)
            first_rows.append(row)
            for j, term in enumerate(["constant", *entry["names"]]):
                coefficients.append({"target": kind, "model": name, "term": term,
                                     "coefficient": fit.params[j], "hac_se": fit.bse[j], "hac_t": fit.tvalues[j]})
        if kind == "log":
            keep = pipeline.named_steps["columns"].keep_
            names = np.array(mc)[keep]
            pca = pipeline.named_steps["pca"]
            for k, component in enumerate(pca.components_):
                top = np.argsort(np.abs(component))[-5:][::-1]
                loadings.append({"factor": f"F{k+1}", "variance_share_pct": 100*pca.explained_variance_ratio_[k],
                                 "top_absolute_loadings": "; ".join(f"{names[j].removeprefix('macro::')} ({component[j]:+.3f})" for j in top)})
    for name, frame in [("in_sample", pd.DataFrame(summaries)), ("first_stage", pd.DataFrame(first_rows)),
                         ("coefficients", pd.DataFrame(coefficients)), ("pca_loadings", pd.DataFrame(loadings))]:
        frame.to_csv(folder/f"{name}.csv", index=False)


def prediction_inference(bonds, predictions, folder, replications):
    rows = []
    for (kind, window), subset in predictions.groupby(["target", "window"]):
        prefix = "rx" if kind == "log" else "sx"
        by_name = {m: g.set_index("ym") for m, g in subset.groupby("model")}
        for g in by_name.values():
            g.index = pd.PeriodIndex(g.index, freq="M")
        dates = by_name["EH"].loc["1990-01":].index
        if not (np.diff(dates.asi8) == 1).all():
            raise ValueError("Overlapping-label HAC requires a contiguous evaluation calendar")
        for name in REFERENCE_MODELS:
            for maturity in [2, 3, 4, 5, "basket"]:
                pred_cols = [f"pred{n}" for n in range(2, 6)] if maturity == "basket" else [f"pred{maturity}"]
                actual_cols = [f"{prefix}{n}" for n in range(2, 6)] if maturity == "basket" else [f"{prefix}{maturity}"]
                y = bonds.loc[dates, actual_cols].mean(axis=1).to_numpy()
                pred = by_name[name].loc[dates, pred_cols].mean(axis=1).to_numpy()
                benchmark = by_name["EH"].loc[dates, pred_cols].mean(axis=1).to_numpy()
                nested = ((maturity == "basket" and name in NESTED_AVERAGE_OLS)
                          or (maturity != "basket" and name == "FB_n"))
                stat, p = loss_test(y, pred, benchmark, nested=nested)
                _, raw_p = loss_test(y, pred, benchmark)
                d = (y-benchmark)**2-(y-pred)**2
                if nested:
                    d += (benchmark-pred)**2
                row = {"target": kind, "window": window, "model": name, "maturity": maturity,
                       "n": len(dates), "start": str(dates.min()), "end": str(dates.max()),
                       "r2_pct": 100*oos_r2(y, pred, benchmark), "rmse_pct": np.sqrt(np.mean((y-pred)**2)),
                       "test": "CW-HAC" if nested else "loss-HAC", "hac_stat": stat, "hac_p": p, "raw_loss_p": raw_p}
                if maturity == "basket":
                    row["bootstrap_replications"] = replications
                    for block in (12, 24):
                        result = block_bootstrap_t(d, block=block, replications=replications)
                        row[f"bootstrap_t_p{block}"] = result["bootstrap_p"]
                rows.append(row)
    table = pd.DataFrame(rows)
    alternatives = table.model != "EH"
    table.loc[alternatives, "holm_p"] = multipletests(table.loc[alternatives, "hac_p"], method="holm")[1]
    for block in (12, 24):
        valid = alternatives & table[f"bootstrap_t_p{block}"].notna()
        table.loc[valid, f"bootstrap_holm_p{block}"] = multipletests(table.loc[valid, f"bootstrap_t_p{block}"], method="holm")[1]
    table.to_csv(folder/"oos_tests.csv", index=False)


def economic_inference(bonds, predictions, folder, replications):
    p = predictions[(predictions.target == "simple") & (predictions.window == "exp")].copy()
    p["average"] = p[[f"pred{n}" for n in range(2, 6)]].mean(axis=1)
    wide = p.pivot(index="ym", columns="model", values="average")
    wide.index = pd.PeriodIndex(wide.index, freq="M")
    summary, trades = annual_backtest(bonds, wide.loc["1990-01":])
    trades.to_csv(folder/"annual_trades.csv", index=False)
    summary.to_csv(folder/"economic_summary.csv")
    eh = trades[trades.model == "EH"].set_index("purchase")
    rows = []
    for model, group in trades.groupby("model"):
        a, b = group.set_index("purchase").align(eh, join="inner", axis=0)
        d = a.return_net.to_numpy()-b.return_net.to_numpy()
        test = block_bootstrap_t(d, lags=2, block=3, replications=replications)
        rows.append({"model": model, "years": len(d), "delta_mean_return_bps": 10000*d.mean(),
                     "return_gain_bootstrap_t_p": test["bootstrap_p"]})
    pd.DataFrame(rows).to_csv(folder/"economic_mean_tests.csv", index=False)


def reference_sensitivity(bonds, predictions, folder):
    """Re-use causal forecasts for alternative origins; retain every reference."""
    scores, grid, extremes, incremental = [], [], [], []
    for (kind, window), subset in predictions.groupby(["target", "window"]):
        prefix = "rx" if kind == "log" else "sx"
        wide = subset.assign(average=subset[[f"pred{n}" for n in range(2, 6)]].mean(axis=1))
        wide = wide.pivot(index="ym", columns="model", values="average")
        wide.index = pd.PeriodIndex(wide.index, freq="M")
        for start in ("1985-01", "1990-01", "1995-01", "2000-01"):
            frame = wide.loc[start:]
            actual = bonds.loc[frame.index, [f"{prefix}{n}" for n in range(2, 6)]].mean(axis=1)
            for name in REFERENCE_MODELS:
                _, p = loss_test(actual, frame[name], frame.EH, nested=name in NESTED_AVERAGE_OLS)
                scores.append({"target": kind, "window": window, "origin": start, "model": name,
                               "r2_pct": 100*oos_r2(actual, frame[name], frame.EH), "hac_p": p})
        frame = wide.loc["1990-01":]
        actual = bonds.loc[frame.index, [f"{prefix}{n}" for n in range(2, 6)]].mean(axis=1)
        for name in REFERENCE_MODELS:
            err = (actual-frame[name])**2
            for date in err.nlargest(3).index:
                extremes.append({"target": kind, "window": window, "model": name, "ym": str(date),
                                 "actual_pct": actual.loc[date], "forecast_pct": frame.loc[date, name],
                                 "squared_error_share_pct": 100*err.loc[date]/err.sum()})
        for name in ("CP_LN5_2stage", "CP_LN5_direct", "CP_LN6_2stage"):
            _, raw_p = loss_test(actual, frame[name], frame.CP_2stage)
            incremental.append({"target": kind, "window": window, "model": name,
                                "r2_vs_cp_pct": 100*oos_r2(actual, frame[name], frame.CP_2stage),
                                "raw_loss_hac_p": raw_p})
        if kind == "simple" and window == "exp":
            for gamma in (1, 2, 3, 5, 10, 20):
                for cap in (1, 1.5, 2, 3):
                    stat, _ = annual_backtest(bonds, frame, gamma=gamma, cap=cap)
                    stat["gamma"], stat["cap"] = gamma, cap
                    grid.append(stat.reset_index())
    for name, table in [("start_sensitivity", pd.DataFrame(scores)), ("risk_sensitivity", pd.concat(grid)),
                        ("extreme_forecasts", pd.DataFrame(extremes)), ("incremental_cp", pd.DataFrame(incremental))]:
        table.to_csv(folder/f"{name}.csv", index=False)


def reference_r2_decomposition(bonds, macro, predictions, folder):
    features = reference_features(bonds, macro)
    mc = [c for c in features if c.startswith("macro::")]
    date = pd.Period("1990-01", freq="M")
    rows = []
    for kind, prefix in (("log", "rx"), ("simple", "sx")):
        targets = [f"{prefix}{n}" for n in range(2, 6)]
        data = features.join(bonds[targets]).dropna(subset=[*[f"f{n}" for n in range(1, 6)], *targets])
        data = data.loc[data[mc].notna().any(axis=1)]
        train = data.loc[(data.index >= "1964-01") & (data.index+12 < date)]
        history = features.loc["1964-01":date]
        history = history.loc[history[mc].notna().any(axis=1)]
        fits, _, _ = fit_reference(train, train, mc, targets, macro_history=history)
        subset = predictions[(predictions.target == kind) & (predictions.window == "exp")].copy()
        subset["average"] = subset[[f"pred{n}" for n in range(2, 6)]].mean(axis=1)
        wide = subset.pivot(index="ym", columns="model", values="average")
        wide.index = pd.PeriodIndex(wide.index, freq="M")
        wide = wide.loc[date:]
        y = bonds.loc[wide.index, targets].mean(axis=1)
        is_y = train[targets].mean(axis=1)
        for name in REFERENCE_MODELS:
            rows.append({"target": kind, "model": name, "initial_train_n": len(train),
                         "initial_is_r2_pct": 100*oos_r2(is_y, fits[name].mean(axis=1), is_y.mean()),
                         "oos_vs_eh_r2_pct": 100*oos_r2(y, wide[name], wide.EH),
                         "oos_vs_test_mean_r2_pct": 100*oos_r2(y, wide[name], y.mean()),
                         "forecast_mse": np.mean((y-wide[name])**2),
                         "eh_mse": np.mean((y-wide.EH)**2), "test_mean_mse": np.mean((y-y.mean())**2)})
    pd.DataFrame(rows).to_csv(folder/"r2_decomposition.csv", index=False)


def monthly_economic_bridge(output):
    rows = []
    for tag in ("monthly_revised", "monthly_vintage"):
        path = ROOT/"output"/"trading_designs"/tag/"trades_long_only.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        names = ["EH/w120", "BuyHold", "EW_R100/w120", "EW/w120", "LN_R100/w120", "EW/select"]
        stats = {}
        for name in names:
            g = frame[frame.model == name].sort_values("purchase")
            n = len(g)//12
            r = (1+g.return_net.to_numpy()[:12*n]).reshape(n, 12).prod(axis=1)-1
            rf = (1+g.risk_free.to_numpy()[:12*n]).reshape(n, 12).prod(axis=1)-1
            stats[name] = annual_performance(r, rf, gamma=5)
        for name, stat in stats.items():
            rows.append({"panel": tag, "model": name, "years": stat["years"],
                         "annual_cer_pct": stat["cer_pct"],
                         "delta_cer_eh_bps": 100*(stat["cer_pct"]-stats["EH/w120"]["cer_pct"]),
                         "delta_cer_passive_bps": 100*(stat["cer_pct"]-stats["BuyHold"]["cer_pct"])})
    pd.DataFrame(rows).to_csv(output/"monthly_economic_bridge.csv", index=False)


def monthly_sample_inference(output):
    """IS companion to the existing one-month extension, with matching inputs."""
    if not (ROOT/"output"/"trading_designs"/"monthly_revised"/"forecast_summary.csv").exists():
        return
    from bond_rp.trading import load_monthly_market
    bonds = load_monthly_market()
    f = available_features(bonds, load_macro_raw_transformed("fredmd"), load_sentiment())
    mc = [c for c in f if c.startswith("macro::")]
    targets = [f"mx{n}" for n in range(2, 6)]
    market = [f"f{n}" for n in range(1, 6)]+["fs_avg", "TI_avg", "SENT", "SENT_SQ"]
    d = f.join(bonds[targets]).loc["1964-01":"2003-12"].dropna(subset=market+targets)
    d = d.loc[d[mc].notna().any(axis=1)]
    y = d[targets].mean(axis=1).to_numpy()
    designs = {"LN": _numeric_pipeline(pca=True).fit_transform(d[mc])}
    for name, cols in {"CP": [f"f{n}" for n in range(1, 6)], "BW": ["SENT", "SENT_SQ"], "TI": ["TI_avg"]}.items():
        designs[name] = _numeric_pipeline().fit_transform(d[cols])
    fits = {name: LinearRegression().fit(X, y).predict(X) for name, X in designs.items()}
    fits["EW"] = np.mean(list(fits.values()), axis=0)
    rows = [{"model": name, "horizon_months": 1, "target": "simple", "n": len(d),
             "start": str(d.index.min()), "end": str(d.index.max()),
             "r2_pct": 100*oos_r2(y, pred, y.mean())} for name, pred in fits.items()]
    pd.DataFrame(rows).to_csv(output/"monthly_in_sample.csv", index=False)


def plot_coursework(output):
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".cache"/"matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4), layout="constrained")
    names = ["CP_2stage", "LN6_2stage", "LN5_2stage", "LN8_2stage", "CP_LN5_2stage"]
    labels = ["CP", "LN6", "LN5", "LN8", "CP + LN5"]
    oos = pd.read_csv(output/"baseline"/"oos_tests.csv")
    r2 = oos[(oos.target == "log") & (oos.window == "exp") & (oos.maturity == "basket")].set_index("model").loc[names, "r2_pct"]
    axes[0].barh(labels, r2, color=["#4477aa" if x >= 0 else "#cc6677" for x in r2])
    axes[0].axvline(0, color="black", lw=.8)
    axes[0].set(title="FB/JLN: annual log-return prediction", xlabel="OOS R² vs EH (%) | 1990-01 to 2012-02")
    a = pd.read_csv(output/"gsw"/"forecast_audit.csv")
    a = a[(a.window == "exp") & a.ym.between("2018-01", "2021-12")]
    x = pd.PeriodIndex(a.ym, freq="M").to_timestamp()
    upper = np.maximum(a.F1_training_min.abs(), a.F1_training_max.abs())
    axes[1].plot(x, a.F1_current.abs(), color="#cc6677", label="Current |F1|")
    axes[1].plot(x, upper, color="#4477aa", ls="--", label="Training max |F1|")
    axes[1].set(title="GSW/FRED: cubic extrapolation risk", ylabel="Standardized PCA score magnitude")
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    econ = pd.read_csv(output/"gsw"/"economic_summary.csv").set_index("model")
    cer = econ.loc[names, "delta_cer_bps"]
    axes[2].barh(labels, cer, color=["#4477aa" if v >= 0 else "#cc6677" for v in cer])
    axes[2].axvline(0, color="black", lw=.8)
    axes[2].set(title="GSW: constrained annual investment", xlabel="ΔCER vs EH (bps) | gamma=5, cap=3\n1990-01 to 2025-01")
    for ax in (axes[0], axes[2]):
        ax.invert_yaxis()
    fig.suptitle("Literature references: prediction accuracy and investment utility are separate evidence", fontsize=12)
    (ROOT/"assets").mkdir(exist_ok=True)
    fig.savefig(ROOT/"assets"/"coursework_check.png", dpi=160)
    plt.close(fig)


def export_readme(output):
    data = {tag: {p.stem: pd.read_csv(p) for p in (output/tag).glob("*.csv")
                  if p.stem not in {"predictions", "annual_trades"}} for tag in ("baseline", "gsw")}
    text = [r"""<a id="coursework-check"></a>

### 5.3 按课程研究要求补齐文献基准和证据链

本节依据本地 `docs/作业要求.md`、`docs/往届建议.md`，并核对 `GroupProjectReading` 的原文公式。正式作业要求优先于往届的概括性建议；课程材料和论文 PDF 均留在本地，不上传。

**研究主线：** 在 EH 常数预期溢价基准上，比较 FB、CP、LN 对国债一年期超额收益的解释及预测能力；再判断统计改进能否转化为投资效用。月度策略是另一个预测期的扩展，不能用其较高夏普代替一年期预测检验。

**已补回的必做规格。** `FB_n` 对每个期限分别使用本期限远期利差；`CP_2stage` 先由五个远期利率预测平均超额收益，形成单一 CP 指数，再回归各期限收益；`LN6_2stage` 使用 F1、F1³、F2、F3、F4、F8 构建宏观指数，再作第二阶段。LN 原论文在其样本中用 BIC 得到这个集合，课程明确给出此规格；在本项目中将它固定为外部文献基准是合理的，但不意味着它必然适合更新后的宏观面板。此前直接称其为本项目的“事后子集”并替换为 LN8，处理不当，现已恢复。

**原文与扩展的对应关系。** 直接核对本地 `GroupProjectReading` 的核心文献、Goh、Laborda–Olmo 以及实时数据论文，按文中的公式区分：

| 依据 | 原文研究设计与经济解释 | 本项目落实及批判性边界 |
| --- | --- | --- |
| Fama–Bliss (1987)，远期利差回归 | 本期限远期利差预测超额收益；利差兼含预期短端利率与风险补偿 | 必做 `FB_n` 已按期限实现；基础报告的平均利差 `FB` 是篮子简化，二者分别保留。预测关系不能自动识别风险补偿的因果机制 |
| [CP (2005)](https://web.stanford.edu/~piazzesi/cp.pdf)，式 (3)–(4)、表 1 | 用平均收益提取共同回报预测指数，再估计期限暴露 | 已补齐两阶段；帐篷形是实证结果，不能强加。拟合曲线、测量误差和历史时期变化都会影响系数 |
| [LN (2009)](https://www.nber.org/papers/w11703)，pp.5038–5039、5049–5051 | 单独使用六项；与 CP 联合时剔除 F2，使用 F1、F1³、F3、F4、F8 五项；宏观状态可能包含曲线未涵盖的信息 | 新增 `LN5_2stage`、`CP_LN5_2stage`，以及表 3 思路的五项加 CP 直接回归 `CP_LN5_direct`。联合六项和 LN8 均明确为扩展，不冒充原文规格。原文还检验递归 BIC 和完整因子重估 bootstrap，本项目未完整复现这两部分 |
| [Goh 等 (2013)](https://ssrn.com/abstract=1914227)，§II.A、式 (1)–(2) | 2–5 年期远期利差的短/长均线交叉，s∈{3,6,9}、l∈{18,24,30,36}，共 48 个信号，另有成交量指标；机制包括信息处理速度与需求调整 | 当前价格偏离均线 TI 和六周期审计属于自行定义的趋势扩展，**不是 Goh 原文指标复现**；12 期没有文献赋予的唯一性，尚未复现原文 48 信号和成交量因子 |
| [Laborda–Olmo (2014)](https://eprints.soton.ac.uk/357416/1/LabordaOlmoJFMfinalversion.pdf)，p.211、式 (12)–(15) | 先将平均收益投影到正交化情绪、其平方及差分，得到债券情绪预测指数，再与 CP/LN 联合 | 当前 `BW` 是 SENT 水平及平方的简化代理，缺少原文正交化、差分和共同指数限制；不能声称删掉差分是在纠正原论文。原始 SENT 与 SENT_ORTH 的选择及历史版本是研究限制 |
| [Ghysels–Horan–Moench](https://www.newyorkfed.org/medialibrary/media/research/staff_reports/sr581.pdf)，§1；[Huang 等 (2020)](https://ssrn.com/abstract=3107612)，§1–2 | 前者指出发布滞后、修订会放大可预测性；后者发现更有效的目标导向方法可能从实时数据提取信息 | 本地前者 PDF 实为 2014 年修订的 Staff Report 581，不能只按文件名当作最终版本。已有历史 FRED-MD 实验，但严格逐系列实时档案未齐；实时 OLS 较弱不意味着所有因子和方法无效，也不构成新方法必胜的保证 |

PCA、第一阶段指数系数和第二阶段期限暴露均在每个预测时点重新估计。监督回归共享同一批满足 **s+12<t** 的已兑现标签；本节无监督 PCA 可以使用当时已公开的宏观观测，即使其对应收益尚未兑现：输入已滞后两月，PCA 最晚参考月为 t−2。这样不会把收益标签提前，也不必额外舍弃一年宏观信息。扩展窗口 PCA 使用 1964 年以来可得观测，滚动 PCA 使用最近 240 月；回归的滚动窗口仍按已兑现标签计算。基础报告及前述策略扩展采用更保守的“成熟标签训练行内 PCA”，与本节实现不同，结果不混用。

主比较为扩展窗口，并补充 240 月滚动窗口。对数收益用于文献公式检验，简单收益另行重新拟合用于投资回测，不对预测的对数收益直接取指数冒充预期简单收益。数据面板、两月发布约定和样本截止日与原文不同，属于文献规格在现有数据上的重估，不是逐表精确复现。

**因子为何可能有效。** [CP 的研究](https://web.stanford.edu/~piazzesi/cp.pdf)强调远期曲线中随时间变化的风险补偿，组合系数形状是经验结果，不能人为强制成帐篷。[LN 的研究](https://www.nber.org/papers/w11703)检验曲线未完全涵盖的宏观状态、实际活动和通胀信息。FB 是更直接的远期利差基准。情绪与趋势则是需求变化、风险承担和调整速度的代理假设，不能把相关性当作因果机制，也不能把股票情绪直接称为已经识别的债券情绪。

**样本内和样本外分开。** 样本内固定为 1964-01–2003-12，使用该区间已知历史文件事后拟合，属于描述分析；样本外从 1990-01 开始，每次训练均满足 s+12<t。两个研究区间有重叠，样本内估计从不被样本外程序复用，故重叠不构成拟合泄漏，但不能称为两份独立证据。下面展示一年期对数篮子收益，IS 调整 R² 取指数构造阶段的平均收益回归；CP/LN 的二阶段篮子预测与一阶段预测数值一致。
"""]
    rows = []
    def pvalue(p):
        return "<0.001" if p < .001 else f"{p:.3f}"
    for tag, label in [("baseline", "FB+JLN"), ("gsw", "GSW+FRED")]:
        ins, oos = data[tag]["first_stage"], data[tag]["oos_tests"]
        for name, first in [("CP_2stage", "CP"), ("LN6_2stage", "LN6"), ("LN5_2stage", "LN5"), ("LN8_2stage", "LN8"), ("CP_LN_raw", "CP_LN_raw")]:
            a = ins[(ins.target == "log") & (ins.model == first)].iloc[0]
            b = oos[(oos.target == "log") & (oos.window == "exp") & (oos.maturity == "basket") & (oos.model == name)].iloc[0]
            rows.append([label, name, a.adjusted_r2_pct, b.r2_pct, pvalue(b.raw_loss_p),
                         pvalue(b.hac_p), pvalue(b.bootstrap_t_p24), pvalue(b.bootstrap_holm_p24)])
    text.append(md(["面板", "规格", "IS 调整 R² (%)", "OOS R² (%)", "原始损失 p", "CW-HAC p", "bootstrap-t p", "bootstrap Holm p"], rows))
    extreme = data["gsw"]["extreme_forecasts"]
    e = extreme[(extreme.target == "log") & (extreme.window == "exp") & (extreme.model == "LN6_2stage")].iloc[0]
    audit = data["gsw"]["forecast_audit"]
    a = audit[(audit.ym == e.ym) & (audit.window == "exp")].iloc[0]
    text.append(f"**异常值按原规格保留并定位。** GSW/FRED 的 LN6 在 {e.ym} 预测为 {e.forecast_pct:.2f} 个对数收益百分点，实际为 {e.actual_pct:.2f}；该月占该规格全期平方预测误差的 {e.squared_error_share_pct:.2f}%。当期标准化 F1={a.F1_current:.2f}，成熟标签训练样本范围为 [{a.F1_training_min:.2f}, {a.F1_training_max:.2f}]，三次项对疫情冲击产生极端外推。没有为抬高 R² 删除这一时点，也没有将 LN8 偷换为原文 LN6。它说明该多项式规格在更新数据上的幅度预测不稳，不能仅凭年度截断仓位表现较好就认可预测模型。完整最差月份与训练因子范围已留档。")
    starts = []
    for tag, label in [("baseline", "FB+JLN"), ("gsw", "GSW+FRED")]:
        table = data[tag]["start_sensitivity"]
        table = table[(table.target == "log") & (table.window == "exp")].pivot(index="model", columns="origin", values="r2_pct")
        for name in ("FB_n", "CP_2stage", "LN6_2stage", "LN5_2stage", "CP_LN5_2stage"):
            starts.append([label, name, *[table.loc[name, s] for s in ("1985-01", "1990-01", "1995-01", "2000-01")]])
    text.append("**六个问题也覆盖恢复后的文献基准。** 下表补充必做规格和 LN5 联合规格的起点检验，均为一年期对数篮子收益、扩展估计、相同面板结束月；异常外推造成的极端负值保留。前面的 15 组合、六周期均线与历史版本审计仍是 LN8/简化情绪/TI 扩展的结果，不能混称原文规格。")
    text.append(md(["面板", "规格", "1985 起 R² (%)", "1990 起", "1995 起", "2000 起"], starts))
    decomposition = []
    for tag, label in [("baseline", "FB+JLN"), ("gsw", "GSW+FRED")]:
        d = data[tag]["r2_decomposition"]
        for r in d[(d.target == "log") & d.model.isin(["FB_n", "LN6_2stage", "LN5_2stage"])].itertuples():
            decomposition.append([label, r.model, r.initial_is_r2_pct, r.oos_vs_eh_r2_pct, r.oos_vs_test_mean_r2_pct])
    text.append("对第⑥问再用这些原文规格复查：下表的 IS 是 1990 年第一次预测实际可用训练标签的拟合值，与上方 1964–2003 描述样本不同。测试期样本均值用到未来信息，仅用于诊断分母，不是可交易基准。不同期、不同分母的 R² 没有必须 IS≥OOS 的关系。")
    text.append(md(["面板", "规格", "初始 IS R² (%)", "OOS / EH R² (%)", "OOS / 测试均值 R² (%)"], decomposition))
    text.append("第④问的 24 组 γ/cap 网格也已对本节全部规格重新计算，保存在 `risk_sensitivity.csv`；不按新结果挑风险偏好。第①问同时保存 LN5/CP 联合相对 **CP 自身** 的增量损失比较，而不只与较弱常数基准比较。同理，前述月度候选中的 LN 是 LN8 扩展；其结果不能移植给 LN6 或原文 BW。")
    vintage_path = output/"gsw_vintage"/"oos_tests.csv"
    if vintage_path.exists():
        vintage = pd.read_csv(vintage_path)
        vintage = vintage[(vintage.target == "log") & (vintage.window == "exp") & (vintage.maturity == "basket")].set_index("model")
        latest = data["gsw"]["predictions"] if "predictions" in data["gsw"] else pd.read_csv(output/"gsw"/"predictions.csv")
        latest = latest[(latest.target == "log") & (latest.window == "exp")].copy()
        latest["average"] = latest[[f"pred{n}" for n in range(2, 6)]].mean(axis=1)
        latest = latest.pivot(index="ym", columns="model", values="average").loc["2000-01":"2024-02"]
        dates = pd.PeriodIndex(latest.index, freq="M")
        actual = load_gsw_bond_data().set_index("ym").loc[dates, [f"rx{n}" for n in range(2, 6)]].mean(axis=1).to_numpy()
        rows = []
        for name in ("FB_n", "CP_2stage", "LN6_2stage", "LN5_2stage", "LN8_2stage", "CP_LN5_2stage"):
            r = vintage.loc[name]
            rows.append([name, 100*oos_r2(actual, latest[name], latest.EH), r.r2_pct,
                         pvalue(r.raw_loss_p), pvalue(r.holm_p), pvalue(r.bootstrap_holm_p24)])
        first = vintage.iloc[0]
        text.append(f"**第⑤问补充：原文 LN5/LN6 的宏观历史版本检验。** 使用已有官方 ZIP，对每个预测月采用 t−1 版本和 t−2 参考月，按该版本的历史面板重新构造 PCA 和两阶段指数；不以新版值填补旧版本。为与本节修订版对应，评价共同期为 {first.start}–{first.end}，{int(first.n)} 个月，扩展窗口、一年期对数篮子收益。训练宏观面板的历史修订、构成和变换差异共同影响结果，不能把两列差值全部归为单一修订效应。")
        text.append(md(["规格", "同期间修订版 R² (%)", "历史版本 R² (%)", "历史原始损失 p", "历史 HAC Holm p", "历史 bootstrap Holm p"], rows))
        text.append("这次原文规格也进行了版本敏感性检查，但不能将历史版本实验自动称为‘通过’。特别保留负 R²、极端预测和校正后 p 值；t−1 月版本约定不能替代逐系列精确发布日志，收益率曲线仍是当前下载版，情绪未取得完整历史档案。")
    else:
        text.append("本节 LN5/LN6 尚未生成宏观历史版本结果；先准备官方 ZIP，再运行 `--with-vintages`。此前 LN8 的版本检验不能替代它们。")
    text.append(r"""样本内第一阶段系数的 HAC t 值、逐期限两阶段 R²、两种收益口径和两种训练窗口的完整结果均留档。第一阶段共同因子是估计得到的，二阶段 t 值不能作为与因子构造独立的显著性证据；本报告没有用目标拟合出的单因子再制造一组“独立显著”结论。

**CW 与 bootstrap-t 的适用范围。** 对能确认与同训练样本 EH 嵌套的平均收益 OLS 规格，报告 [Clark–West 调整](https://www.kansascityfed.org/documents/5368/pdf-RWP05-05.pdf)：

$$d_t=(x_t-\widehat x_{EH,t})^2-(x_t-\widehat x_{M,t})^2+(\widehat x_{EH,t}-\widehat x_{M,t})^2.$$

其余比较使用未调整损失差，所有规格另保存原始损失差 p 值。CW p 值不改变原始 OOS R²，调整后显著也不等于原始 MSE 更低。bootstrap-t 对 d 去均值后按 12/24 月循环区块重抽，每次重新估计 18 阶 Bartlett HAC 标准误，形成学生化统计量；默认重复 1,999 次、种子 42，实际次数记录在结果表，并保存 Holm 校正。HAC 的 Holm 集合含每面板全部非 EH 的收益口径×窗口×期限检验；bootstrap Holm 集合含全部非 EH 篮子检验，不能覆盖整份研究所有历史尝试。它与之前对夏普差的百分位 bootstrap 是不同检验。这里重抽已产生的损失序列，**没有**重新估计整个历史 PCA/预测流水线，不能宣称覆盖全部潜因子和研究者搜索误差；含 PCA 的 CW 渐近近似、极端值及有限样本也要求谨慎。

**宏观经济解释不靠给 PC 编号贴标签。** 以下载荷来自 GSW/FRED 的同一 1964–2003 描述样本，每行列出标准化宏观面板中绝对载荷最大的变量；因子符号可整体翻转。OOS 仍逐期重新提取，不能用这张事后载荷表替代当期信息。
""")
    text.append(md(["主成分", "解释宏观变异 (%)", "绝对载荷最大的五列（括号为载荷）"],
                   [[r.factor, r.variance_share_pct, r.top_absolute_loadings] for r in data["gsw"]["pca_loadings"].itertuples()]))
    text.append("**经济意义需要相对基准。** 正的总收益本身可能仅来自国库券利息或国债久期溢价；因此保留同交易规则 EH、被动篮子和现金，并比较收益、风险及 ΔCER。正式作业允许风险厌恶的均值—方差配置，γ=5 属于此类，并非风险中性。若把 σ 定义为标准差，效用一阶条件使用 μ/(γσ²)；γ=0 的风险中性问题不能直接套入这个公式，必须另加投资约束。")
    rows = []
    for tag, label in [("baseline", "FB+JLN"), ("gsw", "GSW+FRED")]:
        econ = data[tag]["economic_summary"].set_index("model")
        for name in ["EH", "FB_n", "CP_2stage", "LN6_2stage", "LN5_2stage", "CP_LN5_2stage", "BuyHold"]:
            r = econ.loc[name]
            rows.append([label, name, int(r.years), r.cagr_pct, r.sharpe, r.delta_cer_bps])
    text.append("一年期简单收益重新预测、γ=5/cap3、固定 1 月入场的经济检验如下；费用/融资假设与基础年度回测相同。这是理论配置比较，不能把该杠杆规则直接视为实盘可成交账户。")
    text.append(md(["面板", "规格", "年度数", "净 CAGR (%)", "超额夏普", "ΔCER 对 EH (bps)"], rows))
    text.append("![文献基准的预测能力、三次项外推与投资效用](assets/coursework_check.png)\n\n图中短样本预测、GSW 因子外推与年度效用使用各自标注的样本。年度仅固定一月入场并限制仓位，不能用其正 ΔCER 掩盖全期月度预测中的极端误差。")
    bridge = pd.read_csv(output/"monthly_economic_bridge.csv")
    if len(bridge):
        text.append("对新增月度策略，再将实际月度账户收益复利为完整的不重叠年度区块，以相同 γ=5 计算年度 CER。下表明确区分相对同窗口 EH 与相对被动的效用改善，不把 CAGR 差或夏普差直接称为业绩费：")
        text.append(md(["面板", "规则", "完整年度数", "ΔCER 对 EH (bps)", "ΔCER 对被动 (bps)"],
                       [[r.panel, r.model, int(r.years), r.delta_cer_eh_bps, r.delta_cer_passive_bps]
                        for r in bridge.itertuples() if r.model in {"EW_R100/w120", "EW/w120", "LN_R100/w120", "EW/select"}]))
    path = ROOT/"output"/"trading_designs"/"monthly_revised"/"forecast_summary.csv"
    if path.exists():
        q = pd.read_csv(path).set_index("model")
        r = q.loc["EW_R100/w120", "r2_pct"]
        eh = q.loc["EH/w120", "r2_pct"]
        incremental = 100*(1-(1-r/100)/(1-eh/100))
        text.append(f"月度策略也须防止基准选择造成误读：`EW_R100/w120` 收缩后预测相对扩展 EH 的 R²={r:.3f}%，但同为 120 月窗口的 EH 已有 {eh:.3f}%；换成同窗口 EH，增量 R² 仅 {incremental:.3f}%。因此夏普和 CER 点估计改善不等于宏观/情绪因子产生了已确认的额外预测能力。")
        monthly_is = pd.read_csv(output/"monthly_in_sample.csv")
        first = monthly_is.iloc[0]
        rows = [[r.model+"/exp", r.r2_pct, q.loc[r.model+"/exp", "raw_r2_pct"],
                 q.loc[r.model+"/exp", "r2_pct"]] for r in monthly_is.itertuples()]
        text.append(f"**不同预测期的样本内证据。** 对已有一月期扩展补齐相同输入定义的 IS：GSW/FRED、简化情绪和 TI，描述样本为 {first.start}–{first.end}，{int(first.n)} 个月。下面 IS 为未收缩拟合 R²，OOS 为扩展窗口的 1990-01–2024-02 预测，相对逐期 EH；列出原始预测与 50% 向 EH 收缩后的预测，避免混淆。此处 LN 表示八主成分扩展；不同预测期的 R² 分母及收益目标不同，不按绝对大小跨期排名。")
        text.append(md(["一月期规格", "IS R² (%)", "OOS 原始 R² (%)", "OOS 收缩后 R² (%)"], rows))
    text.append("**对 EH 与策略有效性的结论。** 在 FB/JLN 的历史样本中，LN5/LN6 提供正的一年期 OOS R² 和投资效用点估计，但扩大检验集合后尚不足以确认稳定预测优势；这不等于完成了严格的 EH 拒绝。在更新 GSW/FRED 面板上，原文多项式对疫情宏观冲击的外推失败，宏观历史版本也没有消除这个问题，说明样本、非线性和信息版本会实质改变结论。与此同时，一月期、滚动估计、预测收缩和仓位约束形成了可保留的历史候选；这些候选的收益/风险改善与增量 MSE、校正后显著性应分开评价。因此当前有效结论是‘存在依赖数据、时期和交易规则的可预测性与经济价值，尚未确立跨样本稳定的超额投资优势’，而不是由部分失败推导所有策略无效。")
    text.append("该仓库提供代码与可复现研究说明；课程提交的 Word 报告仍需按正式格式压缩为不超过 3000 字的引言、批判性文献综述、数据方法、实证、稳健性和结论，并补齐封面成员信息。旧版局部文献公式脚本及未同步的报告草稿不能代替这套修正后的实证结果。")
    text.append("复现本节：`python scripts/check_coursework_alignment.py --with-vintages --update-readme`，历史宏观 ZIP 可先用 `audit_macro_vintages.py --download` 准备。本地 `output/coursework_audit/` 保存文献规格预测、训练截止日、IS 系数/载荷、逐期限 OOS、CW、bootstrap-t、起点/风险网格、异常预测和经济价值表；数据及本地课程材料不上传。")
    body = "\n\n".join(text)
    (output/"report.md").write_text(body)
    path = ROOT/"README.md"
    content = path.read_text()
    block = "<!-- BEGIN COURSEWORK_CHECK -->\n"+body+"\n<!-- END COURSEWORK_CHECK -->\n\n"
    if "<!-- BEGIN COURSEWORK_CHECK -->" in content:
        content = re.sub(r"<!-- BEGIN COURSEWORK_CHECK -->.*?<!-- END COURSEWORK_CHECK -->\s*", lambda m: block, content, flags=re.S)
    else:
        content = content.replace("## 6. 已修复", block+"## 6. 已修复")
    path.write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"output"/"coursework_audit")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--replications", type=int, default=1999)
    parser.add_argument("--with-vintages", action="store_true", help="Also audit required LN5/LN6 with local official vintage ZIPs")
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()
    for tag, loader, source, files in [("baseline", load_bond_data, "jln", ["Fama_Bliss_price.txt", "macro_raw_data.xlsx"]),
                                       ("gsw", load_gsw_bond_data, "fredmd", ["GSW_yields.csv", "2026-04-MD.csv"])]:
        folder = args.output_dir/tag
        folder.mkdir(parents=True, exist_ok=True)
        bonds, macro = loader().set_index("ym"), load_macro_raw_transformed(source)
        key = cache_key(files)
        marker = folder/"cache_key.txt"
        if not args.refresh and marker.exists() and marker.read_text() == key:
            print(f"Verified reference cache: {tag}", flush=True)
            predictions = pd.read_csv(folder/"predictions.csv")
        else:
            print(f"Computing required reference specifications: {tag}", flush=True)
            predictions, audit = reference_forecasts(bonds, macro, end="2012-02" if tag == "baseline" else "2024-02")
            predictions.to_csv(folder/"predictions.csv", index=False)
            audit.to_csv(folder/"forecast_audit.csv", index=False)
            marker.write_text(key)
        sample_inference(bonds, macro, folder)
        prediction_inference(bonds, predictions, folder, args.replications)
        economic_inference(bonds, predictions, folder, args.replications)
        reference_sensitivity(bonds, predictions, folder)
        reference_r2_decomposition(bonds, macro, predictions, folder)
        print(f"Finished coursework evidence: {tag}", flush=True)
    if args.with_vintages:
        folder = args.output_dir/"gsw_vintage"
        folder.mkdir(parents=True, exist_ok=True)
        files = ["GSW_yields.csv", *VINTAGE_URLS]
        key = cache_key(files)
        marker = folder/"cache_key.txt"
        bonds = load_gsw_bond_data().set_index("ym")
        if not args.refresh and marker.exists() and marker.read_text() == key:
            predictions = pd.read_csv(folder/"predictions.csv")
            print("Verified reference cache: gsw_vintage", flush=True)
        else:
            with VintageArchive(DATA_DIR) as archive:
                predictions, audit = reference_vintage_forecasts(bonds, archive)
            predictions.to_csv(folder/"predictions.csv", index=False)
            audit.to_csv(folder/"forecast_audit.csv", index=False)
            marker.write_text(key)
        prediction_inference(bonds, predictions, folder, args.replications)
        economic_inference(bonds, predictions, folder, args.replications)
        print("Finished historical-vintage coursework references", flush=True)
    monthly_economic_bridge(args.output_dir)
    monthly_sample_inference(args.output_dir)
    sources = [*sorted((ROOT/"bond_rp").glob("*.py")), Path(__file__)]
    (args.output_dir/"metadata.json").write_text(json.dumps({
        "generated_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "bootstrap_replications": args.replications, "with_vintages": args.with_vintages,
        "macro_reference_lag_months": 2, "pca_uses_available_unlabelled_macro": True,
        "supervised_cutoff": "s+12<t", "models": REFERENCE_MODELS,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    }, indent=2))
    if args.update_readme:
        plot_coursework(args.output_dir)
        export_readme(args.output_dir)


if __name__ == "__main__":
    main()
