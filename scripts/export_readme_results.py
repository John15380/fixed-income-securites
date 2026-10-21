"""Reproducible, purged OOS forecasts and non-overlapping annual backtests."""

from __future__ import annotations
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import numpy as np
import pandas as pd

from bond_rp.config import DATA_DIR, LASSO_ALPHAS, PUB_LAG_MONTHS
from bond_rp.data import load_bond_data, load_gsw_bond_data, load_macro_raw_transformed, load_sentiment
from bond_rp.models import ForecastResult, forecast_models
from bond_rp.strategy import annual_backtest
from bond_rp.tables import aligned_forecasts, annual_robustness, markdown_table, prediction_table

PRED_COLUMNS = [("n", "月度样本数"), ("r2_pct", "OOS R² (%)"),
                ("rmse_pct", "RMSE (百分点)"), ("hit_pct", "方向命中 (%)")]
ANNUAL_COLUMNS = [("years", "年度交易数"), ("cagr_pct", "扣费 CAGR (%)"),
                  ("annual_vol_pct", "年收益波动 (%)"), ("sharpe", "超额收益夏普"),
                  ("max_dd_pct", "年度端点回撤 (%)"), ("delta_cer_bps", "ΔCER (bps)")]


def fingerprint(files, settings):
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    paths = [ROOT/"bond_rp"/name for name in ["config.py", "data.py", "models.py"]]
    paths += [Path(DATA_DIR)/name for name in files]
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update(json.dumps(versions(), sort_keys=True).encode())
    return digest.hexdigest()


def versions():
    return {name: importlib.metadata.version(name) for name in
            ["numpy", "pandas", "scikit-learn", "statsmodels", "matplotlib"]}


def cached_forecasts(tag, bonds, macro, sentiment, files, args, lag=PUB_LAG_MONTHS, names=None):
    folder = args.output_dir/tag
    folder.mkdir(parents=True, exist_ok=True)
    key = fingerprint(files, {"lag": lag, "include_ml": args.with_ml, "names": names,
                              "protocol": "mature-labels-v2"})
    meta_path = folder/"cache_key.txt"
    if not args.refresh and meta_path.exists() and meta_path.read_text() == key:
        print(f"Using verified forecast cache: {tag}", flush=True)
        predictions = pd.read_csv(folder/"predictions.csv", index_col="ym")
        predictions.index = pd.PeriodIndex(predictions.index, freq="M")
        audit = pd.read_csv(folder/"audit.csv")
    else:
        print(f"Computing {tag}; publication lag={lag} months", flush=True)
        result = forecast_models(bonds, macro, sentiment, include_ml=args.with_ml,
                                 publication_lag=lag, names=names)
        predictions, audit = result.predictions, result.audit
        predictions.to_csv(folder/"predictions.csv")
        audit.to_csv(folder/"audit.csv", index=False)
        meta_path.write_text(key)
    for _, row in audit.iterrows():
        if pd.Period(row.train_last_label_end, "M") >= pd.Period(row.forecast_month, "M"):
            raise AssertionError("A forecast used an unmatured training label")
    return ForecastResult(predictions, audit)


def evaluate(tag, bonds, result, args):
    aligned = aligned_forecasts(result.predictions, bonds)
    prediction = prediction_table(aligned)
    annual, trades = annual_backtest(bonds, aligned.drop(columns="actual"))
    robustness = annual_robustness(bonds, aligned.drop(columns="actual"))
    folder = args.output_dir/tag
    aligned.to_csv(folder/"aligned_forecasts.csv")
    prediction.to_csv(folder/"forecast_summary.csv")
    annual.to_csv(folder/"annual_summary.csv")
    trades.to_csv(folder/"annual_trades.csv", index=False)
    for name, frame in robustness.items():
        frame.to_csv(folder/(name+"_robustness.csv"), index=False)
    return {"aligned": aligned, "prediction": prediction, "annual": annual,
            "trades": trades, "robustness": robustness}


def plot_results(baseline, extension):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    assets = ROOT/"assets"
    assets.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    ordered = baseline["prediction"].drop(index="EH").sort_values("r2_pct")
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharey=True)
    axes[0].barh(ordered.index, ordered.r2_pct, color="#397b92")
    axes[0].axvline(0, color="#293646", linewidth=0.8)
    axes[0].set_title("Monthly OOS forecast R-squared (%)")
    axes[1].barh(ordered.index, baseline["annual"].loc[ordered.index, "sharpe"], color="#d9956a")
    axes[1].axvline(baseline["annual"].loc["EH", "sharpe"], color="#293646", linestyle="--", label="EH")
    axes[1].axvline(baseline["annual"].loc["BuyHold", "sharpe"], color="#ad4f44", linestyle=":", label="BuyHold")
    axes[1].set_title("Non-overlapping annual net Sharpe")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
        ax.set_axisbelow(True)
    first = baseline["prediction"].iloc[0]
    fig.suptitle("Corrected Treasury return timing | baseline inputs", fontsize=15)
    fig.text(0.02, 0.025, f"Forecasts: {first['start']} to {first['end']}, N={int(first['n'])}. "
             "Annual trading: January entries, 12-month holding.\n"
             "Matured labels only; fold-local preprocessing; lagged current-vintage macro data.", fontsize=9)
    fig.tight_layout(rect=(0, 0.09, 1, 0.95))
    fig.savefig(assets/"model_comparison.png", dpi=170)
    plt.close(fig)

    chosen = extension if extension is not None else baseline
    names = [name for name in ["EH", "BuyHold", "CP", "LN", "GBM_raw", "Ensemble"] if name in chosen["annual"].index]
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for name in names:
        trades = chosen["trades"].loc[chosen["trades"].model == name].sort_values("purchase")
        dates = [trades.purchase.iloc[0].to_timestamp(how="end")] + [p.to_timestamp(how="end") for p in trades.sale]
        wealth = np.r_[1., np.cumprod(1+trades.return_net)]
        ax.plot(dates, wealth, marker=".", markersize=4, label=name)
    ax.set_title("Annual endpoint wealth | " + ("continuous GSW inputs" if extension is not None else "Fama-Bliss inputs"))
    ax.set_ylabel("Net wealth (initial = 1)")
    ax.legend(frameon=False, ncol=3)
    ax.grid(alpha=0.2)
    fig.text(0.02, 0.02, "January-to-January trades; annual endpoints only. Intrayear drawdown is not observed.\n"
             "0.5 bp per leg, entry and exit; financing at the one-year zero-coupon yield is assumed.", fontsize=9)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(assets/"annual_equity.png", dpi=170)
    plt.close(fig)
    if extension is not None:
        common = baseline["aligned"].index.intersection(extension["aligned"].index)
        base_same = prediction_table(baseline["aligned"].loc[common])
        ext_same = prediction_table(extension["aligned"].loc[common])
        names = [c for c in ["FB", "CP", "LN", "GBM_raw", "Ensemble"] if c in base_same.index]
        fig, ax = plt.subplots(figsize=(11, 5))
        x = np.arange(len(names))
        ax.bar(x-0.18, base_same.loc[names, "r2_pct"], width=0.36, label="Fama-Bliss + JLN", color="#397b92")
        ax.bar(x+0.18, ext_same.loc[names, "r2_pct"], width=0.36, label="GSW + FRED-MD", color="#d9956a")
        ax.set_xticks(x, names)
        ax.axhline(0, color="#293646", linewidth=0.8)
        ax.set_ylabel("OOS R-squared (%)")
        ax.set_title("Data-source sensitivity | identical evaluation months")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        fig.text(0.02, 0.025, f"Matched evaluation: {common.min()} to {common.max()}, N={len(common)}. "
                 "No cross-source bond splice.\n"
                 "Both datasets use revised historical values; this is not a real-time vintage test.", fontsize=9)
        fig.tight_layout(rect=(0, 0.11, 1, 1))
        fig.savefig(assets/"extension_check.png", dpi=170)
        plt.close(fig)


def update_readme(baseline, extension, comparison):
    path = ROOT/"README.md"
    content = path.read_text()
    values = {"BASELINE_SAMPLE": sample_text(baseline),
              "BASELINE_TABLE": markdown_table(baseline["prediction"], PRED_COLUMNS),
              "BASELINE_ANNUAL": markdown_table(baseline["annual"], ANNUAL_COLUMNS),
              "ENVIRONMENT": "| 依赖 | 版本 |\n| --- | --- |\n" + "\n".join(f"| {n} | {v} |" for n, v in versions().items())}
    if extension is not None:
        values.update({"EXTENSION_SAMPLE": sample_text(extension),
                       "EXTENSION_TABLE": markdown_table(extension["prediction"], PRED_COLUMNS),
                       "EXTENSION_ANNUAL": markdown_table(extension["annual"], ANNUAL_COLUMNS),
                       "MATCHED_TABLE": comparison_markdown(comparison)})
    selected = extension if extension is not None else baseline
    signal = "Ensemble" if "Ensemble" in selected["annual"].index else "LN"
    ann = selected["annual"].loc[signal]
    pred = selected["prediction"].loc[signal]
    offsets = selected["robustness"]["offset"]
    offset = offsets.loc[offsets.model == signal, "sharpe"]
    top = selected["prediction"].r2_pct.idxmax()
    values["FINDINGS"] = (
        f"1. 在{('连续 GSW 扩展' if extension is not None else '基准')}样本上，{top} 的预测 R² 在参与比较的预测模型（含历史均值基准）中最高，"
        f"为 {selected['prediction'].loc[top, 'r2_pct']:.2f}%；这是模型比较结果，不用于事后挑选组合。\n"
        f"2. 预先固定的 {signal} 规格预测 R² 为 {pred.r2_pct:.2f}%，"
        f"每年 1 月建仓的扣费 CAGR 为 {ann.cagr_pct:.2f}%，超额收益夏普为 {ann.sharpe:.2f}，"
        f"基于 {int(ann.years)} 笔不重叠年度交易。EH 夏普为 {selected['annual'].loc['EH','sharpe']:.2f}，"
        f"等权持有基准的 CAGR 为 {selected['annual'].loc['BuyHold','cagr_pct']:.2f}%、夏普为 "
        f"{selected['annual'].loc['BuyHold','sharpe']:.2f}；组合未在所有风险调整指标上胜过被动持有。\n"
        f"3. 将固定建仓月份从 1 月改为 1–12 月分别回测，该规格的夏普范围为 "
        f"{offset.min():.2f}–{offset.max():.2f}。这 12 条路径相互重叠，不能合并成 12 倍独立样本。\n"
        f"4. 组合预测的损失差 HAC 检验 p 值为 {pred.hac_p:.3f}，Holm 校正后为 {pred.holm_p:.3f}。"
        "不将正的 R² 点估计直接解释为统计显著；预测精度、交易表现与不同数据源的稳健性需要分别判断。"
        "当前结果只支持对所列历史数据和交易假设的研究结论，不等于真实成交记录或实时信息有效性证明。")
    if extension is not None:
        later = comparison.loc[(comparison.panel == "GSW+FRED：2013 年起") & (comparison.model == signal)]
        if not later.empty:
            values["FINDINGS"] += (f"\n5. 单看 2013 年起的较晚样本，该组合预测 R² 为 {later.iloc[0].r2_pct:.2f}%。"
                                   "应单独判断较晚时期的表现，不能从全样本的正 R² 外推跨时期稳健性。")
    values["RESUME"] = (
        "搭建美国国债多因子收益预测与资产配置研究框架，覆盖远期利率、宏观 PCA、情绪及机器学习模型；"
        "实现一年期标签兑现隔离、带隔离区的时间交叉验证、HAC 误差检验与含交易费用的不重叠年度回测，"
        "并检验建仓月份、成本和数据源敏感性。\n\n"
        f"在所列历史数据及融资成本假设下，"
        f"{signal} 的 {int(ann.years)} 笔年度回测扣费 CAGR 为 {ann.cagr_pct:.2f}%，"
        f"超额收益夏普为 {ann.sharpe:.2f}。该指标来自所述理论组合的历史回测。")
    for name, value in values.items():
        pattern = rf"(<!-- BEGIN {name} -->).*?(<!-- END {name} -->)"
        content, count = re.subn(pattern, lambda m: f"{m[1]}\n{value}\n{m[2]}", content, flags=re.S)
        if count != 1:
            raise ValueError(f"Expected one README block: {name}")
    path.write_text(content)


def comparison_markdown(frame):
    text = ["| 数据与时期 | 模型 | 月数 | R² (%) |", "| --- | --- | --- | --- |"]
    for _, row in frame.iterrows():
        text.append(f"| {row.panel} | {row.model} | {int(row.n)} | {row.r2_pct:.2f} |")
    return "\n".join(text)


def sample_text(result):
    first = result["prediction"].iloc[0]
    annual = result["annual"].iloc[0]
    return (f"月度预测评价：**{first['start']}–{first['end']}，{int(first['n'])} 个月**；"
            f"固定 1 月建仓的年度回测：**{annual['start']}–{annual['end']}，{int(annual['years'])} 笔交易**。")


def main(default_full=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-ml", action="store_true", default=default_full)
    parser.add_argument("--with-extensions", action="store_true", default=default_full)
    parser.add_argument("--update-readme", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="Discard matching prediction caches")
    parser.add_argument("--output-dir", type=Path, default=ROOT/"output"/"corrected")
    args = parser.parse_args()
    if args.update_readme and not (args.with_ml and args.with_extensions):
        parser.error("README update requires --with-ml --with-extensions")
    files = ["Fama_Bliss_price.txt", "macro_raw_data.xlsx", "SENTIMENT.xlsx"]
    bonds, macro, sentiment = load_bond_data(), load_macro_raw_transformed(), load_sentiment()
    result = cached_forecasts("baseline", bonds, macro, sentiment, files, args)
    baseline = evaluate("baseline", bonds, result, args)
    extension, comparisons = None, []
    if args.with_extensions:
        ext_files = ["GSW_yields.csv", "2026-04-MD.csv", "SENTIMENT.xlsx"]
        ext_bonds, ext_macro = load_gsw_bond_data(), load_macro_raw_transformed("fredmd")
        ext_result = cached_forecasts("gsw", ext_bonds, ext_macro, sentiment, ext_files, args)
        extension = evaluate("gsw", ext_bonds, ext_result, args)
        common = baseline["aligned"].index.intersection(extension["aligned"].index)
        for title, aligned in [("FB+JLN：共同期", baseline["aligned"].loc[common]),
                                ("GSW+FRED：共同期", extension["aligned"].loc[common]),
                                ("GSW+FRED：2013 年起", extension["aligned"].loc["2013-01":])]:
            table = prediction_table(aligned)
            for model in ["EH", "CP", "LN", "Ensemble"]:
                if model in table.index:
                    row = table.loc[model]
                    comparisons.append({"panel": title, "model": model, "n": row.n, "r2_pct": row.r2_pct})
        # Lag sensitivity: compare identical dates across all three lag assumptions.
        lag_results = {}
        names = ["EH", "FB", "CP", "LN", "BW", "CPLN", "LNBW"]
        for lag in [1, 2, 3]:
            lag_result = cached_forecasts(f"lag{lag}", ext_bonds, ext_macro, sentiment, ext_files, args, lag, names)
            lag_results[lag] = aligned_forecasts(lag_result.predictions, ext_bonds)
        shared = extension["aligned"].index
        for frame in lag_results.values():
            shared = shared.intersection(frame.index)
        lag_tables = []
        for lag, frame in lag_results.items():
            table = prediction_table(frame.loc[shared]).reset_index()
            table["publication_lag"] = lag
            lag_tables.append(table)
        pd.concat(lag_tables).to_csv(args.output_dir/"lag_sensitivity.csv", index=False)
        files += ["GSW_yields.csv", "2026-04-MD.csv"]
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(args.output_dir/"matched_comparison.csv", index=False)
    metadata = {"protocol": "matured-labels; purged CV; fold-local preprocessing; annual non-overlapping trades",
                "target": "simple annual excess return, percentage points", "publication_lag": PUB_LAG_MONTHS,
                "lasso_alpha_grid": LASSO_ALPHAS, "lasso_tuning": "first forecast and each January",
                "generated_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
                "python": sys.version.split()[0],
                "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in [*sorted((ROOT/"bond_rp").glob("*.py")), Path(__file__)]},
                "versions": versions(), "input_sha256": {name: hashlib.sha256((Path(DATA_DIR)/name).read_bytes()).hexdigest()
                                                            for name in files}}
    (args.output_dir/"metadata.json").write_text(json.dumps(metadata, indent=2))
    plot_results(baseline, extension)
    if args.update_readme:
        update_readme(baseline, extension, comparison)
    print("\nBaseline forecast diagnostics:\n"+markdown_table(baseline["prediction"], PRED_COLUMNS))
    print("\nBaseline annual trading:\n"+markdown_table(baseline["annual"], ANNUAL_COLUMNS))
    if extension is not None:
        print("\nContinuous GSW annual trading:\n"+markdown_table(extension["annual"], ANNUAL_COLUMNS))
    print(f"\nSaved corrected results to {args.output_dir}")


if __name__ == "__main__":
    main()
