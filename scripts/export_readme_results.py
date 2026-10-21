"""Recompute README tables and figures from local inputs.

This preserves the original research specification, including its unembargoed
12-month labels. It deliberately does not plot a compounded equity curve.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))

import numpy as np
import pandas as pd

from bond_rp.config import DATA_DIR, GAMMA_BASE, IS_END, OOS_START, TRANSACTION_COST_BPS, W_CAP
from bond_rp.data import (
    load_bond_data, load_bond_data_extended, load_macro_factors_oos,
    load_macro_factors_oos_extended, load_macro_raw_transformed, load_sentiment,
)
from bond_rp.metrics import hit_rate, oos_r2
from bond_rp.models import (
    build_all_oos, oos_eh, oos_fb, oos_cp, oos_ln, oos_ln_lasso,
    oos_rf, oos_gbm, oos_lasso_raw, oos_rf_raw, oos_gbm_raw,
)
from bond_rp.strategy import mv_strategy


def evaluate(predictions: dict, bonds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate every model and its EH benchmark on one identical set of dates."""
    common = bonds[["rx_bar", "y1"]].dropna().index
    for series in predictions.values():
        common = common.intersection(series.dropna().index)
    common = common.sort_values()
    if len(common) < 24:
        raise ValueError("Fewer than 24 common observations; check input coverage.")

    # Match the existing Table 3 specification. These forward labels also
    # include outcomes realized in 1990; this is disclosed in README.md.
    initial = bonds.loc[(bonds.index >= "1964-01") & (bonds.index <= IS_END), "rx_bar"].dropna()
    variance = np.var(initial.to_numpy() / 100, ddof=1)
    actual = bonds.loc[common, "rx_bar"].to_numpy()
    short_yield = bonds.loc[common, "y1"].to_numpy()
    benchmark = predictions["EH"].loc[common].to_numpy()
    reference = mv_strategy(benchmark, actual, short_yield, variance,
                            GAMMA_BASE, W_CAP, TRANSACTION_COST_BPS)
    rows = []
    for name, series in predictions.items():
        prediction = series.loc[common].to_numpy()
        strategy = mv_strategy(prediction, actual, short_yield, variance,
                               GAMMA_BASE, W_CAP, TRANSACTION_COST_BPS)
        rows.append({
            "model": name,
            "start": str(common.min()), "end": str(common.max()), "n": len(common),
            "r2_pct": 100 * oos_r2(actual, prediction, benchmark),
            "hit_pct": 100 * hit_rate(actual, prediction),
            "mean_excess_pct": strategy["mean_er_pct"],
            "std_excess_pct": strategy["std_er_pct"],
            "sharpe": strategy["sharpe"],
            "mean_abs_weight": strategy["mean_w"],
            "mean_turnover": strategy["turnover"],
            "utility_gain_bps": 10000 * (strategy["au"] - reference["au"]),
            "net_utility_gain_bps": 10000 * (strategy["au_net"] - reference["au_net"]),
        })
    predictions_frame = pd.DataFrame({name: s.loc[common] for name, s in predictions.items()})
    predictions_frame["actual_rx_bar"] = actual
    return pd.DataFrame(rows).set_index("model"), predictions_frame


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [
        ("model", "模型"), ("r2_pct", "预测 R² (%)"), ("hit_pct", "方向命中 (%)"),
        ("mean_excess_pct", "加权一年期超额收益均值 (%)"), ("sharpe", "均值/标准差"),
        ("utility_gain_bps", "ΔAU (bps)"), ("net_utility_gain_bps", "ΔAU 扣费 (bps)"),
    ]
    lines = ["| " + " | ".join(label for _, label in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for name, row in frame.iterrows():
        cells = [name] + [f"{row[key]:.2f}" for key, _ in columns[1:]]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_figures(summary: pd.DataFrame, extension: pd.DataFrame | None) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white"})
    ordered = summary.drop(index="EH").sort_values("r2_pct")
    colors = ["#dc9167" if name.endswith("raw") else "#397b92" for name in ordered.index]
    height = max(5.5, 0.38 * len(ordered) + 1.6)
    fig, axes = plt.subplots(1, 2, figsize=(13, height), sharey=True)
    for ax, column, title, suffix in [
        (axes[0], "r2_pct", "Prediction accuracy vs historical mean", "%"),
        (axes[1], "utility_gain_bps", "Mean-variance utility gain vs EH", " bps"),
    ]:
        ax.barh(ordered.index, ordered[column], color=colors, height=0.67)
        ax.axvline(0, color="#283a49", linewidth=0.8)
        ax.set_title(title, fontsize=12, pad=14)
        ax.grid(axis="x", alpha=0.18)
        ax.set_axisbelow(True)
        lower, upper = ordered[column].min(), ordered[column].max()
        span = max(upper - min(lower, 0), 1)
        ax.set_xlim(min(lower, 0) - span * 0.18, max(upper, 0) + span * 0.28)
        for i, value in enumerate(ordered[column]):
            ax.text(value + span * (0.025 if value >= 0 else -0.025), i,
                    f"{value:.1f}{suffix}", va="center",
                    ha="left" if value >= 0 else "right", fontsize=9)
    first = summary.iloc[0]
    fig.suptitle("U.S. Treasury return timing | historical research estimates", fontsize=16, y=0.98)
    fig.text(0.02, 0.035,
             f"Common sample: {first['start']} to {first['end']} | N={int(first['n'])} | "
             "12-month overlapping labels; no label embargo.\n"
             "Utility gains are gross research statistics, not realized annual portfolio returns.",
             color="#56616c", fontsize=9)
    fig.tight_layout(rect=(0, 0.12, 1, 0.93))
    fig.savefig(assets / "model_comparison.png", dpi=170)
    plt.close(fig)

    if extension is not None:
        names = [name for name in extension.index if name != "EH" and name in summary.index]
        fig, ax = plt.subplots(figsize=(10, 5))
        positions = np.arange(len(names))
        ax.bar(positions - 0.18, summary.loc[names, "r2_pct"], width=0.36,
               label="Baseline inputs", color="#397b92")
        ax.bar(positions + 0.18, extension.loc[names, "r2_pct"], width=0.36,
               label="Spliced bond data + FRED-MD", color="#dc9167")
        ax.set_xticks(positions, names)
        ax.set_ylabel("Prediction R-squared vs EH (%)")
        ax.axhline(0, color="#283a49", linewidth=0.8)
        ax.set_title("Extension check: model rankings depend on inputs and sample", pad=14)
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ext_first = extension.iloc[0]
        fig.text(0.02, 0.025,
                 f"Extension: {ext_first['start']} to {ext_first['end']}, N={int(ext_first['n'])}. "
                 "Inputs and dates both change.\n"
                 "Historical research estimates; overlapping labels are not embargoed.",
                 fontsize=9, color="#56616c")
        fig.tight_layout(rect=(0, 0.12, 1, 1))
        fig.savefig(assets / "extension_check.png", dpi=170)
        plt.close(fig)


def update_readme(summary: pd.DataFrame, extension: pd.DataFrame, metadata: dict) -> None:
    """Replace only marked result blocks, keeping prose and source notes intact."""
    path = ROOT / "README.md"
    content = path.read_text(encoding="utf-8")

    def sample_text(frame: pd.DataFrame) -> str:
        row = frame.iloc[0]
        return (f"共同评价月份：**{row['start']} 至 {row['end']}，{int(row['n'])} 个有效观测**。"
                "区间内缺失月份已取交集剔除，不能把区间长度直接当作观测数。")

    def number(name: str, column: str) -> str:
        return f"{summary.loc[name, column]:.2f}"

    best = summary.drop(index="EH")["r2_pct"].idxmax()
    findings = (
        f"1. **宏观信息值得进一步研究。** 在本次共同样本上，LN 的预测 R² 为 "
        f"{number('LN', 'r2_pct')}%，FB 为 {number('FB', 'r2_pct')}%，CP 为 "
        f"{number('CP', 'r2_pct')}%。固定因子组合和递归 LASSO 版本 "
        f"（{number('LN_LASSO', 'r2_pct')}%）的差异说明变量选择会影响结果。\n"
        f"2. **GBM 在当前比较中领先，树模型并非普遍优于线性。** 当前设定中 {best} 的预测 R² 最高，"
        f"为 {number(best, 'r2_pct')}%；RF_raw 为 {number('RF_raw', 'r2_pct')}%，"
        f"LASSO_raw 为 {number('LASSO_raw', 'r2_pct')}%。这是需要在正确时间对齐后重检的模型排序。\n"
        f"3. **预测改善不等于风险调整收益改善。** LN 的加权一年期超额收益均值为 "
        f"{number('LN', 'mean_excess_pct')}%，但其收益/波动比为 {number('LN', 'sharpe')}，"
        f"EH 为 {number('EH', 'sharpe')}；杠杆与波动会改变配置效果。\n"
        f"4. **组合信号不保证优于单模型。** LN+BW 的 R² 为 {number('LNBW', 'r2_pct')}%，"
        f"CP+LN 为 {number('CPLN', 'r2_pct')}%；加入更多信号可能同时引入噪声。\n"
        f"5. **宏观模型的预测优势未在扩展设定中保持。** 更换宏观面板并拼接 GSW 后，LN 的 R² 为 "
        f"{extension.loc['LN', 'r2_pct']:.2f}%，LN_LASSO 为 {extension.loc['LN_LASSO', 'r2_pct']:.2f}%，"
        f"FB 为 {extension.loc['FB', 'r2_pct']:.2f}%。"
        "宏观规格的预测误差高于历史均值基准，不能宣称策略跨样本稳健有效；"
        "需要拆分数据源变化与时期变化后重新检验。"
    )
    environment = ("| 依赖 | 本次版本 |\n| --- | --- |\n" +
                   "\n".join(f"| {name} | {version} |" for name, version in metadata["versions"].items()))
    replacements = {
        "BASELINE_SAMPLE": sample_text(summary), "BASELINE_TABLE": markdown_table(summary),
        "EXTENSION_SAMPLE": sample_text(extension), "EXTENSION_TABLE": markdown_table(extension),
        "FINDINGS": findings, "ENVIRONMENT": environment,
    }
    for name, value in replacements.items():
        pattern = rf"(<!-- BEGIN {name} -->).*?(<!-- END {name} -->)"
        content, count = re.subn(pattern, lambda m: f"{m[1]}\n{value}\n{m[2]}", content, flags=re.S)
        if count != 1:
            raise ValueError(f"Expected exactly one README block: {name}")
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-ml", action="store_true", help="Also fit PCA/raw LASSO, RF, and GBM.")
    parser.add_argument("--with-extensions", action="store_true", help="Also evaluate GSW/FRED-MD inputs.")
    parser.add_argument("--update-readme", action="store_true", help="Update the marked README result blocks.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output")
    args = parser.parse_args()
    if args.update_readme and not (args.with_ml and args.with_extensions):
        parser.error("--update-readme requires --with-ml and --with-extensions")
    warnings.filterwarnings("ignore", category=FutureWarning)

    bonds = load_bond_data()
    factors, last_macro = load_macro_factors_oos()
    sentiment = load_sentiment()
    predictions, indexed = build_all_oos(bonds, factors, last_macro, df_sent=sentiment)
    print("Computing LN_LASSO ...", flush=True)
    predictions["LN_LASSO"] = oos_ln_lasso(indexed, factors)
    raw = load_macro_raw_transformed()
    if args.with_ml:
        for name, function, features in [
            ("RF_pca", oos_rf, factors), ("GBM_pca", oos_gbm, factors),
            ("LASSO_raw", oos_lasso_raw, raw), ("RF_raw", oos_rf_raw, raw),
            ("GBM_raw", oos_gbm_raw, raw),
        ]:
            print(f"Computing {name} ...", flush=True)
            predictions[name] = function(indexed, features)

    summary, forecasts = evaluate(predictions, indexed)
    extension = None
    if args.with_extensions:
        print("Computing extension ...", flush=True)
        extended = load_bond_data_extended().set_index("ym")
        extended_factors, _ = load_macro_factors_oos_extended()
        extension_predictions = {"EH": oos_eh(extended)}
        for name, function in [("FB", oos_fb), ("CP", oos_cp)]:
            extension_predictions[name] = function(extended)
        extension_predictions["LN"] = oos_ln(extended, extended_factors)
        extension_predictions["LN_LASSO"] = oos_ln_lasso(extended, extended_factors)
        extension, extension_forecasts = evaluate(extension_predictions, extended)

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "baseline_summary.csv")
    forecasts.to_csv(output / "baseline_forecasts.csv", index_label="ym")
    (output / "baseline_table.md").write_text(markdown_table(summary) + "\n", encoding="utf-8")
    if extension is not None:
        extension.to_csv(output / "extension_summary.csv")
        extension_forecasts.to_csv(output / "extension_forecasts.csv", index_label="ym")
        (output / "extension_table.md").write_text(markdown_table(extension) + "\n", encoding="utf-8")

    files = ["Fama_Bliss_price.txt", "macro_raw_data.xlsx", "SENTIMENT.xlsx"]
    if args.with_extensions:
        files += ["GSW_yields.csv", "2026-04-MD.csv"]
    metadata = {
        "protocol": "original historical specification; no 12-month label embargo",
        "comparison": "identical nonmissing dates across models within each panel",
        "oos_start_requested": str(OOS_START), "macro_series_count": len(raw.columns),
        "gamma": GAMMA_BASE, "weight_cap": W_CAP,
        "assumed_one_way_cost_bps": TRANSACTION_COST_BPS,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "statsmodels", "scikit-learn", "matplotlib")},
        "inputs_sha256": {name: hashlib.sha256((Path(DATA_DIR) / name).read_bytes()).hexdigest()
                          for name in files},
        "bond_price_range": [str(bonds.ym.min()), str(bonds.ym.max())],
        "macro_factor_range": [str(factors.index.min()), str(factors.index.max())],
        "sentiment_feature_range": [str(sentiment.index.min()), str(sentiment.index.max())],
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    render_figures(summary, extension)
    if args.update_readme:
        update_readme(summary, extension, metadata)
    print("\n" + markdown_table(summary))
    if extension is not None:
        print("\nExtension:\n" + markdown_table(extension))
    print(f"\nWrote local tables to {output}; figures to {ROOT / 'assets'}.")


if __name__ == "__main__":
    main()
