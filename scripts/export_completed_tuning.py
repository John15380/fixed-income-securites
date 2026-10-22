"""Publish the completed fixed-budget study without selecting its test winner."""
import argparse
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".cache"/"matplotlib"))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PANELS = ("core_revised", "headline_vintage")
LABELS = {"core_revised": "修订核心 CPI", "headline_vintage": "历史总 CPI", "headline_revised": "修订总 CPI"}


def row(table, **criteria):
    mask = np.ones(len(table), dtype=bool)
    for key, value in criteria.items():
        mask &= table[key].eq(value).to_numpy()
    result = table.loc[mask]
    if len(result) != 1:
        raise ValueError(f"Expected one review row: {criteria}; found {len(result)}")
    return result.iloc[0]


def plot(output):
    scores = pd.read_csv(output/"all_forecast_scores.csv")
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    names = ["CP05", "LN6", "CP15", "Full_OLS", "EN_real_mse", "EN_GC_mse",
             "XGB_real_mse", "XGB_GC_mse", "ModelSelected_mse"]
    for col, tag in enumerate(PANELS):
        ax = axes[0, col]
        q = scores[scores.panel.eq(tag) & scores.horizon.eq(12) & scores.period.eq("test")]
        values = np.array([[row(q, model=name, maturity=str(n)).r2_pct for n in range(2, 6)] for name in names])
        ax.imshow(values, cmap="RdBu", vmin=-40, vmax=40, aspect="auto")
        ax.set_yticks(range(len(names)), names)
        ax.set_xticks(range(4), ["2y", "3y", "4y", "5y"])
        ax.set_title(tag+": annual forecast OOS R2 (%)")
        for r in range(len(names)):
            for c in range(4):
                v = values[r, c]
                ax.text(c, r, f"{v:.1f}" if abs(v) < 1000 else f"{v:.1e}", ha="center", va="center",
                        fontsize=8, color="white" if abs(v) > 20 else "#172b37")
        ax = axes[1, col]
        trades = pd.read_csv(output/tag/"h1"/"economic_trades.csv")
        for name, label, color in [("ModelSelected/cal", "Early-selected model + allocation calibration", "#146a6d"),
                                   ("CP15/cal", "CP15 + same allocation calibration", "#bf7b28"),
                                   ("EH/cal", "Equally calibrated historical mean", "#687983"),
                                   ("PriorEW3", "Previously retained EW3", "#8062a8"),
                                   ("BuyHold", "Passive basket", "#b64f5d")]:
            g = trades[trades.model.eq(name) & trades.purchase.between("2017-01", "2022-11")].sort_values("purchase")
            if g.empty and name == "PriorEW3":
                continue
            dates = pd.PeriodIndex([g.purchase.iloc[0], *g.sale.tolist()], freq="M").to_timestamp()
            ax.plot(dates, np.r_[1., np.cumprod(1+g.return_net)], color=color, label=label, linewidth=1.7)
        ax.set_title(tag+": monthly net wealth, 71 trades")
        ax.set_ylabel("Wealth (initial = 1)")
        ax.legend(fontsize=7.5, frameon=False)
        ax.grid(alpha=.16)
    fig.suptitle("Expanded search: 768 trials, two seeds, separate investment validation\nAnnual forecasts and monthly investment models have different targets; heatmap colors clipped at +/-40%", fontsize=11)
    fig.savefig(ROOT/"assets"/"completed_tuning.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), constrained_layout=True)
    colors = {"EN_real": "#ac7b34", "EN_GC": "#b95069", "XGB_real": "#436c9c", "XGB_GC": "#14847d"}
    for ax, tag in zip(axes, PANELS):
        trials = pd.read_csv(output/tag/"h1"/"trials.csv")
        for name, color in colors.items():
            for seed, linestyle in ((42, "-"), (73, "--")):
                g = trials[trials.model.eq(name) & trials.seed.eq(seed)].sort_values("number")
                ax.plot(g.number+1, g.value.cummin(), linestyle=linestyle, color=color,
                        label=f"{name}, seed {seed}", linewidth=1.4)
        ax.set_ylim(0, min(2., float(trials.groupby(["model", "seed"]).value.first().max())*1.1))
        ax.axhline(1., color="#777", linewidth=.7)
        ax.set_title(tag+": MONTHLY early-validation loss")
        ax.set_xlabel("Trial per seed and learner group")
        ax.set_ylabel("MSE / common EH MSE (lower is better)")
        ax.legend(ncol=2, fontsize=7, frameon=False)
        ax.grid(alpha=.15)
    fig.savefig(ROOT/"assets"/"tuning_search.png", dpi=170)
    plt.close(fig)


def report(output):
    scores, performance = (pd.read_csv(output/name) for name in ("all_forecast_scores.csv", "all_performance.csv"))
    utility = pd.read_csv(output/"all_utility.csv")
    lines = ['<a id="completed-tuning"></a>', "### 5.7 扩大调参与独立投资校准：补齐后的研究", "",
        "本节保留 §5.6 小预算实验作为对照，纠正其搜索范围窄、输入过度压缩、没有独立经济校准的问题。",
        "先重新执行 FB/CP/LN 等 **41 项原论文数值校验**，再运行四组独立搜索（两个通胀面板 × 一月/一年目标），",
        "共 **768 个 Optuna 候选**：每组 EN 64、XGBoost 128；真实训练与 Copula 各分一半，",
        "各再用种子 42/73 独立搜索。两个候选同时计算、按固定顺序反馈采样器，保留 trial 全表和搜索轨迹。",
        "这是明确预算的历史研究，不能将次数增加称为全局最优或证明调参已经收敛。", "",
        "**三个信息时段分开。** 1994 年起训练；2011 年起的早段验证只使用严格在 2014-01 前兑现的标签，",
        "选择超参数、特征表示、窗口、Copula 比例和模型族。年目标仅有 24 个重叠验证月份，月目标 35 个，",
        "有效样本仍短。2014 年起的晚段另选预测 MSE 收缩；月目标还用 2014-01–2016-11 共 35 笔交易校准投资信号，",
        "分为 18/17 月两段。所有选择在 2017-01 前冻结；回归系数和特征预处理仍按当时可得历史逐月重估。",
        "主评价月交易买入期 2017-01–2022-11（71 笔，最后 2022-12 卖出）；一年目标买入期 2017-01–2021-12（60 个重叠标签）。",
        "后续月交易为 2022-12–2025-11（36 笔，最后 2025-12 卖出）。这些历史期间此前已看过，**不是新鲜、未触碰的测试集**。", "",
        "| 搜索维度 | 本轮范围 |", "| --- | --- |",
        "| 特征表示 | 原 11 个指数/PC；32 个原语；按成熟训练行 1%/99% 截断的 32 个原语 |",
        "| 原语构造 | 五远期利率、八宏观 PC、F1³、一年/2–20 年均值周期、3/6/12 半衰期远期偏差共 15 项、趋势通胀 |",
        "| 训练窗口 | 1994 起扩展、120 月、240 月；起点限制下未满 240 月时使用已有历史 |",
        "| Copula | 0、0.25、0.5、1、2 倍训练行；合成 X/Y 只来自成熟训练集，不计为新增独立观测 |",
        "| ElasticNet | α=10⁻⁴–10，l1_ratio=0.02–1 |",
        "| XGBoost | 25–600 棵树、深度 1–5、学习率 0.005–0.2、child weight 1–80、L2 0.01–1000、L1 10⁻⁶–10、行/列抽样 0.6–1 |", "",
        "32 原语保留非线性模型可利用的交互信息。F1 三次项只将当前 F1 限于成熟训练极值后求立方，",
        "bounded32 再对所有列使用训练分位点；原始 LN6 对照不加该保护，避免把修改冒充原文。",
        "所有候选共用扩展 EH 的预测误差分母，窗口之间能够公平比较。XGBoost 使用每目标一组树的多输出接口，",
        "行列抽样下随机流与旧的四次独立拟合不同，因此明确作为新配置。", "",
        "| 面板/目标 | 早验证选中族 | 表示 | 窗口 | GC 倍数 | 早验证平均 R² (%) | 晚段 MSE 收缩 λ |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for tag in PANELS:
        for horizon in (12, 1):
            frozen = json.loads((output/tag/f"h{horizon}"/"frozen_parameters.json").read_text())
            name = frozen["selected_family"]; config = frozen[name]
            lines.append(f"| {LABELS[tag]}/{horizon}月 | {name} | {config['representation']} | {config['window']} | {config['gc_ratio']:g} | {100*(1-config['selection']['objective']):.2f} | {frozen['mse_shrinkage']['ModelSelected']:.3f} |")
    lines += ["", "| 面板/目标的入选族 | seed42 早验证 R² (%) | seed73 | 入选参数的边界提示 |",
              "| --- | --- | --- | --- |"]
    for tag in PANELS:
        for horizon in (12, 1):
            folder = output/tag/f"h{horizon}"
            f = json.loads((folder/"frozen_parameters.json").read_text())
            d = json.loads((folder/"search_diagnostics.json").read_text())[f["selected_family"]]
            seeds = d["seeds"]
            chosen = next(s for s in seeds if s["seed"] == f[f["selected_family"]]["selection"]["seed"])
            bounds = ", ".join(f"{k}({v['edge']})" for k, v in chosen["boundary_parameters"].items()) or "无连续参数近边界提示"
            lines.append(f"| {LABELS[tag]}/{horizon}月 {f['selected_family']} | {100*(1-seeds[0]['best_value']):.2f} | {100*(1-seeds[1]['best_value']):.2f} | {bounds} |")
    lines += ["", "种子间差异及近边界参数意味着仍有搜索不确定性；不根据测试期排名继续扩大边界。",
              "落选的部分弱正则 EN 候选出现收敛警告。详细正则/树参数及复查文件保存在本地结果目录。"]
    solver_path = output/"selected_en_convergence.json"
    if solver_path.exists():
        solver = json.loads(solver_path.read_text())
        items = solver["models"]
        sections = [m[stage] for m in items for stage in ("early", "later")]
        folds = sum(s["folds"] for s in sections)
        warnings = sum(s["default_warning_count"] for s in sections)
        strict_warnings = sum(s["strict_warning_count"] for s in sections)
        delta = max(s["max_abs_prediction_difference"] for s in sections)
        lines += [f"另对 {len(items)} 个 EN 族获胜配置的 {folds:,} 个预测折复查，默认拟合警告数 {warnings}；",
                  f"对全部早验证折和后期抽样折提高迭代精度，警告数 {strict_warnings}，最大预测差 {100*delta:.4f}bp。",
                  "这支持获胜 EN 的结果不由上述数值警告解释；`scripts/audit_tuning_solver.py` 可独立重跑。"]
    lines += ["", "**投资校准不再等同于 MSE 收缩。** 固定 γ=5、无杠杆多头上限 1、60 月成熟风险估计与单边 0.5bps。",
        "使用 $\mu^{cal}=\mu^{EH}+a(\mu^{model}-\mu^{EH})+b\sigma_t$，其中 a=0–2（步长 0.25）、",
        "b=−0.2–0.2（步长 0.05），共 81 组。按两段净总收益的年化月 CER 增益，",
        "取 0.5×均值＋0.5×较差段来选，抑制收益只来自其中一段的配置。",
        "**EH 也使用相同校准流程**，因此同时报告 EH/cal，避免把简单减仓或现金利息全归为模型信息。",
        "修订总 CPI 面板沿用历史总 CPI 的预测参数、MSE 收缩与投资校准参数，版本对照不重新搜索。", "",
        "| 面板 | 一年预测模型 | 2y R² (%) | 3y | 4y | 5y | 篮子 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for tag in PANELS:
        q = scores[scores.panel.eq(tag) & scores.horizon.eq(12) & scores.period.eq("test")]
        for name in ("CP05", "LN6", "CP15", "EN_real_mse", "EN_GC_mse", "XGB_real_mse", "XGB_GC_mse", "ModelSelected_mse"):
            values = [row(q, model=name, maturity=str(n)).r2_pct for n in (2, 3, 4, 5, "basket")]
            lines.append("| "+" | ".join([LABELS[tag], name, *[f"{v:.2f}" for v in values]])+" |")
    lines += ["", "原始 LN6 的极大负 R² 是未经保护的 F1³ 在疫情宏观极值上的外推失败；",
        "本期两个面板的 LN6 篮子预测平方误差均有超过 99.9% 集中在 2020 年买入月，最大单月为 2020-06。",
        "R² 下界不受零或 −100% 限制。作者样本内公式/数值校验通过，不能保证换数据并递归估计后的样本外稳定性；",
        "本轮保留原始规格展示这一风险，同时让带训练边界保护的 32 维表示参加验证选择。", "",
        "| 主评价面板 | 月度策略 | 扣费 CAGR (%) | 算术年均超额 (%) | 夏普 | 月末回撤 (%) | 平均债券敞口 |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    names = ("ModelSelected/default", "ModelSelected/cal", "CP15/cal", "XGB_GC/cal", "EH/default", "EH/cal", "PriorEW3", "BuyHold")
    for tag in PANELS:
        q = performance[performance.panel.eq(tag) & performance.period.eq("test")]
        for name in names:
            if name == "PriorEW3" and not q.model.eq(name).any():
                continue
            r = row(q, model=name)
            lines.append(f"| {LABELS[tag]} | {name} | {r.cagr_pct:.2f} | {r.mean_excess_pct:.2f} | {r.sharpe:.2f} | {r.max_dd_pct:.2f} | {r.mean_gross_exposure:.2f} |")
    lines += ["", "| 面板 | 早段选中模型/cal | a | b | 对 EH/cal 的年化月 CER 差 (bps) | 配对 95% 区间 | 后续 CAGR / 超额夏普 |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
    for tag in PANELS:
        frozen = json.loads((output/tag/"h1"/"frozen_parameters.json").read_text())
        alloc = json.loads((output/tag/"h1"/"allocation_parameters.json").read_text())["parameters"]["ModelSelected"]
        u = row(utility, panel=tag, period="test", model="ModelSelected/cal", benchmark="EH/cal")
        later = row(performance, panel=tag, period="followup", model="ModelSelected/cal")
        lines.append(f"| {LABELS[tag]} | {frozen['selected_family']} | {alloc['signal_strength']:g} | {alloc['bias_sigma']:g} | {u.utility_gain_bps:.1f} | [{u.ci_low_bps:.1f}, {u.ci_high_bps:.1f}] | {later.cagr_pct:.2f}% / {later.sharpe:.2f} |")
    lines += ["", "| 面板 | 选中模型/cal 的固定参数压力场景 | CAGR (%) | 超额夏普 | 月末回撤 (%) |",
              "| --- | --- | --- | --- | --- |"]
    for tag in PANELS:
        stress = pd.read_csv(output/tag/"h1"/"stress.csv")
        for scenario in ("cost2", "cost5", "gamma2", "gamma10", "delay1"):
            r = row(stress, period="test", model="ModelSelected/cal", scenario=scenario)
            lines.append(f"| {LABELS[tag]} | {scenario} | {r.cagr_pct:.2f} | {r.sharpe:.2f} | {r.max_dd_pct:.2f} |")
    lines += ["", "收益口径与截图可逐项核对：CAGR 复利净总收益；算术年均超额为 12×月净收益减现金收益的均值；",
        "累计超额可按月差相加，不能称作累计复利总收益。夏普使用月超额均值/标准差再乘 √12；",
        "一年重叠预测绝不当作月收益复利。费用同时覆盖买入和卖出，入场预留费用、现金余额非负。",
        "以上为 GSW 拟合曲线的理论零息券回测，未验证实券流动性、成交价及规模容量。", "",
        "完整结果保存全部原始/收缩预测、所有模型的默认/校准账户、EH 公平校准、2/5bps、γ=2/10、",
        "信号延迟一月及分段表现。预测表保留 HAC/全表 Holm 与篮子联合 max-t；经济效用区间为",
        "条件于既有预测的配对 12 月区块 bootstrap，未重估全部历史因子或校正此前搜索，因此不能据此承诺稳定 alpha。", "",
        "![扩大搜索后的年度预测与月度投资](assets/completed_tuning.png)", "",
        "![两种子早段验证搜索轨迹](assets/tuning_search.png)", "",
        "**完成后的判断。** 扩大搜索后，四组早验证均选中 XGB_real，未选 Copula 组；这只说明当前协议下的选择，",
        "不能推成合成数据通常无效。主期一年预测的所选模型仍未稳定超过 EH；CP15 对照保留正 R²，",
        "本轮复杂模型的稳定超额优势仍未获确认。", ""]
    for tag in PANELS:
        q = performance[performance.panel.eq(tag) & performance.period.eq("test")]
        selected = row(q, model="ModelSelected/cal")
        default = row(q, model="ModelSelected/default")
        bench = row(q, model="EH/cal")
        prior = q[q.model.eq("PriorEW3")]
        comparison = f"；既有 EW3 同期 {prior.iloc[0].cagr_pct:.2f}%" if len(prior) else ""
        lines.append(f"{LABELS[tag]}月度所选模型默认/校准后 CAGR 为 {default.cagr_pct:.2f}%/{selected.cagr_pct:.2f}%，"
                     f"同期公平 EH 为 {bench.cagr_pct:.2f}%{comparison}。")
    lines += ["扩大调参解决了原先搜索不足的实施缺项，但本轮结果未支持其超过既有 EW3。",
        "额外经济校准在两个面板的效果也不一致，35 月晚段验证、仓位截断和市场状态变化都有明显局限。",
        "因此保留 EW3 作为历史投资候选，保留 CP15 的文献预测证据，完整展示非线性方法及校准失败的情形。",
        "后续 CPI/时期的局部高收益须同时核对现金利息和债券敞口；全现金的正总收益不计作模型 alpha。",
        "本节同时改变特征、搜索范围和验证分段，与 §5.6 的差异不能单独归因于 trial 数。",
        "截图中的调查预期分歧 ME 因子数据和完整 MS17 无套利 DTSM 仍未纳入本仓库；平滑远期利率只是明确披露的扩展。",
        "因此可以复核上述方法链与本仓库结果，尚未逐项复现另一组未提供代码和数据口径的高收益数字。", "",
        "复现：`uv sync --extra research --extra report`，准备 README §7 的原始数据并运行文献输入下载脚本后，执行：", "",
        "```bash", "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/complete_tuning.py",
        "python scripts/audit_completed_tuning.py", "python scripts/audit_tuning_solver.py",
        "python scripts/export_completed_tuning.py --update-readme", "```", "",
        "`output/completed_tuning/` 保留本轮协议、768 trial、两种子与边界诊断、成熟截止日、参数、预测、",
        "81 点经济校准表及完整现金流。数据和中间结果不上传，README 表格及图片由保存结果生成。",
        "工具实现参考 [Optuna 的可重复性说明](https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-obtain-reproducible-optimization-results)",
        "和 [XGBoost 多目标接口](https://xgboost.readthedocs.io/en/stable/tutorials/multioutput.html)。"]
    body = "\n".join(lines)+"\n"
    (output/"report.md").write_text(body)
    return body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"output"/"completed_tuning")
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()
    # Refuse to publish stale source/data fingerprints or unaudited cash flows.
    from audit_completed_tuning import audit
    audit(args.output_dir)
    plot(args.output_dir)
    body = report(args.output_dir)
    if args.update_readme:
        path = ROOT/"README.md"; text = path.read_text()
        start, end = "<!-- COMPLETED_TUNING:START -->", "<!-- COMPLETED_TUNING:END -->"
        block = start+"\n"+body+end+"\n\n"
        if start in text:
            a, b = text.index(start), text.index(end)+len(end)
            text = text[:a]+block.rstrip()+text[b:]
        else:
            location = text.index("## 6.")
            text = text[:location]+block+text[location:]
        path.write_text(text)
    print("Exported completed tuning tables and figures", flush=True)


if __name__ == "__main__":
    main()
