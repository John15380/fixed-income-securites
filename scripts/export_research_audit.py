"""Build readable evidence and figures for the six specification questions."""

import argparse
from itertools import combinations
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".cache"/"matplotlib"))
import numpy as np
import pandas as pd
from bond_rp.robustness import FAMILIES, LOOKBACKS


def md(headers, rows):
    def fmt(x):
        if isinstance(x, (float, np.floating)):
            return f"{x:.2f}" if np.isfinite(x) else "—"
        return str(x)
    return "\n".join(["| "+" | ".join(headers)+" |", "| "+" | ".join(["---"]*len(headers))+" |",
                      *["| "+" | ".join(map(fmt, row))+" |" for row in rows]])


def load_tables(folder):
    return {tag: {p.stem: pd.read_csv(p) for p in (folder/tag).glob("*.csv")
                  if not p.name.startswith(("predictions", "research_predictions", "annual_trades", "trades_", "selector_"))}
            for tag in ("baseline", "gsw", "vintages")}


def plot(tables):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    assets = ROOT/"assets"
    assets.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), layout="constrained")
    for ax, tag in zip(axes[0], ("baseline", "gsw")):
        f = tables[tag]["forecast_summary"].set_index("model")
        labels = ["+".join(c) for k in range(1, 5) for c in combinations(FAMILIES, k)]
        values = np.array([[f.loc[label+suffix, "r2_pct"] for suffix in ("", "_R10", "_R100")] for label in labels])
        im = ax.imshow(values, cmap="RdBu", vmin=-60, vmax=60, aspect="auto")
        ax.set_xticks(range(3), ["OLS", "Ridge 10", "Ridge 100"])
        ax.set_yticks(range(len(labels)), labels, fontsize=9)
        ax.set_title(f"{'FB + JLN' if tag == 'baseline' else 'GSW + FRED-MD'}: OOS R2 (%)")
        for (r, c), v in np.ndenumerate(values):
            ax.text(c, r, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 40 else "black")
    fig.colorbar(im, ax=list(axes[0]), shrink=.8)
    for tag, color in [("baseline", "#bb714b"), ("gsw", "#367a91")]:
        f = tables[tag]["forecast_summary"].set_index("model")
        axes[1, 0].plot(LOOKBACKS, [f.loc[f"MA{k}", "r2_pct"] for k in LOOKBACKS], marker="o", label=tag, color=color)
    axes[1, 0].axhline(0, color="grey", linewidth=.7)
    axes[1, 0].set(xlabel="Trailing moving-average months", ylabel="OOS R2 (%)", title="All tested MA windows")
    axes[1, 0].legend(frameon=False)
    g = tables["gsw"]["risk_grid"]
    p = g[g.model == "LN"].pivot(index="gamma", columns="cap", values="sharpe")
    ax = axes[1, 1]
    im = ax.imshow(p, vmin=.3, vmax=.6, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(p.columns)), p.columns)
    ax.set_yticks(range(len(p.index)), p.index)
    ax.set(xlabel="Gross exposure cap", ylabel="Risk aversion gamma", title="GSW LN: net annual Sharpe (January)")
    for (r, c), v in np.ndenumerate(p.to_numpy()):
        ax.text(c, r, f"{v:.2f}", ha="center", va="center", color="white" if v > .5 else "black")
    fig.savefig(assets/"specification_audit.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    f = tables["gsw"]["policy_summary"]
    pairs = [("BuyHold", "scalar", "Passive basket"), ("LN", "scalar", "LN: MV, cap 3"),
             ("LN", "long_only", "LN: long only, cap 1"), ("EW4", "vol_target", "EW4: 4% vol, cap 1"),
             ("EW_mature", "vol_target", "Online EW: 4% vol, cap 1"), ("TrendVote", "direct", "Six-window trend vote")]
    colors = ["#777777", "#367a91", "#5d9d89", "#bb714b", "#956baa", "#a7a24c"]
    for i, (model, policy, label) in enumerate(pairs):
        row = f[(f.model == model) & (f.policy == policy)].iloc[0]
        axes[0].barh(i, row.sharpe, color=colors[i])
        axes[0].text(row.sharpe+.007, i, f"{row.sharpe:.2f}", va="center")
    axes[0].set_yticks(range(len(pairs)), [p[2] for p in pairs])
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Net annual excess Sharpe", title="GSW: 35 non-overlapping annual trades", xlim=(0, .65))
    for i, (model, policy, label) in enumerate(pairs[1:5], start=1):
        c = tables["gsw"]["sharpe_intervals"]
        row = c[(c.model == model) & (c.policy == policy)].iloc[0]
        axes[1].plot([row.ci_low, row.ci_high], [i, i], color=colors[i], linewidth=3)
        axes[1].scatter([row.delta_sharpe], [i], color=colors[i], zorder=3)
    axes[1].axvline(0, color="grey", linestyle="--")
    axes[1].set_yticks(range(1, 5), [p[2] for p in pairs[1:5]])
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Sharpe improvement vs passive basket", title="Paired 5-year block bootstrap: 95% intervals")
    fig.savefig(assets/"strategy_research.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    f = tables["vintages"]["forecast_summary"]
    names = ["TI", "LN", "LN_R100", "EW3", "EW_mature"]
    width = .25
    for j, (source, label, color) in enumerate([("revised_full", "2026-04 panel", "#bbbbbb"),
                                                ("revised_matched", "Revised, matched names", "#bb714b"),
                                                ("vintage", "Prior-month vintage", "#367a91")]):
        g = f[(f.source == source) & (f.lag == 2)].set_index("model")
        axes[0].bar(np.arange(len(names))+(j-1)*width, g.loc[names, "r2_pct"], width, label=label, color=color)
    axes[0].set_xticks(range(len(names)), names, rotation=20)
    axes[0].axhline(0, color="grey", linewidth=.7)
    axes[0].set(ylabel="OOS R2 (%)", title="Macro-vintage experiment: two-month reference lag")
    axes[0].legend(frameon=False, fontsize=8)
    policy = tables["vintages"]["policy_summary"]
    names2 = ["LN", "LN_R100", "EW3", "EW_mature", "BuyHold"]
    for j, (method, label, color) in enumerate([("scalar", "MV, cap 3", "#367a91"),
                                               ("long_only", "Long only, cap 1", "#5d9d89"),
                                               ("vol_target", "4% vol budget, cap 1", "#bb714b")]):
        g = policy[(policy.source == "vintage") & (policy.lag == 2) & (policy.policy == method)].set_index("model")
        axes[1].bar(np.arange(len(names2))+(j-1)*width, g.loc[names2, "sharpe"], width, label=label, color=color)
    axes[1].set_xticks(range(len(names2)), names2, rotation=20)
    axes[1].set(ylabel="Net annual excess Sharpe", title="Macro-vintage portfolios; same dates and costs")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(assets/"realtime_audit.png", dpi=170)
    plt.close(fig)


def report(tables):
    b, g, v = (tables[t] for t in ("baseline", "gsw", "vintages"))
    scores = {tag: tables[tag]["forecast_summary"].set_index("model") for tag in ("baseline", "gsw")}
    subsets = ["+".join(c) for k in range(1, 5) for c in combinations(FAMILIES, k)]
    family_rows = [[s, *[scores[tag].loc[s+suffix, "r2_pct"] for tag in ("baseline", "gsw")
                         for suffix in ("", "_R10", "_R100")]] for s in subsets]
    sections = ["""<a id="research-audit"></a>

### 5.1 六项质疑的补充审计与策略探索（2026-10-22）

**先纠正研究表述：不能由现有结果得出“没有好的投资策略”，也不能把历史赢家当成已经确认的策略。** 本轮扩大规格空间、加入递归选择和宏观历史版本检验，保留所有失败方案。新增结果属于已经看过历史数据后的探索；计算过程的无前视约束不能消除研究者的事后选择。

为控制比较，新增因子审计中的所有规格使用四类因子均可用的共同训练月份；1990 年首期为 280 个已兑现标签。因此这里的 EH、CP、LN 等数字与上方按各模型可用历史拟合的基础报告略有差异，两个实验分别保留，不混用。预测评价仍是月度一年期标签，投资评价仍是扣费、不重叠的年度交易。

**① 为什么 LN+CP、LN+BW？** 经济假设是分别结合曲线中的定价信息、宏观状态和情绪代理，但没有理论保证这两组最优。原先仅测试两组不足以支持选择。现在穷举 CP、LN、BW、TI 的全部 15 个非空子集，并对每组报告 OLS、Ridge α=10、100；惩罚施加在训练窗口标准化后的回归变量上，8 个宏观 PC 每期重新估计。

下表是各规格相对共同 EH 的 OOS R²（%），起点均为 1990-01；FB+JLN 截止 2012-02，GSW+FRED 截止 2024-02。Ridge 网格也是本次研究选择，不称为预注册。
""", md(["因子子集", "FB OLS", "FB R10", "FB R100", "GSW OLS", "GSW R10", "GSW R100"], family_rows)]
    sections.append("""短样本中 LN+BW 及其收缩版本较强，延长到 GSW 样本后这种优势消失；加入更多因子也没有稳定改善。因而不再将 LN+CP 或 LN+BW 作为有充分证据支持的唯一选择。

另外比较四个单因子预测等权的 EW4，以及两个只能使用已兑现历史误差的算法：`EW_mature` 在最近 120 个买入月份中，按已兑现误差的 MSE 倒数分配预测权重；`Select_mature` 选其中历史 MSE 最小的规格。候选涵盖 EH、45 个子集/惩罚规格、六个均线模型与多窗口模型，共 53 个。至少累计 60 个已兑现误差后才启动；此前使用 EH。每个误差也满足 s+12<t，权重、选中模型和信息截止日均留档。它们是实际可递归执行的选择规则，不能用事后最好的模型替代其真实表现。

**② 为什么只用 12 期均线？** 原先没有证明 12 期优于其他周期。现在完整比较 3、6、9、12、18、24 期，另将六个指标联合输入 Ridge(100)，并让上述历史选择器有机会选择各周期。指标始终是各期限对数价格减去含当期在内的过去 k 月均值，再跨期限平均；没有用未来价格。
""")
    sections.append(md(["技术窗口", "FB OOS R² (%)", "GSW OOS R² (%)"],
                       [[f"MA{k}", scores["baseline"].loc[f"MA{k}", "r2_pct"], scores["gsw"].loc[f"MA{k}", "r2_pct"]] for k in LOOKBACKS]+
                       [["六窗口 Ridge(100)", scores["baseline"].loc["MA_multi_R100", "r2_pct"], scores["gsw"].loc["MA_multi_R100", "r2_pct"]]]))
    sections.append("12 期在这组 GSW 历史结果中相对较好，但仍是事后观察，不能称为唯一正确周期；正 R² 的技术规格在扩大比较后的 Holm 校正下没有达到 5% 显著性。")
    sections.append("""**③ 换 1985、1995、2000 年是否成立？** 对无调参状态的逐月扩展 OLS/Ridge，给定同一预测月份，改变评价起点不会改变训练规则或拟合；它们可直接使用 1985 年起逐期生成的因果预测。两个递归选择器则在每个起点重新开始累计误差，不沿用更早测试期的选择记录。训练起点始终为 1964 年，未将“更换测试起点”误写成改变训练历史。

下列值为 OOS R²（%）；每张表内部的不同起点使用相同结束月，故改变的是评价时期及其历史均值误差分母。它们是嵌套、相关的历史窗口。
""")
    for tag, label in [("baseline", "FB+JLN（结束 2012-02）"), ("gsw", "GSW+FRED（结束 2024-02）")]:
        p = tables[tag]["start_sensitivity"].pivot(index="model", columns="origin", values="r2_pct")
        names = ["CP", "LN", "CP+LN", "LN+BW", "TI", "EW4", "EW_mature", "Select_mature"]
        sections.extend([f"{label}：", md(["模型", "1985 起", "1990 起", "1995 起", "2000 起"], [[n, *p.loc[n]] for n in names])])
    sections.append("结论随时期变化：不能说原组合在所有起点都成立。TI 的点估计相对稳定，而高维联合 OLS 和按历史误差选单一模型并未自动解决不稳定性。")
    risk = g["risk_grid"]
    rr = []
    for name in ["EH", "LN", "EW4", "EW_mature", "Select_mature"]:
        f = risk[risk.model == name]
        rr.append([name, f.sharpe.min(), f.sharpe.max(), f.cagr_pct.min(), f.cagr_pct.max()])
    sections.append("""**④ 为什么 γ=5、杠杆上限 3？** γ 是均值—方差效用中的风险厌恶偏好，上限是投资约束，二者都不是收益预测模型估计出的最佳值。原先没有证据证明 5 和 3 特别合适；现保留它们作为与基础报告对照的规格，另完整检验 γ∈{1,2,3,5,10,20}、cap∈{1,1.5,2,3} 的 24 个组合，不按夏普挑出“最优参数”。不同 γ 下的 CER 对应不同投资者，不能跨 γ 排名后选赢家。

GSW、1990 年起、固定 1 月入场的敏感性范围如下；被动篮子夏普为 0.40。范围仅覆盖所列网格，尚不意味着每个时期或建仓月份都优于被动。
""")
    sections.append(md(["模型", "夏普最小", "夏普最大", "CAGR 最小 (%)", "CAGR 最大 (%)"], rr))
    sections.append("""同时增加更容易解释的投资约束：`long_only` 将仓位限制在 [0,1]，剩余为一年期国债，不借款也不做空；`vol_target` 只用预测正负决定是否持有，再按“风险预算÷已兑现历史篮子波动”配置，上限为 1。2%、4%、6%、8% 的波动预算全部保存；4% 是展示用的圆整数风险预算，不能称为校准出的最优值。这里估计的是风险资产的超额收益波动，最终账户收益波动不一定等于预算。

![因子组合、均线周期与风险参数的完整审计](assets/specification_audit.png)

**⑤ 是否处理实时版本和发布滞后？** 基础报告仅做了 1/2/3 个月参考期滞后，没有消除修订偏差。本轮新增读取[圣路易斯联储官方历史 FRED-MD](https://www.stlouisfed.org/research/economists/mccracken/fred-databases) 两个 ZIP：1999-08–2014-12、2015-01–2025-12。对月份 t，只使用 t−1 月的历史文件，读取该版本自己的变量与变换码，并重估其完整已知历史的填补、标准化、PCA 和回归。当前变量使用 t−lag 参考月；尾部零散缺失用训练均值处理，整个当期宏观行都不存在时拒绝生成宏观预测。

历史版本标签是月份，未取得逐系列的精确下载/发布日期日志，所以额外延迟一个版本月，保留保守的月度时点约定。[ALFRED 关于 observation date 与 real-time period 的说明](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html)解释了为何参考月份与当时可见版本必须分开。

三个比较对象为：最新版完整面板、最新版中与当期历史文件同名的列、历史版本自身面板。后两者仍可能存在定义/变换码差别；因此差异包含修订、变量构成和可得性的共同影响，不能全部归因于数值修订。历史版本实验不使用 BW，因此不受 2023-12 情绪截止限制，使用全部已兑现债券标签。
""")
    vf = v["forecast_summary"]
    sample = vf[(vf.source == "vintage") & (vf.lag == 2)].iloc[0]
    sections.append(f"宏观历史版本的共同评价期为 **{sample.start}–{sample.end}，{int(sample.n)} 个月**。下表为 OOS R²（%）：")
    rows = []
    for lag in (1, 2, 3):
        for name in ["LN", "LN_R100", "EW3", "EW_mature"]:
            row = [lag, name]
            for source in ("revised_full", "revised_matched", "vintage"):
                s = vf[(vf.source == source) & (vf.lag == lag) & (vf.model == name)]
                row.append(float(s.r2_pct.iloc[0]) if len(s) else np.nan)
            rows.append(row)
    sections.append(md(["参考期滞后", "模型", "新版完整", "新版同名列", "历史版本"], rows))
    sections.append(r"""在上述保守版本约定下，lag=1 时所有 t−1 参考月的宏观行均尚未存在，因此表中的“—”是不可估计，不能解释为零 R²。lag=2/3 使用实际已有数值，未用新版数据回填历史文件的缺失尾部。

**仍未完全修复的边界：** 1990 年起的全样本没有相应的完整月度宏观版本；JLN 与 BW 仍是后验版本，GSW 也使用当前下载的拟合曲线，尚未做债券曲线估计的历史版本重建。因此这里只能称为“宏观历史版本检验”，不能将整个交易系统称为严格 point-in-time。精确发布日期、情绪版本和可交易价格仍需另行补齐。

![宏观历史版本与投资表现](assets/realtime_audit.png)

**⑥ 为什么 OOS R² 可能高于 IS R²？** 两个常见量的分母不同：

$$R^2_{IS}=1-\frac{SSE_{fit,train}}{\sum_{train}(x-\bar x_{train})^2},\qquad
R^2_{OOS,EH}=1-\frac{SSE_{recursive,test}}{\sum_{test}(x-\widehat x_{EH,t})^2}.$$

OLS 的训练拟合最优性质只针对相同样本、相同设计矩阵和同一固定系数问题，不对不同历史时期和不同基准的 R² 给出大小关系。现在同时计算初始训练拟合 R²、递归预测相对 EH 的 R²，以及把递归预测误差放到测试期样本均值分母后的 R²。最后一种均值使用了整个测试期，是诊断基准，不能交易。另保存对测试期事后重新拟合 OLS 的 R²，明确不作为样本外业绩。
""")
    rows = []
    for tag, label in [("baseline", "FB"), ("gsw", "GSW")]:
        f = tables[tag]["r2_decomposition"].set_index("model")
        for name in ["LN", "LN+BW", "TI"]:
            r = f.loc[name]
            rows.append([label, name, r.initial_train_r2_pct, r.oos_r2_eh_pct, r.oos_r2_test_mean_pct])
    sections.append(md(["数据", "模型", "初始 IS R² (%)", "OOS/EH R² (%)", "OOS/测试均值 R² (%)"], rows))
    r = g["r2_decomposition"].set_index("model").loc["TI"]
    sections.append(f"以 GSW/TI 为例：预测 MSE={r.model_mse:.3f}，EH 的 MSE={r.eh_mse:.3f}，测试期样本均值的 MSE={r.test_mean_mse:.3f}。所以相对 EH 是 +{r.oos_r2_eh_pct:.2f}%，换成同一测试期均值则为 {r.oos_r2_test_mean_pct:.2f}%。这里高于初始 IS 的现象主要反映不同分母和时期，不能作为泛化超强的证据。旧版前视造成的异常也已通过标签隔离和训练期内 PCA 修正。")
    sections.append("""**进一步探索：有哪些值得保留的策略？** 将预测误差、经济效用和风险调整表现分开判断。LN 的幅度预测误差较大，并不排除其符号、截断后的仓位在部分年度具有价值；反过来，低预测 MSE 也不保证年度交易获利。这里报告实际产生的扣费仓位收益，不用预测 R² 替代投资回测。

下表为 GSW、1990-01 买入至 2025-01 卖出、35 笔年度交易；费用与基础报告相同。EW4 是 CP/LN/BW/TI 预测等权，不同于前面的含 GBM 的 Ensemble；`TrendVote` 以六种均线为正的比例直接作为篮子仓位，不拟合未来收益。
""")
    ps = g["policy_summary"]
    pairs = [("BuyHold", "scalar", "被动篮子"), ("LN", "scalar", "LN，γ5/cap3"),
             ("LN", "long_only", "LN，仅多头/cap1"), ("EW4", "vol_target", "EW4，4%预算/cap1"),
             ("EW_mature", "vol_target", "历史误差加权，4%/cap1"), ("Select_mature", "scalar", "历史误差选模，γ5/cap3"),
             ("TrendVote", "direct", "六均线投票/cap1")]
    rows = []
    for model, policy, label in pairs:
        r = ps[(ps.model == model) & (ps.policy == policy)].iloc[0]
        rows.append([label, r.cagr_pct, r.sharpe, r.max_dd_pct])
    sections.append(md(["配置规则", "净 CAGR (%)", "超额夏普", "年度端点回撤 (%)"], rows))
    sections.append("为了避免只展示 1990 年这个起点，还检查了候选的年度投资结果。GSW 固定 1 月入场的夏普如下（共同结束于 2025-01）：")
    start_rows = []
    for year in ("1985", "1990", "1995", "2000"):
        scalar = g[f"annual_all_{year}"].set_index("model")
        long_only = g[f"annual_long_only_{year}"].set_index("model")
        start_rows.append([year, scalar.loc["LN", "sharpe"], long_only.loc["LN", "sharpe"], scalar.loc["BuyHold", "sharpe"]])
    sections.append(md(["起始买入年", "LN γ5/cap3", "LN 仅多头/cap1", "被动篮子"], start_rows))
    sections.append("**1985 年起的窗口中，两个 LN 规则均未超过被动夏普**，因此不能宣称起点稳健。1990 年起仅多头 LN 的 12 个建仓月夏普范围为 0.42–0.63，其中 10 个月高于各自同月的被动基准；这些路径相关，不能当成 12 次独立确认。")
    sections.append("""LN 的 1 月入场均值—方差策略是有竞争力的历史候选；仅多头版本降低了借贷和卖空假设。风险预算 EW4/历史误差加权也值得研究，但收益改善主要出现在较早时期，并非稳定的新 alpha。分时期（固定 1 月入场）结果如下：
""")
    f = g["subperiods"]
    sections.append(md(["规则", "买入年份", "年度数", "净 CAGR (%)", "超额夏普"],
                       [[f"{r.model}/{r.policy}", r.period, int(r.years), r.cagr_pct, r.sharpe]
                        for r in f.itertuples() if r.model in {"LN", "EW4", "EW_mature", "BuyHold"} and
                        ((r.policy == "long_only" and r.model in {"LN", "BuyHold"}) or
                         (r.policy == "vol_target" and r.model in {"EW4", "EW_mature"}))]))
    f = g["sharpe_intervals"]
    sections.append("夏普优势的不确定性也必须保留。对同一年份的策略与被动收益做 5 年循环区块、5,000 次配对 bootstrap，得到以下 95% 区间（尚未校正策略搜索）：")
    sections.append(md(["规则", "夏普差", "95% 下界", "95% 上界"],
                       [[f"{model}/{policy}", *(f[(f.model == model) & (f.policy == policy)].iloc[0][k]
                           for k in ("delta_sharpe", "ci_low", "ci_high"))]
                        for model, policy in [("LN", "scalar"), ("LN", "long_only"), ("EW4", "vol_target"), ("EW_mature", "vol_target")]]))
    sections.append("所有这些区间都跨越零，当前证据支持“存在值得继续验证的候选”，不足以声称已经确认跑赢被动。多模型月度检验另保存 Holm p 值；它不能覆盖所有研究者选择、风险参数和策略试验。")
    vp = v["policy_summary"]
    f = vp[(vp.source == "vintage") & (vp.lag == 2)]
    sample = f.iloc[0]
    sections.append(f"在宏观历史版本实验的 **{sample.start}–{sample.end}、{int(sample.years)} 笔年度交易**中，主要候选如下（评价期与上一表不同，不能直接横比 CAGR）：")
    sections.append(md(["规则", "净 CAGR (%)", "超额夏普", "年度端点回撤 (%)"],
                       [[f"{model}/{policy}", *(f[(f.model == model) & (f.policy == policy)].iloc[0][k]
                          for k in ("cagr_pct", "sharpe", "max_dd_pct"))]
                        for model, policy in [("BuyHold", "scalar"), ("LN", "scalar"), ("LN_R100", "scalar"),
                                              ("LN", "long_only"), ("EW_mature", "vol_target")]]))
    sections.append("历史版本候选也保存了 24 组 γ/cap、12 个固定入场月、0–10 bps 成本与配对 bootstrap。上述限制与置信区间仍适用。全部新增逐期预测、选择权重、覆盖率、交易和敏感性矩阵保存在本地 `output/research_audit/`，README 与图片由同一批 CSV 生成；不上传原始数据。")
    sections.append("![候选策略与优势的不确定性](assets/strategy_research.png)")
    return "\n\n".join(sections)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT/"output"/"research_audit")
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()
    tables = load_tables(args.input_dir)
    plot(tables)
    text = report(tables)
    (args.input_dir/"report.md").write_text(text)
    if args.update_readme:
        path = ROOT/"README.md"
        content = path.read_text()
        content = content.replace("#51-六项质疑的补充审计与策略探索2026-10-22", "#research-audit")
        content = content.replace("网格预先设为", "网格固定为")
        content = content.replace("预先固定的 Ensemble 规格", "固定权重的 Ensemble 规格")
        marker = "<!-- BEGIN SPECIFICATION_AUDIT -->"
        block = marker+"\n"+text+"\n<!-- END SPECIFICATION_AUDIT -->\n\n"
        if marker in content:
            content = re.sub(r"<!-- BEGIN SPECIFICATION_AUDIT -->.*?<!-- END SPECIFICATION_AUDIT -->\s*", lambda m: block, content, flags=re.S)
        else:
            content = content.replace("## 6. 已修复", block+"## 6. 已修复")
        row = tables["gsw"]["policy_summary"]
        row = row[(row.model == "LN") & (row.policy == "long_only")].iloc[0]
        resume = ("搭建美国国债收益预测与资产配置研究框架，覆盖曲线、宏观 PCA、情绪与机器学习；"
                  "实现标签兑现隔离、折内预处理、时间验证、宏观历史版本检验和含交易费用的年度回测；"
                  "系统审计 15 个因子子集、多周期技术指标、样本起点与风险约束。\n\n"
                  f"探索性历史回测中，仅多头、无杠杆 LN 配置在 1990–2025 年的 35 笔年度交易上，"
                  f"扣费 CAGR 为 {row.cagr_pct:.2f}%、超额夏普 {row.sharpe:.2f}；"
                  "采用拟合零息曲线与所列交易假设，优势尚未达到统计确认，非实盘业绩。")
        content = re.sub(r"(<!-- BEGIN RESUME -->).*?(<!-- END RESUME -->)",
                         lambda m: m[1]+"\n"+resume+"\n"+m[2], content, flags=re.S)
        path.write_text(content)
    print("Exported specification audit and three figures", flush=True)


if __name__ == "__main__":
    main()
