"""Export the additional trading-design study, including uncertainty and failures."""

import argparse
import os
from pathlib import Path
import re
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".cache"/"matplotlib"))

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from statsmodels.stats.multitest import multipletests
from export_research_audit import md


def diagnostics(folder):
    alpha_rows, intervals, annual_rows = [], [], []
    chosen = ["EW_R100/w120", "EW/w120", "LN_R100/w120", "ALL_R100/w240", "LNBW_R100/w120", "TrendVote",
              "EW/mix", "EW/online", "EW/select"]
    for path in sorted(folder.glob("trades_*.csv")):
        trades = pd.read_csv(path)
        policy = path.stem.removeprefix("trades_")
        passive = trades[trades.model == "BuyHold"].set_index("purchase")
        for name, group in trades.groupby("model"):
            a, b = group.set_index("purchase").align(passive, join="inner", axis=0)
            x, z = (a.return_net-a.risk_free).to_numpy(), (b.return_net-b.risk_free).to_numpy()
            fit = sm.OLS(x, sm.add_constant(z)).fit(cov_type="HAC", cov_kwds={"maxlags": 12, "use_correction": True})
            alpha_rows.append({"model": name, "policy": policy, "alpha_pct": 1200*fit.params[0],
                               "beta": fit.params[1], "alpha_p": float(norm.sf(fit.tvalues[0])) if name != "BuyHold" else 1.})
            count = len(a)//12
            ar = (1+a.return_net.to_numpy()[:count*12]).reshape(count, 12).prod(axis=1)-1
            af = (1+a.risk_free.to_numpy()[:count*12]).reshape(count, 12).prod(axis=1)-1
            excess = ar-af
            annual_rows.append({"model": name, "policy": policy, "annual_blocks": count,
                                "sharpe": excess.mean()/excess.std(ddof=1) if excess.std(ddof=1) > 1e-12 else np.nan,
                                "cagr_pct": 100*((1+ar).prod()**(1/count)-1)})
            if policy != "long_only" or name not in chosen:
                continue
            # Paired circular blocks; preserves each strategy/passive year together.
            def sr(v):
                return np.sqrt(12)*v.mean(axis=-1)/v.std(axis=-1, ddof=1)
            for block in (12, 36):
                rng = np.random.default_rng(42)
                starts = rng.integers(0, len(x), (5000, int(np.ceil(len(x)/block))))
                indices = ((starts[:, :, None]+np.arange(block)) % len(x)).reshape(5000, -1)[:, :len(x)]
                delta = sr(x[indices])-sr(z[indices])
                intervals.append({"model": name, "block_months": block,
                                  "delta_sharpe": float(sr(x)-sr(z)),
                                  "ci_low": np.quantile(delta, .025), "ci_high": np.quantile(delta, .975)})
    alpha = pd.DataFrame(alpha_rows)
    alpha["holm_p"] = multipletests(alpha.alpha_p.fillna(1), method="holm")[1]
    alpha.to_csv(folder/"alpha_diagnostics.csv", index=False)
    pd.DataFrame(intervals).to_csv(folder/"sharpe_intervals.csv", index=False)
    pd.DataFrame(annual_rows).to_csv(folder/"annual_block_summary.csv", index=False)
    return alpha, pd.DataFrame(intervals)


def plots(tables, root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), layout="constrained")
    for ax, tag, title in zip(axes, ("monthly_revised", "monthly_vintage"),
                               ("Revised macro + sentiment", "Historical macro vintages; no sentiment")):
        p = tables[tag]["policy"]
        p = p[(p.policy == "long_only") & p.model.str.contains("/")].copy()
        p[["family", "window"]] = p.model.str.split("/", expand=True)
        grid = p.pivot(index="family", columns="window", values="sharpe")[["exp", "w120", "w240"]]
        grid = grid.sort_index()
        im = ax.imshow(grid, cmap="RdYlBu", vmin=0, vmax=.8, aspect="auto")
        ax.set_xticks(range(3), ["Expanding", "Rolling 10y", "Rolling 20y"])
        ax.set_yticks(range(len(grid)), grid.index)
        ax.set_title(title+"\nMonthly, long only, cap 1; net Sharpe")
        for (r, c), value in np.ndenumerate(grid.to_numpy()):
            ax.text(c, r, f"{value:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if value > .65 or value < .15 else "black")
    fig.colorbar(im, ax=list(axes), shrink=.8)
    fig.savefig(ROOT/"assets"/"monthly_design_grid.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, tag in zip(axes, ("monthly_revised", "monthly_vintage")):
        trades = pd.read_csv(root/tag/"trades_long_only.csv")
        names = ["BuyHold", "EW_R100/w120", "EW/w120", "LN_R100/w120"]
        colors = ["#777777", "#367a91", "#b9744e", "#6d9b7e"]
        for name, color in zip(names, colors):
            g = trades[trades.model == name]
            periods = pd.PeriodIndex([g.purchase.iloc[0], *g.sale], freq="M")
            dates = periods.to_timestamp(how="end")
            ax.plot(dates, np.r_[1., (1+g.return_net).cumprod()], label=name, color=color)
        ax.set(title="Revised panel" if tag == "monthly_revised" else "Historical macro versions", ylabel="Net wealth, initial = 1")
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(ROOT/"assets"/"monthly_strategy_equity.png", dpi=170)
    plt.close(fig)


def report(tables):
    t, v = tables["monthly_revised"], tables["monthly_vintage"]
    selected = ["BuyHold", "EH/w120", "EW_R100/w120", "EW/w120", "LN_R100/w120", "LNBW_R100/w120", "TrendVote"]
    parts = [r"""<a id="trading-designs"></a>

### 5.2 从因子预测到可交易策略：滚动估计与月度配置

**继续探索后的判断：原年度实现限制了策略的适应速度，不能据此否定现有因子的价值。** 维持已兑现标签与宏观可得性约束后，改用月度目标、滚动估计和预测收缩，得到风险收益表现更好的候选。本节是新增的策略设计研究，参数及规则均披露，不把经过探索的历史样本重新称为盲测。

先保留原来的年度目标，测试扩展、10 年和 20 年滚动窗口；单靠滚动估计并未稳定超过原 LN 年度方案。因此另建立**一月期预测、一月持有、每月调仓**的策略。年目标的训练标签仍需等待 12 个月兑现；月目标则只需等待一月兑现，统一要求 s+h<t，没有把未兑现的年收益提前加入训练。

**使用的仍是现有因子：** FB/CP 曲线、LN 的 8 个训练宏观 PC、BW 水平及平方、TI 价格趋势。比较扩展/120/240 月窗口；OLS/Ridge(100)；各组联合回归与等权预测；仅多头、双向均值—方差、波动预算、对角风险配置和初始久期中性五类交易设计。预测分别输出 2/3/4/5 年期收益。完整面板为 63 个基础预测规格，宏观历史版本面板为 51 个；另各加 6 个跨窗口组合/历史均值规格，不将最好的规格悄悄替换为预先指定模型。

展示用的四因子组合把 CP/LN/BW/TI 预测等权平均，再按同一训练窗口历史均值收缩：

$$\widetilde\mu_t=\bar\mu_{train,t}+0.5(\widehat\mu_t-\bar\mu_{train,t}),\qquad
w_t=\operatorname{clip}\left(\frac{\operatorname{mean}_n\widetilde\mu_{n,t}}{5\widehat\sigma_{monthly,t}^2},0,1\right).$$

主展示用 120 月训练窗口、最近 60 个已兑现月收益估计风险，每期等权持有各期限，总仓位在 0–100%，余款持有短期国库券。收缩系数 0/0.25/0.5/1、风险窗口 60/120 月都保留比较；不宣称 0.5 或 120 月已经被证明最优。仅多头方案不使用借款或卖空。

**月收益如何计算。** 使用原 GSW 文件中的 BETA0–3、TAU1–2 重建 Svensson 曲线，并与官方 1–5 年期收益率核对至公布精度。买入 n 年期债券，一个月后按剩余 n−1/12 年重新定价：

$$R_{t,t+1}^{(n)}=\exp\{[-(n-1/12)y_{t+1}(n-1/12)+ny_t(n)]/100\}-1.$$

此处 y 为百分数。该公式计入债券期限缩短，不把相同恒定期限的价格差当成实际持有收益。[GSW 官方方法](https://www.federalreserve.gov/pubs/feds/2006/200628/index.html)支持由曲线计算不同到期日的零息价格，同时也说明极短端拟合有局限。

因此现金基准改用 [Kenneth French 官方 RF](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/f-f_factors.html) 的月度国库券收益，按实际持有的下一月对齐；未引入其股票因子。RF 是已实现现金收益基准，未来现金收益不进入预测或仓位规则。另比较 GSW 一年期收益率折算及短端外推口径。每月按全部本金买入、全部卖出市值扣除费用，并计入入场费用占用现金的利息；双向杠杆方案另计超过净值部分的 50 bps 年化借款利差。卖空融券及真实成交条件仍是模型化假设。

**与被动策略的同月份比较。** 下表为修订版宏观与情绪面板，1990-01 买入至 2024-03 卖出，共 410 个不重叠月收益；单边费用 0.5 bps。夏普统一由月度简单超额收益年化，回撤现在覆盖月末路径，不能与上一节仅观察年度端点的回撤直接横比。被动篮子也每月恢复 2–5 年恒定期限，并使用相同成本。
"""]
    p = t["policy"]
    rows = []
    for name in selected:
        row = p[(p.model == name) & (p.policy == "long_only")].iloc[0]
        rows.append([name, row.cagr_pct, row.sharpe, row.max_dd_pct, 100*row.mean_gross_exposure])
    parts.append(md(["规则", "净 CAGR (%)", "超额夏普", "月末回撤 (%)", "平均风险仓位 (%)"], rows))
    parts.append("`EW_R100/w120` 为四个 Ridge 单因子模型预测等权、10 年滚动估计；`EW/w120` 为对应 OLS 版本。`LNBW_R100/w120` 是联合回归候选，其排名来自本轮事后观察，不能拿它替代完整网格的表现。")
    parts.append("为检查月度夏普年化口径的影响，再把相同交易逐个合成为连续、不重叠的 12 月账户收益，现金收益也同期复利；丢弃末尾不足一年的观测。34 个完整年度区块的夏普如下，不使用 √12：")
    annual = t["annual"]
    parts.append(md(["规则", "完整年度数", "年度区块夏普"],
                    [[n, int(annual[(annual.model == n) & (annual.policy == "long_only")].iloc[0].annual_blocks),
                      annual[(annual.model == n) & (annual.policy == "long_only")].iloc[0].sharpe]
                     for n in ["BuyHold", "EW_R100/w120", "EW/w120", "LN_R100/w120"]]))
    parts.append("月度表的年化夏普是常用描述统计；有自相关时，它不等于真实一年持有期夏普。年度区块检查和下方区块 bootstrap 提供另外两种观察方式。")
    parts.append("**避免再次按历史排名选择 10 年窗口。** 还将 OLS/Ridge × 扩展/10 年/20 年的六个 EW 专家组合，报告固定等权 `EW/mix`、按已兑现预测 MSE 倒数加权的 `EW/online` 和按历史最小 MSE 选择的 `EW/select`。后两者只使用最近 60 个买入月份中已兑现的一月预测误差，至少 36 个误差后启动，此前等权；所有截止日、权重和选择均留档。组合仍收缩到相同权重的各窗口历史均值，交易约束保持一致。")
    rows = []
    for tag, label in [("monthly_revised", "新版四因子"), ("monthly_vintage", "历史宏观")]:
        f = tables[tag]["policy"]
        for n in ("BuyHold", "EW/mix", "EW/online", "EW/select"):
            r = f[(f.model == n) & (f.policy == "long_only")].iloc[0]
            rows.append([label, n, r.cagr_pct, r.sharpe, r.max_dd_pct])
    parts.append(md(["面板", "规则", "净 CAGR (%)", "超额夏普", "月末回撤 (%)"], rows))
    parts.append("递归权重可使用评价起点前已生成、当时能够形成的历史预测作为内部验证，不按展示窗口清空合法历史；历史宏观版本受档案起始月限制，有更长的等权启动阶段。自适应不保证胜过每个固定专家，其实际结果也保留。")
    parts.append("![全部月度预测规格与训练窗口](assets/monthly_design_grid.png)")
    f = t["sensitivity"]
    checks = [("origin", "1985-01"), ("origin", "1990-01"), ("origin", "2000-01"), ("origin", "2013-01"),
              ('{"cost_bps": 2}', None), ('{"cost_bps": 5}', None), ('{"cost_bps": 10}', None),
              ('{"delay": 1}', None), ('{"risk_window": 120}', None)]
    rows = []
    for check, origin in checks:
        s = f[f.check == check]
        if origin:
            s = s[s.origin == origin]
        a, b = (s[s.model == n].iloc[0] for n in ("EW_R100/w120", "BuyHold"))
        labels = {'{"cost_bps": 2}': "单边成本 2 bps", '{"cost_bps": 5}': "单边成本 5 bps",
                  '{"cost_bps": 10}': "单边成本 10 bps", '{"delay": 1}': "延迟一月执行",
                  '{"risk_window": 120}': "风险估计使用 120 月"}
        label = f"{origin[:4]} 年起" if origin else labels[check]
        rows.append([label, a.cagr_pct, a.sharpe, b.cagr_pct, b.sharpe])
    parts.append("**候选能否经受更严格的执行检查？** 对四因子 Ridge 等权版本，仅改变一项条件，使用同一结束月份：")
    parts.append(md(["条件", "策略 CAGR (%)", "策略夏普", "被动 CAGR (%)", "被动夏普"], rows))
    parts.append("延迟一月执行使用上个月已形成的预测和趋势信号，不增加样本末端的一月收益。2–5 bps 下优势点估计仍在；10 bps 下策略超额夏普转负，成本不可忽略。较晚时期的绝对收益能力也较弱，不能从全样本约 0.70 的夏普外推今后的水平。")
    f = t["subperiods"]
    parts.append(md(["规则", "买入年份", "净 CAGR (%)", "超额夏普", "月末回撤 (%)"],
                    [[r.model, r.period, r.cagr_pct, r.sharpe, r.max_dd_pct] for r in f.itertuples()
                     if r.policy == "long_only" and r.model in {"BuyHold", "EW_R100/w120"}]))
    parts.append("**宏观历史版本复核。** 该部分只使用 CP/LN/TI；未取得 BW 历史版本，故不将四因子新版组合包装成严格实时策略。EW 在这里表示三个单因子预测等权，和上表的四因子组合分别报告。样本为 2000-01–2026-02，共 313 个已兑现月收益，沿用同一交易、收缩与风险规则。")
    f = v["policy"]
    parts.append(md(["规则", "净 CAGR (%)", "超额夏普", "月末回撤 (%)"],
                    [[n, *(f[(f.model == n) & (f.policy == "long_only")].iloc[0][c]
                            for c in ("cagr_pct", "sharpe", "max_dd_pct"))]
                     for n in ["BuyHold", "EH/w120", "EW/w120", "EW_R100/w120", "LN_R100/w120"]]))
    parts.append("历史版本中 OLS 等权与 Ridge 等权表现不同，不能说收缩模型处处占优。这里继续保留完整网格及旧年度方案；GSW 曲线与 RF 仍是当前下载版本，历史宏观文件也只提供月度时点，整套系统仍非严格历史逐日可成交记录。")
    parts.append("![月度候选与被动篮子的扣费净值](assets/monthly_strategy_equity.png)")
    parts.append("**统计和搜索的不确定性。** 对月度策略/被动同月份收益做配对循环区块 bootstrap，分别用 12/36 月区块、5,000 次重复。以下是夏普差的 95% 区间，尚未覆盖研究者的全部选择：")
    rows = []
    for tag, label in [("monthly_revised", "新版四因子"), ("monthly_vintage", "历史宏观")]:
        frame = tables[tag]["ci"]
        for name in (["EW_R100/w120", "EW/w120"] if tag == "monthly_revised" else ["EW/w120", "LN_R100/w120"]):
            for block in (12, 36):
                r = frame[(frame.model == name) & (frame.block_months == block)].iloc[0]
                rows.append([label, name, block, r.delta_sharpe, r.ci_low, r.ci_high])
    parts.append(md(["面板", "规则", "区块月数", "夏普差", "95% 下界", "95% 上界"], rows))
    parts.append("另保存预测损失的 HAC/全规格 Holm 校正，以及各交易设计相对被动超额收益回归的 HAC α 与多重比较校正。这些诊断不能把同一段反复研究过的历史变成独立确认。当前更有根据的结论是：**现有因子可以构建有竞争力的配置规则，收益目标、估计窗口和交易频率会显著影响其经济价值；仍需对未来新增数据作独立验证。**")
    a = t["alpha"]
    a = a[(a.model == "EW_R100/w120") & (a.policy == "long_only")].iloc[0]
    parts.append(f"例如四因子 Ridge 候选相对被动的线性超额收益回归，年化 HAC α 点估计为 {a.alpha_pct:.2f}%、单边 p={a.alpha_p:.3f}；涵盖全部模型/交易设计的 Holm p={a.holm_p:.3f}。因此个别未校正区间略高于零也不构成完成模型搜索后的显著性证明。")
    parts.append("新增逐期预测、四期限仓位、月度现金流、全部五类交易设计、失败模型、敏感性与统计诊断保存在本地 `output/trading_designs/`。原始数据继续留在本地，仓库只发布代码、README、依赖和展示图。")
    return "\n\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=ROOT/"output"/"trading_designs")
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()
    tables = {}
    for tag in ("monthly_revised", "monthly_vintage"):
        folder = args.input_dir/tag
        alpha, ci = diagnostics(folder)
        tables[tag] = {"policy": pd.read_csv(folder/"policy_summary.csv"),
                       "sensitivity": pd.read_csv(folder/"sensitivity.csv"),
                       "subperiods": pd.read_csv(folder/"subperiods.csv"), "ci": ci,
                       "annual": pd.read_csv(folder/"annual_block_summary.csv"), "alpha": alpha}
    plots(tables, args.input_dir)
    text = report(tables)
    (args.input_dir/"report.md").write_text(text)
    if args.update_readme:
        path = ROOT/"README.md"
        content = path.read_text()
        block = "<!-- BEGIN TRADING_DESIGNS -->\n"+text+"\n<!-- END TRADING_DESIGNS -->\n\n"
        if "<!-- BEGIN TRADING_DESIGNS -->" in content:
            content = re.sub(r"<!-- BEGIN TRADING_DESIGNS -->.*?<!-- END TRADING_DESIGNS -->\s*", lambda m: block, content, flags=re.S)
        else:
            content = content.replace("## 6. 已修复", block+"## 6. 已修复")
        r = tables["monthly_revised"]["policy"]
        r = r[(r.model == "EW_R100/w120") & (r.policy == "long_only")].iloc[0]
        resume = ("搭建美国国债多因子收益预测与月度资产配置框架，结合曲线、宏观 PCA、情绪与技术指标；"
                  "实现标签兑现隔离、滚动估计、预测收缩、宏观历史版本检验和含成本的债券逐月定价回测。\n\n"
                  f"探索性历史回测中，四因子等权预测、10 年滚动训练、仅多头无杠杆组合在 1990-01–2024-03 "
                  f"410 个非重叠月收益上，扣费 CAGR {r.cagr_pct:.2f}%、月度年化超额夏普 {r.sharpe:.2f}；"
                  "同期被动约为 4.47%/0.58。收益基于拟合零息曲线及所列成本假设，非实盘记录；"
                  "历史搜索与独立验证边界详见策略研究部分。")
        content = re.sub(r"(<!-- BEGIN RESUME -->).*?(<!-- END RESUME -->)", lambda m: m[1]+"\n"+resume+"\n"+m[2], content, flags=re.S)
        path.write_text(content)
    print("Exported monthly-design report, confidence intervals and two figures", flush=True)


if __name__ == "__main__":
    main()
