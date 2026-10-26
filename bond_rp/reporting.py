"""Export compact research tables and figures from the final account ledger."""
from pathlib import Path
import json
import shutil
import numpy as np
import pandas as pd


def md_table(frame, formats=None):
    formats=formats or {}
    f=frame.copy()
    for c,fmt in formats.items():
        f[c]=f[c].map(lambda v:fmt.format(v) if pd.notna(v) else "—")
    return "| "+" | ".join(f.columns)+" |\n| "+" | ".join(["---"]*len(f.columns))+" |\n"+"\n".join(
        "| "+" | ".join(map(str,row))+" |" for row in f.itertuples(index=False,name=None))


def export_report(out,root):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,
                         "axes.spines.right":False,"figure.dpi":160,"savefig.bbox":"tight"})
    assets=root/"assets"
    results=root/"results"
    assets.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    for name in ("performance","paper_comparison","paper_checks","paper_coefficients","sensitivity",
                 "subperiods","forecast_scores","raw_forecast_scores","in_sample","in_sample_coefficients","annual_utility",
                 "active_return_tests","cer_intervals","revised_origins","parameter_grid",
                 "selection_schedule","selection_performance","parameter_family_tests",
                 "factor_validation","macro_information","assumption_performance","r2_accounting","published_cells"):
        shutil.copyfile(out/(name+".csv"),results/(name+".csv"))
    shutil.copyfile(out/"manifest.json",results/"data_manifest.json")
    trades=pd.read_csv(out/"monthly_accounts.csv")
    stats=pd.read_csv(out/"performance.csv").set_index("model")
    labels={"EW3":"CP + Macro8 + Trend","BuyHold":"2–5y passive basket","EH":"Historical-mean timing","Cash":"Treasury bills"}
    colors={"EW3":"#086f83","BuyHold":"#e59938","EH":"#687bb1","Cash":"#91999f"}
    fig,axes=plt.subplots(2,1,figsize=(10.5,6.7),sharex=True,gridspec_kw={"height_ratios":[2,1]})
    for name in ("EW3","BuyHold","EH","Cash"):
        g=trades.loc[trades.model.eq(name)].sort_values("purchase")
        dates=pd.to_datetime([g.purchase.iloc[0]]+g.sale.tolist())
        wealth=np.r_[1.,np.cumprod(1+g.return_net)]
        axes[0].plot(dates,wealth,label=labels[name],color=colors[name],lw=2 if name=="EW3" else 1.25)
        if name!="Cash":
            axes[1].plot(dates,100*(wealth/np.maximum.accumulate(wealth)-1),color=colors[name],lw=1.25)
    axes[0].set_ylabel("Wealth (initial = 1)")
    axes[0].legend(loc="upper left",frameon=False,ncol=2,fontsize=9)
    axes[0].set_title("Treasury return forecasting and risk-constrained timing",loc="left",weight="bold",pad=12)
    axes[1].set_ylabel("Drawdown (%)")
    for ax in axes:ax.grid(alpha=.18)
    fig.text(.13,.01,"Monthly macro vintages | Jan 2000–Feb 2026 account | 0.5 bp per side | Fully funded, long only",fontsize=9,color="#555555")
    fig.tight_layout(rect=[0,.03,1,1]);fig.savefig(assets/"strategy.png");plt.close(fig)
    # Maturity-weight allocation and annual account returns explain the source
    # of the result, including years in which cash does better than the strategy.
    g=trades.loc[trades.model.eq("EW3")].sort_values("purchase")
    fig,ax=plt.subplots(figsize=(10.5,2.7))
    ax.fill_between(pd.to_datetime(g.purchase),0,g.gross_exposure,color=colors["EW3"],alpha=.75)
    ax.set(ylim=(0,1.05),ylabel="Bond allocation",title="Allocation to the equal-weight 2–5y Treasury basket")
    ax.grid(alpha=.18);fig.tight_layout();fig.savefig(assets/"allocation.png");plt.close(fig)
    paper=pd.read_csv(out/"paper_comparison.csv")
    names=["FB in CP05 Table 2","CP05 restricted","LN09 LN6 released","LN09 CP+LN5"]
    fig,axs=plt.subplots(1,4,figsize=(11,3.1),sharey=True)
    for ax,name,title in zip(axs,names,["FB (CP Table 2)","CP (Table 1B)","LN6 (Table 2)","CP + LN5"]):
        d=paper.loc[paper.model.eq(name)]
        x=np.arange(4)
        ax.bar(x-.18,d.published,.36,label="Published",color="#bbcad6")
        ax.bar(x+.18,d.reproduced,.36,label="Reproduced",color=colors["EW3"])
        ax.set_xticks(x,["2y","3y","4y","5y"]);ax.set_title(title,fontsize=10)
        ax.grid(axis="y",alpha=.15)
    axs[0].set_ylabel("R-squared (LN: adjusted)");axs[0].legend(frameon=False,fontsize=8)
    fig.suptitle("Numerical replication using public author archives",weight="bold",fontsize=12)
    fig.tight_layout();fig.savefig(assets/"replication.png");plt.close(fig)
    # Update only the generated numerical section; methods remain hand-reviewed.
    show=stats.loc[["EW3","BuyHold","EH","Cash"],["cagr_pct","sharpe","vol_pct","max_dd_pct"]].reset_index()
    show["model"]=show.model.map({"EW3":"最终组合策略","BuyHold":"等权持有 2–5 年债券","EH":"历史均值择时","Cash":"短期国库券账户"})
    show.columns=["策略","扣费 CAGR","超额夏普","年化波动","最大回撤"]
    table=md_table(show,{"扣费 CAGR":"{:.2f}%","超额夏普":"{:.2f}","年化波动":"{:.2f}%","最大回撤":"{:.2f}%"})
    sub=pd.read_csv(out/"subperiods.csv")
    sub=sub.loc[sub.model.isin(["EW3","BuyHold"])&~sub.period.eq("2013-2026"),["period","model","cagr_pct","sharpe","max_dd_pct"]]
    sub["model"]=sub.model.map({"EW3":"组合策略","BuyHold":"等权持有"})
    sub.columns=["建仓时期","策略","CAGR","超额夏普","最大回撤"]
    subtable=md_table(sub,{"CAGR":"{:.2f}%","超额夏普":"{:.2f}","最大回撤":"{:.2f}%"})
    r2=pd.read_csv(out/"forecast_scores.csv")
    r2=r2.loc[r2.model.eq("EW3")&r2.window.eq("120")&r2.maturity.eq("basket")].set_index("horizon")
    ci=pd.read_csv(out/"cer_intervals.csv").set_index("reference")
    c=ci.loc["BuyHold"]
    ew=stats.loc["EW3"];bh=stats.loc["BuyHold"]
    checks=pd.read_csv(out/"paper_checks.csv")
    cells=pd.read_csv(out/"published_cells.csv")
    block=f'''<!-- RESULTS -->
建仓月份为 **2000-01 至 2026-01**，最后卖出月份为 **2026-02**，共 {int(ew.months)} 个连续月度账户。下表为历史宏观版本主实验；债券买卖双边各扣 0.5 bp，所有策略使用相同现金账户和交易规则。

{table}

组合策略期末净值为初始资金的 **{ew.terminal_wealth:.2f} 倍**。相对持有债券，CAGR 提高 **{ew.cagr_pct-bh.cagr_pct:.2f} 个百分点**，最大回撤绝对值降低 **{100*(1-abs(ew.max_dd_pct)/abs(bh.max_dd_pct)):.1f}%**。夏普用扣费收益减同月国库券收益计算；CAGR 是总收益，二者口径不同。

![累计净值与回撤](assets/strategy.png)

{subtable}

采用 {int(c.years)} 个完整、不重叠的年度账户，γ=5 的年度均值—方差确定性等价收益（CER）相对持有基准提高 **{c.cer_gain_bps:.1f} bp/年**，条件区块自助法 95% 区间为 **[{c.ci_low_bps:.1f}, {c.ci_high_bps:.1f}] bp**。置信区间包含零时，不声称经济增益已显著。

50% 收缩后的组合篮子预测 R²_OOS 为：1 个月 **{100*r2.loc[1,'r2_oos']:.2f}%**，3 个月 **{100*r2.loc[3,'r2_oos']:.2f}%**，12 个月 **{100*r2.loc[12,'r2_oos']:.2f}%**；三个持有期是独立预测评价，不能把年度重叠标签当作月度策略收益。策略的经济效果不等同于稳定提高均值预测精度。

实现与算法一致性检查 **{int(checks.passed.sum())}/{len(checks)} 通过**；逐项核对的论文印刷数值中 **{int(cells.matches_printing.sum())}/{len(cells)}** 在公布精度内一致，其余差异保留并标明。完整数值见 [论文对照表](results/paper_comparison.csv)、[逐单元核验](results/published_cells.csv)和[算法检查](results/paper_checks.csv)。

![论文数值复现](assets/replication.png)
<!-- /RESULTS -->'''
    readme=root/"README.md"
    text=readme.read_text()
    begin=text.index("<!-- RESULTS -->");end=text.index("<!-- /RESULTS -->")+len("<!-- /RESULTS -->")
    readme.write_text(text[:begin]+block+text[end:])
