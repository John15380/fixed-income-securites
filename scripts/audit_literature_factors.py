"""Trace named factors to the supplied readings and isolate the ME input repair.

This is a source audit and a controlled historical comparison, not another
candidate search. The RGDP run must have been retained locally before rerunning.
"""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from bond_rp.literature_audit import survey_paper_audit
from bond_rp.data import load_bond_data
from bond_rp.ms17 import MS17Regression
from bond_rp.peer_allocation import allocation_path,summarize
from bond_rp.peer_learners import xgb_predict
from bond_rp.validated_models import load_long_market

# File references are the actual local reading-list copies, not inferred titles
# from third-party summaries. Files and data themselves remain unpublished.
SOURCES = [
    ("FB", "!Fama and Bliss 1987.pdf", "forward spread f(n)-y(1)",
     "bond_rp/data.py; coursework.py", "逐期限原规格；1987数据版本差异保留"),
    ("CP05", "!Cochrane and Piazzesi 2005.pdf", "Eqs.1-2; mean(rx) on five forwards; restricted maturity loading",
     "bond_rp/coursework.py:common_index", "共同指数及第二阶段已核对"),
    ("LN5/LN6", "!Ludvigson and Ng 2009.pdf", "Eqs.7-8; F1,F1^3,F2,F3,F4,F8; LN5 drops F2",
     "bond_rp/coursework.py; peer_reference.py", "现代宏观面板是适配；COVID三次项外推失败仍保留"),
    ("MS recursive", "MS2017-bond-risk-premia-and-gaussian-term-structure-models.pdf", "Section2.2 Eqs.3-4; R=(1-alpha)*gamma*f+alpha*lag(R)",
     "bond_rp/ms17.py", "联合估计递归系数，不把原样本alpha或固定EMA当作样本外估计"),
    ("ME", "JFQA2017 Expected Business Conditions and Bond Risk Premia.pdf", "SectionIII; Table1; Eqs.6-7; NGDP,PGDP,UNEMP,CPROF,INDPROD,HOUSING x 4 horizons",
     "bond_rp/survey_expectations.py; peer_reference.py", "已修正RGDP为NGDP；原文中位数预期与PPT分歧代理分列"),
    ("CP15", "RFS2015 Expected Returns in Treasury Bonds.pdf", "Eqs.19-21,25,27; core CPI DMA .987/120; yield cycles; duration-scaled 2-20y target",
     "bond_rp/validated_models.py", "原文CMT换为GSW；原文CPI滞后1月，主实验滞后2月"),
    ("Goh48", "Goh et al 2013.pdf", "SectionII.A Eqs.1-2; n=2..5, short=3/6/9, long=18/24/30/36; SectionII.C recursive BIC",
     "bond_rp/refinement.py:goh_signals,goh_predict", "48个远期利差交叉已实现；原文另有15个OBV信号，尚缺对应成交量数据"),
    ("BW3", "Laborda and Olmo 2014 .pdf", "Eq.14; SENT_ORTH, square, first difference -> common fitted index",
     "bond_rp/refinement.py:literal_sentiment", "原文三项在扩展中实现；早期SENT两项是简化代理；历史ORTH版本未取得"),
    ("sPCA", "Are Bond Returns Predictable with Real-Time Macro Data.pdf", "Section3.2; return-slope scaling then first PC",
     "bond_rp/refinement.py:spca_predict", "方法已实现；使用成熟训练标签，不用全样本缩放权重"),
    ("Policy uncertainty", "RF2022 Economic Policy Uncertainty and the Yield Curve.pdf", "Sections4.1-4.2; distinct GPU/MPU weighted components, IP/CPI year-on-year log growth",
     "peer_reference.py:candidate_menu uses aggregate EPU proxy", "总EPU不等于原文GPU/MPU；不能把EPU模板称为该论文因子复现"),
    ("Inflation disagreement", "RFS2017 Inflation Bets on the Long Bond.pdf", "Michigan inflation forecast IQR and debt-supply interaction",
     "未接入原始Michigan IQR及供给交互", "SPF六变量分歧不是本论文的原始因子，也不是Eriksen ME"),
    ("Macro uncertainty", "AER2015 measuring uncertainty.pdf", "Conditional volatility of unforecastable macro components",
     "尚未接入原始不确定性序列", "宏观水平的PCA、VIX或普通滚动波动率均不能冒称JLN uncertainty"),
    ("Stock illiquidity", "Bouwman, Sojli, and Tham 2013.pdf", "Amihud cross-stock aggregate and large/small-cap liquidity difference",
     "尚未接入逐股收益/成交金额", "不是把TED或BAA利差直接改名为原文流动性因子"),
    ("VRP", "Zhou 2012.pdf", "Section2 Eqs.1-3; VIX^2/12 minus a time-t forecast of future realized variance",
     "尚未接入高频已实现方差", "不能直接扣除尚未发生的未来RV；原文主体为1-6月债券，2-5年为适配"),
    ("Debt supply", "RFS2014 Bond Supply and Excess Bond Returns.pdf", "Maturity-weighted government debt/GDP",
     "尚未接入逐期限存量及历史GDP版本", "总债务/GDP不能直接替代久期结构"),
    ("Bank demand", "JF2020 The banking view of bond risk premium.pdf", "Bank interest-rate exposure and bond risk premia",
     "尚未接入银行持仓/资产负债表", "不能用单一宏观利差宣称完整复现银行需求机制"),
    ("Fiscal policy", "JFE2020 Fiscal Policy Driven Bond Risk Premia.pdf", "Government spending level and volatility",
     "尚未接入该文支出及冲击模型", "一般EPU不是财政支出波动冲击"),
    ("Global macro", "MS2021-unspanned-global-macro-risks-in-bond-returns.pdf", "Global macro risks outside the yield span",
     "尚未接入跨国宏观及债券面板", "美国FRED面板不等于全球因子"),
]


def source_table():
    readings=ROOT/"GroupProjectReading"
    rows=[]
    for name,filename,definition,implementation,limitation in SOURCES:
        files=[p for p in readings.rglob(filename) if p.parent.name!="ReadingCN"]
        if len(files)!=1:
            raise ValueError(f"Expected one original reading for {filename}: {files}")
        path=files[0]
        rows.append(dict(factor=name,reading=str(path.relative_to(ROOT)),definition=definition,
            implementation=implementation,limitation=limitation,
            pdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return pd.DataFrame(rows)


def ms_original_sample_audit(output):
    """MS Tables1-2 with six maturities and CRSP forwards/GSW returns.

    The formation period and outcome end are explicitly recorded. Published
    values are comparisons, not a tolerance widened to force a vintage match.
    """
    crsp=load_bond_data().set_index("ym")
    market=load_long_market()
    dates=pd.period_range("1963-01","2003-12",freq="M")
    maturities=[2,3,4,5,7,10]
    f=crsp.loc[dates,[f"f{n}" for n in range(1,6)]].to_numpy()
    y=market.loc[dates,[f"long_rx{n}" for n in maturities]].to_numpy()
    model=MS17Regression().fit(f,y)
    # Independently compute the literal recursion and observation equation.
    previous=model.initial_;r=[]
    for row in f:
        previous=(1-model.alpha_)*(model.gamma_[0]+row@model.gamma_[1:])+model.alpha_*previous
        r.append(previous)
    expected=np.asarray(r)[:,None]*model.loadings_
    np.testing.assert_allclose(model.predict(f),expected,atol=1e-9,rtol=1e-9)
    r2=1-np.sum((y-expected)**2,axis=0)/np.sum((y-y.mean(0))**2,axis=0)
    rows=[]
    for n,b,published_b,value,published_r2 in zip(maturities,model.loadings_,
            [.29,.55,.79,1.01,1.40,1.95],r2,[.357,.400,.430,.450,.473,.486]):
        rows.append(dict(maturity=n,published_loading=published_b,reproduced_loading=b,
            published_r2=published_r2,reproduced_r2=value,
            published_alpha=.6284,reproduced_alpha=model.alpha_,n=len(dates),
            purchase_start=str(dates[0]),purchase_end=str(dates[-1]),last_outcome=str(dates[-1]+12),
            exact_data_vintage=False))
    table=pd.DataFrame(rows)
    table.to_csv(output/"ms_original_sample.csv",index=False)
    return table


def controlled_comparison(current,legacy,output):
    """Fix every old XGB/execution parameter; change only the ME input panel.

    Check all 14 unaffected standardized feature columns against the legacy
    run at every outer fold. No parameter is selected from the resulting test
    scores; reoptimized results are reported as a separate third branch.
    """
    bonds=load_long_market()
    dates=pd.period_range("2017-01","2022-12",freq="M")
    rows,predictions,trades=[],[],[]
    untouched=[j for j in range(15) if j!=9]
    for case in ("dispersion","expectations"):
        now,old=current/case,legacy/case
        folds=pickle.loads((now/"folds.pkl").read_bytes())
        oldfolds=pickle.loads((old/"folds.pkl").read_bytes())
        for date,fold in folds.items():
            previous=oldfolds[date]
            assert fold["train_dates"].equals(previous["train_dates"])
            np.testing.assert_array_equal(fold["y"],previous["y"])
            np.testing.assert_allclose(fold["X"][:,untouched],previous["X"][:,untouched],atol=1e-10,rtol=1e-10)
            np.testing.assert_allclose(fold["x"][:,untouched],previous["x"][:,untouched],atol=1e-10,rtol=1e-10)
        oldparams=json.loads((old/"frozen_parameters.json").read_text())
        execution=json.loads((old/"execution_parameters.json").read_text())["params"]
        newpred=np.vstack([xgb_predict(folds[d],oldparams["L4_GC"]) for d in dates])
        forecast=pd.DataFrame(newpred,index=dates,columns=[f"mu{n}" for n in range(2,6)])
        predictions.append(forecast.assign(case=case,ym=dates.astype(str)))
        path=allocation_path(bonds,forecast,**execution)
        trades.append(path.assign(case=case))
        rows.append(dict(case=case,variant="NGDP_fixed_old_parameters",**summarize(path)))
        for variant,folder in [("RGDP_legacy",old),("NGDP_same_search_budget",now)]:
            perf=pd.read_csv(folder/"performance.csv")
            values=perf.loc[perf.model.eq("L4_GC/optimized")].iloc[0].drop("model").to_dict()
            rows.append(dict(case=case,variant=variant,**values))
    pd.DataFrame(rows).to_csv(output/"gdp_correction_control.csv",index=False)
    pd.concat(predictions,ignore_index=True).to_csv(output/"fixed_parameter_predictions.csv",index=False)
    pd.concat(trades,ignore_index=True).to_csv(output/"fixed_parameter_trades.csv",index=False)
    return pd.DataFrame(rows)


def report(output,catalog,stats,evidence,performance,ms_table):
    lines=['<a id="factor-origins"></a>',"### 5.9 从 GroupProjectReading 原文追溯因子与输入修正","",
        "本轮从课程必做 FB/CP/LN 出发，核对具名扩展的原始公式与实际代码入口。",
        "下表区分已实现的文献规格、数据适配和尚未接入的原文输入；同名或经济动机相似不等于数值复现。","",
        "| 因子 | 读本中的原文依据 | 实现与边界 |","| --- | --- | --- |"]
    for _,r in catalog.iterrows():
        lines.append(f"| {r.factor} | {Path(r.reading).name}；{r.definition} | {r.implementation}；{r.limitation} |")
    lines += ["", "**本轮发现并修正了实际错误。** Eriksen 的 GDP 应使用 SPF **NGDP（名义 GDP）**，先前误用了 RGDP。",
        "识别依据是原文 Table1 的四个预期增长均值，完全不依赖策略收益；不是为了改善测试结果换变量。","",
        "| GDP 口径 | 一季 | 二季 | 三季 | 四季 | 相对原表的均值 RMSE（百分点） |",
        "| --- | --- | --- | --- | --- | --- |",
        "| 论文 Table1 | 1.53 | 3.13 | 4.76 | 6.44 | — |"]
    for name in ("NGDP","RGDP"):
        values=evidence["current_gdp_means"][name]
        lines.append(f"| {name} | "+" | ".join(f"{v:.4f}" for v in values)+f" | {evidence['mean_rmse'][name]:.4f} |")
    lines += ["", "统计区间为 1968Q4–2014Q4；完整24项均值/标准差另存。住宅开工变化按原表单位乘100，",
        "这种固定单位换算在标准化 PCA 中抵消。最新调查档案仍存在部分历史数值差异，",
        f"24项完整季度行共 {evidence['quarterly_rows']} 个；前三 PC 方差占比为 "+
        "/".join(f"{v:.2f}%" for v in evidence["current_first3_variance_pct"])+"，原表约53%/25%/14%。",
        "因此可以确认 GDP 口径错误已修正，仍不能宣称所有调查原表已精确重现。","",
        "**82个候选的出处。** 参考 PPT 第4–5页将它们称为约20个宏观金融序列经固定模板和多个窗口生成的候选，",
        "并没有称其为某篇读本直接提供的82个既定因子。经典六类有明确原文；本项目的20×4＋2候选池仍是公开的重建假设。",
        "Goh48与BW3此前已在§5.4实现，不重复当作本轮新发明；PPT的SPF分歧代理既不同于Eriksen的中位数预期，",
        "也不等于 Hong–Sraer–Yu 的 Michigan 通胀分歧与国债供给交互。","",
        "**修正前后结果。** 测试买入期2017-01–2022-12，72笔月交易、卖出至2023-01；零交易成本。",
        "第一项为保留的错误RGDP旧版本；第二项固定旧XGB和投资校准参数，仅修正ME输入；",
        "第三项沿用上一轮验证区间与相同搜索预算重新估计，不增加候选预算。",
        "受影响的第10列ME以外，全部外层拟合的另14列、训练日期与目标都逐项核验一致。","",
        "| ME 版本 | 实验 | CAGR (%) | 超额夏普 | 最大回撤 (%) |",
        "| --- | --- | --- | --- | --- |"]
    labels={"dispersion":"PPT分歧代理","expectations":"Eriksen原文预期"}
    order=["RGDP_legacy","NGDP_fixed_old_parameters","NGDP_same_search_budget"]
    for case in labels:
        for variant in order:
            r=performance.loc[performance.case.eq(case)&performance.variant.eq(variant)].iloc[0]
            lines.append(f"| {labels[case]} | {variant} | {r.cagr_pct:.2f} | {r.sharpe:.2f} | {r.max_dd_pct:.2f} |")
    lines += ["", "这项修正改善了因子与论文定义的一致性；收益变化不能先验保证为正。重跑仍未达到对方高夏普，",
        "不按GDP口径或两种ME版本的测试排名选择最终定义，也不将修正后模型替换为未经比较验证的主策略。",
        "文献核查并未创造未接触过的新测试集；本节是回溯审计，不能称为新的独立确认。","",
        "**具名基准的第二阶段约束。** CP05与MS参考预测改为直接沿用原文期限载荷，不再统一附加截距。",
        "同时用MS原始六期限（2/3/4/5/7/10年）、CRSP远期利率与GSW收益重做描述性比较；",
        f"1963-01–2003-12买入样本估计α={ms_table.reproduced_alpha.iloc[0]:.4f}，原表为0.6284。",
        "实际逐月递推另用循环独立核算；当期档案与论文原始版本不完全一致，原表差值不作为强行通过的容差。","",
        "| 期限 | 原表载荷 | 重估载荷 | 原表 R² (%) | 重估 R² (%) |",
        "| --- | --- | --- | --- | --- |"]
    for _,r in ms_table.iterrows():
        lines.append(f"| {int(r.maturity)}年 | {r.published_loading:.2f} | {r.reproduced_loading:.2f} | {100*r.published_r2:.1f} | {100*r.reproduced_r2:.1f} |")
    lines += ["", "这项MS比较是原区间的数据适配，不是2017–2022交易结果；不将其α固定带入未来回测。","",
        "原文输入复核：`python scripts/audit_literature_factors.py --sources-only`。",
        "完整修正归因：`python scripts/audit_literature_factors.py --update-readme`，需先保留旧版运行目录",
        "`output/peer_reference_rgdp_legacy/`（对应提交acbdd27），再运行修正后的研究；旧输出不随仓库上传。",
        "原PDF和原始数据继续保留在本地；上传内容为代码、说明与已有结果图。"]
    text="\n".join(lines)+"\n"
    (output/"report.md").write_text(text)
    return text


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current",type=Path,default=ROOT/"output"/"peer_reference")
    parser.add_argument("--legacy",type=Path,default=ROOT/"output"/"peer_reference_rgdp_legacy")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"output"/"literature_factor_audit")
    parser.add_argument("--update-readme",action="store_true")
    parser.add_argument("--sources-only",action="store_true",help="Audit paper inputs without requiring the retained legacy experiment")
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    catalog=source_table();stats,evidence=survey_paper_audit()
    catalog.to_csv(args.output_dir/"factor_sources.csv",index=False)
    stats.to_csv(args.output_dir/"survey_table1.csv",index=False)
    (args.output_dir/"survey_evidence.json").write_text(json.dumps(evidence,indent=2))
    ms_table=ms_original_sample_audit(args.output_dir)
    if args.sources_only:
        print(json.dumps(dict(source_count=len(catalog),survey_series=len(stats),**evidence),indent=2))
        return
    performance=controlled_comparison(args.current,args.legacy,args.output_dir)
    text=report(args.output_dir,catalog,stats,evidence,performance,ms_table)
    if args.update_readme:
        path=ROOT/"README.md";body=path.read_text()
        start,end="<!-- FACTOR_ORIGINS:START -->","<!-- FACTOR_ORIGINS:END -->"
        block=start+"\n"+text+end+"\n\n"
        if start in body:
            a,b=body.index(start),body.index(end)+len(end)
            body=body[:a]+block.rstrip()+body[b:]
        else:
            k=body.index("## 6.");body=body[:k]+block+body[k:]
        path.write_text(body)
    print(performance[["case","variant","cagr_pct","sharpe","max_dd_pct"]].to_string(index=False))
    print("Original sources verified:",len(catalog),"; survey series:",len(stats),"; outer folds controlled:",288)


if __name__=="__main__":
    main()
