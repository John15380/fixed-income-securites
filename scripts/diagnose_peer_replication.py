"""Controlled diagnosis of mismatches with the reference's first 22 slides.

This is retrospective error attribution, not another search for a winning test
strategy. All input/portfolio variants below are reported; no test score chooses
the main model. The peer's own code and prediction series remain unavailable.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache"/"matplotlib"))
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from bond_rp.models import _numeric_pipeline
from bond_rp.peer_allocation import risk_panel
from bond_rp.peer_learners import xgb_predict
from bond_rp.trading import monthly_performance
from rebuild_peer_reference import inputs

CASES=("dispersion","expectations")
DATES=pd.period_range("2017-01","2022-12",freq="M")
TARGETS=[f"rx{n}" for n in range(2,6)]
MU=[f"mu{n}" for n in range(2,6)]


def scores(actual,prediction,benchmark):
    error=(actual-prediction)**2
    base=(actual-benchmark)**2
    r2=1-error.sum(0)/base.sum(0)
    return dict(mean_maturity_r2_pct=100*r2.mean(),
        basket_r2_pct=100*(1-np.sum((actual.mean(1)-prediction.mean(1))**2)/
                          np.sum((actual.mean(1)-benchmark.mean(1))**2)),
        mse=float(error.mean()),max_abs_prediction_pct=float(np.abs(prediction).max()))


def read_predictions(folder):
    p=pd.read_csv(folder/"predictions.csv")
    p["ym"]=pd.PeriodIndex(p.ym,freq="M")
    return {name:g.set_index("ym")[MU] for name,g in p.groupby("model")}


def error_decomposition(bonds,baseline,current,output):
    records,worst,changes=[],[],[]
    actual=bonds.loc[DATES,TARGETS].to_numpy()
    covid=np.asarray((DATES>="2020-01")&(DATES<="2021-12"))
    for case in CASES:
        a=pickle.loads((baseline/case/"folds.pkl").read_bytes())
        b=pickle.loads((current/case/"folds.pkl").read_bytes())
        for date in a:
            # Controlled comparison: adding the L3 lag design must not mutate
            # the 15 contemporaneous XGB inputs or its training targets.
            np.testing.assert_array_equal(a[date]["X"],b[date]["X"])
            np.testing.assert_array_equal(a[date]["y"],b[date]["y"])
            np.testing.assert_array_equal(a[date]["x"].astype("float32"),b[date]["x"].astype("float32"))
        old,new=read_predictions(baseline/case),read_predictions(current/case)
        benchmark=old["EH"].loc[DATES].to_numpy()
        for name in ["L1_CP05","L1_CP15","L1_MS17","L1_Full","L1_Macro","L2_Incr","L3_Real","L3_GC","L4_Real","L4_GC"]:
            p=old[name].loc[DATES].to_numpy()
            loss=((actual-p)**2).sum(1)
            order=np.argsort(loss)[::-1]
            record=dict(case=case,model=name,**scores(actual,p,benchmark),
                sse_share_2020_2021=float(loss[covid].sum()/loss.sum()),
                sse_share_largest3=float(loss[order[:3]].sum()/loss.sum()),
                r2_excluding_2020_2021_pct=scores(actual[~covid],p[~covid],benchmark[~covid])["mean_maturity_r2_pct"])
            records.append(record)
            for j in order[:3]:
                worst.append(dict(case=case,model=name,formation=str(DATES[j]),annual_outcome=str(DATES[j]+12),
                    max_abs_actual_pct=float(np.abs(actual[j]).max()),max_abs_prediction_pct=float(np.abs(p[j]).max()),
                    sse_share=float(loss[j]/loss.sum())))
        for name in ("L3_Real","L3_GC","L4_Real","L4_GC"):
            a,b=old[name].loc[DATES].to_numpy(),new[name].loc[DATES].to_numpy()
            changes.append(dict(case=case,model=name,
                old_r2_pct=scores(actual,a,benchmark)["mean_maturity_r2_pct"],
                with_lag_r2_pct=scores(actual,b,benchmark)["mean_maturity_r2_pct"],
                max_forecast_change_pct=float(np.abs(a-b).max())))
    pd.DataFrame(records).to_csv(output/"error_concentration.csv",index=False)
    pd.DataFrame(worst).to_csv(output/"worst_forecast_months.csv",index=False)
    pd.DataFrame(changes).to_csv(output/"l3_lag_control.csv",index=False)
    return pd.DataFrame(records),pd.DataFrame(changes)


def macro_counterfactuals(bonds,features,baseline,output):
    """Hold all non-LN inputs/labels fixed; change only the stated LN mechanism.

    Known-X PCA includes only predictors available by the decision date, never
    future X or immature labels. It is a diagnostic of the chosen PCA window,
    not a claim that the peer used it. Cubic clipping is likewise an explicitly
    different model, not a silent alteration of the original paper.
    """
    folds={c:pickle.loads((baseline/c/"folds.pkl").read_bytes()) for c in CASES}
    pred={c:{v:[] for v in ("original","known_X_PCA","training_range_cube_clip",
        "fred10iqr_missing","fred10iqr_previous","omit_LN")} for c in CASES}
    tree_pred={c:{v:[] for v in pred[c]} for c in CASES}
    frozen={c:json.loads((baseline/c/"frozen_parameters.json").read_text())["L4_GC"] for c in CASES}
    mc=[c for c in features if c.startswith("macro::")]
    stress=[]
    for date in DATES:
        f=folds["dispersion"][date]
        train=features.loc[f["train_dates"]]
        history=features.loc["1994-01":str(date)]
        # McCracken-Ng FRED-MD section3 footnote5: flag transformed values
        # more than 10 IQR from the median. Bounds here use training X only.
        # Missing/mean is our existing imputer, not the original EM imputer.
        # Previous-value replacement follows Huang et al. section2, with the
        # threshold supplied by FRED-MD; the combined rule is a labelled test.
        quantile=train[mc].quantile([.25,.5,.75])
        width=10*(quantile.loc[.75]-quantile.loc[.25])
        width=width.where(width>1e-12,np.inf)
        clean=history[mc].mask((history[mc]-quantile.loc[.5]).abs()>width)
        previous=clean.ffill()
        panels={"matured":(train[mc],train[mc],features.loc[[date],mc]),
            "known_X":(history[mc],train[mc],features.loc[[date],mc]),
            "fred10iqr_missing":(clean.loc[train.index],clean.loc[train.index],clean.loc[[date]]),
            "fred10iqr_previous":(previous.loc[train.index],previous.loc[train.index],previous.loc[[date]])}
        factors={}
        for window,(fit_X,train_X,now_X) in panels.items():
            pipe=_numeric_pipeline(pca=True).fit(fit_X)
            Z,z=pipe.transform(train_X),pipe.transform(now_X)
            lo,hi=Z[:,0].min(),Z[:,0].max()
            if window=="matured":
                stress.append(dict(formation=str(date),F1=z[0,0],training_min=lo,training_max=hi,
                    cubic_relative_to_training_extreme=float(abs(z[0,0]/max(abs(lo),abs(hi)))**3)))
            for clip in ([False,True] if window=="matured" else [False]):
                a=np.clip(Z[:,0],lo,hi) if clip else Z[:,0]
                b=np.clip(z[:,0],lo,hi) if clip else z[:,0]
                X=np.column_stack([Z[:,0],a**3,Z[:,1:4],Z[:,7]])
                x=np.column_stack([z[:,0],b**3,z[:,1:4],z[:,7]])
                signal=[]
                for cols in ([0,1,3,4,5],list(range(6))):
                    fit=LinearRegression().fit(X[:,cols],f["y"].mean(1))
                    signal.append((fit.predict(X[:,cols]),fit.predict(x[:,cols])))
                key=("training_range_cube_clip" if clip else "known_X_PCA" if window=="known_X"
                     else "original" if window=="matured" else window)
                factors[key]=signal
        for case in CASES:
            fold=folds[case][date]
            for variant in pred[case]:
                X,x=fold["X"][:,:10].copy(),fold["x"][:,:10].copy()
                if variant=="omit_LN":
                    keep=[j for j in range(10) if j not in (7,8)]
                    X,x=X[:,keep],x[:,keep]
                elif variant!="original":
                    for j,(a,b) in zip((7,8),factors[variant]):
                        X[:,j],x[:,j]=a,b
                pred[case][variant].append(LinearRegression().fit(X,fold["y"]).predict(x)[0])
                tree_X,tree_x=fold["X"].copy(),fold["x"].copy()
                if variant=="omit_LN":
                    keep=[j for j in range(15) if j not in (7,8)]
                    tree_X,tree_x=tree_X[:,keep],tree_x[:,keep]
                elif variant!="original":
                    for j,(a,b) in zip((7,8),factors[variant]):
                        scale=max(float(a.std()),1e-8)
                        tree_X[:,j],tree_x[:,j]=(a-a.mean())/scale,(b-a.mean())/scale
                tree_pred[case][variant].append(xgb_predict(dict(X=tree_X,x=tree_x,y=fold["y"]),frozen[case])[0])
    records,rows=[],[]
    actual=bonds.loc[DATES,TARGETS].to_numpy()
    for case in CASES:
        references=read_predictions(baseline/case)
        benchmark=references["EH"].loc[DATES].to_numpy()
        for variant,values in pred[case].items():
            p=np.asarray(values)
            discrepancy=float(np.abs(p-references["L1_Full"].loc[DATES].to_numpy()).max()) if variant=="original" else np.nan
            records.append(dict(case=case,variant=variant,baseline_recompute_max_difference=discrepancy,**scores(actual,p,benchmark)))
            rows += [dict(case=case,variant=variant,ym=str(date),**dict(zip(MU,value))) for date,value in zip(DATES,p)]
    pd.DataFrame(stress).to_csv(output/"macro_extrapolation.csv",index=False)
    pd.DataFrame(records).to_csv(output/"macro_mechanism_controls.csv",index=False)
    pd.DataFrame(rows).to_csv(output/"macro_control_predictions.csv",index=False)
    tree_records=[]
    for case in CASES:
        refs=read_predictions(baseline/case)
        for variant,values in tree_pred[case].items():
            p=np.asarray(values)
            if variant=="original":
                np.testing.assert_allclose(p,refs["L4_GC"].loc[DATES].to_numpy(),atol=1e-9,rtol=1e-9)
            tree_records.append(dict(case=case,variant=variant,**scores(actual,p,refs["EH"].loc[DATES].to_numpy())))
    trees=pd.DataFrame(tree_records);trees.to_csv(output/"xgb_macro_mechanism_controls.csv",index=False)
    return pd.DataFrame(records),pd.DataFrame(stress),trees


def allocation_controls(bonds,baseline,output):
    """Same forecasts/risk history; compare plausible interpretations of slide16.

    The extra rules are diagnostic, not assertions about the peer's hidden code.
    All accounts below keep total allocation <=1, with no borrowing assumption.
    Annual overlapping payoffs are reported separately from monthly NAV.
    """
    mean,variance,_=risk_panel(bonds,DATES)
    actual=bonds.loc[DATES,[f"mr{n}" for n in range(2,6)]].to_numpy()/100
    annual=bonds.loc[DATES,TARGETS].to_numpy()/100
    rf=bonds.rf_month.reindex(DATES).to_numpy()/100
    records=[]
    for case in CASES:
        p=read_predictions(baseline/case)
        forecasts={"RollingMV":mean,"L4_GC":p["L4_GC"].loc[DATES].to_numpy()/100}
        for model,mu in forecasts.items():
            raw=np.maximum(mu,0)/(5*variance)
            weights={"proportional_cap1":raw/np.maximum(raw.sum(1,keepdims=True),1),
                     "four_independent_accounts":np.minimum(raw,1)/4}
            for method,w in weights.items():
                returns=rf+np.sum(w*(actual-rf[:,None]),1)
                diagnostic=np.sum(w*annual,1)
                records.append(dict(case=case,model=model,allocation=method,**monthly_performance(returns,rf),
                    mean_exposure=w.sum(1).mean(),annual_payoff_sum_div12_pct=100*diagnostic.sum()/12,
                    actual_monthly_excess_sum_pct=100*np.sum(returns-rf)))
        old=pd.read_csv(baseline/case/"performance.csv")
        for model in forecasts:
            r=old.loc[old.model.eq(model+"/direct")].iloc[0]
            records.append(dict(case=case,model=model,allocation="current_joint_KKT_cap1",cagr_pct=r.cagr_pct,
                sharpe=r.sharpe,max_dd_pct=r.max_dd_pct,mean_exposure=r.mean_exposure,
                annual_payoff_sum_div12_pct=r.overlapping_payoff_sum_div12_pct,
                actual_monthly_excess_sum_pct=r.cumulative_cash_excess_sum_pct))
    table=pd.DataFrame(records);table.to_csv(output/"allocation_interpretations.csv",index=False)
    return table


def period_controls(bonds,baseline,output):
    records=[]
    for case in CASES:
        p=read_predictions(baseline/case)
        for name in ("L1_FB","L1_CP05","L1_MS17","L1_CP15","L1_Full","L4_GC"):
            for label,dates in [("formation_2017_2022",DATES),
                ("formation_2017_2021_outcomes_by_2022",pd.period_range("2017-01","2021-12",freq="M")),
                ("outcome_year_2017_2022",pd.period_range("2016-01","2021-12",freq="M"))]:
                records.append(dict(case=case,model=name,convention=label,n=len(dates),
                    purchase_start=str(dates[0]),purchase_end=str(dates[-1]),last_outcome=str(dates[-1]+12),
                    causal_frozen_xgb=(name!="L4_GC" or dates[0]>=pd.Period("2017-01","M")),
                    **scores(bonds.loc[dates,TARGETS].to_numpy(),p[name].loc[dates].to_numpy(),p["EH"].loc[dates].to_numpy())))
    t=pd.DataFrame(records);t.to_csv(output/"date_convention_controls.csv",index=False)
    return t


def alignment_table(output,n_macro):
    rows=[
        ("6-7","1994-2010训练/2011-2016验证/2017-2022测试","已对齐文字分段","完整验证段；按标签兑现可用性冻结参数"),
        ("6-7","扩展估计、12个月隔离、训练集预处理","已实现但边界定义未核实","严格purchase+12<decision；PCA只用成熟训练行，是否等于对方训练集含义仍未知"),
        ("11","L3为每个预测变量加入滞后项","原实现遗漏，本轮已补","15个当期＋15个滞后；滞后1月为默认解释，PPT未写具体长度；L4预测不变"),
        ("12","L4同L2_Incr的15个当期变量；深度2、100树","结构已对齐","完整15列成员由第13页图和总数推定；未取得逐列原始数据"),
        ("4-5","82候选、非均匀模板/窗口、PC1-5解释93%","未对齐原始候选池","本项目20×4＋2是自定模板池；筛出17项、PC解释89.29%，不是对方同一输入"),
        ("3,13","LN宏观面板与复合因子","面板及处理未对齐",f"当前FRED-MD为{n_macro}列，PPT为127列；滞后2月、无源数据异常清洗；疫情三次项外推失控"),
        ("3","ME标为Eriksen但描述分歧","定义冲突未消除","原文中位数预期与PPT的IQR代理并列，NGDP已按原表修正；代理不是对方已确认公式"),
        ("11-12","动态EN、滚动Gaussian Copula、强正则","方法已实现，数值设定未核实","具体CV长度、GC边缘/窗口/合成量、L1/L2范围与种子均未知"),
        ("16-19","逐期限MV和B/lambda_dd/lambda_g","只对齐原始MV公式，执行层未复现","gamma5、总上限1、KKT投影、指数回撤刹车和L2仓位惩罚为本项目假设"),
        ("2,6,20","年度目标与2017-2022测试、72个月","测试日期标签口径未核实","本项目以买入月定测试，年度结果延伸2023-12；若按兑现月则对应2016-2021买入"),
        ("20","累计超额、年化收益、Sharpe","尚未同口径","本项目真实月度总收益CAGR；PPT累计超额除6，现金/累计/年化公式未披露"),
        ("16,20","60月滚动MV基准SR0.354","未数值复现","本项目同期间基准超额SR-0.124；不含XGB也未匹配，不能只归因模型调参不足")]
    t=pd.DataFrame(rows,columns=["pdf_pages","peer_statement","status","current_and_evidence"])
    t.to_csv(output/"protocol_alignment.csv",index=False)
    return t


def relative_loss_control(bonds,baseline,output):
    """A common R2 denominator cannot reverse two models' SSE ranking."""
    peer_tree=np.array([.382,.327,.295,.267])
    peer_cycle=np.array([.114,.128,.136,.143])
    actual=bonds.loc[DATES,TARGETS].to_numpy()
    rows=[]
    for case in CASES:
        p=read_predictions(baseline/case)
        ratio=((actual-p["L4_GC"].loc[DATES].to_numpy())**2).sum(0)/((actual-p["L1_CP15"].loc[DATES].to_numpy())**2).sum(0)
        for n,a,b in zip(range(2,6),ratio,(1-peer_tree)/(1-peer_cycle)):
            rows.append(dict(case=case,maturity=n,our_tree_to_cp15_sse=a,peer_tree_to_cp15_sse=b,
                explanation="Within each experiment both models share its own EH denominator; ranking is denominator-independent"))
    t=pd.DataFrame(rows);t.to_csv(output/"relative_loss_control.csv",index=False)
    return t


def figure(bonds,baseline,stress):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,layout="constrained")
    dates=DATES.to_timestamp()
    axes[0].fill_between(dates,stress.training_min,stress.training_max,color="#bdd5ea",alpha=.6,label="Training F1 range")
    axes[0].plot(dates,stress.F1,color="#a33b32",label="Decision-month F1")
    axes[0].set_ylabel("Standardized first macro PC")
    axes[0].set_title("Macro inputs exceed the training range in 2020")
    axes[0].legend(loc="lower left");axes[0].grid(alpha=.15)
    p=read_predictions(baseline/"expectations")
    for name,color in [("L1_Full","#a33b32"),("L4_GC","#297c80")]:
        axes[1].plot(dates,np.abs(p[name].loc[DATES].to_numpy()).max(1),color=color,label=name+" max |forecast|")
    axes[1].plot(dates,np.abs(bonds.loc[DATES,TARGETS]).max(1),color="#454545",label="Max |realized annual excess return|")
    axes[1].set_yscale("log");axes[1].set_ylabel("Absolute annual percentage points (log)")
    axes[1].set_title("Annual forecast magnitudes; expectation ME definition")
    axes[1].legend(loc="upper left");axes[1].grid(alpha=.15)
    fig.suptitle("Replication diagnosis: covariate extrapolation, not a new trading result")
    fig.savefig(ROOT/"assets"/"peer_diagnosis.png",dpi=160);plt.close(fig)


def report(output,alignment,errors,lag,macro,allocation,period,trees):
    lines=['<a id="replication-diagnosis"></a>',"### 5.10 为什么还没有复现对方结果：逐项对照与控制实验","",
        "**目前没有完全按照同一数据和同一执行公式复现。** 本轮发现了PPT明确写出但先前漏做的L3滞后项，",
        "其余差异中既有未披露细节，也有本项目自行加入的假设。测试通过证明当前实现满足检查，不能证明与对方程序一致。","",
        "| PDF页 | 对方披露 | 当前状态 | 可核查差异 |","| --- | --- | --- | --- |"]
    for _,r in alignment.iterrows():
        lines.append(f"| {r.pdf_pages} | {r.peer_statement} | {r.status} | {r.current_and_evidence} |")
    lines += ["", "**① L3遗漏是实际错误，但不是L4差距的原因。** 已补齐30列并在每个内外折重建，",
        "滞后值与当期值用同一个训练得到的因子/PCA基底，不拼接不同月份重估后的PC。",
        "84项测试通过；所有外层XGB训练输入、目标逐项一致，预测输入转float32后也一致，",
        "L4预测的最大变化为0。补齐L3仍未得到PPT的正R²，不能把这项修复包装成高绩效复现。","",
        "| ME版本 | 模型 | 修复前平均OOS R² (%) | 补滞后后 (%) |","| --- | --- | --- | --- |"]
    labels={"dispersion":"PPT分歧代理","expectations":"Eriksen预期"}
    for _,r in lag.iterrows():
        lines.append(f"| {labels[r.case]} | {r.model} | {r.old_r2_pct:.2f} | {r.with_lag_r2_pct:.2f} |")
    lines += ["", "**② 巨大负R²的直接数值机制是疫情宏观输入外推。** 2020-06形成预测时，",
        "F1为-36.34，成熟训练范围约[-5.52,1.83]；三次项相对训练极值放大约285倍。",
        "原文三次项不是程序错误；将未经对应异常处理的现代疫情面板代入它，是尚未对齐的数据适配。","",
        "| ME版本 | 原线性Full最大预测绝对值 (%) | 平方误差集中于2020-2021 (%) | 最大3个月占比 (%) |",
        "| --- | --- | --- | --- |"]
    for _,r in errors[errors.model.eq("L1_Full")].iterrows():
        lines.append(f"| {labels[r.case]} | {r.max_abs_prediction_pct:.1f} | {100*r.sse_share_2020_2021:.4f} | {100*r.sse_share_largest3:.4f} |")
    lines += ["", "仅改变LN构造、保持其他列及收益标签不变的机制诊断如下。`known_X_PCA`允许PCA使用决策时已知的",
        "未成熟标签对应的X，未使用未来X/Y；`training_range_cube_clip`只限制三次项输入。",
        "`fred10iqr_missing`使用训练中位数±10IQR标记异常后用现有训练均值填补；",
        "`fred10iqr_previous`改用当时已知的前值。后者结合了FRED-MD阈值与读本Huang等§2的前值替代原则，",
        "不是宣称对方采用了这个组合，也不是原始FRED-MD的EM缺失值算法。",
        "阈值依据为[McCracken–Ng原文脚注5](https://files.stlouisfed.org/files/htdocs/wp/2015/2015-012.pdf)，",
        "既有项目6IQR保护与这里的10IQR来源规则分别报告。","",
        "| ME版本 | LN机制控制 | Full平均OOS R² (%) | 最大预测绝对值 (%) |",
        "| --- | --- | --- | --- |"]
    for _,r in macro.iterrows():
        lines.append(f"| {labels[r.case]} | {r.variant} | {r.mean_maturity_r2_pct:.2f} | {r.max_abs_prediction_pct:.2f} |")
    lines += ["", "同一批LN输入变动另应用到L4_GC，冻结原XGB和GC参数、不重新搜索，结果如下。其他当期特征保持不变，",
        "不能将它解释为对所有宏观输入/82候选同时清洗的实验。","",
        "| ME版本 | 固定参数L4_GC输入控制 | 平均OOS R² (%) |","| --- | --- | --- |"]
    for _,r in trees.iterrows():
        lines.append(f"| {labels[r.case]} | {r.variant} | {r.mean_maturity_r2_pct:.2f} |")
    lines += ["", "清理异常输入能消除数千个百分点的失控预测，但仍未复现正R²；单纯增加PCA可用历史X也没有解决问题。",
        "这些是机制控制，不按测试成绩挑其中一个替换主模型。疫情之外的L4点估计也不能替代完整测试。","",
        "![疫情宏观外推机制诊断](assets/peer_diagnosis.png)","",
        "**③ 日期定义也未完全核实。** 以兑现年2017–2022而非买入年定义测试，CP15平均R²从17.95%变为12.68%，",
        "接近PPT的13.03%，但四期限逐项并不匹配，不能据此声称已找到对方口径。",
        "L4在该日期解释下也未复现；该L4行使用2016年的买入预测而参数在2017冻结，",
        "只作为日期敏感性诊断，不能当作2016年可执行的样本外策略。","",
        "另直接比较L4_GC与CP15的平方误差：对方四个期限的L4_GC均优于CP15，本项目则均劣于CP15。",
        "两模型在各自实验中共享EH分母；这种排序反转不可能仅由换一个R²基准分母解释。",
        "逐期限误差比保存在`relative_loss_control.csv`，没有将两组实际收益序列假设为相同。","",
        "**④ 投资结果尚未建立同口径比较。** PPT的无模型RollingMV基准SR为0.354，我们的真实月度超额SR为-0.124。",
        "基准不依赖XGB，因此差异不能只归结为没有调好XGB。当前KKT联合上限1、按比例缩放、四个独立期限账户",
        "是三种不同的预算解释；均保持同样预测、60月成熟风险历史、gamma5、零费用，结果如下。","",
        "| ME版本 | 模型 | 预算解释 | 总收益CAGR (%) | 月度超额SR | 月末回撤 (%) |",
        "| --- | --- | --- | --- | --- | --- |"]
    for _,r in allocation.iterrows():
        lines.append(f"| {labels[r.case]} | {r.model} | {r.allocation} | {r.cagr_pct:.2f} | {r.sharpe:.3f} | {r.max_dd_pct:.2f} |")
    lines += ["", "这几种解释均未匹配对方无模型基准，不选最接近其数字的一行充当复现。",
        "本项目CAGR包含现金利息；PPT列名为累计超额且用累计数除6，不是同一指标。",
        "年度预测驱动月度再平衡是本项目的执行选择，PPT没有披露足够的持有/累计公式来确认它相同。","",
        "**可以成立的判断。** 已找到并修复一个明确遗漏，定位了现代宏观输入外推的主导损失机制，",
        "也证实候选池、日期锚点和投资统计尚未逐项匹配。因此本次失败不能证明对方策略不可能，",
        "不能证明这些因子没有投资价值；也不能凭现有证据认定对方存在泄漏。",
        "下一步需要核对的最小原始对象是82列/筛后列名与变换、原始rx与日期索引、逐月预测与权重，以及实际净值累计代码。","",
        "复核：先保留本轮前的`output/peer_reference_before_l3_lag/`（代码版本3d020ce），运行修复后的研究，",
        "再执行`python scripts/diagnose_peer_replication.py --update-readme`。该命令不搜索测试期最优策略。",
        "未使用新的独立测试样本，所有结论均为回溯诊断；原始PDF、数据及中间输出仍不上传。"]
    body="\n".join(lines)+"\n";(output/"report.md").write_text(body)
    return body


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir",type=Path,default=ROOT/"output"/"peer_reference_before_l3_lag")
    parser.add_argument("--current-dir",type=Path,default=ROOT/"output"/"peer_reference")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"output"/"peer_replication_diagnosis")
    parser.add_argument("--update-readme",action="store_true")
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    bonds,features,*_=inputs()
    errors,lag=error_decomposition(bonds,args.baseline_dir,args.current_dir,args.output_dir)
    macro,stress,trees=macro_counterfactuals(bonds,features,args.baseline_dir,args.output_dir)
    allocation=allocation_controls(bonds,args.baseline_dir,args.output_dir)
    period=period_controls(bonds,args.baseline_dir,args.output_dir)
    relative_loss_control(bonds,args.baseline_dir,args.output_dir)
    alignment=alignment_table(args.output_dir,len([c for c in features if c.startswith("macro::")]))
    figure(bonds,args.baseline_dir,stress)
    body=report(args.output_dir,alignment,errors,lag,macro,allocation,period,trees)
    if args.update_readme:
        path=ROOT/"README.md";text=path.read_text()
        start,end="<!-- REPLICATION_DIAGNOSIS:START -->","<!-- REPLICATION_DIAGNOSIS:END -->"
        block=start+"\n"+body+end+"\n\n"
        if start in text:
            a,b=text.index(start),text.index(end)+len(end);text=text[:a]+block.rstrip()+text[b:]
        else:
            k=text.index("## 6.");text=text[:k]+block+text[k:]
        path.write_text(text)
    print("L3 lag control\n",lag.to_string(index=False),flush=True)
    print("LN mechanism controls\n",macro.to_string(index=False),flush=True)
    print("Fixed-parameter XGB LN controls\n",trees.to_string(index=False),flush=True)
    print("Allocation definitions\n",allocation.to_string(index=False),flush=True)
    metadata=dict(status="completed",purpose="retrospective cause diagnosis, no test-period model selection",
        baseline=str(args.baseline_dir),current=str(args.current_dir),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        caveat="Outcome-year 2017-2022 XGB uses 2016 purchase forecasts with parameters frozen in 2017: descriptive date sensitivity only")
    (args.output_dir/"diagnosis.json").write_text(json.dumps(metadata,indent=2))


if __name__=="__main__":
    main()
