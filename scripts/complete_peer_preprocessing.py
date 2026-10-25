"""Complete the macro-preprocessing ablation and compare BOTH peer heatmaps.

Keep the archived raw reconstruction immutable. Refit two declared cleaning
branches, retune with the same 32-trial learner budget, then select using only
annual labels that have matured before 2017. Test results are retrospective.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache"/"matplotlib"))
import numpy as np
import pandas as pd
from bond_rp.peer_preprocessing import METHODS, rebuild_macro_fold
from bond_rp.peer_reference import peer_fold
from bond_rp.peer_learners import elastic_predict, xgb_predict
from bond_rp.peer_allocation import allocation_path, summarize
from rebuild_peer_reference import inputs, search_xgb, economic_evaluation, source_hashes, versions
from export_peer_reference import PEER_R2

PEER_VALIDATION = {
    "L1_FB":[-1.363,-.838,-.356,-.108], "L1_CP05":[.421,.268,.112,.017],
    "L1_MS17":[.361,.191,.043,-.039], "L1_CP15":[.385,.605,.718,.748],
    "L1_Macro":[.220,.454,.645,.712], "L1_Full":[-1.848,-1.359,-.336,.210],
    "L2_PCA":[-.676,-.575,-.252,-.080], "L2_Incr":[-4.292,-3.639,-1.733,-.552],
    "L3_Real":[.009,-.021,.161,.250], "L3_GC":[-1.391,-.827,-.095,.274],
    "L4_Real":[.294,.451,.575,.514], "L4_GC":[.347,.496,.602,.634]}
MU = [f"mu{n}" for n in range(2,6)]
TARGETS = [f"rx{n}" for n in range(2,6)]
DATES = pd.period_range("2011-01","2022-12",freq="M")
VALIDATION = DATES[DATES+12 < pd.Period("2017-01","M")]
TEST = DATES[DATES >= pd.Period("2017-01","M")]


def predictions(folder):
    p = pd.read_csv(folder/"predictions.csv")
    p["ym"] = pd.PeriodIndex(p.ym,freq="M")
    return {name:g.set_index("ym")[MU] for name,g in p.groupby("model")}


def fingerprint(baseline):
    files = [Path(__file__),ROOT/"bond_rp"/"peer_preprocessing.py"]
    for case in ("dispersion","expectations"):
        files += [baseline/case/n for n in ("fit_metadata.json","frozen_parameters.json","folds.pkl")]
        metadata = json.loads((baseline/case/"fit_metadata.json").read_text())
        assert metadata["sources"] == source_hashes() and metadata["versions"] == versions()
    return dict(base=source_hashes(),versions=versions(),
        added={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        methods=list(METHODS),candidate_membership="same pre-2010 frozen 17-column set",
        trials_per_learner=32,threshold="training median +/- 10 IQR; zero IQR unbounded",
        mean_imputation="explicit approximation, not the original FRED-MD EM algorithm",
        selection="minimum mean four-maturity validation MSPE / EH MSPE; labels mature before 2017",
        status="retrospective; all test history has already been inspected")


def run_branch(case, method, baseline, output, data, manifest, inner_cache):
    bonds,features,candidates,selected,_,surveys = data
    folder = output/case/method; folder.mkdir(parents=True,exist_ok=True)
    key = json.dumps(dict(manifest=manifest,case=case,method=method),sort_keys=True)
    if (folder/"completed.json").exists() and (folder/"completed.json").read_text() == key:
        print(f"Reuse completed {case}/{method}",flush=True); return
    base = pickle.loads((baseline/case/"folds.pkl").read_bytes())
    fold_file = folder/"folds.pkl"
    if fold_file.exists() and (folder/"fold_key.txt").read_text() == key:
        folds = pickle.loads(fold_file.read_bytes())
    else:
        folds = {}
        for date,fold in base.items():
            folds[date] = rebuild_macro_fold(fold,features,candidates,selected,method)
        fold_file.write_bytes(pickle.dumps(folds,protocol=pickle.HIGHEST_PROTOCOL))
        (folder/"fold_key.txt").write_text(key)
    print(f"Rebuilt macro/candidate PCs: {case}/{method}",flush=True)
    frozen = {}
    for label in ("L4_Real","L4_GC"):
        frozen[label],_ = search_xgb(folds,VALIDATION,bonds,folder,label)
    (folder/"frozen_parameters.json").write_text(json.dumps(frozen,indent=2))
    records,enet = [],[]
    for date,fold in folds.items():
        predicted = dict(fold["reference"])
        fold["inner_folds"] = []
        for offset in (36,24,12):
            vd = fold["train_dates"][-offset:][:12]
            start,end = vd[0],vd[-1]
            cache_key = (case,start,end)
            if cache_key not in inner_cache:
                inner_cache[cache_key] = peer_fold(bonds,features,surveys[case],candidates,selected,
                    start,predict_dates=vd,include_lags=True)
            inner = rebuild_macro_fold(inner_cache[cache_key],features,candidates,selected,method)
            inner["validation_y"] = bonds.loc[vd,TARGETS].to_numpy()
            fold["inner_folds"].append(inner)
        for label in ("L3_Real","L3_GC"):
            predicted[label],audit = elastic_predict(fold,copula=label.endswith("GC"))
            enet.append(dict(ym=str(date),model=label,**audit))
        for label in ("L4_Real","L4_GC"):
            predicted[label] = xgb_predict(fold,frozen[label])
        for label,values in predicted.items():
            records.append(dict(ym=str(date),model=label,**dict(zip(MU,map(float,values[0])))))
        if date.month == 1:
            print(f"  {case}/{method} predictions {date}",flush=True)
        del fold["inner_folds"]
    p = pd.DataFrame(records)
    assert len(p) == 13*144 and not p.duplicated(["ym","model"]).any()
    p.to_csv(folder/"predictions.csv",index=False)
    pd.DataFrame([f["audit"] for f in folds.values()]).to_csv(folder/"forecast_audit.csv",index=False)
    (folder/"elasticnet_cv_audit.json").write_text(json.dumps(enet,indent=2))
    economic_evaluation(bonds,p,folder)
    (folder/"completed.json").write_text(key)


def evaluate(baseline,output,bonds):
    rows,performance,choices,inference = [],[],[],[]
    folders = {}
    for case in ("dispersion","expectations"):
        for method in METHODS:
            folder = baseline/case if method == "raw" else output/case/method
            folders[(case,method)] = folder
            p = predictions(folder)
            for period,dates in (("validation_matured",VALIDATION),("validation_full_descriptive",DATES[:72]),("test",TEST)):
                actual = bonds.loc[dates,TARGETS].to_numpy()
                eh = p["EH"].loc[dates].to_numpy()
                den = ((actual-eh)**2).sum(0)
                for name,frame in p.items():
                    pred = frame.loc[dates].to_numpy()
                    r2 = 1-((actual-pred)**2).sum(0)/den
                    for j,n in enumerate(range(2,6)):
                        peer = PEER_VALIDATION if period.startswith("validation") else PEER_R2
                        rows.append(dict(case=case,method=method,period=period,model=name,maturity=n,n=len(dates),
                            r2_pct=100*r2[j],mse=float(((actual-pred)**2)[:,j].mean()),
                            max_abs_prediction_pct=float(np.abs(pred).max()),
                            peer_r2_pct=100*peer[name][j] if name in peer else np.nan,
                            eligible_for_tuning=period == "validation_matured"))
            perf = pd.read_csv(folder/"performance.csv")
            for record in perf.to_dict("records"):
                performance.append(dict(case=case,method=method,**record))
        score = pd.DataFrame(rows)
        val = score[score.case.eq(case)&score.period.eq("validation_matured")&score.model.eq("L4_GC")]
        ranking = val.groupby("method").r2_pct.mean().sort_values(ascending=False)
        # Stable tie order retains raw, then mean, then previous.
        selected = max(METHODS,key=lambda m:ranking[m])
        choices.append(dict(case=case,method=selected,validation_mean_r2_pct=float(ranking[selected]),
            validation_last_purchase=str(VALIDATION[-1]),validation_last_label_end=str(VALIDATION[-1]+12),
            rules="choose preprocessing on L4_GC validation loss only; no test metric or investment result used"))
        reference = pd.read_csv(baseline/case/"trades.csv")
        base = reference[reference.model.eq("RollingMV/direct")].return_net.to_numpy()
        rng = np.random.default_rng(42)
        starts = rng.integers(0,72,(1999,6))
        ix = ((starts[:,:,None]+np.arange(12))%72).reshape(1999,72)
        for method in METHODS:
            trades = pd.read_csv(folders[(case,method)]/"trades.csv")
            for model in ("L4_GC/direct","L4_GC/optimized"):
                g = trades[trades.model.eq(model)]
                ret = g.return_net.to_numpy()
                cer = lambda a:12*(a.mean(axis=-1)-2.5*a.var(axis=-1,ddof=1))
                draws = 10000*(cer(ret[ix])-cer(base[ix]))
                inference.append(dict(case=case,method=method,model=model,
                    delta_cer_bps=float(10000*(cer(ret)-cer(base))),
                    ci_low=float(np.quantile(draws,.025)),ci_high=float(np.quantile(draws,.975)),
                    block_months=12,replications=1999,conditional_only=True))
    scores,perf = pd.DataFrame(rows),pd.DataFrame(performance)
    scores.to_csv(output/"all_scores.csv",index=False); perf.to_csv(output/"all_performance.csv",index=False)
    pd.DataFrame(inference).to_csv(output/"conditional_utility.csv",index=False)
    (output/"selection.json").write_text(json.dumps(choices,indent=2))
    return scores,perf,choices,folders


def report(output,scores,perf,choices,folders):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    colors = dict(raw="#7b8793",fred10_mean="#197b78",fred10_previous="#a578b7")
    for j,case in enumerate(("dispersion","expectations")):
        for method in METHODS:
            part = scores[scores.case.eq(case)&scores.method.eq(method)&scores.model.eq("L4_GC")]
            values = [part[part.period.eq(p)].r2_pct.mean() for p in ("validation_matured","test")]
            axes[0,j].plot([0,1],values,"o-",label=method,color=colors[method])
            t = pd.read_csv(folders[(case,method)]/"trades.csv")
            g = t[t.model.eq("L4_GC/direct")]
            axes[1,j].plot(pd.PeriodIndex(g.sale,freq="M").to_timestamp(),np.cumprod(1+g.return_net),
                           label=method,color=colors[method])
        axes[0,j].set_xticks([0,1],["Matured validation (60)","Test (72)"])
        axes[0,j].set_ylabel("Mean maturity OOS R2 (%)"); axes[0,j].set_title(case)
        axes[0,j].axhline(0,color="#aaa",lw=.7)
        axes[1,j].set_ylabel("Actual monthly wealth; initial = 1")
        for ax in axes[:,j]:
            ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.2)
    fig.suptitle("Same factor membership and search budget; preprocessing selected on validation only\nDepth 2 / 100 trees, separate training-only thresholds in every outer and inner fold")
    fig.savefig(ROOT/"assets"/"peer_preprocessing.png",dpi=170);plt.close(fig)
    lines = ['<a id="peer-preprocessing"></a>',"### 5.11 对照完整提取 Markdown：补齐预处理与验证期差异","",
        "本轮继续依据用户提供的 PDF/Markdown 核对。原始 §5.8 重建与失败结果保留；以下是明确标记的预处理改进，",
        "不是对方未公开代码的精确复现。双方在疫情前的验证期也存在差异，不能只把测试差距归因于疫情或调参不足。","",
        "**验证期的两种口径。** 冻结于2017年初时，2011–2015形成的60个年度标签已兑现，可用于调参；",
        "2011–2016全部72个预测也另外评价，以便对应PPT热力图，但其2016年标签在2017年陆续兑现，",
        "只作事后诊断，不参与选模。PPT未明确热力图日期指形成日还是兑现日，不能据此断言其使用了未来信息。","",
        "| 模型 | PPT验证均值R² (%) | 本次60标签：预期ME | 本次72标签：预期ME（描述） |",
        "| --- | --- | --- | --- |"]
    for model in ("L1_FB","L1_CP05","L1_MS17","L1_CP15","L1_Full","L3_GC","L4_Real","L4_GC"):
        q = scores[scores.case.eq("expectations")&scores.method.eq("raw")&scores.model.eq(model)]
        a = q[q.period.eq("validation_matured")].r2_pct.mean()
        b = q[q.period.eq("validation_full_descriptive")].r2_pct.mean()
        lines.append(f"| {model} | {100*np.mean(PEER_VALIDATION[model]):.2f} | {a:.2f} | {b:.2f} |")
    lines += ["","**完整折内清洗。** [FRED-MD原文脚注5](https://files.stlouisfed.org/files/htdocs/wp/2015/2015-012.pdf)规定，",
        "将偏离中位数超过10倍IQR的变换后观测视为缺失。本轮阈值只从成熟训练行估计，分别用于原宏观面板和17个已筛选模板特征，",
        "在提取LN与模板PC之前清洗，再重新构建LN5/LN6及L3的当期＋滞后项。",
        "`fred10_mean`随后使用训练均值填补，属于对原文EM填补的近似；`fred10_previous`使用历史前值，",
        "结合读本Huang等§2的前值替换原则。两者均不声称为PPT的未披露预处理。候选成员、数据版本、CP/MS/CP15/ME与标签保持相同。",
        "每个内层折也独立清洗和提取因子；不把外层阈值或PC带入早期内层验证。",
        "两个ME定义×两种新清洗×Real/GC各32个XGB候选，共新增256次参数试验；深度2、100树保持一致。",
        "EN仍用动态隔离CV，执行层每个分支128次验证期搜索。所有分支均公布，不按测试结果选清洗方法。","",
        "| ME定义 | 清洗 | 验证L4_GC均值R² (%) | 测试L4_GC均值R² (%) | 测试Full均值R² (%) |",
        "| --- | --- | --- | --- | --- |"]
    for case in ("dispersion","expectations"):
        for method in METHODS:
            q = scores[scores.case.eq(case)&scores.method.eq(method)]
            pick = lambda m,p:q[q.model.eq(m)&q.period.eq(p)].r2_pct.mean()
            lines.append(f"| {case} | {method} | {pick('L4_GC','validation_matured'):.2f} | {pick('L4_GC','test'):.2f} | {pick('L1_Full','test'):.2f} |")
    lines += ["","**验证期选择及投资结果。** 下表同时报告原始与两种改进的零成本真实月度账户；",
        "信号仍预测一年收益，逐月按同一债券剩余期限重定价。CAGR、夏普与重叠年标签指标分开。","",
        "| ME定义 | 清洗 | 执行方式 | CAGR (%) | 超额夏普 | 月末回撤 (%) |",
        "| --- | --- | --- | --- | --- | --- |"]
    for case in ("dispersion","expectations"):
        for method in METHODS:
            for model in ("L4_GC/direct","L4_GC/optimized"):
                r = perf[perf.case.eq(case)&perf.method.eq(method)&perf.model.eq(model)].iloc[0]
                lines.append(f"| {case} | {method} | {model.split('/')[1]} | {r.cagr_pct:.2f} | {r.sharpe:.3f} | {r.max_dd_pct:.2f} |")
    lines += [""]
    for choice in choices:
        case,method = choice["case"],choice["method"]
        r = perf[perf.case.eq(case)&perf.method.eq(method)&perf.model.eq("L4_GC/direct")].iloc[0]
        lines.append(f"仅按成熟验证标签选择：**{case} → {method}**，验证平均R²={choice['validation_mean_r2_pct']:.2f}%；"
            f"其测试直接配置CAGR={r.cagr_pct:.2f}%、超额夏普={r.sharpe:.3f}。")
    lines += ["","![预处理改进的验证表现与真实月度净值](assets/peer_preprocessing.png)","",
        "异常处理修复的是极值外推的数值稳定性，是否改善预测和配置分别由表中结果判断；不将预测R²改善自动解释为可交易超额收益。",
        "区块bootstrap效用区间、单边0.5/2/5bps压力、逐期限全部分支与参数日志保存在本地`output/peer_preprocessing/`。",
        "区间仅条件于已选择的分支，未覆盖此前全部研究搜索；所有测试历史已经接触，新增结果仍为回溯探索。",
        "原82项模板清单、ME聚合、GC设置及精确投资方程仍未披露；单一清洗实验不能据此确认剩余复现差距的唯一原因。"]
    text = "\n".join(lines)+"\n"
    (output/"README_section.md").write_text(text)
    readme = ROOT/"README.md"; content = readme.read_text()
    begin,end = "<!-- BEGIN PEER_PREPROCESSING -->","<!-- END PEER_PREPROCESSING -->"
    block = begin+"\n"+text+end
    if begin in content:
        start,finish = content.index(begin),content.index(end)+len(end)
        content = content[:start]+block+content[finish:]
    else:
        marker = "## 6."
        position = content.index(marker)
        content = content[:position]+block+"\n\n"+content[position:]
    readme.write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,default=ROOT/"output"/"peer_reference")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"output"/"peer_preprocessing")
    parser.add_argument("--report-only",action="store_true")
    args = parser.parse_args(); args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest = fingerprint(args.baseline)
    (args.output_dir/"protocol.json").write_text(json.dumps(manifest,indent=2))
    data = inputs(); cache = {}
    if not args.report_only:
        for case in ("dispersion","expectations"):
            for method in METHODS[1:]:
                run_branch(case,method,args.baseline,args.output_dir,data,manifest,cache)
    scores,perf,choices,folders = evaluate(args.baseline,args.output_dir,data[0])
    report(args.output_dir,scores,perf,choices,folders)
    print(json.dumps(dict(status="finished",selection=choices),indent=2),flush=True)


if __name__ == "__main__":
    main()
