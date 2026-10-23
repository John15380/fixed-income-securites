"""Audit and report the source-mapped reconstruction against reference pages 1--22."""
import argparse
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
from statsmodels.stats.multitest import multipletests
from bond_rp.inference import joint_block_bootstrap_t
from bond_rp.metrics import loss_test,oos_r2
from bond_rp.peer_allocation import allocation_path,risk_panel,summarize
from bond_rp.peer_learners import augment,xgb_predict
from sklearn.linear_model import ElasticNet
from bond_rp.validated_models import load_long_market
from rebuild_peer_reference import PROTOCOL,source_hashes,versions

PEER_R2={"L1_FB":[-.082,-.076,-.079,-.084],"L1_CP05":[.018,.024,.027,.028],
    "L1_MS17":[.036,.035,.034,.034],"L1_CP15":[.114,.128,.136,.143],
    "L1_Macro":[.071,.090,.101,.106],"L1_Full":[.198,.156,.130,.120],
    "L2_PCA":[.121,.032,-.010,-.023],"L2_Incr":[-.132,-.219,-.231,-.214],
    "L3_Real":[.004,.031,.057,.072],"L3_GC":[.291,.251,.245,.259],
    "L4_Real":[.318,.267,.214,.171],"L4_GC":[.382,.327,.295,.267]}


def audit_and_scores(output,bonds):
    counts=dict(cases=0,forecast_rows=0,outer_folds=0,trades=0,xgb_trials=0,execution_trials=0)
    score_rows,losses,utility_rows,validation_rows=[],[],[],[]
    dates=pd.period_range("2017-01","2022-12",freq="M")
    actual=bonds.loc[dates,[f"rx{n}" for n in range(2,6)]].to_numpy()
    for case in PROTOCOL["cases"]:
        folder=output/case
        metadata=json.loads((folder/"fit_metadata.json").read_text())
        assert metadata["sources"]==source_hashes() and metadata["versions"]==versions()
        assert metadata["protocol"]==PROTOCOL
        assert json.loads((folder/"evaluation_metadata.json").read_text())==metadata
        p=pd.read_csv(folder/"predictions.csv");p["ym"]=pd.PeriodIndex(p.ym,freq="M")
        assert not p.duplicated(["ym","model"]).any()
        assert np.isfinite(p[[f"mu{n}" for n in range(2,6)]]).all().all()
        assert len(p)==13*144
        for _,g in p.groupby("model"):
            assert pd.PeriodIndex(g.ym,freq="M").equals(pd.period_range("2011-01","2022-12",freq="M"))
        audit=pd.read_csv(folder/"forecast_audit.csv")
        d=pd.PeriodIndex(audit.ym,freq="M")
        assert (pd.PeriodIndex(audit.last_label_end,freq="M")<d).all()
        assert (pd.PeriodIndex(audit.macro_reference,freq="M")==d-2).all()
        assert (pd.to_datetime(audit.survey_release).to_numpy()<=d.to_timestamp(how="end").to_numpy()).all()
        assert audit.n_features.eq(15).all() and audit.ms17_converged.all()
        assert audit.l3_features.eq(30).all() and audit.l3_lag_months.eq(1).all()
        en=json.loads((folder/"elasticnet_cv_audit.json").read_text())
        en_parameters={(r["ym"],r["model"]):r["parameters"] for r in en}
        for record in en:
            for last_label_end,validation_start in record["inner_cutoffs"]:
                assert pd.Period(last_label_end,"M")<pd.Period(validation_start,"M")
        parameters=json.loads((folder/"frozen_parameters.json").read_text())
        assert parameters["validation_last_outcome"]=="2016-12"
        assert parameters["validation_last_purchase"]=="2015-12"
        for name in ("L4_Real","L4_GC"):
            trials=pd.read_csv(folder/(name+"_trials.csv"))
            assert len(trials)==32 and set(trials.seed)=={42,73}
            assert np.isclose(trials.value.min(),parameters[name]["validation_loss"])
            counts["xgb_trials"]+=len(trials)
        folds=pickle.loads((folder/"folds.pkl").read_bytes())
        indexed=p.set_index(["ym","model"])
        for date,fold in folds.items():
            np.testing.assert_array_equal(fold["y"],bonds.loc[fold["train_dates"],[f"rx{n}" for n in range(2,6)]].to_numpy())
            repeated=dict(fold["reference"])
            for model in ("L4_Real","L4_GC"):
                repeated[model]=xgb_predict(fold,parameters[model])
            l3=fold["lagged"]
            mean,std=l3["y"].mean(0),np.maximum(l3["y"].std(0),1e-8)
            for model in ("L3_Real","L3_GC"):
                X,Y=augment(l3["X"],(l3["y"]-mean)/std,1. if model.endswith("GC") else 0.,120)
                config=en_parameters[(str(date),model)]
                repeated[model]=np.column_stack([ElasticNet(max_iter=100000,tol=1e-5,**config[j]).fit(X,Y[:,j]).predict(l3["x"])
                    for j in range(4)])*std+mean
            for model,values in repeated.items():
                saved=indexed.loc[(date,model),[f"mu{n}" for n in range(2,6)]].to_numpy(dtype=float)
                np.testing.assert_allclose(saved,values[0],atol=1e-9,rtol=1e-9)
        execution=json.loads((folder/"execution_parameters.json").read_text())
        trials=pd.read_csv(folder/"execution_trials.csv")
        assert len(trials)==128
        best=trials.loc[trials.score.idxmax()]
        for key,value in execution["params"].items():
            assert np.isclose(best[key],value)
        assert execution["validation_last_outcome"]=="2016-12"
        t=pd.read_csv(folder/"trades.csv")
        forecasts={name:g.set_index("ym")[[f"mu{n}" for n in range(2,6)]] for name,g in p.groupby("model")}
        mean,_,_=risk_panel(bonds,dates)
        forecasts["RollingMV"]=pd.DataFrame(mean*100,index=dates,columns=[f"mu{n}" for n in range(2,6)])
        for name,g in t.groupby("model"):
            model,variant=name.split("/")
            config=execution["params"] if variant=="optimized" else {}
            reproduced=allocation_path(bonds,forecasts[model].loc[dates],**config)
            for key in ["return_net","cash","exposure",*[f"q{n}" for n in range(2,6)]]:
                np.testing.assert_allclose(g[key],reproduced[key],atol=1e-12)
            q=g[[f"q{n}" for n in range(2,6)]].to_numpy()
            assert (q>=-1e-14).all() and (q.sum(1)<=1+1e-12).all()
            assert (pd.PeriodIndex(g.risk_last_label_end,freq="M")<dates).all()
        counts["cases"]+=1;counts["forecast_rows"]+=len(p);counts["outer_folds"]+=len(audit)
        counts["trades"]+=len(t);counts["execution_trials"]+=len(trials)
        q=p[p.ym.isin(dates)];bench=q[q.model.eq("EH")].set_index("ym")
        vp=p[(p.ym>=pd.Period("2011-01","M")) & (p.ym+12<pd.Period("2017-01","M"))]
        ve=vp[vp.model.eq("EH")].set_index("ym")
        for name,g in vp.groupby("model"):
            vd=pd.PeriodIndex(g.ym,freq="M")
            vy=bonds.loc[vd,[f"rx{n}" for n in range(2,6)]].to_numpy()
            vpred=g[[f"mu{n}" for n in range(2,6)]].to_numpy()
            vb=ve.loc[g.ym,[f"mu{n}" for n in range(2,6)]].to_numpy()
            validation_rows.append(dict(case=case,model=name,n=len(g),mean_maturity_r2_pct=
                100*np.mean(1-((vy-vpred)**2).sum(0)/((vy-vb)**2).sum(0))))
        for name,g in q.groupby("model"):
            pred=g[[f"mu{n}" for n in range(2,6)]].to_numpy()
            eh=bench.loc[g.ym,[f"mu{n}" for n in range(2,6)]].to_numpy()
            for j,n in enumerate([2,3,4,5,"basket"]):
                y,yhat,b=((a[:,j] for a in (actual,pred,eh)) if j<4 else (a.mean(1) for a in (actual,pred,eh)))
                _,pv=loss_test(y,yhat,b,lags=18)
                score_rows.append(dict(case=case,model=name,maturity=str(n),n=72,r2_pct=100*oos_r2(y,yhat,b),raw_hac_p=pv,
                                       peer_r2_pct=100*PEER_R2[name][j] if j<4 and name in PEER_R2 else np.nan))
                if j==4 and name!="EH":losses.append((case,name,(y-b)**2-(y-yhat)**2))
        # Paired conditional CER differences, keeping same-month account paths.
        wide=t.pivot(index="purchase",columns="model",values="return_net")
        values=wide.to_numpy();baseline=wide.columns.get_loc("RollingMV/direct")
        cer=lambda a:12*(a.mean(axis=-2)-2.5*a.var(axis=-2,ddof=1))
        point=cer(values);rng=np.random.default_rng(42);starts=rng.integers(0,72,(1999,6))
        ix=((starts[:,:,None]+np.arange(12))%72).reshape(1999,72)
        draws=cer(values[ix])
        for j,name in enumerate(wide.columns):
            gain=10000*(draws[:,j]-draws[:,baseline])
            utility_rows.append(dict(case=case,model=name,delta_cer_bps=10000*(point[j]-point[baseline]),
                ci_low=np.quantile(gain,.025),ci_high=np.quantile(gain,.975),block=12,replications=1999))
    scores=pd.DataFrame(score_rows)
    use=scores.model.ne("EH")
    scores.loc[use,"holm_p"]=multipletests(scores.loc[use,"raw_hac_p"],method="holm")[1]
    joint=joint_block_bootstrap_t(np.column_stack([a for _,_,a in losses]),lags=18,block=12,replications=1999)
    for k,(case,name,_) in enumerate(losses):
        scores.loc[scores.case.eq(case)&scores.model.eq(name)&scores.maturity.eq("basket"),"max_t_p"]=joint["max_t_p"][k]
    scores.to_csv(output/"forecast_scores.csv",index=False)
    pd.DataFrame(validation_rows).to_csv(output/"validation_scores.csv",index=False)
    pd.DataFrame(utility_rows).to_csv(output/"utility.csv",index=False)
    (output/"audit.json").write_text(json.dumps(dict(status="passed",**counts),indent=2))
    return scores,counts


def figures(output,scores):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False})
    names=list(PEER_R2)
    fig,axes=plt.subplots(2,2,figsize=(13,10),constrained_layout=True)
    for j,case in enumerate(PROTOCOL["cases"]):
        ax=axes[0,j]
        a=np.array([[scores.loc[scores.case.eq(case)&scores.model.eq(name)&scores.maturity.eq(str(n)),"r2_pct"].iloc[0]
                     for n in range(2,6)] for name in names])
        ax.imshow(a,cmap="RdBu",vmin=-40,vmax=40,aspect="auto")
        ax.set_yticks(range(len(names)),names);ax.set_xticks(range(4),["2y","3y","4y","5y"])
        ax.set_title(case+" ME: annual OOS R2 (%)")
        for r in range(len(names)):
            for c in range(4):
                value=a[r,c];label=f"{value:.1f}" if abs(value)<1000 else f"{value:.1e}"
                ax.text(c,r,label,ha="center",va="center",fontsize=8,color="white" if abs(value)>20 else "#1c2833")
        ax=axes[1,j];t=pd.read_csv(output/case/"trades.csv")
        for name,label,color in [("L4_GC/optimized","XGB+GC, execution calibrated","#187575"),
                                 ("L4_GC/direct","XGB+GC, direct","#9070af"),
                                 ("L1_CP15/direct","CP15 cycle allocation","#bd8127"),
                                 ("RollingMV/direct","60-observation rolling MV","#6a7887")]:
            g=t[t.model.eq(name)].sort_values("purchase")
            dates=pd.PeriodIndex([g.purchase.iloc[0],*g.sale.tolist()],freq="M").to_timestamp()
            ax.plot(dates,np.r_[1.,np.cumprod(1+g.return_net)],label=label,color=color,linewidth=1.8)
        ax.set_title(case+": actual monthly net wealth (no cost, as undisclosed in slides)")
        ax.set_ylabel("Initial wealth = 1");ax.legend(fontsize=8,frameon=False);ax.grid(alpha=.15)
    fig.suptitle("Reference pages 1-22: L3 uses 15 current + 15 lagged inputs; L4 uses 15 current inputs\nActual MS factor / SPF; unknown source and execution details remain assumptions; colors clipped at +/-40%",fontsize=11)
    fig.savefig(ROOT/"assets"/"peer_reference.png",dpi=170);plt.close(fig)


def report(output,scores,counts):
    labels={"dispersion":"PPT 分歧定义", "expectations":"Eriksen 预期定义"}
    lines=['<a id="peer-reference"></a>',"### 5.8 按新增 PDF 前 22 页重建的对应基线","",
        "参考文件为本地《固定收益证券分析和模型-2026-05-25第3-5节.pdf》前 22 页。",
        "本节与 §5.7 的自行扩展分开：恢复完整 2011–2016 验证段、按期限配置、60 月滚动均值/方差基准，",
        "固定 XGBoost 深度 2、100 棵树；实际实现 MS 递归因子和 SPF 因子，并对齐 10＋5 个输入。",
        "**这是已披露方法的对应重建，不称为同代码、同数据的数值复现。** PDF 未披露的输入及公式见下表，",
        "不通过挑选测试期成绩填补它们，也不把同名模型视为完全相同模型。","",
        "| PDF 页 | 已披露内容 | 本次实现与仍缺的细节 |","| --- | --- | --- |",
        "| 3、9、13 | 六类文献因子，图中含四个利差、LN F5/F6、recursive factor、CP15、ME | Full10：四利差＋CP05＋MS＋CP15＋LN5＋LN6＋ME；完整十列由图与总数推定，变量名全部保存 |",
        "| 4–5 | 约 20 原序列生成 82 候选；绝对 t 值>2.5；筛选截至 2009-12；PC1–5 | 原 82 列清单和公式未给出；本次公开 20×4＋2 的确定模板，不能称为原清单 |",
        "| 6–7 | 1994–2010 / 2011–2016 / 2017–2022；12 月隔离；训练集预处理 | 恢复完整验证段，按日历剔除未兑现标签；成熟训练行拟合填补、缩放与 PCA |",
        "| 11–12 | L3各变量加滞后项；L4使用15个当期变量；动态ElasticNetCV、Copula；XGB深度2、100树 | 已补齐L3的15＋15列；当前及滞后值用同一外层训练基底，内折也重建；L4仍15列；其余正则/GC范围为验证期设定 |",
        "| 16–19 | 各期限正信号、多头、60 月滚动 MV、B/λdd/λg 执行校准 | 按年预测及年风险定仓、真实月价格计净值；具体惩罚方程、γ、仓位上限、费用与优化目标未披露，采用的公式与范围均标为假设 |","",
        "**修正此前对 MS17 的理解。** Feunou–Fontaine 原文 §2.2 的式(3)本身就是可独立估计的递归预测因子，",
        "无须先估计完整无套利 DTSM：$R_t=(1-\u03b1)\u03b3^\u2032[1,f_t]+\u03b1R_{t-1}$，再令 $rx_t^n=b_nR_t+u_t^n$。",
        "本次通过联合非线性最小二乘估计 α、γ、b，约束平均 b=1；R0 取成熟训练均值，保留隔离期内可观察的每个月远期利率。",
        "α=0.6284 是原文年预测样本的估计结果，不当作未来样本已知参数；此前固定 EMA 半衰期不是这项复现。","",
        "参考层的 CP05 与 MS 直接保留原文无额外截距的期限载荷，不再经统一含截距回归改写。",
        "这项修正只影响具名基准预测；复杂模型的15列输入保持不变。","",
        "**ME 的原文与 PPT 存在定义差异。** Eriksen(2017) §III 使用 GDP、GDP 平减指数、失业率、税后利润、",
        "工业生产与房屋开工的个体调查预期：每位受访者相对其自身 nowcast 计算未来 1–4 季变化，再取中位数，",
        "对 24 个序列提取前三 PC 并拟合共同收益指数。PPT 称之为预测分歧；本次并列运行上述原文定义，",
        "以及相同个体变化的横截面 IQR 分歧定义。季度数据按费城联储的**实际发布日期**映射月末，绝不提前铺到季度首月。",
        "当前档案可能含历史勘误，不能据此宣称逐字节重现当年数据版本。宏观/CPI 主面板仍是修订值，滞后两月；历史版本检验另见前文。","",
        "**真实投资保留年度信号。** 不再另拟合一月收益模型来替代 PDF 的年度模型。每期限使用",
        "$w_n\propto\max(\hat\mu_n,0)/(\gamma\hat\sigma_n^2)$，均值和方差口径均为年收益；每月按新信号重新定仓，",
        "下一月将同一债券按剩余期限 n−1/12 重定价，现金计入实际 RF。调仓频率与预测期可以不同，",
        "年度重叠标签不作为实际月收益复利。B∈[0.1,1]，λdd∈[0,30]，λg∈[0,20]，Optuna 128 候选只优化验证期月超额夏普。",
        "本实现用因账户历史回撤缩减的预算 B·exp(λdd·DD) 和凸 L2 仓位惩罚；它们是对未披露公式的明确实现假设。",
        "主表 γ=5、总多头上限 1；PPT 未交代费用，主表先列零费，再保存单边 0.5/2/5bps 敏感性。","",
        "**时间标签明确。** 以买入/形成月定义 2017-01–2022-12，共 72 个年预测和 72 笔月交易；",
        "年标签兑现至 2023-12，月交易卖出至 2023-01。冻结于 2017-01 前，完整验证段内可用的年标签为",
        "2011-01–2015-12 买入的 60 个，月投资校准为 2011-01–2016-11 的 71 笔。2016 未兑现的年标签不会提前用于调参。",
        "所有历史期间此前已接触，本节仍属于回溯研究。","",
        "| 模型 | PDF 平均逐期限 R² (%) | 本次：分歧 ME | 本次：预期 ME |","| --- | --- | --- | --- |"]
    for name,peer in PEER_R2.items():
        values=[scores.loc[scores.case.eq(case)&scores.model.eq(name)&scores.maturity.ne("basket"),"r2_pct"].mean() for case in PROTOCOL["cases"]]
        lines.append(f"| {name} | {100*np.mean(peer):.2f} | {values[0]:.2f} | {values[1]:.2f} |")
    screening=pd.read_csv(output/"dispersion"/"screening.csv")
    first=pd.read_csv(output/"dispersion"/"forecast_audit.csv").iloc[0]
    lines += ["",f"本模板池实际筛出 {int(screening.selected.sum())} 列，首次训练 PC1–5 解释 {100*first.agent_explained5:.2f}% 方差；",
        "PDF 报告 93.0%。这项可观察差异直接说明候选池尚未逐列匹配，不应把新增实验包装为原始输入的复现。",
        "SPF 原文采用季度频率，这里按实际发布日保持到月末，是月度适配；初筛变量集合固定于 2009 年，内层 EN 验证条件于该初始变量集合。","",
        "**ME 原文核对修正。** Eriksen 提供 PDF 第79页 Table1 的 GDP 一至四季预期均值为 1.53、3.13、4.76、6.44。",
        "SPF 的 NGDP 可对应这些数字；先前使用 RGDP 属于因子定义错误，已改为名义 GDP，并用24项原始统计核查。",
        "住宅开工水平变化按原表乘100；这是单位换算，不影响标准化 PCA。原文 ME 是预期中位数，不是分歧；",
        "PPT 分歧版本继续作为单列代理。原始统计及归因依据保存在 `survey_paper_table1.csv` 和 `survey_paper_evidence.json`，",
        "不能将当前调查档案中的剩余修订差异包装成原始版本的逐格精确复现。","",
        "| ME 定义 | 模型 | 完整验证段可兑现标签的平均 R² (%) | 测试平均 R² (%) |","| --- | --- | --- | --- |"]
    val=pd.read_csv(output/"validation_scores.csv")
    for case in PROTOCOL["cases"]:
        for name in ("L3_Real","L3_GC","L4_Real","L4_GC"):
            v=val.loc[val.case.eq(case)&val.model.eq(name),"mean_maturity_r2_pct"].iloc[0]
            t=scores.loc[scores.case.eq(case)&scores.model.eq(name)&scores.maturity.ne("basket"),"r2_pct"].mean()
            lines.append(f"| {labels[case]} | {name} | {v:.2f} | {t:.2f} |")
    lines += ["", "上表按四期限 R² 的均值对应 PDF 图表；篮子 R² 另存，二者不混用。原始 LN 组合在疫情宏观极值上的外推，",
        "使线性组合/EN 出现极大负 R²；保留该失败结果，没有为匹配 PPT 的正值自行加截断。",
        "这需要核对对方原始因子和转换代码，不能据此断言对方发生泄漏，也不能仅由我们未匹配推断其结果不可能。","",
        "| ME 定义 | 月度账户（主表零费） | CAGR (%) | 年均超额 (%) | 超额夏普 | 月末回撤 (%) |","| --- | --- | --- | --- | --- | --- |"]
    for case in PROTOCOL["cases"]:
        perf=pd.read_csv(output/case/"performance.csv")
        for name in ("RollingMV/direct","L1_CP15/direct","L4_Real/direct","L4_GC/direct","L4_GC/optimized"):
            r=perf[perf.model.eq(name)].iloc[0]
            lines.append(f"| {labels[case]} | {name} | {r.cagr_pct:.2f} | {r.mean_excess_pct:.2f} | {r.sharpe:.2f} | {r.max_dd_pct:.2f} |")
    lines += ["", "**收益口径核对。** PDF 第 20 页的 2.51%=15.06%÷6、2.08%≈12.50%÷6、0.58%=3.48%÷6，",
        "符合累计数除以六的算术口径，不能直接叫 CAGR。若 15.06% 是六年复利总收益，对应 CAGR 应约 2.37%。",
        "PPT 没有完整给出现金/RF、净值累计和年度目标到交易收益的映射公式，因此本表不宣称这些定义完全一致。",
        "11.58 是 15.06 与 3.48 的**百分点差**，不自动等于模型的因果收益贡献。结果文件另存重叠年度加权收益的",
        "均值与 Σ/12 诊断口径，和实际月度 NAV 分开；两者差异不能靠改变年化标签消除。","",
        "![按 PDF 前22页重建的预测和实际投资对照](assets/peer_reference.png)","",
        "**目前能够成立的结论。** 与 PDF 相同名称的方法链已进一步对齐，但其 R² 与高夏普数字仍未数值重现。",
        "CP15 的年预测在本数据上有正的点估计，复杂模型及额外执行校准并不保证更高投资收益；",
        "本节保留与此前 EW3 不同的策略研究，不因这次新名称或某个 ME 版本的测试排名替换既有候选。",
        "82 个候选的精确清单、SPF 分歧定义、原始因子数据和执行公式是仍需原代码才能消除的差异。",
        "联合 max-t/全表 Holm 与配对月 CER 区间保存在结果目录，均条件于已保存预测，并未校正此前全部研究搜索。","",
        "来源：[SPF 原始个体数据与说明](https://www.philadelphiafed.org/surveys-and-data/real-time-data-research/individual-forecasts)、",
        "[SPF 发布日历](https://www.philadelphiafed.org/-/media/FRBP/Assets/Surveys-And-Data/survey-of-professional-forecasters/spf-release-dates.txt)、",
        "[Eriksen 原文](https://doi.org/10.1017/S0022109017000369)、[Feunou–Fontaine 原文](https://doi.org/10.1287/mnsc.2016.2602)。",
        "GSW、FRED-MD、核心 CPI、EPU、RF 沿用本仓库数据来源；原始 PDF、调查 Excel 和所有中间结果均不上传。","",
        "```bash","python scripts/fetch_peer_inputs.py",
        "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/rebuild_peer_reference.py",
        "python scripts/export_peer_reference.py --update-readme","```","",
        f"本轮实际审计通过：{counts['forecast_rows']:,} 条预测、{counts['outer_folds']} 个外层拟合、{counts['trades']:,} 笔账户记录，",
        f"{counts['xgb_trials']} 个预测候选和 {counts['execution_trials']} 个执行候选。参数、内折截止日、筛选 t 值、源文件 SHA 和未披露设定映射保存在 `output/peer_reference/`。"]
    body="\n".join(lines)+"\n";(output/"report.md").write_text(body)
    return body


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,default=ROOT/"output"/"peer_reference")
    parser.add_argument("--update-readme",action="store_true")
    args=parser.parse_args();bonds=load_long_market()
    scores,counts=audit_and_scores(args.output_dir,bonds)
    figures(args.output_dir,scores);body=report(args.output_dir,scores,counts)
    if args.update_readme:
        path=ROOT/"README.md";text=path.read_text();start,end="<!-- PEER_REFERENCE:START -->","<!-- PEER_REFERENCE:END -->"
        block=start+"\n"+body+end+"\n\n"
        if start in text:
            a,b=text.index(start),text.index(end)+len(end);text=text[:a]+block.rstrip()+text[b:]
        else:
            k=text.index("## 6.");text=text[:k]+block+text[k:]
        path.write_text(text)
    print(json.dumps(counts),flush=True)


if __name__=="__main__":
    main()
