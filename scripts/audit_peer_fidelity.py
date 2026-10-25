"""Check method identity against peer slides, independently of execution tests.

This report intentionally returns NOT_REPLICATED while disclosed differences
and unknown inputs remain. It does not tune models or relabel counterfactuals
as successful replications. Source pages were visually checked against PDF.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import warnings
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
import statsmodels.api as sm
from bond_rp.config import DATA_DIR
from bond_rp.data import load_macro_raw_transformed
from bond_rp.peer_allocation import risk_panel
from bond_rp.peer_reference import candidate_menu, screen_candidates
from bond_rp.validated_models import load_long_market


def screening_interpretations(bonds, candidates, output):
    """Compare formation cutoff with outcomes known by 2009-12, not future data."""
    names = [f"rx{n}" for n in range(2,6)]
    truncated = bonds.copy()
    truncated.loc[truncated.index+12 > pd.Period("2009-12","M"),names] = np.nan
    records,sets = [],{}
    for label,panel in (("formation_through_2009",bonds),("information_through_2009",truncated)):
        chosen,table = screen_candidates(panel,candidates)
        index = candidates.loc["1994-01":"2009-12"].index
        target = panel.loc[index,names].mean(axis=1)
        valid_dates = index[target.notna()]
        assert str(valid_dates[-1]+12) == ("2010-12" if label.startswith("formation") else "2009-12")
        # Independent statsmodels regression verifies the cutoff comparison,
        # including all rejected candidates, without sharing the manual SE.
        for row in table.to_dict("records"):
            x = candidates.loc[index,row["feature"]]
            use = x.notna()&target.notna()
            independent = sm.OLS(target[use].to_numpy(),sm.add_constant(x[use].to_numpy())).fit()
            np.testing.assert_allclose(independent.tvalues[1],row["t_ols"],atol=1e-8,rtol=1e-8)
            records.append(dict(interpretation=label,feature=row["feature"],n=row["n"],
                t_ols=row["t_ols"],t_hac=row["t_hac"],selected=row["selected"],
                last_formation=str(valid_dates[-1]),last_outcome=str(valid_dates[-1]+12)))
        sets[label] = chosen
    pd.DataFrame(records).to_csv(output/"screening_interpretations.csv",index=False)
    old,new = map(set,(sets["formation_through_2009"],sets["information_through_2009"]))
    return dict(current_n=len(old),information_n=len(new),removed=sorted(old-new),added=sorted(new-old),
                current_label_end="2010-12",information_label_end="2009-12")


def composition(candidates, selected, output):
    peer = dict(sum12=68.,rank12=12.,difference3=4.,momentum6=4.)
    rows = []
    for label,columns in (("all_candidates",list(candidates)),("selected",selected)):
        kind = Counter(c.split("__")[-1] if "__" in c else "level" for c in columns)
        for k in ("sum12","rank12","difference3","momentum6","level"):
            rows.append(dict(population=label,population_n=len(columns),template=k,count=kind[k],
                current_pct=100*kind[k]/len(columns),peer_template_pct=peer.get(k,np.nan)))
    pd.DataFrame(rows).to_csv(output/"candidate_composition.csv",index=False)
    return rows


def allocation_difference(bonds, baseline, output):
    dates = pd.period_range("2017-01","2022-12",freq="M")
    mean,var,_ = risk_panel(bonds,dates)
    rows = []
    for case in ("dispersion","expectations"):
        p = pd.read_csv(baseline/case/"predictions.csv")
        p["ym"] = pd.PeriodIndex(p.ym,freq="M")
        mu = p[p.model.eq("L4_GC")].set_index("ym").loc[dates,[f"mu{n}" for n in range(2,6)]].to_numpy()/100
        for model,forecast in (("RollingMV",mean),("L4_GC",mu)):
            unbounded = np.maximum(forecast,0)/(5*np.maximum(var,1e-10))
            exposure = unbounded.sum(1)
            rows.append(dict(case=case,model=model,months=72,cap_binding_months=int((exposure>1).sum()),
                raw_median_exposure=float(np.median(exposure)),raw_max_exposure=float(exposure.max()),
                explanation="Raw positive MV formula before our added joint KKT budget=1 constraint"))
    pd.DataFrame(rows).to_csv(output/"mv_constraint_effect.csv",index=False)
    return rows


def code_link(path, needle):
    lines = (ROOT/path).read_text().splitlines()
    line = next(i+1 for i,s in enumerate(lines) if needle in s)
    return f"[{path}:{line}]({path}#L{line})"


def report(output, baseline, screening, templates, allocations, n_macro):
    items = [
        ("2", "预测对象", "2–5年零息国债、12个月超额收益", "同一目标类别；价格版本及日期锚点另核", "已核对结构", "bond_rp/validated_models.py", "def load_long_market"),
        ("12", "XGB树结构", "深度2、100树", "max_depth=2、n_estimators=100；其他正则为自定搜索", "已核对结构", "bond_rp/peer_learners.py", "n_estimators=100"),
        ("11–12", "L3/L4维数", "L3各变量加滞后；L4为15个同期输入", "L3为15+15、L4为15；滞后1月及列成员并非原清单确认", "已核对结构", "bond_rp/peer_reference.py", "if include_lags:"),
        ("4–5", "候选模板分布", "累计求和68%、排名12%、差分4%、动量4%；图示分母未注明", "全池四模板各24.39%；筛后为64.71%/23.53%/0%/11.76%，均未匹配图示", "没有复刻图示构成", "bond_rp/peer_reference.py", "for name in raw:"),
        ("4–5", "窗口分布", "12月44%、6月20%、3月20%；剩余未说明", "全池12月40/82、6月20/82、3月20/82、水平2/82；原图的统计对象待确认", "没有复刻图示构成", "bond_rp/peer_reference.py", "__sum12"),
        ("4–5", "候选逐列内容", "约20条序列生成82项；原列名/公式未给出", "自选17个FRED序列+EPU+两曲线源，不能由82这个数证明相同", "未能确认", "bond_rp/peer_reference.py", "MACRO_MENU ="),
        ("4–5", "政策不确定性来源", "来源图标示20%，分母未注明", "直接EPU衍生4/82=4.88%；原图具体序列与统计对象未确认", "没有复刻图示构成", "bond_rp/peer_reference.py", 'raw["EPU"]'),
        ("4–5", "模板PC方差", "前5PC解释93.0%", "原重建首次89.29%；须同时匹配池、样本及预处理，不强行调到93%", "未能确认同口径", "bond_rp/peer_reference.py", "agent_pca ="),
        ("6–7", "筛选截止", "Feature screening truncated at 2009-12", "当前按形成月截止，标签到2010-12；若是信息截止，入选17→19", "日期含义未确认", "bond_rp/peer_reference.py", 'last_purchase="2009-12"'),
        ("3", "LN输入面板", "127个月度序列", f"当前FRED-MD为{n_macro}列，下载版本和异常处理尚未对齐", "明确不一致", "scripts/rebuild_peer_reference.py", 'load_macro_raw_transformed("fredmd")'),
        ("3", "ME因子", "SPF分歧，引用Eriksen", "IQR分歧为自定代理；原文中位数预期另列，未获对方聚合公式", "未能确认", "bond_rp/survey_expectations.py", "dispersion.append"),
        ("6–7", "信息滞后与预处理", "宏观滞后；PCA/缩放只在训练集拟合", "统一滞后2月；逐月以成熟标签行重估，固定初始PCA等解释未核实", "未能确认", "bond_rp/peer_reference.py", "pca = _numeric_pipeline"),
        ("11–12", "GC与正则数值", "rolling Gaussian Copula、强L1/L2", "联合X/Y经验分布、窗口60/120/全历史、比例0.5/1/2及搜索边界均自定", "未能确认", "bond_rp/peer_learners.py", "def augment"),
        ("6–7", "验证可用样本", "验证2011–2016", "调参使用60个已兑现年度标签；另列完整72个描述结果，尚未同口径确认", "日期含义未确认", "scripts/rebuild_peer_reference.py", 'validation = dates[dates+12'),
        ("2,6,20", "测试日期", "2017–2022，72个月，数据区间1994–2022", "当前为形成月2017–2022，年度收益兑现至2023-12；不等于确认对方日期索引", "日期含义未确认", "scripts/rebuild_peer_reference.py", 'date_convention='),
        ("16–17", "原始MV权重", "逐期限正信号mu/(gamma*sigma²)", "先用该式，再加总上限1的KKT约束；约束生效时权重已改变", "附加了自定约束", "bond_rp/peer_allocation.py", "if w.sum() > budget:"),
        ("18–19", "风险预算B_t", "弱预期时自动压低总仓位", "预算B*exp(lambda_dd*历史回撤)没有直接以预测强度确定B_t的步骤；仓位仍随mu变化", "未逐项落实描述", "bond_rp/peer_allocation.py", "effective = budget*np.exp"),
        ("18–19", "回撤惩罚", "识别并惩罚验证期深回撤的配置倾向", "lambda_dd用作当前账户回撤刹车，验证目标仅为Sharpe；不是已确认的原目标", "未能确认", "scripts/rebuild_peer_reference.py", "score = np.sqrt(12)*excess.mean()"),
        ("16–20", "参数与执行", "只做多、60月风险；其他细节未给", "gamma5、上限1、年度风险信号配月度真实价格账户，均为本项目约定", "未能确认", "bond_rp/peer_allocation.py", "def allocation_path"),
        ("20", "收益指标", "累计超额15.06%、年化2.51%、SR1.307", "当前主表为含现金CAGR，月度超额SR；累计超额除6并非CAGR", "明确不是同一指标", "bond_rp/peer_allocation.py", "out = monthly_performance"),
    ]
    records = [dict(pages=a,item=b,source=c,current=d,status=e,code=code_link(f,g)) for a,b,c,d,e,f,g in items]
    pd.DataFrame(records).to_csv(output/"fidelity_matrix.csv",index=False)
    payload = dict(status="NOT_REPLICATED",same_method=False,numerical_replication=False,
        note="Execution checks and additional tuning do not establish peer method identity",
        screening=screening,checks=records)
    (output/"fidelity.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    lines = ['<a id="peer-fidelity"></a>',"### 5.12 是否与对方做了一样的复刻：重新核对原页","",
        "**没有。当前是带自定候选池、信息边界、GC配置和投资执行假设的方法重建及扩展，不能称为同方法复刻。**",
        "§5.11的清洗和重选参改善了部分结果，但没有消除这些差异；86项测试通过也不等于方法一致。",
        "本节重新视觉核对本地PDF第2–7、9、11–14、16–20页，并将材料未披露、日期解释未定和明确不一致分别列出。","",
        "| 原页 | 项目 | 对方披露 | 当前代码 | 核对结论 |",
        "| --- | --- | --- | --- | --- |"]
    for r in records:
        lines.append(f"| {r['pages']} | {r['item']} | {r['source']} | {r['current']}；{r['code']} | {r['status']} |")
    lines += ["","**候选池不是只差未知列名，已披露的模板构成也不同。** 对方图中的模板与窗口比例不是完整分布，",
        "不能猜补剩余部分，更不能通过平均分配模板凑出82列就称为复刻。当前数量与百分比如下：","",
        "原图没有注明右侧比例统计的是82个初始候选还是筛后集合；因此本次同时统计全池和筛后，避免将不同分母直接比较。",
        "图示百分比都是4个百分点的倍数，与25个特征等分母相容；这是可核查线索，不能据此认定筛后就是25项，更不能反向凑25项。","",
        "| 当前集合 | 模板 | 数量 | 当前占比 (%) | 对方图示该模板占比 (%) |","| --- | --- | --- | --- | --- |"]
    for r in templates:
        peer = "未说明" if pd.isna(r["peer_template_pct"]) else f"{r['peer_template_pct']:.0f}"
        lines.append(f"| {r['population']}（{r['population_n']}项） | {r['template']} | {r['count']} | {r['current_pct']:.2f} | {peer} |")
    lines += ["","**2009截止有实际影响。** 原代码是形成月截至2009-12，标签兑现至2010-12；",
        "若将其解释为截至2009年底已经知道的信息，应最多纳入2008-12形成、2009-12兑现的年度收益。",
        f"同一模板池、同一OLS阈值下，入选数从{screening['current_n']}变成{screening['information_n']}；",
        "移除：`"+"`、`".join(screening["removed"])+"`；",
        "增加：`"+"`、`".join(screening["added"])+"`。",
        "两种计算的全部82个t值均用独立statsmodels回归核对；这是一项截止日解释控制，不根据测试成绩选择其中一组。",
        "PPT未说明日期索引，尚不能据此判定对方究竟采用哪种解释，也不能说原代码已确认匹配。","",
        "**权重也不能只核对公式名称。** 本项目额外的总预算1约束会改变原始逐期限MV解。在原对应重建中：","",
        "| ME | 模型 | 72个月中原始MV总权重大于1的月份 | 原始MV总权重中位数 |","| --- | --- | --- | --- |"]
    for r in allocations:
        lines.append(f"| {r['case']} | {r['model']} | {r['cap_binding_months']} | {r['raw_median_exposure']:.2f} |")
    lines += ["","这些未受约束权重只用于说明附加约束影响，不作为可执行、无融资成本的策略发布。",
        "现有1.43%/1.56%等成绩仍是本项目设定下的结果，不能拿它们否定另一组，也不能作为其策略已复现的证据。",
        "接下来要建立方法一致性，需要逐项确认原始rx与日期、82项及筛后列、ME聚合、GC/正则配置和仓位/累计公式；",
        "文献可以约束因子的理论构造，不能唯一反推这些具体研究选择。未知项保持未知，不再以同名或更多调参替代核对。","",
        "复核：`python scripts/audit_peer_fidelity.py --update-readme`。脚本输出状态为`NOT_REPLICATED`，",
        "不覆盖原研究，不把上述截止日控制混入现有主成绩。详情留在本地`output/peer_fidelity/`。"]
    section = "\n".join(lines)+"\n"
    (output/"README_section.md").write_text(section)
    return section,payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,default=ROOT/"output"/"peer_reference")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"output"/"peer_fidelity")
    parser.add_argument("--update-readme",action="store_true")
    args = parser.parse_args()
    args.baseline,args.output_dir = args.baseline.resolve(),args.output_dir.resolve()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    bonds,macro = load_long_market(),load_macro_raw_transformed("fredmd")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore",message="Cannot parse header or footer")
        candidates = candidate_menu(bonds,macro)
    screening = screening_interpretations(bonds,candidates,args.output_dir)
    selected,_ = screen_candidates(bonds,candidates)
    templates = composition(candidates,selected,args.output_dir)
    allocations = allocation_difference(bonds,args.baseline,args.output_dir)
    section,payload = report(args.output_dir,args.baseline,screening,templates,allocations,len(macro.columns))
    files = [Path(__file__),ROOT/"美国国债超额收益预测研究_完整提取.md",
             ROOT/"提取自固定收益证券分析和模型-2026-05-25第3-5节.pdf"]
    files += [ROOT/"bond_rp"/(n+".py") for n in ("peer_reference","peer_allocation","peer_learners","survey_expectations")]
    files += [Path(DATA_DIR)/name for name in ("GSW_yields.csv","2026-04-MD.csv","US_Policy_Uncertainty_Data.xlsx")]
    files += [args.baseline/case/name for case in ("dispersion","expectations")
              for name in ("predictions.csv","forecast_audit.csv")]
    (args.output_dir/"sources.json").write_text(json.dumps({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in files if p.exists()},ensure_ascii=False,indent=2))
    if args.update_readme:
        path = ROOT/"README.md"; text = path.read_text()
        begin,end = "<!-- BEGIN PEER_FIDELITY -->","<!-- END PEER_FIDELITY -->"
        block = begin+"\n"+section+end+"\n\n"
        if begin in text:
            first,last = text.index(begin),text.index(end)+len(end)+2
            text = text[:first]+block+text[last:]
        else:
            first = text.index("## 6.")
            text = text[:first]+block+text[first:]
        path.write_text(text)
    print(json.dumps(dict(status=payload["status"],items=len(payload["checks"]),screening=screening),ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
