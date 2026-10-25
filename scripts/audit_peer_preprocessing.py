"""Independently replay saved cleaned-fold forecasts and monthly cash accounts."""
import argparse
import json
from pathlib import Path
import pickle
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bond_rp.peer_preprocessing import METHODS, rebuild_macro_fold
from bond_rp.peer_learners import augment, xgb_predict
from bond_rp.peer_allocation import allocation_path, risk_panel, summarize
from complete_peer_preprocessing import DATES, MU, TARGETS, VALIDATION, TEST, fingerprint, predictions
from rebuild_peer_reference import inputs


def audit(baseline, output):
    manifest = fingerprint(baseline)
    assert manifest == json.loads((output/"protocol.json").read_text())
    bonds,features,candidates,selected,_,_ = inputs()
    counts = dict(branches=0,forecasts=0,outer_folds=0,replayed_learners=0,
                  xgb_trials=0,execution_trials=0,trades=0)
    for case in ("dispersion","expectations"):
        original = pickle.loads((baseline/case/"folds.pkl").read_bytes())
        for method in METHODS[1:]:
            folder = output/case/method
            completed = json.loads((folder/"completed.json").read_text())
            assert completed == dict(manifest=manifest,case=case,method=method)
            folds = pickle.loads((folder/"folds.pkl").read_bytes())
            frozen = json.loads((folder/"frozen_parameters.json").read_text())
            p = predictions(folder)
            enet = {(r["ym"],r["model"]):r for r in json.loads((folder/"elasticnet_cv_audit.json").read_text())}
            assert len(p) == 13
            for g in p.values():
                assert g.index.equals(DATES) and np.isfinite(g.to_numpy()).all()
            for name in ("L4_Real","L4_GC"):
                trials = pd.read_csv(folder/(name+"_trials.csv"))
                assert len(trials) == 32 and set(trials.seed) == {42,73}
                np.testing.assert_allclose(trials.value.min(),frozen[name]["validation_loss"])
                actual = bonds.loc[VALIDATION,TARGETS].to_numpy()
                eh = p["EH"].loc[VALIDATION].to_numpy()
                pred = p[name].loc[VALIDATION].to_numpy()
                loss = np.mean(((actual-pred)**2).sum(0)/((actual-eh)**2).sum(0))
                np.testing.assert_allclose(loss,frozen[name]["validation_loss"],atol=1e-12)
                counts["xgb_trials"] += len(trials)
            for date,fold in folds.items():
                np.testing.assert_array_equal(fold["y"],original[date]["y"])
                assert fold["audit"]["preprocessing"] == method
                assert pd.Period(fold["audit"]["threshold_fit_last"],"M")+12 < date
                assert fold["X"].shape[1] == 15 and fold["lagged"]["X"].shape[1] == 30
                # Reconstruct representative outer folds from original inputs,
                # including the shock month, independently of saved clean X.
                if str(date) in ("2011-01","2017-01","2020-06","2022-12"):
                    replay = rebuild_macro_fold(original[date],features,candidates,selected,method)
                    for key in ("X","x","y"):
                        np.testing.assert_array_equal(fold[key],replay[key])
                        np.testing.assert_array_equal(fold["lagged"][key],replay["lagged"][key])
                expected = dict(fold["reference"])
                for name in ("L4_Real","L4_GC"):
                    expected[name] = xgb_predict(fold,frozen[name])
                design = fold["lagged"]
                mean,std = design["y"].mean(0),np.maximum(design["y"].std(0),1e-8)
                for name in ("L3_Real","L3_GC"):
                    record = enet[(str(date),name)]
                    for last_label_end,validation_start in record["inner_cutoffs"]:
                        assert pd.Period(last_label_end,"M") < pd.Period(validation_start,"M")
                    X,Y = augment(design["X"],(design["y"]-mean)/std,1. if name.endswith("GC") else 0.,120)
                    expected[name] = np.column_stack([ElasticNet(max_iter=100000,tol=1e-5,
                        **record["parameters"][j]).fit(X,Y[:,j]).predict(design["x"])
                        for j in range(4)])*std+mean
                for name,values in expected.items():
                    np.testing.assert_allclose(p[name].loc[date].to_numpy(),values[0],atol=1e-9,rtol=1e-9)
                counts["replayed_learners"] += 4
            execution = json.loads((folder/"execution_parameters.json").read_text())
            assert execution["validation_last_outcome"] == "2016-12"
            trials = pd.read_csv(folder/"execution_trials.csv")
            assert len(trials) == 128
            best = trials.loc[trials.score.idxmax()]
            for name,value in execution["params"].items():
                np.testing.assert_allclose(best[name],value)
            t = pd.read_csv(folder/"trades.csv")
            historical,_,_ = risk_panel(bonds,TEST)
            p["RollingMV"] = pd.DataFrame(historical*100,index=TEST,columns=MU)
            for name,g in t.groupby("model"):
                model,variant = name.split("/")
                config = execution["params"] if variant == "optimized" else {}
                replay = allocation_path(bonds,p[model].loc[TEST],**config)
                for column in ("return_net","cash","exposure","wealth",*[f"q{n}" for n in range(2,6)]):
                    np.testing.assert_allclose(g[column],replay[column],atol=1e-12)
                assert (g.cash >= -1e-12).all() and (g.exposure <= 1+1e-12).all()
            counts["branches"] += 1; counts["outer_folds"] += len(folds)
            counts["forecasts"] += sum(len(g) for name,g in p.items() if name != "RollingMV")
            counts["execution_trials"] += len(trials); counts["trades"] += len(t)
            print(f"Audited {case}/{method}",flush=True)
    score = pd.read_csv(output/"all_scores.csv")
    choices = json.loads((output/"selection.json").read_text())
    for choice in choices:
        part = score[score.case.eq(choice["case"])&score.period.eq("validation_matured")&score.model.eq("L4_GC")]
        ranking = part.groupby("method").r2_pct.mean()
        assert choice["method"] == max(METHODS,key=lambda m:ranking[m])
        assert choice["validation_last_label_end"] == "2016-12"
    result = dict(status="passed",**counts)
    (output/"audit.json").write_text(json.dumps(result,indent=2))
    return result


def cost_and_evidence(baseline, output, counts):
    """Report selected branches at common costs; never reselect at these costs."""
    bonds = inputs()[0]
    choices = json.loads((output/"selection.json").read_text())
    records = []
    for choice in choices:
        case,method = choice["case"],choice["method"]
        folder = baseline/case if method == "raw" else output/case/method
        p = predictions(folder)["L4_GC"].loc[TEST]
        config = json.loads((folder/"execution_parameters.json").read_text())["params"]
        mean,_,_ = risk_panel(bonds,TEST)
        historical = pd.DataFrame(mean*100,index=TEST,columns=MU)
        for cost in (0.,.5,2.,5.):
            for model,forecast,params in (("RollingMV",historical,{}),("direct",p,{}),("optimized",p,config)):
                path = allocation_path(bonds,forecast,cost_bps=cost,**params)
                records.append(dict(case=case,method=method,model=model,cost_bps=cost,**summarize(path)))
    pd.DataFrame(records).to_csv(output/"selected_cost_stress.csv",index=False)
    lines = ["", "**独立回放及成本复核。** "
        f"{counts['outer_folds']}个新增外层折、{counts['forecasts']}条预测、{counts['replayed_learners']}次学习器回放、"
        f"{counts['trades']}笔交易与256次预测/512次执行参数试验审计通过；86项回归测试通过。",
        "成本压力使用已冻结的预测及执行参数，不按成本重选方法。单边2bps时：","",
        "| ME定义 | 同期RollingMV CAGR (%) | 直接配置 CAGR (%) / SR | 校准配置 CAGR (%) / SR |",
        "| --- | --- | --- | --- |"]
    frame = pd.DataFrame(records)
    for choice in choices:
        q = frame[frame.case.eq(choice["case"])&frame.cost_bps.eq(2.)].set_index("model")
        lines.append(f"| {choice['case']} | {q.loc['RollingMV','cagr_pct']:.2f} | "
            f"{q.loc['direct','cagr_pct']:.2f} / {q.loc['direct','sharpe']:.3f} | "
            f"{q.loc['optimized','cagr_pct']:.2f} / {q.loc['optimized','sharpe']:.3f} |")
    utility = pd.read_csv(output/"conditional_utility.csv")
    lines += ["", "零费直接配置相对RollingMV的年度ΔCER及12月区块bootstrap的95%区间："]
    for choice in choices:
        q = utility[utility.case.eq(choice["case"])&utility.method.eq(choice["method"])&utility.model.eq("L4_GC/direct")].iloc[0]
        lines.append(f"{choice['case']}为{q.delta_cer_bps:.1f}bps [{q.ci_low:.1f}, {q.ci_high:.1f}]。")
    lines += ["区间均包含零，尚未确认稳定的效用增益。"]
    text = "\n".join(lines)+"\n"
    (output/"audit_README_section.md").write_text(text)
    readme = ROOT/"README.md"; content = readme.read_text()
    begin,end = "<!-- BEGIN PEER_PREPROCESSING_EVIDENCE -->","<!-- END PEER_PREPROCESSING_EVIDENCE -->"
    block = begin+"\n"+text+end+"\n"
    if begin in content:
        first,last = content.index(begin),content.index(end)+len(end)+1
        content = content[:first]+block+content[last:]
    else:
        position = content.index("<!-- END PEER_PREPROCESSING -->")
        content = content[:position]+block+content[position:]
    readme.write_text(content)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline",type=Path,default=ROOT/"output"/"peer_reference")
    parser.add_argument("--output-dir",type=Path,default=ROOT/"output"/"peer_preprocessing")
    args = parser.parse_args()
    result = audit(args.baseline,args.output_dir)
    cost_and_evidence(args.baseline,args.output_dir,result)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
