"""Verify original single factors, then run a frozen-validation model experiment."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT/".cache"/"matplotlib"))
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from bond_rp.config import DATA_DIR
from bond_rp.data import load_macro_raw_transformed
from bond_rp.inference import joint_block_bootstrap_t
from bond_rp.metrics import loss_test, oos_r2
from bond_rp.replication import run_replication
from bond_rp.strategy import annual_backtest
from bond_rp.trading import monthly_backtest
from bond_rp.validated_models import (LEARNERS, PRIMARY, REFERENCE, calibration, fred_snapshot,
                                     learner_predict, load_long_market, make_features, prepare_fold)
from bond_rp.vintages import VintageArchive

PROTOCOL = dict(train_start="1994-01", validation_start="2011-01", freeze="2017-01",
                test_sale_end="2022-12", followup_sale_end="2025-12", primary=PRIMARY,
                seed=42, en_trials=12, xgb_trials=16, gc_ratio=1, gamma=5, cap=1,
                cost_bps=.5, macro_reference_lag=2, vintage_lag=1,
                evaluation="retrospective; all periods already inspected in previous research",
                validation_objective="mean maturity-specific SSE / matched EH SSE",
                calibration="validation MSE-optimal common shrinkage in [0,1]")


def source_hashes():
    files = [*sorted((ROOT/"bond_rp").glob("*.py")), Path(__file__),
             *[Path(DATA_DIR)/n for n in ["GSW_yields.csv", "2026-04-MD.csv", "CPILFESL.csv",
               "F-F_Research_Data_Factors_CSV.zip", "fred_md_vintages_1999_2014.zip", "fred_md_vintages_2015_2025.zip"]]]
    return {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def versions():
    import sklearn, xgboost, optuna
    return dict(numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
                xgboost=xgboost.__version__, optuna=optuna.__version__)


def tune(folds, actual, name, folder):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    denominator = np.sum((actual-np.vstack([f["reference"]["EH"] for f in folds]))**2, axis=0)
    def predict(params):
        return np.vstack([learner_predict(f, name, params) for f in folds])
    def objective(trial):
        if name.startswith("EN"):
            params = dict(alpha=trial.suggest_float("alpha", .001, 10, log=True),
                          l1_ratio=trial.suggest_float("l1_ratio", .05, .95))
        else:
            params = dict(n_estimators=trial.suggest_int("n_estimators", 50, 200, step=50),
                          max_depth=trial.suggest_int("max_depth", 1, 3),
                          learning_rate=trial.suggest_float("learning_rate", .02, .15, log=True),
                          min_child_weight=trial.suggest_float("min_child_weight", 5, 30),
                          reg_lambda=trial.suggest_float("reg_lambda", 1, 100, log=True),
                          subsample=1., colsample_bytree=1.)
        return float(np.mean(np.sum((actual-predict(params))**2, axis=0)/denominator))
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    if name.startswith("EN"):
        study.enqueue_trial(dict(alpha=.1, l1_ratio=.5))
    else:
        study.enqueue_trial(dict(n_estimators=100, max_depth=2, learning_rate=.05, min_child_weight=10., reg_lambda=10.))
    study.optimize(objective, n_trials=PROTOCOL["en_trials" if name.startswith("EN") else "xgb_trials"])
    params = study.best_params
    if name.startswith("XGB"):
        params.update(subsample=1., colsample_bytree=1.)
    study.trials_dataframe().to_csv(folder/f"{name}_validation_trials.csv", index=False)
    return params, predict(params)


def panel(bonds, revised, core, tag, horizon, output, *, archive=None, inherited=None):
    folder = output/tag/f"h{horizon}"
    folder.mkdir(parents=True, exist_ok=True)
    fingerprint = hashlib.sha256(json.dumps(dict(sources=source_hashes(), versions=versions(), protocol=PROTOCOL,
            inherited=inherited, tag=tag, horizon=horizon), sort_keys=True).encode()).hexdigest()
    marker = folder/"cache_key.txt"
    needed = ["predictions.csv", "forecast_audit.csv", "validation_predictions.csv", "frozen_parameters.json"]
    if marker.exists() and marker.read_text() == fingerprint and all((folder/n).exists() for n in needed):
        print(f"Verified extension cache {tag}/h{horizon}", flush=True)
        return json.loads((folder/"frozen_parameters.json").read_text())
    macro = revised
    if tag == "core_revised":
        cpi = core
    else:
        raw = pd.read_csv(Path(DATA_DIR)/"2026-04-MD.csv").iloc[1:]
        cpi = pd.Series(pd.to_numeric(raw.CPIAUCSL, errors="coerce").to_numpy(),
                        index=pd.PeriodIndex(pd.to_datetime(raw.sasdate, format="mixed"), freq="M"))
    static = make_features(bonds, macro, cpi)
    targets = [f"{'rx' if horizon == 12 else 'mx'}{n}" for n in range(2, 6)]
    freeze = pd.Period(PROTOCOL["freeze"], "M")
    last = min(bonds.dropna(subset=targets).index.max(), pd.Period(PROTOCOL["followup_sale_end"], "M")-horizon)
    dates = pd.period_range(PROTOCOL["validation_start"], last, freq="M")
    folds, audit = {}, []
    for date in dates:
        features, vintage = static, "revised"
        if archive:
            vintage, macro, levels = fred_snapshot(archive, date)
            features = make_features(bonds, macro, levels)
        f = prepare_fold(bonds, features, date, horizon)
        f["audit"]["vintage"] = str(vintage)
        folds[date] = f
        audit.append(f["audit"])
    # Validation labels must have been realized strictly BEFORE parameter freeze.
    validation_dates = [d for d in folds if d+horizon < freeze]
    validation = [folds[d] for d in validation_dates]
    actual = bonds.loc[validation_dates, targets].to_numpy()
    eh = np.vstack([f["reference"]["EH"] for f in validation])
    selected, validation_predictions = {}, {}
    for name in LEARNERS:
        print(f"  {tag}/h{horizon}: validation {name}, {len(validation)} months", flush=True)
        if inherited:
            params = inherited[name]["params"]
            prediction = np.vstack([learner_predict(f, name, params) for f in validation])
            shrink = inherited[name]["lambda"]
        else:
            params, prediction = tune(validation, actual, name, folder)
            shrink = calibration(actual, prediction, eh)
        selected[name] = dict(params=params, **{"lambda": shrink})
        validation_predictions[name] = prediction
    # Freeze the validation choice of family, rather than selecting its test winner.
    chosen = inherited.get("selected_family") if inherited else min(LEARNERS, key=lambda name:
        np.mean(np.sum((actual-(eh+selected[name]["lambda"]*(validation_predictions[name]-eh)))**2, axis=0)/np.sum((actual-eh)**2, axis=0)))
    selected["selected_family"] = chosen
    (folder/"frozen_parameters.json").write_text(json.dumps(selected, indent=2))
    records = []
    for date, fold in folds.items():
        if date < freeze:
            continue
        predictions = dict(fold["reference"])
        for name in LEARNERS:
            predictions[name] = learner_predict(fold, name, selected[name]["params"])
            predictions[name+"_cal"] = predictions["EH"]+selected[name]["lambda"]*(predictions[name]-predictions["EH"])
        predictions["ValidationSelected"] = predictions[chosen+"_cal"]
        for name, values in predictions.items():
            records.append(dict(ym=str(date), model=name, **{f"mu{n}": float(values[0, n-2]) for n in range(2, 6)}))
    predictions = pd.DataFrame(records)
    predictions.to_csv(folder/"predictions.csv", index=False)
    pd.DataFrame(audit).to_csv(folder/"forecast_audit.csv", index=False)
    validations = []
    for i, date in enumerate(validation_dates):
        for name in LEARNERS:
            for n in range(2, 6):
                validations.append(dict(ym=str(date), model=name, maturity=n, actual=actual[i, n-2],
                                        prediction=validation_predictions[name][i, n-2], benchmark=eh[i, n-2]))
    pd.DataFrame(validations).to_csv(folder/"validation_predictions.csv", index=False)
    marker.write_text(fingerprint)
    return selected


def evaluate(bonds, output):
    all_scores, all_performance = [], []
    for tag in ("core_revised", "headline_vintage", "headline_revised"):
        for horizon in (12, 1):
            folder = output/tag/f"h{horizon}"
            p = pd.read_csv(folder/"predictions.csv")
            p["ym"] = pd.PeriodIndex(p.ym, freq="M")
            target = [f"{'rx' if horizon == 12 else 'mx'}{n}" for n in range(2, 6)]
            for period, sale_start, sale_end in [("test", "2017-01", "2022-12"),
                                                ("followup", "2023-01", PROTOCOL["followup_sale_end"])]:
                subset = p[(p.ym+horizon >= pd.Period(sale_start, "M")) & (p.ym+horizon <= pd.Period(sale_end, "M"))]
                eh = subset[subset.model == "EH"].set_index("ym")
                rows, loss = [], []
                for name, g in subset.groupby("model"):
                    dates = pd.PeriodIndex(g.ym, freq="M")
                    y = bonds.loc[dates, target].to_numpy()
                    pred = g[[f"mu{n}" for n in range(2, 6)]].to_numpy()
                    bench = eh.loc[dates, [f"mu{n}" for n in range(2, 6)]].to_numpy()
                    for j, maturity in enumerate([2, 3, 4, 5, "basket"]):
                        yy, pp, bb = (y[:, j], pred[:, j], bench[:, j]) if j < 4 else (y.mean(axis=1), pred.mean(axis=1), bench.mean(axis=1))
                        _, rawp = loss_test(yy, pp, bb, lags=18 if horizon == 12 else 3)
                        nested = name in {"FB", "PCA5_OLS"} if j < 4 else name in {"CP05", "LN6", "PCA5_OLS"}
                        _, cwp = loss_test(yy, pp, bb, nested=True, lags=18 if horizon == 12 else 3)
                        rows.append(dict(panel=tag, horizon=horizon, period=period, model=name, maturity=maturity,
                                         n=len(dates), purchase_start=str(dates.min()), purchase_end=str(dates.max()),
                                         sale_end=str(dates.max()+horizon), r2_pct=100*oos_r2(yy, pp, bb), raw_hac_p=rawp,
                                         cw_p=cwp if nested else np.nan))
                        if j == 4 and name != "EH":
                            loss.append((name, (yy-bb)**2-(yy-pp)**2))
                table = pd.DataFrame(rows)
                alternatives = table.model != "EH"
                table.loc[alternatives, "holm_p"] = multipletests(table.loc[alternatives, "raw_hac_p"], method="holm")[1]
                loss_matrix = np.column_stack([d for _, d in loss])
                bootstrap = joint_block_bootstrap_t(loss_matrix, block=12 if horizon == 12 else 6,
                                                    lags=18 if horizon == 12 else 3, replications=1999)
                for i, (name, _) in enumerate(loss):
                    mask = table.model.eq(name) & table.maturity.eq("basket")
                    table.loc[mask, "max_t_p"] = bootstrap["max_t_p"][i]
                table.to_csv(folder/f"{period}_forecast_scores.csv", index=False)
                all_scores.append(table)
                if horizon == 1:
                    use = subset.copy(); use.model += "/frozen"
                    legacy = ROOT/"output"/"refinement"/"prior_candidates"/("vintage" if tag == "headline_vintage" else "revised")/"comparison_predictions.csv"
                    if legacy.exists():
                        old = pd.read_csv(legacy)
                        old["ym"] = pd.PeriodIndex(old.ym, freq="M")
                        old = old[old.model.eq("PriorEW3/w120") & old.ym.isin(subset.ym.unique())].copy()
                        old["model"] = "PriorEW3/frozen"
                        if old.ym.nunique() != subset.ym.nunique():
                            raise ValueError("Prior candidate comparison has an incomplete calendar")
                        use = pd.concat([use, old], ignore_index=True)
                    summary, trades = monthly_backtest(bonds, use, policy="long_only", cap=1, shrink=1,
                                                       cost_bps=.5, gamma=5)
                else:
                    # Annual log forecasts are evaluated statistically; the
                    # independently refitted monthly simple-return model trades.
                    continue
                summary["panel"], summary["period"] = tag, period
                summary.to_csv(folder/f"{period}_performance.csv", index=False)
                trades.to_csv(folder/f"{period}_trades.csv", index=False)
                all_performance.append(summary)
                # Annualized monthly mean-variance utility, not annual compounded
                # CER. Paired monthly blocks preserve the matched account paths.
                wide = trades.pivot(index="purchase", columns="model", values="return_net")
                values = wide.to_numpy()
                utility = lambda a: 12*(a.mean(axis=-2)-5*a.var(axis=-2, ddof=1)/2)
                point = utility(values)
                rng = np.random.default_rng(42); n = len(wide)
                starts = rng.integers(0, n, (1999, int(np.ceil(n/12))))
                ix = ((starts[:, :, None]+np.arange(12)) % n).reshape(1999, -1)[:, :n]
                draws = utility(values[ix])
                utility_rows = []
                for benchmark in ["EH/frozen", "BuyHold"]:
                    k = wide.columns.get_loc(benchmark)
                    for j, name in enumerate(wide.columns):
                        gain = 10000*(draws[:, j]-draws[:, k])
                        utility_rows.append(dict(model=name, benchmark=benchmark,
                            delta_annualized_monthly_cer_bps=10000*(point[j]-point[k]),
                            ci_low=np.quantile(gain, .025), ci_high=np.quantile(gain, .975),
                            block_months=12, replications=1999))
                pd.DataFrame(utility_rows).to_csv(folder/f"{period}_utility.csv", index=False)
                for cost in (2, 5):
                    stress, _ = monthly_backtest(bonds, use, policy="long_only", cap=1, shrink=1, cost_bps=cost, gamma=5)
                    stress.to_csv(folder/f"{period}_cost{cost}.csv", index=False)
                for label, kwargs in [("gamma2", dict(gamma=2)), ("gamma10", dict(gamma=10)),
                                      ("delay1", dict(delay=1)), ("risk120", dict(risk_window=120))]:
                    stress, _ = monthly_backtest(bonds, use, policy="long_only", cap=1, shrink=1, cost_bps=.5, **kwargs)
                    stress.to_csv(folder/f"{period}_{label}.csv", index=False)
    pd.concat(all_scores, ignore_index=True).to_csv(output/"all_forecast_scores.csv", index=False)
    pd.concat(all_performance, ignore_index=True).to_csv(output/"all_performance.csv", index=False)


def export(output, update):
    scores = pd.read_csv(output/"all_forecast_scores.csv")
    perf = pd.read_csv(output/"all_performance.csv")
    report = ['<a id="validated-extension"></a>', "### 5.6 基准校验后的 CP15 与验证期机器学习实验", "",
              "本节先执行 §5.5 全部数值校验，通过后才调参。它是按已披露截图提出的后续历史研究，",
              "所有历史期间此前已经看过，因此 2017–2022 不称为完全未触碰的留出集。",
              "训练起点 1994-01；验证预测从 2011-01 开始；2017-01 冻结模型超参数、预测收缩及模型族选择，回归系数仍逐月重估。",
              "只有在冻结前严格兑现的验证标签可参与选择；一年目标的最后验证买入月为 2015-12，",
              "一月目标为 2016-11。测试预测始于 2017-01，收益须在 2022-12 前兑现；后续期间另报。", "",
              "CP15 遵循趋势通胀 DMA(120 月、v=0.987)、逐期限收益率去趋势、一年周期及 2–20 年平均周期，",
              "并用按期限标准化的 2–20 年收益构造单一指数，再预测 2–5 年各期限。曲线改用 GSW，属于数据适配；",
              "一月目标是相同构造在另一预测期的扩展，不冒充原文一年期结果。核心通胀来自 FRED CPILFESL（修订值）；",
              "历史版本使用当时 FRED-MD 的 CPIAUCSL 总通胀，是 CP15 讨论过的替代通胀口径。",
              "headline_revised 完全沿用 headline_vintage 的参数及收缩，形成同口径版本对照。", "",
              "CP_EMA3/6/12 用对应半衰期平滑五个远期利率后重新估计共同指数，是文献启发的历史信息扩展，",
              "没有声称复现 Feunou–Fontaine 的完整无套利 Gaussian DTSM。Full_OLS 联合六个指数；",
              "PCA5 及 FullPCA 单列。ElasticNet/XGBoost 分别保留真实数据和 Gaussian Copula 增强对照；",
              "后者仅拟合成熟训练 X/Y 的经验边际与高斯依赖，加入同等数量合成行，不增加独立历史样本。",
              "Optuna 分别只在验证期选择 12/16 个 EN/XGB 候选。收缩 λ 在相同验证期按四期限 MSE 一次拟合，范围 [0,1]，",
              "之后不按测试成绩调节。预先指定展示 XGB_GC_cal，同时报告验证期选中的模型及全部失败对照。",
              "一年预测的主评价有 60 个买入月份（2017-01–2021-12），实现于 2018-01–2022-12；",
              "一月交易有 71 个买入月份（2017-01–2022-11），实现于 2017-02–2022-12。截图未提供原代码，",
              "本节验证其方法路线，不声称逐项重现其未披露的样本日期、因子、调参和交易设定。",
              "[BLS 确认 2025-10 CPI 发布中断](https://www.bls.gov/cpi/additional-resources/2025-federal-government-shutdown-impact-cpi-faq.htm)，",
              "完整 DMA 窗口从 2025-12 预测起缺失；不回填未来数据，全部面板后续评价统一截至 2025-12 的已实现收益。", "",
              "| 面板 | 一年预测模型 | 2y R² (%) | 3y | 4y | 5y | 篮子 R² (%) |", "| --- | --- | --- | --- | --- | --- | --- |"]
    names = ["CP05", "LN6", "CP15", "CP_EMA6", "Full_OLS", "EN_real_cal", "EN_GC_cal", "XGB_real_cal", PRIMARY, "ValidationSelected"]
    for tag in ("core_revised", "headline_vintage"):
        q = scores[(scores.panel == tag) & (scores.horizon == 12) & (scores.period == "test")]
        for name in names:
            rows = q[q.model == name].set_index("maturity")
            report.append("| "+" | ".join([tag, name]+[f"{rows.loc[str(n), 'r2_pct']:.2f}" for n in [2, 3, 4, 5, "basket"]])+" |")
    report += ["", "| 面板/期间 | 月度交易模型 | CAGR (%) | 年均超额收益 (%) | 夏普 | 月末回撤 (%) |", "| --- | --- | --- | --- | --- | --- |"]
    for tag in ("core_revised", "headline_vintage"):
        q = perf[(perf.panel == tag) & (perf.period == "test")]
        for name in ["EH/frozen", "CP15/frozen", PRIMARY+"/frozen", "ValidationSelected/frozen", "PriorEW3/frozen", "BuyHold"]:
            row = q[q.model == name].iloc[0]
            report.append(f"| {tag}/test | {name} | {row.cagr_pct:.2f} | {row.mean_excess_pct:.2f} | {row.sharpe:.2f} | {row.max_dd_pct:.2f} |")
    report += ["", "| 面板 | 一年篮子模型 | 主评价 R² (%) | 原始 HAC p | 联合 max-t p | 后续期间 R² (%) |", "| --- | --- | --- | --- | --- | --- |"]
    for tag in ("core_revised", "headline_vintage"):
        for name in ["CP15", PRIMARY, "ValidationSelected"]:
            q = scores[(scores.panel == tag) & (scores.horizon == 12) & (scores.maturity == 'basket') & (scores.model == name)]
            test, later = q[q.period == 'test'].iloc[0], q[q.period == 'followup'].iloc[0]
            report.append(f"| {tag} | {name} | {test.r2_pct:.2f} | {test.raw_hac_p:.3f} | {test.max_t_p:.3f} | {later.r2_pct:.2f} |")
    report += ["", "| 面板 | 月度模型 | 年化月 CER 对 EH 差 (bps) | 配对区块 95% 区间 (bps) |", "| --- | --- | --- | --- |"]
    for tag in ("core_revised", "headline_vintage"):
        q = pd.read_csv(output/tag/'h1'/'test_utility.csv')
        for name in ['CP15/frozen', PRIMARY+'/frozen', 'PriorEW3/frozen']:
            row = q[q.model.eq(name) & q.benchmark.eq('EH/frozen')].iloc[0]
            report.append(f"| {tag} | {name} | {row.delta_annualized_monthly_cer_bps:.1f} | [{row.ci_low:.1f}, {row.ci_high:.1f}] |")
    report += ["", "一年期的统计预测表与一月期的交易表使用分别重新拟合的目标，不能混成同一模型的预测与投资业绩。",
               "一年预测不直接作为月收益复利。月度交易沿用同债券折旧重定价、实际现金收益、入场费预留、",
               "γ=5、多头上限 1 与单边 0.5bps；另保存 2/5bps。累计主动收益可按月差相加，CAGR 则按净值复利计算，",
               "两者不混用。另保存 γ=2/10、延迟一月与 120 月风险窗口。效用另按 12×月度 CER 差计算，",
               "配对 12 月区块给出条件于既有预测的区间，不能称为复合年度 CER 或独立实盘证据。",
               "完整逐期限原始 HAC、适用的 CW、Holm 与篮子 max-t 校正和后续期间在本地结果表中。", "",
               "![验证期选择后的预测与交易对照](assets/validated_extension.png)", "",
               "**研究判断。** CP15 在修订数据和历史总通胀版本均出现正的一年期 OOS R²，平滑远期利率也有改善，",
               "因此不能由旧模型的失败推断这些因子没有预测价值。XGBoost+Copula 在修订核心通胀面板主期有所改善，",
               "但没有在历史版本的一年主期重复该优势；验证期选中的 EN_GC 更暴露出疫情外推风险，失败结果保留。",
               "后续区间一些非线性模型有较高 R²，但区间短、联合校正证据仍有限，不能将其单独挑成全期结论。",
               "月度主期 CP15 配置优于相同历史均值配置的收益/回撤点估计，但既有 EW3 在该期收益更高；",
               "CP15 在 2023–2025 月度版本全部持有现金，较高总收益来自现金利息，不解释为债券超额收益。",
               "本轮增加可复核的因子和模型证据，保留原有 EW3 候选，不因新模型名或短期排名替换主策略。", "",
               "来源：[CP15 作者页面](https://pavol.povala.com/publication/2015_journal_cieslak_povala_rfs/)、",
               "[核心 CPI](https://fred.stlouisfed.org/series/CPILFESL)、",
               "[期限结构历史信息](https://publications.gc.ca/site/eng/465305/publication.html)。",
               "复现：`uv sync --extra report --extra research`，先运行 `python scripts/fetch_literature_inputs.py`，",
               "然后 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/validate_and_extend.py --update-readme`。",
               "本地 `output/validated_extension/` 保留参数、全部 Optuna trial、验证预测、逐期截止日、预测及交易流水；原始数据不上传。"]
    body = "\n".join(report)+"\n"
    (output/"report.md").write_text(body)
    if update:
        path = ROOT/"README.md"; text = path.read_text()
        start, end = "<!-- VALIDATED_EXTENSION:START -->", "<!-- VALIDATED_EXTENSION:END -->"
        block = start+"\n"+body+end
        if start in text:
            a, b = text.index(start), text.index(end)+len(end)
            text = text[:a]+block+text[b:]
        else:
            text += "\n\n"+block+"\n"
        path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"output"/"validated_extension")
    parser.add_argument("--update-readme", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    # An executable prerequisite, rather than trusting a stale 'passed' flag.
    _, checks, _, _ = run_replication()
    print(f"Prerequisite: {len(checks)} literature numerical checks passed", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir/"protocol.json").write_text(json.dumps(PROTOCOL, indent=2))
    bonds, macro = load_long_market(), load_macro_raw_transformed("fredmd")
    core = pd.read_csv(Path(DATA_DIR)/"CPILFESL.csv")
    core = pd.Series(pd.to_numeric(core.CPILFESL, errors="coerce").to_numpy(),
                     index=pd.PeriodIndex(pd.to_datetime(core.iloc[:, 0]), freq="M"))
    if not args.evaluate_only:
        for horizon in (12, 1):
            panel(bonds, macro, core, "core_revised", horizon, args.output_dir)
            with VintageArchive(DATA_DIR) as archive:
                params = panel(bonds, macro, core, "headline_vintage", horizon, args.output_dir, archive=archive)
            panel(bonds, macro, core, "headline_revised", horizon, args.output_dir, inherited=params)
        (args.output_dir/"metadata.json").write_text(json.dumps(dict(source_sha256=source_hashes(),
                versions=versions()), indent=2))
    else:
        saved = json.loads((args.output_dir/"metadata.json").read_text())
        if saved["source_sha256"] != source_hashes() or saved['versions'] != versions():
            raise ValueError("Source/data changed; rerun forecasts instead of using stale output")
    evaluate(bonds, args.output_dir)
    export(args.output_dir, args.update_readme)
    from plot_validated_results import main as plot
    plot(args.output_dir)
    print("Finished verified-reference extension", flush=True)


if __name__ == "__main__":
    main()
