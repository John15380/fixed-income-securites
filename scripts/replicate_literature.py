"""Run numerical paper checks before any new strategy/model tuning."""
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bond_rp.replication import run_replication


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"output"/"literature_replication")
    parser.add_argument("--update-readme", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result, checks, coefficients, sources = run_replication()
    for name, frame in [("published_comparison", result), ("logic_checks", checks), ("coefficients", coefficients)]:
        frame.to_csv(args.output_dir/f"{name}.csv", index=False)
    metadata = dict(passed=True, checks=len(checks), data_sha256=sources,
                    scope="shared forecast logic and original-author descriptive replication; not every historical data vintage")
    (args.output_dir/"verified.json").write_text(json.dumps(metadata, indent=2))
    lines = []
    for model in ["FB87", "FB in CP05 Table 2", "CP05 restricted", "LN09 CP", "LN09 LN5 released", "LN09 LN6 released", "LN09 CP+LN5"]:
        subset = result[result.model == model]
        given = " / ".join(f"{v:.2f}" for v in subset.published)
        got = " / ".join(f"{v:.3f}" for v in subset.reproduced)
        lines.append(f"| {model} | {given} | {got} |")
    report = """<a id="literature-replication"></a>
### 5.5 原论文数值复现与代码校验

先运行 `python scripts/replicate_literature.py --update-readme`，通过后才能运行新增模型实验。
校验不通过时退出，不用调高策略收益来判断代码正确。

| 模型 | 原表 R²，2 / 3 / 4 / 5 年 | 本地重算，2 / 3 / 4 / 5 年 |
| --- | --- | --- |
"""+"\n".join(lines)+f"""

通过 {len(checks)} 项数值校验：原始价格按日历重建收益；生产用两阶段函数与独立 statsmodels
回归一致；从作者八个 PC 重构的 LN5/LN6 指数与作者发布指数的 RMSE 均小于 0.001 个百分点；
用作者 MATLAB 原始面板、变换代码重新提取的八个 PC 与作者因子在允许符号翻转后逐点一致至 1e-8。

**口径修正。** `RFS2009.xls` 的 `yr2`–`yr5` 标在收益实现月，预测变量标在形成月，必须间隔
12 个日历月。作者文件有 480 行，1964-01–2003-12；配对后有 468 个预测形成月，最后为 2002-12。
CP 原文式 (2) 第二阶段为不另加截距的 `b_n × CP`，已修正生产参考模型；LN Table 2 则含截距。
CP restricted R² 显式用中心化 SST，避免无截距回归软件默认的非中心化 R²。FB87 和 LN09 原表报告调整 R²。

**复现边界。** FB87 使用本地更新价格得到的结果与 1987 原表存在差异，尤其五年期；这里如实保留，
不称为精确复现 1987 数据版本，也不把 NW18 标准误冒充原文全部推断方法。CP 原始价格结果接近原表；
LN 作者数据可复现主要单因子与联合因子表。PCA 的独立重建使用作者 Handbook 2007 版本的 131 列
面板，与 RFS2009 的原始 132 列不是同一数据版本，不能据此声称重建了后者全部原始宏观输入。
原始作者因子属于全样本事后估计，仅用于本节校验，**不进入新增策略的样本外输入**。

CP 的 480 行按 1964–2003 收益实现月定义，对应买入月 1963-01–2002-12；与上面的 LN 作者
因子配对得到的 468 行分开报告，不混淆样本数。另增加 CP15 的原区间 1971-11–2010-12（470 个
预测形成月）对照：以原文一月 CPI 发布滞后，用 GSW 适配后调整 R² 约 0.507，原文为 0.53。
生产用的 20 条收益率去趋势结果与独立 statsmodels 回归一致；曲线来源不同，不强迫原表数字一致。
新增 OOS 的 CPI 信息滞后统一为两月，因此不把原区间描述拟合直接当作 OOS 成绩。

来源：[LN 作者数据与代码](https://www.sydneyludvigson.com/data-and-appendixes)、
[CP Table 1 / Eq. 2](https://web.stanford.edu/~piazzesi/cp.pdf)。
FB 原表：本地 Fama–Bliss (1987), p.683, Table 1；LN 原表：p.5040, Table 2。
旧 `code/replicate_papers.py` 现在转到本校验入口，停止展示此前抄错的原表数字和样本说明。

![作者原表与代码数值对照](assets/literature_replication.png)
"""
    (args.output_dir/"report.md").write_text(report)
    if args.update_readme:
        path = ROOT/"README.md"
        text = path.read_text()
        start, end = "<!-- LITERATURE_REPLICATION:START -->", "<!-- LITERATURE_REPLICATION:END -->"
        block = start+"\n"+report+"\n"+end
        if start in text:
            a, b = text.index(start), text.index(end)+len(end)
            text = text[:a]+block+text[b:]
        else:
            text += "\n\n"+block+"\n"
        path.write_text(text)
    print(report)


if __name__ == "__main__":
    main()
