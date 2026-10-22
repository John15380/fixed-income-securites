"""Numerical literature checks, using author data only in descriptive replication.

Original full-sample factor files NEVER enter a trading/OOS forecasting pipeline.
Published targets below were checked against the actual PDF tables, not old notes.
"""

from pathlib import Path
import hashlib
import io
import zipfile
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.io import loadmat
from sklearn.decomposition import PCA
from .config import DATA_DIR
from .coursework import common_index
from .data import apply_transform, load_bond_data

AUTHOR_URL = "https://www.sydneyludvigson.com/s/RFS2009.xls"
HANDBOOK_URL = "https://www.sydneyludvigson.com/s/DataReplicationFiles.zip"
LN6 = ["F1", "F1_cube", "F2", "F3", "F4", "F8"]
LN5 = ["F1", "F1_cube", "F3", "F4", "F8"]


def author_ln_data(path=None):
    raw = pd.read_excel(path or Path(DATA_DIR)/"RFS2009.xls", sheet_name="Data Table2", header=None)
    dates = pd.to_datetime(raw.iloc[:, 0], errors="coerce")
    frame = raw.loc[dates.notna()].copy()
    frame.index = pd.PeriodIndex(dates[dates.notna()], freq="M")
    frame.columns = ["date", "CP"]+[f"F{i}" for i in range(1, 9)]+["LN5", "LN6", "blank"]+[f"yr{n}" for n in range(2, 6)]
    frame = frame.drop(columns=["date", "blank"]).astype(float)
    if len(frame) != 480 or frame.index.has_duplicates:
        raise ValueError("Unexpected author archive calendar")
    frame = frame.reindex(pd.period_range("1964-01", "2003-12", freq="M"))
    frame["F1_cube"] = frame.F1**3
    # yr_t is the trailing annual return ending at t. Factors are observed at t.
    # Reindex by calendar, rather than shifting a possibly incomplete row list.
    for n in range(2, 6):
        frame[f"rx{n}"] = frame[f"yr{n}"].reindex(frame.index+12).to_numpy()
    return frame


def run_replication():
    bonds = load_bond_data().set_index("ym")
    author = author_ln_data()
    rows, checks, coefficients = [], [], []

    def check(name, error, tolerance):
        checks.append(dict(check=name, error=float(error), tolerance=tolerance, passed=bool(error <= tolerance)))

    def regress(label, data, terms, published, *, adjusted=True, comparison="author archive"):
        for n, expected in zip(range(2, 6), published):
            fit = sm.OLS(data[f"rx{n}"], sm.add_constant(data[terms], has_constant="add")).fit()
            value = fit.rsquared_adj if adjusted else fit.rsquared
            rows.append(dict(model=label, maturity=n, n=len(data), purchase_start=str(data.index.min()),
                             purchase_end=str(data.index.max()), last_sale=str(data.index.max()+12),
                             statistic="adjusted R2" if adjusted else "R2", published=expected,
                             reproduced=value, difference=value-expected, comparison=comparison))
        return fit

    # Independently reconstruct every calendar-aligned outcome from raw prices.
    raw = pd.read_csv(Path(DATA_DIR)/"Fama_Bliss_price.txt", sep=r"\s+", comment="%",
                      names=["date", "p1", "p2", "p3", "p4", "p5"])
    raw.index = pd.PeriodIndex(pd.to_datetime(raw.date, format="%Y%m%d"), freq="M")
    max_error = 0.
    for n in range(2, 6):
        sale = raw[f"p{n-1}"].reindex(raw.index+12).to_numpy()
        expected = 100*np.log(sale/raw[f"p{n}"].to_numpy())+100*np.log(raw.p1.to_numpy()/100)
        max_error = max(max_error, np.nanmax(np.abs(expected-bonds.loc[raw.index, f"rx{n}"].to_numpy())))
    check("raw calendar returns vs production", max_error, 1e-10)

    # FB87 original sample. Updated CRSP-derived prices do not exactly reproduce
    # the 1987 data vintage. Do not widen a tolerance and call this an exact match.
    d = bonds.loc["1964-01":"1984-12"]
    for n, slope, r2 in zip(range(2, 6), [.91, 1.13, 1.42, .93], [.14, .11, .11, .05]):
        fit = sm.OLS(d[f"rx{n}"], sm.add_constant(d[[f"fs{n}"]])).fit()
        rows.append(dict(model="FB87", maturity=n, n=len(d), purchase_start="1964-01", purchase_end="1984-12",
                         last_sale="1985-12", statistic="adjusted R2", published=r2, reproduced=fit.rsquared_adj,
                         difference=fit.rsquared_adj-r2, comparison="updated prices; not exact 1987 vintage"))
        coefficients.append(dict(model="FB87", term=f"slope_{n}", published=slope, reproduced=fit.params.iloc[1]))

    # The shared CP implementation uses the actual restricted second stage.
    # Independently check the FB specification against CP's published FB table,
    # whose data period is much closer to the available updated price archive.
    d = bonds.loc["1963-01":"2002-12"]
    for n, expected, slope in zip(range(2, 6), [.16, .17, .18, .09], [.99, 1.35, 1.61, 1.27]):
        fit = sm.OLS(d[f"rx{n}"], sm.add_constant(d[[f"fs{n}"]])).fit()
        rows.append(dict(model="FB in CP05 Table 2", maturity=n, n=len(d), purchase_start=str(d.index.min()),
                         purchase_end=str(d.index.max()), last_sale=str(d.index.max()+12), statistic="R2",
                         published=expected, reproduced=fit.rsquared, difference=fit.rsquared-expected,
                         comparison="updated prices; near published rounding"))
        check(f"FB CP05 R2 maturity {n}", abs(fit.rsquared-expected), .006)
        check(f"FB CP05 slope maturity {n}", abs(fit.params.iloc[1]-slope), .01)
    # Monthly annual returns ending in 1964-2003 correspond to purchases in
    # 1963-2002. Keep this 480-row sample distinct from the LN 468-row pairing.
    d = bonds.loc["1963-01":"2002-12"]
    X = d[[f"f{n}" for n in range(1, 6)]].to_numpy()
    y = d[[f"rx{n}" for n in range(2, 6)]].to_numpy()
    prediction, first, second = common_index(X, y, X, second_intercept=False)
    first_ref = sm.OLS(y.mean(axis=1), sm.add_constant(X)).fit()
    predicted_index = first_ref.fittedvalues[:, None]
    independent = np.column_stack([sm.OLS(y[:, j], predicted_index).fit().predict(predicted_index) for j in range(4)])
    check("CP shared estimator vs independent restricted OLS", np.max(np.abs(prediction-independent)), 1e-9)
    for n, expected in zip(range(2, 6), [.31, .34, .37, .34]):
        value = 1-np.sum((y[:, n-2]-prediction[:, n-2])**2)/np.sum((y[:, n-2]-y[:, n-2].mean())**2)
        rows.append(dict(model="CP05 restricted", maturity=n, n=len(d), purchase_start=str(d.index.min()),
                         purchase_end=str(d.index.max()), last_sale=str(d.index.max()+12), statistic="centered R2",
                         published=expected, reproduced=value, difference=value-expected,
                         comparison="raw updated prices; near published rounding"))
        check(f"CP05 published R2 maturity {n}", abs(value-expected), .006)
    for term, given, value in zip(["constant", "f1", "f2", "f3", "f4", "f5"],
                                  [-3.24, -2.14, .81, 3., .80, -2.08], [first.intercept_, *first.coef_]):
        coefficients.append(dict(model="CP05", term=term, published=given, reproduced=value))

    # Released factors are original-sample estimates: an oracle/descriptive
    # replication only, never a tradable forecast. The usable paired sample is 468.
    d = author.dropna(subset=[f"rx{n}" for n in range(2, 6)])
    regress("LN09 CP", d, ["CP"], [.31, .34, .37, .34])
    regress("LN09 LN5 raw terms", d, LN5, [.22, .18, .16, .14])
    # Table 2 uses six raw terms only in the two-year panel; the other panels
    # omit F2. Leave unreported reference cells empty, rather than invent values.
    regress("LN09 LN6 raw terms", d, LN6, [.26, np.nan, np.nan, np.nan])
    regress("LN09 LN5 released", d, ["LN5"], [.22, .19, .17, .14])
    regress("LN09 LN6 released", d, ["LN6"], [.26, .24, .23, .21])
    regress("LN09 CP+LN5", d, ["CP", "LN5"], [.44, .44, .45, .42])
    regress("LN09 CP+five terms", d, ["CP", *LN5], [.45, .44, .45, .41])
    for name, terms in [("LN5", LN5), ("LN6", LN6)]:
        y = d[[f"rx{n}" for n in range(2, 6)]].to_numpy()
        prediction, first, second = common_index(d[terms], y, d[terms])
        index = first.predict(d[terms])
        check(f"{name} reconstructed index vs released index RMSE", np.sqrt(np.mean((index-d[name])**2)), 1e-3)
        independent = np.column_stack([sm.OLS(y[:, j], sm.add_constant(index)).fit().fittedvalues for j in range(4)])
        check(f"{name} production vs independent OLS", np.max(np.abs(prediction-independent)), 1e-9)
    result = pd.DataFrame(rows)
    for row in result[result.model.isin(["LN09 CP", "LN09 LN5 released", "LN09 LN6 released", "LN09 CP+LN5", "LN09 CP+five terms"])].itertuples():
        check(f"published {row.model} maturity {row.maturity}", abs(row.difference), .0051)

    # Independent PCA reproduction using the author's MATLAB raw data, transform
    # codes, and supplied Fhat_T. This is the 2007 handbook version (131 columns),
    # not falsely described as the original RFS 2009 panel of 132 series.
    with zipfile.ZipFile(Path(DATA_DIR)/"LN_handbook_replication.zip") as z:
        original = loadmat(io.BytesIO(z.read("Data&ReplicationFiles/lndata.mat")))
        released = loadmat(io.BytesIO(z.read("Data&ReplicationFiles/Fhat64.mat")))["Fhat_T"]
    values = original["macrodat"][12:]
    codes = original["vartype"].ravel()
    cols = []
    for j, code in enumerate(codes):
        if np.isfinite(values[:, j]).all():
            cols.append(apply_transform(pd.Series(values[:, j]), int(code)).to_numpy()[48:])
    panel = np.column_stack(cols)
    panel = (panel-panel.mean(axis=0))/panel.std(axis=0, ddof=1)
    pca = PCA(n_components=8, svd_solver="full").fit_transform(panel)
    pca /= pca.std(axis=0, ddof=0)
    sign = np.sign(np.sum(pca*released, axis=0))
    check("eight PCA factors vs author MATLAB scores up to sign", np.max(np.abs(pca*sign-released)), 1e-8)
    # CP15 formula check on the original 470-month predictor interval. The
    # paper uses bootstrapped CMT curves; GSW and revised core CPI are explicit
    # data adaptations, so its .53 is a comparison, not a forced pass tolerance.
    from .validated_models import cp15_cycles, inflation_trend, load_long_market
    long = load_long_market().loc["1971-11":"2010-12"].copy()
    raw = pd.read_csv(Path(DATA_DIR)/"CPILFESL.csv")
    cpi = pd.Series(pd.to_numeric(raw.CPILFESL, errors="coerce").to_numpy(),
                    index=pd.PeriodIndex(pd.to_datetime(raw.iloc[:, 0]), freq="M"))
    long["tau"] = inflation_trend(cpi, lag=1).reindex(long.index)
    X, _ = cp15_cycles(long, long, long)
    cols = [f"long_y{n}" for n in range(1, 21)]
    independent = sm.OLS(long[cols], sm.add_constant(long.tau)).fit()
    residual = long[cols]-independent.fittedvalues
    manual = np.column_stack([residual.iloc[:, 0], residual.iloc[:, 1:].mean(axis=1)])
    check("CP15 production detrending vs independent twenty-yield OLS", np.max(np.abs(X-manual)), 1e-10)
    check("CP15 original sample has 470 observations", abs(len(long)-470), 0)
    target = (long[[f"long_rx{n}" for n in range(2, 21)]].to_numpy()/np.arange(2, 21)).mean(axis=1)
    fit = sm.OLS(target, sm.add_constant(X)).fit()
    rows.append(dict(model="CP15 GSW adaptation", maturity="duration basket 2-20", n=len(long),
                     purchase_start="1971-11", purchase_end="2010-12", last_sale="2011-12", statistic="adjusted R2",
                     published=.53, reproduced=fit.rsquared_adj, difference=fit.rsquared_adj-.53,
                     comparison="GSW vs original bootstrapped CMT; descriptive, original one-month CPI lag"))
    result = pd.DataFrame(rows)
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(DATA_DIR)/n for n in
               ["Fama_Bliss_price.txt", "RFS2009.xls", "LN_handbook_replication.zip", "GSW_yields.csv", "CPILFESL.csv", "F-F_Research_Data_Factors_CSV.zip"]]}
    checks = pd.DataFrame(checks)
    if not checks.passed.all():
        raise AssertionError(checks.loc[~checks.passed].to_string(index=False))
    return result, checks, pd.DataFrame(coefficients), sources
