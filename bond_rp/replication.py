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
from .factors import common_index
from .data import apply_transform, load_bond_data

CP_URL = "https://web.stanford.edu/~piazzesi/cp.zip"
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
    with zipfile.ZipFile(Path(DATA_DIR)/"CP2005_replication.zip") as z:
        raw = pd.read_csv(z.open("cp/bondprice.dat"), sep=r"\s+", comment="%",
                          names=["date", "p1", "p2", "p3", "p4", "p5"])
    raw.index = pd.PeriodIndex(pd.to_datetime(raw.date, format="%Y%m%d"), freq="M")
    max_error = 0.
    for n in range(2, 6):
        sale = raw[f"p{n-1}"].reindex(raw.index+12).to_numpy()
        expected = 100*np.log(sale/raw[f"p{n}"].to_numpy())+100*np.log(raw.p1.to_numpy()/100)
        max_error = max(max_error, np.nanmax(np.abs(expected-bonds.loc[raw.index, f"rx{n}"].to_numpy())))
    check("raw calendar returns vs production", max_error, 1e-10)

    # FB87 original sample. CP archive CRSP-derived prices do not exactly reproduce
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
    # using the same original author price archive and formation dates.
    d = bonds.loc["1964-01":"2002-12"]
    for n, expected, slope in zip(range(2, 6), [.16, .17, .18, .09], [.99, 1.35, 1.61, 1.27]):
        fit = sm.OLS(d[f"rx{n}"], sm.add_constant(d[[f"fs{n}"]])).fit()
        rows.append(dict(model="FB in CP05 Table 2", maturity=n, n=len(d), purchase_start=str(d.index.min()),
                         purchase_end=str(d.index.max()), last_sale=str(d.index.max()+12), statistic="R2",
                         published=expected, reproduced=fit.rsquared, difference=fit.rsquared-expected,
                         comparison="CP author prices; near published rounding"))
        check(f"FB CP05 R2 maturity {n}", abs(fit.rsquared-expected), .006)
        check(f"FB CP05 slope maturity {n}", abs(fit.params.iloc[1]-slope), .01)
    # Author fbregslatest.m: beg=140 selects January 1964 formation,
    # ending December 2002. There are 468 forward annual labels (1965-2003).
    d = bonds.loc["1964-01":"2002-12"]
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
                         comparison="raw CP author prices; near published rounding"))
        check(f"CP05 published R2 maturity {n}", abs(value-expected), .006)
    for term, given, value in zip(["constant", "f1", "f2", "f3", "f4", "f5"],
                                  [-3.24, -2.14, .81, 3., .80, -2.08], [first.intercept_, *first.coef_]):
        coefficients.append(dict(model="CP05", term=term, published=given, reproduced=value))
        check(f"CP05 published coefficient {term}", abs(value-given), .0051)

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
    # codes, and supplied Fhat_T. This is the updated handbook panel (131 columns),
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
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(DATA_DIR)/n for n in
               ["CP2005_replication.zip", "RFS2009.xls", "LN_handbook_replication.zip"]]}
    checks = pd.DataFrame(checks)
    if not checks.passed.all():
        raise AssertionError(checks.loc[~checks.passed].to_string(index=False))
    return result, checks, pd.DataFrame(coefficients), sources


def published_cells(comparison,coefficients):
    """Cell-level coverage, including discrepancies; not a claim to all paper tables.

    Targets were checked visually against CP Table 1 and LN Table 2. Printing
    precision and algorithmic identity tolerances are kept separate. A missed
    printed cell is reported, never turned into a passed reconstruction check.
    """
    rows=[]
    def cell(paper,table,model,maturity,statistic,term,target,actual,decimals=2):
        tolerance=.5*10.**(-decimals)
        rows.append(dict(paper=paper,table=table,model=model,maturity=maturity,statistic=statistic,
                         term=term,published=target,reproduced=float(actual),difference=float(actual)-target,
                         printing_tolerance=tolerance,matches_printing=bool(abs(actual-target)<=tolerance+1e-9)))
    for r in comparison.itertuples():
        if pd.notna(r.published):
            paper="FB87" if r.model=="FB87" else "CP05" if r.model.startswith(("CP05","FB in")) else "LN09"
            table="1" if paper=="FB87" or r.model=="CP05 restricted" else "2"
            cell(paper,table,r.model,r.maturity,r.statistic,"R2",r.published,r.reproduced)
    for r in coefficients.itertuples():
        cell("CP05" if r.model=="CP05" else "FB87","1",r.model,"basket", "coefficient",r.term,r.published,r.reproduced)
    b=load_bond_data().set_index("ym").loc["1964-01":"2002-12"]
    X=sm.add_constant(b[[f"f{n}" for n in range(1,6)]])
    y=b[[f"rx{n}" for n in range(2,6)]].to_numpy()
    fit=sm.OLS(y.mean(axis=1),X).fit(cov_type="HAC",cov_kwds={"maxlags":18,"use_correction":False})
    cell("CP05","1A","CP05","basket","R2","first_stage",.35,fit.rsquared)
    for term,reference in zip(X.columns,[1.31,.34,.69,.55,.46,.41]):
        cell("CP05","1A","CP05","basket","NW18_se",term,reference,fit.bse[term])
    beta=fit.params.iloc[1:].to_numpy()
    wald=beta@np.linalg.inv(fit.cov_params().iloc[1:,1:])@beta
    cell("CP05","1A","CP05","basket","NW18_Wald","joint_slopes",105.5,wald,1)
    pred,first,second=common_index(X.iloc[:,1:],y,X.iloc[:,1:],second_intercept=False)
    for n,target,actual in zip(range(2,6),[.47,.87,1.24,1.43],second.coef_.ravel()):
        cell("CP05","1B","CP05",n,"coefficient","b_n",target,actual)

    # Exact author's HH moment convention: each lag's truncated block is
    # separately demeaned. This differs slightly from generic HAC packages.
    xx=X.to_numpy();g=np.r_[first.intercept_,first.coef_];index=xx@g
    bn=second.coef_.ravel();err=y-index[:,None]*bn
    u=np.column_stack([(y.mean(1)-index)[:,None]*xx,err*index[:,None]])
    N=len(xx);S=u.T@u/N
    for k in range(1,13):
        a=u[k:]-u[k:].mean(0);c=u[:-k]-u[:-k].mean(0)
        q=a.T@c/N;S+=q+q.T
    eff=xx.T@xx/N
    jacobian=-np.block([[eff,np.zeros((6,4))],[-y.T@xx/N+2*np.outer(bn,g)@eff,(g@eff@g)*np.eye(4)]])
    inv=np.linalg.inv(jacobian)
    se=np.sqrt(np.diag(inv@S@inv.T/N))
    for n,target,actual in zip(range(2,6),[.03,.02,.01,.04],se[6:]):
        cell("CP05","1B","CP05",n,"two_stage_HH12_se","b_n",target,actual)

    author=author_ln_data().dropna(subset=[f"rx{n}" for n in range(2,6)])
    # For each specification, rows are maturities 2..5; None means unreported.
    definitions=[
        ("CP",["CP"],[[.45],[.85],[1.24],[1.46]],[[8.90],[8.52],[8.58],[7.90]]),
        ("LN5 terms",LN5,[[-.93,.06,.18,-.33,.35],[-1.59,.11,.19,-.53,.64],[-2.05,.16,.18,-.63,.95],[-2.27,.18,.18,-.78,1.13]],
         [[-4.96,2.87,1.87,-2.65,3.83],[-4.68,3.12,1.05,-2.23,3.73],[-4.49,3.20,.68,-1.77,3.75],[-4.10,3.06,.55,-1.80,3.68]]),
        ("LN6 terms",LN6,[[-.93,.06,-.40,.18,-.33,.35],None,None,None],
         [[-5.19,2.78,-3.10,2.24,-2.94,4.35],None,None,None]),
        ("LN5 released",["LN5"],[[.54],[.91],[1.19],[1.36]],[[5.52],[5.28],[5.08],[4.80]]),
        ("LN6 released",["LN6"],[[.50],[.89],[1.20],[1.41]],[[6.78],[6.57],[6.57],[6.47]]),
        ("CP+LN5",["CP","LN5"],[[.39,.43],[.75,.69],[1.11,.87],[1.32,.98]],
         [[6.,5.78],[6.16,5.55],[6.30,5.39],[5.87,5.08]]),
        ("LN5 terms+CP",[*LN5,"CP"],
         [[-.75,.05,.24,-.25,.24,.40],[-1.22,.10,.30,-.36,.44,.76],[-1.51,.14,.35,-.37,.64,1.13],[-1.63,.15,.38,-.48,.76,1.34]],
         [[-4.71,2.71,3.85,-2.61,2.89,5.89],[-4.39,2.96,2.78,-2.12,2.74,6.13],[-4.20,3.08,2.22,-1.50,2.83,6.40],[-3.86,2.95,1.92,-1.54,2.76,6.00]]),
        ("LN6 terms+CP",[*LN6,"CP"],[[-.74,.05,.08,.24,-.24,.24,.41],None,None,None],
         [[-4.48,2.70,.71,3.84,-2.51,2.70,5.22],None,None,None])]
    for name,terms,targets,tvalues in definitions:
        for n,values,trefs in zip(range(2,6),targets,tvalues):
            if values is None:
                continue
            fit=sm.OLS(author[f"rx{n}"],sm.add_constant(author[terms])).fit(
                cov_type="HAC",cov_kwds={"maxlags":18,"use_correction":False})
            for term,expected,expected_t in zip(terms,values,trefs):
                cell("LN09","2",name,n,"coefficient",term,expected,fit.params[term])
                decimals=1 if name=="CP+LN5" and n==2 and term=="CP" else 2
                cell("LN09","2",name,n,"NW18_t",term,expected_t,fit.tvalues[term],decimals)
    return pd.DataFrame(rows)
