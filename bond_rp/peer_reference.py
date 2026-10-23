"""Source-mapped reconstruction of the first 22 reference slides.

The slides disclose 82 candidate features but not their series/templates; the
explicit menu below is an identified assumption, never the authors' exact menu.
The ten literature columns follow slide 13's names and slide 10's column count.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from .config import DATA_DIR
from .coursework import common_index
from .metrics import nw_ols
from .models import _numeric_pipeline
from .ms17 import MS17Regression
from .validated_models import cp15_cycles

MACRO_MENU = ("INDPRO", "PAYEMS", "UNRATE", "RPI", "DPCERA3M086SBEA", "HOUST",
              "PERMIT", "RETAILx", "CMRMTSPLx", "CUMFNS", "CPIAUCSL", "PCEPI",
              "PPICMM", "M2SL", "FEDFUNDS", "BAAFFM", "UMCSENTx")
BASE_NAMES = [f"spread{n}_f{n}{n-1}-r1" for n in range(2, 6)]+[
    "CP05", "recursive_factor", "cp2015_factor_pct", "F5_composite", "F6_composite", "ME_factor"]


def candidate_menu(bonds, macro):
    raw = macro[list(MACRO_MENU)].copy()
    raw.index += 2
    raw = raw.reindex(bonds.index)
    e = pd.read_excel(Path(DATA_DIR)/"US_Policy_Uncertainty_Data.xlsx")
    e = e.loc[pd.to_numeric(e.Year, errors="coerce").notna() & pd.to_numeric(e.Month, errors="coerce").notna()]
    epu = pd.Series(pd.to_numeric(e.iloc[:, 2], errors="coerce").to_numpy(),
        index=pd.PeriodIndex(e.Year.astype(int).astype(str)+"-"+e.Month.astype(int).astype(str), freq="M")+2)
    raw["EPU"] = np.log(epu.where(epu > 0)).reindex(raw.index)
    raw["slope"] = bonds.y5-bonds.y1
    raw["curvature"] = 2*bonds.y3-bonds.y1-bonds.y5
    out = {}
    for name in raw:
        s = raw[name]
        out[name+"__sum12"] = s.rolling(12).sum()
        out[name+"__rank12"] = s.rolling(12).rank(pct=True)
        out[name+"__difference3"] = s-s.shift(3)
        out[name+"__momentum6"] = s.rolling(3).mean()-s.shift(3).rolling(3).mean()
    out["curve_slope_level"], out["curve_curvature_level"] = raw.slope, raw.curvature
    assert len(out) == 82
    return pd.DataFrame(out, index=bonds.index)


def screen_candidates(bonds, candidates):
    """Freeze t-stat screening at purchase 2009-12, with outcomes before 2011."""
    index = candidates.loc["1994-01":"2009-12"].index
    y = bonds.loc[index, [f"rx{n}" for n in range(2, 6)]].mean(axis=1)
    rows = []
    for name in candidates:
        x = candidates.loc[index, name]
        valid = x.notna() & y.notna()
        design = np.column_stack([np.ones(valid.sum()), x[valid]])
        fit = LinearRegression().fit(x[valid].to_numpy()[:, None], y[valid])
        residual = y[valid].to_numpy()-fit.predict(x[valid].to_numpy()[:, None])
        se = np.sqrt(np.sum(residual**2)/(valid.sum()-2)/np.sum((x[valid]-x[valid].mean())**2))
        t_ols = float(fit.coef_[0]/se) if se > 0 else 0.
        t_hac = float(nw_ols(y[valid].to_numpy(), design, lags=18).tvalues[1])
        rows.append(dict(feature=name, n=int(valid.sum()), t_ols=t_ols, t_hac=t_hac,
                         selected=abs(t_ols) > 2.5, last_purchase="2009-12", last_label_end="2010-12"))
    table = pd.DataFrame(rows)
    chosen = table.loc[table.selected, "feature"].tolist()
    if len(chosen) < 5:
        raise ValueError("Fewer than five factors pass the disclosed threshold; cannot silently relax screening")
    return chosen, table


def peer_fold(bonds, features, survey, candidates, selected, date, predict_dates=None):
    date = pd.Period(date, "M")
    prediction_dates = pd.PeriodIndex([date] if predict_dates is None else predict_dates, freq="M")
    if prediction_dates.min() < date:
        raise ValueError("A forecast batch cannot precede fitting")
    history = features.loc["1994-01":str(prediction_dates.max())]
    mc = [c for c in features if c.startswith("macro::")]
    train = history.loc[history.index+12 < date]
    valid = train.tau.notna() & bonds.loc[train.index, [f"rx{n}" for n in range(2, 6)]].notna().all(axis=1)
    train = train.loc[valid]
    if len(train) < 120:
        raise ValueError("Too few mature annual returns")
    now = features.loc[prediction_dates]
    y = bonds.loc[train.index, [f"rx{n}" for n in range(2, 6)]].to_numpy()
    pca = _numeric_pipeline(pca=True).fit(train[mc])
    Z, z = pca.transform(train[mc]), pca.transform(now[mc])
    six = lambda s: np.column_stack([s[:, 0], s[:, 0]**3, s[:, 1:4], s[:, 7]])
    ln6, ln6now = six(Z), six(z)
    ln5, ln5now = ln6[:, [0, 1, 3, 4, 5]], ln6now[:, [0, 1, 3, 4, 5]]
    base, current, refs = {}, {}, {"EH": y.mean(0)[None, :]}
    for name, X, x in [("CP05", train[[f"f{n}" for n in range(1, 6)]], now[[f"f{n}" for n in range(1, 6)]]),
                       ("F5_composite", ln5, ln5now), ("F6_composite", ln6, ln6now)]:
        pred, first, _ = common_index(X, y, x, second_intercept=name != "CP05")
        base[name], current[name], refs[name] = first.predict(X), first.predict(x), pred
    cpX, cpx = cp15_cycles(train, now, train)
    long_y = bonds.loc[train.index, [f"long_rx{n}" for n in range(2, 21)]].to_numpy()
    cp = LinearRegression().fit(cpX, (long_y/np.arange(2, 21)).mean(1))
    base["cp2015_factor_pct"], current["cp2015_factor_pct"] = cp.predict(cpX), cp.predict(cpx)
    f = history[[f"f{n}" for n in range(1, 6)]].to_numpy()
    positions = history.index.get_indexer(train.index)
    ms = MS17Regression().fit(f, y, positions)
    factors = ms.factor(f)
    base["recursive_factor"], current["recursive_factor"] = factors[positions], factors[history.index.get_indexer(prediction_dates)]
    scol = [c for c in survey if c != "release_date"]
    sp = _numeric_pipeline().fit(survey.loc[train.index, scol])
    spca = PCA(n_components=3, svd_solver="full").fit(sp.transform(survey.loc[train.index, scol]))
    SX, sx = spca.transform(sp.transform(survey.loc[train.index, scol])), spca.transform(sp.transform(survey.loc[prediction_dates, scol]))
    me = LinearRegression().fit(SX, y.mean(1))
    base["ME_factor"], current["ME_factor"] = me.predict(SX), me.predict(sx)
    for n in range(2, 6):
        name = f"spread{n}_f{n}{n-1}-r1"
        base[name], current[name] = train[f"fs{n}"].to_numpy(), now[f"fs{n}"].to_numpy()
    X, x = np.column_stack([base[n] for n in BASE_NAMES]), np.column_stack([current[n] for n in BASE_NAMES])
    # Only selected macro-template features enter these PCs, not raw LN PCs.
    imp, scale = SimpleImputer(strategy="mean"), StandardScaler()
    C = scale.fit_transform(imp.fit_transform(candidates.loc[train.index, selected]))
    c = scale.transform(imp.transform(candidates.loc[prediction_dates, selected]))
    agent_pca = PCA(n_components=5, svd_solver="full").fit(C)
    PC, pc = agent_pca.transform(C), agent_pca.transform(c)
    blocks = {"L1_FB": (X[:, :4], x[:, :4]), "L1_CP05": (X[:, [4]], x[:, [4]]),
              "L1_MS17": (X[:, [5]], x[:, [5]]), "L1_CP15": (X[:, [6]], x[:, [6]]),
              "L1_Macro": (X[:, 4:][:, [0, 2, 3, 4, 5]], x[:, 4:][:, [0, 2, 3, 4, 5]]),
              "L1_Full": (X, x), "L2_PCA": (PC, pc),
              "L2_Incr": (np.column_stack([X, PC]), np.column_stack([x, pc]))}
    prediction = {"EH": refs["EH"]}
    for name, (a, b) in blocks.items():
        prediction[name] = LinearRegression().fit(a, y).predict(b)
    prediction["L1_FB"] = np.column_stack([LinearRegression().fit(X[:, [j]], y[:, j]).predict(x[:, [j]])
                                            for j in range(4)])
    full, point = blocks["L2_Incr"]
    scaler = StandardScaler().fit(full)
    return dict(X=scaler.transform(full), x=scaler.transform(point), y=y, train_dates=train.index, dates=prediction_dates,
                reference=prediction, audit=dict(ym=str(date), n_train=len(train),
                    last_label_end=str(train.index.max()+12), macro_reference=str(date-2),
                    survey_release=str(survey.loc[date, "release_date"].date()), ms17_alpha=float(ms.alpha_),
                    ms17_converged=ms.converged_, agent_explained5=float(agent_pca.explained_variance_ratio_.sum()),
                    n_features=15, selected_candidates=len(selected)),
                names=BASE_NAMES+[f"PC{n}" for n in range(1, 6)])
