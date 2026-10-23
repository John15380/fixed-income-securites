"""Raw-input checks against the actual reading-list papers, before model tuning.

Published descriptive statistics identify the GDP series independently of any
investment result. Differences in the current SPF archive are retained rather
than adjusting the data to make all historical table cells agree.
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from .config import DATA_DIR
from .survey_expectations import SPF_SERIES, respondent_expectations

# Eriksen (2017), supplied PDF p.79, Table 1 panels A-D. Column order:
# GDP, GDP deflator, unemployment, corporate profits, IP, housing.
PUBLISHED_MEANS = np.array([
    [1.53, .88, .51, 1.42, .77, .23],
    [3.13, 1.77, .52, 3.19, 1.66, 1.67],
    [4.76, 2.65, -.08, 5.19, 2.63, 3.73],
    [6.44, 3.54, -.97, 7.06, 3.61, 5.79]])
PUBLISHED_STDS = np.array([
    [.57, .50, 2.74, 2.11, .79, 5.67],
    [1.11, .97, 4.78, 3.50, 1.32, 10.12],
    [1.62, 1.42, 5.81, 4.45, 1.64, 14.32],
    [2.15, 1.86, 6.55, 5.09, 1.91, 17.78]])


def survey_paper_audit(directory=DATA_DIR):
    """Compare all 24 expectation inputs and the discarded RGDP interpretation.

    The paper's housing table uses 100 times the level change. A constant unit
    conversion does not change a PCA fitted to standardized survey columns.
    Current archive corrections explain residual discrepancies; no threshold is
    imposed on all 48 historical cells as if this were the author's data vintage.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")
        xls = pd.ExcelFile(Path(directory)/"SPFmicrodata.xlsx")
        panels = {n: respondent_expectations(pd.read_excel(xls,n),n).loc["1968Q4":"2014Q4"]
                  for n in (*SPF_SERIES,"RGDP")}
    rows = []
    for j,name in enumerate(SPF_SERIES):
        for h in range(1,5):
            v=panels[name][f"{name}_h{h}"]
            rows.append(dict(series=name,horizon=h,n=int(v.notna().sum()),
                published_mean=PUBLISHED_MEANS[h-1,j],reproduced_mean=v.mean(),
                published_std=PUBLISHED_STDS[h-1,j],reproduced_std=v.std(ddof=1),
                mean_difference=v.mean()-PUBLISHED_MEANS[h-1,j],
                std_difference=v.std(ddof=1)-PUBLISHED_STDS[h-1,j],
                source="Eriksen 2017 supplied PDF p.79 Table1; archive 1968Q4-2014Q4"))
    means = {n:panels[n].mean().to_numpy() for n in ("NGDP","RGDP")}
    errors = {n:float(np.sqrt(np.mean((means[n]-PUBLISHED_MEANS[:,0])**2))) for n in means}
    if errors["NGDP"] >= errors["RGDP"]:
        raise AssertionError("The current survey input does not support the paper's nominal-GDP identification")
    survey=pd.concat([panels[n] for n in SPF_SERIES],axis=1).dropna()
    pc=PCA(3,svd_solver="full").fit(StandardScaler().fit_transform(survey))
    evidence=dict(gdp_series="NGDP",gdp_identification="original Table1 means, independent of returns",
        published_gdp_means=PUBLISHED_MEANS[:,0].tolist(),
        current_gdp_means={n:means[n].tolist() for n in means},mean_rmse=errors,
        quarterly_rows=len(survey),published_first3_variance_pct=[53.,25.,14.],
        current_first3_variance_pct=(100*pc.explained_variance_ratio_).tolist(),
        source_scope="Current SPF historical archive; not the author's unrevised vintage",
        housing_units="100 times respondent level change; unit conversion cancels under standardization")
    return pd.DataFrame(rows),evidence
