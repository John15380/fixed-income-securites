"""Training-window-only missingness filtering, imputation, scaling and PCA."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class MissingRateFilter(TransformerMixin, BaseEstimator):
    """Select columns using only this fit's observations, including inside CV."""
    def __init__(self, max_missing=0.20):
        self.max_missing = max_missing

    def fit(self, X, y=None):
        values = np.asarray(X, dtype=float)
        self.keep_ = np.mean(~np.isfinite(values), axis=0) <= self.max_missing
        if not self.keep_.any():
            raise ValueError("No usable features in the training window")
        self.n_features_in_ = values.shape[1]
        return self

    def transform(self, X):
        values = np.asarray(X, dtype=float)[:, self.keep_].copy()
        values[~np.isfinite(values)] = np.nan
        return values



def _numeric_pipeline(pca=False):
    steps = [("columns", MissingRateFilter()), ("impute", SimpleImputer(strategy="mean")),
             ("scale", StandardScaler())]
    if pca:
        steps += [("pca", PCA(n_components=8, svd_solver="full")), ("pc_scale", StandardScaler())]
    return Pipeline(steps)
