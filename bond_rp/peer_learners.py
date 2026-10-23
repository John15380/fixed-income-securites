"""The slides' dynamic ElasticNetCV and fixed depth-2/100-tree XGBoost.

Inner validation uses real, chronologically purged rows. Copula fits are rebuilt
inside each inner training split; synthetic rows never become validation data.
"""
import numpy as np
from sklearn.linear_model import ElasticNet, enet_path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from .validated_models import gaussian_augment


def augment(X, y, ratio, window=120):
    if ratio == 0:
        return X, y
    count = min(len(X), window) if window else len(X)
    a, b = gaussian_augment(X[-count:], y[-count:], ratio=ratio)
    return np.vstack([X, a[count:]]), np.vstack([y, b[count:]])


def xgb_predict(fold, config):
    from xgboost import XGBRegressor
    y = fold["y"]
    mean, std = y.mean(0), np.maximum(y.std(0), 1e-8)
    X, Y = augment(fold["X"], (y-mean)/std, config.get("ratio", 0), config.get("window", 120))
    model = XGBRegressor(objective="reg:squarederror", tree_method="hist", multi_strategy="one_output_per_tree",
        n_estimators=100, max_depth=2, n_jobs=1, random_state=42, verbosity=0,
        subsample=1., colsample_bytree=1., **config["params"])
    return model.fit(X, Y).predict(fold["x"])*std+mean


def elastic_predict(fold, *, copula=False):
    X, y, x = fold["X"], fold["y"], fold["x"]
    alphas, l1s = np.logspace(0, -3, 10), (.1, .5, .9)
    losses = np.zeros((len(l1s), len(alphas), 4))
    audited = []
    for inner in fold["inner_folds"]:
        if len(inner["y"]) < 24:
            raise ValueError("Insufficient history after inner annual purge")
        # Supervised literature indices, macro PCA and feature PCA are all
        # independently rebuilt at the inner training boundary by the runner.
        a, b = inner["X"], inner["x"]
        mean, std = inner["y"].mean(0), np.maximum(inner["y"].std(0), 1e-8)
        target = (inner["y"]-mean)/std
        a, target = augment(a, target, 1. if copula else 0., 120)
        center, location = a.mean(0), target.mean(0)
        for j in range(4):
            for k, l1 in enumerate(l1s):
                _, coefficients, _ = enet_path(np.asfortranarray(a-center), target[:, j]-location[j],
                    l1_ratio=l1, alphas=alphas, max_iter=50000, tol=1e-4)
                predictions = ((b-center)@coefficients+location[j])*std[j]+mean[j]
                losses[k, :, j] += ((predictions-inner["validation_y"][:, j, None])**2).mean(0)
        audited.append((inner["audit"]["last_label_end"], str(inner["dates"][0])))
    mean, std = y.mean(0), np.maximum(y.std(0), 1e-8)
    a, target = augment(X, (y-mean)/std, 1. if copula else 0., 120)
    result, parameters = [], []
    for j in range(4):
        k, idx = np.unravel_index(np.argmin(losses[:, :, j]), losses[:, :, j].shape)
        model = ElasticNet(alpha=alphas[idx], l1_ratio=l1s[k], max_iter=100000, tol=1e-5)
        result.append(model.fit(a, target[:, j]).predict(x)*std[j]+mean[j])
        parameters.append(dict(alpha=float(alphas[idx]), l1_ratio=l1s[k]))
    return np.column_stack(result), dict(parameters=parameters, inner_cutoffs=audited)
