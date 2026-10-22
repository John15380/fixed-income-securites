"""Independent learner and deterministic model-search checks."""
import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from bond_rp.tuning_search import predict_config, search


def synthetic_fold(seed=15):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(90, 11))
    y = X[:, [0]]+rng.normal(size=(90, 4))*.2+np.arange(4)
    x = rng.normal(size=(3, 11))
    return dict(X={k: X.copy() for k in ("index11", "primitive32", "bounded32")},
                x={k: x.copy() for k in ("index11", "primitive32", "bounded32")}, y=y)


class TuningSearchTests(unittest.TestCase):
    def test_zero_augmentation_uses_identical_training_observations(self):
        fold = synthetic_fold()
        config = dict(family="EN", representation="index11", window="expanding",
                      gc_ratio=0, params=dict(alpha=.01, l1_ratio=.5))
        with patch("bond_rp.tuning_search.gaussian_augment", side_effect=AssertionError("No synthetic rows at ratio zero")):
            a = predict_config(fold, config)
        from sklearn.linear_model import ElasticNet
        y, X, x = fold["y"], fold["X"]["index11"], fold["x"]["index11"]
        z = (y-y.mean(0))/y.std(0)
        expected = np.column_stack([ElasticNet(alpha=.01, l1_ratio=.5, max_iter=100000, tol=1e-5)
                                    .fit(X, z[:, j]).predict(x) for j in range(4)])*y.std(0)+y.mean(0)
        np.testing.assert_array_equal(a, expected)

    def test_native_xgboost_matches_independent_targets_with_full_sampling(self):
        try:
            from xgboost import XGBRegressor
        except ImportError:
            self.skipTest("Optional research dependency xgboost is unavailable")
        fold = synthetic_fold()
        params = dict(n_estimators=25, max_depth=2, learning_rate=.05,
                      subsample=1., colsample_bytree=1.)
        config = dict(family="XGB", representation="index11", window="expanding", gc_ratio=0, params=params)
        y = fold["y"]; normalized = (y-y.mean(0))/y.std(0)
        expected = np.column_stack([XGBRegressor(objective="reg:squarederror", tree_method="hist",
                  n_jobs=1, random_state=42, verbosity=0, **params)
                  .fit(fold["X"]["index11"], normalized[:, j]).predict(fold["x"]["index11"])
                  for j in range(4)])*y.std(0)+y.mean(0)
        np.testing.assert_allclose(predict_config(fold, config), expected, atol=1e-7, rtol=1e-7)

    def test_copula_and_prediction_are_reproducible_without_mutating_fold(self):
        fold = synthetic_fold()
        before = fold["y"].copy()
        config = dict(family="EN", representation="primitive32", window="120",
                      gc_ratio=.5, params=dict(alpha=.05, l1_ratio=.2))
        np.testing.assert_array_equal(predict_config(fold, config), predict_config(fold, config))
        np.testing.assert_array_equal(fold["y"], before)

    def test_validation_search_is_deterministic_and_covers_each_group_and_seed(self):
        try:
            import optuna  # noqa: F401
        except ImportError:
            self.skipTest("Optional research dependency optuna is unavailable")
        fold = synthetic_fold()
        actual = np.arange(12).reshape(3, 4)/10
        eh = np.zeros_like(actual)
        folds = {window: [fold] for window in ("expanding", "120", "240")}
        def fake_predict(f, config):
            # Search scheduling can change without exposing future observations.
            strength = float(config["params"].get("alpha", config["params"].get("learning_rate", .1)))
            return np.full((3, 4), strength+config["gc_ratio"]*.02)
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            with patch("bond_rp.tuning_search.predict_config", side_effect=fake_predict):
                first, p = search(folds, actual, eh, a, en_trials=8, xgb_trials=8)
                second, q = search(folds, actual, eh, b, en_trials=8, xgb_trials=8)
            self.assertEqual(first, second)
            for key in p:
                np.testing.assert_array_equal(p[key], q[key])
            import pandas as pd
            trials = pd.read_csv(Path(a)/"trials.csv")
            self.assertEqual(len(trials), 16)
            for model in ("EN_real", "EN_GC", "XGB_real", "XGB_GC"):
                part = trials[trials.model == model]
                self.assertEqual(len(part), 4)
                self.assertEqual(set(part.seed), {42, 73})
                self.assertEqual(first[model]["gc_ratio"] == 0, model.endswith("real"))
            diagnostics = json.loads((Path(a)/"search_diagnostics.json").read_text())
            self.assertFalse(diagnostics["protocol"]["pruning"])
            self.assertEqual(diagnostics["protocol"]["workers"], 2)

    @unittest.skipUnless(importlib.util.find_spec("optuna"), "Optional research dependency optuna is unavailable")
    def test_invalid_benchmark_or_augmentation_is_rejected(self):
        fold = synthetic_fold()
        config = dict(family="EN", representation="index11", window="expanding",
                      gc_ratio=-1, params=dict(alpha=.1, l1_ratio=.5))
        with self.assertRaises(ValueError):
            predict_config(fold, config)
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaises(ValueError):
                search({"expanding": [fold]}, np.ones((3, 4)), np.ones((3, 4)), output, en_trials=4, xgb_trials=4)

    @unittest.skipUnless(importlib.util.find_spec("optuna"), "Optional research dependency optuna is unavailable")
    def test_search_rejects_late_validation_and_unmatured_training_labels(self):
        import pandas as pd
        fold = synthetic_fold()
        fold["dates"] = pd.period_range("2013-01", periods=3, freq="M")
        fold["audit"] = dict(horizon=12, last_label_end="2012-12")
        with tempfile.TemporaryDirectory() as output:
            with self.assertRaisesRegex(ValueError, "strictly before 2014-01"):
                search({"expanding": [fold]}, np.ones((3, 4)), np.zeros((3, 4)), output, en_trials=4, xgb_trials=4)
            fold["dates"] = pd.period_range("2011-01", periods=3, freq="M")
            fold["audit"] = dict(horizon=12, last_label_end="2011-01")
            with self.assertRaisesRegex(ValueError, "immature"):
                search({"expanding": [fold]}, np.ones((3, 4)), np.zeros((3, 4)), output, en_trials=4, xgb_trials=4)


if __name__ == "__main__":
    unittest.main()
