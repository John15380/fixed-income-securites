"""Independent timing, representation and clipping checks for tuning inputs."""
import json
import unittest

import numpy as np
import pandas as pd

from bond_rp.tuning_features import PRIMITIVE_LABELS, prepare_tuning_fold
from bond_rp.validated_models import make_features, prepare_fold


def synthetic_inputs():
    rng = np.random.default_rng(297)
    index = pd.period_range("1980-01", "2024-12", freq="M")
    latent = rng.normal(size=(len(index), 12))
    bonds = pd.DataFrame(index=index)
    for n in range(1, 21):
        bonds[f"long_y{n}"] = 4+.02*n+.2*latent[:, 0]+.05*n*latent[:, 1]
    for n in range(1, 6):
        bonds[f"f{n}"] = 4+.04*n+.2*latent[:, 0]+.06*n*latent[:, 2]
        if n > 1:
            bonds[f"fs{n}"] = bonds[f"f{n}"]-bonds.long_y1
    for prefix in ("rx", "mx"):
        for n in range(2, 21):
            value = .1*n+.4*latent[:, 3]+rng.normal(size=len(index))
            bonds[f"long_{prefix}{n}"] = value
            if n <= 5:
                bonds[f"{prefix}{n}"] = value
    macro = pd.DataFrame(latent, index=index, columns=[f"M{n}" for n in range(12)])
    cpi = pd.Series(np.exp(np.arange(len(index))*.002+.001*latent[:, 4]), index=index)
    return bonds, macro, cpi


class TuningFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bonds, cls.macro, cls.cpi = synthetic_inputs()
        cls.features = make_features(cls.bonds, cls.macro, cls.cpi)
        cls.date = pd.Period("2017-01", "M")

    def test_dimensions_mature_windows_and_unchanged_literature_references(self):
        for horizon in (1, 12):
            for window in ("expanding", 120, 240):
                result = prepare_tuning_fold(self.bonds, self.features, self.date, horizon, window)
                expected_start = max(pd.Period("1994-01", "M"), self.date-horizon-window) if isinstance(window, int) else pd.Period("1994-01", "M")
                original = prepare_fold(self.bonds, self.features, self.date, horizon, train_start=str(expected_start))
                np.testing.assert_array_equal(result["X"]["index11"], original["X"])
                np.testing.assert_array_equal(result["x"]["index11"], original["x"])
                np.testing.assert_array_equal(result["y"], original["y"])
                for name in original["reference"]:
                    np.testing.assert_array_equal(result["reference"][name], original["reference"][name])
                for name, width in [("index11", 11), ("primitive32", 32), ("bounded32", 32)]:
                    self.assertEqual(result["X"][name].shape, (len(result["y"]), width))
                    self.assertEqual(result["x"][name].shape, (1, width))
                    self.assertTrue(np.isfinite(result["x"][name]).all())
                if isinstance(window, int):
                    self.assertEqual(len(result["y"]), window)
                self.assertLess(pd.Period(result["audit"]["last_label_end"], "M"), self.date)
                json.dumps(result["audit"], allow_nan=False)

    def test_unmatured_labels_and_unpublished_macro_cannot_change_inputs(self):
        for horizon in (1, 12):
            changed = self.bonds.copy()
            targets = [c for c in changed if c.startswith(("rx", "mx", "long_rx", "long_mx"))]
            changed.loc[changed.index+horizon >= self.date, targets] += 1e6
            macro, cpi = self.macro.copy(), self.cpi.copy()
            macro.loc[macro.index > self.date-2] += 1e6
            cpi.loc[cpi.index > self.date-2] *= 100
            revised = make_features(changed, macro, cpi)
            for window in ("expanding", 120, 240):
                before = prepare_tuning_fold(self.bonds, self.features, self.date, horizon, window)
                after = prepare_tuning_fold(changed, revised, self.date, horizon, window)
                for name in before["X"]:
                    np.testing.assert_allclose(before["X"][name], after["X"][name], atol=1e-12)
                    np.testing.assert_allclose(before["x"][name], after["x"][name], atol=1e-12)
                np.testing.assert_array_equal(before["y"], after["y"])

    def test_bounds_come_from_training_and_current_cube_retains_linear_pc(self):
        # Freeze the known-history PCA by changing an existing historical month
        # and independently reconstruct its score. The new linear PC is retained
        # even when the cubic input alone has to use a training endpoint.
        altered = self.features.copy()
        altered.loc[self.date, "f1"] = 1e8
        macro_columns = [c for c in altered if c.startswith("macro::")]
        altered.loc[self.date, macro_columns] += 1000
        result = prepare_tuning_fold(self.bonds, altered, self.date, 1, 120)
        audit = result["audit"]
        raw = {}
        for name in ("primitive32", "bounded32"):
            scale = audit["feature_scaling"][name]
            raw[name] = result["x"][name][0]*np.array(scale["scale"])+np.array(scale["mean"])
        lower, upper = np.array(audit["bounded32_bounds"]["lower"]), np.array(audit["bounded32_bounds"]["upper"])
        np.testing.assert_allclose(raw["bounded32"], np.clip(raw["primitive32"], lower, upper), atol=1e-10)
        self.assertEqual(raw["primitive32"][PRIMITIVE_LABELS.index("f1")], 1e8)
        f1 = raw["primitive32"][PRIMITIVE_LABELS.index("macro_pc1")]
        cube = raw["primitive32"][PRIMITIVE_LABELS.index("macro_pc1_cubed")]
        bound = audit["cube_f1_bounds"]
        self.assertAlmostEqual(cube, np.clip(f1, bound["lower"], bound["upper"])**3)
        self.assertGreater(abs(f1), max(abs(bound["lower"]), abs(bound["upper"])))
        self.assertGreater(raw["primitive32"][0], upper[0])

    def test_invalid_choices_are_rejected(self):
        for window in ("all", 0, 60, 180, 120.5):
            with self.assertRaises(ValueError):
                prepare_tuning_fold(self.bonds, self.features, self.date, 1, window)
        with self.assertRaises(ValueError):
            prepare_tuning_fold(self.bonds, self.features, self.date, 3)


if __name__ == "__main__":
    unittest.main()
