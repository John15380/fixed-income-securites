"""Required two-stage factor timing and independent studentization references."""

import unittest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import ttest_1samp
from bond_rp.coursework import (fit_reference, reference_features, reference_forecasts,
                                reference_vintage_forecasts)
from bond_rp.inference import block_bootstrap_t, mean_hac_t
from bond_rp.metrics import loss_test
from bond_rp.tables import prediction_table
from bond_rp.strategy import annual_backtest
from test_research_protocol import synthetic_bonds


class StudentizationTests(unittest.TestCase):
    def test_cw_requires_an_explicit_matched_history_for_sentiment(self):
        rng = np.random.default_rng(41)
        aligned = pd.DataFrame({"actual": rng.normal(size=120), "EH": .1,
                                "BW": rng.normal(scale=.2, size=120)},
                               index=pd.period_range("1990-01", periods=120, freq="M"))
        regular = prediction_table(aligned)
        matched = prediction_table(aligned, nested_models={"BW"})
        a = loss_test(aligned.actual, aligned.BW, aligned.EH, nested=False)
        b = loss_test(aligned.actual, aligned.BW, aligned.EH, nested=True)
        self.assertAlmostEqual(regular.loc["BW", "hac_stat"], a[0])
        self.assertAlmostEqual(matched.loc["BW", "hac_stat"], b[0])
        self.assertEqual(regular.loc["BW", "test"], "loss-difference-HAC")

    def test_hac_mean_statistic_matches_statsmodels(self):
        rng = np.random.default_rng(101)
        values = rng.normal(size=300)+.1
        for k in range(1, len(values)):
            values[k] += .4*values[k-1]
        ref = sm.OLS(values, np.ones((len(values), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": 18, "use_correction": True})
        self.assertAlmostEqual(float(mean_hac_t(values)), ref.tvalues[0], places=11)
        self.assertAlmostEqual(float(mean_hac_t(values, lags=0)), ttest_1samp(values, 0).statistic, places=11)

    def test_studentized_block_test_is_scale_invariant_and_directional(self):
        values = np.random.default_rng(7).normal(.5, 1, 200)
        first = block_bootstrap_t(values, block=12, replications=199)
        scaled = block_bootstrap_t(values*100, block=12, replications=199)
        negative = block_bootstrap_t(-values, block=12, replications=199)
        self.assertEqual(first["bootstrap_p"], scaled["bootstrap_p"])
        self.assertAlmostEqual(first["statistic"], scaled["statistic"])
        self.assertLess(first["bootstrap_p"], negative["bootstrap_p"])
        self.assertEqual(block_bootstrap_t(np.zeros(200), replications=199)["bootstrap_p"], 1)


class RequiredFactorTests(unittest.TestCase):
    def inputs(self):
        bonds = synthetic_bonds()
        macro = pd.DataFrame(np.random.default_rng(42).normal(size=(len(bonds), 12)),
                             index=bonds.index, columns=[f"x{i}" for i in range(12)])
        return bonds, macro

    def test_ln_recipe_and_cp_two_stage_average_identity(self):
        bonds, macro = self.inputs()
        features = reference_features(bonds, macro)
        cols = [f"sx{n}" for n in range(2, 6)]
        data = features.join(bonds[cols]).loc["1965-01":"1977-12"].dropna()
        mc = [c for c in data if c.startswith("macro::")]
        predictions, first, _ = fit_reference(data, data, mc, cols)
        self.assertEqual(first["LN6"]["names"], ["F1", "F1_cubed", "F2", "F3", "F4", "F8"])
        self.assertEqual(first["LN5"]["names"], ["F1", "F1_cubed", "F3", "F4", "F8"])
        np.testing.assert_allclose(first["LN6"]["X"][:, 1], first["LN6"]["X"][:, 0]**3)
        for key, model in [("CP", "CP_2stage"), ("LN6", "LN6_2stage")]:
            np.testing.assert_allclose(predictions[model].mean(axis=1), first[key]["fit"].predict(first[key]["X"]))

    def test_required_forecasts_do_not_use_unmatured_targets_or_later_macro(self):
        bonds, macro = self.inputs()
        kwargs = dict(start="1979-01", end="1979-01", windows=(None,), verbose=False)
        first, audit = reference_forecasts(bonds, macro, **kwargs)
        changed, m2 = bonds.copy(), macro.copy()
        targets = [f"{p}{n}" for p in ("rx", "sx") for n in range(2, 6)]
        changed.loc[changed.index+12 >= pd.Period("1979-01", "M"), targets] += 999
        m2.loc["1978-12":] += 999
        second, _ = reference_forecasts(changed, m2, **kwargs)
        np.testing.assert_allclose(first[[f"pred{n}" for n in range(2, 6)]], second[[f"pred{n}" for n in range(2, 6)]])
        self.assertEqual(audit.last_label_end.iloc[0], "1978-12")
        self.assertTrue((first.model == "FB_n").any())
        self.assertEqual(audit.pca_last_reference.iloc[0], "1978-11")

    def test_available_unlabelled_macro_can_inform_pca(self):
        bonds, macro = self.inputs()
        f = reference_features(bonds, macro)
        targets = [f"sx{n}" for n in range(2, 6)]
        data = f.join(bonds[targets]).dropna()
        mc = [c for c in f if c.startswith("macro::")]
        train = data.loc["1964-01":"1977-12"]
        history = f.loc["1964-01":"1979-01"]
        _, _, pipe = fit_reference(train, data.loc[["1979-01"]], mc, targets, macro_history=history)
        reference = history[mc].mean().to_numpy()
        np.testing.assert_allclose(pipe.named_steps["impute"].statistics_, reference)

    def test_vintage_reference_rejects_a_current_month_snapshot(self):
        bonds, macro = self.inputs()
        class Archive:
            def at(self, date):
                return date, macro
        with self.assertRaisesRegex(ValueError, "predate"):
            reference_vintage_forecasts(bonds, Archive(), start="1979-01", end="1979-01", verbose=False)

    def test_unlevered_annual_policy_reserves_entry_fees(self):
        bonds, _ = self.inputs()
        p = pd.DataFrame({"EH": 100.}, index=bonds.loc["1979-01":"1982-01"].index)
        _, trades = annual_backtest(bonds, p, method="long_only", cap=1, cost_bps=10)
        self.assertGreaterEqual(trades.cash_after_entry.min(), -1e-12)
        q = trades.loc[trades.model == "BuyHold", "net_exposure"]
        np.testing.assert_allclose(q, 1/1.001)


if __name__ == "__main__":
    unittest.main()
