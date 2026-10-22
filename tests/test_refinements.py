"""Independent formula checks and future-data perturbation of refinements."""

import itertools
import unittest
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from bond_rp.inference import joint_block_bootstrap_t
from bond_rp.execution import cost_aware_forecasts
from bond_rp.refinement import (BASE_MODELS, causal_shrinkage, goh_predict, goh_signals,
                                ln_design, refinement_forecasts, robust_bounds, spca_predict)
from test_trading_designs import market


class RefinementTests(unittest.TestCase):
    def test_goh_uses_48_forward_spread_crossovers_and_full_calendar(self):
        index = pd.period_range("2000-01", periods=50, freq="M")
        b = pd.DataFrame({f"fs{n}": np.arange(50.) for n in range(2, 6)}, index=index)
        signals = goh_signals(b)
        self.assertEqual(signals.shape[1], 48)
        self.assertTrue(signals.loc["2002-12"].eq(1).all())
        self.assertTrue(signals.loc["2000-12"].isna().all())
        flat = goh_signals(b*0)
        self.assertTrue(flat.loc["2002-12"].eq(0).all())
        missing = goh_signals(b.drop(index[15]))
        self.assertTrue(missing.loc["2002-12", [c for c in missing if c.endswith(":36")]].isna().all())

    def test_cubic_protection_retains_original_linear_term(self):
        z = np.arange(16., dtype=float).reshape(2, 8)
        design = ln_design(z, (-2, 2))
        np.testing.assert_allclose(design[:, 0], z[:, 0])
        np.testing.assert_allclose(design[:, 1], [0, 8])
        frame = pd.DataFrame({"normal": [1., 2, 3, 4, 5], "rare": [0., 0, 0, 0, 1]})
        low, high = robust_bounds(frame)
        self.assertTrue(np.isinf(high["rare"]))
        self.assertLess(low["normal"], 1)

    def test_spca_matches_literal_univariate_regression_recipe(self):
        rng = np.random.default_rng(57)
        X = rng.normal(size=(150, 10))
        y = rng.normal(size=(150, 4))+X[:, [0]]
        x = rng.normal(size=(2, 10))
        slopes = np.array([LinearRegression().fit(X[:, [i]], y.mean(axis=1)).coef_[0] for i in range(10)])
        pca = PCA(1, svd_solver="full").fit(X*slopes)
        expected = LinearRegression().fit(pca.transform(X*slopes), y).predict(pca.transform(x*slopes))
        np.testing.assert_allclose(spca_predict(X, x, y), expected, atol=1e-12)

    def test_goh_factor_count_and_maturity_bic_match_independent_ols(self):
        rng = np.random.default_rng(29)
        latent = rng.normal(size=(180, 3));loading = rng.normal(size=(3, 48))
        X = latent@loading+.2*rng.normal(size=(180, 48));x = rng.normal(size=(3, 48))
        y = rng.normal(size=(180, 4))+.7*latent[:, [1]]
        history = np.vstack([X, rng.normal(size=(40, 3))@loading+.2*rng.normal(size=(40, 48))])
        pred, selected = goh_predict(X, x, y, history)
        scale = StandardScaler().fit(history)
        pca = PCA(8, svd_solver="full").fit(scale.transform(history))
        Z, z = pca.transform(scale.transform(X)), pca.transform(scale.transform(x))
        H = scale.transform(history)
        history_scores = pca.transform(H)
        criteria = []
        for k in range(9):
            reconstructed = history_scores[:, :k]@pca.components_[:k]+pca.mean_
            variance = np.mean((H-reconstructed)**2)
            criteria.append(np.log(variance)+k*(220+48)/(220*48)*np.log(48))
        count = int(np.argmin(criteria))
        self.assertGreater(count, 0)
        for j in range(4):
            choices = []
            for mask in itertools.product((False, True), repeat=count):
                A = np.column_stack([np.ones(len(y)), Z[:, :count][:, mask]])
                fitted = A@np.linalg.lstsq(A, y[:, j], rcond=None)[0]
                bic = len(y)*np.log(np.mean((y[:, j]-fitted)**2))+A.shape[1]*np.log(len(y))
                choices.append((bic, mask))
            _, mask = min(choices)
            self.assertEqual(selected[j], (np.flatnonzero(mask)+1).tolist())
            A = np.column_stack([np.ones(len(y)), Z[:, :count][:, mask]])
            coefficient = np.linalg.lstsq(A, y[:, j], rcond=None)[0]
            expected = np.column_stack([np.ones(len(x)), z[:, :count][:, mask]])@coefficient
            np.testing.assert_allclose(pred[:, j], expected, atol=1e-10)

    def test_forecasts_ignore_future_returns_and_unpublished_macro(self):
        b = market();rng = np.random.default_rng(6)
        macro = pd.DataFrame(rng.normal(size=(len(b), 12)), index=b.index)
        macro.columns = macro.columns.astype(str)
        kwargs = dict(horizon=1, start="1985-01", end="1985-01", windows=(None,), verbose=False)
        first, audit = refinement_forecasts(b, macro, **kwargs)
        future, m2 = b.copy(), macro.copy()
        future.loc[future.index+1 >= pd.Period("1985-01", "M"), [f"mx{n}" for n in range(2, 6)]] += 999
        m2.loc["1984-12":] += 999
        second, _ = refinement_forecasts(future, m2, **kwargs)
        np.testing.assert_allclose(first[[f"mu{n}" for n in range(2, 6)]], second[[f"mu{n}" for n in range(2, 6)]])
        self.assertEqual(audit.last_label_end.iloc[0], "1984-12")

    def test_online_shrinkage_does_not_see_unmatured_errors(self):
        dates = pd.period_range("1980-01", periods=150, freq="M")
        rows = []
        for family in BASE_MODELS:
            for window in ("exp", "w120", "w240"):
                rows += [{"ym": str(t), "model": f"{family}/{window}", **{f"mu{n}": .1 if family == "EH" else .2 for n in range(2, 6)}} for t in dates]
        p = pd.DataFrame(rows);actual = pd.Series(.15, index=dates)
        first, a = causal_shrinkage(p, actual, horizon=12)
        changed = actual.copy();changed.loc[dates+12 >= dates[-1]] = 100
        second, _ = causal_shrinkage(p, changed, horizon=12)
        x = first[(first.ym == str(dates[-1])) & (first.model == "Core/online")]
        z = second[(second.ym == str(dates[-1])) & (second.model == "Core/online")]
        np.testing.assert_allclose(x.mu2, z.mu2)
        self.assertAlmostEqual(x.mu2.iloc[0], .15)
        used = a[a.last_error_end != ""]
        self.assertTrue((pd.PeriodIndex(used.last_error_end, freq="M") < pd.PeriodIndex(used.ym, freq="M")).all())

    def test_joint_bootstrap_adjustment_and_column_permutation(self):
        d = np.random.default_rng(5).normal(size=(100, 3))+.1
        a = joint_block_bootstrap_t(d, replications=199)
        b = joint_block_bootstrap_t(d[:, ::-1], replications=199)
        np.testing.assert_array_equal(a["max_t_p"], b["max_t_p"][::-1])
        self.assertTrue((a["max_t_p"] >= a["bootstrap_p"]).all())

    def test_cost_aware_signal_matches_net_utility_and_ignores_future_cash(self):
        b = market();date = pd.Period("1980-01", "M")
        p = pd.DataFrame([{"ym": str(date), "model": "Core/online", **{f"mu{n}": .1 for n in range(2, 6)}}])
        adjusted = cost_aware_forecasts(b, p, cost_bps=2)
        c, variance, gamma = .0002, .001, 5
        rf = np.expm1(b.loc[date, "y1"]/1200)
        mean = .001*(1-c)-2*c*(1+rf)
        candidate = adjusted.mu2.iloc[0]/100/(gamma*variance)
        self.assertAlmostEqual(candidate, mean/(gamma*(1-c)**2*variance))
        grid = np.linspace(0, 1, 10001)
        utilities = grid*mean-gamma/2*grid**2*(1-c)**2*variance
        self.assertAlmostEqual(candidate, grid[np.argmax(utilities)], delta=.0001)
        changed = b.copy();changed["rf_month"] = 999
        np.testing.assert_allclose(adjusted.mu2, cost_aware_forecasts(changed, p, cost_bps=2).mu2)


if __name__ == "__main__":
    unittest.main()
