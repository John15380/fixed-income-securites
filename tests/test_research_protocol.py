"""Economic identities and leakage regressions; no external datasets required."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV

from bond_rp.data import bond_variables
from bond_rp.metrics import annual_performance, clark_west
from bond_rp.models import (MissingRateFilter, available_features, forecast_models,
                            matured_training, model_pipeline, purged_time_splits)
from bond_rp.strategy import allocation_weights, annual_backtest


def synthetic_bonds():
    rng = np.random.default_rng(1729)
    index = pd.period_range("1964-01", "1983-12", freq="M")
    level = 4 + np.cumsum(rng.normal(0, .12, len(index)))
    prices = pd.DataFrame({f"lp{n}": -n*(level+.12*n) for n in range(1, 6)}, index=index)
    return bond_variables(prices)


class TimingTests(unittest.TestCase):
    def test_training_uses_only_matured_calendar_labels(self):
        index = pd.period_range("1980-01", "1991-12", freq="M").delete([2, 4, 9])
        history = matured_training(pd.DataFrame({"x": 1.}, index=index), pd.Period("1990-01", "M"))
        self.assertEqual(history.index.max(), pd.Period("1988-12", "M"))
        self.assertTrue((history.index+12 < pd.Period("1990-01", "M")).all())

    def test_cv_purges_year_even_with_missing_months(self):
        index = pd.period_range("1964-01", "1984-12", freq="M").delete([5, 11, 20, 67])
        for training, validation in purged_time_splits(index):
            self.assertLess((index[training]+12).max(), index[validation].min())
            self.assertFalse(set(training).intersection(validation))

    def test_minimum_training_window_supports_purged_validation(self):
        index = pd.period_range("1964-01", periods=120, freq="M")
        for training, validation in purged_time_splits(index):
            self.assertGreaterEqual(len(training), 24)
            self.assertLess((index[training]+12).max(), index[validation[0]])

    def test_publication_lag_uses_calendar_not_row_shift(self):
        bonds = synthetic_bonds()
        macro = pd.DataFrame({"a": [11., 33.]}, index=pd.PeriodIndex(["1978-01", "1978-03"], freq="M"))
        features = available_features(bonds, macro, publication_lag=2)
        self.assertEqual(features.loc["1978-03", "macro::a"], 11.)
        self.assertTrue(np.isnan(features.loc["1978-04", "macro::a"]))
        self.assertEqual(features.loc["1978-05", "macro::a"], 33.)

    def test_future_labels_and_inputs_cannot_change_past_forecasts(self):
        bonds = synthetic_bonds()
        rng = np.random.default_rng(45)
        macro = pd.DataFrame(rng.normal(size=(len(bonds), 12)), index=bonds.index)
        macro.columns = [f"x{i}" for i in range(12)]
        original = forecast_models(bonds.loc[:"1979-03"], macro, include_ml=True,
                                   oos_start=pd.Period("1979-01", "M"),
                                   names=["EH", "CP", "LN", "LN_LASSO", "LASSO_raw", "RF_raw", "GBM_raw"], verbose=False)
        changed = bonds.loc[:"1979-03"].copy()
        changed.loc[changed.index+12 >= pd.Period("1979-01", "M"), "sx_bar"] += 1000
        future_macro = macro.copy()
        future_macro.loc["1979-01":] += 1000
        rerun = forecast_models(changed, future_macro, include_ml=True,
                                oos_start=pd.Period("1979-01", "M"),
                                names=["EH", "CP", "LN", "LN_LASSO", "LASSO_raw", "RF_raw", "GBM_raw"], verbose=False)
        np.testing.assert_allclose(original.predictions.loc["1979-01"], rerun.predictions.loc["1979-01"])

    def test_cv_preprocessing_fits_only_fold_training_or_final_refit(self):
        index = pd.period_range("1964-01", periods=150, freq="M")
        rng = np.random.default_rng(33)
        X = pd.DataFrame(rng.normal(size=(150, 10)), index=index, columns=[f"m{i}" for i in range(10)])
        y = X.m0 + rng.normal(size=150)
        splits = purged_time_splits(index)
        allowed = {tuple(index[training]) for training, _ in splits} | {tuple(index)}
        observed = []
        original = MissingRateFilter.fit
        def record_fit(instance, values, target=None):
            observed.append(tuple(values.index))
            return original(instance, values, target)
        with patch.object(MissingRateFilter, "fit", record_fit):
            search = GridSearchCV(model_pipeline("lasso", list(X.columns), pca=True),
                                  {"model__alpha": [.1, 1.]}, cv=splits,
                                  scoring="neg_mean_squared_error", error_score="raise")
            search.fit(X, y)
        self.assertGreater(len(observed), 3)
        self.assertTrue(all(sample in allowed for sample in observed))


class ReturnTests(unittest.TestCase):
    def test_flat_curve_has_zero_simple_excess_return(self):
        index = pd.period_range("1964-01", periods=30, freq="M")
        frame = bond_variables(pd.DataFrame({f"lp{n}": -5*n for n in range(1, 6)}, index=index))
        np.testing.assert_allclose(frame.sx_bar.dropna(), 0, atol=1e-12)
        self.assertAlmostEqual(frame.rf_simple.iloc[0], 100*(np.exp(.05)-1))

    def test_missing_price_does_not_shift_label_to_wrong_month(self):
        bonds = synthetic_bonds()
        raw = bonds[[f"lp{n}" for n in range(1, 6)]].drop(index=pd.Period("1970-01", "M"))
        frame = bond_variables(raw)
        self.assertTrue(np.isnan(frame.loc["1969-01", "sx_bar"]))
        correct = np.expm1((bonds.loc["1970-02", "lp1"]-bonds.loc["1969-02", "lp2"])/100)*100
        self.assertAlmostEqual(frame.loc["1969-02", "simple2"], correct)

    def test_one_missing_maturity_invalidates_the_whole_basket(self):
        bonds = synthetic_bonds()
        prices = bonds[[f"lp{n}" for n in range(1, 6)]].copy()
        prices.loc["1970-01", "lp3"] = np.nan
        outcomes = bond_variables(prices)
        self.assertTrue(np.isnan(outcomes.loc["1969-01", "sx_bar"]))
        self.assertTrue(np.isfinite(outcomes.loc["1969-01", "sx2"]))

    def test_initial_loss_counts_toward_drawdown(self):
        metrics = annual_performance([-.1, .05], [0., 0.])
        self.assertAlmostEqual(metrics["max_dd_pct"], -10.)
        self.assertAlmostEqual(metrics["terminal_wealth"], .9*1.05)
        self.assertAlmostEqual(metrics["cagr_pct"], 100*(np.sqrt(.9*1.05)-1))

    def test_buy_hold_pnl_and_fees_follow_cash_and_asset_identity(self):
        bonds = synthetic_bonds()
        predictions = pd.DataFrame({"EH": 1.}, index=bonds.loc["1979-01":"1982-01"].index)
        _, trades = annual_backtest(bonds, predictions, cost_bps=10)
        row = trades.loc[trades.model == "BuyHold"].iloc[0]
        asset = bonds.loc[row.purchase, [f"simple{n}" for n in range(2, 6)]].mean()/100
        cash = bonds.loc[row.purchase, "rf_simple"]/100
        # Invest one NAV in bonds and borrow the entry fee until sale.
        expected = (1+asset) - .001*(1+cash) - .001*(1+asset) - 1
        self.assertAlmostEqual(row.return_net, expected)
        self.assertTrue((trades.train_last_label_end < trades.purchase).all())
        buys = trades.loc[trades.model == "BuyHold"]
        self.assertTrue(np.all(np.diff(pd.PeriodIndex(buys.purchase).asi8) == 12))

    def test_missing_annual_trade_is_rejected(self):
        bonds = synthetic_bonds()
        index = bonds.loc["1979-01":"1982-01"].index.difference(pd.PeriodIndex(["1980-01"], freq="M"))
        with self.assertRaisesRegex(ValueError, "Missing annual trade"):
            annual_backtest(bonds, pd.DataFrame({"EH": 1.}, index=index))

    def test_covariance_weights_have_correct_units_and_gross_cap(self):
        history = np.random.default_rng(3).normal(0, .03, size=(200, 4))
        for method in ["scalar", "diagonal", "shrinkage"]:
            weights = allocation_weights(.02, history, cap=2., method=method)
            self.assertLessEqual(np.abs(weights).sum(), 2.+1e-10)
        first = allocation_weights(.02, history, cap=100., method="shrinkage")
        scaled = allocation_weights(.2, history*10, cap=100., method="shrinkage")
        np.testing.assert_allclose(first/10, scaled, rtol=1e-5)

    def test_shrinkage_weights_ignore_future_outcomes(self):
        bonds = synthetic_bonds()
        prediction = pd.DataFrame({"EH": 1.}, index=bonds.loc["1979-01":"1982-01"].index)
        _, before = annual_backtest(bonds, prediction, method="shrinkage")
        changed = bonds.copy()
        for n in range(2, 6):
            changed.loc[changed.index+12 >= pd.Period("1979-01", "M"), f"sx{n}"] += 50*n
        _, after = annual_backtest(changed, prediction, method="shrinkage")
        columns = [f"w{n}" for n in range(2, 6)]
        np.testing.assert_allclose(before.loc[before.purchase == pd.Period("1979-01", "M"), columns],
                                   after.loc[after.purchase == pd.Period("1979-01", "M"), columns])

    def test_clark_west_matches_hac_reference(self):
        rng = np.random.default_rng(19)
        actual = np.cumsum(rng.normal(size=200))
        pred, bench = actual*.2, actual*.1
        differences = (actual-bench)**2-(actual-pred)**2+(bench-pred)**2
        fit = sm.OLS(differences, np.ones((200, 1))).fit(
            cov_type="HAC", cov_kwds={"maxlags": 18, "use_correction": True})
        statistic, p = clark_west(actual, pred, bench)
        self.assertAlmostEqual(statistic, fit.tvalues[0])
        self.assertTrue(0 <= p <= 1)


if __name__ == "__main__":
    unittest.main()
