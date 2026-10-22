"""Monthly bond aging, information timing and portfolio cashflow identities."""

from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
import pandas as pd
from bond_rp.data import bond_variables
from bond_rp.trading import (combine_windows, forecast_designs, load_cash_returns, monthly_backtest,
                             monthly_curve_returns, svensson_yields)


def market():
    index = pd.period_range("1964-01", "1985-12", freq="M")
    rng = np.random.default_rng(2026)
    params = pd.DataFrame({"BETA0": 4+np.cumsum(rng.normal(0, .08, len(index))),
                           "BETA1": rng.normal(-.5, .1, len(index)), "BETA2": .5,
                           "BETA3": .2, "TAU1": 1., "TAU2": 3.}, index=index)
    yields = svensson_yields(params, np.arange(1, 6))
    prices = pd.DataFrame({f"lp{n}": -n*yields[:, n-1] for n in range(1, 6)}, index=index)
    return bond_variables(prices).join(monthly_curve_returns(params))


class MonthlyTradingTests(unittest.TestCase):
    def test_flat_curve_one_month_carry_matches_cash(self):
        index = pd.period_range("2000-01", periods=30, freq="M")
        p = pd.DataFrame(np.tile([5, 0, 0, 0, 1, -999.99], (len(index), 1)), index=index)
        frame = monthly_curve_returns(p)
        self.assertAlmostEqual(frame.rf_month.iloc[0], 100*np.expm1(.05/12))
        for n in range(2, 6):
            np.testing.assert_allclose(frame[f"mx{n}"].dropna(), 0, atol=1e-12)
        # Using unchanged constant-maturity prices would incorrectly produce zero total return.
        self.assertGreater(frame.mr5.iloc[0], 0)

    def test_missing_month_cannot_be_bridged_by_next_available_price(self):
        index = pd.period_range("2000-01", periods=30, freq="M").delete(5)
        p = pd.DataFrame(np.tile([5, 0, 0, 0, 1, 2], (len(index), 1)), index=index)
        frame = monthly_curve_returns(p)
        self.assertTrue(np.isnan(frame.loc["2000-05", "mr5"]))
        self.assertTrue(np.isnan(frame.loc["2000-06", "mr5"]))

    def test_cash_rf_is_aligned_to_the_return_month(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"rf.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("f.csv", "Comment\n,Mkt-RF,SMB,HML,RF\n200001,1,2,3,0.4\n200002,2,3,4,0.5\n\n2000,1,2,3,5\n")
            rf = load_cash_returns(path)
            self.assertEqual(rf.loc["1999-12"], .4)
            self.assertEqual(rf.loc["2000-01"], .5)
            self.assertEqual(len(rf), 2)

    def test_monthly_forecasts_ignore_unrealized_labels_and_future_macro(self):
        bonds = market()
        rng = np.random.default_rng(44)
        macro = pd.DataFrame(rng.normal(size=(len(bonds), 12)), index=bonds.index,
                             columns=[f"x{i}" for i in range(12)])
        kwargs = dict(horizon=1, start="1985-01", end="1985-01", verbose=False)
        first, audit = forecast_designs(bonds, macro, **kwargs)
        changed = bonds.copy()
        changed.loc[changed.index+1 >= pd.Period("1985-01", "M"), [f"mx{n}" for n in range(2, 6)]] += 999
        changed_macro = macro.copy()
        changed_macro.loc["1984-12":] += 999
        second, _ = forecast_designs(changed, changed_macro, **kwargs)
        np.testing.assert_allclose(first[[f"mu{n}" for n in range(2, 6)]], second[[f"mu{n}" for n in range(2, 6)]])
        self.assertTrue((pd.PeriodIndex(audit.last_label_end, freq="M") < pd.Period("1985-01", "M")).all())

    def test_monthly_fees_and_cashflows_match_positions(self):
        bonds = market()
        index = pd.period_range("1980-01", "1984-12", freq="M")
        predictions = pd.DataFrame({"ym": index.astype(str), "model": "EH/exp",
                                     **{f"mu{n}": .5 for n in range(2, 6)}})
        _, trades = monthly_backtest(bonds, predictions, start="1980-01", cost_bps=2.)
        g = trades[trades.model == "EH/exp"]
        dates = pd.PeriodIndex(g.purchase, freq="M")
        q = g[[f"q{n}" for n in range(2, 6)]].to_numpy()
        actual = bonds.loc[dates, [f"mr{n}" for n in range(2, 6)]].to_numpy()/100
        rf = bonds.loc[dates, "rf_month"].to_numpy()/100
        gross = rf+(q*(actual-rf[:, None])).sum(axis=1)
        fees = .0002*(np.abs(q).sum(axis=1)*(1+rf)+(np.abs(q)*(1+actual)).sum(axis=1))
        np.testing.assert_allclose(g.return_net, gross-fees)
        self.assertTrue((q >= 0).all())
        self.assertTrue((np.abs(q).sum(axis=1) <= 1+1e-12).all())
        self.assertTrue((g.cash_after_entry >= -1e-12).all())
        # Changing all future outcomes leaves earlier positions unchanged.
        changed = bonds.copy()
        changed.loc["1980-01":, [f"mx{n}" for n in range(2, 6)]] += 999
        _, again = monthly_backtest(changed, predictions, start="1980-01", cost_bps=2.)
        np.testing.assert_allclose(g.iloc[0][[f"q{n}" for n in range(2, 6)]].astype(float),
                                  again[again.model == "EH/exp"].iloc[0][[f"q{n}" for n in range(2, 6)]].astype(float))

    def test_duration_neutral_positions_have_zero_initial_duration(self):
        bonds = market()
        index = pd.period_range("1980-01", "1984-12", freq="M")
        predictions = pd.DataFrame({"ym": index.astype(str), "model": "EH/exp",
                                     **{f"mu{n}": .1*n for n in range(2, 6)}})
        _, trades = monthly_backtest(bonds, predictions, start="1980-01", policy="duration_neutral")
        q = trades[trades.model == "EH/exp"][[f"q{n}" for n in range(2, 6)]].to_numpy()
        np.testing.assert_allclose(q@np.arange(2, 6), 0, atol=1e-9)
        self.assertTrue((np.abs(q).sum(axis=1) <= 1+1e-12).all())

    def test_execution_delay_preserves_common_evaluation_end(self):
        bonds = market()
        index = pd.period_range("1979-01", "1984-12", freq="M")
        predictions = pd.DataFrame({"ym": index.astype(str), "model": "EH/exp",
                                     **{f"mu{n}": .1 for n in range(2, 6)}})
        first, a = monthly_backtest(bonds, predictions, start="1980-01")
        second, b = monthly_backtest(bonds, predictions, start="1980-01", delay=1)
        self.assertEqual(first.months.iloc[0], second.months.iloc[0])
        self.assertEqual(first.end.iloc[0], second.end.iloc[0])
        np.testing.assert_allclose(a[a.model == "BuyHold"].return_net,
                                  b[b.model == "BuyHold"].return_net)

    def test_window_selector_cannot_use_unrealized_forecast_errors(self):
        bonds = market()
        dates = pd.period_range("1975-01", "1984-12", freq="M")
        rows = []
        for window in ("exp", "w120", "w240"):
            for name, value in [("EH", .1), ("EW", .15), ("EW_R100", .2)]:
                rows.append(pd.DataFrame({"ym": dates.astype(str), "model": f"{name}/{window}",
                                          **{f"mu{n}": value for n in range(2, 6)}}))
        forecasts = pd.concat(rows, ignore_index=True)
        first, audit = combine_windows(bonds, forecasts)
        changed = bonds.copy()
        changed.loc[changed.index+1 >= pd.Period("1980-01", "M"), "mx_bar"] += 999
        second, _ = combine_windows(changed, forecasts)
        cols = [f"mu{n}" for n in range(2, 6)]
        np.testing.assert_allclose(first.loc[first.ym <= "1980-01", cols], second.loc[second.ym <= "1980-01", cols])
        a = audit[audit.last_error_end != ""]
        self.assertTrue((pd.PeriodIndex(a.last_error_end, freq="M") < pd.PeriodIndex(a.ym, freq="M")).all())


if __name__ == "__main__":
    unittest.main()
