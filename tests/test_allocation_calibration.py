"""Independent accounting and validation-only investment-selection checks."""

import itertools
import unittest

import numpy as np
import pandas as pd

from bond_rp.allocation_calibration import (
    allocation_path, annualized_utility, calibrate_allocation,
)


def calibration_panel():
    rng = np.random.default_rng(861)
    dates = pd.period_range("2010-01", "2018-12", freq="M")
    columns = [str(n) for n in range(2, 6)]
    signal = rng.normal(size=(len(dates), 4))*.006
    eh = np.full((len(dates), 4), .003)
    mu = eh+signal
    rf = np.full(len(dates), .001)
    actual = rf[:, None]+.00003+signal/100*.7+rng.normal(size=(len(dates), 4))*.0005
    variance = np.full(len(dates), .00002)
    return (
        pd.DataFrame(mu, index=dates, columns=columns),
        pd.DataFrame(eh, index=dates, columns=columns),
        pd.DataFrame(actual, index=dates, columns=columns),
        pd.Series(rf, index=dates),
        pd.Series(variance, index=dates),
        pd.Series((dates-1).astype(str), index=dates),
    )


class AllocationCalibrationTests(unittest.TestCase):
    def test_full_and_partial_investment_match_independent_cash_ledger(self):
        dates = pd.period_range("2015-01", periods=2, freq="M")
        mu = pd.DataFrame([[.05]*4, [.02]*4], index=dates)
        eh = mu*0
        actual = pd.DataFrame([[.04, -.02, .01, 0], [-.03, .02, -.01, .03]], index=dates)
        rf = pd.Series([.002, .001], index=dates)
        variance = pd.Series(.0001, index=dates)
        cutoffs = pd.Series((dates-1).astype(str), index=dates)
        path = allocation_path(mu, eh, actual, rf, variance, cutoffs, cost_bps=25)
        for row, allocation in enumerate([1/1.0025, .4]):
            # Independent cash ledger: buy four equal lots, pay entry fee,
            # accrue remaining cash, sell aged bonds, pay sale-value exit fee.
            lots = np.repeat(allocation/4, 4)
            entry_fee = .0025*sum(lots)
            cash_account = 1-sum(lots)-entry_fee
            sales = sum(lots[j]*(1+actual.iloc[row, j]) for j in range(4))
            exit_fee = .0025*sales
            wealth = cash_account*(1+rf.iloc[row])+sales-exit_fee
            self.assertAlmostEqual(path.iloc[row].return_net, wealth-1, places=14)
            self.assertAlmostEqual(path.iloc[row].cash_after_entry, cash_account, places=14)
            self.assertAlmostEqual(path.iloc[row].entry_fee, entry_fee, places=14)
            self.assertAlmostEqual(path.iloc[row].exit_fee, exit_fee, places=14)
            self.assertAlmostEqual(path.iloc[row].gross_exposure, allocation, places=14)
        self.assertTrue((path.cash_after_entry >= -1e-14).all())
        self.assertTrue((path.gross_exposure <= 1).all())

    def test_realized_future_returns_never_change_current_positions(self):
        args = calibration_panel()
        before = allocation_path(*args, signal_strength=1.25, bias_sigma=-.05)
        modified = [x.copy() for x in args]
        modified[2].iloc[:] = .4
        modified[3].iloc[:] = -.01
        after = allocation_path(*modified, signal_strength=1.25, bias_sigma=-.05)
        np.testing.assert_array_equal(before[[f"q{n}" for n in range(2, 6)]],
                                      after[[f"q{n}" for n in range(2, 6)]])
        self.assertFalse(np.allclose(before.return_net, after.return_net))

    def test_grid_search_matches_independent_exhaustive_utility_calculation(self):
        args = calibration_panel()
        strengths, biases = (0., .5, 1., 2.), (-.05, 0., .05)
        selected, candidates, path = calibrate_allocation(*args, strengths=strengths, biases=biases)
        dates = args[0].index
        use = (dates >= pd.Period("2011-01", "M")) & (dates <= pd.Period("2016-11", "M"))
        dates = dates[use]
        mu, eh, actual = [x.loc[dates].to_numpy() for x in args[:3]]
        rf, variance = [x.loc[dates].to_numpy() for x in args[3:5]]
        masks = (dates.year <= 2013, dates.year >= 2014)

        def independent_path(strength, bias):
            forecast = (eh+strength*(mu-eh))/100+bias*np.sqrt(variance)[:, None]
            investment = np.clip(np.mean(forecast, axis=1)/(5*variance), 0, 1)
            investment = np.minimum(investment, 1/1.00005)
            remaining_cash = 1-investment-investment*.00005
            sale_value = investment*np.mean(1+actual, axis=1)
            return remaining_cash*(1+rf)+sale_value*(1-.00005)-1

        def independent_utilities(returns):
            return np.array([12*(np.mean(returns[mask])-2.5*np.var(returns[mask], ddof=1))
                             for mask in masks])

        baseline = independent_utilities(independent_path(0, 0))
        expected = []
        for strength, bias in itertools.product(strengths, biases):
            returns = independent_path(strength, bias)
            gain = independent_utilities(returns)-baseline
            score = .5*np.mean(gain)+.5*np.min(gain)
            expected.append(score)
        np.testing.assert_allclose(candidates.score_annual, expected, atol=1e-14)
        best = candidates.loc[candidates.selected].iloc[0]
        self.assertAlmostEqual(best.score_annual, max(expected), places=13)
        self.assertEqual(selected, {"signal_strength": best.signal_strength, "bias_sigma": best.bias_sigma})
        np.testing.assert_allclose(path.return_net,
                                   independent_path(selected["signal_strength"], selected["bias_sigma"]), atol=1e-15)
        self.assertGreaterEqual(best.score_annual, -1e-14)  # Unchanged EH remains feasible.
        self.assertEqual(best.months_2011_2013, 36)
        self.assertEqual(best.months_2014_2016, 35)
        self.assertEqual(best.validation_last_label_end, "2016-12")

    def test_freeze_excludes_test_and_immature_validation_outcomes(self):
        args = calibration_panel()
        first, table, path = calibrate_allocation(*args)
        changed = [x.copy() for x in args]
        for value in changed[:-1]:
            value.loc["2016-12":] = np.nan
            value.loc[:"2010-12"] = 1e8
        changed[-1].loc["2016-12":] = "2099-01"
        second, other, next_path = calibrate_allocation(*changed)
        self.assertEqual(first, second)
        pd.testing.assert_frame_equal(table, other)
        pd.testing.assert_frame_equal(path, next_path)
        self.assertEqual(len(table), 81)
        self.assertEqual(path.index[-1], pd.Period("2016-11", "M"))
        # A later freeze may use the December purchase because it has matured.
        _, full_table, full_path = calibrate_allocation(*args, freeze_at="2017-02")
        self.assertEqual(full_path.index[-1], pd.Period("2016-12", "M"))
        self.assertTrue(full_table.months_2014_2016.eq(36).all())

    def test_risk_maturity_and_complete_calendar_are_enforced(self):
        args = list(calibration_panel())
        args[-1] = pd.Series(args[0].index.astype(str), index=args[0].index)
        with self.assertRaisesRegex(ValueError, "immature"):
            allocation_path(*args)
        args = calibration_panel()
        dates = args[0].index.delete(3)
        with self.assertRaisesRegex(ValueError, "calendar"):
            allocation_path(*(x.loc[dates] for x in args))
        args = list(calibration_panel())
        args[4].iloc[0] = -1
        with self.assertRaisesRegex(ValueError, "Variance"):
            allocation_path(*args)

    def test_gamma_is_fixed_and_utility_uses_net_total_return_sample_variance(self):
        values = np.array([.01, -.02, .004, .008])
        expected = 12*(values.sum()/4-2.5*sum((values-values.mean())**2)/3)
        self.assertAlmostEqual(annualized_utility(values), expected)
        _, candidates, _ = calibrate_allocation(*calibration_panel(), strengths=(0., 1.), biases=(0.,))
        self.assertTrue(candidates.gamma.eq(5).all())
        self.assertTrue(candidates.cap.eq(1).all())
        self.assertTrue(candidates.cost_bps.eq(.5).all())
        with self.assertRaisesRegex(ValueError, "unchanged EH"):
            calibrate_allocation(*calibration_panel(), strengths=(1.,), biases=(0.,))

    def test_later_investment_interval_and_custom_split_are_strictly_separate(self):
        args = calibration_panel()
        kwargs = dict(validation_start="2014-01", split_at="2015-07")
        selected, table, path = calibrate_allocation(*args, **kwargs)
        self.assertEqual(str(path.index.min()), "2014-01")
        self.assertEqual(str(path.index.max()), "2016-11")
        self.assertTrue(table.months_first.eq(18).all())
        self.assertTrue(table.months_second.eq(17).all())
        self.assertTrue(table.first_period.eq("2014-01..2015-06").all())
        self.assertTrue(table.second_period.eq("2015-07..2016-11").all())
        self.assertNotIn("utility_2011_2013", table)
        changed = [value.copy() for value in args]
        for value in changed[:-1]:
            value.loc[:"2013-12"] = np.nan
            value.loc["2016-12":] = np.nan
        again, other, _ = calibrate_allocation(*changed, **kwargs)
        self.assertEqual(again, selected)
        pd.testing.assert_frame_equal(other, table)

    def test_eh_can_be_calibrated_by_the_identical_bias_grid(self):
        args = list(calibration_panel())
        args[0] = args[1].copy()
        selected, table, path = calibrate_allocation(
            *args, validation_start="2014-01", split_at="2015-07")
        self.assertEqual(len(table), 81)
        # With identical model and EH predictions, only bias can change any
        # trade.  Strength=1 wins deterministic ties for each chosen bias.
        self.assertEqual(selected["signal_strength"], 1.)
        for _, group in table.groupby("bias_sigma"):
            np.testing.assert_allclose(group.score_annual, group.score_annual.iloc[0], atol=1e-14)
        same = allocation_path(*(value.loc[path.index] for value in args), **selected)
        pd.testing.assert_frame_equal(path, same)
        self.assertGreaterEqual(table.loc[table.selected, "score_annual"].iloc[0], -1e-14)


if __name__ == "__main__":
    unittest.main()
