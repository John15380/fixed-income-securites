"""Independent timing, account identity and matched-evaluation checks."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from bond_rp.tuning_evaluation import evaluate_panel, risk_inputs


def market_panel():
    rng = np.random.default_rng(927)
    dates = pd.period_range("2008-01", "2025-11", freq="M")
    common = rng.normal(.015, .25, len(dates))
    market = pd.DataFrame({"rf_month": .1}, index=dates)
    for maturity in range(2, 6):
        market[f"mx{maturity}"] = common+rng.normal(0, .04, len(dates))
        market[f"mr{maturity}"] = market[f"mx{maturity}"]+market.rf_month
    records = []
    for date in dates[dates >= pd.Period("2014-01", "M")]:
        for model, mean in (("EH", .004), ("Signal", .006*np.sin(date.ordinal/5))):
            records.append({"ym": str(date), "model": model,
                            **{f"mu{n}": mean for n in range(2, 6)}})
    return market, pd.DataFrame(records)


class TuningEvaluationTests(unittest.TestCase):
    def test_risk_variance_matches_strict_calendar_formula_and_ignores_future(self):
        bonds, _ = market_panel()
        date = pd.Period("2017-01", "M")
        result = risk_inputs(bonds, pd.PeriodIndex([date], freq="M"))
        mature = bonds.loc["2011-12":"2016-11", [f"mx{n}" for n in range(2, 6)]]/100
        self.assertEqual(len(mature), 60)
        self.assertAlmostEqual(result[2].iloc[0], mature.mean(axis=1).var(ddof=1), places=15)
        self.assertEqual(result[3].iloc[0], "2016-12")
        self.assertEqual(result[0].columns.tolist(), ["2", "3", "4", "5"])
        changed = bonds.copy()
        changed.loc["2016-12":, [f"mx{n}" for n in range(2, 6)]] = 999
        other = risk_inputs(changed, pd.PeriodIndex([date], freq="M"))
        pd.testing.assert_series_equal(result[2], other[2])
        pd.testing.assert_series_equal(result[3], other[3])
        old = bonds.copy()
        old.loc[:"2011-11", [f"mx{n}" for n in range(2, 6)]] = -999
        pd.testing.assert_series_equal(result[2], risk_inputs(old, [date])[2])

    def test_incomplete_risk_history_and_missing_trade_outcomes_are_rejected(self):
        bonds, _ = market_panel()
        date = pd.Period("2017-01", "M")
        missing = bonds.copy()
        missing.loc["2011-12":"2014-01", "mx2"] = np.nan
        with self.assertRaisesRegex(ValueError, "risk history"):
            risk_inputs(missing, [date])
        missing = bonds.copy()
        missing.loc[date, "mr2"] = np.nan
        with self.assertRaisesRegex(ValueError, "realized"):
            risk_inputs(missing, [date])

    def test_full_evaluation_calendar_cash_ledger_and_saved_utility(self):
        bonds, predictions = market_panel()
        with tempfile.TemporaryDirectory() as directory:
            result = evaluate_panel(bonds, predictions, directory, prior_path=False)
            expected = {"allocation_parameters.json", "fullcalibration.csv", "economic_trades.csv",
                        "performance.csv", "utility.csv", "stress.csv"}
            self.assertEqual({path.name for path in Path(directory).iterdir()}, expected)
            protocol = json.loads((Path(directory)/"allocation_parameters.json").read_text())
            self.assertEqual(protocol["last_validation_purchase"], "2016-11")
            self.assertEqual(protocol["last_validation_outcome"], "2016-12")
            self.assertEqual(len(result["calibration"]), 81*2)
            self.assertTrue(result["calibration"].months_first.eq(18).all())
            self.assertTrue(result["calibration"].months_second.eq(17).all())
            performance = result["performance"]
            for period, length in (("test", 71), ("followup", 36),
                                   ("test_2017_2019", 36), ("test_2020_2022", 35)):
                self.assertTrue(performance.loc[performance.period.eq(period), "months"].eq(length).all())
            trades = result["trades"]
            q = trades[[f"q{n}" for n in range(2, 6)]].to_numpy()
            actual = bonds.reindex(pd.PeriodIndex(trades.purchase, freq="M"))[[f"mr{n}" for n in range(2, 6)]].to_numpy()/100
            cash = trades.cash_after_entry.to_numpy()
            rf = trades.risk_free.to_numpy()
            c = .5/10000
            sales = np.sum(q*(1+actual), axis=1)
            wealth = cash*(1+rf)+sales-c*sales
            np.testing.assert_allclose(trades.return_net, wealth-1, atol=1e-15)
            np.testing.assert_allclose(cash+q.sum(axis=1)+c*q.sum(axis=1), 1., atol=1e-15)
            self.assertTrue((cash >= -1e-14).all())
            self.assertTrue((q >= 0).all())
            self.assertTrue((q.sum(axis=1) <= 1).all())
            self.assertTrue((pd.PeriodIndex(trades.risk_last_label_end, freq="M")
                             < pd.PeriodIndex(trades.purchase, freq="M")).all())
            passive = trades.loc[trades.model.eq("BuyHold")]
            np.testing.assert_allclose(passive.gross_exposure, 1/1.00005, atol=1e-14)
            cash_path = trades.loc[trades.model.eq("Cash")]
            np.testing.assert_allclose(cash_path.return_net, cash_path.risk_free, atol=1e-15)
            identity = result["utility"].loc[lambda f: f.model.eq(f.benchmark)]
            np.testing.assert_allclose(identity[["utility_gain_bps", "ci_low_bps", "ci_high_bps"]], 0., atol=1e-12)
            self.assertEqual(set(result["stress"].scenario), {"cost2", "cost5", "gamma2", "gamma10", "delay1"})
            self.assertTrue(result["stress"].evaluation_gamma.eq(5).all())

    def test_test_outcomes_cannot_change_investment_calibration(self):
        bonds, predictions = market_panel()
        changed = bonds.copy()
        changed.loc["2017-01":, [f"mr{n}" for n in range(2, 6)]] += .2
        changed.loc["2017-01":, [f"mx{n}" for n in range(2, 6)]] -= .1
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            a = evaluate_panel(bonds, predictions, first, prior_path=False)
            b = evaluate_panel(changed, predictions, second, prior_path=False)
        self.assertEqual(a["parameters"]["parameters"], b["parameters"]["parameters"])
        pd.testing.assert_frame_equal(a["calibration"], b["calibration"])
        self.assertFalse(np.allclose(a["performance"].cagr_pct, b["performance"].cagr_pct))

    def test_revised_panel_can_inherit_vintage_calibration_without_reoptimization(self):
        bonds, predictions = market_panel()
        alternative = predictions.copy()
        alternative.loc[alternative.model.eq("Signal"), [f"mu{n}" for n in range(2, 6)]] *= -4
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            original = evaluate_panel(bonds, predictions, first, prior_path=False)
            revised = evaluate_panel(bonds, alternative, second, prior_path=False,
                                     inherited_parameters=original["parameters"])
            self.assertEqual(original["parameters"]["parameters"], revised["parameters"]["parameters"])
            self.assertTrue(revised["calibration"].inherited.all())
            self.assertEqual(len(original["calibration"]), len(revised["calibration"]))
            pd.testing.assert_frame_equal(original["calibration"].drop(columns="inherited"),
                                           revised["calibration"].drop(columns="inherited"), check_dtype=False)
            by_path = evaluate_panel(bonds, alternative, second, prior_path=False,
                                     inherited_parameters=Path(first)/"allocation_parameters.json")
            self.assertEqual(by_path["parameters"]["parameters"], revised["parameters"]["parameters"])


if __name__ == "__main__":
    unittest.main()
