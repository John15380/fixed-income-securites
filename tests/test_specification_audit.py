"""Causality checks for the wider search, vintage loader and portfolio policies."""

import io
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
import pandas as pd

from bond_rp.robustness import family_forecasts, online_combination, technical_features
from bond_rp.strategy import allocation_weights
from bond_rp.vintages import VINTAGE_URLS, VintageArchive, transform_fred_csv
from test_research_protocol import synthetic_bonds


class SpecificationTests(unittest.TestCase):
    def test_family_grid_and_all_ma_forecasts_ignore_future(self):
        bonds = synthetic_bonds()
        rng = np.random.default_rng(19)
        macro = pd.DataFrame(rng.normal(size=(len(bonds), 12)), index=bonds.index,
                             columns=[f"m{i}" for i in range(12)])
        sentiment = pd.DataFrame({"SENT": rng.normal(size=len(bonds))}, index=bonds.index)
        sentiment["SENT_SQ"] = sentiment.SENT**2
        kwargs = dict(start="1979-01", end="1979-01", verbose=False)
        first = family_forecasts(bonds, macro, sentiment, **kwargs).predictions
        changed = bonds.copy()
        changed.loc[changed.index+12 >= pd.Period("1979-01", "M"), "sx_bar"] += 500
        changed_macro, changed_sent = macro.copy(), sentiment.copy()
        changed_macro.loc["1978-12":] *= 1000
        changed_sent.loc["1978-12":] *= 1000
        second = family_forecasts(changed, changed_macro, changed_sent, **kwargs).predictions
        np.testing.assert_allclose(first, second)
        self.assertEqual(len(first.columns), 55)

    def test_online_selector_uses_only_matured_errors(self):
        index = pd.period_range("1985-01", "2005-12", freq="M")
        p = pd.DataFrame({"EH": 1., "A": 2.}, index=index)
        y = pd.Series(2., index=index)
        first, audit = online_combination(p, y, ["EH", "A"])
        changed = y.copy()
        changed.loc[changed.index+12 >= pd.Period("2000-01", "M")] = -999
        second, _ = online_combination(p, changed, ["EH", "A"])
        np.testing.assert_allclose(first.loc[:"2000-01"], second.loc[:"2000-01"])
        check = audit.loc[audit.last_label_end != ""]
        self.assertTrue((pd.PeriodIndex(check.last_label_end, freq="M") <
                         pd.PeriodIndex(check.forecast_month, freq="M")).all())
        self.assertEqual(first.loc["1985-01", "Select_mature"], 1.)

    def test_technical_lookbacks_do_not_read_later_prices(self):
        bonds = synthetic_bonds()
        before = technical_features(bonds)
        bonds.loc["1979-02":, [f"lp{n}" for n in range(2, 6)]] += 999
        after = technical_features(bonds)
        np.testing.assert_allclose(before.loc[:"1979-01"], after.loc[:"1979-01"], equal_nan=True)

    def test_long_only_and_vol_target_respect_cash_budget(self):
        history = np.tile(np.linspace(-.04, .04, 150)[:, None], (1, 4))
        for method in ["long_only", "vol_target"]:
            for mu in [-.1, 0, .01, .2]:
                q = allocation_weights(mu, history, method=method, cap=1)
                self.assertGreaterEqual(q.min(), 0)
                self.assertLessEqual(q.sum(), 1)
                if mu <= 0:
                    self.assertEqual(q.sum(), 0)
        q = allocation_weights(.01, history, method="vol_target", cap=1, target_vol=.02)
        self.assertAlmostEqual(q.sum(), .02/history.mean(axis=1).std(ddof=1))


class VintageTests(unittest.TestCase):
    def test_own_transformation_codes_and_missing_month_calendar(self):
        source = io.StringIO("sasdate,X,Y\nTransform:,2,1\n1/1/1999,10,3\n3/1/1999,40,7\n4/1/1999,60,9\n")
        panel = transform_fred_csv(source)
        self.assertTrue(np.isnan(panel.loc["1999-03", "X"]))
        self.assertEqual(panel.loc["1999-04", "X"], 20)
        self.assertTrue(np.isnan(panel.loc["1999-02", "Y"]))

    def test_future_vintage_cannot_be_used_as_missing_past_vintage(self):
        with tempfile.TemporaryDirectory() as folder:
            for name in VINTAGE_URLS:
                with zipfile.ZipFile(Path(folder)/name, "w") as archive:
                    if "1999" in name:
                        archive.writestr("2000-01.csv", "sasdate,X\nTransform:,1\n12/1/1999,10\n")
                    else:
                        archive.writestr("FRED-MD_2024m03.csv", "sasdate,X\nTransform:,1\n2/1/2024,40\n")
                        archive.writestr("FRED-MD-2025m04.csv", "sasdate,X\nTransform:,1\n3/1/2025,50\n")
            with VintageArchive(folder) as archive:
                date, panel = archive.at("2000-02")
                self.assertEqual(str(date), "2000-01")
                self.assertEqual(panel.X.iloc[0], 10)
                self.assertEqual(archive.at("2024-04")[1].X.iloc[0], 40)
                self.assertEqual(archive.at("2025-05")[1].X.iloc[0], 50)
                with self.assertRaises(KeyError):
                    archive.at("2000-01")


if __name__ == "__main__":
    unittest.main()
