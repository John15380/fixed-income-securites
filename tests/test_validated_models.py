"""Original-data golden checks and independent timing/transform checks."""
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import statsmodels.api as sm
from bond_rp.coursework import common_index
from bond_rp.replication import author_ln_data, run_replication
from bond_rp.validated_models import calibration, gaussian_augment, inflation_trend


class LiteratureTests(unittest.TestCase):
    @unittest.skipUnless(Path('GroupProjectData/LN_handbook_replication.zip').exists(), 'local author data excluded from Git')
    def test_published_factor_tables_and_original_matlab_pca(self):
        table, checks, _, _ = run_replication()
        self.assertTrue(checks.passed.all())
        self.assertGreaterEqual(len(checks), 39)
        d = author_ln_data()
        self.assertEqual(d['rx2'].last_valid_index(), pd.Period('2002-12', 'M'))
        self.assertEqual(d.loc['1964-01', 'rx2'], d.loc['1965-01', 'yr2'])

    def test_cp_restricted_second_stage_matches_independent_ols(self):
        rng = np.random.default_rng(18)
        X = rng.normal(size=(180, 5)); y = rng.normal(size=(180, 4))+X[:, [0]]+3
        now = rng.normal(size=(3, 5))
        pred, first, second = common_index(X, y, now, second_intercept=False)
        z = sm.OLS(y.mean(axis=1), sm.add_constant(X)).fit()
        expected = np.column_stack([sm.OLS(y[:, j], z.fittedvalues).fit().predict(z.predict(sm.add_constant(now))) for j in range(4)])
        np.testing.assert_allclose(pred, expected, atol=1e-10)
        np.testing.assert_allclose(second.intercept_, 0)

    @unittest.skipUnless(Path('GroupProjectData/LN_handbook_replication.zip').exists(), 'local author data excluded from Git')
    def test_actual_production_pca_matches_author_matlab_factors(self):
        import io
        import zipfile
        from scipy.io import loadmat
        from bond_rp.data import apply_transform
        from bond_rp.models import _numeric_pipeline
        with zipfile.ZipFile('GroupProjectData/LN_handbook_replication.zip') as z:
            raw = loadmat(io.BytesIO(z.read('Data&ReplicationFiles/lndata.mat')))
            released = loadmat(io.BytesIO(z.read('Data&ReplicationFiles/Fhat64.mat')))['Fhat_T']
        values = raw['macrodat'][12:]
        panel = np.column_stack([apply_transform(pd.Series(values[:, j]), int(code)).to_numpy()[48:]
                                 for j, code in enumerate(raw['vartype'].ravel()) if np.isfinite(values[:, j]).all()])
        scores = _numeric_pipeline(pca=True).fit_transform(panel)
        signs = np.sign(np.sum(scores*released, axis=0))
        np.testing.assert_allclose(scores*signs, released, atol=1e-8, rtol=0)

    def test_cp15_trend_uses_calendar_lag_and_literal_finite_sum(self):
        ix = pd.period_range('1980-01', periods=180, freq='M')
        levels = pd.Series(np.exp(np.arange(180)*.003), index=ix)
        trend = inflation_trend(levels)
        self.assertAlmostEqual(trend.loc['1994-12'], 3.6*(1-.987**120), places=10)
        changed = levels.copy(); changed.loc['1994-11':] *= 100
        self.assertEqual(trend.loc['1994-12'], inflation_trend(changed).loc['1994-12'])
        missing = inflation_trend(levels.drop(ix[100]))
        self.assertTrue(np.isnan(missing.loc['1994-12']))

    def test_copula_is_reproducible_bounded_and_keeps_original_rows(self):
        rng = np.random.default_rng(24)
        X = rng.normal(size=(140, 5)); y = X[:, [0]]+rng.normal(size=(140, 4))
        a, b = gaussian_augment(X, y)
        c, d = gaussian_augment(X, y)
        np.testing.assert_array_equal(a, c); np.testing.assert_array_equal(b, d)
        np.testing.assert_array_equal(a[:140], X); np.testing.assert_array_equal(b[:140], y)
        self.assertEqual(a.shape, (280, 5))
        self.assertTrue((b.min(axis=0) >= y.min(axis=0)).all())
        self.assertTrue((b.max(axis=0) <= y.max(axis=0)).all())

    def test_calibration_matches_bounded_validation_least_squares(self):
        rng = np.random.default_rng(99)
        b = rng.normal(size=(60, 4)); p = b+rng.normal(size=(60, 4)); y = b+.3*(p-b)
        self.assertAlmostEqual(calibration(y, p, b), .3)
        self.assertEqual(calibration(b-(p-b), p, b), 0)
        self.assertEqual(calibration(b+3*(p-b), p, b), 1)

    @unittest.skipUnless(Path('GroupProjectData/GSW_yields.csv').exists(), 'local market inputs excluded from Git')
    def test_new_factors_cannot_consume_immature_returns_or_unpublished_macro(self):
        from bond_rp.validated_models import load_long_market, make_features, prepare_fold
        from bond_rp.data import load_macro_raw_transformed
        b = load_long_market(); m = load_macro_raw_transformed('fredmd')
        raw = pd.read_csv('GroupProjectData/CPILFESL.csv')
        cpi = pd.Series(raw.CPILFESL.to_numpy(), index=pd.PeriodIndex(pd.to_datetime(raw.iloc[:, 0]), freq='M'))
        date = pd.Period('2017-01', 'M')
        for h in (1, 12):
            before = prepare_fold(b, make_features(b, m, cpi), date, h)
            changed, other, levels = b.copy(), m.copy(), cpi.copy()
            outcomes = [c for c in b if c.startswith(('rx', 'mx', 'long_rx', 'long_mx'))]
            changed.loc[changed.index+h >= date, outcomes] += 1000
            other.loc[other.index > date-2] += 1000
            levels.loc[levels.index > date-2] *= 100
            after = prepare_fold(changed, make_features(changed, other, levels), date, h)
            for key in ['X', 'x', 'y']:
                np.testing.assert_allclose(before[key], after[key], atol=1e-10)
            for key in before['reference']:
                np.testing.assert_allclose(before['reference'][key], after['reference'][key], atol=1e-10)
            self.assertLess(pd.Period(before['audit']['last_label_end'], 'M'), date)


if __name__ == '__main__':
    unittest.main()
