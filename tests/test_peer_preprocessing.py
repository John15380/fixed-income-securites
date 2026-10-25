"""Shock handling and timing checks for the source-grounded cleaning branches."""
import unittest
import numpy as np
import pandas as pd
from bond_rp.peer_preprocessing import clean_panel, rebuild_macro_fold, _evaluation_inputs


class PeerPreprocessingTests(unittest.TestCase):
    def test_training_threshold_and_previous_value_never_use_future_observations(self):
        dates = pd.period_range("2000-01", periods=12, freq="M")
        panel = pd.DataFrame({"x":[0.,1.,2.,3.,4.,5.,1000.,2000.,7.,8.,9.,10.],
                              "constant":[1.]*12}, index=dates)
        clean, audit = clean_panel(panel, dates[:6], "fred10_previous")
        np.testing.assert_array_equal(clean.x.iloc[6:9], [5.,5.,7.])
        self.assertEqual(audit["flagged_train"], 0)
        self.assertEqual(audit["flagged_total"], 2)
        changed = panel.copy(); changed.iloc[9:] = 1e9
        after, _ = clean_panel(changed, dates[:6], "fred10_previous")
        pd.testing.assert_frame_equal(clean.iloc[:9], after.iloc[:9])
        missing, _ = clean_panel(panel, dates[:6], "fred10_mean")
        self.assertTrue(missing.x.iloc[6:8].isna().all())

    def test_rebuilt_factors_ignore_future_and_keep_original_references_and_labels(self):
        from test_tuning_features import synthetic_inputs
        from bond_rp.validated_models import make_features
        from bond_rp.peer_reference import peer_fold
        bonds, macro, cpi = synthetic_inputs()
        features = make_features(bonds, macro, cpi)
        rng = np.random.default_rng(112); date = pd.Period("2017-01", "M")
        survey = pd.DataFrame(rng.normal(size=(len(bonds),24)), index=bonds.index)
        survey["release_date"] = (survey.index-1).to_timestamp()
        candidates = pd.DataFrame(rng.normal(size=(len(bonds),8)), index=bonds.index)
        fold = peer_fold(bonds,features,survey,candidates,list(candidates),date,
                         predict_dates=[date,date+1],include_lags=True)
        evaluated = _evaluation_inputs(fold)
        np.testing.assert_allclose(evaluated.loc[date],fold["x"][0],atol=1e-12)
        # A large, currently available shock must change the raw LN forecast;
        # both declared cleaning rules should replace it before cubing F1.
        mc = [c for c in features if c.startswith("macro::")]
        features.loc[date,mc] = 1e8
        candidates.loc[date] = 1e8
        for method in ("fred10_mean", "fred10_previous"):
            before = rebuild_macro_fold(fold,features,candidates,list(candidates),method)
            future = features.copy(); future.loc[future.index>date+1,mc] = 1e12
            future_c = candidates.copy(); future_c.loc[future_c.index>date+1] = 1e12
            after = rebuild_macro_fold(fold,future,future_c,list(candidates),method)
            for key in ("X","x","y"):
                np.testing.assert_array_equal(before[key],after[key])
            for key in ("X","x","y"):
                np.testing.assert_array_equal(before["lagged"][key],after["lagged"][key])
            for name in ("EH","L1_CP05","L1_MS17","L1_CP15","L1_FB"):
                np.testing.assert_array_equal(before["reference"][name],fold["reference"][name])
            self.assertLess(before["audit"]["threshold_fit_last"],str(date-12))
            self.assertLess(np.abs(before["x"]).max(),100.)
            # t+1's lag is t in the same newly fitted factor basis.
            train = pd.DataFrame(before["X"],index=before["train_dates"])
            previous = train.reindex(before["lagged"]["train_dates"]-1).to_numpy()
            expected = (before["x"][0]-previous.mean(0))/previous.std(0)
            np.testing.assert_allclose(before["lagged"]["x"][1,15:],expected,atol=1e-10)
        np.testing.assert_array_equal(fold["y"],before["y"])


if __name__ == "__main__":
    unittest.main()
