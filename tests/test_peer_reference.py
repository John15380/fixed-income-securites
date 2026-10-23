"""Independent equations, survey construction, embargo and cash-flow checks."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from bond_rp.ms17 import MS17Regression, recursive_factor
from bond_rp.peer_allocation import allocation_path, diagonal_allocation, risk_panel
from bond_rp.survey_expectations import respondent_expectations, release_dates


class PeerReferenceTests(unittest.TestCase):
    def test_ms17_equation_and_identifiable_synthetic_parameters(self):
        rng = np.random.default_rng(62)
        f = rng.normal(size=(240,5))
        gamma = np.array([1., .1, -.3, .5, .7, -.2])
        r, previous = [], 2.5
        for row in f:
            previous = .4*(gamma[0]+row@gamma[1:])+.6*previous
            r.append(previous)
        np.testing.assert_allclose(recursive_factor(f,gamma,.6,2.5), r, atol=1e-14)
        y = np.array(r)[:,None]*np.array([.4,.8,1.2,1.6])
        fit = MS17Regression().fit(f,y,initial=2.5)
        self.assertTrue(fit.converged_)
        self.assertAlmostEqual(fit.alpha_,.6,places=5)
        self.assertAlmostEqual(fit.loadings_.mean(),1.,places=12)
        np.testing.assert_allclose(fit.predict(f),y,atol=1e-6)

    def test_recursive_factor_keeps_unlabelled_calendar_gap(self):
        rng = np.random.default_rng(15)
        f=rng.normal(size=(180,5)); y=rng.normal(size=(140,4))
        a=MS17Regression().fit(f,y)
        modified=f.copy(); modified[170:]+=1000
        b=MS17Regression().fit(modified,y)
        np.testing.assert_allclose(a.gamma_,b.gamma_,atol=1e-8)
        np.testing.assert_allclose(a.factor(f)[:170],b.factor(modified)[:170],atol=1e-8)
        self.assertNotEqual(a.factor(f)[-1],b.factor(modified)[-1])

    def test_spf_median_of_individual_growth_and_missing_housing(self):
        f=pd.DataFrame(dict(YEAR=[2011]*3,QUARTER=[1]*3,RGDP2=[1,10,100]))
        for h in range(3,7): f[f"RGDP{h}"]=[2,11,130]
        result=respondent_expectations(f,"RGDP")
        self.assertAlmostEqual(result.iloc[0,0],100*np.log(1.3))
        self.assertNotAlmostEqual(result.iloc[0,0],100*np.log(11/10))
        h=f.rename(columns={c:c.replace("RGDP","HOUSING") for c in f})
        h.loc[0,"HOUSING3"]=-999
        self.assertAlmostEqual(respondent_expectations(h,"HOUSING").iloc[0,0],15.5)

    def test_actual_spf_release_calendar_has_delayed_2019q1(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/"dates.txt"
            p.write_text("2019 Q1 3/12/19** 3/22/19**\n Q2 5/7/19 5/10/19\n")
            dates=release_dates(p)
        self.assertEqual(dates[pd.Period("2019Q1")],pd.Timestamp("2019-03-22"))
        self.assertEqual(dates[pd.Period("2019Q2")],pd.Timestamp("2019-05-10"))

    def test_constrained_per_maturity_solution_satisfies_kkt(self):
        mu=np.array([.1,.02,-.1,.03]); variance=np.array([.01,.02,.04,.03])
        w=diagonal_allocation(mu,variance,budget=.7,concentration=2.)
        self.assertAlmostEqual(w.sum(),.7,places=10)
        self.assertTrue((w>=0).all());self.assertEqual(w[2],0)
        gradient=mu-(5*variance+2*variance.mean())*w
        np.testing.assert_allclose(gradient[w>0],np.repeat(gradient[w>0][0],(w>0).sum()),atol=1e-12)
        self.assertTrue((gradient[w==0]<=gradient[w>0][0]+1e-12).all())

    def test_annual_signal_cash_ledger_and_strict_risk_history(self):
        rng=np.random.default_rng(20)
        index=pd.period_range("1990-01","2024-12",freq="M")
        b=pd.DataFrame(index=index);b["rf_month"]=.1
        for n in range(2,6):
            b[f"rx{n}"]=rng.normal(.5,2,len(index))
            b[f"mr{n}"]=rng.normal(.15,.4,len(index))
        dates=pd.period_range("2017-01","2022-12",freq="M")
        forecast=pd.DataFrame(1.,index=dates,columns=[f"mu{n}" for n in range(2,6)])
        path=allocation_path(b,forecast,cost_bps=2,drawdown=10,concentration=2)
        q=path[[f"q{n}" for n in range(2,6)]].to_numpy()
        cash=1-q.sum(1)*(1+2/10000)
        actual=b.loc[dates,[f"mr{n}" for n in range(2,6)]].to_numpy()/100
        expected=cash*1.001+(q*(1+actual)).sum(1)*(1-2/10000)-1
        np.testing.assert_allclose(path.return_net,expected,atol=1e-14)
        self.assertTrue((cash>=-1e-14).all())
        self.assertTrue((pd.PeriodIndex(path.risk_last_label_end,freq="M")<dates).all())
        _,v,_=risk_panel(b,dates[:1])
        modified=b.copy();modified.loc[modified.index+12>=dates[0],[f"rx{n}" for n in range(2,6)]]+=1000
        np.testing.assert_array_equal(v,risk_panel(modified,dates[:1])[1])

    def test_all_fifteen_features_ignore_immature_labels_and_future_macro(self):
        from test_tuning_features import synthetic_inputs
        from bond_rp.validated_models import make_features
        from bond_rp.peer_reference import peer_fold
        bonds,macro,cpi=synthetic_inputs()
        rng=np.random.default_rng(83); date=pd.Period("2017-01","M")
        survey=pd.DataFrame(rng.normal(size=(len(bonds),24)),index=bonds.index)
        survey["release_date"]=(survey.index-1).to_timestamp()
        candidates=pd.DataFrame(rng.normal(size=(len(bonds),8)),index=bonds.index,columns=[f"c{j}" for j in range(8)])
        before=peer_fold(bonds,make_features(bonds,macro,cpi),survey,candidates,list(candidates),date)
        changed=bonds.copy();targets=[c for c in bonds if c.startswith(("rx","long_rx"))]
        changed.loc[changed.index+12>=date,targets]+=1e6
        m=macro.copy();m.loc[m.index>date-2]+=1e6
        c=cpi.copy();c.loc[c.index>date-2]*=100
        after=peer_fold(changed,make_features(changed,m,c),survey,candidates,list(candidates),date)
        np.testing.assert_array_equal(before["X"],after["X"])
        np.testing.assert_array_equal(before["x"],after["x"])
        np.testing.assert_array_equal(before["y"],after["y"])
        self.assertEqual(before["X"].shape[1],15)


if __name__=="__main__":
    unittest.main()
