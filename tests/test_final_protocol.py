"""Economic identities and adversarial calendar/information-set checks."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import numpy as np
import pandas as pd
import statsmodels.api as sm
from bond_rp.data import bond_variables
from bond_rp.market import monthly_curve_returns, svensson_yields
from bond_rp.forecasting import training_sample, fit_forecasts
from bond_rp.factors import common_index, ln6_terms
from bond_rp.inference import mean_hac_t, block_bootstrap_t
from bond_rp.vintages import VintageArchive, VINTAGE_URLS
from bond_rp.portfolio import simulate, performance, basket_account
from bond_rp.validation import r2_components, choose_policy

class ResearchProtocol(unittest.TestCase):
    def test_cp_matches_independent_restricted_regression(self):
        rng=np.random.default_rng(14)
        X=rng.normal(size=(200,5));y=rng.normal(size=(200,4))+X@rng.normal(size=(5,4))
        p,first,second=common_index(X,y,X,second_intercept=False)
        index=sm.OLS(y.mean(axis=1),sm.add_constant(X)).fit().fittedvalues
        expected=np.column_stack([sm.OLS(y[:,j],index).fit().fittedvalues for j in range(4)])
        np.testing.assert_allclose(p,expected,atol=1e-11)
        np.testing.assert_allclose(p.mean(axis=1),index,atol=1e-11)
        self.assertFalse(second.fit_intercept)

    def test_ln_literal_terms(self):
        z=np.arange(24).reshape(3,8)
        np.testing.assert_array_equal(ln6_terms(z),np.column_stack([z[:,0],z[:,0]**3,z[:,1],z[:,2],z[:,3],z[:,7]]))

    def test_calendar_hole_does_not_shift_annual_sale(self):
        idx=pd.period_range('2000-01','2002-01',freq='M').delete(12)
        lp=pd.DataFrame({f'lp{n}':np.arange(len(idx))*.1-n for n in range(1,6)},index=idx)
        b=bond_variables(lp)
        self.assertTrue(np.isnan(b.loc['2000-01','rx2']))
        expected=lp.loc['2001-02','lp1']-lp.loc['2000-02','lp2']+lp.loc['2000-02','lp1']
        self.assertAlmostEqual(b.loc['2000-02','rx2'],expected)

    def test_strict_maturity_cutoff_at_all_horizons(self):
        idx=pd.period_range('1990-01','2020-12',freq='M')
        frame=pd.DataFrame({'x':np.arange(len(idx))},index=idx)
        for h in (1,3,12):
            t=pd.Period('2020-01','M')
            train=training_sample(frame,t,h,120)
            self.assertEqual(len(train),120)
            self.assertEqual(train.index.max()+h,t-1)
            self.assertFalse((train.index+h>=t).any())

    def test_same_close_alternative_keeps_the_window_length(self):
        dates=pd.period_range('1990-01','2020-12',freq='M')
        frame=pd.DataFrame({'x':np.arange(len(dates))},index=dates)
        t=pd.Period('2020-01','M')
        for h in (1,3,12):
            a=training_sample(frame,t,h,120,embargo=0)
            self.assertEqual(len(a),120)
            self.assertEqual(a.index.max()+h,t)

    def test_training_preprocessing_ignores_future_rows(self):
        rng=np.random.default_rng(7)
        idx=pd.period_range('2000-01',periods=200,freq='M')
        columns=[f'macro::{i}' for i in range(12)]+[f'f{i}' for i in range(1,6)]+[f'fs{i}' for i in range(2,6)]+['trend12']
        f=pd.DataFrame(rng.normal(size=(200,22)),index=idx,columns=columns)
        target=[f'y{n}' for n in range(4)]
        f[target]=rng.normal(size=(200,4))
        t=idx[150]
        before,_=fit_forecasts(training_sample(f,t,12,120),f.loc[[t]],columns[:12],target)
        f.loc[f.index>t,:]=1e12
        after,_=fit_forecasts(training_sample(f,t,12,120),f.loc[[t]],columns[:12],target)
        for name in before:np.testing.assert_allclose(before[name],after[name],atol=1e-12)

    def test_flat_curve_ages_same_bond(self):
        idx=pd.period_range('2000-01',periods=3,freq='M')
        p=pd.DataFrame(np.tile([4.,0,0,0,2.,-999.99],(3,1)),index=idx)
        r=monthly_curve_returns(p)
        for n in range(2,6):
            np.testing.assert_allclose(r[f'mr{n}'].iloc[:2],100*np.expm1(.04/12),atol=1e-12)
        self.assertTrue(r.filter(regex='^mr').iloc[-1].isna().all())

    def test_svensson_against_scalar_formula(self):
        p=np.array([[3.,-1.,2.,.7,1.5,3.2]])
        n=2-1/12
        a,b=n/1.5,n/3.2
        expected=3-(1-np.exp(-a))/a+2*((1-np.exp(-a))/a-np.exp(-a))+.7*((1-np.exp(-b))/b-np.exp(-b))
        self.assertAlmostEqual(svensson_yields(p,[n])[0,0],expected,places=13)

    def test_future_vintage_never_substitutes_for_missing_one(self):
        with TemporaryDirectory() as tmp:
            for name in VINTAGE_URLS:
                with ZipFile(Path(tmp)/name,'w') as z:
                    if '1999' in name:
                        z.writestr('2000-02.csv','sasdate,X\nTransform:,1\n1/1/2000,1\n')
            with VintageArchive(tmp) as archive:
                with self.assertRaises(KeyError):archive.at('2000-02')

    def test_fully_funded_entry_exit_ledger_and_initial_drawdown(self):
        idx=pd.period_range('2000-01',periods=120,freq='M')
        m=pd.DataFrame({'rf_month':.2},index=idx)
        for n in range(2,6):
            m[f'mr{n}']=np.where(np.arange(120)%2,1.,-1.)
            m[f'mx{n}']=m[f'mr{n}']-.2
        dates=idx[70:]
        p=pd.DataFrame([dict(ym=str(d),horizon=1,window='120',model=name,**{f'mu{n}':1. for n in range(2,6)}) for d in dates for name in ('EH','EW3')])
        _,trades=simulate(m,p,start=str(dates[0]),end=str(dates[-1]),cost_bps=5)
        g=trades[trades.model.eq('BuyHold')]
        c=5/10000;q=1/(1+c)
        expected=(1-q-c*q)*1.002+q*(1+m.mr2.reindex(dates).to_numpy()/100)*(1-c)-1
        np.testing.assert_allclose(g.return_net,expected,atol=1e-14)
        self.assertGreaterEqual(g.cash_after_entry.min(),-1e-12)
        self.assertAlmostEqual(performance(np.r_[-.1,np.zeros(11)],np.zeros(12))['max_dd_pct'],-10)

    def test_hac_matches_statsmodels_and_bootstrap_scale(self):
        rng=np.random.default_rng(11)
        x=rng.normal(size=180)+.05
        fit=sm.OLS(x,np.ones((len(x),1))).fit(cov_type='HAC',cov_kwds={'maxlags':6,'use_correction':True})
        self.assertAlmostEqual(float(mean_hac_t(x,6)),fit.tvalues[0],places=12)
        a=block_bootstrap_t(x,lags=6,block=12,replications=199)
        b=block_bootstrap_t(x*100,lags=6,block=12,replications=199)
        self.assertEqual(a['bootstrap_p'],b['bootstrap_p'])

    def test_borrowing_charges_entry_fees_and_funding_spread(self):
        w=np.array([.5,1.,2.,3.])
        r=np.array([.01]*4);rf=np.array([.002]*4)
        account=basket_account(w,r,rf,cost_bps=5,borrow_spread_bps=120)
        q=account['gross_exposure'];cash=1-q-q*.0005
        expected=cash*(1+rf)+q*(1+r)*(1-.0005)-np.maximum(-cash,0)*.001-1
        np.testing.assert_allclose(account['return_net'],expected,atol=1e-14)
        self.assertEqual(account['financing'][0],0)
        self.assertAlmostEqual(account['financing'][2],1.001*.001)

    def test_r2_decomposition_exposes_a_weak_mean_benchmark(self):
        y=np.array([-1.,1.,-1.,1.])
        p=np.zeros(4);b=np.full(4,3.)
        s=r2_components(y,p,b)
        self.assertAlmostEqual(s['r2_centered'],0.)
        self.assertAlmostEqual(s['r2_oos'],.9)
        self.assertAlmostEqual(s['r2_oos'],1-(1-s['r2_centered'])/s['eh_denominator_ratio'])

    def test_validation_selection_cannot_use_future_account_returns(self):
        rng=np.random.default_rng(5)
        dates=pd.period_range('2000-01',periods=200,freq='M')
        returns=rng.normal(0,.01,(200,3))
        cutoff=pd.Period('2010-01','M')
        before,info=choose_policy(returns,dates,cutoff)
        returns[dates+1>=cutoff]=rng.normal(100,1,(np.sum(dates+1>=cutoff),3))
        after,changed=choose_policy(returns,dates,cutoff)
        self.assertEqual(before,after)
        self.assertEqual(info,changed)
        self.assertLess(pd.Period(info['last_validation_sale'],'M'),cutoff)

if __name__=='__main__':
    unittest.main()
