"""Independent replay of saved holdings, cash, fees, calendar cutoffs and metrics."""
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from bond_rp.config import DATA_DIR
from bond_rp.market import load_monthly_market


def main():
    out=ROOT/"output"/"final"
    manifest=json.loads((out/"manifest.json").read_text())
    for name,digest in manifest["inputs_sha256"].items():
        assert hashlib.sha256((DATA_DIR/name).read_bytes()).hexdigest()==digest,name
    for name,digest in manifest["source_sha256"].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    market=load_monthly_market()
    trades=pd.read_csv(out/"monthly_accounts.csv")
    performance=pd.read_csv(out/"performance.csv").set_index("model")
    c=manifest["strategy"]["one_way_cost_bps"]/10000
    max_error=0.
    for name,g in trades.groupby("model"):
        g=g.sort_values("purchase")
        dates=pd.PeriodIndex(g.purchase,freq="M")
        assert (np.diff(dates.asi8)==1).all()
        assert (pd.PeriodIndex(g.sale,freq="M")==dates+1).all()
        assert (pd.PeriodIndex(g.risk_last_label_end,freq="M")<dates).all()
        q=g[[f"q{n}" for n in range(2,6)]].to_numpy()
        rf=market.rf_month.reindex(dates).to_numpy()/100
        actual=market.loc[dates,[f"mr{n}" for n in range(2,6)]].to_numpy()/100
        assert (q>=0).all() and (q.sum(axis=1)<=1+1e-12).all()
        entry_fee=c*q.sum(axis=1)
        cash=1-q.sum(axis=1)-entry_fee
        assert (cash>=-1e-12).all()
        sale_value=(q*(1+actual)).sum(axis=1)
        r=cash*(1+rf)+sale_value-c*sale_value-1
        max_error=max(max_error,np.max(abs(r-g.return_net)))
        np.testing.assert_allclose(r,g.return_net,atol=1e-14,rtol=0)
        np.testing.assert_allclose(cash,g.cash_after_entry,atol=1e-14,rtol=0)
        wealth=np.r_[1,np.cumprod(1+r)]
        row=performance.loc[name]
        np.testing.assert_allclose(100*(wealth[-1]**(12/len(r))-1),row.cagr_pct,atol=1e-11)
        np.testing.assert_allclose(100*np.min(wealth/np.maximum.accumulate(wealth)-1),row.max_dd_pct,atol=1e-11)
    n=0
    for label in ("archival_monthly","archival_horizons","revised"):
        a=pd.read_csv(out/label/"forecast_audit.csv")
        dates=pd.PeriodIndex(a.ym,freq="M")
        assert (pd.PeriodIndex(a.last_label_end,freq="M")<dates).all()
        assert (pd.PeriodIndex(a.latest_macro_reference,freq="M")<=dates-2).all()
        if label.startswith("archival"):
            assert (pd.PeriodIndex(a.vintage,freq="M")==dates-1).all()
        n+=len(a)
    checks=pd.read_csv(out/"paper_checks.csv")
    assert checks.passed.all()
    # The final model family must contain every singleton, all pairwise
    # combinations, and the fixed three-signal ensemble. This makes the stated
    # factor choice an executable ablation rather than a narrative assertion.
    required={"CP","Macro8","Trend12","CP_Macro8","CP_Trend12",
              "Macro8_Trend12","EW3"}
    assert required.issubset(set(performance.index))

    # The 324-cell design grid is retrospective.  Its base cell must replay the
    # published strategy exactly, and the causal schedules may only consume
    # accounts whose sales precede the selection date.
    grid=pd.read_csv(out/"parameter_grid.csv",dtype={"window":str})
    assert grid.config.nunique()==324 and set(grid.period)=={"full","validation","later"}
    base=grid.loc[(grid.config=="cfg000")&(grid.period=="full")].iloc[0]
    assert (base.window,base.ma,base.gamma,base.shrink,base.cap)==("120",12,5.,.5,1.)
    for column in ("cagr_pct","sharpe","vol_pct","max_dd_pct","terminal_wealth"):
        np.testing.assert_allclose(base[column],performance.loc["EW3",column],atol=1e-12,rtol=0)
    schedule=pd.read_csv(out/"selection_schedule.csv",dtype={"window":str})
    selected=pd.PeriodIndex(schedule.selected_at,freq="M")
    sale=pd.PeriodIndex(schedule.last_validation_sale,freq="M")
    purchase=pd.PeriodIndex(schedule.purchase,freq="M")
    assert (sale<selected).all() and (selected<=purchase).all()
    assert (schedule.validation_months>=60).all()
    family=pd.read_csv(out/"parameter_family_tests.csv")
    assert family.family_configs.eq(324).all() and set(family.period)=={"full","later"}
    factor=pd.read_csv(out/"factor_validation.csv")
    assert factor.model.nunique()==9 and set(factor.period)=={"validation","later"}
    assert factor.groupby("period").selected_before_cutoff.sum().eq(1).all()
    assert factor.last_validation_sale.lt(factor.selection_cutoff).all()
    assert factor.validation_months.ge(60).all()
    finance=pd.read_csv(out/"financing_grid.csv")
    assert len(finance)==108 and set(finance.direction)=={"long_only","long_short"}
    assert set(finance.gamma)=={2.,5.,10.} and set(finance.upper_bound)=={1.,2.,3.}
    assert set(finance.borrowing_spread_bps)=={0.,50.,200.}
    assert finance.months.eq(313).all() and finance.start.eq("2000-01").all()
    assert finance.borrowing_months.ge(0).all() and finance.borrowing_months.le(313).all()
    long=finance.loc[finance.direction.eq("long_only")]
    assert long.short_months.eq(0).all() and long.lower_bound.eq(0).all()
    short=finance.loc[finance.direction.eq("long_short")]
    assert short.short_months.gt(0).all() and short.lower_bound.lt(0).all()
    summary=pd.read_csv(out/"financing_summary.csv")
    assert set(summary.label)=={"unlevered_long_only","financed_long_short_cap2",
        "financed_long_short_cap3","cost_grid_best_cagr","cost_grid_best_sharpe",
        "grid_best_cagr","grid_best_sharpe"}
    assert summary.loc[summary.label.eq("grid_best_cagr"),"rank_cagr"].iat[0]==1
    assert summary.loc[summary.label.eq("grid_best_sharpe"),"rank_sharpe"].iat[0]==1
    cost=finance.loc[(finance.borrowing_spread_bps==50)&(
        ((finance.direction=="long_only")&(finance.short_borrow_bps==0))|
        ((finance.direction=="long_short")&(finance.short_borrow_bps==25)))]
    best_cost_cagr=summary.loc[summary.label.eq("cost_grid_best_cagr")].iloc[0]
    best_cost_sharpe=summary.loc[summary.label.eq("cost_grid_best_sharpe")].iloc[0]
    assert np.isclose(best_cost_cagr.cagr_pct,cost.cagr_pct.max())
    assert np.isclose(best_cost_sharpe.sharpe,cost.sharpe.max())

    # Nonlinear learners are tuned before 2007, selected and shrunk before
    # 2013, then evaluated on the common 2013+ account calendar.
    nonlinear_search=pd.read_csv(out/"nonlinear_search.csv")
    assert len(nonlinear_search)==192 and nonlinear_search.family.nunique()==4
    assert nonlinear_search.groupby("family").selected.sum().eq(1).all()
    nonlinear_selection=pd.read_csv(out/"nonlinear_selection.csv")
    assert len(nonlinear_selection)==4 and nonlinear_selection.family_winner.sum()==1
    assert nonlinear_selection.accepted.sum() in (0,1)
    assert nonlinear_selection.shrinkage.between(0,1).all()
    model_performance=pd.read_csv(out/"model_performance.csv")
    assert model_performance.model.nunique()==5
    assert set(model_performance.period)=={"selection_2007_2012","holdout_2013_2026","full_2000_2026"}
    assert model_performance.groupby("model").months.sum().gt(0).all()
    model_scores=pd.read_csv(out/"model_forecast_scores.csv")
    assert len(model_scores)==50 and model_scores.model.nunique()==5
    assert set(model_scores.period)=={"selection_2007_2012","holdout_2013_2026"}
    protocol=json.loads((out/"nonlinear_protocol.json").read_text())
    assert protocol["evaluation"].startswith("2013-01")

    maturity=pd.read_csv(out/"maturity_allocation.csv")
    assert maturity.policy.nunique()==7 and len(maturity)==21
    long=maturity.loc[maturity.policy_group.eq("long_only")]
    assert long.groupby("period").selected_before_2013.sum().eq(1).all()
    assert set(maturity.period)=={"validation_2000_2012","holdout_2013_2026","full_2000_2026"}

    # Historical monthly-vintage coverage is auditable month by month.  File
    # labels and reference dates are separate; the t-1 file's latest populated
    # reference is normally t-2, which is the information used by the strategy.
    macro=pd.read_csv(out/"macro_information.csv")
    purchase=pd.PeriodIndex(macro.purchase,freq="M")
    assert len(macro)==312 and (pd.PeriodIndex(macro.vintage,freq="M")==purchase-1).all()
    assert (pd.PeriodIndex(macro.last_reference,freq="M")<=purchase-2).all()
    assert macro.reference_lag1_available.eq(0).all()
    assert macro.reference_lag2_available.gt(0).all()

    # Every relaxed-information and leverage comparison uses the identical
    # 2000-01..2025-12 calendar.  Its own forecast audit must encode the timing
    # assumptions explicitly rather than silently changing the sample.
    assumptions=pd.read_csv(out/"assumption_performance.csv")
    assert assumptions.months.eq(312).all()
    assert assumptions.start.eq("2000-01").all() and assumptions.last_purchase.eq("2025-12").all()
    timing={"archival_lag3":(1,3,1),"current_vintage_lag2":(0,2,1),
            "current_vintage_lag1":(0,1,1),"same_close_labels":(1,2,0),
            "same_close_all":(0,1,0)}
    for name,(delay,lag,embargo) in timing.items():
        audit=pd.read_csv(out/"validation"/name/"forecast_audit.csv")
        dates=pd.PeriodIndex(audit.ym,freq="M")
        assert audit.macro_lag.eq(lag).all() and audit.embargo.eq(embargo).all()
        assert (pd.PeriodIndex(audit.vintage,freq="M")==dates-delay).all()
        assert (pd.PeriodIndex(audit.latest_macro_reference,freq="M")==dates-lag).all()
        label_end=pd.PeriodIndex(audit.last_label_end,freq="M")
        assert (label_end<dates).all() if embargo else (label_end<=dates).all()

    # OOS R2, same-sample centered R2, and the historical-mean denominator are
    # linked by an exact identity.  The post-hoc calibration column is diagnostic
    # only and cannot enter trading decisions.
    accounting=pd.read_csv(out/"r2_accounting.csv")
    identity=1-(1-accounting.r2_centered)/accounting.eh_denominator_ratio
    np.testing.assert_allclose(accounting.r2_oos,identity,atol=1e-12,rtol=0)
    assert (accounting.calibrated_signal_r2+1e-12>=accounting.r2_centered).all()

    # Published-cell coverage includes known, visible data-vintage and printed-
    # precision misses.  Only the separate algorithmic checks are all-pass.
    cells=pd.read_csv(out/"published_cells.csv")
    assert set(cells.paper)=={"FB87","CP05","LN09"} and len(cells)>=200
    assert cells.matches_printing.any() and (~cells.matches_printing).any()
    mismatch=cells.loc[~cells.matches_printing]
    assert (mismatch.difference.abs()>mismatch.printing_tolerance).all()
    print(f"Verified {len(trades)} monthly account rows; max independent replay error {max_error:.3g}")
    print(f"Verified {n} forecast training/vintage audit rows and {len(checks)} paper checks")
    print(f"Verified 324 parameter cells, {len(schedule)} causal selections, 9 factor candidates, 108 financing policies and {len(accounting)} R2 identities")
    print(f"Published-table cell coverage: {int(cells.matches_printing.sum())}/{len(cells)} match printed precision")
    print("Input and source SHA-256 fingerprints match the completed run")


if __name__=="__main__":
    main()
