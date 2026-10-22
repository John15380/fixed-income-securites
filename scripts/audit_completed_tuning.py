"""Check actual saved forecasts, selection dates, versions and cash ledgers."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from complete_tuning import (MODELS, PROTOCOL, TAGS, fingerprint, hashes, load_inputs, versions)
from bond_rp.replication import run_replication
from bond_rp.tuning_evaluation import risk_inputs


def audit(output, *, check_papers=True):
    output = Path(output)
    if check_papers:
        _, checks, _, _ = run_replication()
        assert len(checks) == 41
    metadata = json.loads((output/"metadata.json").read_text())
    assert metadata["sources"] == hashes(), "Code or raw data changed since evaluation"
    assert metadata["versions"] == versions(), "Dependency versions changed"
    assert metadata["protocol"] == PROTOCOL
    bonds, _, _ = load_inputs()
    counters = dict(paper_checks=41, panels=0, optimization_trials=0,
                    forecast_rows=0, fold_audits=0, economic_trades=0,
                    calibration_candidates=0)
    numerical_files = {}
    for tag in TAGS:
        for horizon in (1, 12):
            folder = output/tag/f"h{horizon}"
            frozen = json.loads((folder/"frozen_parameters.json").read_text())
            inherited = None
            if tag == "headline_revised":
                inherited = json.loads((output/"headline_vintage"/f"h{horizon}"/"frozen_parameters.json").read_text())
                for name in MODELS:
                    assert frozen[name] == inherited[name]
                assert frozen["mse_shrinkage"] == inherited["mse_shrinkage"]
            assert (folder/"cache_key.txt").read_text() == fingerprint(tag, horizon, inherited), "Stale forecast cache"
            fitmeta = json.loads((folder/"fit_metadata.json").read_text())
            assert fitmeta["sources"] == hashes(fitting_only=True)
            assert fitmeta["versions"] == versions()
            dating = frozen["dating"]
            assert pd.Period(dating["model_validation_last_label_end"], "M") < pd.Period("2014-01", "M")
            assert pd.Period(dating["allocation_validation_last_label_end"], "M") < pd.Period("2017-01", "M")
            assert frozen["selected_family"] == min(MODELS, key=lambda name: frozen[name]["selection"]["objective"])
            if inherited is None:
                trials = pd.read_csv(folder/"trials.csv")
                assert len(trials) == 192 and trials.state.eq("COMPLETE").all()
                assert trials.groupby("model").size().to_dict() == dict(EN_GC=32, EN_real=32, XGB_GC=64, XGB_real=64)
                assert set(trials.seed) == {42, 73}
                for name in MODELS:
                    assert np.isclose(trials.loc[trials.model.eq(name), "value"].min(), frozen[name]["selection"]["objective"])
                counters["optimization_trials"] += len(trials)
            p = pd.read_csv(folder/"predictions.csv")
            assert not p.duplicated(["ym", "model"]).any()
            assert np.isfinite(p[[f"mu{n}" for n in range(2, 6)]]).all().all()
            end = "2025-11" if horizon == 1 else "2024-12"
            dates = pd.period_range("2014-01", end, freq="M")
            for name, group in p.groupby("model"):
                assert pd.PeriodIndex(group.ym, freq="M").equals(dates), (tag, horizon, name)
            groups = {name: group.set_index("ym")[[f"mu{n}" for n in range(2, 6)]].to_numpy()
                      for name, group in p.groupby("model")}
            np.testing.assert_allclose(groups["ModelSelected"], groups[frozen["selected_family"]], atol=1e-12)
            for name in (*MODELS, "ModelSelected"):
                shrink = frozen["mse_shrinkage"][name]
                assert 0 <= shrink <= 1
                np.testing.assert_allclose(groups[name+"_mse"], groups["EH"]+shrink*(groups[name]-groups["EH"]), atol=1e-10)
            a = pd.read_csv(folder/"forecast_audit.csv")
            decision = pd.PeriodIndex(a.ym, freq="M")
            assert (pd.PeriodIndex(a.last_label_end, freq="M") < decision).all()
            assert (pd.PeriodIndex(a.macro_reference, freq="M") == decision-2).all()
            assert (pd.PeriodIndex(a.history_end, freq="M") <= decision).all()
            assert a.n_train.ge(120).all()
            if tag == "headline_vintage":
                assert (pd.PeriodIndex(a.vintage, freq="M") == decision-1).all()
            for window in ("120", "240"):
                assert a.loc[a.window.astype(str).eq(window), "n_train"].le(int(window)).all()
            counters["forecast_rows"] += len(p)
            counters["fold_audits"] += len(a)
            counters["panels"] += 1
            if horizon == 1:
                allocation = json.loads((folder/"allocation_parameters.json").read_text())
                assert allocation["last_validation_outcome"] == "2016-12"
                assert allocation["first_test_purchase"] == "2017-01"
                if inherited is not None:
                    original = json.loads((output/"headline_vintage"/"h1"/"allocation_parameters.json").read_text())
                    assert allocation["parameters"] == original["parameters"]
                calibration = pd.read_csv(folder/"fullcalibration.csv")
                assert calibration.groupby("model").size().eq(81).all()
                assert calibration.groupby("model").selected.sum().eq(1).all()
                assert calibration.months_first.eq(18).all() and calibration.months_second.eq(17).all()
                counters["calibration_candidates"] += len(calibration)
                t = pd.read_csv(folder/"economic_trades.csv")
                for name, g in t.groupby("model"):
                    d = pd.PeriodIndex(g.purchase, freq="M")
                    assert d.equals(pd.period_range("2017-01", "2025-11", freq="M"))
                    assert (pd.PeriodIndex(g.sale, freq="M") == d+1).all()
                    asset, rf, variance, cut = risk_inputs(bonds, d)
                    q = g[[f"q{n}" for n in range(2, 6)]].to_numpy()
                    exposure = q.sum(axis=1)
                    assert (q >= -1e-14).all(), "Hidden short exposure in a long-only account"
                    np.testing.assert_allclose(q, np.repeat((exposure/4)[:, None], 4, axis=1), atol=1e-14)
                    c = .5/10000
                    expected_cash = 1-exposure-c*exposure
                    sale = (q*(1+asset.to_numpy())).sum(axis=1)
                    expected = expected_cash*(1+rf.to_numpy())+sale*(1-c)-1
                    np.testing.assert_allclose(g.return_net, expected, atol=1e-12)
                    np.testing.assert_allclose(g.cash_after_entry, expected_cash, atol=1e-12)
                    np.testing.assert_allclose(g.risk_variance, variance, atol=1e-14)
                    assert g.risk_last_label_end.tolist() == cut.tolist()
                    assert (expected_cash >= -1e-14).all() and (exposure <= 1+1e-14).all()
                counters["economic_trades"] += len(t)
                prior = allocation.get("prior_ew3_source")
                if prior:
                    numerical_files[prior] = hashlib.sha256(Path(prior).read_bytes()).hexdigest()
    assert counters["optimization_trials"] == 768
    result = dict(checks="passed", **counters, prior_prediction_sha256=numerical_files)
    (output/"audit.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT/"output"/"completed_tuning")
    audit(parser.parse_args().output_dir)
