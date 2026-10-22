"""Date-known feature alternatives for the expanded tuning experiment.

The original eleven-index specification and its literature reference forecasts
are delegated unchanged to ``validated_models.prepare_fold``. Primitive inputs
retain information that supervised common-index compression can otherwise hide
from a nonlinear learner. All additional bounds and scales use matured rows.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .models import _numeric_pipeline
from .validated_models import cp15_cycles, prepare_fold


INDEX_LABELS = ["CP05", "LN6", "CP15", "CP_EMA3", "CP_EMA6", "CP_EMA12"] + [
    f"macro_pc{n}" for n in range(1, 6)
]
PRIMITIVE_LABELS = (
    [f"f{n}" for n in range(1, 6)]
    + [f"macro_pc{n}" for n in range(1, 9)]
    + ["macro_pc1_cubed", "cp15_cycle1", "cp15_cycle_mean2_20"]
    + [f"ema{half}_minus_f{n}" for half in (3, 6, 12) for n in range(1, 6)]
    + ["tau"]
)


def prepare_tuning_fold(bonds, features, date, horizon, window="expanding"):
    """Build unchanged index11 and two 32-column alternatives at one date.

    ``window`` is ``'expanding'``, 120 or 240 (numeric strings are accepted).
    With a complete calendar, a finite window selects exactly that many labels
    satisfying ``purchase_month + horizon < date``. The 1994-01 lower bound and
    the original function's minimum of 120 usable training observations apply.

    ``X`` and ``x`` map representation names to standardized numeric arrays;
    ``y`` and ``reference`` retain the original function's output. The primitive
    cube alone clips current F1 at matured-training extrema before cubing; the
    corresponding linear F1 remains uncapped. ``bounded32`` additionally clips
    every primitive column to training 1%/99% quantiles before standardization.
    The audit includes raw bounds and fitted scales as JSON-compatible values.
    """
    date = pd.Period(date, "M")
    if horizon not in (1, 12):
        raise ValueError("Tuning horizons must be 1 or 12 months")
    if window == "expanding":
        normalized_window = "expanding"
        start = pd.Period("1994-01", "M")
    elif str(window) in {"120", "240"}:
        normalized_window = int(window)
        start = max(pd.Period("1994-01", "M"), date-horizon-normalized_window)
    else:
        raise ValueError("window must be 'expanding', 120, or 240")

    original = prepare_fold(bonds, features, date, horizon, train_start=str(start))
    macro_columns = [c for c in features if c.startswith("macro::")]
    history = features.loc[str(start):str(date)]
    history = history[history.tau.notna() & history[macro_columns].notna().any(axis=1)]
    train = history.loc[history.index+horizon < date]
    prefix = "rx" if horizon == 12 else "mx"
    targets = [f"{prefix}{n}" for n in range(2, 6)]
    long_targets = [f"long_{prefix}{n}" for n in range(2, 21)]
    train = train.loc[bonds.loc[train.index, targets+long_targets].notna().all(axis=1)]
    now = features.loc[[date]]
    # Use the identical known-history PCA convention as the original baseline;
    # fitting it again avoids changing or depending on its private return data.
    pca = _numeric_pipeline(pca=True).fit(history[macro_columns])
    Z, z = pca.transform(train[macro_columns]), pca.transform(now[macro_columns])
    cube_min, cube_max = float(Z[:, 0].min()), float(Z[:, 0].max())
    cycle, cycle_now = cp15_cycles(train, now, history)
    forwards = [f"f{n}" for n in range(1, 6)]
    differences = lambda frame: np.column_stack([
        frame[f"ema{half}_f{n}"].to_numpy()-frame[f"f{n}"].to_numpy()
        for half in (3, 6, 12) for n in range(1, 6)
    ])
    raw = np.column_stack([train[forwards], Z, Z[:, [0]]**3,
                           cycle, differences(train), train[["tau"]]])
    current = np.column_stack([now[forwards], z,
                               np.clip(z[:, [0]], cube_min, cube_max)**3,
                               cycle_now, differences(now), now[["tau"]]])
    if raw.shape != (len(original["y"]), 32):
        raise ValueError("Primitive feature rows do not match the baseline's matured labels")
    if not np.isfinite(raw).all() or not np.isfinite(current).all():
        raise ValueError("Tuning primitives require finite date-known inputs")
    np.testing.assert_allclose(original["y"], bonds.loc[train.index, targets].to_numpy())

    lower, upper = np.quantile(raw, [.01, .99], axis=0)
    X, x = {"index11": original["X"]}, {"index11": original["x"]}
    scaling = {}
    for name, values, point in [
        ("primitive32", raw, current),
        ("bounded32", np.clip(raw, lower, upper), np.clip(current, lower, upper)),
    ]:
        scaler = StandardScaler().fit(values)
        X[name], x[name] = scaler.transform(values), scaler.transform(point)
        scaling[name] = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
    audit = dict(original["audit"])
    audit.update(
        window=normalized_window,
        requested_train_start=str(start),
        feature_labels={"index11": list(INDEX_LABELS), "primitive32": list(PRIMITIVE_LABELS),
                        "bounded32": list(PRIMITIVE_LABELS)},
        feature_counts={"index11": 11, "primitive32": 32, "bounded32": 32},
        cube_f1_bounds={"lower": cube_min, "upper": cube_max},
        bounded32_bounds={"quantiles": [.01, .99], "lower": lower.tolist(), "upper": upper.tolist()},
        feature_scaling=scaling,
        last_training_purchase=str(train.index.max()),
    )
    return {"X": X, "x": x, "y": original["y"], "reference": original["reference"],
            "dates": original["dates"], "audit": audit}
