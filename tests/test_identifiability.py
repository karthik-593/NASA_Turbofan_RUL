"""Stage A4: early identifiability of the subpopulation (D51)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbofan.analysis.identifiability import early_features, identifiability
from turbofan.config import KEEP
from turbofan.evaluation.cv import make_folds
from turbofan.features.engineering import add_features


def _engines(
    n_units: int = 40, separable: bool = True, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    """Two groups of engines; if ``separable`` they differ in s9's early level."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        grp = u % 2
        for c in range(1, 121):
            row: dict[str, float] = {"unit": u, "cycle": c}
            for i in range(1, 4):
                row[f"op{i}"] = float(rng.uniform(0, 1))
            for i in range(1, 22):
                row[f"s{i}"] = float(rng.normal(0, 1) + 0.01 * c)
            if separable:
                row["s9"] += 2.0 * grp
            rows.append(row)
    labels = pd.Series([u % 2 for u in range(1, n_units + 1)], index=range(1, n_units + 1))
    return pd.DataFrame(rows), labels


def test_early_features_use_only_the_first_n_cycles() -> None:
    train, _ = _engines(n_units=4)
    _, stats = add_features(train, KEEP, "FD001")
    x30 = early_features(train, "FD001", 30, stats)
    poisoned = train.copy()
    poisoned.loc[poisoned["cycle"] > 30, KEEP] = 1e6  # the future must not matter
    pd.testing.assert_frame_equal(x30, early_features(poisoned, "FD001", 30, stats))
    assert list(x30.columns) == [f"{s}_{k}" for s in KEEP for k in ("mean", "slope")]


def test_too_short_engines_raise() -> None:
    train, _ = _engines(n_units=2)
    _, stats = add_features(train, KEEP, "FD001")
    with pytest.raises(ValueError, match="fewer than 200"):
        early_features(train, "FD001", 200, stats)


@pytest.mark.parametrize(("separable", "lo", "hi"), [(True, 0.9, 1.0), (False, 0.0, 0.8)])
def test_auc_tracks_separability(separable: bool, lo: float, hi: float) -> None:
    train, labels = _engines(separable=separable)
    folds = make_folds(labels, n_folds=4, n_repeats=2, seed=0)
    t = identifiability(
        train, labels, "FD001", folds, [30], 1.0, np.random.default_rng(0), 200, 0.95
    )
    r = t.iloc[0]
    assert lo <= r["auc"] <= hi and r["ci_lo"] <= r["auc"] <= r["ci_hi"]
    assert r["n_engines"] == 40 and len(r["auc_per_repeat"]) == 2
