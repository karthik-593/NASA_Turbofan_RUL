"""Single k-regime feature path vs the former two-path ``add_features`` (tests/legacy_features).

params.yaml replaced "global z-score for FD001/FD003, per-regime for FD002/FD004" with one
code path: KMeans with k = n_regimes, then a per-regime z-score. For k = 1 that must be the
old global z-score bit for bit, on train, validation and test frames alike. For k = 6 the
per-regime mean/std are now computed per regime subset with Series.mean/std instead of a
pandas groupby, which rounds differently in the last bits. Dividing by a regime's small std
amplifies that: measured 2026-10-03, max |diff| 1.75e-11 (FD002) and 1.03e-11 (FD004) over
train/val/test — far below any modelling-relevant amount.

The rolling slope is computed in closed form since D52 (the legacy path calls ``np.polyfit``
per window): equal to rounding, max |diff| 1.8e-15 on the four training files (2026-10-04), so
the slope columns are compared to ``SLOPE_ATOL`` and every other column stays bit-identical.

Requires data/raw to be present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.legacy_features import legacy_add_features
from turbofan.config import KEEP, feat_cols
from turbofan.data.loader import load_dataset
from turbofan.evaluation.comparison import split_engines
from turbofan.features.engineering import add_features

pytestmark = pytest.mark.requires_data

RAW = "data/raw"
MULTI_REGIME_ATOL = 1e-10  # z-scores are O(1)-O(10); max observed 1.75e-11
SLOPE_ATOL = 1e-12  # closed-form vs polyfit slope (D52); max observed 1.8e-15
SLOPES = [f"{s}_slope" for s in KEEP]
EXACT = [c for c in feat_cols(KEEP) if c not in SLOPES]


def _frames(dataset: str) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    tr, te, _ = load_dataset(dataset, RAW)
    e_tr, e_va = split_engines(tr)
    train = tr[tr["unit"].isin(e_tr)]
    new_tr, st_new = add_features(train, KEEP, dataset)
    old_tr, st_old = legacy_add_features(train, KEEP, dataset)
    out = {"train": (new_tr, old_tr)}
    for name, frame in (("val", tr[tr["unit"].isin(e_va)]), ("test", te)):
        out[name] = (
            add_features(frame, KEEP, dataset, stats=st_new)[0],
            legacy_add_features(frame, KEEP, dataset, stats=st_old)[0],
        )
    return out


@pytest.mark.parametrize("dataset", ["FD001", "FD003"])
def test_single_regime_features_bit_identical(dataset: str) -> None:
    for split, (new, old) in _frames(dataset).items():
        assert new.index.equals(old.index), split
        assert np.array_equal(new[EXACT].to_numpy(), old[EXACT].to_numpy()), split
        slope_diff = np.abs(new[SLOPES].to_numpy() - old[SLOPES].to_numpy()).max()
        assert slope_diff <= SLOPE_ATOL, (split, slope_diff)


@pytest.mark.parametrize("dataset", ["FD002", "FD004"])
def test_multi_regime_features_match_to_rounding(dataset: str) -> None:
    for split, (new, old) in _frames(dataset).items():
        assert new.index.equals(old.index), split
        diff = np.abs(new[feat_cols(KEEP)].to_numpy() - old[feat_cols(KEEP)].to_numpy())
        assert diff.max() <= MULTI_REGIME_ATOL, (split, diff.max())
