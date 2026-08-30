"""Serving-vs-training parity — the project's #1 correctness guard.

The window the API feeds the model (request -> to_frame -> add_features -> last window)
must be bit-for-bit identical to the window the training/eval path builds for the same
engine. If these two paths ever drift, predictions silently degrade. This test is
model-free: it compares the feature windows directly, so it needs no trained bundle.

Requires data/raw to be present.
"""

from __future__ import annotations

import numpy as np
import pytest

from turbofan.config import KEEP, SENSOR_N_COLS, SEQ_LEN
from turbofan.data.loader import load_dataset
from turbofan.evaluation.comparison import split_engines
from turbofan.features.engineering import add_features
from turbofan.models.lstm_model import make_last_windows
from turbofan.serving.schemas import CycleReading
from turbofan.serving.service import to_frame

RAW = "data/raw"


def _cycles_for_unit(unit_df):
    out = []
    for _, r in unit_df.sort_values("cycle").iterrows():
        out.append(
            CycleReading(
                op1=float(r.op1),
                op2=float(r.op2),
                op3=float(r.op3),
                sensors={s: float(r[s]) for s in KEEP},
            )
        )
    return out


@pytest.mark.parametrize("dataset", ["FD001", "FD002"])
def test_serving_features_match_training(dataset):
    tr, te, _rul = load_dataset(dataset, RAW)
    tr_e, _va_e = split_engines(tr)
    _feat_tr, stats = add_features(tr[tr["unit"].isin(tr_e)], KEEP, dataset)

    # a test engine with enough history
    unit = next(u for u in te["unit"].unique() if (te["unit"] == u).sum() >= SEQ_LEN)
    te_u = te[te["unit"] == unit]

    # training/eval path: features straight from the raw test rows
    feat_train_path, _ = add_features(te_u, KEEP, dataset, stats=stats)
    X_a, _ = make_last_windows(feat_train_path, SENSOR_N_COLS, SEQ_LEN)

    # serving path: same rows arriving as an API request
    df_serving = to_frame(_cycles_for_unit(te_u))
    feat_serving_path, _ = add_features(df_serving, KEEP, dataset, stats=stats)
    X_b, _ = make_last_windows(feat_serving_path, SENSOR_N_COLS, SEQ_LEN)

    assert X_a.shape == X_b.shape
    assert np.allclose(X_a, X_b, atol=1e-10), "serving feature window diverged from training"
