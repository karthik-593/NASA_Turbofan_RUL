"""Serving-vs-training parity — the project's #1 correctness guard.

The window the API feeds the model (request -> to_frame -> add_features -> padded/masked
last window, ``service.serving_window``) must be bit-for-bit identical to the window the
training/eval path builds for the same engine (``add_features`` on the raw rows ->
``make_last_windows``). If these two paths ever drift, predictions silently degrade.

Covered on all four datasets: a full-length engine, every test engine shorter than
``SEQ_LEN`` (padded, D11; FD002/FD004 have them) and engines truncated to exactly
``serving.min_history`` cycles. Model-free: it compares the feature windows directly, so it
needs no trained bundle — only the fitted feature state and a manifest.

Requires data/raw to be present (reads test-set *features* only; no labels).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbofan.config import KEEP, MIN_HISTORY, SENSOR_N_COLS, SEQ_LEN, WINDOW
from turbofan.data.loader import load_dataset
from turbofan.evaluation.comparison import split_engines
from turbofan.features.engineering import add_features
from turbofan.models.lstm_model import make_last_windows
from turbofan.serving.schemas import CycleReading
from turbofan.serving.service import serving_window
from turbofan.training.bundle import Bundle

pytestmark = pytest.mark.requires_data

RAW = "data/raw"
DATASETS = ["FD001", "FD002", "FD003", "FD004"]
N_TRUNCATED = 5  # engines per dataset also checked at exactly min_history cycles


def _cycles_for_unit(unit_df: pd.DataFrame) -> list[CycleReading]:
    return [
        CycleReading(
            op1=float(r.op1),
            op2=float(r.op2),
            op3=float(r.op3),
            sensors={s: float(r[s]) for s in KEEP},
        )
        for _, r in unit_df.sort_values("cycle").iterrows()
    ]


def _bundle(dataset: str, stats: dict) -> Bundle:
    manifest = {
        "dataset": dataset,
        "version": "parity",
        "config": {"keep": KEEP, "window": WINDOW, "seq_len": SEQ_LEN},
    }
    return Bundle(model=None, stats=stats, manifest=manifest, path=None)


@pytest.fixture(scope="module", params=DATASETS)
def fitted(request: pytest.FixtureRequest) -> tuple[str, pd.DataFrame, Bundle]:
    dataset = request.param
    tr, te, _rul = load_dataset(dataset, RAW)
    tr_e, _va_e = split_engines(tr)
    _feat_tr, stats = add_features(tr[tr["unit"].isin(tr_e)], KEEP, dataset)
    return dataset, te, _bundle(dataset, stats)


def _engines_to_check(te: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    lengths = te.groupby("unit")["cycle"].size()
    cases = [("full", te[te["unit"] == lengths.idxmax()])]
    for u in lengths[(lengths < SEQ_LEN) & (lengths >= MIN_HISTORY)].index:
        cases.append((f"short-{u}", te[te["unit"] == u]))
    for u in lengths[lengths >= MIN_HISTORY].index[:N_TRUNCATED]:
        g = te[te["unit"] == u].sort_values("cycle")
        cases.append((f"trunc-{u}", g.head(MIN_HISTORY)))
    return cases


def test_serving_window_matches_training(fitted: tuple[str, pd.DataFrame, Bundle]) -> None:
    dataset, te, bundle = fitted
    for name, eng in _engines_to_check(te):
        feat, _ = add_features(eng, KEEP, dataset, stats=bundle.stats)
        X_train_path, _ = make_last_windows(feat, SENSOR_N_COLS, SEQ_LEN)
        X_serving = serving_window(_cycles_for_unit(eng), bundle)
        assert X_serving.shape == X_train_path.shape == (1, SEQ_LEN, len(SENSOR_N_COLS) + 1)
        assert np.array_equal(X_serving, X_train_path), f"{dataset} {name}: window diverged"
        n_real = min(len(eng), SEQ_LEN)
        assert X_serving[0, :, -1].sum() == n_real, f"{dataset} {name}: mask"


def test_real_short_test_engines_are_covered(fitted: tuple[str, pd.DataFrame, Bundle]) -> None:
    """Audit §A: FD002 has 6 and FD004 11 test engines shorter than SEQ_LEN; none of them
    is below min_history, so all are servable and all go through the padded path above."""
    dataset, te, _ = fitted
    lengths = te.groupby("unit")["cycle"].size()
    expected_short = {"FD001": 0, "FD002": 6, "FD003": 0, "FD004": 11}[dataset]
    assert int((lengths < SEQ_LEN).sum()) == expected_short
    assert int((lengths < MIN_HISTORY).sum()) == 0
