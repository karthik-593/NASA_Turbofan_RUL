"""Tests for src/turbofan/training/train.py — _fit_and_eval()."""

from __future__ import annotations

import numpy as np
import pandas as pd

from turbofan.config import FEAT_COLS, SENSOR_N_COLS
from turbofan.models.baselines import RidgeRUL
from turbofan.models.lstm_model import LSTMRUL
from turbofan.training.train import _fit_and_eval


def _make_df(cols: list[str], n_units: int, cycles_per_unit: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for uid in range(1, n_units + 1):
        for cyc in range(1, cycles_per_unit + 1):
            row: dict = {"unit": uid, "cycle": cyc, "rul": float(cycles_per_unit - cyc)}
            for col in cols:
                row[col] = float(rng.normal())
            rows.append(row)
    return pd.DataFrame(rows)


class TestFlatKind:
    def test_returns_fitted_model_and_matching_length_arrays(self) -> None:
        feat_tr = _make_df(FEAT_COLS, n_units=4, cycles_per_unit=10, seed=0)
        feat_va = _make_df(FEAT_COLS, n_units=2, cycles_per_unit=10, seed=1)
        feat_te = _make_df(FEAT_COLS, n_units=3, cycles_per_unit=10, seed=2)
        rul_te = pd.Series([9.0, 9.0, 9.0], index=[1, 2, 3])
        model = RidgeRUL()

        fitted, y, pred = _fit_and_eval(model, "flat", feat_tr, feat_va, feat_te, rul_te)

        assert fitted is model
        assert len(y) == 3
        assert len(pred) == 3

    def test_true_rul_overrides_rul_column(self) -> None:
        feat_tr = _make_df(FEAT_COLS, n_units=4, cycles_per_unit=10, seed=0)
        feat_va = _make_df(FEAT_COLS, n_units=2, cycles_per_unit=10, seed=1)
        feat_te = _make_df(FEAT_COLS, n_units=2, cycles_per_unit=5, seed=2)
        feat_te["rul"] = 999.0  # sentinel; must not appear in y
        rul_te = pd.Series([10.0, 20.0], index=[1, 2])

        _, y, _ = _fit_and_eval(RidgeRUL(), "flat", feat_tr, feat_va, feat_te, rul_te)

        assert set(y) == {10.0, 20.0}


class TestSequenceKind:
    def test_returns_fitted_model_and_matching_length_arrays(self) -> None:
        feat_tr = _make_df(SENSOR_N_COLS, n_units=4, cycles_per_unit=8, seed=0)
        feat_va = _make_df(SENSOR_N_COLS, n_units=2, cycles_per_unit=8, seed=1)
        feat_te = _make_df(SENSOR_N_COLS, n_units=3, cycles_per_unit=8, seed=2)
        rul_te = pd.Series([4.0, 4.0, 4.0], index=[1, 2, 3])
        model = LSTMRUL(n_features=len(SENSOR_N_COLS), hidden=4, layers=1, max_epochs=2)

        fitted, y, pred = _fit_and_eval(model, "sequence", feat_tr, feat_va, feat_te, rul_te)

        assert fitted is model
        assert len(y) == 3
        assert len(pred) == 3
