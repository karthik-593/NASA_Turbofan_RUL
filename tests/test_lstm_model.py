"""Tests for src/turbofan/models/lstm_model.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbofan.models.lstm_model import (
    LSTMRUL,
    RUL_CAP,
    make_last_windows,
    make_sequences,
)

FEAT_COLS = ["f0", "f1", "f2"]


def _make_feat_df(n_units: int = 3, cycles_per_unit: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for uid in range(1, n_units + 1):
        for cyc in range(1, cycles_per_unit + 1):
            row: dict = {
                "unit": uid,
                "cycle": cyc,
                "rul": float(cycles_per_unit - cyc),
            }
            for col in FEAT_COLS:
                row[col] = float(rng.normal())
            rows.append(row)
    return pd.DataFrame(rows)


def _fitted(seq_len: int = 5) -> tuple[LSTMRUL, np.ndarray, np.ndarray]:
    df_tr = _make_feat_df(n_units=4, cycles_per_unit=10, seed=0)
    df_va = _make_feat_df(n_units=2, cycles_per_unit=10, seed=1)
    X_tr, y_tr = make_sequences(df_tr, FEAT_COLS, seq_len=seq_len)
    X_va, y_va = make_sequences(df_va, FEAT_COLS, seq_len=seq_len)
    model = LSTMRUL(
        n_features=len(FEAT_COLS),
        hidden=4,
        layers=1,
        max_epochs=2,
        patience=1,
        batch_size=8,
    )
    model.fit(X_tr, y_tr, X_va, y_va)
    return model, X_tr, y_tr


class TestMakeSequences:
    def test_output_shapes(self) -> None:
        df = _make_feat_df(n_units=2, cycles_per_unit=6)
        X, y = make_sequences(df, FEAT_COLS, seq_len=4)
        assert X.shape == (12, 4, len(FEAT_COLS))
        assert y.shape == (12,)

    def test_dtype_is_float32(self) -> None:
        df = _make_feat_df(n_units=1, cycles_per_unit=5)
        X, y = make_sequences(df, FEAT_COLS, seq_len=3)
        assert X.dtype == np.float32
        assert y.dtype == np.float32

    def test_short_history_is_left_padded_with_zeros(self) -> None:
        """First window of an engine, before seq_len cycles exist, must be zero-padded."""
        df = _make_feat_df(n_units=1, cycles_per_unit=5)
        X, _ = make_sequences(df, FEAT_COLS, seq_len=5)
        first_window = X[0]
        np.testing.assert_array_equal(first_window[:4], np.zeros((4, len(FEAT_COLS))))

    def test_target_is_rul_at_window_end(self) -> None:
        df = _make_feat_df(n_units=1, cycles_per_unit=6)
        _, y = make_sequences(df, FEAT_COLS, seq_len=3)
        expected = df.sort_values("cycle")["rul"].to_numpy(dtype=np.float32)
        np.testing.assert_array_equal(y, expected)


class TestMakeLastWindows:
    def test_one_window_per_engine(self) -> None:
        df = _make_feat_df(n_units=4, cycles_per_unit=6)
        X, units = make_last_windows(df, FEAT_COLS, seq_len=3)
        assert X.shape == (4, 3, len(FEAT_COLS))
        assert list(units) == [1, 2, 3, 4]

    def test_window_ends_at_last_cycle(self) -> None:
        df = _make_feat_df(n_units=1, cycles_per_unit=6)
        X, _ = make_last_windows(df, FEAT_COLS, seq_len=3)
        expected = df.sort_values("cycle")[FEAT_COLS].to_numpy(dtype=np.float32)[-3:]
        np.testing.assert_array_equal(X[0], expected)

    def test_short_history_is_left_padded(self) -> None:
        df = _make_feat_df(n_units=1, cycles_per_unit=2)
        X, _ = make_last_windows(df, FEAT_COLS, seq_len=5)
        np.testing.assert_array_equal(X[0][:3], np.zeros((3, len(FEAT_COLS))))


class TestFitPredict:
    def test_fit_returns_self(self) -> None:
        model = LSTMRUL(n_features=len(FEAT_COLS), hidden=4, layers=1, max_epochs=1)
        X_tr, y_tr = make_sequences(_make_feat_df(seed=0), FEAT_COLS, seq_len=5)
        assert model.fit(X_tr, y_tr, X_tr, y_tr) is model

    def test_predict_output_shape_matches_input(self) -> None:
        model, X_tr, _ = _fitted()
        assert model.predict(X_tr).shape == (len(X_tr),)

    def test_predict_before_fit_raises(self) -> None:
        model = LSTMRUL(n_features=len(FEAT_COLS))
        X, _ = make_sequences(_make_feat_df(), FEAT_COLS, seq_len=5)
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_best_val_loss_recorded(self) -> None:
        model, _, _ = _fitted()
        assert model.best_val_loss_ is not None
        assert model.best_val_loss_ >= 0.0


class TestClipping:
    def test_clip_true_bounds_predictions_to_0_and_cap(self) -> None:
        model, X_tr, _ = _fitted()
        pred = model.predict(X_tr, clip=True)
        assert np.all(pred >= 0.0)
        assert np.all(pred <= RUL_CAP)


class TestSaveLoad:
    def test_save_before_fit_raises(self, tmp_path: Path) -> None:
        model = LSTMRUL(n_features=len(FEAT_COLS))
        with pytest.raises(RuntimeError):
            model.save(tmp_path / "model.pt")

    def test_load_reproduces_predictions(self, tmp_path: Path) -> None:
        model, X_tr, _ = _fitted()
        path = tmp_path / "model.pt"
        model.save(path)
        loaded = LSTMRUL.load(path)
        np.testing.assert_allclose(loaded.predict(X_tr), model.predict(X_tr), atol=1e-6)

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        model, _, _ = _fitted()
        path = tmp_path / "nested" / "dir" / "model.pt"
        model.save(path)
        assert path.exists()
