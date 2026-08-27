"""Tests for src/turbofan/models/xgboost_model.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbofan.models.xgboost_model import RUL_CAP, XGBoostRUL


def _make_xy(n: int = 80, n_features: int = 5, seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    cols = [f"f{i}" for i in range(n_features)]
    X = pd.DataFrame(rng.normal(size=(n, n_features)), columns=cols)
    y = rng.uniform(0.0, RUL_CAP, size=n)
    return X, y


def _fitted(n_estimators: int = 20) -> tuple[XGBoostRUL, pd.DataFrame, np.ndarray]:
    X_train, y_train = _make_xy(seed=0)
    X_val, y_val = _make_xy(n=20, seed=1)
    model = XGBoostRUL(params={"n_estimators": n_estimators})
    model.fit(X_train, y_train, X_val, y_val)
    return model, X_train, y_train


class TestFitPredict:
    def test_fit_returns_self(self) -> None:
        X_train, y_train = _make_xy(seed=0)
        X_val, y_val = _make_xy(n=20, seed=1)
        model = XGBoostRUL(params={"n_estimators": 5})
        assert model.fit(X_train, y_train, X_val, y_val) is model

    def test_predict_output_length_matches_input(self) -> None:
        model, X_train, _ = _fitted()
        assert len(model.predict(X_train)) == len(X_train)

    def test_predict_before_fit_raises(self) -> None:
        model = XGBoostRUL()
        with pytest.raises(RuntimeError):
            model.predict(_make_xy()[0])

    def test_feature_names_captured_from_dataframe_columns(self) -> None:
        model, X_train, _ = _fitted()
        assert model.feature_names_ == list(X_train.columns)

    def test_accepts_numpy_array_input(self) -> None:
        X_train, y_train = _make_xy(seed=0)
        X_val, y_val = _make_xy(n=20, seed=1)
        model = XGBoostRUL(params={"n_estimators": 5})
        model.fit(X_train.to_numpy(), y_train, X_val.to_numpy(), y_val)
        assert len(model.predict(X_train.to_numpy())) == len(X_train)


class TestClipping:
    def test_clip_true_bounds_predictions_to_0_and_cap(self) -> None:
        model, X_train, _ = _fitted()
        pred = model.predict(X_train, clip=True)
        assert np.all(pred >= 0.0)
        assert np.all(pred <= RUL_CAP)

    def test_clip_false_may_exceed_cap(self) -> None:
        """With a tiny/underfit model on a wide y range, unclipped predictions
        are not guaranteed in-range — clip=False must not silently clip them."""
        X_train, y_train = _make_xy(seed=0)
        X_val, y_val = _make_xy(n=20, seed=1)
        model = XGBoostRUL(params={"n_estimators": 5})
        model.fit(X_train, y_train, X_val, y_val, verbose=False)
        clipped = model.predict(X_train, clip=True, rul_cap=1.0)
        unclipped = model.predict(X_train, clip=False, rul_cap=1.0)
        assert not np.allclose(clipped, unclipped)
        assert np.all(clipped <= 1.0)


class TestParams:
    def test_custom_params_override_defaults(self) -> None:
        model = XGBoostRUL(params={"max_depth": 3})
        assert model.params["max_depth"] == 3

    def test_default_params_present_when_none_given(self) -> None:
        model = XGBoostRUL()
        assert model.params["n_estimators"] == 500


class TestProperties:
    def test_feature_importances_raises_before_fit(self) -> None:
        model = XGBoostRUL()
        with pytest.raises(RuntimeError):
            _ = model.feature_importances_

    def test_feature_importances_indexed_by_feature_name(self) -> None:
        model, X_train, _ = _fitted()
        assert set(model.feature_importances_.index) == set(X_train.columns)

    def test_feature_importances_sorted_descending(self) -> None:
        model, _, _ = _fitted()
        values = model.feature_importances_.to_numpy()
        assert list(values) == sorted(values, reverse=True)

    def test_best_iteration_raises_before_fit(self) -> None:
        model = XGBoostRUL()
        with pytest.raises(RuntimeError):
            _ = model.best_iteration_

    def test_best_iteration_is_int_after_fit(self) -> None:
        model, _, _ = _fitted()
        assert isinstance(model.best_iteration_, int)


class TestSaveLoad:
    def test_load_reproduces_predictions(self, tmp_path: Path) -> None:
        model, X_train, _ = _fitted()
        path = tmp_path / "model.pkl"
        model.save(path)
        loaded = XGBoostRUL.load(path)
        np.testing.assert_array_equal(loaded.predict(X_train), model.predict(X_train))

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        model, _, _ = _fitted()
        path = tmp_path / "nested" / "dir" / "model.pkl"
        model.save(path)
        assert path.exists()
