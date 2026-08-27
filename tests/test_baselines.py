"""Tests for src/turbofan/models/baselines.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from turbofan.models.baselines import RUL_CAP, MeanBaseline, RandomForestRUL, RidgeRUL

MODEL_CLASSES = [MeanBaseline, RidgeRUL, RandomForestRUL]


def _make_xy(n: int = 40, n_features: int = 5, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features))
    y = rng.uniform(0.0, RUL_CAP, size=n)
    return X, y


class TestMeanBaseline:
    def test_predicts_training_mean_for_every_row(self) -> None:
        X, y = _make_xy()
        model = MeanBaseline().fit(X, y)
        pred = model.predict(X, clip=False)
        assert np.allclose(pred, np.mean(y))

    def test_predict_output_length_matches_input(self) -> None:
        X, y = _make_xy(n=17)
        model = MeanBaseline().fit(X, y)
        assert len(model.predict(X)) == 17

    def test_fit_returns_self(self) -> None:
        X, y = _make_xy()
        model = MeanBaseline()
        assert model.fit(X, y) is model


class TestClipping:
    """Clipping behaviour is shared logic (_clip) — verified once per model class."""

    @pytest.mark.parametrize("cls", MODEL_CLASSES)
    def test_clip_true_bounds_predictions_to_0_and_cap(self, cls: type) -> None:
        X, y = _make_xy(n=60)
        y_extreme = np.where(np.arange(60) % 2 == 0, -1000.0, 1000.0)
        model = cls().fit(X, y_extreme)
        pred = model.predict(X, clip=True)
        assert np.all(pred >= 0.0)
        assert np.all(pred <= RUL_CAP)

    @pytest.mark.parametrize("cls", MODEL_CLASSES)
    def test_clip_false_allows_out_of_range_values(self, cls: type) -> None:
        X, y = _make_xy(n=60)
        y_extreme = np.full(60, 1000.0)
        model = cls().fit(X, y_extreme)
        pred = model.predict(X, clip=False)
        assert np.any(pred > RUL_CAP)


class TestRidgeRUL:
    def test_fit_predict_roundtrip_shape(self) -> None:
        X, y = _make_xy()
        model = RidgeRUL().fit(X, y)
        assert model.predict(X).shape == (len(X),)

    def test_fit_returns_self(self) -> None:
        X, y = _make_xy()
        model = RidgeRUL()
        assert model.fit(X, y) is model


class TestRandomForestRUL:
    def test_fit_predict_roundtrip_shape(self) -> None:
        X, y = _make_xy()
        model = RandomForestRUL(n_estimators=10).fit(X, y)
        assert model.predict(X).shape == (len(X),)

    def test_fit_returns_self(self) -> None:
        X, y = _make_xy()
        model = RandomForestRUL(n_estimators=10)
        assert model.fit(X, y) is model

    def test_extra_kwargs_forwarded_to_sklearn(self) -> None:
        model = RandomForestRUL(n_estimators=10, max_depth=3)
        assert model.model_.max_depth == 3


class TestSaveLoad:
    @pytest.mark.parametrize("cls", MODEL_CLASSES)
    def test_load_reproduces_predictions(self, cls: type, tmp_path: Path) -> None:
        X, y = _make_xy()
        model = cls().fit(X, y)
        path = tmp_path / "model.pkl"
        model.save(path)
        loaded = cls.load(path)
        assert np.allclose(loaded.predict(X), model.predict(X))

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        X, y = _make_xy()
        model = MeanBaseline().fit(X, y)
        path = tmp_path / "nested" / "dir" / "model.pkl"
        model.save(path)
        assert path.exists()
