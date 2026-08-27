"""Baseline RUL models for the C-MAPSS comparison.

Thin wrappers exposing the same fit/predict/save/load surface as XGBoostRUL so the
comparison can treat every flat-feature model identically. The val arguments are accepted
for interface symmetry and ignored by models that don't need early stopping.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Self, cast

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RUL_CAP = 125.0

__all__ = ["MeanBaseline", "RidgeRUL", "RandomForestRUL", "RUL_CAP"]


def _clip(p: npt.ArrayLike, cap: float = RUL_CAP) -> npt.NDArray[np.float64]:
    return np.asarray(np.clip(np.asarray(p, dtype=float), 0.0, cap), dtype=np.float64)


def _pickle_save(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.parent / (p.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    tmp.replace(p)


def _pickle_load(path: str | Path) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


class MeanBaseline:
    """Predicts the training-set mean RUL. The floor every model must beat."""

    def __init__(self) -> None:
        self.mean_: float | None = None

    def fit(
        self,
        X_train: Any,
        y_train: npt.ArrayLike,
        X_val: Any = None,
        y_val: Any = None,
    ) -> Self:
        self.mean_ = float(np.mean(np.asarray(y_train, dtype=float)))
        return self

    def predict(self, X: Any, clip: bool = True) -> npt.NDArray[np.float64]:
        fill = self.mean_ if self.mean_ is not None else float("nan")
        p = np.full(len(X), fill, dtype=float)
        return _clip(p) if clip else p

    def save(self, path: str | Path) -> None:
        _pickle_save(self, path)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        return cast(Self, _pickle_load(path))


class RidgeRUL:
    """StandardScaler + Ridge. Tells us whether the problem needs nonlinearity."""

    def __init__(self, alpha: float = 1.0):
        self.model_ = make_pipeline(StandardScaler(), Ridge(alpha=alpha))

    def fit(
        self,
        X_train: Any,
        y_train: npt.ArrayLike,
        X_val: Any = None,
        y_val: Any = None,
    ) -> Self:
        self.model_.fit(X_train, y_train)
        return self

    def predict(self, X: Any, clip: bool = True) -> npt.NDArray[np.float64]:
        p = self.model_.predict(X)
        return _clip(p) if clip else p

    def save(self, path: str | Path) -> None:
        _pickle_save(self, path)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        return cast(Self, _pickle_load(path))


class RandomForestRUL:
    """Random forest — a bagging ensemble, a different bias/variance profile than boosting."""

    def __init__(self, n_estimators: int = 300, random_state: int = 42, **kw: Any) -> None:
        self.model_ = RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state, n_jobs=-1, **kw
        )

    def fit(
        self,
        X_train: Any,
        y_train: npt.ArrayLike,
        X_val: Any = None,
        y_val: Any = None,
    ) -> Self:
        self.model_.fit(X_train, y_train)
        return self

    def predict(self, X: Any, clip: bool = True) -> npt.NDArray[np.float64]:
        p = self.model_.predict(X)
        return _clip(p) if clip else p

    def save(self, path: str | Path) -> None:
        _pickle_save(self, path)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        return cast(Self, _pickle_load(path))
