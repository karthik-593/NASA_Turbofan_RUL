"""XGBoost model for C-MAPSS turbofan RUL prediction.

A thin, serialisable wrapper around XGBRegressor. The feature vector is flat
(one row per engine cycle) and the target is capped RUL. Hyperparameter search
and evaluation live in the modeling notebook, not here — this module is just the
model object: fit, predict, persist.

Usage
-----
    model = XGBoostRUL(params=best_params)
    model.fit(X_train, y_train, X_val, y_val)
    preds = model.predict(X_test)
    model.save("models/xgboost_FD001.pkl")
    model = XGBoostRUL.load("models/xgboost_FD001.pkl")
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

RUL_CAP: float = 125.0
_EARLY_STOP: int = 50

_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "eval_metric": "rmse",
    "random_state": 42,
    "n_jobs": -1,
}

__all__ = ["XGBoostRUL", "RUL_CAP"]


class XGBoostRUL:
    """Serialisable XGBRegressor wrapper for RUL prediction.

    Parameters
    ----------
    params:
        XGBoost hyperparameters, merged over the module defaults. Anything
        XGBRegressor accepts (including ``device``) is passed straight through.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params: dict[str, Any] = {**_DEFAULT_PARAMS, **(params or {})}
        self.model_: XGBRegressor | None = None
        self.feature_names_: list[str] | None = None

    # -- properties ----------------------------------------------------------

    @property
    def feature_importances_(self) -> pd.Series:
        """Gain-based feature importances, sorted descending."""
        if self.model_ is None:
            raise RuntimeError("Call fit() before accessing feature_importances_.")
        names = self.feature_names_ or [
            str(i) for i in range(len(self.model_.feature_importances_))
        ]
        return pd.Series(
            self.model_.feature_importances_, index=names, name="importance"
        ).sort_values(ascending=False)

    @property
    def best_iteration_(self) -> int:
        """Boosting round at which early stopping fired."""
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        return int(self.model_.best_iteration)

    # -- training ------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame | np.ndarray,
        y_train: np.ndarray,
        X_val: pd.DataFrame | np.ndarray,
        y_val: np.ndarray,
        early_stopping_rounds: int = _EARLY_STOP,
        verbose: bool = False,
    ) -> XGBoostRUL:
        """Fit with early stopping on the validation set."""
        if isinstance(X_train, pd.DataFrame):
            self.feature_names_ = list(X_train.columns)
        self.model_ = XGBRegressor(early_stopping_rounds=early_stopping_rounds, **self.params)
        self.model_.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=verbose)
        return self

    # -- inference -----------------------------------------------------------

    def predict(
        self,
        X: pd.DataFrame | np.ndarray,
        clip: bool = True,
        rul_cap: float = RUL_CAP,
    ) -> np.ndarray:
        """Predict RUL, clipped to [0, rul_cap] by default."""
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        preds = np.asarray(self.model_.predict(X), dtype=np.float64)
        return np.asarray(np.clip(preds, 0.0, rul_cap), dtype=np.float64) if clip else preds

    # -- persistence ---------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Pickle the wrapper to *path*, using an atomic write to avoid leaving
        a corrupt file behind if the process is interrupted mid-write."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.parent / (p.name + ".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(self, f)
        tmp.replace(p)

    @classmethod
    def load(cls, path: str | Path) -> XGBoostRUL:
        """Load a wrapper saved with save()."""
        with open(path, "rb") as f:
            return cast(XGBoostRUL, pickle.load(f))
