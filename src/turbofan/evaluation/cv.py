"""Protocol-v2 cross-validation folds over training engines (D46). Never reads the test set.

``make_folds``: repeated ``StratifiedGroupKFold`` — groups = engine (an engine's rows never
sit on both sides of a split), stratified by the engine's subpopulation label (D35) so every
held-out fold contains both degradation subpopulations in proportion. Within each fold's
training engines a stratified share (``cv.inner_val_frac``) is set aside for early stopping,
so early stopping never sees the held-out engines either.

``prepare_fold``: every fitted piece of state — regime model, per-regime normalization,
operating envelope — is fitted by ``add_features`` on the fold's inner-training engines only
and reused unchanged on its inner-validation and held-out engines. Training labels are capped
at the candidate's ``rul_cap``; held-out rows keep the uncapped true RUL (``rul_true``),
known because training trajectories run to failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from turbofan.config import MODEL_SENSORS, PARAMS
from turbofan.features.engineering import add_features

__all__ = ["Fold", "FoldData", "make_folds", "prepare_fold", "true_rul"]

_CV = PARAMS["cv"]


@dataclass(frozen=True)
class Fold:
    repeat: int
    fold: int
    train_units: npt.NDArray[np.int64]  # inner-training engines: fit features and model
    val_units: npt.NDArray[np.int64]  # inner-validation engines: early stopping only
    test_units: npt.NDArray[np.int64]  # held-out engines: evaluation only

    @property
    def fit_units(self) -> npt.NDArray[np.int64]:
        """All of the fold's training-side engines (inner train + inner validation)."""
        return np.sort(np.concatenate([self.train_units, self.val_units]))


@dataclass(frozen=True)
class FoldData:
    fold: Fold
    feat_tr: pd.DataFrame
    feat_va: pd.DataFrame
    feat_te: pd.DataFrame
    stats: dict[str, Any]


def make_folds(
    labels: pd.Series[int],
    n_folds: int = _CV["n_folds"],
    n_repeats: int = _CV["n_repeats"],
    seed: int = _CV["seed"],
    inner_val_frac: float = _CV["inner_val_frac"],
) -> list[Fold]:
    """``n_repeats`` x ``n_folds`` folds over the engines in ``labels`` (index = unit,
    value = subpopulation label). Deterministic given ``seed``."""
    if labels.index.has_duplicates:
        raise ValueError("labels must have one entry per engine")
    units = labels.index.to_numpy(dtype=np.int64)
    y = labels.to_numpy()
    folds = []
    for r in range(n_repeats):
        skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed * 1000 + r)
        for f, (fit_idx, test_idx) in enumerate(skf.split(units.reshape(-1, 1), y, groups=units)):
            tr_idx, va_idx = train_test_split(
                fit_idx,
                test_size=inner_val_frac,
                stratify=y[fit_idx],
                random_state=seed * 1000 + r * 100 + f,
            )
            folds.append(
                Fold(
                    repeat=r,
                    fold=f,
                    train_units=np.sort(units[tr_idx]),
                    val_units=np.sort(units[va_idx]),
                    test_units=np.sort(units[test_idx]),
                )
            )
    return folds


def true_rul(df: pd.DataFrame) -> pd.Series[int]:
    """Uncapped remaining life of each row of complete run-to-failure trajectories."""
    out: pd.Series[int] = df.groupby("unit")["cycle"].transform("max") - df["cycle"]
    return out


def prepare_fold(
    train: pd.DataFrame,
    dataset: str,
    fold: Fold,
    rul_cap: float,
    sensors: list[str] = MODEL_SENSORS,
) -> FoldData:
    """Features for one fold, all state fitted on the fold's inner-training engines only.

    Every frame gets ``rul_true`` (uncapped) and ``rul`` (capped at ``rul_cap``, the training
    label)."""
    d = train.copy()
    d["rul_true"] = true_rul(d)
    d["rul"] = d["rul_true"].clip(upper=rul_cap)
    feat_tr, stats = add_features(d[d["unit"].isin(fold.train_units)], sensors, dataset)
    feat_va, _ = add_features(d[d["unit"].isin(fold.val_units)], sensors, dataset, stats=stats)
    feat_te, _ = add_features(d[d["unit"].isin(fold.test_units)], sensors, dataset, stats=stats)
    return FoldData(fold=fold, feat_tr=feat_tr, feat_va=feat_va, feat_te=feat_te, stats=stats)
