"""Fit a registry candidate and predict with it — shared by CV and ``final_eval``.

One code path for both, so the model the final evaluation scores is trained exactly like the
ones model selection compared. ``kind`` is the registry's: 'flat' (the engineered feature
columns, one row per cycle) or 'sequence' (padded + masked windows of the normalized
channels, one per cycle).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from turbofan.config import FEAT_COLS, SENSOR_N_COLS, SEQ_LEN
from turbofan.models.lstm_model import make_last_windows, make_sequences

__all__ = [
    "fit_candidate",
    "iteration_budget",
    "refit_candidate",
    "predict_every_cycle",
    "predict_last_cycle",
]

F64 = npt.NDArray[np.float64]


def fit_candidate(model: Any, kind: str, feat_tr: pd.DataFrame, feat_va: pd.DataFrame) -> Any:
    """Fit on ``feat_tr``; ``feat_va`` only for early stopping. Targets = the ``rul`` column
    (already capped at the candidate's rul_cap)."""
    if kind == "flat":
        model.fit(
            feat_tr[FEAT_COLS],
            feat_tr["rul"].to_numpy(),
            feat_va[FEAT_COLS],
            feat_va["rul"].to_numpy(),
        )
    elif kind == "sequence":
        X_tr, y_tr = make_sequences(feat_tr, SENSOR_N_COLS, SEQ_LEN)
        X_va, y_va = make_sequences(feat_va, SENSOR_N_COLS, SEQ_LEN)
        model.fit(X_tr, y_tr, X_va, y_va)
    else:
        raise ValueError(f"unknown candidate kind {kind!r}")
    return model


def iteration_budget(model: Any) -> int | None:
    """Rounds / epochs a fitted candidate used (the early-stopped best): boosting rounds for
    XGBoost, the best epoch for the LSTM, None for models with no iteration count."""
    if hasattr(model, "best_epoch_"):
        return None if model.best_epoch_ is None else int(model.best_epoch_)
    if hasattr(model, "best_iteration_"):
        return int(model.best_iteration_) + 1  # best_iteration is 0-based
    return None


def refit_candidate(model: Any, kind: str, feat_all: pd.DataFrame, budget: int | None) -> Any:
    """Fit on every engine in ``feat_all`` with no validation set (D48). Iterative models
    train for exactly ``budget`` rounds / epochs; the others ignore it and require None."""
    if not hasattr(model, "fit_fixed"):
        if budget is not None:
            raise ValueError(f"{type(model).__name__} has no iteration count; budget must be None")
    elif budget is None:
        raise ValueError(
            f"{type(model).__name__} needs an iteration budget to refit without validation"
        )
    X: Any
    y: Any
    if kind == "flat":
        X, y = feat_all[FEAT_COLS], feat_all["rul"].to_numpy()
    elif kind == "sequence":
        X, y = make_sequences(feat_all, SENSOR_N_COLS, SEQ_LEN)
    else:
        raise ValueError(f"unknown candidate kind {kind!r}")
    if hasattr(model, "fit_fixed"):
        model.fit_fixed(X, y, budget)
    else:
        model.fit(X, y)
    return model


def predict_every_cycle(model: Any, kind: str, feat: pd.DataFrame) -> pd.DataFrame:
    """A prediction at every cycle of every engine in ``feat`` (sorted by unit, cycle):
    columns unit, cycle, rul_true, pred."""
    f = feat.sort_values(["unit", "cycle"])
    if kind == "flat":
        pred = np.asarray(model.predict(f[FEAT_COLS]), dtype=float)
    elif kind == "sequence":
        X, _ = make_sequences(f, SENSOR_N_COLS, SEQ_LEN)  # same unit, cycle order as f
        pred = np.asarray(model.predict(X), dtype=float)
    else:
        raise ValueError(f"unknown candidate kind {kind!r}")
    return pd.DataFrame(
        {
            "unit": f["unit"].to_numpy(),
            "cycle": f["cycle"].to_numpy(),
            "rul_true": f["rul_true"].to_numpy(),
            "pred": pred,
        }
    )


def predict_last_cycle(
    model: Any, kind: str, feat: pd.DataFrame
) -> tuple[npt.NDArray[np.int64], F64]:
    """One prediction per engine at its last observed cycle: (units, predictions)."""
    if kind == "flat":
        last = feat.sort_values(["unit", "cycle"]).groupby("unit").tail(1)
        return last["unit"].to_numpy(np.int64), np.asarray(model.predict(last[FEAT_COLS]), float)
    if kind == "sequence":
        X, units = make_last_windows(feat, SENSOR_N_COLS, SEQ_LEN)
        return units, np.asarray(model.predict(X), dtype=float)
    raise ValueError(f"unknown candidate kind {kind!r}")
