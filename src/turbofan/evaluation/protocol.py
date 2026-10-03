"""Evaluation protocol — the scoring contract every model is held to.

``eval_lc`` (verbatim from notebook 02) extracts each engine's prediction at its last
observed cycle vs ground truth — used by the flat models. ``score`` turns any
(y_true, y_pred) pair into the one metric row used by the comparison, so flat and
sequence models are scored identically. Headline: critical-zone [0-25] RMSE
(``critical_rmse``); tiebreaker: NASA score (``nasa``); also global RMSE and late %.

Label consistency: training labels are capped at ``rul_cap`` but NASA's test truth is not.
``score`` therefore reports every metric against both — ``<metric>_vs_uncapped`` and
``<metric>_vs_capped`` (truth clipped at ``rul_cap``) — and the unsuffixed headline keys
use whichever ``params.yaml`` ``eval.cap_test_truth`` selects (default: uncapped).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from turbofan.config import CAP_TEST_TRUTH, FEAT_COLS, RUL_CAP
from turbofan.evaluation.metrics import (
    cmapss_score,
    late_prediction_pct,
    per_bucket_metrics,
    rmse,
)

__all__ = ["eval_lc", "predict_last_cycle", "score"]


def predict_last_cycle(
    model: Any,
    feat_df: pd.DataFrame,
    true_rul: pd.Series[float] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Extract one (y_true, y_pred) per engine at its last observed cycle.
    feat_df has columns unit, cycle, optional rul, plus FEAT_COLS.
    true_rul: pd.Series indexed 1..N by unit — pass for the test set."""
    last_idx = np.asarray(feat_df.groupby("unit")["cycle"].idxmax().to_numpy())
    units = feat_df.loc[last_idx, "unit"].to_numpy()
    X = feat_df.loc[last_idx, FEAT_COLS]
    y = (
        true_rul.loc[units].to_numpy().astype(float)
        if true_rul is not None
        else feat_df.loc[last_idx, "rul"].to_numpy().astype(float)
    )
    return y, model.predict(X)


def eval_lc(
    model: Any,
    feat_df: pd.DataFrame,
    true_rul: pd.Series[float] | None = None,
) -> tuple[dict[str, float], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    # Evaluate at each engine's last observed cycle only.
    # feat_df  : has columns unit, cycle, optional rul, plus FEAT_COLS.
    # true_rul : pd.Series indexed 1..N by unit number — pass for test set.
    # Returns  : (metrics_dict, y_true_array, y_pred_array)
    y, pred = predict_last_cycle(model, feat_df, true_rul)
    s = score(y, pred)
    out = {"rmse": s.pop("global_rmse"), "cmapss_score": s.pop("nasa"), "n_engines": s.pop("n")}
    out.update(s)
    return out, y, pred


def _metrics(y: npt.NDArray[np.float64], pred: npt.NDArray[np.float64]) -> dict[str, float]:
    out = {
        "global_rmse": rmse(y, pred),
        "nasa": cmapss_score(y, pred),
        "late_pct": late_prediction_pct(y, pred),
    }
    out.update(per_bucket_metrics(y, pred))
    return out


def score(
    y: Any,
    pred: Any,
    *,
    cap_truth: bool = CAP_TEST_TRUTH,
    rul_cap: float = RUL_CAP,
) -> dict[str, float]:
    """One metric row for any model, from a (y_true, y_pred) pair.

    Headline keys (``global_rmse``, ``nasa``, ``late_pct``, ``<bucket>_rmse``) are computed
    against the truth ``cap_truth`` selects; every one is also reported as
    ``<key>_vs_uncapped`` and ``<key>_vs_capped``. ``n`` = number of predictions.
    """
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    uncapped = _metrics(y, pred)
    capped = _metrics(np.minimum(y, rul_cap), pred)
    out: dict[str, float] = dict(capped if cap_truth else uncapped)
    out["n"] = len(y)
    out.update({f"{k}_vs_uncapped": v for k, v in uncapped.items()})
    out.update({f"{k}_vs_capped": v for k, v in capped.items()})
    return out
