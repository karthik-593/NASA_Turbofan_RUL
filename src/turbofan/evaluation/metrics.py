"""Evaluation metrics for RUL prediction.

Standard metrics used across the C-MAPSS literature:

RMSE
    Root Mean Squared Error.  Symmetric, penalises large errors quadratically.

MAE
    Mean Absolute Error.  Symmetric, easy to interpret in cycle units.

CMAPSS Score (S)
    Asymmetric exponential penalty defined in Saxena et al. (2008).

    For each engine unit *i* with prediction error d_i = ŷ_i - y_i:

        s_i = exp(-d_i / 13) - 1    if d_i < 0   (early prediction — less penalty)
        s_i = exp( d_i / 10) - 1    if d_i ≥ 0   (late prediction  — more penalty)

        S = Σ s_i

    The asymmetry reflects operational reality: predicting failure later than
    it occurs (d > 0) is more dangerous than predicting it too early.

Reference
---------
Saxena, A., Goebel, K., Simon, D., & Eklund, N. (2008).
    Damage propagation modeling for aircraft engine run-to-failure simulation.
    International Conference on Prognostics and Health Management (PHM).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from turbofan.config import MAINTENANCE_BUCKETS

# Derived from MAINTENANCE_BUCKETS — single source of truth in config.py.
_BUCKETS: list[tuple[float, float, str]] = [
    (lo, hi, f"{name}_rmse") for name, lo, hi in MAINTENANCE_BUCKETS
]


def _to_array(x: ArrayLike) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root Mean Squared Error.

    Parameters
    ----------
    y_true:
        Ground-truth RUL values.
    y_pred:
        Predicted RUL values.

    Returns
    -------
    float
        RMSE in the same units as the RUL (cycles).
    """
    y_true_arr = _to_array(y_true)
    y_pred_arr = _to_array(y_pred)
    return float(np.sqrt(np.mean((y_pred_arr - y_true_arr) ** 2)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Error.

    Parameters
    ----------
    y_true:
        Ground-truth RUL values.
    y_pred:
        Predicted RUL values.

    Returns
    -------
    float
        MAE in the same units as the RUL (cycles).
    """
    y_true_arr = _to_array(y_true)
    y_pred_arr = _to_array(y_pred)
    return float(np.mean(np.abs(y_pred_arr - y_true_arr)))


def cmapss_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Standard C-MAPSS asymmetric scoring function (lower is better).

    .. math::

        S = \\sum_{i=1}^{N} s_i, \\quad
        s_i = \\begin{cases}
            e^{-d_i/13} - 1 & d_i < 0 \\\\
            e^{\\phantom{-}d_i/10} - 1 & d_i \\ge 0
        \\end{cases}

    where :math:`d_i = \\hat{y}_i - y_i`.

    Parameters
    ----------
    y_true:
        Ground-truth RUL values, one per engine unit.
    y_pred:
        Predicted RUL values, one per engine unit, in the same order.

    Returns
    -------
    float
        Summed score across all units.  The minimum achievable value is 0
        (perfect predictions).  Scores grow exponentially with error magnitude
        and grow faster for late predictions (d > 0) than early ones (d < 0).

    Examples
    --------
    >>> cmapss_score([100], [100])
    0.0
    >>> cmapss_score([100], [90])   # early prediction (d = -10)
    1.1581055339484458
    >>> cmapss_score([100], [110])  # late prediction  (d = +10)
    1.718281828459045
    """
    y_true_arr = _to_array(y_true)
    y_pred_arr = _to_array(y_pred)

    d = y_pred_arr - y_true_arr  # positive → predicted later than actual (dangerous)

    scores = np.where(d < 0, np.expm1(-d / 13.0), np.expm1(d / 10.0))
    return float(scores.sum())


def per_bucket_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Per-RUL-bucket RMSE.

    Buckets: [0-25] Critical, [25-50] Urgent, [50-100] Monitor, [100+] Healthy.
    Returns NaN for any bucket with no samples.
    """
    y_true_arr = _to_array(y_true)
    y_pred_arr = _to_array(y_pred)
    result: dict[str, float] = {}
    for lo, hi, name in _BUCKETS:
        mask = (y_true_arr >= lo) & (y_true_arr < hi)
        if mask.sum() > 0:
            result[name] = float(np.sqrt(np.mean((y_pred_arr[mask] - y_true_arr[mask]) ** 2)))
        else:
            result[name] = float("nan")
    return result


def late_prediction_pct(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Percentage of predictions that are late (predicted RUL > true RUL)."""
    y_true_arr = _to_array(y_true)
    y_pred_arr = _to_array(y_pred)
    return float(np.mean(y_pred_arr > y_true_arr) * 100.0)
