"""Feature engineering for the C-MAPSS RUL pipeline.

`add_features` builds the lean feature set used everywhere downstream (modeling, comparison,
training, serving): per sensor, the normalized value plus a causal rolling mean and rolling
slope. Normalization is per operating regime, through one code path for every dataset:
KMeans with k = ``params.yaml`` ``n_regimes[dataset]`` on standardized op settings, then a
z-score per sensor within each regime. k = 1 is the global z-score (FD001/FD003); k = 6 is
the per-regime normalization of FD002/FD004 (D03, D04a/b). Normalization state is fitted on
the training engines and reused on val/test/serving by passing the returned `stats` back in.
All operations are right-aligned (no future leakage) and computed per engine.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from turbofan.config import (
    N_REGIMES,
    OP_COLS,
    REGIME_KMEANS_N_INIT,
    REGIME_KMEANS_SEED,
    WINDOW,
)

__all__ = ["add_features", "assign_regimes", "FEATURE_STATE_KEYS"]

# Keys a fitted feature state must carry; a state without them predates params.yaml.
FEATURE_STATE_KEYS = ("n_regimes", "sensors", "op_sc", "km", "s_mean", "s_std")

# Floor on a regime's sensor std so a constant sensor cannot divide by zero (D06).
_STD_FLOOR = 1e-9


def _fit_regimes(d: pd.DataFrame, k: int) -> tuple[StandardScaler, KMeans]:
    op_sc = StandardScaler().fit(d[OP_COLS])
    km = KMeans(n_clusters=k, n_init=REGIME_KMEANS_N_INIT, random_state=REGIME_KMEANS_SEED)
    km.fit(op_sc.transform(d[OP_COLS]))
    return op_sc, km


def assign_regimes(d: pd.DataFrame, stats: dict[str, Any]) -> npt.NDArray[np.intp]:
    """Regime label per row of ``d`` (row order preserved) from a fitted feature state."""
    labels: npt.NDArray[np.intp] = stats["km"].predict(stats["op_sc"].transform(d[OP_COLS]))
    return labels


def _fit_norm(
    d: pd.DataFrame, labels: npt.NDArray[np.intp], sensors: list[str], k: int
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """Per-sensor, per-regime mean and std, indexed by regime label.

    Computed on each regime's row subset with ``Series.mean/std``: for k = 1 that is
    bit-identical to the former global ``d[s].mean()/std()``, whereas pandas' groupby
    aggregations differ from them in the last bits.
    """
    s_mean: dict[str, list[float]] = {s: [] for s in sensors}
    s_std: dict[str, list[float]] = {s: [] for s in sensors}
    for r in range(k):
        rows = d.loc[labels == r]
        for s in sensors:
            sd = rows[s].std()
            s_mean[s].append(float(rows[s].mean()))
            s_std[s].append(float(max(1.0 if np.isnan(sd) else sd, _STD_FLOOR)))
    return s_mean, s_std


def _check_state(stats: dict[str, Any], sensors: list[str]) -> None:
    missing = [k for k in FEATURE_STATE_KEYS if k not in stats]
    if missing:
        raise ValueError(
            f"feature state lacks {missing}: it predates the params.yaml regime model "
            "(bundle_schema < 3) — retrain the bundle"
        )
    if list(stats["sensors"]) != list(sensors):
        raise ValueError(
            f"feature state was fitted on sensors {stats['sensors']}, called with {sensors}"
        )


def add_features(
    d: pd.DataFrame,
    sensors: list[str],
    dataset_name: str,
    window: int = WINDOW,
    stats: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """3 features per sensor: normalized value, rolling mean, rolling slope.

    stats=None on train (fits and returns stats); pass the returned stats on val/test.
    """
    d = d.sort_values(["unit", "cycle"]).copy()
    if stats is None:
        k = N_REGIMES[dataset_name]
        op_sc, km = _fit_regimes(d, k)
        stats = {"n_regimes": k, "sensors": list(sensors), "op_sc": op_sc, "km": km}
        labels = assign_regimes(d, stats)
        stats["s_mean"], stats["s_std"] = _fit_norm(d, labels, sensors, k)
    else:
        _check_state(stats, sensors)
        labels = assign_regimes(d, stats)
    for s in sensors:
        mu = np.asarray(stats["s_mean"][s], dtype=float)[labels]
        sig = np.asarray(stats["s_std"][s], dtype=float)[labels]
        d[f"{s}_n"] = (d[s].to_numpy(dtype=float) - mu) / sig
    g = d.groupby("unit")
    for s in sensors:
        sn = f"{s}_n"
        d[f"{s}_mean"] = g[sn].transform(lambda x: x.rolling(window, min_periods=1).mean())
        d[f"{s}_slope"] = g[sn].transform(
            lambda x: x.rolling(window, min_periods=2).apply(
                lambda w: np.polyfit(np.arange(len(w)), w, 1)[0], raw=True
            )
        )
    return d.fillna(0), stats
