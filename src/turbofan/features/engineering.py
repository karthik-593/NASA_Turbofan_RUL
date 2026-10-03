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

# A sample std needs two rows; a regime with fewer cannot be normalized (D06, rule 8).
_MIN_REGIME_ROWS = 2


class FeatureError(ValueError):
    """Input or fitted state that add_features refuses to paper over (rule 8)."""


def _check_input(d: pd.DataFrame, sensors: list[str]) -> None:
    cols = ["unit", "cycle", *OP_COLS, *sensors]
    missing = [c for c in cols if c not in d.columns]
    if missing:
        raise FeatureError(f"input lacks columns {missing}")
    if d.empty:
        raise FeatureError("input has no rows")
    nan = d[cols].isna().sum()
    if nan.any():
        raise FeatureError(f"NaN in input columns: {nan[nan > 0].to_dict()}")


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
    aggregations differ from them in the last bits. A regime with < 2 training rows, or a
    sensor that is constant within a regime, raises — the former fallbacks (std ``fillna(1)``
    and a 1e-9 floor) never fired on the train/val split (audit §D, D06) and would only have
    hidden a bad sensor list or regime count.
    """
    s_mean: dict[str, list[float]] = {s: [] for s in sensors}
    s_std: dict[str, list[float]] = {s: [] for s in sensors}
    for r in range(k):
        rows = d.loc[labels == r]
        if len(rows) < _MIN_REGIME_ROWS:
            raise FeatureError(
                f"regime {r} has {len(rows)} training row(s); need >= {_MIN_REGIME_ROWS} — "
                "n_regimes is too large for this data"
            )
        for s in sensors:
            sd = float(rows[s].std())
            if sd == 0.0:
                raise FeatureError(
                    f"sensor {s} is constant within regime {r} on the training data — "
                    "remove it from params.yaml sensors.use"
                )
            s_mean[s].append(float(rows[s].mean()))
            s_std[s].append(sd)
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
    k = stats["n_regimes"]
    if any(len(stats[key][s]) != k for key in ("s_mean", "s_std") for s in sensors):
        raise FeatureError(f"feature state has per-regime stats inconsistent with k = {k}")


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
    _check_input(d, sensors)
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
        unseen = sorted(set(np.unique(labels).tolist()) - set(range(stats["n_regimes"])))
        if unseen:
            raise FeatureError(f"regime label(s) {unseen} have no fitted normalization")
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
    # The one legitimate NaN: the slope needs two points, so it is undefined at each engine's
    # first cycle. Defined explicitly as 0 (no trend observed yet), the value the former
    # blanket fillna(0) gave it. Any other NaN is a bug and raises.
    first_cycle = ~d["unit"].duplicated().to_numpy()
    slope_cols = [f"{s}_slope" for s in sensors]
    d.loc[first_cycle, slope_cols] = 0.0
    out_cols = [f"{s}_{kind}" for kind in ("n", "mean", "slope") for s in sensors]
    nan = d[out_cols].isna().sum()
    if nan.any():
        raise FeatureError(f"unexpected NaN in features: {nan[nan > 0].to_dict()}")
    return d, stats
