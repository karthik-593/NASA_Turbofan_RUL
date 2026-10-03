"""Oracle: ``add_features`` exactly as it was before ``params.yaml`` (git 0acb61b).

Kept verbatim (only the module-level constants inlined) so the parity tests can prove the
single k-regime code path reproduces the former two-path implementation. Not used by
``src/``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

MULTI_REGIME = frozenset({"FD002", "FD004"})
REGIME_N_CLUSTERS = 6
REGIME_KMEANS_N_INIT = 10
REGIME_KMEANS_SEED = 0


def legacy_add_features(
    d: pd.DataFrame,
    sensors: list[str],
    dataset_name: str,
    window: int = 20,
    stats: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    # 3 features per sensor: normalized value, rolling mean, rolling slope.
    # FD002/FD004: per-regime normalization (KMeans k=6 on scaled op settings).
    # FD001/FD003: global z-score per sensor.
    # stats=None on train (fits and returns stats); pass returned stats on val/test.
    d = d.sort_values(["unit", "cycle"]).copy()
    multi = dataset_name in MULTI_REGIME
    if stats is None:
        if multi:
            op_sc = StandardScaler().fit(d[["op1", "op2", "op3"]])
            km = KMeans(
                n_clusters=REGIME_N_CLUSTERS,
                n_init=REGIME_KMEANS_N_INIT,
                random_state=REGIME_KMEANS_SEED,
            )
            km.fit(op_sc.transform(d[["op1", "op2", "op3"]]))
            d["_r"] = km.predict(op_sc.transform(d[["op1", "op2", "op3"]]))
            s_mean: dict[str, Any] = {s: d.groupby("_r")[s].mean().to_dict() for s in sensors}
            s_std: dict[str, Any] = {
                s: d.groupby("_r")[s].std().fillna(1).clip(lower=1e-9).to_dict() for s in sensors
            }
            stats = {"multi": True, "op_sc": op_sc, "km": km, "s_mean": s_mean, "s_std": s_std}
        else:
            s_mean = {s: float(d[s].mean()) for s in sensors}
            s_std = {s: float(max(d[s].std(), 1e-9)) for s in sensors}
            stats = {"multi": False, "s_mean": s_mean, "s_std": s_std}
    else:
        if stats["multi"]:
            d["_r"] = stats["km"].predict(stats["op_sc"].transform(d[["op1", "op2", "op3"]]))
    for s in sensors:
        if stats["multi"]:
            mu = d["_r"].map(stats["s_mean"][s]).fillna(0)
            sig = d["_r"].map(stats["s_std"][s]).fillna(1)
        else:
            mu, sig = stats["s_mean"][s], stats["s_std"][s]
        d[f"{s}_n"] = (d[s] - mu) / sig
    g = d.groupby("unit")
    for s in sensors:
        sn = f"{s}_n"
        d[f"{s}_mean"] = g[sn].transform(lambda x: x.rolling(window, min_periods=1).mean())
        d[f"{s}_slope"] = g[sn].transform(
            lambda x: x.rolling(window, min_periods=2).apply(
                lambda w: np.polyfit(np.arange(len(w)), w, 1)[0], raw=True
            )
        )
    if "_r" in d.columns:
        d = d.drop(columns=["_r"])
    return d.fillna(0), stats
