"""Engine subpopulation labels (D35) — for CV stratification and per-group reporting ONLY.

Reproduces ``reports/data_audit.md`` §I's permutation-supported 2-way split: per training
engine, the sign of the Spearman correlation between each sensor's within-regime z-score and
the engine's (uncapped) time to failure, over its **entire run-to-failure trajectory**;
KMeans with k = 2 on those ±1 sign vectors; labels renumbered by cluster size (0 = largest).

The label uses the engine's future (its full trajectory up to failure), so it is never
observable at inference time and must never become a model input — directly or through a
derived feature. ``tests/test_subpopulation.py`` checks that no feature, model, training or
serving module imports this one.

Sensors: the dataset's ``params.yaml`` ``sensors.<ds>.candidate_pool`` (the non-constant
sensors of audit §E) minus any sensor whose correlation is undefined in some engine (constant
within that engine), as in audit §I.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.cluster import KMeans

from turbofan.config import N_REGIMES, PARAMS, SENSOR_POOL
from turbofan.features.engineering import _fit_regimes, assign_regimes

__all__ = ["sign_vectors", "subpopulation_labels"]

_SUB = PARAMS["subpopulation"]


def _within_regime_z(train: pd.DataFrame, dataset: str, sensors: list[str]) -> pd.DataFrame:
    """Within-regime z-scores as audit §E computed them: regime model fitted on all of
    ``train`` (the pipeline's KMeans settings), z = 0 where a sensor is single-valued within a
    regime (no variation there — the pipeline's normalizer would refuse that cell instead)."""
    d = train.sort_values(["unit", "cycle"])
    k = N_REGIMES[dataset]
    op_sc, km = _fit_regimes(d, k)
    reg = assign_regimes(d, {"op_sc": op_sc, "km": km})
    g = d[sensors].groupby(reg)
    single = g.transform("nunique") == 1
    z = ((d[sensors] - g.transform("mean")) / g.transform("std")).mask(single, 0.0)
    z["unit"] = d["unit"].to_numpy()
    z["ttf"] = (d.groupby("unit")["cycle"].transform("max") - d["cycle"]).to_numpy()
    return z


def sign_vectors(train: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Per engine (rows, indexed by unit) the sign of Spearman ρ(z_sensor, time to failure)
    for every candidate sensor whose ρ is defined in every engine (columns)."""
    pool = list(SENSOR_POOL[dataset])
    z = _within_regime_z(train, dataset, pool)
    ranks = z.groupby("unit")[[*pool, "ttf"]].rank(method="average")
    ranks["unit"] = z["unit"]
    # a sensor constant within an engine has no defined ρ there: NaN (dropped below), and
    # numpy's divide-by-zero warning for it is expected
    with np.errstate(invalid="ignore", divide="ignore"):
        rho = pd.DataFrame({u: g[pool].corrwith(g["ttf"]) for u, g in ranks.groupby("unit")}).T
    defined = [s for s in pool if rho[s].notna().all()]
    signs = pd.DataFrame(np.sign(rho[defined].to_numpy(float)), index=rho.index, columns=defined)
    if (signs == 0).any().any():
        raise ValueError(f"{dataset}: a sensor has ρ exactly 0 in some engine — sign undefined")
    signs.index.name = "unit"
    return signs


def subpopulation_labels(train: pd.DataFrame, dataset: str) -> pd.Series[int]:
    """Subpopulation label per training engine (index = unit), 0 = the larger group.

    ``train`` must hold complete run-to-failure trajectories (the training file): the label
    is a property of the whole trajectory, which is why it can stratify folds but never feed
    a model.
    """
    X = sign_vectors(train, dataset)
    km = KMeans(
        n_clusters=_SUB["k"], n_init=_SUB["kmeans_n_init"], random_state=_SUB["kmeans_random_state"]
    ).fit(X.to_numpy())
    raw: npt.NDArray[np.intp] = km.labels_
    order = np.argsort(-np.bincount(raw), kind="stable")
    labels = np.argsort(order)[raw]
    return pd.Series(labels.astype(int), index=X.index, name="subpopulation")
