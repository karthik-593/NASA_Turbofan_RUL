"""Training operating envelope, checked at serving (D41).

The region of input space the model was trained on, learned from the training rows by
``add_features`` and stored in the feature state:

- op settings: each regime's maximum distance from its KMeans centroid in standardized
  op-setting space. A cycle further from its nearest centroid than ``(1 + margin) x`` that
  radius is from an operating regime the model never saw (KMeans would otherwise silently
  assign it to the nearest one).
- sensors: each sensor's [min, max] within each regime, widened by ``margin x`` the range.

``margin`` is ``params.yaml`` ``validation.envelope_margin``. ``check_envelope`` returns
human-readable reasons; the API turns a non-empty list into HTTP 422.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from turbofan.config import OP_COLS

__all__ = ["fit_envelope", "check_envelope"]


def _centroid_dist(d: pd.DataFrame, stats: dict[str, Any]) -> npt.NDArray[np.float64]:
    z = stats["op_sc"].transform(d[OP_COLS])
    labels = stats["km"].predict(z)
    return np.asarray(
        np.linalg.norm(z - stats["km"].cluster_centers_[labels], axis=1), dtype=np.float64
    )


def fit_envelope(
    d: pd.DataFrame, labels: npt.NDArray[np.intp], stats: dict[str, Any], sensors: list[str]
) -> dict[str, Any]:
    """Per-regime op-setting radius and sensor [min, max] from the training rows."""
    dist = _centroid_dist(d, stats)
    k = stats["n_regimes"]
    return {
        "op_radius": [float(dist[labels == r].max()) for r in range(k)],
        "sensor_min": {s: [float(d.loc[labels == r, s].min()) for r in range(k)] for s in sensors},
        "sensor_max": {s: [float(d.loc[labels == r, s].max()) for r in range(k)] for s in sensors},
    }


def check_envelope(d: pd.DataFrame, stats: dict[str, Any], margin: float) -> list[str]:
    """Reasons ``d``'s rows fall outside the training envelope (empty list = inside).

    ``d`` is a raw frame (op settings + the feature state's sensors), one row per cycle.
    """
    env = stats["envelope"]
    z = stats["op_sc"].transform(d[OP_COLS])
    labels = stats["km"].predict(z)
    dist = np.linalg.norm(z - stats["km"].cluster_centers_[labels], axis=1)
    reasons: list[str] = []
    cycles = d["cycle"].to_numpy() if "cycle" in d.columns else np.arange(1, len(d) + 1)
    radius = np.asarray(env["op_radius"])[labels] * (1 + margin)
    for i in np.flatnonzero(dist > radius):
        reasons.append(
            f"cycle {cycles[i]}: op settings {d[OP_COLS].iloc[i].tolist()} are outside every "
            f"training operating regime (distance {dist[i]:.3g} > {radius[i]:.3g})"
        )
    for s in stats["sensors"]:
        lo = np.asarray(env["sensor_min"][s])[labels]
        hi = np.asarray(env["sensor_max"][s])[labels]
        pad = (hi - lo) * margin
        v = d[s].to_numpy(dtype=float)
        for i in np.flatnonzero((v < lo - pad) | (v > hi + pad)):
            reasons.append(
                f"cycle {cycles[i]}: {s} = {v[i]:.6g} outside training range "
                f"[{lo[i] - pad[i]:.6g}, {hi[i] + pad[i]:.6g}] for regime {labels[i]}"
            )
    return reasons
