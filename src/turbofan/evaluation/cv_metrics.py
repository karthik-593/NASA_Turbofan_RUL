"""Protocol-v2 metrics with engine-level bootstrap CIs (D46, rule 4).

Points within an engine are strongly correlated, so uncertainty comes from resampling
**engines**: each bootstrap replicate draws engines with replacement and pools all residuals
of the drawn engines (an engine drawn twice counts twice). Every metric is a ratio of
per-engine sums, so a replicate is a weighted sum over engines with multinomial weights —
exact, and vectorized over all replicates.

Residual e = prediction - truth (positive = late, the dangerous side). Metrics, each against
**uncapped** truth (true cycles to failure) and **capped** truth (min(truth, rul_cap), what
the model was trained to predict); buckets are assigned by the truth used:

- ``rmse``, ``mae``, ``late_pct``, ``mean_signed_error``;
- ``nasa_mean_per_engine``: the C-MAPSS score of each point, averaged within each engine,
  then over engines (every engine weighs the same, however long its trajectory);
- per bucket (``config.MAINTENANCE_BUCKETS``): ``<bucket>_rmse``, ``<bucket>_late_pct``,
  ``<bucket>_mean_signed_error``, with ``n_engines`` / ``n_points`` that fall in the bucket.

Headline (D18): ``critical_rmse`` in the deployment view — identical under both truths, since
truth below 25 cycles is never capped.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd

from turbofan.config import MAINTENANCE_BUCKETS, PARAMS

__all__ = [
    "HEADLINE",
    "TRUTHS",
    "engine_weights",
    "engine_aggregates",
    "metrics_from_weights",
    "metric_table",
]

HEADLINE = "critical_rmse"
TRUTHS = ("uncapped", "capped")
N_BOOT: int = PARAMS["bootstrap"]["n_boot"]
CI_LEVEL: float = PARAMS["bootstrap"]["ci_level"]

F64 = npt.NDArray[np.float64]


def _nasa(e: F64) -> F64:
    out: F64 = np.where(e < 0, np.expm1(-e / 13.0), np.expm1(e / 10.0))
    return out


def engine_aggregates(points: pd.DataFrame, truth: str, rul_cap: float) -> pd.DataFrame:
    """Per-engine sufficient statistics (one row per unit) for ``truth`` in TRUTHS.

    ``points``: one row per prediction with columns ``unit``, ``rul_true``, ``pred``.
    """
    if truth not in TRUTHS:
        raise ValueError(f"truth must be one of {TRUTHS}, got {truth!r}")
    y = points["rul_true"].to_numpy(float)
    if truth == "capped":
        y = np.minimum(y, rul_cap)
    e = points["pred"].to_numpy(float) - y
    if not np.isfinite(e).all():
        raise ValueError("non-finite residual — predictions or truth contain NaN/inf")
    cols: dict[str, F64] = {
        "n": np.ones_like(e),
        "se": e,
        "se2": e**2,
        "sabs": np.abs(e),
        "late": (e > 0).astype(float),
        "nasa": _nasa(e),
    }
    for name, lo, hi in MAINTENANCE_BUCKETS:
        m = ((y >= lo) & (y < hi)).astype(float)
        cols[f"{name}:n"] = m
        cols[f"{name}:se"] = m * e
        cols[f"{name}:se2"] = m * e**2
        cols[f"{name}:late"] = m * (e > 0)
    agg: pd.DataFrame = pd.DataFrame(cols).groupby(points["unit"].to_numpy()).sum()
    return agg


def engine_weights(n_engines: int, n_boot: int, rng: np.random.Generator) -> F64:
    """``(n_boot, n_engines)`` multiplicities: each row is one resample of engines with
    replacement (row sums = n_engines)."""
    w: F64 = rng.multinomial(n_engines, np.full(n_engines, 1.0 / n_engines), size=n_boot).astype(
        float
    )
    return w


def metrics_from_weights(agg: pd.DataFrame, w: F64) -> dict[str, F64]:
    """Every metric for each weight row ``w`` (shape ``(r, n_engines)``) -> arrays of length r.

    A metric whose denominator is zero in a replicate (e.g. an empty bucket) is NaN there.
    """
    a = {c: agg[c].to_numpy(float) for c in agg.columns}
    tot = w @ a["n"]
    with np.errstate(invalid="ignore", divide="ignore"):
        out: dict[str, F64] = {
            "rmse": np.sqrt(w @ a["se2"] / tot),
            "mae": w @ a["sabs"] / tot,
            "late_pct": 100.0 * (w @ a["late"]) / tot,
            "mean_signed_error": w @ a["se"] / tot,
            "nasa_mean_per_engine": (w @ (a["nasa"] / a["n"])) / w.sum(axis=1),
            "n_points": tot,
            "n_engines": w @ np.ones(len(agg)),
        }
        for name, _lo, _hi in MAINTENANCE_BUCKETS:
            nb = w @ a[f"{name}:n"]
            out[f"{name}_rmse"] = np.sqrt(w @ a[f"{name}:se2"] / nb)
            out[f"{name}_late_pct"] = 100.0 * (w @ a[f"{name}:late"]) / nb
            out[f"{name}_mean_signed_error"] = w @ a[f"{name}:se"] / nb
            out[f"{name}_n_points"] = nb
            out[f"{name}_n_engines"] = w @ (a[f"{name}:n"] > 0).astype(float)
    return out


def metric_table(
    points: pd.DataFrame,
    rul_cap: float,
    rng: np.random.Generator,
    n_boot: int = N_BOOT,
    ci_level: float = CI_LEVEL,
) -> pd.DataFrame:
    """Tidy table: one row per (truth, metric) with estimate, CI and n.

    ``n_engines`` / ``n_points`` columns are the counts behind each estimate (bucket counts
    for bucket metrics). Counts carry no CI.
    """
    if not 0 < ci_level < 1:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    q = [(1 - ci_level) / 2, 1 - (1 - ci_level) / 2]
    rows = []
    for truth in TRUTHS:
        agg = engine_aggregates(points, truth, rul_cap)
        est = metrics_from_weights(agg, np.ones((1, len(agg))))
        boot = metrics_from_weights(agg, engine_weights(len(agg), n_boot, rng))
        for m, v in est.items():
            if m.endswith(("n_points", "n_engines")):
                continue
            bucket = m.split("_", 1)[0] if m.split("_", 1)[0] in _BUCKET_NAMES else None
            n_eng = est[f"{bucket}_n_engines"][0] if bucket else est["n_engines"][0]
            n_pts = est[f"{bucket}_n_points"][0] if bucket else est["n_points"][0]
            bs = boot[m]
            defined = np.isfinite(bs)
            lo, hi = np.quantile(bs[defined], q) if defined.any() else (np.nan, np.nan)
            rows.append(
                {
                    "truth": truth,
                    "metric": m,
                    "estimate": float(v[0]),
                    "ci_lo": float(lo),
                    "ci_hi": float(hi),
                    "n_engines": int(n_eng),
                    "n_points": int(n_pts),
                    "boot_defined_share": float(defined.mean()),
                }
            )
    return pd.DataFrame(rows)


_BUCKET_NAMES = {name for name, _lo, _hi in MAINTENANCE_BUCKETS}
