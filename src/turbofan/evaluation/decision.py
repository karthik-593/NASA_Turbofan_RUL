"""Decision curves for the maintenance rule "remove when predicted RUL <= T" (D46).

Each held-out trajectory (per repeat and seed) is run forward with the rule: the engine is
removed at the first cycle whose prediction is <= T. Its **lead time** is the true RUL at that
cycle — the life left when it came off — which is also the **life wasted** by the removal. A
trajectory the rule never triggers on fails in service.

Per threshold T: share of failures caught with lead time >= L (for each L), share failing in
service, and mean wasted life over removed engines. CIs resample engines (all of an engine's
trajectories together), as for every protocol-v2 metric.

Models are compared at **matched operating points** (``matched_budget``): the same T removes
at different wasted life for different models, so comparing at equal T is not like-for-like.
Instead, for a wasted-life budget B, each model's caught share (lead >= L) is read off its own
curve where mean wasted life equals B. Mean wasted life is not monotone in T in a sample (the
set of removed engines changes with T), so its running maximum over increasing T is used and B is
matched at the lowest T that reaches it, by linear interpolation between grid points. A budget
outside a curve's range is undefined (NaN) for that curve; the share of bootstrap replicates
in which it is defined is reported.
"""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt
import pandas as pd

from turbofan.config import PARAMS
from turbofan.evaluation.cv_metrics import CI_LEVEL, N_BOOT, engine_weights

__all__ = [
    "THRESHOLDS",
    "LEAD_TIMES",
    "MATCHED_LEAD_TIME",
    "WASTED_LIFE_BUDGETS",
    "removal_lead_times",
    "decision_curve",
    "engine_sums",
    "caught_at_budgets",
    "require_increasing",
    "caught_at_budget",
    "matched_budget",
]

_D = PARAMS["decision_curve"]
THRESHOLDS: list[float] = [
    float(t) for t in np.arange(_D["t_min"], _D["t_max"] + _D["t_step"] / 2, _D["t_step"])
]
LEAD_TIMES: list[int] = list(_D["lead_times"])
MATCHED_LEAD_TIME: int = int(_D["matched_lead_time"])
WASTED_LIFE_BUDGETS: list[float] = [float(b) for b in _D["wasted_life_budgets"]]

F64 = npt.NDArray[np.float64]


def removal_lead_times(points: pd.DataFrame, thresholds: list[float] = THRESHOLDS) -> pd.DataFrame:
    """One row per trajectory (unit, repeat, seed): lead time at removal for each threshold
    (column = T; NaN = never removed, i.e. failed in service)."""
    keys = [k for k in ("unit", "repeat", "seed") if k in points.columns]
    t = np.asarray(thresholds, dtype=float)
    rows = []
    for key, g in points.sort_values([*keys, "cycle"]).groupby(keys):
        run_min = np.minimum.accumulate(g["pred"].to_numpy(float))  # non-increasing
        rul = g["rul_true"].to_numpy(float)
        # first index with run_min <= T: run_min is non-increasing, so search its negation
        first = np.searchsorted(-run_min, -t, side="left")
        lead = np.where(first < len(rul), rul[np.minimum(first, len(rul) - 1)], np.nan)
        rows.append([*(key if isinstance(key, tuple) else (key,)), *lead])
    out = pd.DataFrame(rows, columns=[*keys, *t])
    out[keys] = out[keys].astype(int)
    return out


def engine_sums(
    points: pd.DataFrame, thresholds: list[float], lead_times: list[int]
) -> tuple[npt.NDArray[np.int64], dict[str, F64]]:
    """Per-engine sufficient statistics of the rule at each T: ``(engines, sums)`` with each
    array ``(n_engines, n_T)``; engines sorted by unit id. Trajectories of one engine are
    summed together so that bootstrap weights resample engines."""
    lead = removal_lead_times(points, thresholds)
    lt = lead[list(map(float, thresholds))].to_numpy(float)  # (n_traj, n_T)
    uniq, inv = np.unique(lead["unit"].to_numpy(), return_inverse=True)

    def per_engine(x: F64) -> F64:
        out: F64 = np.zeros((len(uniq), x.shape[1]))
        np.add.at(out, inv, x)
        return out

    removed = ~np.isnan(lt)
    sums = {
        "n": per_engine(np.ones_like(lt)),
        "failed": per_engine((~removed).astype(float)),
        "removed": per_engine(removed.astype(float)),
        "wasted": per_engine(np.where(removed, lt, 0.0)),
        **{f"caught_{L}": per_engine((removed & (lt >= L)).astype(float)) for L in lead_times},
    }
    return uniq, sums


def require_increasing(thresholds: list[float]) -> None:
    if any(b <= a for a, b in zip(thresholds, thresholds[1:], strict=False)):
        raise ValueError("thresholds must be strictly increasing for operating-point matching")


def caught_at_budget(caught: F64, wasted: F64, budget: float) -> float:
    """Caught share at the T where mean wasted life reaches ``budget`` (both arrays indexed by
    increasing T). NaN when the curve never reaches the budget."""
    ok = np.isfinite(wasted) & np.isfinite(caught)
    if not ok.any():
        return float("nan")
    w = np.maximum.accumulate(wasted[ok])
    if not w[0] <= budget <= w[-1]:
        return float("nan")
    return float(np.interp(budget, w, caught[ok]))


def caught_at_budgets(sums: dict[str, F64], w: F64, budgets: list[float], lead_time: int) -> F64:
    """``(r, n_budgets)``: caught share (lead >= L) at each budget, per weight row of ``w``."""
    with np.errstate(invalid="ignore", divide="ignore"):
        caught = 100 * (w @ sums[f"caught_{lead_time}"]) / (w @ sums["n"])
        wasted = (w @ sums["wasted"]) / (w @ sums["removed"])
    out: F64 = np.array(
        [
            [caught_at_budget(c, ws, b) for b in budgets]
            for c, ws in zip(caught, wasted, strict=True)
        ]
    )
    return out


def matched_budget(
    points: pd.DataFrame,
    rng: np.random.Generator,
    budgets: list[float] = WASTED_LIFE_BUDGETS,
    lead_time: int = MATCHED_LEAD_TIME,
    thresholds: list[float] = THRESHOLDS,
    n_boot: int = N_BOOT,
    ci_level: float = CI_LEVEL,
) -> pd.DataFrame:
    """One row per budget: caught share (lead >= ``lead_time``) at that mean wasted life, with
    engine-bootstrap CI, ``n_engines`` and the share of replicates in which it is defined."""
    require_increasing(thresholds)
    uniq, sums = engine_sums(points, thresholds, [lead_time])
    est = caught_at_budgets(sums, np.ones((1, len(uniq))), budgets, lead_time)[0]
    boot = caught_at_budgets(sums, engine_weights(len(uniq), n_boot, rng), budgets, lead_time)
    a = (1 - ci_level) / 2
    rows = []
    for j, b in enumerate(budgets):
        d = boot[:, j][np.isfinite(boot[:, j])]
        lo, hi = np.quantile(d, [a, 1 - a]) if len(d) else (np.nan, np.nan)
        rows.append(
            {
                "budget": b,
                "estimate": float(est[j]),
                "ci_lo": float(lo),
                "ci_hi": float(hi),
                "n_engines": len(uniq),
                "boot_defined_share": float(np.isfinite(boot[:, j]).mean()),
            }
        )
    return pd.DataFrame(rows)


def decision_curve(
    points: pd.DataFrame,
    rng: np.random.Generator,
    thresholds: list[float] = THRESHOLDS,
    lead_times: list[int] = LEAD_TIMES,
    n_boot: int = N_BOOT,
    ci_level: float = CI_LEVEL,
) -> pd.DataFrame:
    """Tidy curve: one row per (T, quantity) with estimate and engine-bootstrap CI.

    Quantities: ``caught_lead_ge_<L>_pct``, ``failed_in_service_pct``, ``mean_wasted_life``.
    """
    uniq, sums = engine_sums(points, thresholds, lead_times)
    n_traj = int(sums["n"][:, 0].sum())

    def quantities(w: F64) -> dict[str, F64]:  # w: (r, n_engines) -> each (r, n_T)
        n = w @ sums["n"]
        with np.errstate(invalid="ignore", divide="ignore"):
            q = {f"caught_lead_ge_{L}_pct": 100 * (w @ sums[f"caught_{L}"]) / n for L in lead_times}
            q["failed_in_service_pct"] = 100 * (w @ sums["failed"]) / n
            q["mean_wasted_life"] = (w @ sums["wasted"]) / (w @ sums["removed"])
        return q

    est = quantities(np.ones((1, len(uniq))))
    boot = quantities(engine_weights(len(uniq), n_boot, rng))
    a = (1 - ci_level) / 2
    rows = []
    for name, v in est.items():
        # mean wasted life is undefined (NaN) at a T where no engine is ever removed; its CI
        # is then NaN too, which is the honest value
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", "All-NaN slice", RuntimeWarning)
            lo = np.nanquantile(boot[name], a, axis=0)
            hi = np.nanquantile(boot[name], 1 - a, axis=0)
        for j, T in enumerate(thresholds):
            rows.append(
                {
                    "threshold": T,
                    "quantity": name,
                    "estimate": float(v[0, j]),
                    "ci_lo": float(lo[j]),
                    "ci_hi": float(hi[j]),
                    "n_engines": len(uniq),
                    "n_trajectories": n_traj,
                }
            )
    return pd.DataFrame(rows)
