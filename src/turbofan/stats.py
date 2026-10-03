"""Engine-level uncertainty for reported numbers (CLAUDE.md rule 4).

Every number a report or a protocol-v2 evaluation publishes carries an uncertainty estimate
that resamples *engines*, never rows: rows within one engine are strongly dependent, so a
row-level bootstrap would understate the uncertainty. Two helpers live here:

- ``bootstrap_ci`` — percentile bootstrap CI of any statistic of per-engine values.
- ``ks_2samp_ci`` — two-sample Kolmogorov–Smirnov distance with a bootstrap CI that resamples
  engines on both sides; one side may contribute several values per engine (e.g. many
  truncation instances per validation engine).

Both take the generator, the number of resamples and the CI level explicitly — there are no
defaults (rule 5: parameters live with the caller, not in ``src/``). NaN input raises
(rule 8): missing per-engine values must be dropped and reported by the caller, not
silently skipped here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats as _sps

__all__ = ["bootstrap_ci", "ks_2samp_ci"]

Stat = Callable[..., Any]  # called as stat(array, axis=...) -> array, and stat(array) -> scalar


def _quantile_levels(ci_level: float) -> list[float]:
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1), got {ci_level}")
    alpha = 1 - ci_level
    return [alpha / 2, 1 - alpha / 2]


def _check_n_boot(n_boot: int) -> None:
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")


def bootstrap_ci(
    values: npt.ArrayLike,
    rng: np.random.Generator,
    *,
    n_boot: int,
    ci_level: float,
    stat: Stat = np.median,
) -> tuple[float, float, float, int]:
    """``(estimate, lo, hi, n)`` — percentile bootstrap of ``stat`` over per-engine values.

    ``values`` holds one number per engine; each resample draws ``n`` engines with
    replacement. ``stat`` must accept ``axis=1`` (applied to all resamples at once).
    """
    _check_n_boot(n_boot)
    q = _quantile_levels(ci_level)
    v = np.asarray(values, dtype=float)
    if v.ndim != 1 or len(v) == 0:
        raise ValueError(f"values must be a non-empty 1-D array, got shape {v.shape}")
    if np.isnan(v).any():
        raise ValueError("NaN in bootstrap input — drop and report them explicitly first")
    n = len(v)
    idx = rng.integers(0, n, size=(n_boot, n))
    bs = stat(v[idx], axis=1)
    lo, hi = np.quantile(bs, q)
    return float(stat(v)), float(lo), float(hi), n


def ks_2samp_ci(
    groups_a: Sequence[npt.ArrayLike],
    b: npt.ArrayLike,
    rng: np.random.Generator,
    *,
    n_boot: int,
    ci_level: float,
) -> tuple[float, float, float]:
    """``(statistic, lo, hi)`` — two-sample KS distance between pooled ``groups_a`` and ``b``.

    ``groups_a`` is one array per engine (any number of values each); ``b`` holds one value
    per engine. Each resample draws engines with replacement on both sides independently —
    all of a drawn engine's values from ``groups_a``. A percentile bootstrap of a distance
    is biased upward, so the point estimate can sit at or below the CI's lower end.
    """
    _check_n_boot(n_boot)
    q = _quantile_levels(ci_level)
    ga = [np.asarray(g, dtype=float) for g in groups_a]
    bb = np.asarray(b, dtype=float)
    if not ga or any(len(g) == 0 for g in ga) or len(bb) == 0:
        raise ValueError("ks_2samp_ci needs at least one non-empty group and a non-empty b")
    if any(np.isnan(g).any() for g in ga) or np.isnan(bb).any():
        raise ValueError("NaN in KS input — drop and report them explicitly first")
    est = _sps.ks_2samp(np.concatenate(ga), bb).statistic
    na, nb = len(ga), len(bb)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        ia = rng.integers(0, na, na)
        ib = rng.integers(0, nb, nb)
        bs[i] = _sps.ks_2samp(np.concatenate([ga[k] for k in ia]), bb[ib]).statistic
    lo, hi = np.quantile(bs, q)
    return float(est), float(lo), float(hi)
