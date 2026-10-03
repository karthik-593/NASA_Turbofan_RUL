"""Paired comparison of two CV candidates (D46) — replaces ``comparison.contender_gap``.

Two candidates are compared only on **identical** held-out points: same dataset, same
engines, same cycles, same repeat x fold split, same seeds. Two complementary readings:

- **Paired engine bootstrap** of the metric difference (A - B): every replicate draws one set
  of engines and evaluates both candidates on it, so engine-to-engine difficulty cancels.
  Also returned: the per-engine differences behind it.
- **Wilcoxon signed-rank** over the fold x seed results: the metric of A and of B on each
  (repeat, fold, seed) held-out set, as paired samples.

Never across datasets (FD00x results are never pooled or averaged). Candidates trained with
different ``rul_cap`` are only comparable on cap-invariant metrics (D01): critical-bucket
RMSE (truth < 25 is never capped) and RMSE against uncapped truth; anything else raises
``CapComparisonError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from turbofan.evaluation.cv_metrics import (
    CI_LEVEL,
    HEADLINE,
    N_BOOT,
    engine_aggregates,
    engine_weights,
    metrics_from_weights,
)

__all__ = ["CVResult", "CapComparisonError", "CAP_INVARIANT", "paired_compare"]

KEYS = ["unit", "cycle", "repeat", "fold", "seed"]
# (metric, truth) pairs that do not depend on the training cap (D01)
CAP_INVARIANT = frozenset({(HEADLINE, "uncapped"), (HEADLINE, "capped"), ("rmse", "uncapped")})


class CapComparisonError(ValueError):
    """A capped-truth (or cap-dependent) metric compared across different rul_cap values."""


@dataclass(frozen=True)
class CVResult:
    """One candidate's held-out predictions on one dataset (one view)."""

    dataset: str
    name: str
    rul_cap: float
    points: pd.DataFrame  # KEYS + rul_true, pred


def _paired_frame(a: CVResult, b: CVResult) -> pd.DataFrame:
    for r in (a, b):
        missing = [c for c in [*KEYS, "rul_true", "pred"] if c not in r.points.columns]
        if missing:
            raise ValueError(f"{r.name}: points lack {missing}")
        if r.points.duplicated(KEYS).any():
            raise ValueError(f"{r.name}: duplicate prediction keys")
    m = a.points[[*KEYS, "rul_true", "pred"]].merge(
        b.points[[*KEYS, "rul_true", "pred"]],
        on=KEYS,
        how="outer",
        suffixes=("_a", "_b"),
        indicator=True,
    )
    if (m["_merge"] != "both").any():
        n = int((m["_merge"] != "both").sum())
        raise ValueError(
            f"not paired: {n} prediction keys are present for only one candidate — compare "
            "candidates on the same engines, cycles, folds and seeds"
        )
    if not np.array_equal(m["rul_true_a"].to_numpy(), m["rul_true_b"].to_numpy()):
        raise ValueError("not paired: true RUL differs between candidates on the same key")
    return m.drop(columns="_merge")


def _side(m: pd.DataFrame, s: str) -> pd.DataFrame:
    return pd.DataFrame({"unit": m["unit"], "rul_true": m["rul_true_a"], "pred": m[f"pred_{s}"]})


def paired_compare(
    a: CVResult,
    b: CVResult,
    rng: np.random.Generator,
    metric: str = HEADLINE,
    truth: str = "uncapped",
    n_boot: int = N_BOOT,
    ci_level: float = CI_LEVEL,
) -> dict[str, Any]:
    """Paired comparison of ``metric`` (A - B; negative = A has lower error)."""
    if a.dataset != b.dataset:
        raise ValueError(
            f"comparisons are per dataset — got {a.dataset} vs {b.dataset}; FD00x results "
            "are never pooled or averaged"
        )
    if a.rul_cap != b.rul_cap and (metric, truth) not in CAP_INVARIANT:
        raise CapComparisonError(
            f"{a.name} (rul_cap {a.rul_cap}) vs {b.name} (rul_cap {b.rul_cap}): "
            f"{metric} against {truth} truth depends on the cap; use one of "
            f"{sorted(CAP_INVARIANT)}"
        )
    m = _paired_frame(a, b)
    # capped truth with differing caps is refused above, so one cap serves both sides
    agg_a = engine_aggregates(_side(m, "a"), truth, a.rul_cap)
    agg_b = engine_aggregates(_side(m, "b"), truth, a.rul_cap)
    ones = np.ones((1, len(agg_a)))
    est_a = metrics_from_weights(agg_a, ones)[metric][0]
    est_b = metrics_from_weights(agg_b, ones)[metric][0]
    w = engine_weights(len(agg_a), n_boot, rng)
    diff = metrics_from_weights(agg_a, w)[metric] - metrics_from_weights(agg_b, w)[metric]
    diff = diff[np.isfinite(diff)]
    q = [(1 - ci_level) / 2, 1 - (1 - ci_level) / 2]
    lo, hi = np.quantile(diff, q)

    eye = np.eye(len(agg_a))
    per_engine = pd.DataFrame(
        {
            "a": metrics_from_weights(agg_a, eye)[metric],
            "b": metrics_from_weights(agg_b, eye)[metric],
        },
        index=agg_a.index,
    )
    per_engine["diff"] = per_engine["a"] - per_engine["b"]

    by_split = []
    for key, g in m.groupby(["repeat", "fold", "seed"]):
        ga = engine_aggregates(_side(g, "a"), truth, a.rul_cap)
        gb = engine_aggregates(_side(g, "b"), truth, a.rul_cap)
        o = np.ones((1, len(ga)))
        by_split.append(
            {
                "split": key,
                "a": metrics_from_weights(ga, o)[metric][0],
                "b": metrics_from_weights(gb, o)[metric][0],
            }
        )
    splits = pd.DataFrame(by_split)
    d = (splits["a"] - splits["b"]).to_numpy()
    d = d[np.isfinite(d)]
    if len(d) and np.any(d != 0):
        wil = sps.wilcoxon(d)
        w_stat, w_p = float(wil.statistic), float(wil.pvalue)
    else:
        w_stat, w_p = float("nan"), float("nan")  # no split differs: no test to run

    defined = per_engine["diff"].dropna()
    return {
        "dataset": a.dataset,
        "a": a.name,
        "b": b.name,
        "metric": metric,
        "truth": truth,
        "estimate_a": float(est_a),
        "estimate_b": float(est_b),
        "diff": float(est_a - est_b),
        "diff_ci_lo": float(lo),
        "diff_ci_hi": float(hi),
        "n_engines": len(agg_a),
        "n_points": len(m),
        "per_engine_diff_median": float(defined.median()) if len(defined) else float("nan"),
        "share_engines_a_better": float((defined < 0).mean()) if len(defined) else float("nan"),
        "n_engines_with_metric": len(defined),
        "wilcoxon_n_splits": len(d),
        "wilcoxon_stat": w_stat,
        "wilcoxon_p": w_p,
        "per_engine": per_engine,
        "per_split": splits,
    }
