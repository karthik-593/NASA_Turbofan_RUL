"""Paired comparison of two CV candidates (D46) — replaces ``comparison.contender_gap``.

Two candidates are compared only on **identical** held-out points: same dataset, same
engines, same cycles, same repeat x fold split, same seeds. Two complementary readings:

- **Paired engine bootstrap** of the metric difference (A - B): every replicate draws one set
  of engines and evaluates both candidates on it, so engine-to-engine difficulty cancels.
  Also returned: the per-engine differences behind it.
- **Wilcoxon signed-rank** over the fold x seed results: the metric of A and of B on each
  (repeat, fold, seed) held-out set, as paired samples.

A third, decision-level reading, ``matched_budget_compare``: the caught share (lead >= L) of
each candidate at the *same mean wasted life*, not at the same removal threshold T — the same T
removes at different wasted life for different models, so a same-T comparison is not
like-for-like. Paired engine bootstrap, the curves re-matched in every replicate.

Never across datasets (FD00x results are never pooled or averaged). Candidates trained with
different ``rul_cap`` are only comparable on cap-invariant metrics (D01): critical-bucket
RMSE (truth < 25 is never capped), RMSE against uncapped truth, and any bucket metric against
uncapped truth whose bucket lies wholly below both caps (e.g. urgent RMSE / late % at caps >= 50:
neither model's target is capped there); anything else raises ``CapComparisonError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats as sps

from turbofan.config import MAINTENANCE_BUCKETS
from turbofan.evaluation.cv_metrics import (
    CI_LEVEL,
    HEADLINE,
    N_BOOT,
    engine_aggregates,
    engine_weights,
    metrics_from_weights,
)
from turbofan.evaluation.decision import (
    MATCHED_LEAD_TIME,
    THRESHOLDS,
    WASTED_LIFE_BUDGETS,
    caught_at_budgets,
    engine_sums,
    require_increasing,
)

__all__ = [
    "CVResult",
    "CapComparisonError",
    "CAP_INVARIANT",
    "cap_invariant",
    "paired_compare",
    "matched_budget_compare",
]

KEYS = ["unit", "cycle", "repeat", "fold", "seed"]
# (metric, truth) pairs that do not depend on the training cap (D01)
CAP_INVARIANT = frozenset({(HEADLINE, "uncapped"), (HEADLINE, "capped"), ("rmse", "uncapped")})


def cap_invariant(metric: str, truth: str, caps: tuple[float, float]) -> bool:
    """Whether ``metric`` against ``truth`` means the same for models trained at ``caps``."""
    if (metric, truth) in CAP_INVARIANT:
        return True
    if truth != "uncapped":
        return False
    for name, _lo, hi in MAINTENANCE_BUCKETS:
        if metric.startswith(f"{name}_") and hi <= min(caps):
            return True
    return False


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
    if a.rul_cap != b.rul_cap and not cap_invariant(metric, truth, (a.rul_cap, b.rul_cap)):
        raise CapComparisonError(
            f"{a.name} (rul_cap {a.rul_cap}) vs {b.name} (rul_cap {b.rul_cap}): "
            f"{metric} against {truth} truth depends on the cap; use one of "
            f"{sorted(CAP_INVARIANT)} or a bucket metric against uncapped truth whose bucket "
            "lies below both caps"
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


def matched_budget_compare(
    a: CVResult,
    b: CVResult,
    rng: np.random.Generator,
    budgets: list[float] = WASTED_LIFE_BUDGETS,
    lead_time: int = MATCHED_LEAD_TIME,
    thresholds: list[float] = THRESHOLDS,
    n_boot: int = N_BOOT,
    ci_level: float = CI_LEVEL,
) -> pd.DataFrame:
    """Caught share (lead >= ``lead_time``) of A and B at equal mean wasted life, one row per
    budget: each side's estimate with CI, the paired difference A - B (positive = A catches
    more) with CI, ``n_engines`` and the share of replicates where both sides are defined.

    Every replicate draws one set of engines, re-derives both curves on it, matches each at the
    budget, and differences them. Datasets and pairing are checked as in ``paired_compare``.
    """
    if a.dataset != b.dataset:
        raise ValueError(
            f"comparisons are per dataset — got {a.dataset} vs {b.dataset}; FD00x results "
            "are never pooled or averaged"
        )
    require_increasing(thresholds)
    m = _paired_frame(a, b)
    side = {
        s: m[[*KEYS, "rul_true_a"]]
        .rename(columns={"rul_true_a": "rul_true"})
        .assign(pred=m[f"pred_{s}"])
        for s in ("a", "b")
    }
    sums = {}
    engines = {}
    for s, pts in side.items():
        engines[s], sums[s] = engine_sums(pts, thresholds, [lead_time])
    if not np.array_equal(engines["a"], engines["b"]):
        raise ValueError("not paired: the candidates cover different engines")
    n_eng = len(engines["a"])
    est = {s: caught_at_budgets(sums[s], np.ones((1, n_eng)), budgets, lead_time)[0] for s in sums}
    w = engine_weights(n_eng, n_boot, rng)
    boot = {s: caught_at_budgets(sums[s], w, budgets, lead_time) for s in sums}
    q = [(1 - ci_level) / 2, 1 - (1 - ci_level) / 2]

    def ci(x: npt.NDArray[np.float64]) -> tuple[float, float]:
        x = x[np.isfinite(x)]
        lo, hi = np.quantile(x, q) if len(x) else (np.nan, np.nan)
        return float(lo), float(hi)

    rows = []
    for j, bud in enumerate(budgets):
        diff = boot["a"][:, j] - boot["b"][:, j]
        row: dict[str, Any] = {"budget": bud}
        for s, name in (("a", a.name), ("b", b.name)):
            lo, hi = ci(boot[s][:, j])
            row[f"{s}"] = name
            row[f"estimate_{s}"] = float(est[s][j])
            row[f"ci_lo_{s}"], row[f"ci_hi_{s}"] = lo, hi
        d_lo, d_hi = ci(diff)
        row.update(
            diff=float(est["a"][j] - est["b"][j]),
            diff_ci_lo=d_lo,
            diff_ci_hi=d_hi,
            n_engines=n_eng,
            boot_defined_share=float(np.isfinite(diff).mean()),
        )
        rows.append(row)
    return pd.DataFrame(rows)
