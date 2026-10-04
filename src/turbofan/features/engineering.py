"""Feature engineering for the C-MAPSS RUL pipeline.

`add_features` builds the lean feature set used everywhere downstream (modeling, comparison,
training, serving): per sensor, the normalized value plus a causal rolling mean and rolling
slope. Normalization is per operating regime, through one code path for every dataset:
KMeans with k = ``params.yaml`` ``n_regimes[dataset]`` on standardized op settings, then a
z-score per sensor within each regime. k = 1 is the global z-score (FD001/FD003); k = 6 is
the per-regime normalization of FD002/FD004 (D03, D04a/b). Normalization state is fitted on
the training engines and reused on val/test/serving by passing the returned `stats` back in.
All operations are right-aligned (no future leakage) and computed per engine.

Optional blocks (``params.yaml`` ``features``, D51 Stage A3 and the post-Stage-A addendum), all
fitted on the training engines only and stored in the returned state:

- a health index (``hi``, ``hi_mean``, ``hi_slope``) — PC1 of the within-regime z of the base
  sensors (``pooled``) or of the direction-consistent ones (``consistent``: >=
  ``hi_consistent_share`` of training engines share the majority sign of Spearman rho(z, uncapped
  time to failure)), oriented to rise toward failure;
- one indicator column per fitted regime (``regime_onehot``);
- the baseline deviation (``baseline``): per sensor, ``<s>_base`` = the mean z over the engine's
  cycles 1..min(t, ``baseline_cycles``) and ``<s>_dev`` = ``<s>_mean`` - ``<s>_base`` — causal,
  no fitted state beyond the normalization;
- the subpopulation probability (``subpop_prob``): a logistic regression on the mean and OLS slope
  of each base sensor's z over cycles 1..``subpop_prob_cycles``, fitted on the training engines
  against D35 labels the caller derives on those same engines (``subpop_labels``); the training
  engines' own value is cross-fitted. ``subpop_p`` is missing (NaN) before that cycle.

The last two read an engine's history from its first cycle, so each engine must start at cycle 1
(a truncated history raises). Extra sensors arrive through ``sensors``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from turbofan.config import (
    FEATURE_BLOCKS,
    HI_COLS,
    KEEP,
    N_REGIMES,
    OP_COLS,
    REGIME_KMEANS_N_INIT,
    REGIME_KMEANS_SEED,
    SUBPOP_COL,
    WINDOW,
    FeatureBlocks,
    baseline_cols,
    regime_onehot_cols,
)
from turbofan.features.envelope import fit_envelope

__all__ = ["add_features", "assign_regimes", "FEATURE_STATE_KEYS"]

# Keys a fitted feature state must carry; a state without them predates params.yaml.
FEATURE_STATE_KEYS = ("n_regimes", "sensors", "op_sc", "km", "s_mean", "s_std", "envelope")

# A sample std needs two rows; a regime with fewer cannot be normalized (D06, rule 8).
_MIN_REGIME_ROWS = 2

F64 = npt.NDArray[np.float64]


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
            "(bundle_schema < 5) — retrain the bundle"
        )
    if list(stats["sensors"]) != list(sensors):
        raise ValueError(
            f"feature state was fitted on sensors {stats['sensors']}, called with {sensors}"
        )
    k = stats["n_regimes"]
    if any(len(stats[key][s]) != k for key in ("s_mean", "s_std") for s in sensors):
        raise FeatureError(f"feature state has per-regime stats inconsistent with k = {k}")


def _per_engine_spearman(d: pd.DataFrame, col: str) -> pd.Series[float]:
    """Spearman rho(col, rul_true) per engine (NaN where either is constant in the engine)."""
    ranks = d[["unit", col, "rul_true"]].copy()
    g = ranks.groupby("unit")
    ranks[col] = g[col].rank()
    ranks["rul_true"] = g["rul_true"].rank()
    out: pd.Series[float] = ranks.groupby("unit")[[col, "rul_true"]].apply(
        lambda x: x[col].corr(x["rul_true"])
    )
    return out


def _fit_health_index(d: pd.DataFrame, blocks: FeatureBlocks) -> dict[str, Any]:
    """PC1 of the base sensors' within-regime z on the training rows (D51)."""
    if "rul_true" not in d.columns:
        raise FeatureError("fitting a health index needs the uncapped 'rul_true' column")
    base = list(KEEP)
    if blocks.health_index == "consistent":
        chosen = []
        for s in base:
            rho = _per_engine_spearman(d, f"{s}_n").dropna()
            if rho.empty:
                continue
            share = max(float((rho > 0).mean()), float((rho < 0).mean()))
            if share >= blocks.hi_consistent_share:
                chosen.append(s)
        if not chosen:
            raise FeatureError("no direction-consistent sensor on these training engines")
    else:
        chosen = base
    z = d[[f"{s}_n" for s in chosen]].to_numpy(float)
    pca = PCA(n_components=1).fit(z)
    comp = pca.components_[0]
    score = (z - pca.mean_) @ comp
    # orient to rise toward failure: score must fall as time to failure grows
    sign = -1.0 if np.corrcoef(score, d["rul_true"].to_numpy(float))[0, 1] > 0 else 1.0
    return {
        "mode": blocks.health_index,
        "sensors": chosen,
        "mean": pca.mean_.tolist(),
        "component": (sign * comp).tolist(),
        "explained_variance_ratio": float(pca.explained_variance_ratio_[0]),
    }


_SLOPE_CHUNK = 4096  # rows per batch of full windows (bounds the gathered window memory)


def _trailing_slope(y: F64, pos: npt.NDArray[np.intp], window: int) -> F64:
    """OLS slope of each column of ``y`` over the trailing ``window`` rows of the same engine.

    ``y``: (n, k), rows sorted by unit then cycle; ``pos``: each row's 0-based position within
    its engine. A row with m = min(pos + 1, window) >= 2 points gets slope
    sum((t - mean t) * y) / sum((t - mean t)^2) over t = 0..m-1 — the least-squares slope
    ``np.polyfit(t, y, 1)[0]`` computed (the former implementation; equal to it to ~1e-13, D52)
    — and NaN at m = 1. The windows of full length are gathered from one sliding view, the
    shorter ones (the first window - 1 rows of each engine) per length."""
    out = np.full(y.shape, np.nan)

    def fit(win: F64, m: int) -> F64:  # win: (r, m, k), oldest row first
        t = np.arange(m, dtype=float)
        t -= t.mean()
        res: F64 = np.einsum("rmk,m->rk", win, t) / float(t @ t)
        return res

    if window < 2:
        raise FeatureError(f"a rolling slope needs window >= 2, got {window}")
    full = np.flatnonzero(pos >= window - 1)
    for c in range(0, len(full), _SLOPE_CHUNK):
        rows = full[c : c + _SLOPE_CHUNK]
        idx = rows[:, None] - np.arange(window - 1, -1, -1)
        out[rows] = fit(y[idx], window)
    for m in range(2, window):
        rows = np.flatnonzero(pos == m - 1)
        if rows.size:
            out[rows] = fit(y[rows[:, None] - np.arange(m - 1, -1, -1)], m)
    return out


def _rolling(g: Any, col: str, window: int) -> tuple[pd.Series[float], pd.Series[float]]:
    mean = g[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
    d = g.obj
    slope = _trailing_slope(d[[col]].to_numpy(float), g.cumcount().to_numpy(), window)[:, 0]
    return mean, pd.Series(slope, index=d.index)


def _require_history_from_cycle_one(d: pd.DataFrame, block: str) -> None:
    first = d.groupby("unit")["cycle"].min()
    late = first[first != 1]
    if len(late):
        raise FeatureError(
            f"the {block} block reads each engine's history from cycle 1; engines "
            f"{late.index.tolist()[:10]} start later (a truncated history would shift it)"
        )


def _early_summary(d: pd.DataFrame, n: int) -> pd.DataFrame:
    """Per engine with >= n cycles (index = unit): the mean and OLS slope of each base sensor's
    within-regime z over cycles 1..n — the A4 description (``analysis.identifiability``)."""
    first = d[d["cycle"] <= n]
    full = first.groupby("unit")["cycle"].count()
    first = first[first["unit"].isin(full[full == n].index)]
    t = first["cycle"].to_numpy(float) - (n + 1) / 2.0
    tt = float(np.sum((np.arange(1, n + 1) - (n + 1) / 2.0) ** 2))
    g = first.groupby("unit")
    cols: dict[str, pd.Series[float]] = {}
    for s in KEEP:
        cols[f"{s}_early_mean"] = g[f"{s}_n"].mean()
        cols[f"{s}_early_slope"] = (first[f"{s}_n"] * t).groupby(first["unit"]).sum() / tt
    return pd.DataFrame(cols)


def _subpop_classifier(blocks: FeatureBlocks) -> Pipeline:
    return make_pipeline(
        StandardScaler(), LogisticRegression(C=blocks.subpop_prob_c, max_iter=1000)
    )


def _fit_subpop(
    x: pd.DataFrame, labels: pd.Series[int] | None, blocks: FeatureBlocks
) -> tuple[Pipeline, pd.Series[float]]:
    """The classifier on all training engines, and each training engine's cross-fitted
    probability (inner stratified folds: the classifier never scores an engine it saw)."""
    if labels is None:
        raise FeatureError("fitting the subpopulation-probability block needs subpop_labels")
    y = labels.reindex(x.index)
    if y.isna().any():
        raise FeatureError(f"no subpopulation label for engines {y[y.isna()].index.tolist()}")
    y = y.astype(int)
    if y.nunique() != 2:
        raise FeatureError("the training engines hold one subpopulation only")
    inner = StratifiedKFold(
        n_splits=blocks.subpop_prob_inner_folds, shuffle=True, random_state=blocks.subpop_prob_seed
    )
    oof = cross_val_predict(_subpop_classifier(blocks), x, y, cv=inner, method="predict_proba")
    clf = _subpop_classifier(blocks).fit(x, y)
    return clf, pd.Series(oof[:, 1], index=x.index)


def add_features(
    d: pd.DataFrame,
    sensors: list[str],
    dataset_name: str,
    window: int = WINDOW,
    stats: dict[str, Any] | None = None,
    blocks: FeatureBlocks = FEATURE_BLOCKS,
    subpop_labels: pd.Series[int] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """3 features per sensor: normalized value, rolling mean, rolling slope; plus the optional
    blocks of ``blocks``.

    stats=None on train (fits and returns stats); pass the returned stats on val/test.
    ``subpop_labels`` (index = unit): the D35 labels of the training engines, needed only to fit
    the subpopulation-probability block.
    """
    _check_input(d, sensors)
    d = d.sort_values(["unit", "cycle"]).copy()
    fitting = stats is None
    if stats is None:
        k = N_REGIMES[dataset_name]
        op_sc, km = _fit_regimes(d, k)
        stats = {"n_regimes": k, "sensors": list(sensors), "op_sc": op_sc, "km": km}
        labels = assign_regimes(d, stats)
        stats["s_mean"], stats["s_std"] = _fit_norm(d, labels, sensors, k)
        stats["envelope"] = fit_envelope(d, labels, stats, sensors)
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
    hi_cols: list[str] = []
    if blocks.health_index != "none":
        if fitting:
            stats["hi"] = _fit_health_index(d, blocks)
        elif "hi" not in stats:
            raise FeatureError("feature state lacks the fitted health index")
        hi = stats["hi"]
        if hi["mode"] != blocks.health_index:
            raise FeatureError(
                f"state health index is {hi['mode']!r}, asked {blocks.health_index!r}"
            )
        z = d[[f"{s}_n" for s in hi["sensors"]]].to_numpy(float)
        d["hi"] = (z - np.asarray(hi["mean"])) @ np.asarray(hi["component"])
        hi_cols = HI_COLS
    elif "hi" in stats:
        raise FeatureError("feature state carries a health index but none was requested")
    g = d.groupby("unit")
    for s in sensors:
        d[f"{s}_mean"], d[f"{s}_slope"] = _rolling(g, f"{s}_n", window)
    if hi_cols:
        d["hi_mean"], d["hi_slope"] = _rolling(g, "hi", window)
    onehot: list[str] = []
    if blocks.regime_onehot:
        onehot = regime_onehot_cols(stats["n_regimes"])
        for r, c in enumerate(onehot):
            d[c] = (labels == r).astype(float)
    base_cols: list[str] = []
    if blocks.baseline:
        _require_history_from_cycle_one(d, "baseline")
        early = (d["cycle"] <= blocks.baseline_cycles).to_numpy()
        z = d[[f"{s}_n" for s in sensors]].to_numpy(float)
        z_early = pd.DataFrame(np.where(early[:, None], z, 0.0), index=d.index)
        count = pd.Series(early.astype(float), index=d.index).groupby(d["unit"]).cumsum()
        # count >= 1 on every row: each engine starts at cycle 1 (checked above)
        base = z_early.groupby(d["unit"]).cumsum().to_numpy() / count.to_numpy()[:, None]
        for j, s in enumerate(sensors):
            d[f"{s}_base"] = base[:, j]
            d[f"{s}_dev"] = d[f"{s}_mean"] - d[f"{s}_base"]
        base_cols = baseline_cols(sensors)
    if blocks.subpop_prob:
        _require_history_from_cycle_one(d, "subpopulation-probability")
        n = blocks.subpop_prob_cycles
        x = _early_summary(d, n)
        if fitting:
            stats["subpop"], prob = _fit_subpop(x, subpop_labels, blocks)
            stats["subpop_cycles"] = n
        elif "subpop" not in stats:
            raise FeatureError("feature state lacks the fitted subpopulation classifier")
        elif stats["subpop_cycles"] != n:
            raise FeatureError(f"state subpopulation block uses {stats['subpop_cycles']} cycles")
        else:
            prob = pd.Series(stats["subpop"].predict_proba(x)[:, 1], index=x.index)
        later = d["cycle"].to_numpy() >= n
        d[SUBPOP_COL] = np.where(later, d["unit"].map(prob).to_numpy(float), np.nan)
        if np.isnan(d[SUBPOP_COL].to_numpy()[later]).any():
            raise FeatureError("an engine past its first n cycles got no subpopulation probability")
    elif "subpop" in stats:
        raise FeatureError("feature state carries a subpopulation classifier; none was requested")
    # The one legitimate NaN: the slope needs two points, so it is undefined at each engine's
    # first cycle. Defined explicitly as 0 (no trend observed yet), the value the former
    # blanket fillna(0) gave it. Any other NaN is a bug and raises.
    first_cycle = ~d["unit"].duplicated().to_numpy()
    slope_cols = [f"{s}_slope" for s in sensors] + (["hi_slope"] if hi_cols else [])
    d.loc[first_cycle, slope_cols] = 0.0
    out_cols = [f"{s}_{kind}" for kind in ("n", "mean", "slope") for s in sensors]
    out_cols += hi_cols + onehot + base_cols  # subpop_p: NaN before its cycle, checked above
    nan = d[out_cols].isna().sum()
    if nan.any():
        raise FeatureError(f"unexpected NaN in features: {nan[nan > 0].to_dict()}")
    return d, stats
