"""Early identifiability of an engine's subpopulation (D51, Stage A4).

The D35 subpopulation label is a property of an engine's *whole* trajectory. Here it is predicted
**causally** from the engine's first N cycles only. If that works well enough, the label could
become a serving-time feature; if not, it can only stratify folds and split reports.

Per fold of the repeated StratifiedGroupKFold (stratified by the label itself):

- the within-regime normalization is fitted on the fold's training engines (full trajectories);
- each engine is described by the mean and the OLS slope of every base sensor's within-regime z
  over cycles 1..N;
- a standardized L2 logistic regression is fitted on the training engines and scores the held-out
  ones.

Each engine's out-of-fold probability is averaged over the repeats, and the ROC AUC over engines
gets an engine-bootstrap CI. Training files only.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from turbofan.config import KEEP, FeatureBlocks
from turbofan.evaluation.cv import Fold
from turbofan.features.engineering import add_features

__all__ = ["early_features", "identifiability"]

BASE = FeatureBlocks()  # the label is predicted from base sensors only


def early_features(
    train: pd.DataFrame, dataset: str, n: int, stats: dict[str, Any]
) -> pd.DataFrame:
    """One row per engine (index = unit): mean and OLS slope of each base sensor's
    within-regime z over the engine's cycles 1..n. Engines shorter than n raise."""
    first = train[train["cycle"] <= n]
    short = first.groupby("unit")["cycle"].max()
    if (short < n).any():
        raise ValueError(f"engines with fewer than {n} cycles: {short[short < n].index.tolist()}")
    feat, _ = add_features(first, KEEP, dataset, stats=stats, blocks=BASE)
    t = np.arange(1, n + 1, dtype=float)
    t = t - t.mean()
    cols: dict[str, pd.Series[float]] = {}
    g = feat.sort_values(["unit", "cycle"]).groupby("unit")
    for s in KEEP:
        cols[f"{s}_mean"] = g[f"{s}_n"].mean()
        cols[f"{s}_slope"] = g[f"{s}_n"].apply(
            lambda z: float(np.dot(t, z.to_numpy()) / np.dot(t, t))
        )
    return pd.DataFrame(cols)


def identifiability(
    train: pd.DataFrame,
    labels: pd.Series[int],
    dataset: str,
    folds: list[Fold],
    n_cycles: list[int],
    c: float,
    rng: np.random.Generator,
    n_boot: int,
    ci_level: float,
) -> pd.DataFrame:
    """One row per N: AUC of the repeat-averaged out-of-fold probability, its engine-bootstrap
    CI, n engines, and the AUC of each repeat."""
    rows = []
    norm: dict[tuple[int, int], dict[str, Any]] = {}
    for f in folds:  # normalization state per fold, fitted on its training engines
        _, norm[(f.repeat, f.fold)] = add_features(
            train[train["unit"].isin(f.fit_units)], KEEP, dataset, blocks=BASE
        )
    repeats = sorted({f.repeat for f in folds})
    for n in n_cycles:
        oof = pd.DataFrame(index=labels.index, columns=repeats, dtype=float)
        for f in folds:
            x = early_features(train, dataset, n, norm[(f.repeat, f.fold)])
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000))
            clf.fit(x.loc[list(f.fit_units)], labels.loc[list(f.fit_units)])
            oof.loc[list(f.test_units), f.repeat] = clf.predict_proba(x.loc[list(f.test_units)])[
                :, 1
            ]
        if oof.isna().any().any():
            raise RuntimeError("an engine has no out-of-fold prediction in some repeat")
        p = oof.mean(axis=1).to_numpy()
        y = labels.loc[oof.index].to_numpy()
        auc = float(roc_auc_score(y, p))
        idx = rng.integers(0, len(y), size=(n_boot, len(y)))
        boot = np.array([roc_auc_score(y[i], p[i]) for i in idx if len(np.unique(y[i])) == 2])
        a = (1 - ci_level) / 2
        rows.append(
            {
                "n_cycles": n,
                "auc": auc,
                "ci_lo": float(np.quantile(boot, a)),
                "ci_hi": float(np.quantile(boot, 1 - a)),
                "n_engines": len(y),
                "n_boot_defined": len(boot),
                "auc_per_repeat": [float(roc_auc_score(y, oof[r].to_numpy())) for r in repeats],
            }
        )
    return pd.DataFrame(rows)
