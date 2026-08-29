"""Model-selection driver — the single code path behind notebook 03.

Every candidate runs through the same protocol: train-only normalization stats reused
on val/test, the same first-80% engine split, one prediction per test engine at its last
cycle vs NASA ground truth, scored by ``protocol.score``. A registry plus two thin
adapters (flat features vs LSTM sequences) keep it honest: adding a model is one line and
there is no second place the protocol can drift. Fixed, untuned configs — XGBoost uses
library defaults, NOT notebook 02's tuned params (disclosed in notebook 03).

``run_comparison_multiseed`` re-runs chosen contenders over several model seeds to put
variance bands on the headline metric. The engine split and the features are
seed-independent and built once per dataset, so only model init/training varies — the
robustness question the screen leaves open. Split-composition variance is a separate
source and is deliberately not tested here.

Loader contract: load_dataset(name, raw) -> (train_df, test_df, rul_series), train_df
with a capped 'rul' column, rul_series indexed 1..N by unit.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
import pandas as pd

from turbofan.config import (
    DATASETS,
    FEAT_COLS,
    KEEP,
    SEED,
    SENSOR_N_COLS,
    SEQ_LEN,
    SPLIT_FRAC,
    xgb_device,
)
from turbofan.data.loader import load_dataset
from turbofan.evaluation.protocol import predict_last_cycle, score
from turbofan.features.engineering import add_features
from turbofan.models.baselines import MeanBaseline, RandomForestRUL, RidgeRUL
from turbofan.models.lstm_model import LSTMRUL, make_last_windows, make_sequences
from turbofan.models.xgboost_model import XGBoostRUL

__all__ = [
    "Candidate",
    "build_registry",
    "split_engines",
    "prepare",
    "run_comparison",
    "decision_summary",
    "run_comparison_multiseed",
    "multiseed_summary",
    "contender_gap",
]


class Candidate(NamedTuple):
    """kind: 'flat' (42 engineered features) | 'sequence' (14 normalized channels, windowed)"""

    factory: Callable[[], Any]
    kind: str


def build_registry(device: str | None = None, seed: int = SEED) -> dict[str, Candidate]:
    """The five candidates at fixed, untuned configs. Add a model = add a line.

    ``seed`` flows into every stochastic model so a registry can be rebuilt per seed.
    """
    dev = device or xgb_device()
    return {
        "mean": Candidate(lambda: MeanBaseline(), "flat"),
        "ridge": Candidate(lambda: RidgeRUL(alpha=1.0), "flat"),
        "rf": Candidate(lambda: RandomForestRUL(n_estimators=300, random_state=seed), "flat"),
        # library DEFAULTS on purpose — not 02's tuned params; disclosed in notebook 03.
        "xgboost": Candidate(
            lambda: XGBoostRUL(params={"device": dev, "random_state": seed}), "flat"
        ),
        "lstm": Candidate(lambda: LSTMRUL(n_features=len(SENSOR_N_COLS), seed=seed), "sequence"),
    }


def split_engines(
    train_df: pd.DataFrame, frac: float = SPLIT_FRAC
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """First-``frac`` of engines by id order — deterministic, identical to notebook 02."""
    engines = train_df["unit"].unique()
    k = int(len(engines) * frac)
    return np.asarray(engines[:k], dtype=np.int64), np.asarray(engines[k:], dtype=np.int64)


def prepare(
    name: str, raw: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series[int], dict[str, Any]]:
    """Load + feature-engineer one dataset. Seed-independent, so reusable across seeds.

    Returns (feat_tr, feat_va, feat_te, rul_te, stats); ``stats`` is the fitted feature
    state the training step persists into the artifact bundle.
    """
    tr, te, rul_te = load_dataset(name, raw)
    tr_e, va_e = split_engines(tr)
    feat_tr, stats = add_features(tr[tr["unit"].isin(tr_e)], KEEP, name)
    feat_va, _ = add_features(tr[tr["unit"].isin(va_e)], KEEP, name, stats=stats)
    feat_te, _ = add_features(te, KEEP, name, stats=stats)
    return feat_tr, feat_va, feat_te, rul_te, stats


def _eval_flat(
    model: Any,
    feat_tr: pd.DataFrame,
    feat_va: pd.DataFrame,
    feat_te: pd.DataFrame,
    rul_te: pd.Series[int],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    model.fit(
        feat_tr[FEAT_COLS],
        feat_tr["rul"].to_numpy(),
        feat_va[FEAT_COLS],
        feat_va["rul"].to_numpy(),
    )
    y, pred = predict_last_cycle(model, feat_te, true_rul=rul_te.astype(float))
    return y, pred


def _eval_sequence(
    model: Any,
    feat_tr: pd.DataFrame,
    feat_va: pd.DataFrame,
    feat_te: pd.DataFrame,
    rul_te: pd.Series[int],
    cols: list[str] = SENSOR_N_COLS,
    seq_len: int = SEQ_LEN,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    X_tr, y_tr = make_sequences(feat_tr, cols, seq_len)
    X_va, y_va = make_sequences(feat_va, cols, seq_len)
    model.fit(X_tr, y_tr, X_va, y_va)
    X_w, units = make_last_windows(feat_te, cols, seq_len)
    y = rul_te.loc[list(units)].to_numpy().astype(float)  # align ground truth to window order
    pred = model.predict(X_w)
    return y, pred


def _evaluate(
    model: Any,
    kind: str,
    feat_tr: pd.DataFrame,
    feat_va: pd.DataFrame,
    feat_te: pd.DataFrame,
    rul_te: pd.Series[int],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    if kind == "flat":
        return _eval_flat(model, feat_tr, feat_va, feat_te, rul_te)
    return _eval_sequence(model, feat_tr, feat_va, feat_te, rul_te)


def run_comparison(
    raw: str | Path,
    datasets: tuple[str, ...] = DATASETS,
    registry: dict[str, Candidate] | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run every candidate over every dataset under one protocol; return tidy rows."""
    registry = registry or build_registry()
    rows = []
    for name in datasets:
        feat_tr, feat_va, feat_te, rul_te, _ = prepare(name, raw)
        for mname, cand in registry.items():
            y, pred = _evaluate(cand.factory(), cand.kind, feat_tr, feat_va, feat_te, rul_te)
            row = score(y, pred)
            rows.append({"dataset": name, "model": mname, **row})
            if verbose:
                crit, nasa = row["critical_rmse"], row["nasa"]
                print(f"{name:6s} {mname:8s} crit={crit:6.2f}  nasa={nasa:8.0f}")
        if verbose:
            print(f"-- {name} done --")
    return pd.DataFrame(rows)


def decision_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series[int]]:
    """Aggregate into decision evidence: (agg, wins).

    agg  — per model, mean critical-zone RMSE and mean NASA across datasets, best-first.
    wins — count of datasets each model wins on critical-zone RMSE.
    """
    agg = (
        df.groupby("model")
        .agg(mean_crit=("critical_rmse", "mean"), mean_nasa=("nasa", "mean"))
        .sort_values(["mean_crit", "mean_nasa"])
    )
    wins = df.loc[df.groupby("dataset")["critical_rmse"].idxmin(), "model"].value_counts()
    return agg, wins


# -- variance check ---------------------------------------------------------------------


def run_comparison_multiseed(
    raw: str | Path,
    seeds: tuple[int, ...] = (42, 7, 123, 2024, 99),
    models: tuple[str, ...] = ("xgboost", "lstm"),
    datasets: tuple[str, ...] = DATASETS,
    verbose: bool = True,
) -> pd.DataFrame:
    """Re-run ``models`` over ``seeds`` to band the headline metric.

    Features + split are built once per dataset and reused across seeds (only model
    init/training varies). Returns tidy rows tagged with a 'seed' column.

    Cost note: this is ``len(seeds) * len(datasets)`` fits per model. The LSTM fits on
    FD002/FD004 dominate — start with 3 seeds for a fast first pass if needed.
    """
    rows = []
    for name in datasets:
        feat_tr, feat_va, feat_te, rul_te, _ = prepare(name, raw)
        for seed in seeds:
            reg = build_registry(seed=seed)
            for mname in models:
                cand = reg[mname]
                y, pred = _evaluate(cand.factory(), cand.kind, feat_tr, feat_va, feat_te, rul_te)
                row = score(y, pred)
                rows.append({"dataset": name, "model": mname, "seed": seed, **row})
                if verbose:
                    print(f"{name:6s} {mname:8s} seed={seed:<5d} crit={row['critical_rmse']:6.2f}")
        if verbose:
            print(f"-- {name} done ({len(seeds)} seeds) --")
    return pd.DataFrame(rows)


def multiseed_summary(df_ms: pd.DataFrame, metric: str = "critical_rmse") -> pd.DataFrame:
    """Per dataset×model: mean / std / min / max of ``metric`` across seeds."""
    return df_ms.groupby(["dataset", "model"])[metric].agg(["mean", "std", "min", "max"]).round(2)


def contender_gap(
    df_ms: pd.DataFrame, a: str = "lstm", b: str = "xgboost", metric: str = "critical_rmse"
) -> pd.DataFrame:
    """Head-to-head per dataset. gap = mean_b - mean_a (positive => a is better on RMSE).

    ``separated_1sigma`` is True when |gap| exceeds the summed ±1σ bands — a simple,
    conservative read of whether the difference survives seed variance.
    """
    s = df_ms.groupby(["dataset", "model"])[metric].agg(["mean", "std"])
    rows = []
    for ds in df_ms["dataset"].unique():
        ma, sa = s.loc[(ds, a), "mean"], s.loc[(ds, a), "std"]
        mb, sb = s.loc[(ds, b), "mean"], s.loc[(ds, b), "std"]
        gap = mb - ma
        rows.append(
            {
                "dataset": ds,
                f"{a}_mean": round(ma, 2),
                f"{a}_std": round(sa, 2),
                f"{b}_mean": round(mb, 2),
                f"{b}_std": round(sb, 2),
                "gap_b_minus_a": round(gap, 2),
                "separated_1sigma": bool(abs(gap) > (sa + sb)),
            }
        )
    return pd.DataFrame(rows).set_index("dataset")
