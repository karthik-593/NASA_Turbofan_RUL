"""Train the chosen model and write a versioned artifact bundle.

    python -m turbofan.training.train --dataset FD001            # ships the LSTM
    python -m turbofan.training.train --dataset FD001 --model xgboost

Deploy-first: trains at the registry's fixed config (no tuning) on the same first-80%
engine split used for model selection, and records the test-set critical-zone RMSE of
*this* artifact so you know exactly which draw you shipped (the LSTM is seed-sensitive).
Tuning is a later, measurable upgrade and does not change the bundle contract the API loads.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from turbofan.config import FEAT_COLS, SEED, SENSOR_N_COLS, SEQ_LEN
from turbofan.evaluation.comparison import build_registry, prepare
from turbofan.evaluation.protocol import predict_last_cycle, score
from turbofan.models.lstm_model import make_last_windows, make_sequences
from turbofan.training.bundle import new_version, save_bundle


def _fit_and_eval(
    model: Any,
    kind: str,
    feat_tr: pd.DataFrame,
    feat_va: pd.DataFrame,
    feat_te: pd.DataFrame,
    rul_te: pd.Series[int],
) -> tuple[Any, npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Fit on train (val for early stopping) and score on the NASA test set."""
    if kind == "flat":
        model.fit(
            feat_tr[FEAT_COLS],
            feat_tr["rul"].to_numpy(),
            feat_va[FEAT_COLS],
            feat_va["rul"].to_numpy(),
        )
        y, pred = predict_last_cycle(model, feat_te, true_rul=rul_te.astype(float))
    else:
        X_tr, y_tr = make_sequences(feat_tr, SENSOR_N_COLS, SEQ_LEN)
        X_va, y_va = make_sequences(feat_va, SENSOR_N_COLS, SEQ_LEN)
        model.fit(X_tr, y_tr, X_va, y_va)
        X_w, units = make_last_windows(feat_te, SENSOR_N_COLS, SEQ_LEN)
        y = rul_te.loc[list(units)].to_numpy().astype(float)
        pred = model.predict(X_w)
    return model, y, pred


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a model and write an artifact bundle.")
    ap.add_argument("--dataset", required=True, choices=("FD001", "FD002", "FD003", "FD004"))
    ap.add_argument("--model", default="lstm", choices=("mean", "ridge", "rf", "xgboost", "lstm"))
    ap.add_argument("--raw", default="data/raw", help="raw C-MAPSS directory")
    ap.add_argument("--out", default="models", help="bundle output root")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--version", default=None, help="bundle version id (default: UTC timestamp)")
    args = ap.parse_args()

    cand = build_registry(seed=args.seed)[args.model]
    feat_tr, feat_va, feat_te, rul_te, stats = prepare(args.dataset, args.raw)
    model, y, pred = _fit_and_eval(cand.factory(), cand.kind, feat_tr, feat_va, feat_te, rul_te)

    # Each top-level key maps to a dict of named metrics (matches "test"'s shape) — a bare
    # float here would silently violate that shape for any caller iterating metrics.items().
    metrics: dict[str, dict[str, float]] = {"test": score(y, pred)}
    val_loss = getattr(model, "best_val_loss_", None)
    if val_loss is not None:
        metrics["val"] = {"loss": float(val_loss)}

    version = args.version or new_version()
    d = save_bundle(
        args.out, args.dataset, args.model, version, model, stats, seed=args.seed, metrics=metrics
    )

    t = metrics["test"]
    print(f"bundle         : {d}")
    print(
        f"test crit RMSE : {t['critical_rmse']:.2f}  "
        f"| nasa {t['nasa']:.0f}  | global {t['global_rmse']:.2f}  | late {t['late_pct']:.1f}%"
    )


if __name__ == "__main__":
    main()
