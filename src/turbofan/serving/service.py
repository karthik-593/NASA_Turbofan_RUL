"""Prediction service — the serving feature path, kept HTTP-free so it can be tested
directly (and so the serving-vs-training parity test can call it).

This MUST run the exact same pipeline as training: build the engineered features with
``add_features`` using the bundle's persisted train-time ``stats``, take the last
``SEQ_LEN`` window of the normalized channels, and predict. The parity test guards that
this path reproduces the training path bit-for-bit.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from turbofan.config import KEEP, SENSOR_N_COLS, SEQ_LEN, maintenance_bucket
from turbofan.features.engineering import add_features
from turbofan.models.lstm_model import make_last_windows
from turbofan.serving.schemas import CycleReading
from turbofan.training.bundle import Bundle


class ShortHistory(ValueError):
    """Raised when fewer than SEQ_LEN cycles are supplied."""


def to_frame(cycles: Sequence[CycleReading], unit: int = 1) -> pd.DataFrame:
    """Turn request cycles (chronological) into the raw frame add_features expects."""
    rows = []
    for i, c in enumerate(cycles, start=1):
        row = {"unit": unit, "cycle": i, "op1": c.op1, "op2": c.op2, "op3": c.op3}
        row.update(c.sensors)
        rows.append(row)
    return pd.DataFrame(rows)


def predict_rul(cycles: Sequence[CycleReading], bundle: Bundle) -> dict[str, Any]:
    """Predict RUL for one engine from its cycle history using a loaded bundle."""
    if len(cycles) < SEQ_LEN:
        raise ShortHistory(f"need at least {SEQ_LEN} cycles of history, got {len(cycles)}")

    dataset = bundle.manifest["dataset"]
    feat, _ = add_features(to_frame(cycles), KEEP, dataset, stats=bundle.stats)
    X_w, _units = make_last_windows(feat, SENSOR_N_COLS, SEQ_LEN)
    rul = float(bundle.model.predict(X_w)[0])

    bucket = maintenance_bucket(rul)
    band = bundle.manifest.get("metrics", {}).get("test", {}).get(f"{bucket}_rmse")
    return {
        "predicted_rul": round(rul, 2),
        "maintenance_bucket": bucket,
        "confidence": {
            "error_band_cycles": round(band, 2) if band is not None else None,
            "basis": "test-set RMSE in this RUL bucket for the shipped artifact",
        },
        "n_cycles_used": SEQ_LEN,
        "dataset": dataset,
        "model_version": bundle.manifest["version"],
    }
