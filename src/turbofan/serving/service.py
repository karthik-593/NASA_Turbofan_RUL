"""Prediction service — the serving feature path, kept HTTP-free so it can be tested
directly (and so the serving-vs-training parity test can call it).

This MUST run the exact same pipeline as training: build the engineered features with
``add_features`` using the bundle's persisted train-time ``stats``, take the last
``seq_len`` window of the normalized channels — padded and masked by the same builder as
training when the history is shorter (D11) — and predict. Window length, sensors and
rolling window come from the bundle's manifest (what the model was trained with), not from
the current params.yaml. The parity test guards that this path reproduces the training
path bit-for-bit.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from turbofan.config import MIN_HISTORY, maintenance_bucket, sensor_n_cols
from turbofan.features.engineering import add_features
from turbofan.models.lstm_model import make_last_windows
from turbofan.serving.schemas import CycleReading
from turbofan.training.bundle import Bundle

PENDING_BASIS = "pending calibrated intervals"


class ShortHistory(ValueError):
    """Raised when fewer than ``serving.min_history`` cycles are supplied."""


def to_frame(cycles: Sequence[CycleReading], unit: int = 1) -> pd.DataFrame:
    """Turn request cycles (chronological) into the raw frame add_features expects."""
    rows = []
    for i, c in enumerate(cycles, start=1):
        row = {"unit": unit, "cycle": i, "op1": c.op1, "op2": c.op2, "op3": c.op3}
        row.update(c.sensors)
        rows.append(row)
    return pd.DataFrame(rows)


def serving_window(cycles: Sequence[CycleReading], bundle: Bundle) -> npt.NDArray[np.float32]:
    """The model input for one request: ``(1, seq_len, n_channels + 1)``."""
    if len(cycles) < MIN_HISTORY:
        raise ShortHistory(f"need at least {MIN_HISTORY} cycles of history, got {len(cycles)}")
    cfg = bundle.manifest["config"]
    sensors = list(cfg["keep"])
    feat, _ = add_features(
        to_frame(cycles), sensors, bundle.manifest["dataset"], cfg["window"], stats=bundle.stats
    )
    X_w, _units = make_last_windows(feat, sensor_n_cols(sensors), cfg["seq_len"])
    return X_w


def predict_rul(cycles: Sequence[CycleReading], bundle: Bundle) -> dict[str, Any]:
    """Predict RUL for one engine from its cycle history using a loaded bundle."""
    dataset = bundle.manifest["dataset"]
    rul = float(bundle.model.predict(serving_window(cycles, bundle))[0])
    # The model is trained on labels capped at rul_cap and its output is clipped there, so a
    # prediction at the cap means "at least rul_cap cycles", not that number exactly.
    rul_at_cap = rul >= bundle.manifest["config"]["rul_cap"]

    # No error band until calibrated intervals exist (D26): the old band was the shipped
    # model's test-set RMSE per bucket — calibrated on the test set (rule 3) and looked up by
    # predicted rather than true bucket.
    return {
        "predicted_rul": round(rul, 2),
        "rul_at_cap": rul_at_cap,
        "maintenance_bucket": maintenance_bucket(rul),
        "confidence": {"error_band_cycles": None, "basis": PENDING_BASIS},
        "n_cycles_used": min(len(cycles), bundle.manifest["config"]["seq_len"]),
        "dataset": dataset,
        "model_version": bundle.manifest["version"],
    }
