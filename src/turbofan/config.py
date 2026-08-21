"""Project-wide constants and configuration for the C-MAPSS RUL pipeline.

Single source of truth for the values that must stay identical across feature
engineering, evaluation, model selection, training, and serving. Importing these
instead of re-typing literals is what keeps the train and serve paths from drifting.
"""

from __future__ import annotations

import shutil

RUL_CAP: float = 125.0  # RUL is capped: a healthy engine's RUL is uninformative above this
WINDOW: int = 20  # rolling mean/slope window for the engineered flat features
SEQ_LEN: int = 30  # LSTM input sequence length
SEED: int = 42
SPLIT_FRAC: float = 0.8  # train/val engine split — first-80% by engine order, deterministic

DATASETS: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")
MULTI_REGIME: frozenset[str] = frozenset({"FD002", "FD004"})  # per-regime normalization datasets

# Raw column layout: unit, cycle, 3 operating settings, 21 sensors.
COLS: list[str] = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

# The lean 14-sensor set carried from the data-understanding notebook.
KEEP: list[str] = [
    "s2",
    "s3",
    "s4",
    "s7",
    "s8",
    "s9",
    "s11",
    "s12",
    "s13",
    "s14",
    "s15",
    "s17",
    "s20",
    "s21",
]

# 42 engineered features for the flat models: normalized value, rolling mean, rolling slope.
FEAT_COLS: list[str] = (
    [f"{s}_n" for s in KEEP] + [f"{s}_mean" for s in KEEP] + [f"{s}_slope" for s in KEEP]
)

# 14 normalized sensor channels — the LSTM's input (no engineered mean/slope).
SENSOR_N_COLS: list[str] = [f"{s}_n" for s in KEEP]


def xgb_device() -> str:
    """Return 'cuda' if an NVIDIA GPU is visible, else 'cpu' — mirrors notebook 02."""
    return "cuda" if shutil.which("nvidia-smi") is not None else "cpu"


# -- maintenance action buckets (serving) -----------------------------------------------
# Bounds MUST stay identical to the buckets in evaluation.metrics.per_bucket_metrics;
# metrics.py derives its _BUCKETS list from this tuple so there is one definition.
MAINTENANCE_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("critical", 0.0, 25.0),
    ("urgent", 25.0, 50.0),
    ("monitor", 50.0, 100.0),
    ("healthy", 100.0, float("inf")),
)


def maintenance_bucket(rul: float) -> str:
    """Map a predicted RUL (cycles) to its maintenance action bucket."""
    for name, lo, hi in MAINTENANCE_BUCKETS:
        if lo <= rul < hi:
            return name
    return MAINTENANCE_BUCKETS[-1][0]
