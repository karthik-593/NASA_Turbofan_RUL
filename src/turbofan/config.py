"""Project-wide configuration for the C-MAPSS RUL pipeline, read from ``params.yaml``.

Single source of truth for the values that must stay identical across feature engineering,
evaluation, model selection, training and serving (CLAUDE.md rule 5: every parameter lives in
``params.yaml``, no magic numbers in ``src/``). The file is read once, at import; a missing or
malformed key raises instead of falling back to a default (rule 8).

Location: ``$TURBOFAN_PARAMS`` if set, else the first ``params.yaml`` found walking up from the
current directory, then from this file (the repo root in a checkout). The Docker image sets
``TURBOFAN_PARAMS`` explicitly.
"""

from __future__ import annotations

import math
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PARAMS_ENV = "TURBOFAN_PARAMS"
PARAMS_FILE = "params.yaml"


class ParamsError(RuntimeError):
    """params.yaml is missing, unreadable or does not match the expected schema."""


def find_params_file() -> Path:
    env = os.environ.get(PARAMS_ENV)
    if env:
        p = Path(env)
        if not p.is_file():
            raise ParamsError(f"${PARAMS_ENV}={env} does not point to a file")
        return p
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for d in (start, *start.parents):
            if (d / PARAMS_FILE).is_file():
                return d / PARAMS_FILE
    raise ParamsError(
        f"no {PARAMS_FILE} found above {Path.cwd()} or {Path(__file__).parent}; "
        f"run from the repo or set ${PARAMS_ENV}"
    )


def load_params(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path is not None else find_params_file()
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ParamsError(f"{p} must hold a mapping at top level")
    return raw


def _get(d: Mapping[str, Any], dotted: str, typ: type | tuple[type, ...]) -> Any:
    cur: Any = d
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            raise ParamsError(f"params.yaml: missing key '{dotted}'")
        cur = cur[part]
    if not isinstance(cur, typ) or isinstance(cur, bool) and typ is not bool:
        raise ParamsError(f"params.yaml: '{dotted}' must be {typ}, got {type(cur).__name__}")
    return cur


PARAMS: dict[str, Any] = load_params()

DATASETS: tuple[str, ...] = ("FD001", "FD002", "FD003", "FD004")

SEED: int = _get(PARAMS, "seed", int)
RUL_CAP: float = float(_get(PARAMS, "rul_cap", (int, float)))
SPLIT_FRAC: float = float(_get(PARAMS, "split_frac", float))
WINDOW: int = _get(PARAMS, "window", int)
SEQ_LEN: int = _get(PARAMS, "seq_len", int)


def _sensor_list(ds: str, key: str) -> list[str]:
    v = _get(PARAMS, f"sensors.{ds}.{key}", list)
    if not v or not all(isinstance(s, str) for s in v) or len(set(v)) != len(v):
        raise ParamsError(f"params.yaml: sensors.{ds}.{key} must be unique sensor names")
    return list(v)


SENSORS: dict[str, list[str]] = {ds: _sensor_list(ds, "use") for ds in DATASETS}
SENSOR_POOL: dict[str, list[str]] = {ds: _sensor_list(ds, "candidate_pool") for ds in DATASETS}
for _ds in DATASETS:
    if not set(SENSORS[_ds]) <= set(SENSOR_POOL[_ds]):
        raise ParamsError(f"params.yaml: sensors.{_ds}.use must be within its candidate_pool")

N_REGIMES: dict[str, int] = {ds: _get(PARAMS, f"n_regimes.{ds}", int) for ds in DATASETS}
if any(k < 1 for k in N_REGIMES.values()):
    raise ParamsError(f"params.yaml: n_regimes must be >= 1, got {N_REGIMES}")
# Derived, not configured: a dataset is multi-regime iff it has more than one regime.
MULTI_REGIME: frozenset[str] = frozenset(ds for ds, k in N_REGIMES.items() if k > 1)
REGIME_KMEANS_N_INIT: int = _get(PARAMS, "regime_kmeans.n_init", int)
REGIME_KMEANS_SEED: int = _get(PARAMS, "regime_kmeans.random_state", int)

CAP_TEST_TRUTH: bool = _get(PARAMS, "eval.cap_test_truth", bool)
ENVELOPE_MARGIN: float = float(_get(PARAMS, "validation.envelope_margin", (int, float)))
if ENVELOPE_MARGIN < 0:
    raise ParamsError(
        f"params.yaml: validation.envelope_margin must be >= 0, got {ENVELOPE_MARGIN}"
    )
MIN_HISTORY: int = _get(PARAMS, "serving.min_history", int)
if not 1 <= MIN_HISTORY <= SEQ_LEN:
    raise ParamsError(f"params.yaml: serving.min_history must be in 1..seq_len, got {MIN_HISTORY}")

MODEL_PARAMS: dict[str, Any] = _get(PARAMS, "models", dict)
for _key in ("k", "kmeans_n_init", "kmeans_random_state"):
    _get(PARAMS, f"subpopulation.{_key}", int)
for _key in ("n_folds", "n_repeats", "seed"):
    _get(PARAMS, f"cv.{_key}", int)
for _key in ("benchmark_view.bin_width", "benchmark_view.points_per_engine", "bootstrap.n_boot"):
    _get(PARAMS, _key, int)
_get(PARAMS, "bootstrap.ci_level", float)
for _key in ("t_min", "t_max", "t_step"):
    _get(PARAMS, f"decision_curve.{_key}", int)
_get(PARAMS, "decision_curve.lead_times", list)
if not 0.0 < _get(PARAMS, "cv.inner_val_frac", float) < 1.0:
    raise ParamsError("params.yaml: cv.inner_val_frac must be in (0, 1)")
SEEDS: tuple[int, ...] = tuple(_get(PARAMS, "seeds", list))

# Raw column layout: unit, cycle, 3 operating settings, 21 sensors.
OP_COLS: list[str] = ["op1", "op2", "op3"]
COLS: list[str] = ["unit", "cycle", *OP_COLS] + [f"s{i}" for i in range(1, 22)]

# Backward-compatible names for the shared default sensor set. Every dataset uses the same
# 14 sensors today (params.yaml sensors.*.use); code that is per-dataset uses SENSORS[ds].
if any(SENSORS[ds] != SENSORS[DATASETS[0]] for ds in DATASETS):
    raise ParamsError(
        "params.yaml: sensors.*.use differ by dataset, but KEEP / FEAT_COLS / SENSOR_N_COLS "
        "still assume one shared list — make them per-dataset before changing this"
    )
KEEP: list[str] = list(SENSORS[DATASETS[0]])


def feat_cols(sensors: list[str]) -> list[str]:
    """Engineered flat features: normalized value, rolling mean, rolling slope per sensor."""
    return (
        [f"{s}_n" for s in sensors]
        + [f"{s}_mean" for s in sensors]
        + [f"{s}_slope" for s in sensors]
    )


def sensor_n_cols(sensors: list[str]) -> list[str]:
    """The normalized sensor channels — the LSTM's input (no engineered mean/slope)."""
    return [f"{s}_n" for s in sensors]


HEALTH_INDEX_MODES = ("none", "consistent", "pooled")
HI_COLS = ["hi", "hi_mean", "hi_slope"]
SUBPOP_COL = "subpop_p"


def baseline_cols(sensors: list[str]) -> list[str]:
    """The baseline-deviation block: per sensor, its early-life level and today's offset."""
    return [f"{s}_base" for s in sensors] + [f"{s}_dev" for s in sensors]


@dataclass(frozen=True)
class FeatureBlocks:
    """Optional feature blocks on top of the base set (params.yaml ``features``, D51)."""

    extra_sensors: tuple[str, ...] = ()
    health_index: str = "none"
    hi_consistent_share: float = 0.9
    regime_onehot: bool = False
    baseline: bool = False
    baseline_cycles: int = 30
    subpop_prob: bool = False
    subpop_prob_cycles: int = 30
    subpop_prob_c: float = 1.0
    subpop_prob_inner_folds: int = 5
    subpop_prob_seed: int = 0

    @property
    def active(self) -> bool:
        return (
            bool(self.extra_sensors)
            or self.health_index != "none"
            or self.regime_onehot
            or self.baseline
            or self.subpop_prob
        )


def _feature_blocks() -> FeatureBlocks:
    extra = _get(PARAMS, "features.extra_sensors", list)
    hi = _get(PARAMS, "features.health_index", str)
    share = float(_get(PARAMS, "features.hi_consistent_share", (int, float)))
    onehot = _get(PARAMS, "features.regime_onehot", bool)
    if hi not in HEALTH_INDEX_MODES:
        raise ParamsError(f"params.yaml: features.health_index must be one of {HEALTH_INDEX_MODES}")
    if not 0.5 < share <= 1.0:
        raise ParamsError("params.yaml: features.hi_consistent_share must be in (0.5, 1]")
    blocks = FeatureBlocks(
        tuple(extra),
        hi,
        share,
        onehot,
        baseline=_get(PARAMS, "features.baseline", bool),
        baseline_cycles=_get(PARAMS, "features.baseline_cycles", int),
        subpop_prob=_get(PARAMS, "features.subpop_prob", bool),
        subpop_prob_cycles=_get(PARAMS, "features.subpop_prob_cycles", int),
        subpop_prob_c=float(_get(PARAMS, "features.subpop_prob_c", (int, float))),
        subpop_prob_inner_folds=_get(PARAMS, "features.subpop_prob_inner_folds", int),
        subpop_prob_seed=_get(PARAMS, "features.subpop_prob_seed", int),
    )
    if blocks.baseline_cycles < 1 or blocks.subpop_prob_cycles < 2:
        raise ParamsError(
            "params.yaml: features.baseline_cycles must be >= 1 and subpop_prob_cycles >= 2"
        )
    if blocks.subpop_prob_inner_folds < 2 or blocks.subpop_prob_c <= 0:
        raise ParamsError(
            "params.yaml: features.subpop_prob_inner_folds must be >= 2 and subpop_prob_c > 0"
        )
    if blocks.extra_sensors or blocks.regime_onehot:
        ds = _get(PARAMS, "pipeline.dataset", str)
        if ds not in DATASETS:
            raise ParamsError(f"params.yaml: pipeline.dataset {ds!r} not in {DATASETS}")
        bad = [s for s in extra if s in KEEP or s not in SENSOR_POOL[ds]]
        if bad or len(set(extra)) != len(extra):
            raise ParamsError(
                f"params.yaml: features.extra_sensors {extra} must be distinct sensors of "
                f"{ds}'s candidate_pool outside sensors.use (offending: {bad})"
            )
        if blocks.regime_onehot and N_REGIMES[ds] < 2:
            raise ParamsError(f"params.yaml: features.regime_onehot needs regimes; {ds} has 1")
    return blocks


def regime_onehot_cols(k: int) -> list[str]:
    return [f"regime_{r}" for r in range(k)]


FEATURE_BLOCKS: FeatureBlocks = _feature_blocks()
# The sensors a model is built from: sensors.use plus any extra sensors of this run.
MODEL_SENSORS: list[str] = KEEP + list(FEATURE_BLOCKS.extra_sensors)
_ONEHOT: list[str] = (
    regime_onehot_cols(N_REGIMES[_get(PARAMS, "pipeline.dataset", str)])
    if FEATURE_BLOCKS.regime_onehot
    else []
)
_HI_ON = FEATURE_BLOCKS.health_index != "none"
_BASELINE: list[str] = baseline_cols(MODEL_SENSORS) if FEATURE_BLOCKS.baseline else []
# Flat (XGBoost) features: every active block.
FEAT_COLS: list[str] = (
    feat_cols(MODEL_SENSORS)
    + (HI_COLS if _HI_ON else [])
    + _ONEHOT
    + _BASELINE
    + ([SUBPOP_COL] if FEATURE_BLOCKS.subpop_prob else [])
)
# LSTM channels: the normalized sensors, the health-index score (the LSTM learns its own
# smoothing, so not hi_mean / hi_slope), the baseline block and the regime indicators. The
# subpopulation probability is missing before its cycle, which a channel cannot carry, so a
# sequence model refuses that block (evaluation.fitting).
SENSOR_N_COLS: list[str] = (
    sensor_n_cols(MODEL_SENSORS) + (["hi"] if _HI_ON else []) + _BASELINE + _ONEHOT
)


def xgb_device() -> str:
    """Return 'cuda' if an NVIDIA GPU is visible, else 'cpu' — mirrors notebook 02."""
    return "cuda" if shutil.which("nvidia-smi") is not None else "cpu"


# -- maintenance action buckets ---------------------------------------------------------
def _buckets() -> tuple[tuple[str, float, float], ...]:
    raw = _get(PARAMS, "buckets", dict)
    out = []
    for name, bounds in raw.items():
        if not (isinstance(bounds, list) and len(bounds) == 2):
            raise ParamsError(f"params.yaml: buckets.{name} must be [lo, hi]")
        out.append((str(name), float(bounds[0]), float(bounds[1])))
    out.sort(key=lambda b: b[1])  # by lower bound; YAML mapping order is not a contract
    for (_, _, hi), (n2, lo2, _) in zip(out, out[1:], strict=False):
        if hi != lo2:
            raise ParamsError(f"params.yaml: buckets are not contiguous at '{n2}'")
    if out[0][1] != 0.0 or not math.isinf(out[-1][2]):
        raise ParamsError("params.yaml: buckets must start at 0 and end at .inf")
    return tuple(out)


# Bounds MUST stay identical to the buckets in evaluation.metrics.per_bucket_metrics;
# metrics.py derives its _BUCKETS list from this tuple so there is one definition.
MAINTENANCE_BUCKETS: tuple[tuple[str, float, float], ...] = _buckets()


def maintenance_bucket(rul: float) -> str:
    """Map a predicted RUL (cycles) to its maintenance action bucket."""
    for name, lo, hi in MAINTENANCE_BUCKETS:
        if lo <= rul < hi:
            return name
    return MAINTENANCE_BUCKETS[-1][0]
