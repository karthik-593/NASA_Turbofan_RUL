"""Versioned artifact bundles — model + feature state + provenance.

A bundle is everything the serving path needs to reproduce a prediction:

    models/<dataset>/<model>/<version>/
        model.bin           # written by the model wrapper's own .save()
        feature_state.pkl   # the add_features `stats` (regimes + per-regime norm), via joblib
        manifest.json       # config, metrics, seed, lib versions, provenance

The feature state ships *with* the model on purpose: a sensor value is meaningless
without its operating regime, so train-time normalization must be reapplied identically
at inference. ``load_bundle`` is what the API (step 3) calls.
"""

from __future__ import annotations

import importlib
import json
import platform
from collections import namedtuple
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

import joblib

from turbofan import config as cfg

BUNDLE_SCHEMA = 1
MODEL_FILE = "model.bin"
FEATURE_STATE_FILE = "feature_state.pkl"
MANIFEST_FILE = "manifest.json"

# String registry — importing bundle.py never pulls in xgboost/torch/sklearn until
# load_bundle() is actually called. Serving startup stays dependency-free.
_MODEL_REGISTRY: dict[str, tuple[str, str]] = {
    "mean": ("turbofan.models.baselines", "MeanBaseline"),
    "ridge": ("turbofan.models.baselines", "RidgeRUL"),
    "rf": ("turbofan.models.baselines", "RandomForestRUL"),
    "xgboost": ("turbofan.models.xgboost_model", "XGBoostRUL"),
    "lstm": ("turbofan.models.lstm_model", "LSTMRUL"),
}


def __getattr__(name: str) -> object:
    """Lazy MODEL_CLASSES for callers that import it by name."""
    if name == "MODEL_CLASSES":
        return {k: getattr(importlib.import_module(v[0]), v[1]) for k, v in _MODEL_REGISTRY.items()}
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


Bundle = namedtuple("Bundle", "model stats manifest path")

_TRACKED_LIBS = ("torch", "xgboost", "scikit-learn", "numpy", "pandas")


def _lib_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": platform.python_version()}
    for name in _TRACKED_LIBS:
        try:
            out[name] = _pkg_version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def new_version() -> str:
    """UTC timestamp version id; lexically sortable, so max() is the latest."""
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def bundle_dir(out_root: str | Path, dataset: str, model_name: str, version: str) -> Path:
    return Path(out_root) / dataset / model_name / version


def save_bundle(
    out_root: str | Path,
    dataset: str,
    model_name: str,
    version: str,
    model: Any,
    stats: object,
    *,
    seed: int,
    metrics: dict[str, dict[str, float]],
) -> Path:
    """Write a complete bundle; return its directory."""
    d = bundle_dir(out_root, dataset, model_name, version)
    d.mkdir(parents=True, exist_ok=True)
    model.save(str(d / MODEL_FILE))
    joblib.dump(stats, d / FEATURE_STATE_FILE)
    manifest = {
        "bundle_schema": BUNDLE_SCHEMA,
        "dataset": dataset,
        "model": model_name,
        "version": version,
        "created_at": datetime.now(UTC).isoformat(),
        "seed": seed,
        "config": {
            "rul_cap": cfg.RUL_CAP,
            "window": cfg.WINDOW,
            "seq_len": cfg.SEQ_LEN,
            "keep": list(cfg.KEEP),
            "feat_cols": list(cfg.FEAT_COLS),
            "sensor_n_cols": list(cfg.SENSOR_N_COLS),
        },
        "metrics": metrics,
        "lib_versions": _lib_versions(),
        "files": {"model": MODEL_FILE, "feature_state": FEATURE_STATE_FILE},
    }
    (d / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    return d


def resolve_version(
    out_root: str | Path, dataset: str, model_name: str, version: str = "latest"
) -> str:
    base = Path(out_root) / dataset / model_name
    if version != "latest":
        if not (base / version).is_dir():
            raise FileNotFoundError(f"No bundle at {base / version}")
        return version
    versions = sorted(p.name for p in base.iterdir() if p.is_dir()) if base.is_dir() else []
    if not versions:
        raise FileNotFoundError(f"No bundles under {base}")
    return versions[-1]


def load_bundle(
    out_root: str | Path,
    dataset: str,
    model_name: str = "lstm",
    version: str = "latest",
    device: str | None = None,
) -> Bundle:
    """Load a bundle for inference: (model, stats, manifest, path)."""
    version = resolve_version(out_root, dataset, model_name, version)
    d = bundle_dir(out_root, dataset, model_name, version)
    manifest = json.loads((d / MANIFEST_FILE).read_text())
    mod_path, cls_name = _MODEL_REGISTRY[model_name]
    cls = getattr(importlib.import_module(mod_path), cls_name)
    model_path = str(d / manifest["files"]["model"])
    model = cls.load(model_path, device=device) if model_name == "lstm" else cls.load(model_path)
    stats = joblib.load(d / manifest["files"]["feature_state"])
    return Bundle(model=model, stats=stats, manifest=manifest, path=d)
