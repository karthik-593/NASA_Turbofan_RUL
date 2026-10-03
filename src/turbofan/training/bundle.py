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
import os
import platform
import subprocess
from collections import namedtuple
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

import joblib

from turbofan import config as cfg

# 3: feature state carries a regime model for every dataset (params.yaml n_regimes, D38).
# 4: LSTM input gains a mask channel; short histories padded with the first real cycle (D11).
# 5: feature state carries the training operating envelope checked at serving (D41).
BUNDLE_SCHEMA = 5
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


def _git(repo_dir: Path, *args: str) -> str:
    try:
        res = subprocess.run(
            ["git", "-C", str(repo_dir), *args], capture_output=True, text=True, check=False
        )
    except FileNotFoundError as e:
        raise RuntimeError("git executable not found; cannot record bundle provenance") from e
    if res.returncode != 0:
        raise RuntimeError(
            f"`git {' '.join(args)}` failed in {repo_dir}: {res.stderr.strip()} — "
            "bundles must be built from a git checkout so the manifest can record the commit"
        )
    return res.stdout.strip()


def _try_git_sha(repo_dir: Path) -> str | None:
    """HEAD SHA, or None if git is missing or ``repo_dir`` isn't a git checkout."""
    try:
        res = subprocess.run(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return None
    return res.stdout.strip() if res.returncode == 0 else None


_TRUTHY_ENV = {"1", "true", "yes"}


def git_provenance(repo_dir: Path | None = None) -> dict[str, str | bool]:
    """Commit SHA of the code, and whether tracked files differ from it (untracked ignored).

    Falls back to the ``GIT_COMMIT`` / ``GIT_DIRTY`` env vars when ``repo_dir`` is not a git
    checkout (e.g. a built container with the ``.git`` directory excluded) — set both, or
    neither; a lone env var is treated as missing. Raises only if there is no git checkout
    *and* the env vars are not set, since the manifest must not silently omit provenance.
    """
    repo_dir = repo_dir or Path(__file__).resolve().parent
    sha = _try_git_sha(repo_dir)
    if sha is not None:
        # The checkout exists (sha succeeded); a failure here is a real anomaly, not a
        # missing-checkout case, so let `_git` raise instead of silently treating it as clean.
        dirty = bool(_git(repo_dir, "status", "--porcelain", "--untracked-files=no"))
        return {"git_commit": sha, "git_dirty": dirty}

    env_sha, env_dirty = os.environ.get("GIT_COMMIT"), os.environ.get("GIT_DIRTY")
    if env_sha is not None and env_dirty is not None:
        return {"git_commit": env_sha, "git_dirty": env_dirty.strip().lower() in _TRUTHY_ENV}

    raise RuntimeError(
        f"{repo_dir} is not a git checkout and GIT_COMMIT/GIT_DIRTY env vars are not both "
        "set; bundles must be built from a git checkout or have both env vars set so the "
        "manifest can record provenance"
    )


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
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a complete bundle; return its directory. ``extra`` adds top-level manifest keys
    (e.g. the locked spec's hash); it may not overwrite a key the bundle sets itself."""
    provenance = git_provenance()
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
        **provenance,
        "config": {
            "rul_cap": cfg.RUL_CAP,
            "window": cfg.WINDOW,
            "seq_len": cfg.SEQ_LEN,
            "keep": list(cfg.KEEP),
            "n_regimes": cfg.N_REGIMES[dataset],
            "feat_cols": list(cfg.FEAT_COLS),
            "sensor_n_cols": list(cfg.SENSOR_N_COLS),
        },
        "metrics": metrics,
        "lib_versions": _lib_versions(),
        "files": {"model": MODEL_FILE, "feature_state": FEATURE_STATE_FILE},
    }
    clash = sorted(set(extra or {}) & set(manifest))
    if clash:
        raise ValueError(f"extra manifest keys clash with the bundle's own: {clash}")
    manifest.update(extra or {})
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


def load_bundle_dir(d: str | Path, device: str | None = None) -> Bundle:
    """Load a bundle from an already-resolved directory (model name read from its manifest).

    Split out of ``load_bundle`` so a caller that already has the bundle's files sitting in
    some directory — an MLflow pyfunc artifact restored under its own path, say, with no
    ``<out_root>/<dataset>/<model>/<version>`` nesting left — can load it without needing to
    reconstruct that nesting.
    """
    d = Path(d)
    manifest = json.loads((d / MANIFEST_FILE).read_text())
    schema = manifest.get("bundle_schema")
    if schema != BUNDLE_SCHEMA:
        raise ValueError(
            f"bundle {d} has bundle_schema {schema}, this code reads {BUNDLE_SCHEMA} — "
            "its feature state or model input is incompatible; retrain it"
        )
    model_name = manifest["model"]
    mod_path, cls_name = _MODEL_REGISTRY[model_name]
    cls = getattr(importlib.import_module(mod_path), cls_name)
    model_path = str(d / manifest["files"]["model"])
    model = cls.load(model_path, device=device) if model_name == "lstm" else cls.load(model_path)
    stats = joblib.load(d / manifest["files"]["feature_state"])
    return Bundle(model=model, stats=stats, manifest=manifest, path=d)


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
    return load_bundle_dir(d, device=device)
