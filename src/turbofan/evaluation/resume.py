"""Resumable cross-validation: skip splits MLflow already holds as FINISHED (D49).

A fold x seed fit of one model is identified by

- a **config hash**: dataset, model, seed, repeat, fold, ``rul_cap``, the engines in play, the
  DVC data hash, and every ``params.yaml`` section that changes what a fit produces
  (``FIT_SECTIONS`` plus the model's own ``models.<name>`` entry); and
- a **source fingerprint**: the git tree hash of ``src/`` at HEAD, plus a digest of any
  uncommitted change under ``src/`` (so an edit never silently reuses stale predictions).

Each finished split logs its held-out predictions (``predictions.parquet``) and
``split_meta.json`` (fit seconds, iteration budget) to its child run, tagged with both keys.
A resumed run looks those up and reloads instead of refitting. A matching FINISHED run
without its artifacts raises: it cannot be reused and must not be quietly refitted over.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from turbofan import tracking
from turbofan.config import PARAMS

__all__ = [
    "FIT_SECTIONS",
    "PREDICTIONS_FILE",
    "META_FILE",
    "config_hash",
    "src_fingerprint",
    "finished_splits",
    "load_split",
    "log_split",
]

# Every top-level params.yaml section is classified as one or the other; a test fails on an
# unclassified one, so a new fit-affecting section cannot silently stay out of the key (as
# ``features`` did at first — D49, found 2026-10-03 in Stage A).
# Sections that change the predictions of a fit:
FIT_SECTIONS = (
    "seed",
    "sensors",
    "n_regimes",
    "regime_kmeans",
    "window",
    "seq_len",
    "subpopulation",
    "cv",
    "features",
)
# Sections that do not: evaluation-side (applied to stored predictions), selection, serving,
# pipeline plumbing, or passed explicitly (rul_cap, models.<name>, the engine subset).
NOT_FIT_SECTIONS = (
    "rul_cap",
    "rul_cap_grid",
    "split_frac",
    "seq_len_grid",
    "models",
    "eval",
    "buckets",
    "seeds",
    "benchmark_view",
    "decision_curve",
    "bootstrap",
    "validation",
    "serving",
    "pipeline",
    "release",
    "final_eval",
    "selection",
    "tuning",
    "smoke",
)
PREDICTIONS_FILE = "predictions.parquet"
META_FILE = "split_meta.json"
_REPO = Path(__file__).resolve().parents[3]
HASH_TAG = "fit_config_hash"
SRC_TAG = "src_fingerprint"


def _git(*args: str) -> str:
    res = subprocess.run(
        ["git", "-C", str(_REPO), *args], capture_output=True, text=True, check=False
    )
    if res.returncode != 0:
        raise RuntimeError(
            f"`git {' '.join(args)}` failed: {res.stderr.strip()} — resumable CV needs a git "
            "checkout to fingerprint src/"
        )
    return res.stdout.strip()


def src_fingerprint() -> str:
    """``<tree hash of src/ at HEAD>`` or ``<tree>+<digest of uncommitted changes>``."""
    tree = _git("rev-parse", "HEAD:src")
    changed = _git("status", "--porcelain", "--untracked-files=all", "--", "src")
    if not changed:
        return tree
    h = hashlib.sha256(_git("diff", "HEAD", "--", "src").encode())
    for line in sorted(changed.splitlines()):
        if line.startswith("??"):  # untracked: not in the diff, so hash its content
            h.update((_REPO / line[3:].strip().strip('"')).read_bytes())
    return f"{tree}+{h.hexdigest()[:16]}"


def config_hash(
    *,
    dataset: str,
    model: str,
    seed: int,
    repeat: int,
    fold: int,
    n_folds: int,
    n_repeats: int,
    rul_cap: float,
    engines: list[int] | None,
) -> str:
    doc: dict[str, Any] = {
        "dataset": dataset,
        "model": model,
        "seed": seed,
        "repeat": repeat,
        "fold": fold,
        "n_folds": n_folds,
        "n_repeats": n_repeats,
        "rul_cap": rul_cap,
        "engines": sorted(engines) if engines is not None else None,
        "dvc_data_hash": tracking._dvc_data_hash(),
        "params": {k: PARAMS[k] for k in FIT_SECTIONS},
        "model_params": PARAMS["models"].get(model),
    }
    return hashlib.sha256(json.dumps(doc, sort_keys=True, default=str).encode()).hexdigest()


def finished_splits(dataset: str, model: str, fingerprint: str) -> dict[str, str]:
    """``{config hash: MLflow run id}`` of FINISHED CV child runs for this dataset and model
    produced by exactly this source."""
    mlflow.set_tracking_uri(tracking._tracking_uri())
    exp = mlflow.get_experiment_by_name(tracking._experiment_name())
    if exp is None:
        return {}
    runs = mlflow.MlflowClient().search_runs(
        [exp.experiment_id],
        filter_string=(
            "attributes.status = 'FINISHED' and tags.run_type = 'cv' and "
            f"tags.cv_role = 'child' and tags.{SRC_TAG} = '{fingerprint}' and "
            f"params.dataset = '{dataset}' and params.model = '{model}'"
        ),
        max_results=50_000,
    )
    return {r.data.tags[HASH_TAG]: r.info.run_id for r in runs if HASH_TAG in r.data.tags}


def log_split(pts: pd.DataFrame, meta: dict[str, Any]) -> None:
    """Attach a split's predictions and metadata to the active child run."""
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / PREDICTIONS_FILE
        pts.to_parquet(path, index=False)
        mlflow.log_artifact(str(path))
    mlflow.log_dict(meta, META_FILE)


def load_split(run_id: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Predictions and metadata a finished child run logged."""
    client = mlflow.MlflowClient()
    names = {a.path for a in client.list_artifacts(run_id)}
    missing = {PREDICTIONS_FILE, META_FILE} - names
    if missing:
        raise RuntimeError(
            f"MLflow run {run_id} is FINISHED and matches this split but lacks {sorted(missing)}; "
            "delete the run or change the config to recompute it"
        )
    with tempfile.TemporaryDirectory() as d:
        pts = pd.read_parquet(client.download_artifacts(run_id, PREDICTIONS_FILE, d))
        meta: dict[str, Any] = json.loads(
            Path(client.download_artifacts(run_id, META_FILE, d)).read_text(encoding="utf-8")
        )
    return pts, meta
