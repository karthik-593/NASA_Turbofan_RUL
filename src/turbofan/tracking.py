"""MLflow integration: run tracking and production-model registration.

Two concerns live here:

1. ``run()`` — a context manager for one MLflow run that always logs the provenance
   CLAUDE.md rule 6 requires (git SHA + dirty flag, the DVC data hash, every
   ``turbofan.config`` value, library versions, dataset/model/seed) plus a ``run_type``
   tag distinguishing what the run is for:

   - ``cv``                     — a protocol-v2 model-selection/robustness screen that
                                   never reads the sealed test set. Reserved: nothing in
                                   this repo produces it yet.
   - ``legacy_test_selection``  — pre-protocol-v2 runs that used the test set for
                                   selection; not valid evidence. ``evaluation.comparison``
                                   currently reads the test set during screening — exactly
                                   the rule-3 violation ``docs/decisions.md`` already
                                   flags — so those runs get this tag, not ``cv``, so they
                                   can never be mistaken for a protocol-v2-clean screen.
   - ``final_test``             — the one sealed read of the NASA test set per locked
                                   candidate (CLAUDE.md rule 3). Nothing in this repo
                                   produces this tag yet.
   - ``train_prod``             — ``training/train.py``, which ships a bundle.

   Tracking is opt-out, not opt-in: ``run()`` raises ``TrackingUnavailable`` unless a
   reachable MLflow backend is configured (``MLFLOW_TRACKING_URI``, default
   ``http://localhost:5000``), so a run is never silently un-logged. Pass ``track=False``
   (``--no-track`` in ``training/train.py``) to skip tracking entirely.

2. ``BundleModel`` / ``log_production_model`` / ``register_production_model`` — an
   ``mlflow.pyfunc`` wrapper around a turbofan artifact bundle (model + feature_state +
   manifest, see ``training.bundle``), so the exact same bundle training writes and
   ``serving.service`` reads can also be registered in the MLflow Model Registry, named
   ``turbofan-rul-<dataset>`` and promoted via aliases (``challenger`` / ``champion``),
   not the deprecated stage API. Nothing calls ``register_production_model`` yet — this is
   the tested code path for when a later step decides to promote a model, not an
   automatic action of any run.
"""

from __future__ import annotations

import os
import platform
from collections.abc import Iterator
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any, Literal, cast

import mlflow
import pandas as pd
import yaml
from mlflow.entities.model_registry import ModelVersion
from mlflow.models.model import ModelInfo
from mlflow.pyfunc import log_model as _pyfunc_log_model
from mlflow.pyfunc.model import PythonModel, PythonModelContext

from turbofan import config as cfg
from turbofan.serving.schemas import CycleReading
from turbofan.serving.service import predict_rul
from turbofan.training.bundle import Bundle, git_provenance, load_bundle_dir

__all__ = [
    "TrackingUnavailable",
    "run",
    "BundleModel",
    "log_production_model",
    "register_production_model",
]

RunType = Literal["cv", "legacy_test_selection", "final_test", "train_prod"]

DEFAULT_TRACKING_URI = "http://localhost:5000"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRACKED_LIBS = ("mlflow", "torch", "xgboost", "scikit-learn", "numpy", "pandas")


class TrackingUnavailable(RuntimeError):
    """No MLflow backend is reachable and tracking was not explicitly disabled."""


def _lib_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": platform.python_version()}
    for name in _TRACKED_LIBS:
        try:
            out[name] = _pkg_version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def _config_params() -> dict[str, str]:
    """Every top-level constant in turbofan.config (its UPPER_SNAKE_CASE names),
    stringified for mlflow.log_params. params.yaml replaces this as the source of
    truth in a later step; for now config.py is read directly so nothing drifts
    between what's logged and what the code actually ran with.

    Prefixed with ``cfg_`` so e.g. config.SEED (the registry's default seed) can never
    collide with this run's own ``seed`` param — MLflow's file-store backend stores each
    param as a same-named file, and 'SEED' vs 'seed' collide on a case-insensitive
    filesystem (Windows, default macOS).
    """
    return {f"cfg_{name}": str(value) for name, value in vars(cfg).items() if name.isupper()}


def _dvc_data_hash(dvc_file: Path | None = None) -> str:
    """The ``outs[0].md5`` hash from ``data/raw.dvc`` — the data version this run used."""
    dvc_file = dvc_file or (_REPO_ROOT / "data" / "raw.dvc")
    if not dvc_file.exists():
        raise FileNotFoundError(
            f"{dvc_file} not found — run `dvc add data/raw` (or `dvc pull` to restore the "
            "existing pointer) so this run's data provenance can be recorded"
        )
    doc = yaml.safe_load(dvc_file.read_text())
    outs = doc.get("outs") or []
    if not outs or "md5" not in outs[0]:
        raise ValueError(f"{dvc_file} has no usable md5 hash: {doc!r}")
    return str(outs[0]["md5"])


def _tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def _check_reachable(uri: str) -> None:
    # mlflow's HTTP client defaults to 7 retries with exponential backoff (tens of seconds
    # before it gives up) — fine for a flaky request mid-run, useless for a one-shot
    # reachability probe, so this check overrides both to fail fast.
    prev_retries = os.environ.get("MLFLOW_HTTP_REQUEST_MAX_RETRIES")
    prev_timeout = os.environ.get("MLFLOW_HTTP_REQUEST_TIMEOUT")
    os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "1"
    os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "3"
    try:
        mlflow.MlflowClient(tracking_uri=uri).search_experiments(max_results=1)
    except Exception as e:
        raise TrackingUnavailable(
            f"No MLflow backend reachable at {uri!r} ({e}). Start one "
            "(`docker compose up mlflow`), point MLFLOW_TRACKING_URI elsewhere, or pass "
            "track=False / --no-track to skip tracking for this run — tracking does not "
            "silently no-op."
        ) from e
    finally:
        for var, prev in (
            ("MLFLOW_HTTP_REQUEST_MAX_RETRIES", prev_retries),
            ("MLFLOW_HTTP_REQUEST_TIMEOUT", prev_timeout),
        ):
            if prev is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = prev


@contextmanager
def run(
    *,
    dataset: str,
    model: str,
    seed: int,
    run_type: RunType,
    track: bool = True,
    nested: bool = False,
    extra_params: dict[str, Any] | None = None,
    extra_tags: dict[str, str] | None = None,
) -> Iterator[mlflow.ActiveRun | None]:
    """Context manager for one MLflow run; yields ``None`` and logs nothing if ``track=False``.

    Always logs: git commit SHA + dirty flag, the DVC data hash, every ``turbofan.config``
    value, library versions, ``dataset``/``model``/``seed``, and the ``run_type`` tag.
    """
    if not track:
        yield None
        return

    uri = _tracking_uri()
    mlflow.set_tracking_uri(uri)
    _check_reachable(uri)

    # Gather everything before opening the run so a missing/bad provenance source fails
    # fast, with no half-populated run left behind.
    prov = git_provenance()
    dvc_hash = _dvc_data_hash()
    params = {"dataset": dataset, "model": model, "seed": seed, **_config_params()}
    if extra_params:
        params.update({k: str(v) for k, v in extra_params.items()})

    with mlflow.start_run(nested=nested) as active_run:
        mlflow.log_params(params)
        tags = {
            "run_type": run_type,
            "git_commit": prov["git_commit"],
            "git_dirty": str(prov["git_dirty"]),
            "dvc_data_hash": dvc_hash,
        }
        tags.update({f"lib.{lib}": ver or "not installed" for lib, ver in _lib_versions().items()})
        if extra_tags:
            tags.update(extra_tags)
        mlflow.set_tags(tags)
        yield active_run


# -- production-model registration (pyfunc) ----------------------------------------------


class BundleModel(PythonModel):
    """Wraps one turbofan artifact bundle (model + feature_state + manifest) as a single
    deployable mlflow.pyfunc model — the same bundle format training writes and
    serving.service reads, so a registered model and the FastAPI service never diverge.

    ``predict`` takes a DataFrame of one engine's cycle history, chronological (oldest
    first): columns ``op1``, ``op2``, ``op3``, plus every sensor in ``config.KEEP``. It
    returns the same shape as ``POST /predict`` with ``confidence`` flattened into
    ``confidence_error_band_cycles`` / ``confidence_basis`` columns.
    """

    _bundle: Bundle

    def load_context(self, context: PythonModelContext) -> None:
        self._bundle = load_bundle_dir(context.artifacts["bundle"], device="cpu")

    def predict(
        self,
        context: PythonModelContext,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        cycles = [
            CycleReading(
                op1=float(row["op1"]),
                op2=float(row["op2"]),
                op3=float(row["op3"]),
                sensors={s: float(row[s]) for s in cfg.KEEP},
            )
            for _, row in model_input.iterrows()
        ]
        result = predict_rul(cycles, self._bundle)
        confidence = result.pop("confidence")
        result.update({f"confidence_{k}": v for k, v in confidence.items()})
        return pd.DataFrame([result])


def log_production_model(bundle_dir: str | Path, dataset: str) -> ModelInfo:
    """Log a bundle as an mlflow.pyfunc model under the currently active run.

    Caller is responsible for the surrounding ``with tracking.run(...):`` and for
    registering the result (``register_production_model``) — this only logs the model
    artifact.
    """
    return cast(
        ModelInfo,
        _pyfunc_log_model(
            name=f"turbofan-rul-{dataset}",
            python_model=BundleModel(),
            artifacts={"bundle": str(bundle_dir)},
        ),
    )


def register_production_model(
    model_uri: str, dataset: str, alias: Literal["challenger", "champion"] = "challenger"
) -> ModelVersion:
    """Register a logged model into the MLflow Model Registry and set an alias.

    Uses aliases, not the deprecated stage API. Requires a DB-backed MLflow backend (the
    docker-compose sqlite server qualifies; a plain local file store does not in
    production, though mlflow's file-store registry support is enough for tests).
    """
    name = f"turbofan-rul-{dataset}"
    model_version = mlflow.register_model(model_uri, name)
    mlflow.MlflowClient().set_registered_model_alias(name, alias, model_version.version)
    return model_version
