"""Stage ``train_prod``: refit the locked spec on **all** training engines and write a bundle.

The locked spec (``release.spec``, ``evaluation.spec`` format) names the candidate. The model is
fitted on every training engine — no inner validation share — for the iteration budget CV found:
the median early-stopped round / epoch of that model over ``cv_select``'s splits
(``reports/cv_select/metrics.json``, D48). Feature state (regimes, normalization, envelope) is
fitted on all training engines too. Reads training files only; the bundle's manifest carries no
test-set metric (rule 3).

Writes ``models/prod/<dataset>/<model>/current`` (a bundle ``serving`` and ``register`` read),
``reports/train_prod/metrics.json`` and an MLflow run (``run_type=train_prod``).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import mlflow

from turbofan import tracking
from turbofan.config import KEEP, RUL_CAP
from turbofan.data.loader import load_train
from turbofan.evaluation.comparison import build_registry
from turbofan.evaluation.cv import true_rul
from turbofan.evaluation.fitting import refit_candidate
from turbofan.evaluation.spec import load_spec, spec_hash
from turbofan.features.engineering import add_features
from turbofan.pipeline.context import Context, write_json
from turbofan.training.bundle import bundle_dir, save_bundle

BUNDLE_VERSION = "current"


def iteration_budget_from_cv(cv_metrics: Path, dataset: str, model: str) -> int | None:
    """Median early-stopped iteration of ``model`` in the cv_select metrics (None for models
    with no iteration count). Refuses metrics from another dataset or cap."""
    if not cv_metrics.is_file():
        raise FileNotFoundError(f"{cv_metrics} missing — run the cv_select stage first")
    doc = json.loads(cv_metrics.read_text(encoding="utf-8"))
    if doc["dataset"] != dataset or float(doc["rul_cap"]) != RUL_CAP:
        raise ValueError(
            f"{cv_metrics} is for {doc['dataset']} at rul_cap {doc['rul_cap']}; the spec needs "
            f"{dataset} at {RUL_CAP} — rerun cv_select"
        )
    if model not in doc["models"]:
        raise ValueError(
            f"{cv_metrics} has no results for {model!r} (has {sorted(doc['models'])}); the "
            "locked candidate must have been through cv_select"
        )
    med = doc["models"][model].get("iteration_budget_median")
    return None if med is None else max(1, round(float(med)))


def run(ctx: Context) -> Path:
    spec = load_spec(ctx.spec)
    ds, name, seed = spec["dataset"], spec["model"], int(spec["seed"])
    if ds != ctx.dataset:
        raise ValueError(f"spec dataset {ds} != params pipeline.dataset {ctx.dataset}")
    budget = iteration_budget_from_cv(ctx.cv_metrics, ds, name)

    train = load_train(ds, ctx.raw)
    if ctx.smoke:
        engines = json.loads(ctx.engines_file.read_text(encoding="utf-8"))["engines"]
        train = train[train["unit"].isin(engines)]
    d = train.copy()
    d["rul_true"] = true_rul(d)
    d["rul"] = d["rul_true"].clip(upper=RUL_CAP)
    feat_all, stats = add_features(d, KEEP, ds)
    n_engines, n_rows = int(feat_all["unit"].nunique()), len(feat_all)

    h = spec_hash(spec)
    with tracking.run(
        dataset=ds,
        model=name,
        seed=seed,
        run_type="train_prod",
        extra_params={
            "rul_cap": RUL_CAP,
            "refit": "all_training_engines",
            "iteration_budget": budget,
            "n_engines": n_engines,
        },
        extra_tags={"spec_hash": h},
    ) as active:
        cand = build_registry(seed=seed)[name]
        model = refit_candidate(cand.factory(), cand.kind, feat_all, budget)
        out = bundle_dir(ctx.models_prod, ds, name, BUNDLE_VERSION)
        if out.exists():
            shutil.rmtree(out)  # a refit replaces the bundle, it never merges into one
        refit_stats: dict[str, float] = {"n_engines": float(n_engines), "n_rows": float(n_rows)}
        if budget is not None:
            refit_stats["iteration_budget"] = float(budget)
        d_out = save_bundle(
            ctx.models_prod,
            ds,
            name,
            BUNDLE_VERSION,
            model,
            stats,
            seed=seed,
            metrics={"refit": refit_stats},
        )
        if active is not None:
            mlflow.log_metrics(refit_stats)
            mlflow.log_artifacts(str(d_out), artifact_path="bundle")
        doc: dict[str, Any] = {
            "dataset": ds,
            "model": name,
            "seed": seed,
            "rul_cap": RUL_CAP,
            "spec_hash": h,
            "bundle": d_out.relative_to(ctx.root).as_posix(),
            "mlflow_run_id": active.info.run_id if active is not None else None,
            **refit_stats,
        }
    write_json(ctx.train_metrics, doc)
    return d_out
