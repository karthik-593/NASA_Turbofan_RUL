"""The one sealed read of the NASA test set per locked candidate (rule 3, D46).

    python -m turbofan.evaluation.final_eval --spec locked.yaml --confirm

The spec (YAML) names one candidate exactly: ``dataset``, ``model`` (registry name),
``seed``, ``rul_cap``, and must say ``locked: true`` — it is the outcome of protocol-v2
selection on training data, frozen before this runs. The candidate is retrained on all
training engines (a stratified inner share for early stopping only, as in CV), then scored
once on ``test_FD00x.txt`` against ``RUL_FD00x.txt``: one prediction per test engine at its
last observed cycle, metrics with engine-bootstrap CIs. Logged to MLflow with
``run_type=final_test`` and the spec's hash; a second run for the same spec hash is refused.

Refuses to do anything without ``--confirm``. Selection code must never import this module
(``tests/test_final_eval.py`` enforces it): nothing in model selection may reach the test set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from turbofan import tracking
from turbofan.analysis.subpopulation import subpopulation_labels
from turbofan.config import DATASETS, KEEP, PARAMS, RUL_CAP
from turbofan.data.loader import load_dataset, load_train
from turbofan.evaluation.comparison import build_registry
from turbofan.evaluation.cv import true_rul
from turbofan.evaluation.cv_metrics import metric_table
from turbofan.evaluation.fitting import fit_candidate, predict_last_cycle
from turbofan.features.engineering import add_features

__all__ = ["load_spec", "spec_hash", "main"]

REQUIRED = ("dataset", "model", "seed", "rul_cap", "locked")


class FinalEvalRefused(RuntimeError):
    """final_eval declined to read the test set."""


def load_spec(path: str | Path) -> dict[str, Any]:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise FinalEvalRefused(f"{path}: a spec is a YAML mapping")
    missing = [k for k in REQUIRED if k not in spec]
    if missing:
        raise FinalEvalRefused(f"{path}: spec lacks {missing}")
    if spec["locked"] is not True:
        raise FinalEvalRefused(
            f"{path}: spec is not locked (locked: true) — finish selection first"
        )
    if spec["dataset"] not in DATASETS:
        raise FinalEvalRefused(f"{path}: unknown dataset {spec['dataset']!r}")
    if float(spec["rul_cap"]) != RUL_CAP:
        raise FinalEvalRefused(
            f"{path}: rul_cap {spec['rul_cap']} != params.yaml rul_cap {RUL_CAP}; model "
            "clipping follows params.yaml, so set rul_cap there before a final evaluation"
        )
    return spec


def spec_hash(spec: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


def _already_evaluated(h: str) -> list[str]:
    mlflow.set_tracking_uri(tracking._tracking_uri())
    exp = mlflow.get_experiment_by_name(tracking._experiment_name())
    if exp is None:
        return []
    runs = mlflow.MlflowClient().search_runs(
        [exp.experiment_id],
        filter_string=f"tags.run_type = 'final_test' and tags.spec_hash = '{h}'",
    )
    return [r.info.run_id for r in runs]


def evaluate(spec: dict[str, Any], raw: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Retrain on all training engines and score the test set once. Returns (metric table,
    run details)."""
    ds, seed, cap = spec["dataset"], int(spec["seed"]), float(spec["rul_cap"])
    train = load_train(ds, raw)
    labels = subpopulation_labels(train, ds)
    tr_u, va_u = train_test_split(
        labels.index.to_numpy(),
        test_size=PARAMS["cv"]["inner_val_frac"],
        stratify=labels.to_numpy(),
        random_state=seed,
    )
    d = train.copy()
    d["rul_true"] = true_rul(d)
    d["rul"] = d["rul_true"].clip(upper=cap)
    feat_tr, stats = add_features(d[d["unit"].isin(tr_u)], KEEP, ds)
    feat_va, _ = add_features(d[d["unit"].isin(va_u)], KEEP, ds, stats=stats)
    cand = build_registry(seed=seed)[spec["model"]]
    model = fit_candidate(cand.factory(), cand.kind, feat_tr, feat_va)

    _tr, test, rul_test = load_dataset(ds, raw)  # the sealed read
    feat_te, _ = add_features(test, KEEP, ds, stats=stats)
    units, pred = predict_last_cycle(model, cand.kind, feat_te)
    points = pd.DataFrame(
        {"unit": units, "rul_true": rul_test.reindex(units).to_numpy(float), "pred": pred}
    )
    table = metric_table(points, cap, np.random.default_rng(seed))
    return table, {"n_test_engines": len(points)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Score one locked candidate on the NASA test set.")
    ap.add_argument("--spec", required=True, help="locked candidate spec (YAML)")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument(
        "--confirm",
        action="store_true",
        help="required: this reads the sealed test set, once per locked candidate",
    )
    args = ap.parse_args(argv)
    if not args.confirm:
        raise FinalEvalRefused(
            "final_eval reads the sealed NASA test set and may run once per locked candidate; "
            "re-run with --confirm if that is what you intend"
        )
    spec = load_spec(args.spec)
    h = spec_hash(spec)
    previous = _already_evaluated(h)
    if previous:
        raise FinalEvalRefused(f"spec {h[:12]} was already scored on the test set: runs {previous}")

    with tracking.run(
        dataset=spec["dataset"],
        model=spec["model"],
        seed=int(spec["seed"]),
        run_type="final_test",
        extra_params={"rul_cap": spec["rul_cap"]},
        extra_tags={"spec_hash": h},
    ) as active:
        table, info = evaluate(spec, args.raw)
        assert active is not None
        mlflow.log_dict(spec, "locked_spec.yaml")
        mlflow.log_text(table.to_csv(index=False), "final_test_metrics.csv")
        mlflow.log_metrics(
            {
                f"test_{t}_{m}": float(v)
                for t, m, v in zip(table["truth"], table["metric"], table["estimate"], strict=True)
            }
        )
        mlflow.log_metrics({k: float(v) for k, v in info.items()})
    print(table[table["truth"] == "uncapped"].to_string(index=False))


if __name__ == "__main__":
    main()
