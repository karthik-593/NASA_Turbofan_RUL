"""The one sealed read of the NASA test set per locked candidate (rule 3, D46, D48).

    python -m turbofan.evaluation.final_eval --spec locked.yaml --bundle <dir> --confirm

The spec (YAML) names one candidate exactly: ``dataset``, ``model`` (registry name),
``seed``, ``rul_cap``, and must say ``locked: true`` — it is the outcome of protocol-v2
selection on training data, frozen before this runs. What is scored is the **shipped artifact**:
the ``train_prod`` bundle built from this very spec (its manifest must carry the spec's hash and
a ``refit`` record, and no test-set metric). The bundle's model and feature state score
``test_FD00x.txt`` against ``RUL_FD00x.txt`` once: one prediction per test engine at its last
observed cycle, metrics with engine-bootstrap CIs. Logged to MLflow with ``run_type=final_test``
and the spec's hash; a second run for the same spec hash is refused.

Before scoring it checks the test files' structure only (schema, row / unit counts against
``configs/data_expectations.json``, ``data.integrity``) and refuses on any deviation.

Refuses to do anything without ``--confirm``. Selection code must never import this module
(``tests/test_final_eval.py`` enforces it): nothing in model selection may reach the test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from turbofan import tracking
from turbofan.config import KEEP, PARAMS
from turbofan.data.integrity import check_test_integrity, load_expectations
from turbofan.data.loader import load_dataset
from turbofan.evaluation.comparison import build_registry
from turbofan.evaluation.cv_metrics import metric_table
from turbofan.evaluation.fitting import predict_last_cycle
from turbofan.evaluation.spec import FinalEvalRefused, load_spec, spec_hash
from turbofan.features.engineering import add_features
from turbofan.training.bundle import MANIFEST_FILE, Bundle, load_bundle_dir

__all__ = ["FinalEvalRefused", "load_spec", "spec_hash", "main"]


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


def check_bundle(spec: dict[str, Any], bundle: str | Path) -> dict[str, Any]:
    """Refuse a bundle that is not the shipped artifact of this spec; returns its manifest.
    Reads the manifest only — no model is loaded and no data is touched."""
    path = Path(bundle) / MANIFEST_FILE
    if not path.is_file():
        raise FinalEvalRefused(f"{bundle}: no {MANIFEST_FILE} — not a bundle directory")
    man: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    for key in ("dataset", "model", "seed"):
        if man.get(key) != spec[key]:
            raise FinalEvalRefused(
                f"{bundle}: bundle {key} {man.get(key)!r} != spec {key} {spec[key]!r}"
            )
    if man.get("spec_hash") != spec_hash(spec):
        raise FinalEvalRefused(
            f"{bundle}: bundle was not built from this spec (spec_hash "
            f"{str(man.get('spec_hash'))[:12]} != {spec_hash(spec)[:12]}) — rerun train_prod"
        )
    metrics = man.get("metrics", {})
    if "refit" not in metrics or "test" in metrics:
        raise FinalEvalRefused(
            f"{bundle}: not a train_prod bundle (needs a 'refit' record and no 'test' metrics)"
        )
    return man


def evaluate(
    spec: dict[str, Any], raw: str | Path, bundle: Bundle
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score the bundle on the test set once. Returns (metric table, run details)."""
    ds, seed, cap = spec["dataset"], int(spec["seed"]), float(spec["rul_cap"])
    kind = build_registry(seed=seed)[spec["model"]].kind
    _tr, test, rul_test = load_dataset(ds, raw)  # the sealed read
    feat_te, _ = add_features(test, KEEP, ds, stats=bundle.stats)
    units, pred = predict_last_cycle(bundle.model, kind, feat_te)
    points = pd.DataFrame(
        {"unit": units, "rul_true": rul_test.reindex(units).to_numpy(float), "pred": pred}
    )
    table = metric_table(points, cap, np.random.default_rng(seed))
    return table, {"n_test_engines": len(points)}


def _metrics_doc(
    spec: dict[str, Any], h: str, table: pd.DataFrame, info: dict[str, Any]
) -> dict[str, Any]:
    doc: dict[str, Any] = {
        "ran": True,
        "dataset": spec["dataset"],
        "model": spec["model"],
        "spec_hash": h,
        **info,
    }
    for truth, g in table.groupby("truth"):
        doc[str(truth)] = {
            r.metric: {
                "estimate": r.estimate,
                "ci_lo": r.ci_lo,
                "ci_hi": r.ci_hi,
                "n_engines": r.n_engines,
                "n_points": r.n_points,
            }
            for r in g.itertuples()
        }
    return doc


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Score one locked candidate on the NASA test set.")
    ap.add_argument("--spec", required=True, help="locked candidate spec (YAML)")
    ap.add_argument("--bundle", required=True, help="the train_prod bundle directory to score")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument(
        "--expectations",
        default=str(Path(__file__).resolve().parents[3] / PARAMS["final_eval"]["expectations"]),
        help="recorded test-file row / unit counts (default: params final_eval.expectations)",
    )
    ap.add_argument("--metrics-out", help="also write the metric table here as JSON (DVC metrics)")
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
    manifest = check_bundle(spec, args.bundle)
    # structure only (rule 3): schema and counts of the test files, before anything is scored
    integrity = check_test_integrity(
        spec["dataset"], args.raw, load_expectations(args.expectations)
    )
    bundle = load_bundle_dir(args.bundle)

    with tracking.run(
        dataset=spec["dataset"],
        model=spec["model"],
        seed=int(spec["seed"]),
        run_type="final_test",
        extra_params={"rul_cap": spec["rul_cap"]},
        extra_tags={"spec_hash": h, "bundle_version": str(manifest["version"])},
    ) as active:
        table, info = evaluate(spec, args.raw, bundle)
        assert active is not None
        mlflow.log_dict(spec, "locked_spec.yaml")
        mlflow.log_dict(integrity, "test_integrity.json")
        mlflow.log_text(table.to_csv(index=False), "final_test_metrics.csv")
        mlflow.log_metrics(
            {
                f"test_{t}_{m}": float(v)
                for t, m, v in zip(table["truth"], table["metric"], table["estimate"], strict=True)
            }
        )
        mlflow.log_metrics({k: float(v) for k, v in info.items()})
    print(table[table["truth"] == "uncapped"].to_string(index=False))
    if args.metrics_out:
        out = Path(args.metrics_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_metrics_doc(spec, h, table, info), indent=2) + "\n", "utf-8")


if __name__ == "__main__":
    main()
