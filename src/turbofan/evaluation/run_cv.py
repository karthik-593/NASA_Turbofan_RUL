"""Protocol-v2 cross-validation runner (D46). Reads training files only.

    python -m turbofan.evaluation.run_cv --dataset FD001 --models mean ridge xgboost lstm

Per dataset: subpopulation labels (D35) -> repeated stratified group folds -> per fold, the
feature state fitted on its inner-training engines (built once, shared by every model) ->
per model x fold x seed, fit and predict at every held-out cycle -> deployment and benchmark
views -> metric tables with engine-bootstrap CIs (overall and per subpopulation) and decision
curves.

MLflow (D46/D43): one parent run per model (``run_type=cv``) holding the aggregated metrics
and the tables/predictions as artifacts, one nested child run per fold x seed with that
split's metrics; every run carries the git/DVC/device tags of ``tracking.run``.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from turbofan import tracking
from turbofan.analysis.subpopulation import subpopulation_labels
from turbofan.config import PARAMS, RUL_CAP, SEED
from turbofan.data.loader import load_train
from turbofan.evaluation.comparison import build_registry
from turbofan.evaluation.cv import FoldData, make_folds, prepare_fold
from turbofan.evaluation.cv_metrics import (
    HEADLINE,
    engine_aggregates,
    metric_table,
    metrics_from_weights,
)
from turbofan.evaluation.decision import decision_curve
from turbofan.evaluation.fitting import fit_candidate, predict_every_cycle
from turbofan.evaluation.views import benchmark_view, deployment_view, nasa_test_label_histogram

__all__ = ["ModelResult", "CVRun", "run_cv", "main"]

_CV = PARAMS["cv"]
GROUPS = ("all", "subpop_0", "subpop_1")
# (view, truth, metric) rows that are not reported (D46): the NASA score against uncapped truth
# in the deployment view is dominated by early-life cycles no capped model can reach
NOT_REPORTED = frozenset({("deployment", "uncapped", "nasa_mean_per_engine")})


@dataclass
class ModelResult:
    model: str
    rul_cap: float
    points: pd.DataFrame  # every held-out cycle: KEYS + rul_true, pred, subpop
    benchmark: pd.DataFrame
    benchmark_shortfall: dict[int, int]
    metrics: pd.DataFrame  # view, group, truth, metric, estimate, ci_lo, ci_hi, n_...
    decision: pd.DataFrame
    fit_seconds: float  # fit + predict wall time summed over fold x seed
    parent_run_id: str | None = None


@dataclass
class CVRun:
    dataset: str
    n_folds: int
    n_repeats: int
    seeds: list[int]
    subpop_sizes: dict[int, int]
    feature_seconds: float
    results: dict[str, ModelResult] = field(default_factory=dict)


def _split_metrics(pts: pd.DataFrame, rul_cap: float) -> dict[str, float]:
    one: dict[str, dict[str, Any]] = {}
    for truth in ("uncapped", "capped"):
        agg = engine_aggregates(pts, truth, rul_cap)
        one[truth] = metrics_from_weights(agg, np.ones((1, len(agg))))
    out = {k: float(one["uncapped"][k][0]) for k in (HEADLINE, "rmse", "mae")}
    out["nasa_mean_per_engine_capped"] = float(one["capped"]["nasa_mean_per_engine"][0])
    return out


def _evaluate(
    pts: pd.DataFrame, label_share: pd.Series[float], rul_cap: float, seed: int
) -> tuple[pd.DataFrame, dict[int, int], pd.DataFrame, pd.DataFrame]:
    deploy = deployment_view(pts)
    bench_parts: list[pd.DataFrame] = []
    shortfall: dict[int, int] = {}
    for _key, g in deploy.groupby(["repeat", "fold", "seed"]):
        split = [int(g[c].iloc[0]) for c in ("repeat", "fold", "seed")]
        rng = np.random.default_rng([seed, *split])
        b, short = benchmark_view(g, label_share, rng)
        bench_parts.append(b)
        for k, v in short.items():
            shortfall[k] = shortfall.get(k, 0) + v
    bench = pd.concat(bench_parts, ignore_index=True)
    tables: list[pd.DataFrame] = []
    for view, frame in (("deployment", deploy), ("benchmark", bench)):
        for group in GROUPS:
            sub = frame if group == "all" else frame[frame["subpop"] == int(group[-1])]
            t = metric_table(sub, rul_cap, np.random.default_rng([seed, len(tables)]))
            tables.append(t.assign(view=view, group=group))
    metrics = pd.concat(tables, ignore_index=True)
    dropped = pd.MultiIndex.from_frame(metrics[["view", "truth", "metric"]]).isin(NOT_REPORTED)
    metrics = metrics[~dropped].reset_index(drop=True)
    curve = decision_curve(deploy, np.random.default_rng([seed, 999]))
    return bench, shortfall, metrics, curve


def run_cv(
    dataset: str,
    models: list[str],
    seeds: list[int],
    raw: str | Path = "data/raw",
    track: bool = True,
    rul_cap: float = RUL_CAP,
    n_folds: int = _CV["n_folds"],
    n_repeats: int = _CV["n_repeats"],
    verbose: bool = True,
) -> CVRun:
    train = load_train(dataset, raw)
    labels = subpopulation_labels(train, dataset)
    folds = make_folds(labels, n_folds=n_folds, n_repeats=n_repeats, seed=_CV["seed"])
    label_share = nasa_test_label_histogram(dataset, raw)

    t0 = time.perf_counter()
    fold_data: list[FoldData] = [prepare_fold(train, dataset, f, rul_cap) for f in folds]
    run = CVRun(
        dataset=dataset,
        n_folds=n_folds,
        n_repeats=n_repeats,
        seeds=list(seeds),
        subpop_sizes=dict(
            zip(
                labels.value_counts().sort_index().index.to_numpy(int).tolist(),
                labels.value_counts().sort_index().to_numpy(int).tolist(),
                strict=True,
            )
        ),
        feature_seconds=time.perf_counter() - t0,
    )
    if verbose:
        print(f"{dataset}: {len(folds)} folds prepared in {run.feature_seconds:.0f} s", flush=True)

    for name in models:
        common = {
            "n_folds": n_folds,
            "n_repeats": n_repeats,
            "seeds": list(seeds),
            "rul_cap": rul_cap,
        }
        with tracking.run(
            dataset=dataset,
            model=name,
            seed=-1,
            run_type="cv",
            track=track,
            extra_params=common,
            extra_tags={"cv_role": "parent", "protocol": "v2"},
        ) as parent:
            parts, fit_s = [], 0.0
            for fd in fold_data:
                f = fd.fold
                for s in seeds:
                    with tracking.run(
                        dataset=dataset,
                        model=name,
                        seed=s,
                        run_type="cv",
                        track=track,
                        nested=parent is not None,
                        extra_params={**common, "repeat": f.repeat, "fold": f.fold},
                        extra_tags={"cv_role": "child", "protocol": "v2"},
                    ) as child:
                        cand = build_registry(seed=s)[name]
                        t1 = time.perf_counter()
                        model = fit_candidate(cand.factory(), cand.kind, fd.feat_tr, fd.feat_va)
                        pts = predict_every_cycle(model, cand.kind, fd.feat_te)
                        dt = time.perf_counter() - t1
                        fit_s += dt
                        pts = pts.assign(repeat=f.repeat, fold=f.fold, seed=s)
                        pts["subpop"] = labels.loc[pts["unit"]].to_numpy()
                        parts.append(pts)
                        if child is not None:
                            m = _split_metrics(deployment_view(pts), rul_cap)
                            mlflow.log_metrics({**m, "fit_predict_seconds": dt})
                if verbose:
                    print(f"  {name:8s} repeat {f.repeat} fold {f.fold} done", flush=True)
            points = pd.concat(parts, ignore_index=True)
            bench, short, metrics, curve = _evaluate(points, label_share, rul_cap, SEED)
            res = ModelResult(
                model=name,
                rul_cap=rul_cap,
                points=points,
                benchmark=bench,
                benchmark_shortfall=short,
                metrics=metrics,
                decision=curve,
                fit_seconds=fit_s,
                parent_run_id=parent.info.run_id if parent is not None else None,
            )
            if parent is not None:
                _log_parent(res)
            run.results[name] = res
            if verbose:
                h = metrics[
                    (metrics["view"] == "deployment")
                    & (metrics["group"] == "all")
                    & (metrics["truth"] == "uncapped")
                    & (metrics["metric"] == HEADLINE)
                ].iloc[0]
                print(
                    f"{name:8s} {HEADLINE} {h['estimate']:.2f} [{h['ci_lo']:.2f}, {h['ci_hi']:.2f}]"
                    f"  fit+predict {fit_s:.0f} s",
                    flush=True,
                )
    return run


def _log_parent(res: ModelResult) -> None:
    m = res.metrics
    flat: dict[str, Any] = {}
    sel = m[(m["group"] == "all")]
    for r in sel.itertuples():
        base = f"{r.view}_{r.truth}_{r.metric}"
        flat[base] = r.estimate
        flat[f"{base}_ci_lo"] = r.ci_lo
        flat[f"{base}_ci_hi"] = r.ci_hi
    flat["fit_predict_seconds_total"] = res.fit_seconds
    mlflow.log_metrics({k: float(v) for k, v in flat.items() if np.isfinite(v)})
    mlflow.log_text(m.to_csv(index=False), "cv_metrics.csv")
    mlflow.log_text(res.decision.to_csv(index=False), "decision_curve.csv")
    mlflow.log_text(res.points.to_csv(index=False), "heldout_predictions.csv")
    mlflow.log_dict(
        {str(k): v for k, v in res.benchmark_shortfall.items()}, "benchmark_shortfall.json"
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Protocol-v2 cross-validation (training data only).")
    ap.add_argument("--dataset", required=True, choices=("FD001", "FD002", "FD003", "FD004"))
    ap.add_argument("--models", nargs="+", default=["mean", "ridge", "xgboost", "lstm"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[SEED])
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--no-track", action="store_true")
    args = ap.parse_args(argv)
    run_cv(args.dataset, args.models, args.seeds, args.raw, track=not args.no_track)


if __name__ == "__main__":
    main()
