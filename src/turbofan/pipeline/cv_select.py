"""Stage ``cv_select``: protocol-v2 cross-validation, resumable, report + metrics + plots.

Runs ``evaluation.report_v2.run_and_report`` (training files only; MLflow-tracked; splits
MLflow already holds as FINISHED for this exact config and source are reloaded, D49) and
distils the run into the files DVC tracks:

- ``reports/protocol_v2.md`` + figures + env sidecar (from the report writer);
- ``reports/cv_select/metrics.json``: per model the headline critical-bucket RMSE with its
  engine-bootstrap CI and n, other deployment-view summaries, and the median early-stopped
  iteration budget over splits (what ``train_prod`` refits with);
- ``reports/cv_select/plots/*.csv``: decision curves, one column per model.

Selection never imports ``final_eval`` or the test-reading loader (tests/test_pipeline.py).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from turbofan.analysis.subpopulation import subpopulation_labels
from turbofan.config import RUL_CAP
from turbofan.data.loader import load_train
from turbofan.evaluation.cv_metrics import HEADLINE
from turbofan.evaluation.decision import MATCHED_LEAD_TIME
from turbofan.evaluation.report_v2 import run_and_report
from turbofan.evaluation.run_cv import CVRun
from turbofan.pipeline.context import Context, select_smoke_engines, write_json


def _pick(m: pd.DataFrame, truth: str, metric: str) -> pd.Series[Any]:
    sel = m[
        (m["view"] == "deployment")
        & (m["group"] == "all")
        & (m["truth"] == truth)
        & (m["metric"] == metric)
    ]
    if len(sel) != 1:
        raise RuntimeError(f"expected one deployment/all/{truth}/{metric} row, got {len(sel)}")
    return sel.iloc[0]


def metrics_doc(run: CVRun) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for name, res in run.results.items():
        h = _pick(res.metrics, "uncapped", HEADLINE)
        entry: dict[str, Any] = {
            HEADLINE: float(h["estimate"]),
            f"{HEADLINE}_ci_lo": float(h["ci_lo"]),
            f"{HEADLINE}_ci_hi": float(h["ci_hi"]),
            "n_engines": int(h["n_engines"]),
            "n_points": int(h["n_points"]),
            "rmse": float(_pick(res.metrics, "uncapped", "rmse")["estimate"]),
            "mae": float(_pick(res.metrics, "uncapped", "mae")["estimate"]),
            "nasa_mean_per_engine_capped": float(
                _pick(res.metrics, "capped", "nasa_mean_per_engine")["estimate"]
            ),
            "n_splits": len(res.points[["repeat", "fold", "seed"]].drop_duplicates()),
            "resumed_splits": res.resumed_splits,
        }
        if res.iteration_budgets:
            entry["iteration_budget_median"] = float(np.median(res.iteration_budgets))
        bad = [k for k, v in entry.items() if isinstance(v, float) and not math.isfinite(v)]
        if bad:
            raise ValueError(f"{name}: non-finite metrics {bad}")  # rule 8: no hidden NaN
        models[name] = entry
    return {"dataset": run.dataset, "rul_cap": RUL_CAP, "models": models}


def write_plot_data(ctx: Context, run: CVRun) -> None:
    ctx.cv_plots.mkdir(parents=True, exist_ok=True)
    for fname, quantity in (
        ("decision_caught.csv", f"caught_lead_ge_{MATCHED_LEAD_TIME}_pct"),
        ("decision_wasted.csv", "mean_wasted_life"),
    ):
        cols = {}
        for name, res in run.results.items():
            c = res.decision
            s = c[c["quantity"] == quantity].set_index("threshold")["estimate"]
            cols[name] = s
        wide = pd.DataFrame(cols)
        wide.index.name = "threshold"
        wide.reset_index().to_csv(ctx.cv_plots / fname, index=False)


def subpop_doc(run: CVRun) -> dict[str, Any]:
    """Headline per subpopulation (D35), with CI and n, per model."""
    out: dict[str, Any] = {}
    for name, res in run.results.items():
        m = res.metrics
        rows = m[
            (m["view"] == "deployment")
            & (m["truth"] == "uncapped")
            & (m["metric"] == HEADLINE)
            & (m["group"] != "all")
        ]
        out[name] = {
            str(r["group"]): {
                "estimate": float(r["estimate"]),
                "ci_lo": float(r["ci_lo"]),
                "ci_hi": float(r["ci_hi"]),
                "n_engines": int(r["n_engines"]),
            }
            for r in rows.to_dict("records")
        }
    return out


def run_selection(ctx: Context) -> CVRun:
    """A tagged model-selection run (D51): CV only — no protocol report — writing the run's
    metrics, its held-out predictions per model and its environment under ``selection/<tag>/``."""
    import time

    import torch

    from turbofan import repro
    from turbofan.config import FEATURE_BLOCKS, PARAMS, SEQ_LEN, WINDOW
    from turbofan.evaluation.run_cv import run_cv

    seeds = {f"model_seed_{s}": s for s in ctx.seeds} | {"cv_seed": PARAMS["cv"]["seed"]}
    env = repro.environment_context(seeds=seeds, n_threads=torch.get_num_threads())
    t0 = time.perf_counter()
    cv_run = run_cv(
        ctx.dataset,
        ctx.models,
        ctx.seeds,
        ctx.raw,
        track=True,
        resume=ctx.resume,
        n_folds=ctx.n_folds,
        n_repeats=ctx.n_repeats,
    )
    wall = time.perf_counter() - t0
    doc = metrics_doc(cv_run)
    doc.update(
        tag=ctx.tag,
        window=WINDOW,
        seq_len=SEQ_LEN,
        feature_blocks={
            "extra_sensors": list(FEATURE_BLOCKS.extra_sensors),
            "health_index": FEATURE_BLOCKS.health_index,
            "regime_onehot": FEATURE_BLOCKS.regime_onehot,
        },
        subpopulations=subpop_doc(cv_run),
        wall_seconds=wall,
        feature_seconds=cv_run.feature_seconds,
        fit_seconds={n: r.fit_seconds for n, r in cv_run.results.items()},
    )
    for name, res in cv_run.results.items():
        res.points.to_parquet(ctx.points(name), index=False)
    write_json(ctx.env_file, repro.with_device_state(env))
    write_json(ctx.cv_metrics, doc)  # written last: its presence marks a complete run
    return cv_run


def run(ctx: Context) -> CVRun:
    if ctx.tag:
        return run_selection(ctx)
    engines: list[int] | None = None
    if ctx.smoke:
        assert ctx.n_engines is not None
        engines = select_smoke_engines(
            subpopulation_labels(load_train(ctx.dataset, ctx.raw), ctx.dataset), ctx.n_engines
        )
        write_json(ctx.engines_file, {"dataset": ctx.dataset, "engines": engines})
    cv_run, _path = run_and_report(
        ctx.dataset,
        ctx.models,
        ctx.seeds,
        ctx.raw,
        resume=ctx.resume,
        max_folds=ctx.max_folds,
        engines=engines,
        n_folds=ctx.n_folds,
        n_repeats=ctx.n_repeats,
        report=ctx.report,
        fig_dir=ctx.fig_dir,
    )
    write_json(ctx.cv_metrics, metrics_doc(cv_run))
    write_plot_data(ctx, cv_run)
    return cv_run
