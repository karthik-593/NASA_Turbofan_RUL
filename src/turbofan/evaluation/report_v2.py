"""Write ``reports/protocol_v2.md``: the protocol and a sanity run of it (D46).

    python -m turbofan.evaluation.report_v2 --dataset FD001

Runs ``run_cv`` (training files only, MLflow-tracked) and writes the report from the results —
every number in it is computed here, none typed. Figures go to
``reports/figures/protocol_v2/``; the environment sidecar to ``reports/protocol_v2.env.json``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from turbofan import repro
from turbofan.config import PARAMS, RUL_CAP, SEED
from turbofan.data.loader import load_train
from turbofan.evaluation.compare import CVResult, matched_budget_compare, paired_compare
from turbofan.evaluation.cv_metrics import HEADLINE
from turbofan.evaluation.decision import (
    LEAD_TIMES,
    MATCHED_LEAD_TIME,
    WASTED_LIFE_BUDGETS,
    matched_budget,
)
from turbofan.evaluation.run_cv import NOT_REPORTED, CVRun, run_cv
from turbofan.evaluation.views import deployment_view

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "reports" / "protocol_v2.md"
FIG_DIR = ROOT / "reports" / "figures" / "protocol_v2"
DATASETS = ("FD001", "FD002", "FD003", "FD004")
PAIRS = (("lstm", "xgboost"), ("xgboost", "ridge"), ("ridge", "mean"))


def _ci(r: pd.Series, p: int = 2) -> str:
    return f"{r['estimate']:.{p}f} [{r['ci_lo']:.{p}f}, {r['ci_hi']:.{p}f}]"


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def _pick(m: pd.DataFrame, view: str, group: str, truth: str, metric: str) -> pd.Series:
    return m[
        (m["view"] == view)
        & (m["group"] == group)
        & (m["truth"] == truth)
        & (m["metric"] == metric)
    ].iloc[0]


def _headline_table(run: CVRun, view: str) -> pd.DataFrame:
    rows = []
    for name, res in run.results.items():
        r = _pick(res.metrics, view, "all", "uncapped", HEADLINE)
        rows.append(
            {
                "model": name,
                f"{HEADLINE} [95% CI]": _ci(r),
                "n engines": int(r["n_engines"]),
                "n points": int(r["n_points"]),
            }
        )
    return pd.DataFrame(rows)


def _subpop_table(run: CVRun) -> pd.DataFrame:
    rows = []
    for name, res in run.results.items():
        row = {"model": name}
        for g in ("subpop_0", "subpop_1"):
            r = _pick(res.metrics, "deployment", g, "uncapped", HEADLINE)
            row[f"{g} (n={int(r['n_engines'])} engines)"] = _ci(r)
        rows.append(row)
    return pd.DataFrame(rows)


def _full_table(run: CVRun, view: str, truth: str) -> pd.DataFrame:
    metrics = [
        "rmse",
        "mae",
        "late_pct",
        "mean_signed_error",
        "nasa_mean_per_engine",
        "critical_rmse",
        "critical_late_pct",
        "critical_mean_signed_error",
        "urgent_rmse",
        "monitor_rmse",
        "healthy_rmse",
    ]
    rows = []
    for metric in metrics:
        row = {"metric": metric}
        for name, res in run.results.items():
            if (view, truth, metric) in NOT_REPORTED:
                row[name] = "not reported (D46)"
                continue
            r = _pick(res.metrics, view, "all", truth, metric)
            row[name] = _ci(r, 1) if np.isfinite(r["estimate"]) else "—"
        rows.append(row)
    return pd.DataFrame(rows)


def _bucket_counts(run: CVRun, view: str) -> pd.DataFrame:
    res = next(iter(run.results.values()))
    rows = []
    for truth in ("uncapped", "capped"):
        for b in ("critical", "urgent", "monitor", "healthy"):
            r = _pick(res.metrics, view, "all", truth, f"{b}_rmse")
            rows.append(
                {
                    "truth": truth,
                    "bucket": b,
                    "n engines": int(r["n_engines"]),
                    "n points": int(r["n_points"]),
                }
            )
    return pd.DataFrame(rows)


def _comparisons(run: CVRun) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for a, b in PAIRS:
        if a not in run.results or b not in run.results:
            continue
        ra, rb = run.results[a], run.results[b]
        out = paired_compare(
            CVResult(run.dataset, a, ra.rul_cap, deployment_view(ra.points)),
            CVResult(run.dataset, b, rb.rul_cap, deployment_view(rb.points)),
            np.random.default_rng([SEED, len(rows)]),
        )
        rows.append(
            {
                "A − B": f"{a} − {b}",
                f"Δ {HEADLINE} [95% CI]": (
                    f"{out['diff']:.2f} [{out['diff_ci_lo']:.2f}, {out['diff_ci_hi']:.2f}]"
                ),
                "engines where A better": (
                    f"{out['share_engines_a_better']:.0%} of {out['n_engines_with_metric']}"
                ),
                "Wilcoxon p (fold × seed)": (
                    f"{out['wilcoxon_p']:.2g} (n={out['wilcoxon_n_splits']})"
                ),
            }
        )
    return pd.DataFrame(rows)


def _decision_fig(run: CVRun, fig_dir: Path) -> tuple[Path, pd.DataFrame]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=100)
    rows = []
    L = MATCHED_LEAD_TIME
    for name, res in run.results.items():
        c = res.decision
        n_eng = int(c["n_engines"].iloc[0])
        n_traj = int(c["n_trajectories"].iloc[0])
        for ax, q in ((axes[0], f"caught_lead_ge_{L}_pct"), (axes[1], "mean_wasted_life")):
            s = c[c["quantity"] == q]
            ax.plot(s["threshold"], s["estimate"], label=name)
            ax.fill_between(s["threshold"], s["ci_lo"], s["ci_hi"], alpha=0.15)
        for T in (25.0, 50.0):
            row = {"model": name, "T (cycles)": int(T)}
            for q in [f"caught_lead_ge_{x}_pct" for x in LEAD_TIMES] + [
                "failed_in_service_pct",
                "mean_wasted_life",
            ]:
                r = c[(c["quantity"] == q) & (c["threshold"] == T)].iloc[0]
                row[q] = _ci(r, 1) if np.isfinite(r["estimate"]) else "—"
            rows.append(row)
    axes[0].set(
        title=(
            f"{run.dataset}: failures caught with lead ≥ {L} cycles\n"
            f"(n={n_eng} engines, {n_traj} trajectories)"
        ),
        xlabel="removal threshold T (predicted RUL, cycles)",
        ylabel="% of held-out trajectories",
    )
    axes[1].set(
        title=f"{run.dataset}: mean wasted life at removal (n={n_eng} engines)",
        xlabel="removal threshold T (predicted RUL, cycles)",
        ylabel="cycles of life left at removal",
    )
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.suptitle(
        "Rule: remove when predicted RUL ≤ T — CV held-out trajectories, 95% engine-bootstrap bands"
    )
    fig.tight_layout()
    path = fig_dir / f"decision_{run.dataset}.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path, pd.DataFrame(rows)


def _tradeoff(run: CVRun, fig_dir: Path) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    """Trade-off curves (caught share vs mean wasted life, one point per T) and the models
    compared at matched wasted-life budgets: per model, and paired per model pair."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = MATCHED_LEAD_TIME
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.8), dpi=100)
    for name, res in run.results.items():
        c = res.decision
        x = c[c["quantity"] == "mean_wasted_life"].set_index("threshold")["estimate"]
        y = c[c["quantity"] == f"caught_lead_ge_{L}_pct"].set_index("threshold")["estimate"]
        ax.plot(x, y, marker="o", ms=3, label=name)
    for budget in WASTED_LIFE_BUDGETS:
        ax.axvline(budget, color="grey", ls=":", lw=1)
    n_eng = int(next(iter(run.results.values())).decision["n_engines"].iloc[0])
    ax.set(
        title=(
            f"{run.dataset}: failures caught with lead ≥ {L} cycles vs life wasted\n"
            f"(one point per T; dotted = matched budgets; n={n_eng} engines)"
        ),
        xlabel="mean wasted life at removal (cycles)",
        ylabel=f"% of held-out trajectories caught with lead ≥ {L}",
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = fig_dir / f"tradeoff_{run.dataset}.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)

    per_model = []
    for i, (name, res) in enumerate(run.results.items()):
        t = matched_budget(deployment_view(res.points), np.random.default_rng([SEED, 500 + i]))
        for r in t.to_dict("records"):
            per_model.append(
                {
                    "model": name,
                    "wasted-life budget": f"{r['budget']:g}",
                    f"caught lead ≥ {L} (%) [95% CI]": (
                        f"{r['estimate']:.1f} [{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]"
                        if np.isfinite(r["estimate"])
                        else "budget not reached"
                    ),
                    "n engines": r["n_engines"],
                    "bootstrap replicates defined": f"{r['boot_defined_share']:.0%}",
                }
            )
    paired = []
    for k, (a, b) in enumerate(PAIRS):
        if a not in run.results or b not in run.results:
            continue
        ra, rb = run.results[a], run.results[b]
        t = matched_budget_compare(
            CVResult(run.dataset, a, ra.rul_cap, deployment_view(ra.points)),
            CVResult(run.dataset, b, rb.rul_cap, deployment_view(rb.points)),
            np.random.default_rng([SEED, 600 + k]),
        )
        for r in t.to_dict("records"):
            paired.append(
                {
                    "A − B": f"{a} − {b}",
                    "wasted-life budget": f"{r['budget']:g}",
                    f"Δ caught lead ≥ {L} (pp) [95% CI]": (
                        f"{r['diff']:.1f} [{r['diff_ci_lo']:.1f}, {r['diff_ci_hi']:.1f}]"
                        if np.isfinite(r["diff"])
                        else "budget not reached by both"
                    ),
                    "n engines": r["n_engines"],
                    "bootstrap replicates defined": f"{r['boot_defined_share']:.0%}",
                }
            )
    return path, pd.DataFrame(per_model), pd.DataFrame(paired)


def _compute_estimate(run: CVRun, raw: str | Path) -> tuple[pd.DataFrame, str]:
    """Scale this run's measured per-fit times to the full sweep: every dataset (by training
    rows), every rul_cap in the grid, every seed in params.yaml, and every seq_len in the grid
    for the LSTM. Feature preparation is shared across models and caps (labels only change)."""
    rows_per = {d: len(load_train(d, raw)) for d in DATASETS}
    scale = {d: rows_per[d] / rows_per[run.dataset] for d in DATASETS}
    n_splits = run.n_folds * run.n_repeats
    seeds = len(PARAMS["seeds"])
    caps = len(PARAMS["rul_cap_grid"])
    seq_lens = len(PARAMS["seq_len_grid"])
    out = []
    total = 0.0
    for name, res in run.results.items():
        per_fit = res.fit_seconds / (n_splits * len(run.seeds))
        configs = caps * (seq_lens if name == "lstm" else 1)
        fits = sum(1 for _ in DATASETS) * n_splits * seeds * configs
        secs = sum(per_fit * scale[d] for d in DATASETS) * n_splits * seeds * configs
        total += secs
        out.append(
            {
                "model": name,
                "measured s / fit (FD001)": f"{per_fit:.1f}",
                "configs per dataset": configs,
                "fits": fits,
                "estimated hours": f"{secs / 3600:.1f}",
            }
        )
    feat = (
        run.feature_seconds
        / n_splits
        * sum(scale.values())
        * n_splits
        * len(PARAMS["seq_len_grid"])
    )
    total += feat
    out.append(
        {
            "model": "feature preparation",
            "measured s / fit (FD001)": f"{run.feature_seconds / n_splits:.1f} per fold",
            "configs per dataset": seq_lens,
            "fits": len(DATASETS) * n_splits * seq_lens,
            "estimated hours": f"{feat / 3600:.1f}",
        }
    )
    note = (
        f"Total ≈ {total / 3600:.1f} h on this machine, assuming: {len(DATASETS)} datasets, "
        f"{run.n_folds} folds × {run.n_repeats} repeats, {seeds} seeds (`params.yaml` `seeds`), "
        f"{caps} caps (`rul_cap_grid`), {seq_lens} sequence lengths for the LSTM (`seq_len_grid`), "
        "the four models of this run at default hyperparameters, per-fit time proportional to "
        "training rows, and feature preparation repeated per sequence-length setting (an upper "
        "bound: the window grid does not change the flat features). No hyperparameter search is "
        "included — each tuned configuration multiplies its model's row."
    )
    return pd.DataFrame(out), note


PROTOCOL = """\
## Protocol (D46)

Model selection never touches the NASA test set. Everything below runs on the **training
files** (`data.loader.load_train`); the only test-side input is the *label histogram* of
`RUL_FD00x.txt`, used to shape the benchmark view's sampling.

1. **Subpopulations (D35).** `analysis.subpopulation` reproduces data-audit §I's
   permutation-supported 2-way split (sign of each engine's sensor-vs-time-to-failure
   correlation over its full trajectory, KMeans k = 2). Used only to stratify folds and to
   split reports — never a model input (enforced by an import test).
2. **Folds.** Repeated `StratifiedGroupKFold` over training engines: groups = engine,
   stratified by subpopulation (`params.yaml` `cv`). Inside each fold a stratified share of
   the fold's training engines is held out for early stopping; every fitted state (regimes,
   normalization, envelope) is fitted on the remaining inner-training engines only.
3. **Views on held-out engines** (trajectories run to failure, so true RUL is known
   uncapped). *Deployment* (primary, selects models): a prediction at every cycle from
   `serving.min_history` on. *Benchmark* (secondary, comparability only): truncation points
   sampled per held-out fold so the true-RUL histogram matches the NASA test labels (audit §B),
   RUL above the cap included.
4. **Metrics**, each against capped and uncapped truth: RMSE, MAE, late %, mean signed error
   (prediction − truth), NASA score as a mean per engine, and per maintenance bucket RMSE,
   late %, signed error with n engines and n points. **Headline: critical-bucket RMSE,
   deployment view** (identical under both truths).
5. **Uncertainty.** Points within an engine are correlated, so every CI resamples engines
   (1,000 replicates), pooling all residuals of each drawn engine across repeats and seeds.
6. **Comparisons** (`evaluation.compare`) are paired — same engines, cycles, folds, seeds:
   the metric difference with a paired engine-bootstrap CI, per-engine differences, and a
   Wilcoxon signed-rank test over fold × seed results. Never pooled across datasets.
7. **Cap comparisons (D01).** Candidates with different `rul_cap` are compared only on
   critical-bucket RMSE or RMSE against uncapped truth; the comparison code refuses any other
   metric.
8. **Decision curves.** Rule "remove when predicted RUL ≤ T" run forward on held-out
   trajectories: % of failures caught with lead time ≥ L, % failing in service, mean wasted
   life (true RUL at removal), versus T. Models are compared at **matched operating points**
   (caught % at fixed wasted-life budgets, paired engine bootstrap), not at equal T, which is
   not like-for-like.
9. **Final evaluation** (`evaluation.final_eval`) is the one sealed test read per locked
   candidate: it refuses without `--confirm` or a `locked: true` spec, refuses a second run of
   the same spec, and logs `run_type=final_test`. Selection code may not import it.
10. **Tracking.** One MLflow parent run per model (`run_type=cv`) with aggregated metrics and
    CIs, one child run per fold × seed; device tags on every run (D43).
"""


def write_report(
    run: CVRun,
    raw: str | Path,
    ctx: dict[str, object],
    wall: float,
    report: Path | None = None,
    fig_dir: Path | None = None,
) -> Path:
    report = report or REPORT
    fig_dir = fig_dir or FIG_DIR
    fig, decision_tab = _decision_fig(run, fig_dir)
    trade_fig, trade_model, trade_paired = _tradeoff(run, fig_dir)
    est, est_note = _compute_estimate(run, raw)
    short = {k: v for res in run.results.values() for k, v in res.benchmark_shortfall.items()}
    n_fits = run.n_folds * run.n_repeats * len(run.seeds)
    fit_rows = pd.DataFrame(
        [
            {
                "model": n,
                "fit + predict, all folds (s)": f"{r.fit_seconds:.0f}",
                "per fit (s)": f"{r.fit_seconds / n_fits:.1f}",
                "MLflow parent run": f"`{r.parent_run_id}`",
            }
            for n, r in run.results.items()
        ]
    )
    devices = ctx["devices"]
    assert isinstance(devices, dict)
    parts = [
        "# Evaluation protocol v2 — description and sanity run",
        "",
        "Produced by `python -m turbofan.evaluation.report_v2`; every number below is computed by "
        "that run from the training files. **No NASA test input was read** (the benchmark view "
        "used the test label histogram only).",
        "",
        PROTOCOL,
        f"## Sanity run — {run.dataset}",
        "",
        f"Models at their current default configurations (`params.yaml` `models`), "
        f"{run.n_folds} folds × {run.n_repeats} repeats, seed{'s' if len(run.seeds) > 1 else ''} "
        f"{', '.join(map(str, run.seeds))}, `rul_cap` {RUL_CAP:g}. Subpopulation sizes: "
        + ", ".join(f"group {k}: {v} engines" for k, v in run.subpop_sizes.items())
        + f". Devices: LSTM on {devices['torch']}, XGBoost on {devices['xgboost']}"
        + (f" ({devices['gpu_name']})" if devices.get("gpu_name") else "")
        + f". Wall-clock for the whole run: {wall / 60:.1f} min (fold features "
        f"{run.feature_seconds:.0f} s).",
        "",
        f"**Headline — {HEADLINE}, deployment view** (95% engine-bootstrap CI):",
        "",
        _md(_headline_table(run, "deployment")),
        "",
        "Benchmark view, same metric (secondary; matches the NASA test label histogram):",
        "",
        _md(_headline_table(run, "benchmark")),
        "",
        "Benchmark-view bins no held-out trajectory could supply (target points short): "
        + (", ".join(f"RUL {k}–{k + 9}: {v}" for k, v in sorted(short.items())) or "none")
        + ".",
        "",
        f"**Per subpopulation** — {HEADLINE}, deployment view:",
        "",
        _md(_subpop_table(run)),
        "",
        "**All metrics — deployment view, uncapped truth** (estimate [95% CI]):",
        "",
        _md(_full_table(run, "deployment", "uncapped")),
        "",
        "Deployment view, capped truth (what the models are trained to predict):",
        "",
        _md(_full_table(run, "deployment", "capped")),
        "",
        "Points per bucket (deployment view; identical for every model):",
        "",
        _md(_bucket_counts(run, "deployment")),
        "",
        f"**Paired comparisons** — {HEADLINE}, deployment view (negative Δ = A lower error):",
        "",
        _md(_comparisons(run)),
        "",
        "**Decision curves** (rule: remove when predicted RUL ≤ T):",
        "",
        f"![decision curves]({fig.relative_to(report.parent).as_posix()})",
        "",
        "Same-T table — **not like-for-like**: the same T removes at different wasted life for "
        "different models, so differences here mix better prediction with a different "
        "operating point. Compare models in the matched-budget tables below.",
        "",
        _md(decision_tab),
        "",
        f"**Trade-off curves** — caught share (lead ≥ {MATCHED_LEAD_TIME} cycles) against mean "
        "wasted life, one point per T:",
        "",
        f"![trade-off curves]({trade_fig.relative_to(report.parent).as_posix()})",
        "",
        "**Matched operating points** — each model read at the T where its mean wasted life "
        "equals the budget (linear interpolation along the T grid; 95% engine-bootstrap CI with "
        "the curve re-matched in every replicate):",
        "",
        _md(trade_model),
        "",
        "Paired differences at matched budgets (same engines, folds, seeds; positive Δ = A "
        "catches more):",
        "",
        _md(trade_paired),
        "",
        "**Wall-clock per model** (fit + predict, summed over folds × seeds):",
        "",
        _md(fit_rows),
        "",
        "## Compute estimate for the full sweep",
        "",
        _md(est),
        "",
        est_note,
        "",
        repro.render_markdown(ctx),
    ]
    # exactly one trailing newline (render_markdown already ends with one; the end-of-file
    # pre-commit hook rejects a trailing blank line)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(parts).rstrip("\n") + "\n", encoding="utf-8")
    repro.write_context(ctx, report)
    return report


def run_and_report(
    dataset: str,
    models: list[str],
    seeds: list[int],
    raw: str | Path,
    *,
    resume: bool = False,
    max_folds: int | None = None,
    engines: list[int] | None = None,
    n_folds: int | None = None,
    n_repeats: int | None = None,
    report: Path | None = None,
    fig_dir: Path | None = None,
) -> tuple[CVRun, Path]:
    """Run protocol-v2 CV (MLflow-tracked) and write the report; returns (run, report path)."""
    run_seeds = {f"model_seed_{s}": s for s in seeds}
    run_seeds.update({"cv_seed": PARAMS["cv"]["seed"], "bootstrap_and_views_seed": SEED})
    # git/data provenance first, before anything is written (cf. D34 / item 0)
    ctx = repro.environment_context(seeds=run_seeds, n_threads=torch.get_num_threads())
    t0 = time.perf_counter()
    run = run_cv(
        dataset,
        models,
        seeds,
        raw,
        track=True,
        resume=resume,
        max_folds=max_folds,
        engines=engines,
        n_folds=PARAMS["cv"]["n_folds"] if n_folds is None else n_folds,
        n_repeats=PARAMS["cv"]["n_repeats"] if n_repeats is None else n_repeats,
    )
    # device state only exists after the models ran (GPU resolved, CUDA initialized)
    ctx = repro.with_device_state(ctx)
    path = write_report(run, raw, ctx, time.perf_counter() - t0, report=report, fig_dir=fig_dir)
    return run, path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Protocol-v2 sanity run and report.")
    ap.add_argument("--dataset", default="FD001")
    ap.add_argument("--models", nargs="+", default=["mean", "ridge", "xgboost", "lstm"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[SEED])
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--resume", action="store_true", help="reload splits MLflow already holds")
    args = ap.parse_args(argv)
    _run, path = run_and_report(args.dataset, args.models, args.seeds, args.raw, resume=args.resume)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
