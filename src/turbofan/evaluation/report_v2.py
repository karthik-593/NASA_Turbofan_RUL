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
from turbofan.evaluation.compare import CVResult, paired_compare
from turbofan.evaluation.cv_metrics import HEADLINE
from turbofan.evaluation.decision import LEAD_TIMES
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


def _decision_fig(run: CVRun) -> tuple[Path, pd.DataFrame]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=100)
    rows = []
    L = LEAD_TIMES[len(LEAD_TIMES) // 2]
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
    path = FIG_DIR / f"decision_{run.dataset}.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path, pd.DataFrame(rows)


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
   life (true RUL at removal), versus T.
9. **Final evaluation** (`evaluation.final_eval`) is the one sealed test read per locked
   candidate: it refuses without `--confirm` or a `locked: true` spec, refuses a second run of
   the same spec, and logs `run_type=final_test`. Selection code may not import it.
10. **Tracking.** One MLflow parent run per model (`run_type=cv`) with aggregated metrics and
    CIs, one child run per fold × seed; device tags on every run (D43).
"""


def write_report(run: CVRun, raw: str | Path, ctx: dict[str, object], wall: float) -> Path:
    fig, decision_tab = _decision_fig(run)
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
        f"![decision curves]({fig.relative_to(REPORT.parent).as_posix()})",
        "",
        _md(decision_tab),
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
    REPORT.write_text("\n".join(parts).rstrip("\n") + "\n", encoding="utf-8")
    repro.write_context(ctx, REPORT)
    return REPORT


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Protocol-v2 sanity run and report.")
    ap.add_argument("--dataset", default="FD001")
    ap.add_argument("--models", nargs="+", default=["mean", "ridge", "xgboost", "lstm"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[SEED])
    ap.add_argument("--raw", default="data/raw")
    args = ap.parse_args(argv)
    seeds = {f"model_seed_{s}": s for s in args.seeds}
    seeds.update({"cv_seed": PARAMS["cv"]["seed"], "bootstrap_and_views_seed": SEED})
    # provenance first, before anything is written (cf. D34 / item 0)
    ctx = repro.environment_context(seeds=seeds, n_threads=torch.get_num_threads())
    t0 = time.perf_counter()
    run = run_cv(args.dataset, args.models, args.seeds, args.raw, track=True)
    path = write_report(run, args.raw, ctx, time.perf_counter() - t0)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
