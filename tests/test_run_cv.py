"""The protocol-v2 CV runner end to end on synthetic C-MAPSS files (no real data needed)."""

from __future__ import annotations

from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from turbofan.config import MIN_HISTORY
from turbofan.evaluation.cv_metrics import HEADLINE
from turbofan.evaluation.run_cv import run_cv

N_UNITS = 24


def _write_raw(raw: Path) -> None:
    """train_FD001.txt (26 space-separated columns) with two degradation subpopulations, and
    RUL_FD001.txt. No test_FD001.txt on purpose: the runner must never need it."""
    rng = np.random.default_rng(0)
    rows = []
    for u in range(1, N_UNITS + 1):
        life = 60 + 3 * u
        flip = -1.0 if u % 3 == 0 else 1.0
        for c in range(1, life + 1):
            frac = c / life
            ops = [rng.normal(0, 0.002), rng.normal(0, 0.0002), 100.0]
            sens = [520.0 + 10 * frac + rng.normal(0, 0.5) for _ in range(21)]
            sens[8] = 9000.0 + flip * 30 * frac + rng.normal(0, 1)  # s9 direction differs
            sens[13] = 8000.0 + flip * 30 * frac + rng.normal(0, 1)  # s14 likewise
            rows.append([u, c, *ops, *sens])
    raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(raw / "train_FD001.txt", sep=" ", header=False, index=False)
    pd.Series(rng.integers(5, 140, 50)).to_csv(raw / "RUL_FD001.txt", header=False, index=False)


@pytest.fixture
def raw(tmp_path: Path) -> Path:
    r = tmp_path / "raw"
    _write_raw(r)
    return r


def test_runner_shape_views_and_no_test_inputs(raw: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import turbofan.data.loader as loader

    opened: list[str] = []
    real = loader._read_txt
    monkeypatch.setattr(loader, "_read_txt", lambda p: opened.append(Path(p).name) or real(p))
    run = run_cv(
        "FD001", ["mean", "ridge"], [7], raw, track=False, n_folds=3, n_repeats=2, verbose=False
    )
    assert opened == ["train_FD001.txt"]  # RUL labels are read by the benchmark view only

    res = run.results["ridge"]
    per_engine = res.points.groupby(["repeat", "seed"])["unit"].nunique()
    assert (per_engine == N_UNITS).all()  # every engine held out once per repeat
    assert res.points.groupby(["repeat", "seed", "unit"])["fold"].nunique().max() == 1
    assert res.benchmark["cycle"].min() >= MIN_HISTORY

    m = res.metrics
    assert set(m["view"]) == {"deployment", "benchmark"}
    assert set(m["group"]) == {"all", "subpop_0", "subpop_1"}
    head = m[(m["view"] == "deployment") & (m["group"] == "all") & (m["metric"] == HEADLINE)]
    assert len(head) == 2 and head["estimate"].notna().all()
    assert (head["ci_lo"] <= head["estimate"]).all() and (head["estimate"] <= head["ci_hi"]).all()
    assert head["n_engines"].iloc[0] == N_UNITS
    assert sum(run.subpop_sizes.values()) == N_UNITS
    assert not res.decision.empty and res.fit_seconds > 0

    nasa = m[m["metric"] == "nasa_mean_per_engine"]
    assert not ((nasa["view"] == "deployment") & (nasa["truth"] == "uncapped")).any()
    assert ((nasa["view"] == "deployment") & (nasa["truth"] == "capped")).any()
    assert ((nasa["view"] == "benchmark") & (nasa["truth"] == "uncapped")).any()


def test_mlflow_parent_and_child_runs(
    raw: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    run = run_cv("FD001", ["mean"], [1, 2], raw, track=True, n_folds=3, n_repeats=1, verbose=False)
    client = mlflow.MlflowClient(uri)
    exp = client.get_experiment_by_name("turbofan-rul")
    runs = client.search_runs([exp.experiment_id])
    parents = [r for r in runs if r.data.tags.get("cv_role") == "parent"]
    children = [r for r in runs if r.data.tags.get("cv_role") == "child"]
    assert len(parents) == 1 and len(children) == 3 * 2
    pid = parents[0].info.run_id
    assert pid == run.results["mean"].parent_run_id
    assert all(c.data.tags["mlflow.parentRunId"] == pid for c in children)
    assert all(r.data.tags["run_type"] == "cv" for r in runs)
    assert all("device.torch" in r.data.tags for r in runs)
    assert f"deployment_uncapped_{HEADLINE}_ci_lo" in parents[0].data.metrics
    pm = parents[0].data.metrics
    assert "deployment_uncapped_nasa_mean_per_engine" not in pm
    assert "deployment_capped_nasa_mean_per_engine" in pm
    assert "benchmark_uncapped_nasa_mean_per_engine" in pm
    assert all("nasa_mean_per_engine" not in c.data.metrics for c in children)
    assert all("nasa_mean_per_engine_capped" in c.data.metrics for c in children)
    arts = {a.path for a in client.list_artifacts(pid)}
    assert {"cv_metrics.csv", "decision_curve.csv", "heldout_predictions.csv"} <= arts


def test_report_writer_end_to_end(
    raw: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import torch

    import turbofan.evaluation.report_v2 as rep
    from turbofan import repro

    monkeypatch.setattr(rep, "REPORT", tmp_path / "reports" / "protocol_v2.md")
    monkeypatch.setattr(rep, "FIG_DIR", tmp_path / "reports" / "figures" / "protocol_v2")
    monkeypatch.setattr(rep, "DATASETS", ("FD001",))
    run = run_cv(
        "FD001", ["mean", "ridge"], [7], raw, track=False, n_folds=3, n_repeats=1, verbose=False
    )
    ctx = repro.environment_context(seeds={"s": 7}, n_threads=torch.get_num_threads())
    path = rep.write_report(run, raw, ctx, wall=12.0)
    text = path.read_text(encoding="utf-8")
    assert "## Sanity run — FD001" in text and "ridge − mean" in text
    assert "## Compute estimate for the full sweep" in text
    assert (rep.FIG_DIR / "decision_FD001.png").exists()
    assert path.with_suffix(".env.json").exists()
