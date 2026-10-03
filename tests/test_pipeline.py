"""The DVC pipeline stages (D48): guards, refit budget, and cv_select -> train_prod -> register
end to end on synthetic C-MAPSS files with a file-based MLflow store."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import pytest
import yaml

import turbofan.pipeline.context as pc
from tests.test_run_cv import N_UNITS, _write_raw
from turbofan.config import RUL_CAP
from turbofan.pipeline import cv_select, final_eval_stage, register, train_prod
from turbofan.pipeline.__main__ import _merge
from turbofan.pipeline.context import Context, select_smoke_engines
from turbofan.training.bundle import load_bundle_dir

SRC = Path(__file__).resolve().parents[1] / "src" / "turbofan"
PIPELINE = SRC / "pipeline"


def _ctx(tmp_path: Path, **over: Any) -> Context:
    base: dict[str, Any] = {
        "smoke": True,
        "root": tmp_path / "out",
        "raw": tmp_path / "raw",
        "dataset": "FD001",
        "models": ["mean", "xgboost"],
        "seeds": [1],
        "resume": True,
        "n_folds": 2,
        "n_repeats": 1,
        "max_folds": 1,
        "n_engines": 12,
        "spec": tmp_path / "out" / "locked.yaml",
        "final_eval_enabled": False,
    }
    base.update(over)
    return Context(**base)


def _spec(ctx: Context, model: str = "xgboost") -> None:
    ctx.spec.parent.mkdir(parents=True, exist_ok=True)
    ctx.spec.write_text(
        yaml.safe_dump(
            {"dataset": ctx.dataset, "model": model, "seed": 1, "rul_cap": RUL_CAP, "locked": True}
        ),
        encoding="utf-8",
    )


class TestSmokeEngines:
    def test_balanced_deterministic_and_stratifiable(self) -> None:
        labels = pd.Series([0] * 30 + [1] * 12, index=range(1, 43))
        a = select_smoke_engines(labels, 10)
        assert a == select_smoke_engines(labels, 10) and len(a) == 10
        assert labels.loc[a].value_counts().to_dict() == {0: 5, 1: 5}

    def test_small_group_is_filled_from_the_other(self) -> None:
        labels = pd.Series([0] * 20 + [1] * 3, index=range(1, 24))
        picked = select_smoke_engines(labels, 10)
        assert len(picked) == 10 and labels.loc[picked].value_counts()[1] >= 2

    def test_refuses_a_selection_that_cannot_stratify(self) -> None:
        labels = pd.Series([0] * 20 + [1], index=range(1, 22))
        with pytest.raises(ValueError, match="< 2 engines"):
            select_smoke_engines(labels, 10)


class TestContext:
    def test_unknown_model_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(pc.PARAMS["pipeline"], "models", ["mean", "svm"])
        with pytest.raises(ValueError, match="unknown models"):
            pc.build_context(smoke=False)

    def test_final_eval_enabled_must_be_a_bool(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setitem(pc.PARAMS["final_eval"], "enabled", "yes")
        with pytest.raises(ValueError, match="true/false"):
            pc.build_context(smoke=False)

    def test_default_params_build_and_final_eval_is_off(self) -> None:
        ctx = pc.build_context(smoke=False)
        assert ctx.final_eval_enabled is False and ctx.root == pc.REPO and ctx.max_folds is None

    def test_smoke_context_is_isolated(self) -> None:
        ctx = pc.build_context(smoke=True)
        assert ctx.root == pc.SMOKE_DIR and ctx.max_folds == 1 and ctx.n_engines == 10

    def test_merge_overrides_nested_keys_only(self) -> None:
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        assert _merge(base, {"a": {"y": 9}}) == {"a": {"x": 1, "y": 9}, "b": 3}
        assert base["a"]["y"] == 2  # not mutated


class TestFinalEvalStage:
    def _spy(self, monkeypatch: pytest.MonkeyPatch, ctx: Context) -> list[list[str]]:
        calls: list[list[str]] = []

        def fake_run(cmd: list[str], check: bool) -> None:
            calls.append(cmd)
            ctx.final_metrics.write_text(json.dumps({"ran": True}), encoding="utf-8")

        monkeypatch.setattr(subprocess, "run", fake_run)
        return calls

    def test_disabled_writes_status_and_reads_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ctx = _ctx(tmp_path, smoke=False, final_eval_enabled=False)
        calls = self._spy(monkeypatch, ctx)
        doc = final_eval_stage.run(ctx, confirm=True)
        assert doc["ran"] is False and calls == []
        assert json.loads(ctx.final_metrics.read_text(encoding="utf-8"))["ran"] is False

    def test_smoke_never_reads_the_test_set_even_if_enabled_and_confirmed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ctx = _ctx(tmp_path, smoke=True, final_eval_enabled=True)
        calls = self._spy(monkeypatch, ctx)
        monkeypatch.setenv(final_eval_stage.CONFIRM_ENV, "1")
        assert final_eval_stage.run(ctx, confirm=True)["ran"] is False and calls == []

    def test_enabled_without_confirmation_refuses(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ctx = _ctx(tmp_path, smoke=False, final_eval_enabled=True)
        calls = self._spy(monkeypatch, ctx)
        monkeypatch.delenv(final_eval_stage.CONFIRM_ENV, raising=False)
        with pytest.raises(RuntimeError, match="not confirmed"):
            final_eval_stage.run(ctx, confirm=False)
        assert calls == []

    @pytest.mark.parametrize("how", ["flag", "env"])
    def test_enabled_and_confirmed_delegates_to_final_eval(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, how: str
    ) -> None:
        ctx = _ctx(tmp_path, smoke=False, final_eval_enabled=True)
        ctx.final_metrics.parent.mkdir(parents=True)
        ctx.train_metrics.parent.mkdir(parents=True)
        ctx.train_metrics.write_text(json.dumps({"bundle": "models/prod/b"}), encoding="utf-8")
        calls = self._spy(monkeypatch, ctx)
        monkeypatch.delenv(final_eval_stage.CONFIRM_ENV, raising=False)
        if how == "env":
            monkeypatch.setenv(final_eval_stage.CONFIRM_ENV, "1")
        assert final_eval_stage.run(ctx, confirm=how == "flag") == {"ran": True}
        (cmd,) = calls
        assert cmd[2] == "turbofan.evaluation.final_eval" and "--confirm" in cmd
        assert cmd[cmd.index("--spec") + 1] == str(ctx.spec)
        assert cmd[cmd.index("--bundle") + 1] == str(ctx.root / "models/prod/b")


class TestSealedBoundary:
    def _imports(self, f: Path) -> set[str]:
        out: set[str] = set()
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                out.add(node.module)
                out.update(f"{node.module}.{a.name}" for a in node.names)
            elif isinstance(node, ast.Import):
                out.update(a.name for a in node.names)
        return out

    def test_no_stage_imports_final_eval_or_reads_the_test_loader(self) -> None:
        for f in PIPELINE.glob("*.py"):
            assert "turbofan.evaluation.final_eval" not in self._imports(f), f.name
            assert "load_dataset" not in f.read_text(encoding="utf-8"), f.name

    def test_importing_the_pipeline_package_pulls_in_no_config(self) -> None:
        code = "import sys, turbofan.pipeline; print('turbofan.config' in sys.modules)"
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout.strip()
        assert out == "False"  # --smoke must be able to set TURBOFAN_PARAMS first


class TestIterationBudget:
    def _cv(self, tmp_path: Path, **over: Any) -> Path:
        doc = {
            "dataset": "FD001",
            "rul_cap": RUL_CAP,
            "models": {"xgboost": {"iteration_budget_median": 41.5}, "ridge": {}},
        }
        doc.update(over)
        p = tmp_path / "cv.json"
        p.write_text(json.dumps(doc), encoding="utf-8")
        return p

    def test_median_is_rounded_and_baselines_have_none(self, tmp_path: Path) -> None:
        p = self._cv(tmp_path)
        assert train_prod.iteration_budget_from_cv(p, "FD001", "xgboost") == 42
        assert train_prod.iteration_budget_from_cv(p, "FD001", "ridge") is None

    def test_refuses_other_dataset_other_cap_or_unseen_model(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="rerun cv_select"):
            train_prod.iteration_budget_from_cv(self._cv(tmp_path), "FD002", "xgboost")
        with pytest.raises(ValueError, match="rerun cv_select"):
            train_prod.iteration_budget_from_cv(self._cv(tmp_path, rul_cap=1), "FD001", "xgboost")
        with pytest.raises(ValueError, match="no results for 'lstm'"):
            train_prod.iteration_budget_from_cv(self._cv(tmp_path), "FD001", "lstm")

    def test_missing_cv_metrics_is_loud(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="cv_select"):
            train_prod.iteration_budget_from_cv(tmp_path / "nope.json", "FD001", "xgboost")


def test_cv_select_train_prod_register_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    import turbofan.evaluation.report_v2 as report_v2

    monkeypatch.setattr(report_v2, "DATASETS", ("FD001",))  # the compute estimate loads these
    _write_raw(tmp_path / "raw")
    ctx = _ctx(tmp_path)

    run = cv_select.run(ctx)
    assert ctx.report.is_file() and (ctx.fig_dir / "tradeoff_FD001.png").is_file()
    doc = json.loads(ctx.cv_metrics.read_text(encoding="utf-8"))
    assert doc["dataset"] == "FD001" and doc["rul_cap"] == RUL_CAP
    xgb = doc["models"]["xgboost"]
    assert xgb["iteration_budget_median"] >= 1 and xgb["n_splits"] == 1  # max_folds=1
    assert "iteration_budget_median" not in doc["models"]["mean"]
    assert set(run.results) == {"mean", "xgboost"}
    caught = pd.read_csv(ctx.cv_plots / "decision_caught.csv")
    assert list(caught.columns) == ["threshold", "mean", "xgboost"]
    engines = json.loads(ctx.engines_file.read_text(encoding="utf-8"))["engines"]
    assert len(engines) == 12 and set(engines) <= set(range(1, N_UNITS + 1))

    # a second run reloads the split from MLflow
    assert cv_select.run(ctx).results["xgboost"].resumed_splits == 1

    _spec(ctx)
    bundle = train_prod.run(ctx)
    tp = json.loads(ctx.train_metrics.read_text(encoding="utf-8"))
    assert tp["n_engines"] == 12.0  # every smoke engine, no inner validation share held back
    assert tp["iteration_budget"] == round(xgb["iteration_budget_median"])
    loaded = load_bundle_dir(bundle, device="cpu")
    assert loaded.model.model_.get_booster().num_boosted_rounds() == tp["iteration_budget"]
    assert "test" not in loaded.manifest["metrics"]  # no test-set number in a prod bundle (rule 3)

    reg = register.run(ctx)
    assert reg["alias"] == "challenger" and reg["version"] == 1
    client = mlflow.MlflowClient(uri)
    mv = client.get_model_version_by_alias("turbofan-rul-FD001", "challenger")
    assert int(mv.version) == 1
    tags = client.get_run(reg["mlflow_run_id"]).data.tags
    assert tags["run_type"] == "register"
    assert client.get_run(tp["mlflow_run_id"]).data.tags["run_type"] == "train_prod"


def test_train_prod_refuses_an_unlocked_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    ctx = _ctx(tmp_path)
    ctx.spec.parent.mkdir(parents=True)
    ctx.spec.write_text(
        yaml.safe_dump(
            {"dataset": "FD001", "model": "ridge", "seed": 1, "rul_cap": RUL_CAP, "locked": False}
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="not locked"):
        train_prod.run(ctx)


def _write_test_files(raw: Path, n_test: int = 10) -> None:
    """test_FD001.txt (first ``n_test`` training engines cut short) and its RUL labels."""
    train = pd.read_csv(raw / "train_FD001.txt", sep=" ", header=None)
    parts, remaining = [], []
    for u in range(1, n_test + 1):
        g = train[train[0] == u]
        keep = 20 + 3 * u
        parts.append(g.iloc[:keep])
        remaining.append(len(g) - keep)
    pd.concat(parts).to_csv(raw / "test_FD001.txt", sep=" ", header=False, index=False)
    pd.Series(remaining).to_csv(raw / "RUL_FD001.txt", header=False, index=False)


def _expectations(tmp_path: Path, raw: Path, n_test: int = 10) -> Path:
    test = pd.read_csv(raw / "test_FD001.txt", sep=" ", header=None)
    doc = {
        "datasets": {
            "FD001": {
                "test_rows": len(test),
                "test_cols": test.shape[1],
                "test_units": n_test,
                "rul_rows": n_test,
                "rul_cols": 1,
            }
        }
    }
    p = tmp_path / "expectations.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def test_final_eval_scores_the_train_prod_bundle_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import turbofan.evaluation.final_eval as fe

    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    _write_raw(tmp_path / "raw")
    _write_test_files(tmp_path / "raw")
    ctx = _ctx(tmp_path, models=["ridge"], smoke=False, max_folds=None, n_engines=None)
    ctx.cv_metrics.parent.mkdir(parents=True)
    ctx.cv_metrics.write_text(
        json.dumps({"dataset": "FD001", "rul_cap": RUL_CAP, "models": {"ridge": {}}}),
        encoding="utf-8",
    )
    _spec(ctx, model="ridge")
    bundle = train_prod.run(ctx)

    out = tmp_path / "final.json"
    argv = [
        "--spec", str(ctx.spec), "--bundle", str(bundle), "--raw", str(ctx.raw),
        "--confirm", "--metrics-out", str(out),
        "--expectations", str(_expectations(tmp_path, ctx.raw)),
    ]  # fmt: skip
    fe.main(argv)
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["ran"] is True and doc["n_test_engines"] == 10
    assert doc["uncapped"]["rmse"]["n_engines"] == 10
    runs = mlflow.MlflowClient(uri).search_runs(
        [mlflow.MlflowClient(uri).get_experiment_by_name("turbofan-rul").experiment_id],
        filter_string="tags.run_type = 'final_test'",
    )
    assert len(runs) == 1
    assert runs[0].data.tags["bundle_version"] == "current"
    assert runs[0].data.tags["spec_hash"] == doc["spec_hash"]
    with pytest.raises(fe.FinalEvalRefused, match="already scored"):
        fe.main(argv)


def test_final_eval_refuses_a_bundle_built_from_another_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import turbofan.evaluation.final_eval as fe

    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    _write_raw(tmp_path / "raw")
    _write_test_files(tmp_path / "raw")
    ctx = _ctx(tmp_path, models=["ridge"], smoke=False, max_folds=None, n_engines=None)
    ctx.cv_metrics.parent.mkdir(parents=True)
    ctx.cv_metrics.write_text(
        json.dumps({"dataset": "FD001", "rul_cap": RUL_CAP, "models": {"ridge": {}}}),
        encoding="utf-8",
    )
    _spec(ctx, model="ridge")
    bundle = train_prod.run(ctx)
    other = yaml.safe_load(ctx.spec.read_text(encoding="utf-8")) | {"seed": 7}
    ctx.spec.write_text(yaml.safe_dump(other), encoding="utf-8")  # re-locked after the refit
    argv = ["--spec", str(ctx.spec), "--bundle", str(bundle), "--raw", str(ctx.raw), "--confirm"]
    with pytest.raises(fe.FinalEvalRefused, match="seed"):
        fe.main(argv)
