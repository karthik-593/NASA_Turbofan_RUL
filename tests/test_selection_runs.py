"""Tagged model-selection runs (D51): --set overrides, isolation, outputs."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pandas as pd
import pytest
import yaml

from turbofan.pipeline.__main__ import SELECTION_DIR, parse_sets


def test_parse_sets_builds_nested_yaml_values() -> None:
    got = parse_sets(["rul_cap=90", "pipeline.models=[xgboost]", "features.regime_onehot=true"])
    assert got == {
        "rul_cap": 90,
        "pipeline": {"models": ["xgboost"]},
        "features": {"regime_onehot": True},
    }
    with pytest.raises(ValueError, match="key=value"):
        parse_sets(["rul_cap"])


def _entry(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "turbofan.pipeline", *args],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONUTF8": "1", **(env or {})},
        check=False,
    )


@pytest.mark.parametrize(
    ("args", "msg"),
    [
        (["cv_select", "--set", "rul_cap=90"], "--set needs --tag"),
        (["validate", "--tag", "x"], "--tag applies to"),
        (["cv_select", "--tag", "x", "--smoke"], "--tag applies to"),
    ],
)
def test_cli_guards(args: list[str], msg: str) -> None:
    res = _entry(*args)
    assert res.returncode != 0 and msg in res.stderr


def test_unknown_override_key_is_refused() -> None:
    tag = f"_test/{uuid.uuid4().hex[:8]}"
    try:
        res = _entry("cv_select", "--tag", tag, "--set", "rul_kap=90")
        assert res.returncode != 0 and "no such key" in res.stderr
    finally:
        shutil.rmtree(SELECTION_DIR / tag.split("/")[0] / tag.split("/")[1], ignore_errors=True)


@pytest.mark.requires_data
def test_tagged_run_writes_isolated_outputs_and_tags_mlflow(tmp_path: Path) -> None:
    import mlflow

    tag = f"_test/{uuid.uuid4().hex[:8]}"
    uri = (tmp_path / "mlruns").as_uri()
    sets = [
        "pipeline.models=[ridge]", "pipeline.seeds=[42]", "cv.n_folds=2", "cv.n_repeats=1",
        "rul_cap=105", "window=30", "bootstrap.n_boot=50",
    ]  # fmt: skip
    args = ["cv_select", "--tag", tag, *[a for s in sets for a in ("--set", s)]]
    try:
        res = _entry(*args, env={"MLFLOW_TRACKING_URI": uri})
        assert res.returncode == 0, res.stderr[-2000:]
        out = SELECTION_DIR / tag
        params = yaml.safe_load((out / "params.yaml").read_text(encoding="utf-8"))
        assert params["rul_cap"] == 105 and params["window"] == 30
        doc = json.loads((out / "reports" / "cv_select" / "metrics.json").read_text("utf-8"))
        assert doc["tag"] == tag and doc["rul_cap"] == 105.0 and doc["window"] == 30
        assert set(doc["subpopulations"]["ridge"]) == {"subpop_0", "subpop_1"}
        pts = pd.read_parquet(out / "points_ridge.parquet")
        assert pts["unit"].nunique() == 100 and pts["fold"].nunique() == 2
        assert json.loads((out / "env.json").read_text("utf-8"))["devices"]
        client = mlflow.MlflowClient(uri)
        exp = client.get_experiment_by_name("turbofan-rul")
        runs = client.search_runs([exp.experiment_id])
        assert runs and all(r.data.tags.get("selection_tag") == tag for r in runs)
        # the repo's own reports are untouched by a selection run
        assert not (out / "reports" / "protocol_v2.md").exists()
        again = _entry(*args, env={"MLFLOW_TRACKING_URI": uri})
        assert again.returncode == 0, again.stderr[-2000:]
        doc2 = json.loads((out / "reports" / "cv_select" / "metrics.json").read_text("utf-8"))
        assert doc2["models"]["ridge"]["resumed_splits"] == 2  # reloaded, not refitted
    finally:
        shutil.rmtree(SELECTION_DIR / tag, ignore_errors=True)
