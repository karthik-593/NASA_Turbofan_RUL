"""Tests for turbofan.tracking — the run() context manager and the pyfunc wrapper.

All tests use a temporary MLflow file-store URI (file:///...), never a server: the file
store supports both the tracking API and (with a deprecation warning) the model registry,
which is exactly what the round-trip tests below need, and it needs no setup/teardown.
"""

from __future__ import annotations

from pathlib import Path

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
import pytest

from turbofan import tracking
from turbofan.config import KEEP, RUL_CAP, SENSOR_N_COLS, SEQ_LEN
from turbofan.evaluation.protocol import score
from turbofan.features.engineering import add_features
from turbofan.models.lstm_model import LSTMRUL, make_last_windows, make_sequences
from turbofan.training.bundle import load_bundle_dir, save_bundle


@pytest.fixture
def tracking_uri(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    # The target dir must NOT already exist: mlflow's FileStore only auto-creates the
    # 'Default' experiment (id 0) when it initializes a brand-new root; an existing-but-
    # empty directory (e.g. tmp_path itself) raises "Could not find experiment with ID 0".
    uri = (tmp_path / "mlruns").as_uri()
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    return uri


def _train_tiny_bundle(out_root: Path) -> tuple[Path, pd.DataFrame]:
    """A 2-engine LSTM bundle on synthetic data — enough to exercise the pyfunc
    wrapper's plumbing, not a quality fit (mirrors tests/test_api.py's fixture)."""
    rng = np.random.default_rng(0)
    rows = []
    for unit, final_rul in enumerate([10, 70], start=1):
        for cyc in range(1, 40):
            row = {"unit": unit, "cycle": cyc, "op1": 0.0, "op2": 0.0, "op3": 0.0}
            row.update({s: float(rng.normal()) for s in KEEP})
            row["rul"] = min(final_rul + (39 - cyc), RUL_CAP)
            rows.append(row)
    df = pd.DataFrame(rows)
    feat, stats = add_features(df, KEEP, "FD001")

    X, y = make_sequences(feat, SENSOR_N_COLS, SEQ_LEN)
    model = LSTMRUL(
        n_features=len(SENSOR_N_COLS),
        hidden=4,
        layers=1,
        max_epochs=1,
        patience=1,
        batch_size=8,
        seed=0,
    )
    model.fit(X, y, X, y)

    X_last, units = make_last_windows(feat, SENSOR_N_COLS, SEQ_LEN)
    y_last = feat.groupby("unit")["rul"].last().loc[units].to_numpy()
    metrics = {"test": score(y_last, model.predict(X_last))}

    bundle_dir = save_bundle(out_root, "FD001", "lstm", "v1", model, stats, seed=0, metrics=metrics)
    return bundle_dir, feat


# -- _dvc_data_hash ---------------------------------------------------------------------


class TestDvcDataHash:
    def test_reads_md5_from_dvc_file(self, tmp_path: Path) -> None:
        p = tmp_path / "raw.dvc"
        p.write_text(
            "outs:\n- md5: deadbeefcafebabe.dir\n  size: 1\n  nfiles: 1\n  hash: md5\n  path: raw\n"
        )
        assert tracking._dvc_data_hash(p) == "deadbeefcafebabe.dir"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="dvc"):
            tracking._dvc_data_hash(tmp_path / "nope.dvc")

    def test_file_with_no_outs_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.dvc"
        p.write_text("not_outs: []\n")
        with pytest.raises(ValueError, match="no usable md5"):
            tracking._dvc_data_hash(p)


# -- _config_params -----------------------------------------------------------------------


class TestConfigParams:
    def test_keys_are_cfg_prefixed(self) -> None:
        params = tracking._config_params()
        assert params  # non-empty
        assert all(k.startswith("cfg_") for k in params)

    def test_includes_known_constant(self) -> None:
        assert tracking._config_params()["cfg_RUL_CAP"] == "125.0"

    def test_no_collision_with_run_metadata_keys(self) -> None:
        """Regression test: config.SEED vs a run's own 'seed' param differ only by case,
        which collided on a case-insensitive filesystem (Windows/macOS) before the cfg_
        prefix was added — see the fix in evaluation.comparison's integration commit."""
        params = tracking._config_params()
        assert "seed" not in params
        assert "dataset" not in params
        assert "model" not in params


# -- run() ------------------------------------------------------------------------------


class TestRun:
    def test_track_false_yields_none(self, tracking_uri: str) -> None:
        with tracking.run(
            dataset="FD001", model="lstm", seed=0, run_type="cv", track=False
        ) as active:
            assert active is None

    def test_track_false_skips_even_when_unreachable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:1")
        with tracking.run(
            dataset="FD001", model="lstm", seed=0, run_type="cv", track=False
        ) as active:
            assert active is None

    def test_unreachable_backend_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:1")
        with (
            pytest.raises(tracking.TrackingUnavailable),
            tracking.run(dataset="FD001", model="lstm", seed=0, run_type="cv"),
        ):
            pass

    def test_logs_provenance_tags_and_params(self, tracking_uri: str) -> None:
        with tracking.run(
            dataset="FD001", model="xgboost", seed=7, run_type="train_prod"
        ) as active:
            assert active is not None
            run_id = active.info.run_id

        r = mlflow.MlflowClient(tracking_uri=tracking_uri).get_run(run_id)
        assert r.data.tags["run_type"] == "train_prod"
        assert len(r.data.tags["git_commit"]) == 40
        assert r.data.tags["git_dirty"] in ("True", "False")
        assert r.data.tags["dvc_data_hash"]
        assert r.data.tags["lib.mlflow"]
        assert r.data.params["dataset"] == "FD001"
        assert r.data.params["model"] == "xgboost"
        assert r.data.params["seed"] == "7"
        assert r.data.params["cfg_RUL_CAP"] == "125.0"

    def test_legacy_test_selection_run_type_accepted(self, tracking_uri: str) -> None:
        """evaluation.comparison tags its runs this way, not 'cv' — see its docstring:
        it scores against the test set during selection, so it isn't valid evidence and
        must not be mistaken for a protocol-v2-clean 'cv' screen."""
        with tracking.run(
            dataset="FD001", model="xgboost", seed=1, run_type="legacy_test_selection"
        ) as active:
            assert active is not None
            run_id = active.info.run_id

        r = mlflow.MlflowClient(tracking_uri=tracking_uri).get_run(run_id)
        assert r.data.tags["run_type"] == "legacy_test_selection"

    def test_extra_params_and_tags_logged(self, tracking_uri: str) -> None:
        with tracking.run(
            dataset="FD001",
            model="lstm",
            seed=1,
            run_type="cv",
            extra_params={"n_trials": 5},
            extra_tags={"note": "x"},
        ) as active:
            assert active is not None
            run_id = active.info.run_id

        r = mlflow.MlflowClient(tracking_uri=tracking_uri).get_run(run_id)
        assert r.data.params["n_trials"] == "5"
        assert r.data.tags["note"] == "x"

    def test_nested_run_has_correct_parent(self, tracking_uri: str) -> None:
        with tracking.run(dataset="FD001", model="lstm", seed=-1, run_type="cv") as parent:
            assert parent is not None
            with tracking.run(
                dataset="FD001", model="lstm", seed=1, run_type="cv", nested=True
            ) as child:
                assert child is not None
                child_id, parent_id = child.info.run_id, parent.info.run_id

        child_run = mlflow.MlflowClient(tracking_uri=tracking_uri).get_run(child_id)
        assert child_run.data.tags["mlflow.parentRunId"] == parent_id


# -- pyfunc round-trip --------------------------------------------------------------------


class TestPyfuncRoundTrip:
    def test_bundle_model_predicts_directly(self, tmp_path: Path) -> None:
        """Unit-level: BundleModel.predict against a loaded bundle, no mlflow logging."""
        bundle_dir, feat = _train_tiny_bundle(tmp_path / "bundle_root")
        model = tracking.BundleModel()
        model._bundle = load_bundle_dir(bundle_dir, device="cpu")

        req = feat[feat["unit"] == 2][["op1", "op2", "op3"] + KEEP].tail(SEQ_LEN + 5)
        out = model.predict(context=None, model_input=req)

        assert isinstance(out, pd.DataFrame)
        assert len(out) == 1
        assert out.loc[0, "n_cycles_used"] == SEQ_LEN
        assert out.loc[0, "dataset"] == "FD001"
        assert "confidence_error_band_cycles" in out.columns
        assert "confidence_basis" in out.columns
        assert 0.0 <= out.loc[0, "predicted_rul"] <= RUL_CAP

    def test_log_register_alias_load_predict(self, tracking_uri: str, tmp_path: Path) -> None:
        """Full round-trip: log as pyfunc, register, set alias, load back, predict —
        exactly what a later 'promote to challenger' step would do."""
        # tracking_uri only sets the env var; set it explicitly here too since this test
        # calls mlflow.* directly rather than through tracking.run() (which does this
        # itself) — otherwise a global URI left behind by an earlier test in the same
        # process could still be active.
        mlflow.set_tracking_uri(tracking_uri)
        bundle_dir, feat = _train_tiny_bundle(tmp_path / "bundle_root")

        with mlflow.start_run():
            info = tracking.log_production_model(bundle_dir, "FD001")
            mv = tracking.register_production_model(info.model_uri, "FD001", alias="challenger")

        assert mv.name == "turbofan-rul-FD001"

        client = mlflow.MlflowClient(tracking_uri=tracking_uri)
        fetched = client.get_model_version_by_alias("turbofan-rul-FD001", "challenger")
        assert fetched.version == mv.version

        loaded = mlflow.pyfunc.load_model(info.model_uri)
        req = feat[feat["unit"] == 1][["op1", "op2", "op3"] + KEEP].tail(SEQ_LEN + 5)
        pred = loaded.predict(req)

        assert len(pred) == 1
        assert pred.loc[0, "maintenance_bucket"] in ("critical", "urgent", "monitor", "healthy")
        assert 0.0 <= pred.loc[0, "predicted_rul"] <= RUL_CAP
