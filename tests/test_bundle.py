"""Tests for turbofan.training.bundle — save/load roundtrip.

Uses Ridge (CPU, fast) and synthetic data; no LSTM, no real dataset.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest

from turbofan.evaluation.comparison import build_registry
from turbofan.training.bundle import (
    BUNDLE_SCHEMA,
    Bundle,
    git_provenance,
    load_bundle,
    new_version,
    resolve_version,
    save_bundle,
)

_RNG = np.random.default_rng(0)
_X = _RNG.standard_normal((20, 5)).astype(np.float32)
_y = _RNG.uniform(0, 125, 20).astype(np.float32)
_STATS = {"s2_mean": [0.0] * 5, "s2_std": [1.0] * 5}
_METRICS = {"test": {"critical_rmse": 12.34, "nasa": 999.0}}


def _ridge():
    cand = build_registry()["ridge"]
    model = cand.factory()
    model.fit(_X, _y)
    return model


# -- helpers -------------------------------------------------------------------


def test_new_version_is_string():
    v = new_version()
    assert isinstance(v, str)
    assert v.endswith("Z")


# -- save ----------------------------------------------------------------------


def test_save_creates_three_files(tmp_path):
    path = save_bundle(
        tmp_path, "FD001", "ridge", "v1", _ridge(), _STATS, seed=42, metrics=_METRICS
    )
    assert (path / "model.bin").exists()
    assert (path / "feature_state.pkl").exists()
    assert (path / "manifest.json").exists()


def test_manifest_fields(tmp_path):
    path = save_bundle(
        tmp_path, "FD001", "ridge", "v1", _ridge(), _STATS, seed=42, metrics=_METRICS
    )
    m = json.loads((path / "manifest.json").read_text())
    assert m["dataset"] == "FD001"
    assert m["model"] == "ridge"
    assert m["seed"] == 42
    assert m["metrics"]["test"]["critical_rmse"] == pytest.approx(12.34)
    assert "bundle_schema" in m
    assert "lib_versions" in m
    assert "config" in m
    assert "files" in m


def test_manifest_records_git_commit_and_dirty_flag(tmp_path):
    path = save_bundle(
        tmp_path, "FD001", "ridge", "v1", _ridge(), _STATS, seed=42, metrics=_METRICS
    )
    m = json.loads((path / "manifest.json").read_text())
    assert re.fullmatch(r"[0-9a-f]{40}", m["git_commit"])
    assert isinstance(m["git_dirty"], bool)


def test_git_provenance_outside_repo_raises(tmp_path, monkeypatch):
    monkeypatch.delenv("GIT_COMMIT", raising=False)
    monkeypatch.delenv("GIT_DIRTY", raising=False)
    with pytest.raises(RuntimeError, match="git checkout"):
        git_provenance(tmp_path)


def test_git_provenance_falls_back_to_env_vars_outside_repo(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_COMMIT", "deadbeef")
    monkeypatch.setenv("GIT_DIRTY", "true")
    assert git_provenance(tmp_path) == {"git_commit": "deadbeef", "git_dirty": True}


def test_git_provenance_env_dirty_false(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_COMMIT", "deadbeef")
    monkeypatch.setenv("GIT_DIRTY", "false")
    assert git_provenance(tmp_path) == {"git_commit": "deadbeef", "git_dirty": False}


def test_git_provenance_raises_on_lone_env_var(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_COMMIT", "deadbeef")
    monkeypatch.delenv("GIT_DIRTY", raising=False)
    with pytest.raises(RuntimeError, match="git checkout"):
        git_provenance(tmp_path)


def test_git_provenance_prefers_real_git_checkout_over_env(monkeypatch):
    monkeypatch.setenv("GIT_COMMIT", "deadbeef")
    monkeypatch.setenv("GIT_DIRTY", "true")
    repo_dir = Path(__file__).resolve().parent
    result = git_provenance(repo_dir)
    assert result["git_commit"] != "deadbeef"
    assert re.fullmatch(r"[0-9a-f]{40}", result["git_commit"])


def test_explicit_version_in_path(tmp_path):
    path = save_bundle(
        tmp_path, "FD001", "ridge", "v42", _ridge(), _STATS, seed=42, metrics=_METRICS
    )
    assert path.name == "v42"


# -- resolve_version -----------------------------------------------------------


def test_resolve_version_latest(tmp_path):
    model = _ridge()
    save_bundle(tmp_path, "FD001", "ridge", "v1", model, _STATS, seed=42, metrics=_METRICS)
    save_bundle(tmp_path, "FD001", "ridge", "v2", model, _STATS, seed=42, metrics=_METRICS)
    assert resolve_version(tmp_path, "FD001", "ridge") == "v2"


def test_resolve_version_explicit(tmp_path):
    model = _ridge()
    save_bundle(tmp_path, "FD001", "ridge", "v1", model, _STATS, seed=42, metrics=_METRICS)
    assert resolve_version(tmp_path, "FD001", "ridge", "v1") == "v1"


def test_resolve_version_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_version(tmp_path, "FD999", "ridge")


# -- load ----------------------------------------------------------------------


def test_load_returns_bundle(tmp_path):
    save_bundle(tmp_path, "FD001", "ridge", "v1", _ridge(), _STATS, seed=42, metrics=_METRICS)
    b = load_bundle(tmp_path, "FD001", "ridge")
    assert isinstance(b, Bundle)
    assert b.path is not None


def test_load_manifest_roundtrip(tmp_path):
    save_bundle(tmp_path, "FD001", "ridge", "v1", _ridge(), _STATS, seed=42, metrics=_METRICS)
    b = load_bundle(tmp_path, "FD001", "ridge")
    assert b.manifest["dataset"] == "FD001"
    assert b.manifest["model"] == "ridge"
    assert b.manifest["version"] == "v1"


def test_load_stats_roundtrip(tmp_path):
    save_bundle(tmp_path, "FD001", "ridge", "v1", _ridge(), _STATS, seed=42, metrics=_METRICS)
    b = load_bundle(tmp_path, "FD001", "ridge")
    assert list(b.stats.keys()) == list(_STATS.keys())


def test_predictions_identical_after_reload(tmp_path):
    model = _ridge()
    pred_before = model.predict(_X)
    save_bundle(tmp_path, "FD001", "ridge", "v1", model, _STATS, seed=42, metrics=_METRICS)
    b = load_bundle(tmp_path, "FD001", "ridge")
    np.testing.assert_array_almost_equal(pred_before, b.model.predict(_X))


def test_load_picks_latest(tmp_path):
    model = _ridge()
    save_bundle(
        tmp_path,
        "FD001",
        "ridge",
        "v1",
        model,
        _STATS,
        seed=42,
        metrics={"test": {"critical_rmse": 10.0}},
    )
    save_bundle(
        tmp_path,
        "FD001",
        "ridge",
        "v2",
        model,
        _STATS,
        seed=42,
        metrics={"test": {"critical_rmse": 5.0}},
    )
    b = load_bundle(tmp_path, "FD001", "ridge")
    assert b.manifest["version"] == "v2"


def test_load_specific_version(tmp_path):
    model = _ridge()
    save_bundle(
        tmp_path,
        "FD001",
        "ridge",
        "v1",
        model,
        _STATS,
        seed=42,
        metrics={"test": {"critical_rmse": 10.0}},
    )
    save_bundle(
        tmp_path,
        "FD001",
        "ridge",
        "v2",
        model,
        _STATS,
        seed=42,
        metrics={"test": {"critical_rmse": 5.0}},
    )
    b = load_bundle(tmp_path, "FD001", "ridge", version="v1")
    assert b.manifest["metrics"]["test"]["critical_rmse"] == pytest.approx(10.0)


def test_manifest_records_regime_count(tmp_path):
    d = save_bundle(tmp_path, "FD002", "ridge", "v1", _ridge(), _STATS, seed=42, metrics=_METRICS)
    m = json.loads((d / "manifest.json").read_text())
    assert m["bundle_schema"] == BUNDLE_SCHEMA
    assert m["config"]["n_regimes"] == 6


def test_load_refuses_older_bundle_schema(tmp_path):
    d = save_bundle(tmp_path, "FD001", "ridge", "v1", _ridge(), _STATS, seed=42, metrics=_METRICS)
    m = json.loads((d / "manifest.json").read_text())
    m["bundle_schema"] = BUNDLE_SCHEMA - 1
    (d / "manifest.json").write_text(json.dumps(m))
    with pytest.raises(ValueError, match="retrain it"):
        load_bundle(tmp_path, "FD001", "ridge")
