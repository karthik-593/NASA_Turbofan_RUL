"""Resume keys: the config hash and the source fingerprint (D49)."""

from __future__ import annotations

import pytest

import turbofan.evaluation.resume as rs

BASE = {
    "dataset": "FD001",
    "model": "ridge",
    "seed": 1,
    "repeat": 0,
    "fold": 0,
    "n_folds": 3,
    "n_repeats": 1,
    "rul_cap": 125.0,
    "engines": None,
}


@pytest.mark.parametrize(
    "change",
    [
        {"model": "xgboost"},
        {"seed": 2},
        {"fold": 1},
        {"repeat": 1},
        {"rul_cap": 105.0},
        {"n_folds": 5},
        {"engines": [1, 2, 3]},
    ],
)
def test_config_hash_changes_with_every_field(change: dict[str, object]) -> None:
    assert rs.config_hash(**BASE) != rs.config_hash(**{**BASE, **change})  # type: ignore[arg-type]


def test_config_hash_is_stable_and_engine_order_free() -> None:
    assert rs.config_hash(**BASE) == rs.config_hash(**BASE)  # type: ignore[arg-type]
    a = rs.config_hash(**{**BASE, "engines": [3, 1, 2]})  # type: ignore[arg-type]
    assert a == rs.config_hash(**{**BASE, "engines": [1, 2, 3]})  # type: ignore[arg-type]


def test_config_hash_follows_fit_params_and_model_params(monkeypatch: pytest.MonkeyPatch) -> None:
    h = rs.config_hash(**BASE)  # type: ignore[arg-type]
    monkeypatch.setitem(rs.PARAMS, "window", rs.PARAMS["window"] + 1)
    assert rs.config_hash(**BASE) != h  # type: ignore[arg-type]
    monkeypatch.undo()
    monkeypatch.setitem(rs.PARAMS["models"]["ridge"], "alpha", 9.0)
    assert rs.config_hash(**BASE) != h  # type: ignore[arg-type]
    monkeypatch.undo()
    for key, value in (
        ("health_index", "pooled"),
        ("regime_onehot", True),
        ("extra_sensors", ["s6"]),
    ):
        monkeypatch.setitem(rs.PARAMS["features"], key, value)
        assert rs.config_hash(**BASE) != h, key  # type: ignore[arg-type]
        monkeypatch.undo()
    # an evaluation-side section does not invalidate a fit
    monkeypatch.setitem(rs.PARAMS["bootstrap"], "n_boot", 7)
    assert rs.config_hash(**BASE) == h  # type: ignore[arg-type]


def test_fingerprint_is_the_tree_when_clean_and_adds_a_digest_when_dirty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replies = {"rev-parse": "treehash", "status": "", "diff": "patch"}
    monkeypatch.setattr(rs, "_git", lambda *a: replies[a[0]])
    assert rs.src_fingerprint() == "treehash"
    replies["status"] = " M src/turbofan/x.py"
    dirty = rs.src_fingerprint()
    assert dirty.startswith("treehash+") and dirty != "treehash"
    replies["diff"] = "other patch"
    assert rs.src_fingerprint() != dirty


def test_every_params_section_is_classified_for_the_resume_key() -> None:
    sections = set(rs.PARAMS)
    fit, not_fit = set(rs.FIT_SECTIONS), set(rs.NOT_FIT_SECTIONS)
    assert not fit & not_fit
    assert sections - fit - not_fit == set(), "classify new params.yaml sections in resume.py"
