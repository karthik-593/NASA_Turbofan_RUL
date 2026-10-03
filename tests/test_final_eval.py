"""final_eval guards: --confirm, locked spec, once per spec; selection code can't reach it."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import turbofan.evaluation.final_eval as fe

SRC = Path(__file__).resolve().parents[1] / "src" / "turbofan"
# protocol-v2 selection modules: must not import final_eval, nor the loader that opens test files
SELECTION = ("cv", "cv_metrics", "views", "compare", "decision", "fitting", "run_cv")


@pytest.fixture
def no_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if anything reaches the data."""

    def boom(*_a: object, **_k: object) -> None:
        raise AssertionError("data was loaded")

    monkeypatch.setattr(fe, "load_dataset", boom)


BUNDLE = "no-such-bundle"  # refusals must fire before any bundle is opened


def _spec(tmp_path: Path, **over: object) -> Path:
    spec = {"dataset": "FD001", "model": "ridge", "seed": 42, "rul_cap": 125, "locked": True}
    spec.update(over)
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return p


class TestRefusals:
    def test_refuses_without_confirm(self, tmp_path: Path, no_data: None) -> None:
        with pytest.raises(fe.FinalEvalRefused, match="--confirm"):
            fe.main(["--spec", str(_spec(tmp_path)), "--bundle", BUNDLE])

    def test_refuses_unlocked_spec(self, tmp_path: Path, no_data: None) -> None:
        with pytest.raises(fe.FinalEvalRefused, match="not locked"):
            fe.main(["--spec", str(_spec(tmp_path, locked=False)), "--bundle", BUNDLE, "--confirm"])

    def test_refuses_incomplete_spec(self, tmp_path: Path, no_data: None) -> None:
        p = tmp_path / "s.yaml"
        p.write_text(yaml.safe_dump({"dataset": "FD001", "locked": True}), encoding="utf-8")
        with pytest.raises(fe.FinalEvalRefused, match="lacks"):
            fe.main(["--spec", str(p), "--bundle", BUNDLE, "--confirm"])

    def test_refuses_cap_other_than_params(self, tmp_path: Path, no_data: None) -> None:
        with pytest.raises(fe.FinalEvalRefused, match="rul_cap"):
            fe.main(["--spec", str(_spec(tmp_path, rul_cap=105)), "--bundle", BUNDLE, "--confirm"])

    def test_refuses_second_run_for_same_spec(
        self, tmp_path: Path, no_data: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fe, "_already_evaluated", lambda h: ["run123"])
        with pytest.raises(fe.FinalEvalRefused, match="already scored"):
            fe.main(["--spec", str(_spec(tmp_path)), "--bundle", BUNDLE, "--confirm"])

    def test_spec_hash_is_order_independent(self) -> None:
        a = {"dataset": "FD001", "model": "lstm", "seed": 1, "rul_cap": 125, "locked": True}
        assert fe.spec_hash(a) == fe.spec_hash(dict(reversed(list(a.items()))))


def _imports(f: Path) -> set[str]:
    tree = ast.parse(f.read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            out.add(node.module)
            out.update(f"{node.module}.{a.name}" for a in node.names)
        elif isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
    return out


class TestImportBan:
    def test_no_module_but_final_eval_imports_it(self) -> None:
        offenders = [
            str(f.relative_to(SRC))
            for f in SRC.rglob("*.py")
            if f.name != "final_eval.py" and "turbofan.evaluation.final_eval" in _imports(f)
        ]
        assert offenders == []

    def test_selection_modules_never_use_the_test_loader(self) -> None:
        for name in SELECTION:
            f = SRC / "evaluation" / f"{name}.py"
            if not f.exists():
                continue
            src = f.read_text(encoding="utf-8")
            assert "load_dataset" not in src, f"{name}.py references load_dataset (reads test)"
            assert "final_eval" not in _imports(f), name

    def test_selection_modules_do_not_load_it_at_runtime(self) -> None:
        mods = ", ".join(
            f"turbofan.evaluation.{m}"
            for m in SELECTION
            if (SRC / "evaluation" / f"{m}.py").exists()
        )
        code = f"import sys, {mods}; print('turbofan.evaluation.final_eval' in sys.modules)"
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "False"


SPEC = {"dataset": "FD001", "model": "ridge", "seed": 42, "rul_cap": 125, "locked": True}


def _manifest(tmp_path: Path, **over: object) -> Path:
    import json

    man = {
        "dataset": "FD001",
        "model": "ridge",
        "seed": 42,
        "version": "current",
        "spec_hash": fe.spec_hash(SPEC),
        "metrics": {"refit": {"n_engines": 100.0}},
    }
    man.update(over)
    d = tmp_path / "bundle"
    d.mkdir(exist_ok=True)
    (d / "manifest.json").write_text(json.dumps(man), encoding="utf-8")
    return d


class TestBundleChecks:
    def test_accepts_the_bundle_built_from_this_spec(self, tmp_path: Path) -> None:
        assert fe.check_bundle(SPEC, _manifest(tmp_path))["version"] == "current"

    @pytest.mark.parametrize(
        ("over", "match"),
        [
            ({"dataset": "FD002"}, "dataset"),
            ({"model": "lstm"}, "model"),
            ({"seed": 7}, "seed"),
            ({"spec_hash": "0" * 64}, "not built from this spec"),
            ({"spec_hash": None}, "not built from this spec"),
            ({"metrics": {"test": {"critical_rmse": 4.0}}}, "not a train_prod bundle"),
            ({"metrics": {"refit": {}, "test": {"critical_rmse": 4.0}}}, "not a train_prod bundle"),
        ],
    )
    def test_refuses_any_other_bundle(
        self, tmp_path: Path, over: dict[str, object], match: str
    ) -> None:
        with pytest.raises(fe.FinalEvalRefused, match=match):
            fe.check_bundle(SPEC, _manifest(tmp_path, **over))

    def test_refuses_a_directory_without_a_manifest(self, tmp_path: Path) -> None:
        with pytest.raises(fe.FinalEvalRefused, match="not a bundle directory"):
            fe.check_bundle(SPEC, tmp_path)

    def test_refuses_before_any_data_is_read(self, tmp_path: Path, no_data: None) -> None:
        bad = _manifest(tmp_path, spec_hash="0" * 64)
        with pytest.raises(fe.FinalEvalRefused, match="not built from this spec"):
            fe.main(["--spec", str(_spec(tmp_path)), "--bundle", str(bad), "--confirm"])
