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

    for name in ("load_dataset", "load_train"):
        monkeypatch.setattr(fe, name, boom)


def _spec(tmp_path: Path, **over: object) -> Path:
    spec = {"dataset": "FD001", "model": "ridge", "seed": 42, "rul_cap": 125, "locked": True}
    spec.update(over)
    p = tmp_path / "spec.yaml"
    p.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return p


class TestRefusals:
    def test_refuses_without_confirm(self, tmp_path: Path, no_data: None) -> None:
        with pytest.raises(fe.FinalEvalRefused, match="--confirm"):
            fe.main(["--spec", str(_spec(tmp_path))])

    def test_refuses_unlocked_spec(self, tmp_path: Path, no_data: None) -> None:
        with pytest.raises(fe.FinalEvalRefused, match="not locked"):
            fe.main(["--spec", str(_spec(tmp_path, locked=False)), "--confirm"])

    def test_refuses_incomplete_spec(self, tmp_path: Path, no_data: None) -> None:
        p = tmp_path / "s.yaml"
        p.write_text(yaml.safe_dump({"dataset": "FD001", "locked": True}), encoding="utf-8")
        with pytest.raises(fe.FinalEvalRefused, match="lacks"):
            fe.main(["--spec", str(p), "--confirm"])

    def test_refuses_cap_other_than_params(self, tmp_path: Path, no_data: None) -> None:
        with pytest.raises(fe.FinalEvalRefused, match="rul_cap"):
            fe.main(["--spec", str(_spec(tmp_path, rul_cap=105)), "--confirm"])

    def test_refuses_second_run_for_same_spec(
        self, tmp_path: Path, no_data: None, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fe, "_already_evaluated", lambda h: ["run123"])
        with pytest.raises(fe.FinalEvalRefused, match="already scored"):
            fe.main(["--spec", str(_spec(tmp_path)), "--confirm"])

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


def test_metrics_out_writes_the_table_as_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    import pandas as pd

    table = pd.DataFrame(
        {
            "truth": ["uncapped", "capped"],
            "metric": ["rmse", "rmse"],
            "estimate": [10.0, 9.0],
            "ci_lo": [8.0, 7.0],
            "ci_hi": [12.0, 11.0],
            "n_engines": [100, 100],
            "n_points": [100, 100],
        }
    )
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    monkeypatch.setattr(fe, "evaluate", lambda spec, raw: (table, {"n_test_engines": 100}))
    out = tmp_path / "out" / "m.json"
    fe.main(["--spec", str(_spec(tmp_path)), "--confirm", "--metrics-out", str(out)])
    doc = json.loads(out.read_text(encoding="utf-8"))
    assert doc["ran"] is True and doc["n_test_engines"] == 100 and doc["model"] == "ridge"
    assert doc["uncapped"]["rmse"] == {
        "estimate": 10.0,
        "ci_lo": 8.0,
        "ci_hi": 12.0,
        "n_engines": 100,
        "n_points": 100,
    }
    assert doc["capped"]["rmse"]["estimate"] == 9.0
