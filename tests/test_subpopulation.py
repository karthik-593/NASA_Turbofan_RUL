"""Subpopulation labels (D35): reproduce audit §I, deterministic, never a model input."""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbofan.analysis.subpopulation import sign_vectors, subpopulation_labels

SRC = Path(__file__).resolve().parents[1] / "src" / "turbofan"
# Everything that builds model inputs, trains, or serves must not see the labels.
NO_IMPORT_DIRS = ("features", "models", "serving", "training", "data")


def _synthetic(n_units: int = 12, n_cycles: int = 40, seed: int = 0) -> pd.DataFrame:
    """Two groups of engines whose s2 trends in opposite directions toward failure."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        flip = -1.0 if u <= n_units // 3 else 1.0
        for c in range(1, n_cycles + 1):
            row = {"unit": u, "cycle": c, "op1": 0.0, "op2": 0.0, "op3": 100.0}
            for i in range(1, 22):
                row[f"s{i}"] = float(rng.normal(0, 1) + 0.2 * c)
            row["s2"] = float(flip * 0.2 * c + rng.normal(0, 0.1))
            rows.append(row)
    return pd.DataFrame(rows)


class TestLabels:
    def test_separates_opposite_trend_groups(self) -> None:
        lab = subpopulation_labels(_synthetic(), "FD001")
        assert sorted(lab.value_counts().tolist()) == [4, 8]
        assert (lab.loc[1:4] == 1).all() and (lab.loc[5:] == 0).all()  # 0 = larger group

    def test_deterministic(self) -> None:
        df = _synthetic(seed=3)
        pd.testing.assert_series_equal(
            subpopulation_labels(df, "FD001"), subpopulation_labels(df, "FD001")
        )

    def test_row_order_does_not_matter(self) -> None:
        df = _synthetic(seed=4)
        shuffled = df.sample(frac=1.0, random_state=0)
        pd.testing.assert_series_equal(
            subpopulation_labels(df, "FD001"), subpopulation_labels(shuffled, "FD001")
        )

    def test_one_label_per_engine(self) -> None:
        df = _synthetic()
        lab = subpopulation_labels(df, "FD001")
        assert list(lab.index) == sorted(df["unit"].unique())


@pytest.mark.requires_data
@pytest.mark.parametrize(
    ("dataset", "sizes", "n_sensors"),
    [
        ("FD001", [60, 40], 14),
        ("FD002", [183, 77], 17),
        ("FD003", [56, 44], 14),
        ("FD004", [148, 101], 17),
    ],
)
def test_reproduces_audit_section_I(dataset: str, sizes: list[int], n_sensors: int) -> None:
    """reports/data_audit.md §I: the permutation-supported k = 2 split's cluster sizes and the
    number of sensors in the sign vector."""
    from turbofan.data.loader import load_dataset

    tr, _te, _rul = load_dataset(dataset, "data/raw")
    assert sign_vectors(tr, dataset).shape[1] == n_sensors
    assert subpopulation_labels(tr, dataset).value_counts().sort_index().tolist() == sizes


class TestNeverAModelInput:
    """D35: the labels use the engine's whole future; they may stratify folds and split
    reports, never feed a model."""

    def test_no_feature_model_training_serving_or_data_module_imports_it(self) -> None:
        offenders = []
        for d in NO_IMPORT_DIRS:
            for f in (SRC / d).rglob("*.py"):
                tree = ast.parse(f.read_text(encoding="utf-8"))
                for node in ast.walk(tree):
                    names = []
                    if isinstance(node, ast.ImportFrom) and node.module:
                        names = [node.module]
                    elif isinstance(node, ast.Import):
                        names = [a.name for a in node.names]
                    if any(n.startswith("turbofan.analysis") for n in names):
                        offenders.append(str(f.relative_to(SRC)))
        assert offenders == []

    def test_feature_builders_do_not_load_it_at_runtime(self) -> None:
        code = (
            "import sys, turbofan.features.engineering, turbofan.models.lstm_model, "
            "turbofan.models.xgboost_model, turbofan.serving.app, turbofan.training.train; "
            "print(any(m.startswith('turbofan.analysis') for m in sys.modules))"
        )
        out = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert out.stdout.strip() == "False"
