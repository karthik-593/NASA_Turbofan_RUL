"""The test-file structure check final_eval runs before scoring (rule 3: structure only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from turbofan.data.integrity import DataIntegrityError, check_test_integrity, load_expectations

N_UNITS = 6


def _write(raw: Path, rows_per_unit: int = 12) -> dict[str, dict[str, int]]:
    rng = np.random.default_rng(0)
    rows = [
        [u, c, *rng.normal(size=3), *rng.normal(size=21)]
        for u in range(1, N_UNITS + 1)
        for c in range(1, rows_per_unit + 1)
    ]
    raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(raw / "test_FD001.txt", sep=" ", header=False, index=False)
    pd.Series(range(5, 5 + N_UNITS)).to_csv(raw / "RUL_FD001.txt", header=False, index=False)
    return {
        "FD001": {
            "test_rows": N_UNITS * rows_per_unit,
            "test_cols": 26,
            "test_units": N_UNITS,
            "rul_rows": N_UNITS,
            "rul_cols": 1,
        }
    }


def test_matching_files_pass_and_return_the_counts(tmp_path: Path) -> None:
    exp = _write(tmp_path)
    assert check_test_integrity("FD001", tmp_path, exp) == exp["FD001"]


@pytest.mark.parametrize("key", ["test_rows", "test_cols", "test_units", "rul_rows", "rul_cols"])
def test_every_count_is_checked(tmp_path: Path, key: str) -> None:
    exp = _write(tmp_path)
    exp["FD001"][key] += 1
    with pytest.raises(DataIntegrityError, match=f"{key}: .* != expected"):
        check_test_integrity("FD001", tmp_path, exp)


def test_all_deviations_are_listed_together(tmp_path: Path) -> None:
    exp = _write(tmp_path)
    exp["FD001"]["test_rows"] += 1
    exp["FD001"]["rul_rows"] += 1
    with pytest.raises(DataIntegrityError) as e:
        check_test_integrity("FD001", tmp_path, exp)
    assert "test_rows" in str(e.value) and "rul_rows" in str(e.value)


def test_schema_violation_is_reported(tmp_path: Path) -> None:
    exp = _write(tmp_path)
    df = pd.read_csv(tmp_path / "test_FD001.txt", sep=" ", header=None)
    df.iloc[3, 7] = np.nan
    df.to_csv(tmp_path / "test_FD001.txt", sep=" ", header=False, index=False)
    with pytest.raises(DataIntegrityError, match="schema"):
        check_test_integrity("FD001", tmp_path, exp)


def test_rul_count_must_match_test_engines(tmp_path: Path) -> None:
    exp = _write(tmp_path)
    pd.Series(range(N_UNITS + 2)).to_csv(tmp_path / "RUL_FD001.txt", header=False, index=False)
    with pytest.raises(DataIntegrityError, match="RUL values"):
        check_test_integrity("FD001", tmp_path, exp)


def test_missing_file_and_unknown_dataset(tmp_path: Path) -> None:
    exp = _write(tmp_path)
    (tmp_path / "RUL_FD001.txt").unlink()
    with pytest.raises(DataIntegrityError, match="missing test-side file"):
        check_test_integrity("FD001", tmp_path, exp)
    with pytest.raises(DataIntegrityError, match="no recorded expectations"):
        check_test_integrity("FD002", tmp_path, exp)


def test_the_committed_expectations_cover_every_dataset() -> None:
    exp = load_expectations(Path(__file__).parents[1] / "configs" / "data_expectations.json")
    assert set(exp) == {"FD001", "FD002", "FD003", "FD004"}
    assert all(
        {"test_rows", "test_cols", "test_units", "rul_rows", "rul_cols"} <= set(v)
        for v in exp.values()
    )


@pytest.mark.requires_data
@pytest.mark.parametrize("dataset", ["FD001", "FD002", "FD003", "FD004"])
def test_real_test_files_pass(dataset: str) -> None:
    exp = load_expectations(Path(__file__).parents[1] / "configs" / "data_expectations.json")
    got = check_test_integrity(dataset, "data/raw", exp)
    assert got == {k: exp[dataset][k] for k in got}
    json.dumps(got)  # plain ints, loggable as an MLflow artifact
