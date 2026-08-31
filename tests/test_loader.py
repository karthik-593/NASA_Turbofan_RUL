"""Tests for the C-MAPSS data loader.

These tests use synthetic fixture data so they run without the real dataset.
The fixtures are generated to be structurally identical to the actual files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from turbofan.data.loader import (
    _COLUMN_NAMES,
    CMAPSSDataset,
    _compute_train_rul,
    _read_txt,
    load_cmapss,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_train_txt(units: int = 3, cycles_per_unit: int = 10) -> str:
    """Generate synthetic train file content matching C-MAPSS format."""
    rows: list[str] = []
    for unit in range(1, units + 1):
        for cycle in range(1, cycles_per_unit + 1):
            # unit_id, cycle, 3 op settings, 21 sensors — all dummy values
            values = [str(unit), str(cycle)] + ["0.0"] * 24
            rows.append(" ".join(values) + " ")  # trailing space, as in real files
    return "\n".join(rows) + "\n"


def _make_test_txt(units: int = 3, cycles_per_unit: int = 5) -> str:
    """Generate synthetic test file content (shorter trajectories)."""
    rows: list[str] = []
    for unit in range(1, units + 1):
        for cycle in range(1, cycles_per_unit + 1):
            values = [str(unit), str(cycle)] + ["0.0"] * 24
            rows.append(" ".join(values) + " ")
    return "\n".join(rows) + "\n"


def _make_rul_txt(units: int = 3, rul_value: int = 50) -> str:
    """Generate synthetic RUL file (one value per test unit)."""
    return "\n".join([str(rul_value)] * units) + "\n"


@pytest.fixture()
def cmapss_dir(tmp_path: Path) -> Path:
    """Write synthetic C-MAPSS files for FD001 into a temp directory."""
    train_content = _make_train_txt(units=5, cycles_per_unit=20)
    test_content = _make_test_txt(units=5, cycles_per_unit=10)
    rul_content = _make_rul_txt(units=5, rul_value=40)

    (tmp_path / "train_FD001.txt").write_text(train_content)
    (tmp_path / "test_FD001.txt").write_text(test_content)
    (tmp_path / "RUL_FD001.txt").write_text(rul_content)
    return tmp_path


# ---------------------------------------------------------------------------
# _read_txt
# ---------------------------------------------------------------------------


class TestReadTxt:
    def test_returns_dataframe(self, cmapss_dir: Path) -> None:
        df = _read_txt(cmapss_dir / "train_FD001.txt")
        assert isinstance(df, pd.DataFrame)

    def test_column_count(self, cmapss_dir: Path) -> None:
        df = _read_txt(cmapss_dir / "train_FD001.txt")
        assert len(df.columns) == 26, f"Expected 26 columns, got {len(df.columns)}"

    def test_column_names(self, cmapss_dir: Path) -> None:
        df = _read_txt(cmapss_dir / "train_FD001.txt")
        assert list(df.columns) == _COLUMN_NAMES

    def test_no_null_columns(self, cmapss_dir: Path) -> None:
        """Trailing whitespace in raw files must not produce spurious NaN columns."""
        df = _read_txt(cmapss_dir / "train_FD001.txt")
        assert not df.isnull().all().any()

    def test_row_count(self, cmapss_dir: Path) -> None:
        df = _read_txt(cmapss_dir / "train_FD001.txt")
        # 5 units × 20 cycles = 100 rows
        assert len(df) == 100

    def test_unit_ids(self, cmapss_dir: Path) -> None:
        df = _read_txt(cmapss_dir / "train_FD001.txt")
        assert set(df["unit_id"].unique()) == {1, 2, 3, 4, 5}

    def test_sensor_column_names(self, cmapss_dir: Path) -> None:
        df = _read_txt(cmapss_dir / "train_FD001.txt")
        sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
        assert len(sensor_cols) == 21
        assert sensor_cols == [f"sensor_{i}" for i in range(1, 22)]

    def test_op_setting_column_names(self, cmapss_dir: Path) -> None:
        df = _read_txt(cmapss_dir / "train_FD001.txt")
        op_cols = [c for c in df.columns if c.startswith("op_setting_")]
        assert op_cols == ["op_setting_1", "op_setting_2", "op_setting_3"]


# ---------------------------------------------------------------------------
# _compute_train_rul
# ---------------------------------------------------------------------------


class TestComputeTrainRUL:
    def _make_df(self) -> pd.DataFrame:
        """Two units: unit 1 has 10 cycles, unit 2 has 5 cycles."""
        rows = [(1, c) for c in range(1, 11)] + [(2, c) for c in range(1, 6)]
        return pd.DataFrame(rows, columns=["unit_id", "cycle"])

    def test_rul_column_added(self) -> None:
        df = _compute_train_rul(self._make_df(), rul_cap=125)
        assert "rul" in df.columns

    def test_rul_at_last_cycle_is_zero(self) -> None:
        df = _compute_train_rul(self._make_df(), rul_cap=125)
        last_cycles = df.groupby("unit_id")["cycle"].transform("max")
        assert (df.loc[df["cycle"] == last_cycles, "rul"] == 0).all()

    def test_rul_values_unit1(self) -> None:
        df = _compute_train_rul(self._make_df(), rul_cap=125)
        unit1 = df[df["unit_id"] == 1].sort_values("cycle")
        # max_cycle=10; RUL = 10-cycle, capped at 125
        expected = list(range(9, -1, -1))  # [9, 8, ..., 0]
        assert list(unit1["rul"]) == expected

    def test_rul_cap_applied(self) -> None:
        """RUL should never exceed rul_cap even for early cycles."""
        rows = [(1, c) for c in range(1, 201)]  # 200-cycle unit
        df = pd.DataFrame(rows, columns=["unit_id", "cycle"])
        df = _compute_train_rul(df, rul_cap=125)
        assert df["rul"].max() == 125

    def test_rul_cap_zero_means_always_zero(self) -> None:
        df = _compute_train_rul(self._make_df(), rul_cap=0)
        assert (df["rul"] == 0).all()

    def test_original_df_not_mutated(self) -> None:
        original = self._make_df()
        original_copy = original.copy()
        _compute_train_rul(original, rul_cap=125)
        pd.testing.assert_frame_equal(original, original_copy)


# ---------------------------------------------------------------------------
# load_cmapss
# ---------------------------------------------------------------------------


class TestLoadCMAPSS:
    def test_returns_dict(self, cmapss_dir: Path) -> None:
        result = load_cmapss(cmapss_dir, subset="FD001")
        assert isinstance(result, dict)
        assert "FD001" in result

    def test_dataset_type(self, cmapss_dir: Path) -> None:
        result = load_cmapss(cmapss_dir, subset="FD001")
        assert isinstance(result["FD001"], CMAPSSDataset)

    def test_train_shape(self, cmapss_dir: Path) -> None:
        ds = load_cmapss(cmapss_dir, subset="FD001")["FD001"]
        # 5 units × 20 cycles, 26 raw cols + rul = 27 columns
        assert ds.train.shape == (100, 27)

    def test_test_shape(self, cmapss_dir: Path) -> None:
        ds = load_cmapss(cmapss_dir, subset="FD001")["FD001"]
        # 5 units × 10 cycles, 26 columns (no rul)
        assert ds.test.shape == (50, 26)

    def test_train_has_rul_column(self, cmapss_dir: Path) -> None:
        ds = load_cmapss(cmapss_dir, subset="FD001")["FD001"]
        assert "rul" in ds.train.columns

    def test_test_has_no_rul_column(self, cmapss_dir: Path) -> None:
        ds = load_cmapss(cmapss_dir, subset="FD001")["FD001"]
        assert "rul" not in ds.test.columns

    def test_test_rul_length(self, cmapss_dir: Path) -> None:
        ds = load_cmapss(cmapss_dir, subset="FD001")["FD001"]
        assert len(ds.test_rul) == 5  # one per test unit

    def test_test_rul_values(self, cmapss_dir: Path) -> None:
        ds = load_cmapss(cmapss_dir, subset="FD001")["FD001"]
        assert (ds.test_rul == 40).all()

    def test_test_rul_index_is_one_based(self, cmapss_dir: Path) -> None:
        ds = load_cmapss(cmapss_dir, subset="FD001")["FD001"]
        assert list(ds.test_rul.index) == [1, 2, 3, 4, 5]

    def test_rul_cap_respected(self, cmapss_dir: Path) -> None:
        ds = load_cmapss(cmapss_dir, subset="FD001", rul_cap=5)["FD001"]
        assert ds.train["rul"].max() <= 5

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Expected file not found"):
            load_cmapss(tmp_path, subset="FD001")

    def test_unknown_subset_raises(self, cmapss_dir: Path) -> None:
        with pytest.raises(ValueError, match="Unknown sub-dataset"):
            load_cmapss(cmapss_dir, subset="FD999")  # type: ignore[arg-type]

    def test_feature_columns_excludes_metadata(self, cmapss_dir: Path) -> None:
        ds = load_cmapss(cmapss_dir, subset="FD001")["FD001"]
        for excluded in ("unit_id", "cycle", "rul"):
            assert excluded not in ds.feature_columns

    def test_feature_columns_count(self, cmapss_dir: Path) -> None:
        ds = load_cmapss(cmapss_dir, subset="FD001")["FD001"]
        # 3 op settings + 21 sensors = 24
        assert len(ds.feature_columns) == 24

    def test_n_train_units(self, cmapss_dir: Path) -> None:
        ds = load_cmapss(cmapss_dir, subset="FD001")["FD001"]
        assert ds.n_train_units == 5

    def test_n_test_units(self, cmapss_dir: Path) -> None:
        ds = load_cmapss(cmapss_dir, subset="FD001")["FD001"]
        assert ds.n_test_units == 5

    def test_column_names_in_train(self, cmapss_dir: Path) -> None:
        ds = load_cmapss(cmapss_dir, subset="FD001")["FD001"]
        expected_prefix = ["unit_id", "cycle", "op_setting_1", "op_setting_2", "op_setting_3"]
        assert list(ds.train.columns[:5]) == expected_prefix

    def test_all_sensor_columns_present(self, cmapss_dir: Path) -> None:
        ds = load_cmapss(cmapss_dir, subset="FD001")["FD001"]
        for i in range(1, 22):
            assert f"sensor_{i}" in ds.train.columns

    def test_path_as_string(self, cmapss_dir: Path) -> None:
        """load_cmapss should accept a plain string path."""
        result = load_cmapss(str(cmapss_dir), subset="FD001")
        assert "FD001" in result

    def test_repr(self, cmapss_dir: Path) -> None:
        ds = load_cmapss(cmapss_dir, subset="FD001")["FD001"]
        assert "FD001" in repr(ds)
        assert "train_units=5" in repr(ds)


# ---------------------------------------------------------------------------
# Integration: multiple fixture files → load all subsets
# ---------------------------------------------------------------------------


@pytest.fixture()
def full_cmapss_dir(tmp_path: Path) -> Path:
    """Write synthetic files for all four sub-datasets."""
    for name in ("FD001", "FD002", "FD003", "FD004"):
        (tmp_path / f"train_{name}.txt").write_text(_make_train_txt(units=4, cycles_per_unit=8))
        (tmp_path / f"test_{name}.txt").write_text(_make_test_txt(units=4, cycles_per_unit=4))
        (tmp_path / f"RUL_{name}.txt").write_text(_make_rul_txt(units=4, rul_value=30))
    return tmp_path


class TestLoadAllSubsets:
    def test_returns_four_datasets(self, full_cmapss_dir: Path) -> None:
        result = load_cmapss(full_cmapss_dir)
        assert set(result.keys()) == {"FD001", "FD002", "FD003", "FD004"}

    def test_all_have_correct_train_columns(self, full_cmapss_dir: Path) -> None:
        result = load_cmapss(full_cmapss_dir)
        for name, ds in result.items():
            assert list(ds.train.columns) == _COLUMN_NAMES + ["rul"], name

    def test_subset_list(self, full_cmapss_dir: Path) -> None:
        result = load_cmapss(full_cmapss_dir, subset=["FD001", "FD003"])
        assert set(result.keys()) == {"FD001", "FD003"}
