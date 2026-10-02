"""C-MAPSS data loader.

Handles all four sub-datasets (FD001–FD004).  Each dataset ships as three files:
  - train_FD00x.txt   — multi-unit run-to-failure trajectories
  - test_FD00x.txt    — truncated trajectories (last cycle is the prediction point)
  - RUL_FD00x.txt     — ground-truth RUL at the last observed cycle of each test unit

RUL label construction for training data
-----------------------------------------
True remaining life = max_cycle_for_unit - current_cycle.
Following the standard C-MAPSS convention we cap this at ``rul_cap`` (default 125)
so that the model is not penalised for predicting high RUL when the engine is still
healthy and far from failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd

from turbofan.config import RUL_CAP

# The 26 raw columns in every C-MAPSS text file (space-separated, no header).
_COLUMN_NAMES: list[str] = [
    "unit_id",
    "cycle",
    "op_setting_1",
    "op_setting_2",
    "op_setting_3",
    *[f"sensor_{i}" for i in range(1, 22)],
]

SubDataset = Literal["FD001", "FD002", "FD003", "FD004"]
ALL_SUBDATASETS: tuple[SubDataset, ...] = ("FD001", "FD002", "FD003", "FD004")

DEFAULT_RUL_CAP: int = int(RUL_CAP)


def _read_txt(path: Path) -> pd.DataFrame:
    """Read a whitespace-delimited C-MAPSS file into a DataFrame.

    The files contain trailing whitespace on every row, which produces a
    spurious NaN column when parsed with ``sep=r"\\s+"``.  We drop it.
    """
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
    # Drop any all-NaN trailing columns caused by trailing whitespace.
    df = df.dropna(axis=1, how="all")
    df.columns = pd.Index(_COLUMN_NAMES[: len(df.columns)])
    return df


def _compute_train_rul(df: pd.DataFrame, rul_cap: int) -> pd.DataFrame:
    """Add a ``rul`` column to a training DataFrame.

    For each engine unit the maximum observed cycle is the failure cycle.
    RUL at cycle *t* = min(max_cycle - t, rul_cap).
    """
    max_cycle = df.groupby("unit_id")["cycle"].transform("max")
    df = df.copy()
    df["rul"] = (max_cycle - df["cycle"]).clip(upper=rul_cap)
    return df


@dataclass
class CMAPSSDataset:
    """Container for one C-MAPSS sub-dataset.

    Attributes
    ----------
    name:
        Sub-dataset identifier, e.g. ``"FD001"``.
    train:
        Training trajectories with an appended ``rul`` column.
    test:
        Truncated test trajectories (no ``rul`` column — use ``test_rul``).
    test_rul:
        Ground-truth RUL values at the last observed cycle of each test unit.
        Index aligns with test unit order (1-based engine number).
    rul_cap:
        The cap applied to training RUL labels.
    """

    name: str
    train: pd.DataFrame
    test: pd.DataFrame
    test_rul: pd.Series[int]
    rul_cap: int = field(default=DEFAULT_RUL_CAP)

    @property
    def feature_columns(self) -> list[str]:
        """All sensor + operational-setting column names (excludes unit_id, cycle, rul)."""
        exclude = {"unit_id", "cycle", "rul"}
        return [c for c in self.train.columns if c not in exclude]

    @property
    def n_train_units(self) -> int:
        return int(self.train["unit_id"].nunique())

    @property
    def n_test_units(self) -> int:
        return int(self.test["unit_id"].nunique())

    def __repr__(self) -> str:
        return (
            f"CMAPSSDataset({self.name}, "
            f"train_units={self.n_train_units}, "
            f"test_units={self.n_test_units}, "
            f"rul_cap={self.rul_cap})"
        )


def load_cmapss(
    data_dir: str | Path,
    subset: SubDataset | list[SubDataset] | None = None,
    rul_cap: int = DEFAULT_RUL_CAP,
) -> dict[str, CMAPSSDataset]:
    """Load one or more C-MAPSS sub-datasets from *data_dir*.

    Parameters
    ----------
    data_dir:
        Directory containing the raw ``train_FD00x.txt``, ``test_FD00x.txt``,
        and ``RUL_FD00x.txt`` files.
    subset:
        Which sub-datasets to load.  ``None`` (default) loads all four.
        Pass a single string like ``"FD001"`` or a list to load a subset.
    rul_cap:
        Maximum RUL label value applied to training data (default 125).

    Returns
    -------
    dict[str, CMAPSSDataset]
        Mapping from sub-dataset name (e.g. ``"FD001"``) to its dataset object.

    Raises
    ------
    FileNotFoundError
        If any expected file is missing from *data_dir*.
    ValueError
        If an unknown sub-dataset name is requested.
    """
    data_dir = Path(data_dir)

    if subset is None:
        names: list[SubDataset] = list(ALL_SUBDATASETS)
    elif isinstance(subset, str):
        names = [subset]
    else:
        names = list(subset)

    unknown = set(names) - set(ALL_SUBDATASETS)
    if unknown:
        raise ValueError(f"Unknown sub-dataset(s): {unknown}. Choose from {ALL_SUBDATASETS}.")

    datasets: dict[str, CMAPSSDataset] = {}

    for name in names:
        train_path = data_dir / f"train_{name}.txt"
        test_path = data_dir / f"test_{name}.txt"
        rul_path = data_dir / f"RUL_{name}.txt"

        for p in (train_path, test_path, rul_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Expected file not found: {p}\n"
                    f"Download the C-MAPSS dataset and place the .txt files in {data_dir}"
                )

        train_df = _read_txt(train_path)
        train_df = _compute_train_rul(train_df, rul_cap)

        test_df = _read_txt(test_path)

        # RUL file has one value per test unit, no header.
        rul_series: pd.Series[int] = pd.read_csv(rul_path, header=None, names=["rul"])["rul"]
        # Use 1-based unit index to match unit_id in the test DataFrame.
        rul_series.index = np.arange(1, len(rul_series) + 1)

        datasets[name] = CMAPSSDataset(
            name=name,
            train=train_df,
            test=test_df,
            test_rul=rul_series,
            rul_cap=rul_cap,
        )

    return datasets


# ---------------------------------------------------------------------------
# Notebook-compatible thin adapter
# ---------------------------------------------------------------------------

# Column rename: production loader names → notebook-style names used by
# add_features, eval_lc, and the modeling/comparison notebooks.
_NOTEBOOK_COL_MAP: dict[str, str] = {
    "unit_id": "unit",
    "op_setting_1": "op1",
    "op_setting_2": "op2",
    "op_setting_3": "op3",
    **{f"sensor_{i}": f"s{i}" for i in range(1, 22)},
}


def load_dataset(
    name: str,
    raw: str | Path,
    rul_cap: int = DEFAULT_RUL_CAP,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series[int]]:
    """Return ``(train_df, test_df, rul_series)`` with notebook-style column names.

    Thin adapter over ``load_cmapss`` that renames columns so the output matches
    the ``unit`` / ``opN`` / ``sN`` convention used by ``add_features``, ``eval_lc``,
    and the modeling notebooks.  No data duplication; all heavy lifting stays in
    ``load_cmapss``.

    Parameters
    ----------
    name:
        Sub-dataset identifier, e.g. ``"FD001"``.
    raw:
        Directory containing the raw C-MAPSS text files.
    rul_cap:
        RUL cap applied to training labels (default 125).

    Returns
    -------
    (train_df, test_df, rul_series)
        ``train_df`` has a capped ``rul`` column and notebook-style column names.
        ``rul_series`` is indexed 1..N by unit number.
    """
    ds = load_cmapss(raw, subset=cast(SubDataset, name), rul_cap=int(rul_cap))[name]
    train = ds.train.rename(columns=_NOTEBOOK_COL_MAP)
    test = ds.test.rename(columns=_NOTEBOOK_COL_MAP)
    return train, test, ds.test_rul
