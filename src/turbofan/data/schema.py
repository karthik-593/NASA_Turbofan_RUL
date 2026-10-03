"""Raw-file schemas checked at load (pandera).

``validate_raw`` / ``validate_rul``: expected columns and dtypes, no NaN, ids >= 1, each
engine's cycles unique and contiguous from 1, and the RUL file (non-negative integers, one
per test engine). A file that breaks them raises ``ValueError`` with the failing cases.
Load-time only — the serving image does not import this module (or pandera).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa

__all__ = ["RAW_SCHEMA", "RUL_SCHEMA", "validate_raw", "validate_rul"]

_RAW_SENSORS = [f"sensor_{i}" for i in range(1, 22)]
_RAW_OPS = ["op_setting_1", "op_setting_2", "op_setting_3"]


def _numeric(s: pd.Series[Any]) -> bool:
    return bool(pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s))


def _contiguous_cycles(df: pd.DataFrame) -> bool:
    g = df.groupby("unit_id")["cycle"]
    return bool(((g.min() == 1) & (g.max() == g.size())).all())


RAW_SCHEMA = pa.DataFrameSchema(
    {
        "unit_id": pa.Column(np.int64, pa.Check.ge(1), nullable=False),
        "cycle": pa.Column(np.int64, pa.Check.ge(1), nullable=False),
        **{
            c: pa.Column(checks=pa.Check(_numeric, element_wise=False), nullable=False)
            for c in _RAW_OPS + _RAW_SENSORS
        },
    },
    checks=pa.Check(_contiguous_cycles, error="cycles per unit must be exactly 1..n"),
    unique=["unit_id", "cycle"],
    strict="filter",  # the loader appends 'rul' to train; it is checked by the loader itself
    ordered=False,
)

RUL_SCHEMA = pa.SeriesSchema(np.int64, pa.Check.ge(0), nullable=False, name="rul")


def validate_raw(df: pd.DataFrame, what: str) -> None:
    """Raise if a raw train/test frame breaks the schema; ``what`` names it in the error."""
    missing = [c for c in ["unit_id", "cycle", *_RAW_OPS, *_RAW_SENSORS] if c not in df.columns]
    if missing:
        raise ValueError(f"{what}: missing columns {missing}")
    try:
        RAW_SCHEMA.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        raise ValueError(f"{what} failed schema validation:\n{e.failure_cases}") from e


def validate_rul(rul: pd.Series[int], n_test_units: int, what: str) -> None:
    try:
        RUL_SCHEMA.validate(rul, lazy=True)
    except pa.errors.SchemaErrors as e:
        raise ValueError(f"{what} failed schema validation:\n{e.failure_cases}") from e
    if len(rul) != n_test_units:
        raise ValueError(f"{what}: {len(rul)} RUL values for {n_test_units} test units")
