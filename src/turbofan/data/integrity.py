"""Structure check of the sealed test files, run by ``final_eval`` before it scores anything.

Rule 3 allows the test files to be read only by ``final_eval``; this check is part of that read
and looks at **structure only**: the schema (columns, dtypes, no NaN, contiguous cycles, a
non-negative integer RUL per test engine) and the row / column / unit counts against
``configs/data_expectations.json`` (generated 2026-10-02 by reading the raw files directly). It
catches a corrupted pull or a DVC remote serving different data. It computes no metric and takes
no decision from a value. No module but ``final_eval`` may import it (tests/test_integrity.py).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from turbofan.data.loader import _read_txt
from turbofan.data.schema import validate_raw, validate_rul

__all__ = ["DataIntegrityError", "load_expectations", "check_test_integrity"]

COUNTS = ("test_rows", "test_cols", "test_units", "rul_rows", "rul_cols")


class DataIntegrityError(ValueError):
    """The test files do not match the recorded structure."""


def load_expectations(path: str | Path) -> dict[str, dict[str, int]]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    out: dict[str, dict[str, int]] = doc["datasets"]
    return out


def check_test_integrity(
    dataset: str, raw: str | Path, expectations: dict[str, dict[str, int]]
) -> dict[str, int]:
    """Counts of ``test_<dataset>.txt`` / ``RUL_<dataset>.txt``; raises ``DataIntegrityError``
    listing every deviation from the expectations or the schema."""
    if dataset not in expectations:
        raise DataIntegrityError(f"no recorded expectations for {dataset}")
    want = expectations[dataset]
    test_path, rul_path = Path(raw) / f"test_{dataset}.txt", Path(raw) / f"RUL_{dataset}.txt"
    for p in (test_path, rul_path):
        if not p.is_file():
            raise DataIntegrityError(f"missing test-side file {p}")
    problems: list[str] = []
    test = _read_txt(test_path)
    rul_raw = pd.read_csv(rul_path, header=None)
    try:
        validate_raw(test, str(test_path))
        rul = rul_raw[0].astype(np.int64).rename("rul")
        validate_rul(rul, int(test["unit_id"].nunique()), str(rul_path))
    except ValueError as e:
        problems.append(f"schema: {e}")
    got = {
        "test_rows": len(test),
        "test_cols": test.shape[1],
        "test_units": int(test["unit_id"].nunique()),
        "rul_rows": len(rul_raw),
        "rul_cols": rul_raw.shape[1],
    }
    problems += [f"{k}: {got[k]} != expected {want[k]}" for k in COUNTS if got[k] != want[k]]
    if set(test["unit_id"].unique()) != set(range(1, got["test_units"] + 1)):
        problems.append(f"unit ids are not 1..{got['test_units']}")
    if problems:
        raise DataIntegrityError(
            f"{dataset} test files failed the integrity check:\n" + "\n".join(problems)
        )
    return got
