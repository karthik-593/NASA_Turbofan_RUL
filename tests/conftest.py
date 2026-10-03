"""Shared fixtures/hooks for the test suite.

Implements the ``requires_data`` marker: tests that need the real C-MAPSS raw
files (pulled via ``dvc pull``, not committed to git). Behaviour when
``data/raw`` is missing:

- ``REQUIRE_DATA_TESTS=1`` set (Jenkins, where the data is expected to be
  present): FAIL, so a broken/missing data pull is caught loudly.
- Otherwise (GitHub Actions, which does not pull the data): SKIP with reason
  "data not pulled".
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Unit tests fit tiny models and must not initialize CUDA in the shared pytest process: once
# any test does, every later repro.cpu_deterministic() raises GpuUsedError. Hide the GPU from
# torch here, before anything imports it; the real CUDA path is covered by
# tests/test_gpu.py in a subprocess. Set TURBOFAN_TEST_GPU=1 to leave the GPU visible (D45).
if os.environ.get("TURBOFAN_TEST_GPU") != "1":
    # "-1", not "": with an empty value this torch build reports is_available() == True
    # alongside device_count() == 0.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

_DATASETS = ("FD001", "FD002", "FD003", "FD004")
RAW_DIR = Path("data/raw")


def _required_raw_files() -> list[str]:
    files = [f"RUL_{ds}.txt" for ds in _DATASETS]
    files += [f"{kind}_{ds}.txt" for ds in _DATASETS for kind in ("train", "test")]
    return files


def _data_present() -> bool:
    return RAW_DIR.is_dir() and all((RAW_DIR / f).exists() for f in _required_raw_files())


def pytest_runtest_setup(item: pytest.Item) -> None:
    if item.get_closest_marker("requires_data") is None:
        return
    if _data_present():
        return
    if os.environ.get("REQUIRE_DATA_TESTS") == "1":
        pytest.fail(
            "data/raw is missing and REQUIRE_DATA_TESTS=1 — run `dvc pull` to fetch it",
            pytrace=False,
        )
    pytest.skip("data not pulled")
