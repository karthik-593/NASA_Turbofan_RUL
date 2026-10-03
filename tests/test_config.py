"""Tests for turbofan.config — params.yaml is the single source and is read strictly."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml

from turbofan import config as cfg

REPO_PARAMS = Path(__file__).resolve().parents[1] / "params.yaml"


def _import_config_with(params: dict, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    """Import turbofan.config in a fresh process pointed at a modified params file."""
    p = tmp_path / "params.yaml"
    p.write_text(yaml.safe_dump(params), encoding="utf-8")
    env = {**os.environ, cfg.PARAMS_ENV: str(p)}
    return subprocess.run(
        [sys.executable, "-c", "import turbofan.config"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _repo_params() -> dict:
    return yaml.safe_load(REPO_PARAMS.read_text(encoding="utf-8"))


class TestValues:
    def test_reads_repo_params_file(self) -> None:
        assert cfg.find_params_file().resolve() == REPO_PARAMS

    def test_values_unchanged_from_pre_params_constants(self) -> None:
        """params.yaml moved the constants; it must not have changed any of them."""
        keep, buckets, seeds = cfg.KEEP, cfg.MAINTENANCE_BUCKETS, cfg.SEEDS
        assert cfg.RUL_CAP == 125.0
        assert (cfg.WINDOW, cfg.SEQ_LEN, cfg.SEED, cfg.SPLIT_FRAC) == (20, 30, 42, 0.8)
        assert (cfg.REGIME_KMEANS_N_INIT, cfg.REGIME_KMEANS_SEED) == (10, 0)
        assert keep == [
            "s2", "s3", "s4", "s7", "s8", "s9", "s11",
            "s12", "s13", "s14", "s15", "s17", "s20", "s21",
        ]  # fmt: skip
        assert len(cfg.FEAT_COLS) == 42 and len(cfg.SENSOR_N_COLS) == 14
        assert buckets == (
            ("critical", 0.0, 25.0),
            ("urgent", 25.0, 50.0),
            ("monitor", 50.0, 100.0),
            ("healthy", 100.0, float("inf")),
        )
        assert seeds == (42, 7, 123, 2024, 99)

    def test_multi_regime_is_derived_from_n_regimes(self) -> None:
        n_regimes, multi = cfg.N_REGIMES, cfg.MULTI_REGIME
        assert n_regimes == {"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6}
        assert multi == frozenset({"FD002", "FD004"})

    def test_candidate_pool_is_audit_E_non_constant_sensors(self) -> None:
        """reports/data_audit.md §E: constant sensors per dataset are excluded."""
        constant = {
            "FD001": {"s1", "s5", "s10", "s16", "s18", "s19"},
            "FD002": {"s1", "s5", "s18", "s19"},
            "FD003": {"s1", "s5", "s16", "s18", "s19"},
            "FD004": {"s1", "s5", "s18", "s19"},
        }
        for ds, const in constant.items():
            expected = {f"s{i}" for i in range(1, 22)} - const
            assert set(cfg.SENSOR_POOL[ds]) == expected, ds
            assert set(cfg.SENSORS[ds]) <= set(cfg.SENSOR_POOL[ds])


class TestStrictLoading:
    def test_missing_key_raises(self, tmp_path: Path) -> None:
        params = _repo_params()
        del params["window"]
        res = _import_config_with(params, tmp_path)
        assert res.returncode != 0
        assert "missing key 'window'" in res.stderr

    def test_wrong_type_raises(self, tmp_path: Path) -> None:
        params = _repo_params()
        params["seq_len"] = "30"
        res = _import_config_with(params, tmp_path)
        assert res.returncode != 0
        assert "'seq_len' must be" in res.stderr

    def test_zero_regimes_raises(self, tmp_path: Path) -> None:
        params = _repo_params()
        params["n_regimes"]["FD001"] = 0
        res = _import_config_with(params, tmp_path)
        assert "n_regimes must be >= 1" in res.stderr

    def test_sensor_outside_pool_raises(self, tmp_path: Path) -> None:
        params = _repo_params()
        params["sensors"]["FD001"]["use"] = [*params["sensors"]["FD001"]["use"], "s1"]
        res = _import_config_with(params, tmp_path)
        assert "within its candidate_pool" in res.stderr

    def test_gapped_buckets_raise(self, tmp_path: Path) -> None:
        params = _repo_params()
        params["buckets"]["urgent"] = [30, 50]
        res = _import_config_with(params, tmp_path)
        assert "not contiguous" in res.stderr

    def test_env_pointing_nowhere_raises(self, tmp_path: Path) -> None:
        env = {**os.environ, cfg.PARAMS_ENV: str(tmp_path / "nope.yaml")}
        res = subprocess.run(
            [sys.executable, "-c", "import turbofan.config"],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert res.returncode != 0
        assert "does not point to a file" in res.stderr

    def test_valid_copy_imports(self, tmp_path: Path) -> None:
        res = _import_config_with(_repo_params(), tmp_path)
        assert res.returncode == 0, res.stderr
