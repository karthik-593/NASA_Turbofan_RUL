"""Tests for turbofan.stats — engine-level bootstrap CI and two-sample KS with CI."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as sps

from turbofan.stats import bootstrap_ci, ks_2samp_ci

N_BOOT = 500
CI = 0.95


def _reference_boot(values, rng, n_boot, ci_level, stat=np.median):
    """The data-audit notebook's original implementation, kept verbatim as the oracle."""
    alpha = 1 - ci_level
    v = np.asarray(values, dtype=float)
    n = len(v)
    idx = rng.integers(0, n, size=(n_boot, n))
    bs = stat(v[idx], axis=1)
    lo, hi = np.quantile(bs, [alpha / 2, 1 - alpha / 2])
    return float(stat(v)), float(lo), float(hi), n


def _reference_ks(groups_a, b, rng, n_boot, ci_level):
    alpha = 1 - ci_level
    est = sps.ks_2samp(np.concatenate(groups_a), b).statistic
    na, nb = len(groups_a), len(b)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        ia = rng.integers(0, na, na)
        ib = rng.integers(0, nb, nb)
        bs[i] = sps.ks_2samp(np.concatenate([groups_a[k] for k in ia]), b[ib]).statistic
    lo, hi = np.quantile(bs, [alpha / 2, 1 - alpha / 2])
    return float(est), float(lo), float(hi)


class TestBootstrapCI:
    def test_matches_audit_reference_bit_for_bit(self) -> None:
        v = np.random.default_rng(1).normal(size=137)
        got = bootstrap_ci(v, np.random.default_rng(7), n_boot=N_BOOT, ci_level=CI)
        want = _reference_boot(v, np.random.default_rng(7), N_BOOT, CI)
        assert got == want

    def test_custom_stat_matches_reference(self) -> None:
        v = np.random.default_rng(2).integers(0, 2, size=80).astype(float)
        got = bootstrap_ci(v, np.random.default_rng(3), n_boot=N_BOOT, ci_level=CI, stat=np.mean)
        want = _reference_boot(v, np.random.default_rng(3), N_BOOT, CI, stat=np.mean)
        assert got == want

    def test_returns_n_and_ordered_interval(self) -> None:
        v = np.random.default_rng(4).normal(size=60)
        est, lo, hi, n = bootstrap_ci(v, np.random.default_rng(5), n_boot=N_BOOT, ci_level=CI)
        assert n == 60
        assert lo <= est <= hi

    def test_constant_values_give_degenerate_interval(self) -> None:
        est, lo, hi, _ = bootstrap_ci(
            np.full(25, 3.0), np.random.default_rng(0), n_boot=N_BOOT, ci_level=CI
        )
        assert est == lo == hi == 3.0

    def test_same_seed_same_result(self) -> None:
        v = np.random.default_rng(6).normal(size=40)
        a = bootstrap_ci(v, np.random.default_rng(9), n_boot=N_BOOT, ci_level=CI)
        b = bootstrap_ci(v, np.random.default_rng(9), n_boot=N_BOOT, ci_level=CI)
        assert a == b

    def test_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            bootstrap_ci([1.0, np.nan], np.random.default_rng(0), n_boot=N_BOOT, ci_level=CI)

    @pytest.mark.parametrize("values", [[], [[1.0, 2.0]]])
    def test_empty_or_2d_raises(self, values) -> None:
        with pytest.raises(ValueError, match="1-D"):
            bootstrap_ci(values, np.random.default_rng(0), n_boot=N_BOOT, ci_level=CI)

    @pytest.mark.parametrize("ci_level", [0.0, 1.0, 1.5])
    def test_bad_ci_level_raises(self, ci_level: float) -> None:
        with pytest.raises(ValueError, match="ci_level"):
            bootstrap_ci([1.0, 2.0], np.random.default_rng(0), n_boot=N_BOOT, ci_level=ci_level)

    def test_bad_n_boot_raises(self) -> None:
        with pytest.raises(ValueError, match="n_boot"):
            bootstrap_ci([1.0, 2.0], np.random.default_rng(0), n_boot=0, ci_level=CI)


class TestKs2sampCI:
    @staticmethod
    def _groups(seed: int, n_engines: int, shift: float = 0.0) -> list[np.ndarray]:
        rng = np.random.default_rng(seed)
        return [rng.normal(shift, size=rng.integers(3, 12)) for _ in range(n_engines)]

    def test_matches_audit_reference_bit_for_bit(self) -> None:
        ga = self._groups(0, 20)
        b = np.random.default_rng(1).normal(0.5, size=90)
        got = ks_2samp_ci(ga, b, np.random.default_rng(11), n_boot=200, ci_level=CI)
        want = _reference_ks(ga, b, np.random.default_rng(11), 200, CI)
        assert got == want

    def test_point_estimate_is_scipy_ks_on_pooled_groups(self) -> None:
        ga = self._groups(2, 15)
        b = np.random.default_rng(3).normal(size=50)
        est, _, _ = ks_2samp_ci(ga, b, np.random.default_rng(0), n_boot=50, ci_level=CI)
        assert est == sps.ks_2samp(np.concatenate(ga), b).statistic

    def test_disjoint_samples_have_distance_one(self) -> None:
        ga = [np.array([0.0, 1.0]), np.array([2.0])]
        b = np.array([10.0, 11.0, 12.0])
        est, lo, hi = ks_2samp_ci(ga, b, np.random.default_rng(0), n_boot=50, ci_level=CI)
        assert est == lo == hi == 1.0

    def test_identical_samples_have_distance_zero(self) -> None:
        b = np.arange(10, dtype=float)
        est, lo, hi = ks_2samp_ci([b], b, np.random.default_rng(0), n_boot=50, ci_level=CI)
        assert est == 0.0
        assert lo <= hi

    def test_resamples_whole_engines(self) -> None:
        """With one engine on side a, every resample draws that engine: the bootstrap
        distribution of side a is fixed, so only side b's resampling moves the statistic."""
        ga = [np.array([0.0, 0.0, 0.0])]
        b = np.array([5.0, 6.0])
        est, lo, hi = ks_2samp_ci(ga, b, np.random.default_rng(0), n_boot=50, ci_level=CI)
        assert est == lo == hi == 1.0

    def test_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            ks_2samp_ci(
                [np.array([1.0, np.nan])],
                np.array([1.0]),
                np.random.default_rng(0),
                n_boot=10,
                ci_level=CI,
            )

    def test_empty_input_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            ks_2samp_ci([], np.array([1.0]), np.random.default_rng(0), n_boot=10, ci_level=CI)
        with pytest.raises(ValueError, match="non-empty"):
            ks_2samp_ci(
                [np.array([1.0])], np.array([]), np.random.default_rng(0), n_boot=10, ci_level=CI
            )
