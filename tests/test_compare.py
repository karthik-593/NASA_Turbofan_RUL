"""Paired comparisons: pairing enforced, per dataset only, cap guard, both readings."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbofan.evaluation.compare import (
    CAP_INVARIANT,
    CapComparisonError,
    CVResult,
    paired_compare,
)


def _points(noise: float, bias: float = 0.0, seed: int = 0, n_units: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for rep in range(2):
        for u in range(1, n_units + 1):
            fold = u % 5
            for c in range(1, 81):
                t = 80 - c
                rows.append(
                    {
                        "unit": u,
                        "cycle": c,
                        "repeat": rep,
                        "fold": fold,
                        "seed": 42,
                        "rul_true": t,
                        "pred": min(t, 125) + bias + rng.normal(0, noise),
                    }
                )
    return pd.DataFrame(rows)


def _res(name: str, pts: pd.DataFrame, cap: float = 125.0, ds: str = "FD001") -> CVResult:
    return CVResult(dataset=ds, name=name, rul_cap=cap, points=pts)


class TestGuards:
    def test_refuses_two_datasets(self) -> None:
        p = _points(1.0)
        with pytest.raises(ValueError, match="never pooled"):
            paired_compare(_res("a", p), _res("b", p, ds="FD002"), np.random.default_rng(0))

    def test_refuses_unpaired_points(self) -> None:
        p = _points(1.0)
        with pytest.raises(ValueError, match="not paired"):
            paired_compare(_res("a", p), _res("b", p.iloc[10:]), np.random.default_rng(0))

    @pytest.mark.parametrize(
        ("metric", "truth"),
        [
            ("rmse", "capped"),
            ("healthy_rmse", "uncapped"),
            ("mae", "uncapped"),
            ("nasa_mean_per_engine", "capped"),
        ],
    )
    def test_cap_guard_refuses_cap_dependent_metrics(self, metric: str, truth: str) -> None:
        p = _points(1.0)
        with pytest.raises(CapComparisonError, match="depends on the cap"):
            paired_compare(
                _res("a", p, cap=125.0),
                _res("b", p, cap=105.0),
                np.random.default_rng(0),
                metric=metric,
                truth=truth,
                n_boot=20,
            )

    @pytest.mark.parametrize(("metric", "truth"), sorted(CAP_INVARIANT))
    def test_cap_guard_allows_cap_invariant_metrics(self, metric: str, truth: str) -> None:
        p = _points(1.0)
        out = paired_compare(
            _res("a", p, cap=125.0),
            _res("b", _points(2.0, seed=1), cap=105.0),
            np.random.default_rng(0),
            metric=metric,
            truth=truth,
            n_boot=50,
        )
        assert np.isfinite(out["diff"])

    def test_same_cap_allows_any_metric(self) -> None:
        p = _points(1.0)
        out = paired_compare(
            _res("a", p),
            _res("b", p),
            np.random.default_rng(0),
            metric="mae",
            truth="capped",
            n_boot=20,
        )
        assert out["diff"] == 0.0


class TestReadings:
    def test_identical_candidates(self) -> None:
        p = _points(1.0)
        out = paired_compare(_res("a", p), _res("b", p), np.random.default_rng(0), n_boot=100)
        assert out["diff"] == out["diff_ci_lo"] == out["diff_ci_hi"] == 0.0
        assert np.isnan(out["wilcoxon_p"])  # no split differs

    def test_detects_a_clearly_better_candidate(self) -> None:
        good, bad = _points(1.0, seed=1), _points(6.0, seed=2)
        out = paired_compare(
            _res("good", good), _res("bad", bad), np.random.default_rng(0), n_boot=300
        )
        assert out["diff"] < 0 and out["diff_ci_hi"] < 0
        assert out["wilcoxon_n_splits"] == 2 * 5
        assert out["wilcoxon_p"] < 0.01
        assert out["share_engines_a_better"] > 0.9
        assert out["n_engines"] == 30

    def test_pairing_cancels_shared_engine_difficulty(self) -> None:
        """Both candidates share large per-engine errors; only a small constant offset
        separates them. The paired CI is much narrower than the per-candidate spread."""
        rng = np.random.default_rng(3)
        base = _points(0.0)
        hard = rng.normal(0, 10, 31)
        base["pred"] = base["pred"] + hard[base["unit"]]
        other = base.copy()
        other["pred"] = other["pred"] + 0.5
        out = paired_compare(
            _res("a", base),
            _res("b", other),
            np.random.default_rng(0),
            metric="mean_signed_error",
            n_boot=300,
        )
        assert out["diff"] == pytest.approx(-0.5)
        assert out["diff_ci_hi"] - out["diff_ci_lo"] < 1e-9
