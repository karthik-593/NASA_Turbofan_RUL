"""Tests for turbofan.evaluation.protocol — score() and eval_lc()."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbofan.config import FEAT_COLS
from turbofan.evaluation.protocol import eval_lc, score

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _ZeroModel:
    def predict(self, X):
        return np.zeros(len(X))


class _ConstModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def _make_feat_df(n_units: int = 3, cycles_per_unit: int = 5) -> pd.DataFrame:
    """Synthetic feat_df with FEAT_COLS plus unit, cycle, rul columns."""
    rows = []
    for uid in range(1, n_units + 1):
        for cyc in range(1, cycles_per_unit + 1):
            row: dict = {"unit": uid, "cycle": cyc, "rul": float(cycles_per_unit - cyc)}
            for col in FEAT_COLS:
                row[col] = 0.0
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# score()
# ---------------------------------------------------------------------------


class TestScore:
    def test_perfect_prediction_global_rmse_zero(self) -> None:
        y = np.array([10.0, 20.0, 30.0])
        out = score(y, y)
        assert out["global_rmse"] == pytest.approx(0.0)

    def test_perfect_prediction_nasa_zero(self) -> None:
        y = np.array([10.0, 20.0, 30.0])
        out = score(y, y)
        assert out["nasa"] == pytest.approx(0.0)

    def test_known_global_rmse(self) -> None:
        y = np.array([0.0, 0.0])
        pred = np.array([3.0, 4.0])
        # MSE = (9 + 16) / 2 = 12.5  →  RMSE = sqrt(12.5)
        out = score(y, pred)
        assert out["global_rmse"] == pytest.approx(np.sqrt(12.5))

    def test_late_pct_all_late(self) -> None:
        y = np.array([10.0, 20.0, 30.0])
        pred = y + 1.0
        out = score(y, pred)
        assert out["late_pct"] == pytest.approx(100.0)

    def test_late_pct_none_late(self) -> None:
        y = np.array([10.0, 20.0, 30.0])
        pred = y - 1.0
        out = score(y, pred)
        assert out["late_pct"] == pytest.approx(0.0)

    def test_late_pct_half(self) -> None:
        y = np.array([10.0, 20.0])
        pred = np.array([15.0, 15.0])  # first late, second early
        out = score(y, pred)
        assert out["late_pct"] == pytest.approx(50.0)

    def test_critical_rmse_covers_only_low_rul(self) -> None:
        """critical_rmse must use only samples with true RUL < 25."""
        y = np.array([10.0, 60.0])  # [0] in critical, [1] in monitor
        pred = np.array([15.0, 60.0])  # error 5 in critical, 0 in monitor
        out = score(y, pred)
        assert out["critical_rmse"] == pytest.approx(5.0)
        assert out["monitor_rmse"] == pytest.approx(0.0)

    def test_all_expected_keys_present(self) -> None:
        out = score(np.array([10.0]), np.array([10.0]))
        for k in (
            "global_rmse",
            "nasa",
            "late_pct",
            "n",
            "critical_rmse",
            "urgent_rmse",
            "monitor_rmse",
            "healthy_rmse",
        ):
            assert k in out, f"Missing key: {k}"

    def test_n_equals_input_length(self) -> None:
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        out = score(y, y)
        assert out["n"] == 5

    def test_nasa_late_higher_than_early(self) -> None:
        """NASA score penalises late predictions more than early ones of equal magnitude."""
        y = np.array([50.0])
        assert score(y, y + 10)["nasa"] > score(y, y - 10)["nasa"]

    def test_accepts_list_inputs(self) -> None:
        """score() should accept plain Python lists, not just numpy arrays."""
        out = score([10.0, 20.0], [10.0, 20.0])
        assert out["global_rmse"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# eval_lc()
# ---------------------------------------------------------------------------


class TestEvalLc:
    def test_picks_last_cycle_per_unit(self) -> None:
        """eval_lc must evaluate each unit at its maximum-cycle row."""
        df = _make_feat_df(n_units=2, cycles_per_unit=5)
        # cycles_per_unit=5 → last cycle has rul=0 for every unit
        _, y, _ = eval_lc(_ZeroModel(), df)
        np.testing.assert_array_equal(y, [0.0, 0.0])

    def test_n_engines_equals_unit_count(self) -> None:
        df = _make_feat_df(n_units=4, cycles_per_unit=3)
        out, _, _ = eval_lc(_ZeroModel(), df)
        assert out["n_engines"] == 4

    def test_returns_three_tuple(self) -> None:
        df = _make_feat_df()
        result = eval_lc(_ZeroModel(), df)
        assert isinstance(result, tuple) and len(result) == 3

    def test_metrics_dict_has_standard_keys(self) -> None:
        df = _make_feat_df()
        out, _, _ = eval_lc(_ZeroModel(), df)
        for k in ("rmse", "cmapss_score", "late_pct", "n_engines"):
            assert k in out, f"Missing key: {k}"

    def test_true_rul_overrides_rul_column(self) -> None:
        """When true_rul is passed it must be used instead of the 'rul' column."""
        df = _make_feat_df(n_units=2, cycles_per_unit=3)
        df["rul"] = 999.0  # sentinel; must NOT appear in y
        true_rul = pd.Series([10.0, 20.0], index=[1, 2])
        _, y, _ = eval_lc(_ZeroModel(), df, true_rul=true_rul)
        assert set(y) == {10.0, 20.0}

    def test_prediction_values_forwarded(self) -> None:
        """y_pred in the returned tuple must come from model.predict."""
        df = _make_feat_df(n_units=3, cycles_per_unit=2)
        _, _, pred = eval_lc(_ConstModel(42.0), df)
        np.testing.assert_array_equal(pred, [42.0, 42.0, 42.0])

    def test_bucket_keys_present(self) -> None:
        df = _make_feat_df(n_units=2, cycles_per_unit=5)
        out, _, _ = eval_lc(_ZeroModel(), df)
        assert "critical_rmse" in out
