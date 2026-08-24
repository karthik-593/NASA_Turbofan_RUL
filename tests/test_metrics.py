"""Tests for evaluation metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from turbofan.evaluation.metrics import cmapss_score, mae, rmse

# ---------------------------------------------------------------------------
# RMSE
# ---------------------------------------------------------------------------


class TestRMSE:
    def test_perfect_prediction(self) -> None:
        assert rmse([10, 20, 30], [10, 20, 30]) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        # errors = [2, -2] → MSE = 4 → RMSE = 2
        assert rmse([10, 10], [12, 8]) == pytest.approx(2.0)

    def test_single_element(self) -> None:
        assert rmse([5], [8]) == pytest.approx(3.0)

    def test_symmetry(self) -> None:
        """RMSE is symmetric in the sign of the error."""
        assert rmse([0], [5]) == pytest.approx(rmse([0], [-5]))

    def test_returns_float(self) -> None:
        result = rmse([1, 2], [3, 4])
        assert isinstance(result, float)

    def test_numpy_arrays(self) -> None:
        y_true = np.array([10.0, 20.0])
        y_pred = np.array([11.0, 19.0])
        assert rmse(y_true, y_pred) == pytest.approx(1.0)

    def test_large_errors_penalised_more(self) -> None:
        """RMSE penalises large errors more than MAE."""
        # Two predictions with the same MAE but different distributions
        assert rmse([0, 0], [10, 10]) < rmse([0, 0], [20, 0])


# ---------------------------------------------------------------------------
# MAE
# ---------------------------------------------------------------------------


class TestMAE:
    def test_perfect_prediction(self) -> None:
        assert mae([10, 20, 30], [10, 20, 30]) == pytest.approx(0.0)

    def test_known_value(self) -> None:
        # |3| + |−1| = 4, mean = 2
        assert mae([10, 10], [13, 9]) == pytest.approx(2.0)

    def test_single_element(self) -> None:
        assert mae([100], [90]) == pytest.approx(10.0)

    def test_symmetry(self) -> None:
        assert mae([0], [7]) == pytest.approx(mae([0], [-7]))

    def test_returns_float(self) -> None:
        assert isinstance(mae([1], [2]), float)

    def test_mae_leq_rmse(self) -> None:
        """MAE ≤ RMSE always holds (Cauchy–Schwarz)."""
        y_true = [10, 20, 15, 5]
        y_pred = [12, 18, 20, 3]
        assert mae(y_true, y_pred) <= rmse(y_true, y_pred) + 1e-9


# ---------------------------------------------------------------------------
# CMAPSS Score
# ---------------------------------------------------------------------------


class TestCMAPSSScore:
    def test_perfect_prediction_is_zero(self) -> None:
        assert cmapss_score([100], [100]) == pytest.approx(0.0)

    def test_multiple_perfect_predictions(self) -> None:
        assert cmapss_score([10, 20, 30], [10, 20, 30]) == pytest.approx(0.0)

    def test_late_prediction_positive_score(self) -> None:
        """d > 0 (late prediction) must give a positive score."""
        score = cmapss_score([100], [110])  # d = +10
        assert score > 0

    def test_early_prediction_positive_score(self) -> None:
        """d < 0 (early prediction) still gives a positive penalty.

        exp(-d/13) - 1 with d=-10 → exp(10/13)-1 > 0.
        Both early and late predictions incur a positive cost; the asymmetry
        is in the *rate of growth*, not the sign.
        """
        score = cmapss_score([100], [90])  # d = -10
        assert score > 0

    def test_late_penalised_more_than_early(self) -> None:
        """Absolute score for d=+k must exceed absolute score for d=−k."""
        delta = 15
        late = cmapss_score([100], [100 + delta])  # d = +15
        early = cmapss_score([100], [100 - delta])  # d = -15
        assert abs(late) > abs(early)

    def test_known_late_value(self) -> None:
        # d = +10 → exp(10/10) - 1 = e - 1 ≈ 1.71828
        expected = math.exp(10 / 10) - 1
        assert cmapss_score([0], [10]) == pytest.approx(expected, rel=1e-5)

    def test_known_early_value(self) -> None:
        # d = -13 → exp(13/13) - 1 = e - 1 ≈ 1.71828  (but NEGATIVE because early)
        # Wait: score = exp(-d/13) - 1 = exp(13/13) - 1 = e - 1
        # But the score is positive for d<0 according to the formula…
        # Re-check: s_i = exp(-d/13) - 1 for d<0
        # d = -13 → -d = 13 → exp(13/13)-1 = e-1 ≈ 1.718, which is > 0
        # So early predictions also give positive s_i?  No — re-read the spec:
        # The original paper sums are always >= 0; the score represents a penalty.
        # Both cases return positive values (the penalty for error in either direction).
        # d<0 means predicted RUL < actual → EARLY prediction, penalty = exp(-d/13)-1
        # Since d<0, -d>0, exp(-d/13)>1, so score > 0. ✓
        d = -13.0
        expected = math.exp(-d / 13.0) - 1  # = e - 1
        assert cmapss_score([100], [100 + d]) == pytest.approx(expected, rel=1e-5)

    def test_score_is_summed_not_averaged(self) -> None:
        """Score should be the SUM across units, not the mean."""
        single = cmapss_score([100], [110])
        double = cmapss_score([100, 100], [110, 110])
        assert double == pytest.approx(2 * single)

    def test_mixed_early_and_late(self) -> None:
        # d1 = +10 → exp(1)-1;  d2 = -10 → exp(10/13)-1
        expected = (math.exp(10 / 10) - 1) + (math.exp(10 / 13) - 1)
        assert cmapss_score([100, 100], [110, 90]) == pytest.approx(expected, rel=1e-5)

    def test_returns_float(self) -> None:
        assert isinstance(cmapss_score([1], [1]), float)

    def test_numpy_input(self) -> None:
        y_true = np.array([100.0, 80.0])
        y_pred = np.array([110.0, 70.0])
        result = cmapss_score(y_true, y_pred)
        assert isinstance(result, float)

    def test_score_grows_with_error_magnitude(self) -> None:
        """Larger errors should produce larger scores."""
        small = cmapss_score([100], [105])
        large = cmapss_score([100], [120])
        assert large > small

    def test_boundary_d_equals_zero(self) -> None:
        """d = 0 must produce score 0 (both branches converge)."""
        assert cmapss_score([50], [50]) == pytest.approx(0.0)
