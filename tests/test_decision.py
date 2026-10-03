"""Decision curves for "remove when predicted RUL <= T"."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbofan.evaluation.decision import (
    caught_at_budget,
    decision_curve,
    matched_budget,
    removal_lead_times,
)


def _traj(n_units: int = 10, life: int = 100, bias: float = 0.0, repeats: int = 1) -> pd.DataFrame:
    rows = []
    for r in range(repeats):
        for u in range(1, n_units + 1):
            for c in range(1, life + 1):
                t = life - c
                rows.append(
                    {
                        "unit": u,
                        "repeat": r,
                        "seed": 42,
                        "cycle": c,
                        "rul_true": t,
                        "pred": t + bias,
                    }
                )
    return pd.DataFrame(rows)


def _q(cur: pd.DataFrame, name: str, T: float) -> float:
    return float(cur[(cur["quantity"] == name) & (cur["threshold"] == T)]["estimate"].iloc[0])


def test_perfect_model_removes_at_threshold() -> None:
    lead = removal_lead_times(_traj(), thresholds=[0.0, 20.0, 50.0])
    assert (lead[20.0] == 20).all() and (lead[50.0] == 50).all() and (lead[0.0] == 0).all()


def test_late_model_never_triggers_low_thresholds() -> None:
    lead = removal_lead_times(_traj(bias=30.0), thresholds=[10.0, 40.0])
    assert lead[10.0].isna().all()  # prediction never reaches 10: fails in service
    assert (lead[40.0] == 10).all()


def test_first_trigger_counts_even_if_prediction_rises_again() -> None:
    p = _traj(n_units=1, life=50)
    p.loc[p["cycle"] == 5, "pred"] = 0.0  # one early dip
    lead = removal_lead_times(p, thresholds=[10.0])
    assert lead[10.0].iloc[0] == 45


def test_curve_quantities() -> None:
    cur = decision_curve(
        _traj(),
        np.random.default_rng(0),
        thresholds=[0.0, 15.0, 30.0],
        lead_times=[10, 20],
        n_boot=50,
    )
    assert _q(cur, "caught_lead_ge_10_pct", 15.0) == 100.0
    assert _q(cur, "caught_lead_ge_20_pct", 15.0) == 0.0
    assert _q(cur, "mean_wasted_life", 30.0) == 30.0
    assert _q(cur, "failed_in_service_pct", 0.0) == 0.0  # perfect model triggers at RUL 0
    late = decision_curve(
        _traj(bias=50.0), np.random.default_rng(0), thresholds=[20.0], lead_times=[10], n_boot=20
    )
    assert _q(late, "failed_in_service_pct", 20.0) == 100.0


def test_trajectories_of_one_engine_resampled_together() -> None:
    cur = decision_curve(
        _traj(n_units=8, repeats=3),
        np.random.default_rng(0),
        thresholds=[15.0],
        lead_times=[10],
        n_boot=30,
    )
    row = cur[cur["quantity"] == "caught_lead_ge_10_pct"].iloc[0]
    assert row["n_engines"] == 8 and row["n_trajectories"] == 24
    assert row["ci_lo"] == row["ci_hi"] == pytest.approx(100.0)


def test_caught_at_budget_interpolates_and_flags_unreachable() -> None:
    wasted = np.array([np.nan, 10.0, 20.0, 30.0])
    caught = np.array([0.0, 0.0, 100.0, 100.0])
    assert caught_at_budget(caught, wasted, 15.0) == 50.0
    assert caught_at_budget(caught, wasted, 30.0) == 100.0
    assert np.isnan(caught_at_budget(caught, wasted, 5.0))  # below the curve
    assert np.isnan(caught_at_budget(caught, wasted, 31.0))  # above the curve
    assert np.isnan(caught_at_budget(caught, np.full(4, np.nan), 15.0))


def test_caught_at_budget_uses_running_max_of_non_monotone_wasted() -> None:
    wasted = np.array([10.0, 30.0, 20.0, 40.0])  # dips at T index 2
    caught = np.array([0.0, 50.0, 60.0, 100.0])
    # running max 10, 30, 30, 40: budget 35 lies between the last two points
    assert caught_at_budget(caught, wasted, 35.0) == pytest.approx(80.0)


def test_matched_budget_perfect_model() -> None:
    t = matched_budget(
        _traj(n_units=8),
        np.random.default_rng(0),
        budgets=[15.0, 25.0, 500.0],
        lead_time=20,
        thresholds=[0.0, 10.0, 20.0, 30.0],
        n_boot=40,
    )
    est = t.set_index("budget")
    assert est.loc[15.0, "estimate"] == 50.0  # linear between T=10 (0%) and T=20 (100%)
    assert est.loc[25.0, "estimate"] == 100.0
    assert np.isnan(est.loc[500.0, "estimate"]) and est.loc[500.0, "boot_defined_share"] == 0.0
    assert (t["n_engines"] == 8).all()


def test_matched_budget_requires_increasing_thresholds() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        matched_budget(_traj(), np.random.default_rng(0), thresholds=[10.0, 5.0])
