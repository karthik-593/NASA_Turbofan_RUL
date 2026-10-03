"""Protocol-v2 views and metrics: definitions, capped/uncapped truth, engine bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbofan.evaluation.cv_metrics import (
    HEADLINE,
    engine_aggregates,
    engine_weights,
    metric_table,
    metrics_from_weights,
)
from turbofan.evaluation.views import benchmark_view, deployment_view

CAP = 125.0


def _points(life: int = 200, n_units: int = 4, bias: float = 0.0) -> pd.DataFrame:
    rows = []
    for u in range(1, n_units + 1):
        for c in range(1, life + 1):
            t = life - c
            rows.append({"unit": u, "cycle": c, "rul_true": t, "pred": min(t, CAP) + bias})
    return pd.DataFrame(rows)


def _metric(tab: pd.DataFrame, truth: str, m: str) -> pd.Series:
    return tab[(tab["truth"] == truth) & (tab["metric"] == m)].iloc[0]


class TestMetricDefinitions:
    def test_perfect_capped_model(self) -> None:
        """Predicts min(truth, cap) exactly: zero error against capped truth, error only
        above the cap against uncapped truth."""
        tab = metric_table(_points(), CAP, np.random.default_rng(0), n_boot=50)
        assert _metric(tab, "capped", "rmse")["estimate"] == 0.0
        assert _metric(tab, "uncapped", "rmse")["estimate"] > 0.0
        for truth in ("capped", "uncapped"):
            assert _metric(tab, truth, HEADLINE)["estimate"] == 0.0

    def test_bias_shows_in_signed_error_and_late(self) -> None:
        tab = metric_table(_points(bias=3.0), CAP, np.random.default_rng(0), n_boot=50)
        row = _metric(tab, "capped", "mean_signed_error")
        assert row["estimate"] == pytest.approx(3.0)
        assert _metric(tab, "capped", "late_pct")["estimate"] == 100.0
        assert _metric(tab, "capped", "critical_rmse")["estimate"] == pytest.approx(3.0)
        assert _metric(tab, "capped", "mae")["estimate"] == pytest.approx(3.0)

    def test_bucket_counts(self) -> None:
        tab = metric_table(_points(life=200, n_units=4), CAP, np.random.default_rng(0), n_boot=20)
        crit = _metric(tab, "uncapped", "critical_rmse")
        assert crit["n_points"] == 4 * 25 and crit["n_engines"] == 4
        healthy_unc = _metric(tab, "uncapped", "healthy_rmse")
        healthy_cap = _metric(tab, "capped", "healthy_rmse")
        assert healthy_unc["n_points"] == healthy_cap["n_points"] == 4 * 100

    def test_nasa_is_mean_per_engine(self) -> None:
        """A long and a short engine with the same per-point score weigh the same."""
        a = pd.DataFrame({"unit": 1, "rul_true": np.arange(100.0), "pred": np.arange(100.0) + 10})
        b = pd.DataFrame({"unit": 2, "rul_true": np.arange(10.0), "pred": np.arange(10.0) + 10})
        agg = engine_aggregates(pd.concat([a, b]), "uncapped", CAP)
        v = metrics_from_weights(agg, np.ones((1, 2)))["nasa_mean_per_engine"][0]
        assert v == pytest.approx(np.expm1(1.0))

    def test_nan_prediction_raises(self) -> None:
        p = _points()
        p.loc[3, "pred"] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            metric_table(p, CAP, np.random.default_rng(0), n_boot=10)


class TestEngineBootstrap:
    def test_weights_resample_engines(self) -> None:
        w = engine_weights(7, 500, np.random.default_rng(0))
        assert w.shape == (500, 7) and (w.sum(axis=1) == 7).all()

    def test_resamples_engines_not_rows(self) -> None:
        """Engine A: 1000 perfect points; engine B: one point 10 cycles off. RMSE is 0 in a
        replicate exactly when B's residual is absent. Resampling engines: B is absent when A
        is drawn twice, 1/4 of replicates. Resampling the 1001 rows would leave B's point out
        in (1000/1001)^1001 ~ 37% of replicates instead."""
        a = pd.DataFrame({"unit": 1, "rul_true": np.full(1000, 50.0), "pred": 50.0})
        b = pd.DataFrame({"unit": 2, "rul_true": [50.0], "pred": [60.0]})
        agg = engine_aggregates(pd.concat([a, b]), "uncapped", CAP)
        rmse = metrics_from_weights(agg, engine_weights(2, 4000, np.random.default_rng(1)))["rmse"]
        share_zero = np.mean(rmse == 0.0)
        assert share_zero == pytest.approx(0.25, abs=0.03)
        assert abs(share_zero - (1000 / 1001) ** 1001) > 0.08  # not the row-bootstrap rate

    def test_ci_brackets_estimate_and_is_reproducible(self) -> None:
        rng_pts = np.random.default_rng(5)
        p = _points(n_units=20)
        p["pred"] = p["pred"] + rng_pts.normal(0, 5, len(p))
        t1 = metric_table(p, CAP, np.random.default_rng(3), n_boot=200)
        t2 = metric_table(p, CAP, np.random.default_rng(3), n_boot=200)
        pd.testing.assert_frame_equal(t1, t2)
        r = _metric(t1, "uncapped", "rmse")
        assert r["ci_lo"] <= r["estimate"] <= r["ci_hi"]


class TestViews:
    def test_deployment_view_starts_at_min_history(self) -> None:
        p = _points(life=50, n_units=2)
        d = deployment_view(p, min_history=10)
        assert d["cycle"].min() == 10 and len(d) == 2 * 41

    def test_benchmark_view_matches_label_share(self) -> None:
        p = _points(life=300, n_units=40)
        share = pd.Series({0: 0.2, 10: 0.3, 150: 0.5})  # bins of width 10 by lower edge
        pts, short = benchmark_view(
            p, share, np.random.default_rng(0), points_per_engine=5, bin_width=10, min_history=10
        )
        assert short == {}
        assert len(pts) == 40 * 5
        got = ((pts["rul_true"] // 10) * 10).value_counts(normalize=True).sort_index()
        assert got.to_dict() == pytest.approx({0: 0.2, 10: 0.3, 150: 0.5})

    def test_benchmark_view_reports_unreachable_bins(self) -> None:
        p = _points(life=60, n_units=4)  # no point with true RUL >= 60
        share = pd.Series({0: 0.5, 200: 0.5})
        pts, short = benchmark_view(
            p, share, np.random.default_rng(0), points_per_engine=4, bin_width=10, min_history=1
        )
        assert short == {200: 8} and len(pts) == 8

    def test_benchmark_view_includes_rul_above_cap(self) -> None:
        p = _points(life=300, n_units=10)
        share = pd.Series({130: 1.0})
        pts, _ = benchmark_view(
            p, share, np.random.default_rng(0), points_per_engine=2, bin_width=10, min_history=1
        )
        assert (pts["rul_true"] > CAP).all()


@pytest.mark.requires_data
def test_label_histogram_reads_only_the_rul_file(monkeypatch: pytest.MonkeyPatch) -> None:
    import turbofan.evaluation.views as views

    opened: list[str] = []
    real = views.pd.read_csv
    monkeypatch.setattr(
        views.pd, "read_csv", lambda p, *a, **k: opened.append(str(p)) or real(p, *a, **k)
    )
    share = views.nasa_test_label_histogram("FD001", "data/raw")
    assert len(opened) == 1 and opened[0].endswith("RUL_FD001.txt")
    assert share.sum() == pytest.approx(1.0)
