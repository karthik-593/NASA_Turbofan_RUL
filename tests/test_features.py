"""Tests for src/turbofan/features/engineering.py.

All tests use synthetic DataFrames so the real C-MAPSS data is not required.
The synthetic data is structurally identical to real C-MAPSS files (notebook-style
column names: unit, cycle, op1-3, s1-s21).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbofan.config import FEAT_COLS, KEEP, N_REGIMES
from turbofan.features.engineering import (
    FEATURE_STATE_KEYS,
    FeatureError,
    _trailing_slope,
    add_features,
)


def _make_nb_df(n_units: int = 3, cycles_per_unit: int = 40, seed: int = 0) -> pd.DataFrame:
    """Synthetic C-MAPSS DataFrame with notebook-style column names (unit, sN, opN)."""
    rng = np.random.default_rng(seed)
    rows = []
    for uid in range(1, n_units + 1):
        for cyc in range(1, cycles_per_unit + 1):
            row: dict = {"unit": uid, "cycle": cyc}
            for i in range(1, 4):
                row[f"op{i}"] = float(rng.uniform(0.0, 1.0))
            for i in range(1, 22):
                row[f"s{i}"] = float(rng.normal(0.0, 1.0) + cyc * 0.01)
            rows.append(row)
    return pd.DataFrame(rows)


class TestAddFeatures:
    def test_output_includes_feat_cols(self) -> None:
        out, _ = add_features(_make_nb_df(), KEEP, "FD001")
        for col in FEAT_COLS:
            assert col in out.columns, f"Missing column: {col}"

    def test_no_nans_in_feat_cols(self) -> None:
        out, _ = add_features(_make_nb_df(), KEEP, "FD001")
        assert not out[FEAT_COLS].isnull().any().any()

    def test_returns_complete_feature_state(self) -> None:
        _, stats = add_features(_make_nb_df(), KEEP, "FD001")
        assert set(FEATURE_STATE_KEYS) <= set(stats)

    def test_single_regime_dataset_has_one_regime(self) -> None:
        _, stats = add_features(_make_nb_df(), KEEP, "FD001")
        assert stats["n_regimes"] == 1 == N_REGIMES["FD001"]
        assert all(len(v) == 1 for v in stats["s_mean"].values())

    def test_stats_reuse_on_held_out_frame(self) -> None:
        train = _make_nb_df(n_units=4, cycles_per_unit=40, seed=0)
        held = _make_nb_df(n_units=2, cycles_per_unit=20, seed=1)
        _, stats = add_features(train, KEEP, "FD001")
        out, _ = add_features(held, KEEP, "FD001", stats=stats)
        assert not out[FEAT_COLS].isnull().any().any()
        for col in FEAT_COLS:
            assert col in out.columns

    def test_deterministic(self) -> None:
        df = _make_nb_df()
        out1, _ = add_features(df, KEEP, "FD001")
        out2, _ = add_features(df, KEEP, "FD001")
        pd.testing.assert_frame_equal(
            out1[FEAT_COLS].reset_index(drop=True), out2[FEAT_COLS].reset_index(drop=True)
        )

    def test_no_future_leakage(self) -> None:
        """Features at cycle t must not depend on cycle t+1."""
        train = _make_nb_df(n_units=4, cycles_per_unit=50, seed=99)
        _, stats = add_features(train, KEEP, "FD001")

        engine_df = _make_nb_df(n_units=1, cycles_per_unit=30, seed=7)
        feat_orig, _ = add_features(engine_df, KEEP, "FD001", stats=stats)

        engine_mod = engine_df.copy()
        last_cycle = engine_mod["cycle"].max()
        last_mask = engine_mod["cycle"] == last_cycle
        for s in KEEP:
            engine_mod.loc[last_mask, s] = 999.0
        feat_mod, _ = add_features(engine_mod, KEEP, "FD001", stats=stats)

        penult = last_cycle - 1
        orig_row = feat_orig.loc[engine_df["cycle"] == penult, FEAT_COLS].reset_index(drop=True)
        mod_row = feat_mod.loc[engine_mod["cycle"] == penult, FEAT_COLS].reset_index(drop=True)
        pd.testing.assert_frame_equal(orig_row, mod_row)

    def test_multi_regime_fd002_no_nans(self) -> None:
        """add_features on FD002-style data uses per-regime normalization without error."""
        rng = np.random.default_rng(42)
        centroids = [
            (0, 0, 100),
            (10, 0.25, 100),
            (20, 0.7, 100),
            (25, 0.62, 60),
            (35, 0.84, 100),
            (42, 0.84, 100),
        ]
        rows = []
        for uid in range(1, 5):
            for cyc in range(1, 41):
                c = centroids[cyc % 6]
                row: dict = {
                    "unit": uid,
                    "cycle": cyc,
                    "op1": float(c[0]) + rng.uniform(-0.01, 0.01),
                    "op2": float(c[1]) + rng.uniform(-0.001, 0.001),
                    "op3": float(c[2]),
                }
                for s in KEEP:
                    row[s] = float(rng.normal(0, 1))
                rows.append(row)
        df = pd.DataFrame(rows)
        out, stats = add_features(df, KEEP, "FD002")
        assert stats["n_regimes"] == N_REGIMES["FD002"] == 6
        assert not out[FEAT_COLS].isnull().any().any()

    def test_k1_regime_is_global_zscore(self) -> None:
        """k = 1 through the regime path equals a plain global z-score, bit for bit."""
        df = _make_nb_df(n_units=4, seed=3)
        out, _ = add_features(df, KEEP, "FD001")
        d = df.sort_values(["unit", "cycle"])
        for s in KEEP:
            expected = (d[s].to_numpy() - d[s].mean()) / d[s].std()
            assert np.array_equal(out[f"{s}_n"].to_numpy(), expected), s

    def test_stats_from_older_bundle_raise(self) -> None:
        legacy_state = {"multi": False, "s_mean": {}, "s_std": {}}
        with pytest.raises(ValueError, match="predates the params.yaml"):
            add_features(_make_nb_df(), KEEP, "FD001", stats=legacy_state)

    def test_stats_sensor_mismatch_raises(self) -> None:
        _, stats = add_features(_make_nb_df(), KEEP, "FD001")
        with pytest.raises(ValueError, match="fitted on sensors"):
            add_features(_make_nb_df(), KEEP[:-1], "FD001", stats=stats)


class TestNoSilentFallbacks:
    """Rule 8 / D06: every former fillna/floor either raises or is an explicit definition."""

    def test_first_cycle_slope_is_explicit_zero(self) -> None:
        out, _ = add_features(_make_nb_df(), KEEP, "FD001")
        first = out.groupby("unit").head(1)
        assert (first[[f"{s}_slope" for s in KEEP]] == 0.0).all().all()
        later = out.groupby("unit").nth(1)
        assert (later[[f"{s}_slope" for s in KEEP]] != 0.0).any().any()

    def test_single_cycle_engine_is_valid(self) -> None:
        _, stats = add_features(_make_nb_df(), KEEP, "FD001")
        one = _make_nb_df(n_units=1, cycles_per_unit=1, seed=5)
        out, _ = add_features(one, KEEP, "FD001", stats=stats)
        assert not out[FEAT_COLS].isna().any().any()

    def test_nan_sensor_raises(self) -> None:
        df = _make_nb_df()
        df.loc[5, "s2"] = np.nan
        with pytest.raises(FeatureError, match="NaN in input columns.*s2"):
            add_features(df, KEEP, "FD001")

    def test_nan_op_setting_raises_at_serving(self) -> None:
        _, stats = add_features(_make_nb_df(), KEEP, "FD001")
        df = _make_nb_df(n_units=1, seed=2)
        df.loc[0, "op1"] = np.nan
        with pytest.raises(FeatureError, match="op1"):
            add_features(df, KEEP, "FD001", stats=stats)

    def test_missing_column_raises(self) -> None:
        with pytest.raises(FeatureError, match="lacks columns"):
            add_features(_make_nb_df().drop(columns=["s7"]), KEEP, "FD001")

    def test_constant_sensor_in_regime_raises(self) -> None:
        df = _make_nb_df()
        df["s3"] = 1.0
        with pytest.raises(FeatureError, match="s3 is constant within regime 0"):
            add_features(df, KEEP, "FD001")

    def test_regime_with_too_few_rows_raises(self) -> None:
        df = _make_nb_df(n_units=1, cycles_per_unit=8)
        df[["op1", "op2", "op3"]] = 0.0
        df.loc[0, ["op1", "op2", "op3"]] = 100.0  # one outlier row forms its own cluster
        with pytest.raises(FeatureError, match="regime .* has 1 training row"):
            add_features(df, KEEP, "FD002")

    def test_inconsistent_state_raises(self) -> None:
        _, stats = add_features(_make_nb_df(), KEEP, "FD001")
        stats["s_mean"]["s2"] = [0.0, 0.0]
        with pytest.raises(FeatureError, match="inconsistent with k = 1"):
            add_features(_make_nb_df(seed=1), KEEP, "FD001", stats=stats)


class TestClosedFormSlope:
    """D52: the rolling slope is the least-squares slope np.polyfit gave, without the per-window
    Python call."""

    @pytest.mark.parametrize("window", [2, 5, 20, 45])
    def test_matches_polyfit_on_engines_shorter_and_longer_than_the_window(
        self, window: int
    ) -> None:
        rng = np.random.default_rng(window)
        lengths = [1, 3, window - 1, window, window + 7, 3 * window]
        unit = np.repeat(np.arange(len(lengths)), [max(n, 1) for n in lengths])
        y = rng.normal(size=(len(unit), 3)).cumsum(axis=0)
        pos = pd.Series(unit).groupby(unit).cumcount().to_numpy()
        got = _trailing_slope(y, pos, window)
        for u in np.unique(unit):
            rows = np.flatnonzero(unit == u)
            for j, r in enumerate(rows):
                w = y[rows[max(0, j - window + 1) : j + 1]]
                if len(w) < 2:
                    assert np.isnan(got[r]).all()
                else:
                    ref = np.polyfit(np.arange(len(w)), w, 1)[0]
                    np.testing.assert_allclose(got[r], ref, rtol=0, atol=1e-12)

    def test_window_below_two_raises(self) -> None:
        with pytest.raises(FeatureError, match="window >= 2"):
            _trailing_slope(np.zeros((4, 1)), np.arange(4), 1)
