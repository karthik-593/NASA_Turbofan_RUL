"""Tests for src/turbofan/features/engineering.py.

All tests use synthetic DataFrames so the real C-MAPSS data is not required.
The synthetic data is structurally identical to real C-MAPSS files (notebook-style
column names: unit, cycle, op1-3, s1-s21).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from turbofan.config import FEAT_COLS, KEEP
from turbofan.features.engineering import add_features


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

    def test_returns_stats_dict_with_multi_key(self) -> None:
        _, stats = add_features(_make_nb_df(), KEEP, "FD001")
        assert isinstance(stats, dict) and "multi" in stats

    def test_single_regime_multi_is_false(self) -> None:
        _, stats = add_features(_make_nb_df(), KEEP, "FD001")
        assert stats["multi"] is False

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
        assert stats["multi"] is True
        assert not out[FEAT_COLS].isnull().any().any()
