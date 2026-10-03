"""Input validation: raw-file schema at load, operating envelope at serving."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbofan.config import ENVELOPE_MARGIN, KEEP
from turbofan.data.schema import validate_raw, validate_rul
from turbofan.features.engineering import add_features
from turbofan.features.envelope import check_envelope

_SENSORS = [f"sensor_{i}" for i in range(1, 22)]


def _raw(n_units: int = 2, n_cycles: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for u in range(1, n_units + 1):
        for c in range(1, n_cycles + 1):
            row = {"unit_id": u, "cycle": c, "op_setting_1": 0.0, "op_setting_2": 0.0}
            row["op_setting_3"] = 100.0
            row.update({s: float(rng.normal()) for s in _SENSORS})
            rows.append(row)
    return pd.DataFrame(rows)


class TestRawSchema:
    def test_valid_frame_passes(self) -> None:
        validate_raw(_raw(), "synthetic")

    def test_int_valued_sensor_column_passes(self) -> None:
        df = _raw()
        df["sensor_17"] = 392  # integer-valued sensors parse as int64 in the real files
        validate_raw(df, "synthetic")

    def test_nan_raises(self) -> None:
        df = _raw()
        df.loc[3, "sensor_4"] = np.nan
        with pytest.raises(ValueError, match="sensor_4"):
            validate_raw(df, "synthetic")

    def test_non_numeric_raises(self) -> None:
        df = _raw()
        df["sensor_2"] = "x"
        with pytest.raises(ValueError, match="sensor_2"):
            validate_raw(df, "synthetic")

    def test_duplicate_cycle_raises(self) -> None:
        df = pd.concat([_raw(), _raw().iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="failed schema validation"):
            validate_raw(df, "synthetic")

    def test_gap_in_cycles_raises(self) -> None:
        df = _raw()
        df = df[~((df["unit_id"] == 1) & (df["cycle"] == 3))]
        with pytest.raises(ValueError, match="1..n"):
            validate_raw(df, "synthetic")

    def test_unit_zero_raises(self) -> None:
        df = _raw()
        df.loc[df["unit_id"] == 1, "unit_id"] = 0
        with pytest.raises(ValueError, match="unit_id"):
            validate_raw(df, "synthetic")

    def test_missing_column_raises(self) -> None:
        with pytest.raises(ValueError, match="missing columns"):
            validate_raw(_raw().drop(columns=["sensor_9"]), "synthetic")


class TestRulSchema:
    def test_valid(self) -> None:
        validate_rul(pd.Series([3, 0, 120], name="rul"), 3, "rul")

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="failed schema validation"):
            validate_rul(pd.Series([3, -1], name="rul"), 2, "rul")

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="2 RUL values for 3 test units"):
            validate_rul(pd.Series([3, 4], name="rul"), 3, "rul")


def _nb_df(n_units: int = 4, n_cycles: int = 40, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        for c in range(1, n_cycles + 1):
            row = {"unit": u, "cycle": c, "op1": rng.normal(0, 0.002), "op2": 0.0, "op3": 100.0}
            row.update({s: float(rng.normal()) for s in KEEP})
            rows.append(row)
    return pd.DataFrame(rows)


class TestEnvelope:
    @pytest.fixture(scope="class")
    def stats(self) -> dict:
        return add_features(_nb_df(), KEEP, "FD001")[1]

    def test_training_rows_are_inside(self, stats: dict) -> None:
        assert check_envelope(_nb_df(), stats, margin=0.0) == []

    def test_sensor_beyond_range_is_reported(self, stats: dict) -> None:
        df = _nb_df(n_units=1, n_cycles=3, seed=1)
        df.loc[1, "s7"] = 50.0
        reasons = check_envelope(df, stats, margin=ENVELOPE_MARGIN)
        assert len(reasons) == 1
        assert reasons[0].startswith("cycle 2: s7 = 50 outside training range")

    def test_margin_widens_the_range(self, stats: dict) -> None:
        hi = stats["envelope"]["sensor_max"]["s7"][0]
        lo = stats["envelope"]["sensor_min"]["s7"][0]
        df = _nb_df(n_units=1, n_cycles=1, seed=1)
        df.loc[0, "s7"] = hi + 0.1 * (hi - lo)
        assert check_envelope(df, stats, margin=0.0)
        assert not check_envelope(df, stats, margin=0.25)

    def test_unseen_operating_regime_is_reported(self, stats: dict) -> None:
        df = _nb_df(n_units=1, n_cycles=2, seed=1)
        df.loc[0, ["op1", "op2", "op3"]] = [35.0, 0.84, 60.0]  # an FD002-style regime
        reasons = check_envelope(df, stats, margin=ENVELOPE_MARGIN)
        assert any("outside every training operating regime" in r for r in reasons)


@pytest.mark.requires_data
class TestEnvelopeOnRealData:
    """Evidence for D41, computed on train/validation only (rule 3: no test rows)."""

    @pytest.mark.parametrize("dataset", ["FD001", "FD002", "FD003", "FD004"])
    def test_configured_margin_accepts_every_validation_row(self, dataset: str) -> None:
        from turbofan.data.loader import load_dataset
        from turbofan.evaluation.comparison import split_engines

        tr, _te, _ = load_dataset(dataset, "data/raw")
        e_tr, e_va = split_engines(tr)
        _, stats = add_features(tr[tr["unit"].isin(e_tr)], KEEP, dataset)
        va = tr[tr["unit"].isin(e_va)]
        rejected = {u: check_envelope(g, stats, ENVELOPE_MARGIN) for u, g in va.groupby("unit")}
        assert not any(rejected.values()), {u: r[:2] for u, r in rejected.items() if r}

    def test_multi_regime_data_is_rejected_by_single_regime_model(self) -> None:
        from turbofan.data.loader import load_dataset

        tr1, _, _ = load_dataset("FD001", "data/raw")
        tr2, _, _ = load_dataset("FD002", "data/raw")
        _, stats = add_features(tr1, KEEP, "FD001")
        reasons = check_envelope(tr2[tr2["unit"] == 1], stats, ENVELOPE_MARGIN)
        assert sum("operating regime" in r for r in reasons) > 0


def test_serving_app_does_not_import_pandera() -> None:
    """The Docker image installs requirements.lock, which has no pandera: the load-time
    schema must stay out of the serving import graph."""
    import subprocess
    import sys

    code = "import sys, turbofan.serving.app; print('pandera' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"
