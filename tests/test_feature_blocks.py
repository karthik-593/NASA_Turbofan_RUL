"""Optional feature blocks of Stage A3 (D51): health index, regime one-hot, extra sensors."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from turbofan.config import HI_COLS, KEEP, FeatureBlocks
from turbofan.features.engineering import FeatureError, add_features

CONSISTENT = FeatureBlocks(health_index="consistent")
POOLED = FeatureBlocks(health_index="pooled")


def _engines(n_units: int = 8, seed: int = 0, flip: frozenset[str] = frozenset()) -> pd.DataFrame:
    """Every KEEP sensor drifts with age; sensors in ``flip`` drift up in half the engines and
    down in the other half (direction-inconsistent)."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        life = 60 + 5 * u
        for c in range(1, life + 1):
            frac = c / life
            row: dict[str, float] = {"unit": u, "cycle": c, "rul_true": life - c}
            for i in range(1, 4):
                row[f"op{i}"] = float(rng.uniform(0, 1))
            for i in range(1, 22):
                s = f"s{i}"
                direction = (-1.0 if u % 2 else 1.0) if s in flip else 1.0
                row[s] = float(direction * 3 * frac + rng.normal(0, 0.3))
            rows.append(row)
    return pd.DataFrame(rows)


class TestHealthIndex:
    def test_consistent_drops_direction_mixed_sensors(self) -> None:
        _, stats = add_features(
            _engines(flip=frozenset({"s9", "s14"})), KEEP, "FD001", blocks=CONSISTENT
        )
        chosen = stats["hi"]["sensors"]
        assert "s9" not in chosen and "s14" not in chosen
        assert set(chosen) == set(KEEP) - {"s9", "s14"}

    def test_pooled_keeps_every_base_sensor(self) -> None:
        _, stats = add_features(_engines(flip=frozenset({"s9"})), KEEP, "FD001", blocks=POOLED)
        assert stats["hi"]["sensors"] == KEEP

    @pytest.mark.parametrize("blocks", [CONSISTENT, POOLED])
    def test_oriented_to_rise_toward_failure(self, blocks: FeatureBlocks) -> None:
        out, _ = add_features(_engines(), KEEP, "FD001", blocks=blocks)
        assert np.corrcoef(out["hi"], out["rul_true"])[0, 1] < -0.5
        assert not out[HI_COLS].isna().any().any()
        first = ~out["unit"].duplicated()
        assert (out.loc[first, "hi_slope"] == 0).all()

    def test_fitted_on_training_rows_and_reused_unchanged(self) -> None:
        train, held = _engines(seed=0), _engines(n_units=3, seed=1)
        _, stats = add_features(train, KEEP, "FD001", blocks=CONSISTENT)
        before = dict(stats["hi"])
        out, _ = add_features(
            held.drop(columns="rul_true"), KEEP, "FD001", stats=stats, blocks=CONSISTENT
        )
        assert stats["hi"] == before  # no refit on held-out rows, no truth needed there
        z = out[[f"{s}_n" for s in before["sensors"]]].to_numpy(float)
        expected = (z - np.asarray(before["mean"])) @ np.asarray(before["component"])
        np.testing.assert_allclose(out["hi"].to_numpy(), expected)

    def test_fit_needs_the_uncapped_truth(self) -> None:
        with pytest.raises(FeatureError, match="rul_true"):
            add_features(_engines().drop(columns="rul_true"), KEEP, "FD001", blocks=POOLED)

    def test_state_and_request_must_agree(self) -> None:
        _, plain = add_features(_engines(), KEEP, "FD001")
        with pytest.raises(FeatureError, match="lacks the fitted health index"):
            add_features(_engines(), KEEP, "FD001", stats=plain, blocks=POOLED)
        _, with_hi = add_features(_engines(), KEEP, "FD001", blocks=POOLED)
        with pytest.raises(FeatureError, match="none was requested"):
            add_features(_engines(), KEEP, "FD001", stats=with_hi)
        with pytest.raises(FeatureError, match="asked 'consistent'"):
            add_features(_engines(), KEEP, "FD001", stats=with_hi, blocks=CONSISTENT)

    def test_no_consistent_sensor_raises(self) -> None:
        mixed = _engines(flip=frozenset(KEEP))
        with pytest.raises(FeatureError, match="no direction-consistent sensor"):
            add_features(mixed, KEEP, "FD001", blocks=CONSISTENT)


class TestRegimeOnehot:
    def test_one_indicator_per_regime_summing_to_one(self) -> None:
        out, stats = add_features(
            _engines(), KEEP, "FD002", blocks=FeatureBlocks(regime_onehot=True)
        )
        cols = [f"regime_{r}" for r in range(stats["n_regimes"])]
        assert stats["n_regimes"] == 6
        assert (out[cols].sum(axis=1) == 1).all() and set(np.unique(out[cols])) <= {0.0, 1.0}


class TestBaseUnchanged:
    def test_default_blocks_add_no_column(self) -> None:
        out_default, _ = add_features(_engines(), KEEP, "FD001")
        out_explicit, _ = add_features(_engines(), KEEP, "FD001", blocks=FeatureBlocks())
        pd.testing.assert_frame_equal(out_default, out_explicit)
        assert not ({"hi", "hi_mean", "hi_slope"} & set(out_default.columns))
        assert not any(c.startswith("regime_") for c in out_default.columns)


def _config_under(tmp_path: Path, features: dict[str, object], dataset: str = "FD002") -> str:
    """Import turbofan.config in a fresh process under params with these features."""
    src = Path(__file__).resolve().parents[1] / "params.yaml"
    doc = yaml.safe_load(src.read_text(encoding="utf-8"))
    doc["features"].update(features)
    doc["pipeline"]["dataset"] = dataset
    p = tmp_path / "params.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    code = (
        "import turbofan.config as c; "
        "print(len(c.FEAT_COLS), len(c.SENSOR_N_COLS), c.MODEL_SENSORS[-1], "
        "c.FEATURE_BLOCKS.active)"
    )
    res = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**__import__("os").environ, "TURBOFAN_PARAMS": str(p)},
        check=False,
    )
    return (
        res.stdout.strip() if res.returncode == 0 else "ERR " + res.stderr.strip().splitlines()[-1]
    )


class TestConfig:
    def test_columns_follow_the_blocks(self, tmp_path: Path) -> None:
        # 14 base sensors + s6 = 15 sensors x 3 = 45, + 3 HI columns, + 6 regime indicators
        got = _config_under(
            tmp_path,
            {"extra_sensors": ["s6"], "health_index": "pooled", "regime_onehot": True},
        )
        assert got == "54 21 s6 True"

    def test_extra_sensor_must_come_from_the_datasets_pool(self, tmp_path: Path) -> None:
        assert "candidate_pool" in _config_under(tmp_path, {"extra_sensors": ["s10"]}, "FD001")
        assert "candidate_pool" in _config_under(tmp_path, {"extra_sensors": ["s2"]})

    def test_regime_onehot_needs_regimes(self, tmp_path: Path) -> None:
        assert "has 1" in _config_under(tmp_path, {"regime_onehot": True}, "FD001")

    def test_unknown_health_index_mode(self, tmp_path: Path) -> None:
        assert "health_index" in _config_under(tmp_path, {"health_index": "pc2"})


def test_bundles_refuse_feature_blocks(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import turbofan.training.bundle as bundle

    monkeypatch.setattr(bundle.cfg, "FEATURE_BLOCKS", FeatureBlocks(regime_onehot=True))
    with pytest.raises(NotImplementedError, match="feature blocks"):
        bundle.save_bundle(tmp_path, "FD002", "xgboost", "v", object(), {}, seed=1, metrics={})
