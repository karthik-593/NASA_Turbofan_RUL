"""Protocol-v2 folds: engine-disjoint, stratified, state fitted on fold-train only, and the
test set never opened."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbofan.config import KEEP
from turbofan.evaluation.cv import make_folds, prepare_fold, true_rul


def _labels(n: int = 50, n_minor: int = 20) -> pd.Series:
    lab = np.zeros(n, dtype=int)
    lab[np.random.default_rng(0).choice(n, n_minor, replace=False)] = 1
    return pd.Series(lab, index=np.arange(1, n + 1), name="subpopulation")


def _raw(n_units: int = 30, n_cycles: int = 35, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_units + 1):
        for c in range(1, n_cycles + u % 7 + 1):
            row = {"unit": u, "cycle": c, "op1": rng.normal(0, 0.002), "op2": 0.0, "op3": 100.0}
            row.update({s: float(rng.normal() + 0.05 * c) for s in KEEP})
            rows.append(row)
    return pd.DataFrame(rows)


class TestMakeFolds:
    def test_no_engine_on_two_sides(self) -> None:
        for f in make_folds(_labels(), n_folds=5, n_repeats=3, seed=0):
            sides = [set(f.train_units), set(f.val_units), set(f.test_units)]
            assert not (sides[0] & sides[1]) and not (sides[0] & sides[2])
            assert not (sides[1] & sides[2])
            assert set().union(*sides) == set(_labels().index)

    def test_each_engine_held_out_once_per_repeat(self) -> None:
        folds = make_folds(_labels(), n_folds=5, n_repeats=3, seed=0)
        for r in range(3):
            held = np.concatenate([f.test_units for f in folds if f.repeat == r])
            assert sorted(held) == list(_labels().index)

    def test_stratification_balanced(self) -> None:
        """Each held-out fold's minority-group count is within one engine of n_minor / K."""
        lab = _labels(n=50, n_minor=20)
        for f in make_folds(lab, n_folds=5, n_repeats=3, seed=0):
            n1 = int(lab.loc[f.test_units].sum())
            assert abs(n1 - 20 / 5) <= 1, (f.repeat, f.fold, n1)
            frac_val = lab.loc[f.val_units].mean()
            assert abs(frac_val - 0.4) <= 0.15

    def test_deterministic_and_repeats_differ(self) -> None:
        a = make_folds(_labels(), seed=0)
        b = make_folds(_labels(), seed=0)
        assert all(np.array_equal(x.test_units, y.test_units) for x, y in zip(a, b, strict=True))
        r0 = [tuple(f.test_units) for f in a if f.repeat == 0]
        r1 = [tuple(f.test_units) for f in a if f.repeat == 1]
        assert r0 != r1

    def test_configured_shape(self) -> None:
        from turbofan.config import PARAMS

        folds = make_folds(_labels())
        assert len(folds) == PARAMS["cv"]["n_folds"] * PARAMS["cv"]["n_repeats"]


class TestPrepareFold:
    def test_state_fitted_on_fold_train_only(self) -> None:
        raw = _raw()
        lab = pd.Series(np.arange(30) % 2, index=np.arange(1, 31))
        fold = make_folds(lab, n_folds=5, n_repeats=1, seed=0)[0]
        fd = prepare_fold(raw, "FD001", fold, rul_cap=125.0)
        own = raw[raw["unit"].isin(fold.train_units)]
        for s in KEEP:
            assert fd.stats["s_mean"][s] == [float(own.sort_values(["unit", "cycle"])[s].mean())]
        # perturbing held-out and inner-validation engines leaves the fitted state unchanged
        moved = raw.copy()
        other = moved["unit"].isin(np.concatenate([fold.test_units, fold.val_units]))
        moved.loc[other, KEEP] += 50.0
        fd2 = prepare_fold(moved, "FD001", fold, rul_cap=125.0)
        assert fd2.stats["s_mean"] == fd.stats["s_mean"]
        assert fd2.stats["s_std"] == fd.stats["s_std"]

    def test_frames_hold_their_own_engines_and_both_labels(self) -> None:
        raw = _raw()
        lab = pd.Series(np.arange(30) % 2, index=np.arange(1, 31))
        fold = make_folds(lab, n_folds=5, n_repeats=1, seed=0)[1]
        fd = prepare_fold(raw, "FD001", fold, rul_cap=20.0)
        assert set(fd.feat_tr["unit"]) == set(fold.train_units)
        assert set(fd.feat_va["unit"]) == set(fold.val_units)
        assert set(fd.feat_te["unit"]) == set(fold.test_units)
        assert fd.feat_te["rul"].max() == 20.0
        assert fd.feat_te["rul_true"].max() > 20.0  # uncapped truth kept for evaluation

    def test_true_rul_is_uncapped_cycles_to_failure(self) -> None:
        raw = _raw(n_units=2)
        t = true_rul(raw)
        last = raw.groupby("unit")["cycle"].transform("max")
        assert (t == last - raw["cycle"]).all()
        assert (t[raw["cycle"] == last] == 0).all()


@pytest.mark.requires_data
def test_load_train_never_opens_test_or_rul_files(monkeypatch: pytest.MonkeyPatch) -> None:
    import turbofan.data.loader as loader

    opened: list[str] = []
    real = loader._read_txt
    monkeypatch.setattr(loader, "_read_txt", lambda p: opened.append(str(p)) or real(p))
    real_csv = loader.pd.read_csv
    monkeypatch.setattr(
        loader.pd, "read_csv", lambda p, *a, **k: opened.append(str(p)) or real_csv(p, *a, **k)
    )
    tr = loader.load_train("FD001", "data/raw")
    assert opened and all("train_FD001" in p for p in opened), opened
    assert tr["rul"].max() == 125
