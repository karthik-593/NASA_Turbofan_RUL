"""Tests for turbofan.evaluation.comparison — build_registry() and split_engines()."""

from __future__ import annotations

import pandas as pd

from turbofan.evaluation.comparison import Candidate, build_registry, split_engines

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tr(n_units: int = 10) -> pd.DataFrame:
    """Minimal training DataFrame with a 'unit' column."""
    return pd.DataFrame({"unit": list(range(1, n_units + 1)), "cycle": 1})


# ---------------------------------------------------------------------------
# build_registry()
# ---------------------------------------------------------------------------


class TestBuildRegistry:
    def test_has_exactly_five_keys(self) -> None:
        reg = build_registry()
        assert set(reg.keys()) == {"mean", "ridge", "rf", "xgboost", "lstm"}

    def test_flat_model_kinds(self) -> None:
        reg = build_registry()
        for name in ("mean", "ridge", "rf", "xgboost"):
            assert reg[name].kind == "flat", f"{name} should have kind='flat'"

    def test_lstm_is_sequence(self) -> None:
        reg = build_registry()
        assert reg["lstm"].kind == "sequence"

    def test_all_values_are_candidates(self) -> None:
        reg = build_registry()
        for name, cand in reg.items():
            assert isinstance(cand, Candidate), f"{name} is not a Candidate"

    def test_factories_return_non_none(self) -> None:
        reg = build_registry()
        for name, cand in reg.items():
            assert cand.factory() is not None, f"{name} factory returned None"

    def test_factory_returns_fresh_instance(self) -> None:
        """Each call to factory() must return a distinct object."""
        reg = build_registry()
        a = reg["mean"].factory()
        b = reg["mean"].factory()
        assert a is not b

    def test_cpu_device_accepted(self) -> None:
        """build_registry should work with an explicit device string."""
        reg = build_registry(device="cpu")
        assert "xgboost" in reg


# ---------------------------------------------------------------------------
# split_engines()
# ---------------------------------------------------------------------------


class TestSplitEngines:
    def test_8_2_split_for_10_engines(self) -> None:
        tr_e, va_e = split_engines(_make_tr(10))
        assert len(tr_e) == 8
        assert len(va_e) == 2

    def test_no_overlap(self) -> None:
        tr_e, va_e = split_engines(_make_tr(10))
        assert len(set(tr_e) & set(va_e)) == 0

    def test_union_covers_all_engines(self) -> None:
        tr_e, va_e = split_engines(_make_tr(10))
        assert set(tr_e) | set(va_e) == set(range(1, 11))

    def test_default_frac_is_0_8(self) -> None:
        tr_e, va_e = split_engines(_make_tr(20))
        assert len(tr_e) == 16
        assert len(va_e) == 4

    def test_first_k_engines_in_train(self) -> None:
        """First-frac fraction of engines by order must go to train."""
        tr_e, va_e = split_engines(_make_tr(10))
        assert set(tr_e) == set(range(1, 9))
        assert set(va_e) == {9, 10}

    def test_custom_frac(self) -> None:
        tr_e, va_e = split_engines(_make_tr(10), frac=0.5)
        assert len(tr_e) == 5
        assert len(va_e) == 5
