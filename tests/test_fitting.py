"""Iteration budgets and refit-on-all-engines (D48)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from turbofan.config import FEAT_COLS
from turbofan.evaluation.fitting import iteration_budget, refit_candidate
from turbofan.models.baselines import MeanBaseline, RidgeRUL
from turbofan.models.xgboost_model import XGBoostRUL


def _feat(n: int = 60) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.normal(size=(n, len(FEAT_COLS))), columns=FEAT_COLS)
    df["unit"] = 1
    df["cycle"] = np.arange(1, n + 1)
    df["rul"] = rng.uniform(0, 125, n)
    return df


class _Fake:
    def __init__(self, **attrs: object) -> None:
        self.__dict__.update(attrs)


class TestIterationBudget:
    def test_lstm_uses_best_epoch(self) -> None:
        assert iteration_budget(_Fake(best_epoch_=7)) == 7
        assert iteration_budget(_Fake(best_epoch_=None)) is None

    def test_xgboost_is_zero_based_round_plus_one(self) -> None:
        assert iteration_budget(_Fake(best_iteration_=41)) == 42

    def test_model_without_iterations(self) -> None:
        assert iteration_budget(MeanBaseline()) is None


class TestRefit:
    def test_baselines_fit_on_all_rows_and_refuse_a_budget(self) -> None:
        f = _feat()
        m = refit_candidate(RidgeRUL(), "flat", f, None)
        assert len(m.predict(f[FEAT_COLS])) == len(f)
        with pytest.raises(ValueError, match="budget must be None"):
            refit_candidate(MeanBaseline(), "flat", f, 5)

    def test_iterative_model_needs_a_budget(self) -> None:
        with pytest.raises(ValueError, match="needs an iteration budget"):
            refit_candidate(XGBoostRUL(), "flat", _feat(), None)

    def test_xgboost_trains_exactly_the_budget(self) -> None:
        m = refit_candidate(XGBoostRUL(params={"n_estimators": 400}), "flat", _feat(), 6)
        assert m.model_.get_booster().num_boosted_rounds() == 6

    def test_unknown_kind_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown candidate kind"):
            refit_candidate(RidgeRUL(), "tree", _feat(), None)
