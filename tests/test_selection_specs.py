"""Content-addressed specs and the addendum / Stage B-D rules of the orchestrator (D51)."""

from __future__ import annotations

from typing import Any

import pytest

import turbofan.pipeline.selection as sel
from turbofan.pipeline.__main__ import parse_sets


def _xgb(**kw: Any) -> sel.Spec:
    return sel.Spec("FD002", "xgboost", 90, 45, **kw)


class TestSpec:
    def test_tag_is_content_addressed(self) -> None:
        a = _xgb(blocks=("regime_onehot", "hi_consistent"))
        b = _xgb(blocks=("hi_consistent", "regime_onehot"))
        assert a == b and a.tag == b.tag
        assert a.tag == "runs/FD002/xgboost/cap090_w045_hi_consistent_regime_onehot"
        assert _xgb().tag != _xgb(seeds=(42, 7)).tag != _xgb(n_repeats=1).tag

    def test_overrides_round_trip_through_the_entry_points_parser(self) -> None:
        s = _xgb(
            blocks=("baseline",),
            tuned=(("learning_rate", 1e-05), ("max_depth", 7), ("n_estimators", 2000)),
            n_repeats=1,
            n_boot=200,
        )
        over = parse_sets(list(s.sets))
        assert over["rul_cap"] == 90 and over["window"] == 45
        assert over["features"] == {"baseline": True}
        assert over["models"]["xgboost"]["params"] == {
            "learning_rate": 1e-05,  # PyYAML reads a bare 1e-05 as a string; _scalar does not
            "max_depth": 7,
            "n_estimators": 2000,
        }
        assert over["cv"] == {"n_repeats": 1} and over["bootstrap"] == {"n_boot": 200}

    def test_lstm_tuned_keys_live_under_models_lstm(self) -> None:
        s = sel.Spec("FD001", "lstm", 90, 45, seq_len=60, tuned=(("hidden", 64),))
        assert parse_sets(list(s.sets))["models"] == {"lstm": {"hidden": 64}}
        assert "L060" in s.tag

    def test_json_round_trip(self) -> None:
        s = _xgb(blocks=("hi_pooled",), tuned=(("subsample", 0.75),), seeds=(42, 7))
        assert sel.Spec.from_json(s.to_json()) == s

    def test_unknown_block_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown or repeated"):
            _xgb(blocks=("hi_magic",))


class TestAdoptedChanges:
    def test_every_change_from_the_defaults_has_its_leave_one_out_spec(self) -> None:
        fin = _xgb(blocks=("hi_consistent", "regime_onehot"), tuned=(("max_depth", 4),))
        got = dict(sel.adopted_changes(fin))
        assert list(got) == ["cap", "window", "hi_consistent", "regime_onehot", "tuning"]
        assert got["cap"].rul_cap == 125 and got["window"].window == 20
        assert got["regime_onehot"].blocks == ("hi_consistent",)
        assert got["tuning"].tuned == ()

    def test_lstm_reverts_seq_len_not_window(self) -> None:
        fin = sel.Spec("FD001", "lstm", 90, 45, seq_len=45)
        assert [n for n, _ in sel.adopted_changes(fin)] == ["cap", "seq_len"]


def _dc(*cis: tuple[float, float]) -> list[dict[str, Any]]:
    return [{"diff_ci_lo": lo, "diff_ci_hi": hi} for lo, hi in cis]


def test_decision_better_needs_one_budget_better_and_none_worse() -> None:
    assert sel._decision_better(_dc((0.5, 3.0), (-1.0, 1.0)), +1)
    assert not sel._decision_better(_dc((0.5, 3.0), (-3.0, -0.1)), +1)
    assert not sel._decision_better(_dc((-1.0, 1.0)), +1)
    assert sel._decision_better(_dc((-3.0, -0.5)), -1)


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """No runs; comparisons from a table keyed by (challenger tag, incumbent tag)."""
    table: dict[tuple[str, str], tuple[float, float, float]] = {}
    state: dict[str, Any] = {"table": table, "cap_ok": True, "checked": []}

    def compare(led: sel.Ledger, sub: str, inc: Any, ch: Any, **kw: Any) -> dict[str, Any]:
        est, lo, hi = table.get((ch.tag, inc.tag), (0.0, -1.0, 1.0))
        row = {
            "estimate_challenger": est,
            "diff_ci_lo": lo,
            "diff_ci_hi": hi,
            "better": hi < 0,
            "worse": lo > 0,
        }
        led.comparisons.append(row)
        return row

    monkeypatch.setattr(sel, "_run", lambda led, c: led.configs.setdefault(c.tag, {}))
    monkeypatch.setattr(sel, "compare", compare)
    monkeypatch.setattr(
        sel, "decision_check", lambda led, s, inc, ch, **kw: state["checked"].append(ch.tag) or []
    )
    monkeypatch.setattr(sel, "cap_criteria", lambda *a: (state["cap_ok"], []))
    monkeypatch.setattr(sel, "_headline", lambda led, c: table.get(("est", c.tag), (9.0,))[0])
    return state


class TestBlockScreen:
    def test_single_adopted_block_wins(self, fake: dict[str, Any]) -> None:
        inc = _xgb()
        fake["table"][(inc.plus("regime_onehot").tag, inc.tag)] = (5.0, -0.5, -0.1)
        got = sel.block_screen(sel.Ledger("FD002"), "B2", inc, ["regime_onehot", "hi_pooled"])
        assert got == inc.plus("regime_onehot")

    def test_both_health_indices_only_the_better_enters_the_union(
        self, fake: dict[str, Any]
    ) -> None:
        inc = _xgb()
        t = fake["table"]
        for b in ("hi_consistent", "hi_pooled", "regime_onehot"):
            t[(inc.plus(b).tag, inc.tag)] = (5.0, -0.5, -0.1)
        t[("est", inc.plus("hi_pooled").tag)] = (4.0,)
        union = inc.plus("hi_pooled", "regime_onehot")
        t[(union.tag, inc.tag)] = (3.0, -0.9, -0.2)
        led = sel.Ledger("FD002")
        got = sel.block_screen(
            led, "A3-rerun", inc, ["hi_consistent", "hi_pooled", "regime_onehot"]
        )
        assert got == union
        assert any("hi_pooled" in n for n in led.notes)


class TestConfirm:
    def test_changes_whose_ci_includes_zero_are_dropped(self, fake: dict[str, Any]) -> None:
        fin = _xgb(blocks=("hi_consistent", "regime_onehot"), seeds=(1, 2))
        t = fake["table"]
        for _name, minus in sel.adopted_changes(fin):  # fin beats every leave-one-out ...
            t[(fin.tag, minus.tag)] = (4.0, -0.8, -0.2)
        t[(fin.tag, fin.minus("regime_onehot").tag)] = (4.0, -0.3, 0.1)  # ... but this one
        assert sel.confirm(sel.Ledger("FD002"), fin) == fin.minus("regime_onehot")

    def test_cap_needs_the_cap_check_too(self, fake: dict[str, Any]) -> None:
        fin = _xgb()
        for _name, minus in sel.adopted_changes(fin):
            fake["table"][(fin.tag, minus.tag)] = (4.0, -0.8, -0.2)
        fake["cap_ok"] = False
        assert sel.confirm(sel.Ledger("FD002"), fin) == fin.with_(rul_cap=125)

    def test_several_dropped_run_the_reduced_spec_unless_the_full_one_beats_it(
        self, fake: dict[str, Any]
    ) -> None:
        fin = _xgb(blocks=("hi_consistent", "regime_onehot"))
        reduced = fin.minus("hi_consistent", "regime_onehot")
        t = fake["table"]
        for _name, minus in sel.adopted_changes(fin):
            t[(fin.tag, minus.tag)] = (4.0, -0.8, -0.2)
        for b in ("hi_consistent", "regime_onehot"):
            t[(fin.tag, fin.minus(b).tag)] = (4.0, -0.3, 0.1)
        assert sel.confirm(sel.Ledger("FD002"), fin) == reduced
        t[(fin.tag, reduced.tag)] = (4.0, -0.6, -0.1)  # interaction: together they matter
        led = sel.Ledger("FD002")
        assert sel.confirm(led, fin) == fin and any("interaction" in n for n in led.notes)


class TestLocking:
    @pytest.fixture
    def run_d(self, fake: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> Any:
        xgb = _xgb(seeds=(1,))
        lstm = sel.Spec("FD002", "lstm", 90, 45, seeds=(1,))
        written: list[tuple[str, bool, str]] = []
        doc = {"specs": {"xgboost": xgb.to_json(), "lstm": lstm.to_json()}}
        monkeypatch.setattr(sel, "_load", lambda stage, ds: doc)
        monkeypatch.setattr(sel, "SEEDS", (1,))
        monkeypatch.setattr(sel, "confirm", lambda led, s: s)
        monkeypatch.setattr(
            sel,
            "write_spec",
            lambda s, locked, status, led: written.append((s.model, locked, status)),
        )

        def go(rmse: tuple[float, float, float], dc: list[dict[str, Any]]) -> Any:
            fake["table"][(lstm.tag, xgb.tag)] = rmse
            monkeypatch.setattr(sel, "decision_check", lambda *a, **k: dc)
            written.clear()
            led = sel.stage_d("FD002")
            return led.extra["outcome"], list(written)

        return go

    def test_rmse_winner_also_better_on_decision_is_locked(self, run_d: Any) -> None:
        out, written = run_d((4.0, -0.9, -0.2), _dc((1.0, 4.0)))
        assert out["winner"] == "lstm" and out["locked"]
        assert ("lstm", True, "locked") in written

    def test_rmse_winner_not_better_on_decision_leaves_both_pending(self, run_d: Any) -> None:
        out, written = run_d((4.0, -0.9, -0.2), _dc((-1.0, 2.0)))
        assert not out["locked"]
        assert sorted(written) == [
            ("lstm", False, "locked-pending"),
            ("xgboost", False, "locked-pending"),
        ]

    def test_tie_goes_to_xgboost_unless_the_lstm_wins_the_decision_check(self, run_d: Any) -> None:
        out, _ = run_d((4.0, -0.5, 0.4), _dc((-1.0, 2.0)))
        assert out["winner"] == "xgboost" and out["locked"]
        out, _ = run_d((4.0, -0.5, 0.4), _dc((0.5, 2.0)))
        assert out["winner"] == "xgboost" and not out["locked"]
