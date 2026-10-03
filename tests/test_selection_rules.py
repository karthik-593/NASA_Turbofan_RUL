"""The pre-registered selection rule of the Stage A orchestrator (D51)."""

from __future__ import annotations

from typing import Any

import pytest

import turbofan.pipeline.selection as sel


def _cfg(name: str) -> sel.Config:
    return sel.Config(f"A/FD001/{name}", name, ())


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, Any]]:
    """Comparisons come from a table: challenger label -> (estimate, ci_lo, ci_hi)."""
    table: dict[str, dict[str, Any]] = {}
    checked: list[str] = []

    def compare(led: sel.Ledger, substage: str, inc: sel.Config, ch: sel.Config) -> dict[str, Any]:
        est, lo, hi = table[ch.label]
        row = {"estimate_challenger": est, "diff_ci_lo": lo, "diff_ci_hi": hi, "better": hi < 0}
        led.comparisons.append(row)
        return row

    monkeypatch.setattr(sel, "compare", compare)
    monkeypatch.setattr(sel, "decision_check", lambda led, s, inc, ch: checked.append(ch.label))
    table["_checked"] = checked  # type: ignore[assignment]
    return table


def test_ties_keep_the_incumbent(fake: dict[str, Any]) -> None:
    fake.update({"a": (10.0, -1.0, 0.5), "b": (9.0, -2.0, 0.1)})  # CIs include 0
    led = sel.Ledger("FD001")
    assert sel.pick(led, "A1", _cfg("inc"), [_cfg("a"), _cfg("b")]).label == "inc"
    assert led.winners["A1"] == "inc" and fake["_checked"] == []


def test_lowest_estimate_among_the_better_ones(fake: dict[str, Any]) -> None:
    fake.update({"a": (8.0, -3.0, -0.5), "b": (7.5, -4.0, -0.2), "c": (7.0, -5.0, 0.3)})
    led = sel.Ledger("FD001")
    # c has the lowest estimate but its CI includes 0, so it cannot win
    assert sel.pick(led, "A2", _cfg("inc"), [_cfg("a"), _cfg("b"), _cfg("c")]).label == "b"
    assert fake["_checked"] == ["b"]  # the adopted change gets the decision check


def test_worse_challengers_never_win(fake: dict[str, Any]) -> None:
    fake.update({"a": (12.0, 0.5, 2.0)})
    assert sel.pick(sel.Ledger("FD001"), "A1", _cfg("inc"), [_cfg("a")]).label == "inc"
