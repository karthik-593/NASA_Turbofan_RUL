"""Model-selection orchestrator (D51): ``python -m turbofan.pipeline select A --dataset FD001``.

Applies the rules pre-registered in ``reports/model_selection.md`` §0. Each configuration runs in
its own process through the pipeline entry point (``cv_select --tag T --set ...``) — parameters are
read at import, and the entry point carries resume (D49) and the GPU-teardown workaround (D50). A
configuration whose ``selection/T`` output already holds a complete run with the same overrides is
not run again. The comparisons read the runs' held-out predictions and write
``reports/selection/stage_<S>_<dataset>.json`` and ``.md``.

Stage A (XGBoost, seed ``selection.seed``):

- A1 ``rul_cap`` over ``rul_cap_grid`` vs the params.yaml cap;
- A2 ``window`` over ``selection.window_grid`` vs the params.yaml window, at the A1 cap;
- A3 feature blocks, each alone vs base at the A1/A2 winners (+ A3-combined if >= 2 adopted);
- A4 early identifiability (``identifiability`` stage).

A challenger replaces its incumbent only if the paired Δ critical RMSE CI lies entirely below 0;
among several, the lowest point estimate wins. Ties keep the incumbent (simpler / status quo).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from turbofan.config import N_REGIMES, PARAMS, RUL_CAP, WINDOW
from turbofan.evaluation.compare import CVResult, matched_budget_compare, paired_compare
from turbofan.evaluation.cv_metrics import HEADLINE
from turbofan.evaluation.views import deployment_view
from turbofan.pipeline.__main__ import SELECTION_DIR, parse_sets
from turbofan.pipeline.context import REPO

__all__ = ["Config", "stage_a"]

OUT = REPO / "reports" / "selection"
MODEL = "xgboost"


@dataclass(frozen=True)
class Config:
    tag: str
    label: str
    sets: tuple[str, ...]
    stage: str = "cv_select"


@dataclass
class Ledger:
    dataset: str
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    winners: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _metrics_path(tag: str) -> Path:
    return SELECTION_DIR / tag / "reports" / "cv_select" / "metrics.json"


def _complete(cfg: Config) -> bool:
    over = SELECTION_DIR / cfg.tag / "overrides.yaml"
    if not (_metrics_path(cfg.tag).is_file() and over.is_file()):
        return False
    return bool(yaml.safe_load(over.read_text(encoding="utf-8")) == parse_sets(list(cfg.sets)))


def ensure(cfg: Config) -> dict[str, Any]:
    """Run ``cfg`` through the pipeline entry point unless a complete run exists; return its
    metrics."""
    if not _complete(cfg):
        args = [sys.executable, "-m", "turbofan.pipeline", cfg.stage, "--tag", cfg.tag]
        for s in cfg.sets:
            args += ["--set", s]
        print(f"[select] run {cfg.tag}: {' '.join(cfg.sets)}", flush=True)
        res = subprocess.run(args, env={**os.environ, "PYTHONUTF8": "1"}, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"configuration {cfg.tag} failed (exit {res.returncode})")
        if not _complete(cfg):
            raise RuntimeError(f"configuration {cfg.tag} exited 0 but left no complete run")
    else:
        print(f"[select] reuse {cfg.tag}", flush=True)
    doc: dict[str, Any] = json.loads(_metrics_path(cfg.tag).read_text(encoding="utf-8"))
    return doc


def _result(tag: str, dataset: str, cap: float) -> CVResult:
    pts = pd.read_parquet(SELECTION_DIR / tag / f"points_{MODEL}.parquet")
    return CVResult(dataset, tag, cap, deployment_view(pts))


def _rng(*keys: str) -> np.random.Generator:
    return np.random.default_rng(
        [int(PARAMS["selection"]["seed"]), *(zlib.crc32(k.encode()) for k in keys)]
    )


def compare(led: Ledger, substage: str, inc: Config, ch: Config) -> dict[str, Any]:
    a = led.configs[ch.tag]
    b = led.configs[inc.tag]
    out = paired_compare(
        _result(ch.tag, led.dataset, a["rul_cap"]),
        _result(inc.tag, led.dataset, b["rul_cap"]),
        _rng(substage, ch.tag, inc.tag),
    )
    row = {
        "substage": substage,
        "challenger": ch.label,
        "incumbent": inc.label,
        "challenger_tag": ch.tag,
        "incumbent_tag": inc.tag,
        "estimate_challenger": out["estimate_a"],
        "estimate_incumbent": out["estimate_b"],
        "diff": out["diff"],
        "diff_ci_lo": out["diff_ci_lo"],
        "diff_ci_hi": out["diff_ci_hi"],
        "better": bool(out["diff_ci_hi"] < 0),
        "worse": bool(out["diff_ci_lo"] > 0),
        "share_engines_challenger_better": out["share_engines_a_better"],
        "wilcoxon_p": out["wilcoxon_p"],
        "wilcoxon_n_splits": out["wilcoxon_n_splits"],
        "n_engines": out["n_engines"],
        "n_points": out["n_points"],
    }
    led.comparisons.append(row)
    return row


def decision_check(led: Ledger, substage: str, inc: Config, ch: Config) -> None:
    a, b = led.configs[ch.tag], led.configs[inc.tag]
    t = matched_budget_compare(
        _result(ch.tag, led.dataset, a["rul_cap"]),
        _result(inc.tag, led.dataset, b["rul_cap"]),
        _rng("decision", substage, ch.tag, inc.tag),
    )
    for r in t.to_dict("records"):
        led.decisions.append(
            {
                "substage": substage,
                "challenger": ch.label,
                "incumbent": inc.label,
                "budget": float(r["budget"]),
                "caught_challenger": float(r["estimate_a"]),
                "caught_incumbent": float(r["estimate_b"]),
                "diff": float(r["diff"]),
                "diff_ci_lo": float(r["diff_ci_lo"]),
                "diff_ci_hi": float(r["diff_ci_hi"]),
                "n_engines": int(r["n_engines"]),
                "boot_defined_share": float(r["boot_defined_share"]),
            }
        )


def pick(led: Ledger, substage: str, inc: Config, challengers: list[Config]) -> Config:
    """Pre-registered rule: the challenger with the lowest point estimate among those whose
    paired Δ CI lies entirely below 0; else the incumbent."""
    rows = [compare(led, substage, inc, ch) for ch in challengers]
    better = [
        (r["estimate_challenger"], ch)
        for r, ch in zip(rows, challengers, strict=True)
        if r["better"]
    ]
    winner = min(better, key=lambda x: x[0])[1] if better else inc
    led.winners[substage] = winner.label
    if winner is not inc:
        decision_check(led, substage, inc, winner)
    return winner


def _headline(led: Ledger, cfg: Config) -> float:
    return float(led.configs[cfg.tag]["models"][MODEL][HEADLINE])


def _run(led: Ledger, cfg: Config) -> None:
    led.configs[cfg.tag] = ensure(cfg)


def stage_a(dataset: str) -> Ledger:
    led = Ledger(dataset)
    seed = int(PARAMS["selection"]["seed"])
    base = (
        f"pipeline.dataset={dataset}",
        f"pipeline.models=[{MODEL}]",
        f"pipeline.seeds=[{seed}]",
        "pipeline.resume=true",
    )

    def cap_cfg(cap: int) -> Config:
        return Config(f"A/{dataset}/cap{cap:03d}", f"cap {cap}", (*base, f"rul_cap={cap}"))

    # A1 — cap (incumbent: params.yaml rul_cap)
    caps = [int(c) for c in PARAMS["rul_cap_grid"]]
    inc_cap = int(RUL_CAP)
    if inc_cap not in caps:
        raise ValueError(f"rul_cap {inc_cap} is not in rul_cap_grid {caps}")
    a1 = {c: cap_cfg(c) for c in caps}
    for cfg in a1.values():
        _run(led, cfg)
    w1 = pick(led, "A1", a1[inc_cap], [a1[c] for c in caps if c != inc_cap])
    cap = int(led.configs[w1.tag]["rul_cap"])

    # A2 — window at the A1 cap (incumbent: params.yaml window, i.e. the A1 winner itself)
    windows = [int(w) for w in PARAMS["selection"]["window_grid"]]
    if WINDOW not in windows:
        raise ValueError(f"window {WINDOW} is not in selection.window_grid {windows}")
    a2_inc = Config(w1.tag, f"cap {cap}, window {WINDOW}", w1.sets)
    a2 = [
        Config(f"{w1.tag}_w{w}", f"cap {cap}, window {w}", (*w1.sets, f"window={w}"))
        for w in windows
        if w != WINDOW
    ]
    for cfg in a2:
        _run(led, cfg)
    w2 = pick(led, "A2", a2_inc, a2)

    # A3 — feature blocks, each alone vs base at the A1/A2 winners
    blocks: dict[str, tuple[str, ...]] = {
        "hi_consistent": ("features.health_index=consistent",),
        "hi_pooled": ("features.health_index=pooled",),
    }
    if N_REGIMES[dataset] > 1:
        blocks["regime_onehot"] = ("features.regime_onehot=true",)
    extra = PARAMS["selection"]["extra_sensors"][dataset]
    blocks["sensors_plus"] = (f"features.extra_sensors=[{', '.join(extra)}]",)
    a3_inc = Config(w2.tag, f"{w2.label} (base features)", w2.sets)
    a3 = {
        name: Config(f"{w2.tag}_{name}", f"{w2.label} + {name}", (*w2.sets, *sets))
        for name, sets in blocks.items()
    }
    for cfg in a3.values():
        _run(led, cfg)
    rows = [compare(led, "A3", a3_inc, cfg) for cfg in a3.values()]
    adopted = [name for (name, _), r in zip(a3.items(), rows, strict=True) if r["better"]]
    winner = a3_inc
    if len(adopted) >= 2:
        sets: list[str] = list(w2.sets)
        combined_name = "+".join(adopted)
        hi = [n for n in adopted if n.startswith("hi_")]
        if len(hi) == 2:
            # both health indices adopted: one column set only, so the better one enters the union
            best_hi = min(hi, key=lambda n: _headline(led, a3[n]))
            led.notes.append(
                f"A3-combined: both health indices adopted; {best_hi} (lower point estimate) is "
                "the one combined, as the two share the same three columns"
            )
            adopted_c = [n for n in adopted if not n.startswith("hi_")] + [best_hi]
        else:
            adopted_c = adopted
        for n in adopted_c:
            sets += list(blocks[n])
        comb = Config(f"{w2.tag}_combined", f"{w2.label} + {combined_name}", tuple(sets))
        _run(led, comb)
        r = compare(led, "A3-combined", a3_inc, comb)
        if r["better"]:
            winner = comb
        else:
            winner = min(
                (a3[n] for n in adopted),
                key=lambda c: led.configs[c.tag]["models"][MODEL][HEADLINE],
            )
    elif adopted:
        winner = a3[adopted[0]]
    led.winners["A3"] = winner.label
    if winner is not a3_inc:
        decision_check(led, "A3", a3_inc, winner)
    led.winners["stage_A"] = winner.label
    led.winners["stage_A_tag"] = winner.tag

    # A4 — early identifiability (reported, never forced into the feature set)
    a4 = Config(
        f"A/{dataset}/A4", "A4 identifiability", (f"pipeline.dataset={dataset}",), "identifiability"
    )
    led.configs[a4.tag] = ensure(a4)
    return led


def _ci(est: float, lo: float, hi: float, p: int = 2) -> str:
    return f"{est:.{p}f} [{lo:.{p}f}, {hi:.{p}f}]"


def render(led: Ledger) -> str:
    ds = led.dataset
    lines = [f"### Stage A — {ds}", ""]
    lines += [
        "| config | critical RMSE [95% CI] | n engines | subpop 0 | subpop 1 | wall (min) |",
        "|---|---|---|---|---|---|",
    ]
    for tag, doc in led.configs.items():
        if "models" not in doc:
            continue
        m = doc["models"][MODEL]
        sp = doc["subpopulations"][MODEL]
        lines.append(
            f"| `{tag}` | {_ci(m[HEADLINE], m[f'{HEADLINE}_ci_lo'], m[f'{HEADLINE}_ci_hi'])} | "
            f"{m['n_engines']} | "
            + " | ".join(
                f"{_ci(sp[g]['estimate'], sp[g]['ci_lo'], sp[g]['ci_hi'])} (n={sp[g]['n_engines']})"
                for g in ("subpop_0", "subpop_1")
            )
            + f" | {doc['wall_seconds'] / 60:.1f} |"
        )
    lines += [
        "",
        "Paired comparisons (Δ = challenger − incumbent, critical RMSE; better = CI entirely < 0):",
        "",
        "| step | challenger | incumbent | Δ [95% CI] | engines challenger better "
        "| Wilcoxon p (n splits) | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in led.comparisons:
        verdict = "**better**" if r["better"] else ("worse" if r["worse"] else "not different")
        lines.append(
            f"| {r['substage']} | {r['challenger']} | {r['incumbent']} | "
            f"{_ci(r['diff'], r['diff_ci_lo'], r['diff_ci_hi'])} | "
            f"{r['share_engines_challenger_better']:.0%} of {r['n_engines']} | "
            f"{r['wilcoxon_p']:.2g} ({r['wilcoxon_n_splits']}) | {verdict} |"
        )
    if led.decisions:
        lines += [
            "",
            "Decision check for each adopted change (caught % with lead ≥ 20 at matched mean "
            "wasted life; Δ = challenger − incumbent, positive = challenger catches more):",
            "",
            "| step | challenger vs incumbent | budget (cycles) | caught % challenger "
            "| caught % incumbent | Δ [95% CI] | n engines |",
            "|---|---|---|---|---|---|---|",
        ]
        for r in led.decisions:
            lines.append(
                f"| {r['substage']} | {r['challenger']} vs {r['incumbent']} | {r['budget']:g} | "
                f"{r['caught_challenger']:.1f} | {r['caught_incumbent']:.1f} | "
                f"{_ci(r['diff'], r['diff_ci_lo'], r['diff_ci_hi'], 1)} | {r['n_engines']} |"
            )
    a4 = led.configs.get(f"A/{ds}/A4")
    if a4:
        lines += [
            "",
            f"A4 — subpopulation from the first N cycles (logistic regression, AUC [95% CI], "
            f"strong = CI lower bound ≥ {a4['strong_threshold_ci_lo']:.2f}; subpopulation sizes "
            f"{a4['subpopulation_sizes']}):",
            "",
            "| N cycles | AUC [95% CI] | per repeat | n engines | strong |",
            "|---|---|---|---|---|",
        ]
        for r in a4["results"]:
            per = ", ".join(f"{x:.3f}" for x in r["auc_per_repeat"])
            lines.append(
                f"| {r['n_cycles']} | {_ci(r['auc'], r['ci_lo'], r['ci_hi'], 3)} | {per} | "
                f"{r['n_engines']} | {'yes' if r['strong'] else 'no'} |"
            )
    lines += [
        "",
        "Winners: "
        + "; ".join(f"{k}: {v}" for k, v in led.winners.items() if not k.endswith("_tag")),
    ]
    for n in led.notes:
        lines.append(f"- {n}")
    return "\n".join(lines) + "\n"


def write(led: Ledger, stage: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"stage_{stage}_{led.dataset}"
    doc = {
        "dataset": led.dataset,
        "winners": led.winners,
        "comparisons": led.comparisons,
        "decisions": led.decisions,
        "notes": led.notes,
        "configs": led.configs,
    }
    stem.with_suffix(".json").write_text(json.dumps(doc, indent=2, default=float) + "\n", "utf-8")
    stem.with_suffix(".md").write_text(render(led), encoding="utf-8")
    return stem


def main(stage: str, datasets: list[str]) -> None:
    if stage != "A":
        raise NotImplementedError(f"stage {stage} is not implemented yet (stop after Stage A, D51)")
    for ds in datasets:
        led = stage_a(ds)
        print(f"[select] wrote {write(led, stage)}", flush=True)
