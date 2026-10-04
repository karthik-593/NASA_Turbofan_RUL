"""Model-selection orchestrator (D51): ``python -m turbofan.pipeline select <stage> --dataset ...``.

Applies the rules pre-registered in ``reports/model_selection.md`` (§0 and the post-Stage-A
addendum). Each configuration runs in its own process through the pipeline entry point
(``cv_select --tag T --set ...``) — parameters are read at import, and the entry point carries
resume (D49) and the GPU-teardown workaround (D50). A configuration whose ``selection/T`` output
already holds a complete run with the same overrides is not run again. The comparisons read the
runs' held-out predictions and write ``reports/selection/stage_<S>_<dataset>.json`` and ``.md``.

Stages:

- ``A`` — XGBoost screening (seed ``selection.seed``): A1 cap, A2 window, A3 feature blocks,
  A4 early identifiability. Run before the closed-form slope (D52); tags ``A/<ds>/...``.
- ``A1check`` — cap 90 vs 125 on metrics where neither cap binds (urgent RMSE, critical / urgent
  late %, matched-budget decision check), from the A1 runs' MLflow predictions; no new runs. A
  failure stops the addendum.
- ``X`` — the addendum: A2-ext windows (A3 re-run if the window changes), A5a baseline
  deviation, A4 re-run with a label-shuffle control, A5b subpopulation probability.
- ``B`` — LSTM: B1 ``seq_len``, B2 channel blocks.
- ``C`` — Optuna tuning on the repeat-1 folds, evaluated on repeats 2-3.
- ``D`` — 5-seed confirmation, leave-one-out check of every adopted change, finalist
  comparison, locked / locked-pending specs (``specs/``).

From ``X`` on, a run is a ``Spec`` and its tag is derived from its content
(``runs/<ds>/<model>/<slug>``), so an identical configuration requested by a later stage — a
Stage C incumbent, a Stage D leave-one-out spec — is the same run, reused.

A challenger replaces its incumbent only if the paired Δ critical RMSE CI lies entirely below 0;
among several, the lowest point estimate wins. Ties keep the incumbent (simpler / status quo).
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zlib
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from turbofan.config import N_REGIMES, PARAMS, RUL_CAP, SEEDS, SEQ_LEN, WINDOW
from turbofan.evaluation.compare import CVResult, matched_budget_compare, paired_compare
from turbofan.evaluation.cv_metrics import HEADLINE, metric_table
from turbofan.evaluation.views import deployment_view
from turbofan.pipeline.__main__ import SELECTION_DIR, parse_sets
from turbofan.pipeline.context import REPO

__all__ = ["Config", "Spec", "stage_a", "a1_check", "stage_x", "stage_b", "stage_c", "stage_d"]

OUT = REPO / "reports" / "selection"
SPECS_DIR = REPO / "specs"
MODEL = "xgboost"  # Stage A
RUN_PREFIX = "runs"  # content-addressed run tags from the addendum on (D52)
SEL = PARAMS["selection"]
SEED = int(SEL["seed"])
DEFAULT_CAP = int(RUL_CAP)
BLOCK_ORDER = (
    "hi_consistent",
    "hi_pooled",
    "regime_onehot",
    "sensors_plus",
    "baseline",
    "subpop_prob",
)
# A1-check / cap leave-one-out: metrics no cap >= 50 binds (truth < 50), "worse" = Δ CI > 0
CAP_CHECK_METRICS = ("urgent_rmse", "critical_late_pct", "urgent_late_pct")


@dataclass(frozen=True)
class Config:
    """A Stage A configuration, or a non-CV analysis run (A4)."""

    tag: str
    label: str
    sets: tuple[str, ...]
    stage: str = "cv_select"


def _scalar(v: Any) -> str:
    """A YAML scalar that parses back to ``v`` (PyYAML reads ``1e-05`` as a string)."""
    out = yaml.safe_dump(v, default_flow_style=True).split("\n")[0]
    if yaml.safe_load(out) != v:
        raise ValueError(f"{v!r} does not round-trip through YAML as {out!r}")
    return out


def block_sets(name: str, dataset: str) -> tuple[str, ...]:
    if name in ("hi_consistent", "hi_pooled"):
        return (f"features.health_index={name[3:]}",)
    if name == "regime_onehot":
        return ("features.regime_onehot=true",)
    if name == "sensors_plus":
        return (f"features.extra_sensors=[{', '.join(SEL['extra_sensors'][dataset])}]",)
    if name in ("baseline", "subpop_prob"):
        return (f"features.{name}=true",)
    raise KeyError(f"unknown feature block {name!r}")


@dataclass(frozen=True)
class Spec:
    """One candidate configuration of one model on one dataset (addendum and later stages)."""

    dataset: str
    model: str
    rul_cap: int
    window: int
    seq_len: int = SEQ_LEN
    blocks: tuple[str, ...] = ()
    tuned: tuple[tuple[str, Any], ...] = ()  # keys under models.xgboost.params / models.lstm
    seeds: tuple[int, ...] = (SEED,)
    n_repeats: int | None = None  # None = params.yaml cv.n_repeats
    n_boot: int | None = None  # None = params.yaml bootstrap.n_boot

    def __post_init__(self) -> None:
        unknown = [b for b in self.blocks if b not in BLOCK_ORDER]
        if unknown or len(set(self.blocks)) != len(self.blocks):
            raise ValueError(f"blocks {self.blocks}: unknown or repeated")
        object.__setattr__(self, "blocks", tuple(sorted(self.blocks, key=BLOCK_ORDER.index)))
        object.__setattr__(self, "tuned", tuple(sorted(tuple(kv) for kv in self.tuned)))
        object.__setattr__(self, "seeds", tuple(int(s) for s in self.seeds))
        if self.model not in ("xgboost", "lstm"):
            raise ValueError(f"model must be xgboost or lstm, got {self.model!r}")

    def with_(self, **kw: Any) -> Spec:
        return replace(self, **kw)

    def plus(self, *blocks: str) -> Spec:
        return replace(self, blocks=(*self.blocks, *blocks))

    def minus(self, *blocks: str) -> Spec:
        return replace(self, blocks=tuple(b for b in self.blocks if b not in blocks))

    @property
    def sets(self) -> tuple[str, ...]:
        s = [
            f"pipeline.dataset={self.dataset}",
            f"pipeline.models=[{self.model}]",
            f"pipeline.seeds=[{', '.join(str(x) for x in self.seeds)}]",
            "pipeline.resume=true",
            f"rul_cap={self.rul_cap}",
            f"window={self.window}",
            f"seq_len={self.seq_len}",
        ]
        for b in self.blocks:
            s += block_sets(b, self.dataset)
        prefix = "models.xgboost.params." if self.model == "xgboost" else "models.lstm."
        s += [f"{prefix}{k}={_scalar(v)}" for k, v in self.tuned]
        if self.n_repeats is not None:
            s.append(f"cv.n_repeats={self.n_repeats}")
        if self.n_boot is not None:
            s.append(f"bootstrap.n_boot={self.n_boot}")
        return tuple(s)

    @property
    def tag(self) -> str:
        parts = [f"cap{self.rul_cap:03d}", f"w{self.window:03d}"]
        if self.model == "lstm":
            parts.append(f"L{self.seq_len:03d}")
        parts += list(self.blocks)
        if self.tuned:
            h = hashlib.sha256(json.dumps(self.tuned, sort_keys=True).encode()).hexdigest()
            parts.append(f"tuned-{h[:8]}")
        if self.seeds != (SEED,):
            parts.append("seeds-" + "-".join(str(x) for x in self.seeds))
        if self.n_repeats is not None:
            parts.append(f"rep{self.n_repeats}")
        if self.n_boot is not None:
            parts.append(f"nb{self.n_boot}")
        return f"{RUN_PREFIX}/{self.dataset}/{self.model}/" + "_".join(parts)

    @property
    def label(self) -> str:
        bits = [self.model, f"cap {self.rul_cap}"]
        bits.append(
            f"window {self.window}" if self.model == "xgboost" else f"seq_len {self.seq_len}"
        )
        bits += list(self.blocks)
        if self.tuned:
            bits.append("tuned")
        if len(self.seeds) > 1:
            bits.append(f"{len(self.seeds)} seeds")
        return ", ".join(bits)

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["tuned"] = [list(kv) for kv in self.tuned]
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> Spec:
        return cls(
            **{
                **d,
                "blocks": tuple(d["blocks"]),
                "tuned": tuple(map(tuple, d["tuned"])),
                "seeds": tuple(d["seeds"]),
            }
        )


Candidate = Config | Spec


@dataclass
class Ledger:
    dataset: str
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    configs: dict[str, dict[str, Any]] = field(default_factory=dict)
    winners: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    specs: dict[str, dict[str, Any]] = field(default_factory=dict)
    sections: list[str] = field(default_factory=list)  # extra markdown, rendered in order
    extra: dict[str, Any] = field(default_factory=dict)  # extra JSON


def _metrics_path(tag: str) -> Path:
    return SELECTION_DIR / tag / "reports" / "cv_select" / "metrics.json"


def _complete(tag: str, sets: tuple[str, ...]) -> bool:
    over = SELECTION_DIR / tag / "overrides.yaml"
    if not (_metrics_path(tag).is_file() and over.is_file()):
        return False
    return bool(yaml.safe_load(over.read_text(encoding="utf-8")) == parse_sets(list(sets)))


def ensure(cfg: Candidate) -> dict[str, Any]:
    """Run ``cfg`` through the pipeline entry point unless a complete run exists; return its
    metrics."""
    stage = cfg.stage if isinstance(cfg, Config) else "cv_select"
    if not _complete(cfg.tag, cfg.sets):
        args = [sys.executable, "-m", "turbofan.pipeline", stage, "--tag", cfg.tag]
        for s in cfg.sets:
            args += ["--set", s]
        print(f"[select] run {cfg.tag}: {' '.join(cfg.sets)}", flush=True)
        res = subprocess.run(args, env={**os.environ, "PYTHONUTF8": "1"}, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"configuration {cfg.tag} failed (exit {res.returncode})")
        if not _complete(cfg.tag, cfg.sets):
            raise RuntimeError(f"configuration {cfg.tag} exited 0 but left no complete run")
    else:
        print(f"[select] reuse {cfg.tag}", flush=True)
    doc: dict[str, Any] = json.loads(_metrics_path(cfg.tag).read_text(encoding="utf-8"))
    return doc


def _run(led: Ledger, cfg: Candidate) -> None:
    led.configs[cfg.tag] = ensure(cfg)


def _model_of(doc: dict[str, Any]) -> str:
    (name,) = doc["models"]
    return str(name)


def _result(led: Ledger, tag: str, repeats: tuple[int, ...] | None = None) -> CVResult:
    doc = led.configs[tag]
    pts = pd.read_parquet(SELECTION_DIR / tag / f"points_{_model_of(doc)}.parquet")
    if repeats is not None:
        pts = pts[pts["repeat"].isin(repeats)]
    return CVResult(led.dataset, tag, float(doc["rul_cap"]), deployment_view(pts))


def _rng(*keys: str) -> np.random.Generator:
    return np.random.default_rng([SEED, *(zlib.crc32(k.encode()) for k in keys)])


def _row(substage: str, ch: Candidate, inc: Candidate, out: dict[str, Any]) -> dict[str, Any]:
    return {
        "substage": substage,
        "challenger": ch.label,
        "incumbent": inc.label,
        "challenger_tag": ch.tag,
        "incumbent_tag": inc.tag,
        "metric": out["metric"],
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


def compare(
    led: Ledger,
    substage: str,
    inc: Candidate,
    ch: Candidate,
    repeats: tuple[int, ...] | None = None,
    metric: str = HEADLINE,
) -> dict[str, Any]:
    out = paired_compare(
        _result(led, ch.tag, repeats),
        _result(led, inc.tag, repeats),
        _rng(substage, ch.tag, inc.tag, metric),
        metric=metric,
    )
    row = _row(substage, ch, inc, out)
    if repeats is not None:
        row["repeats"] = list(repeats)
    led.comparisons.append(row)
    return row


def decision_check(
    led: Ledger,
    substage: str,
    inc: Candidate,
    ch: Candidate,
    repeats: tuple[int, ...] | None = None,
) -> list[dict[str, Any]]:
    t = matched_budget_compare(
        _result(led, ch.tag, repeats),
        _result(led, inc.tag, repeats),
        _rng("decision", substage, ch.tag, inc.tag),
    )
    rows = []
    for r in t.to_dict("records"):
        rows.append(
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
    led.decisions.extend(rows)
    return rows


def pick(led: Ledger, substage: str, inc: Candidate, challengers: list[Candidate]) -> Candidate:
    """Pre-registered rule: the challenger with the lowest point estimate among those whose
    paired Δ CI lies entirely below 0; else the incumbent."""
    for c in (inc, *challengers):
        if isinstance(c, Spec) and c.tag not in led.configs:
            _run(led, c)
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


def _headline(led: Ledger, cfg: Candidate) -> float:
    doc = led.configs[cfg.tag]
    return float(doc["models"][_model_of(doc)][HEADLINE])


def fittable_extra_sensors(
    dataset: str, sensors: list[str]
) -> tuple[list[str], dict[str, list[int]]]:
    """The extra sensors ``add_features`` can normalize: not constant within any operating regime
    (regimes fitted on all training engines, as params.yaml defines them). Returns (fittable,
    {sensor: regimes where it is constant})."""
    from turbofan.data.loader import load_train
    from turbofan.features.engineering import _fit_regimes, assign_regimes

    train = load_train(dataset, REPO / "data" / "raw")
    op_sc, km = _fit_regimes(train, N_REGIMES[dataset])
    labels = assign_regimes(train, {"op_sc": op_sc, "km": km})
    constant = {
        s: [r for r in range(N_REGIMES[dataset]) if train.loc[labels == r, s].nunique() < 2]
        for s in sensors
    }
    return [s for s in sensors if not constant[s]], {s: v for s, v in constant.items() if v}


# -- Stage A (run 2026-10-03/04, before D52; kept as it ran) ----------------------------------


def stage_a(dataset: str) -> Ledger:
    led = Ledger(dataset)
    base = (
        f"pipeline.dataset={dataset}",
        f"pipeline.models=[{MODEL}]",
        f"pipeline.seeds=[{SEED}]",
        "pipeline.resume=true",
    )

    def cap_cfg(cap: int) -> Config:
        return Config(f"A/{dataset}/cap{cap:03d}", f"cap {cap}", (*base, f"rul_cap={cap}"))

    # A1 — cap (incumbent: params.yaml rul_cap)
    caps = [int(c) for c in PARAMS["rul_cap_grid"]]
    inc_cap = DEFAULT_CAP
    if inc_cap not in caps:
        raise ValueError(f"rul_cap {inc_cap} is not in rul_cap_grid {caps}")
    a1 = {c: cap_cfg(c) for c in caps}
    for cfg in a1.values():
        _run(led, cfg)
    w1 = pick(led, "A1", a1[inc_cap], [a1[c] for c in caps if c != inc_cap])
    cap = int(led.configs[w1.tag]["rul_cap"])

    # A2 — window at the A1 cap (incumbent: params.yaml window, i.e. the A1 winner itself)
    windows = [int(w) for w in SEL["window_grid"]]
    if WINDOW not in windows:
        raise ValueError(f"window {WINDOW} is not in selection.window_grid {windows}")
    assert isinstance(w1, Config)
    a2_inc = Config(w1.tag, f"cap {cap}, window {WINDOW}", w1.sets)
    a2: list[Candidate] = [
        Config(f"{w1.tag}_w{w}", f"cap {cap}, window {w}", (*w1.sets, f"window={w}"))
        for w in windows
        if w != WINDOW
    ]
    for c in a2:
        _run(led, c)
    w2 = pick(led, "A2", a2_inc, a2)
    assert isinstance(w2, Config)

    # A3 — feature blocks, each alone vs base at the A1/A2 winners
    blocks: dict[str, tuple[str, ...]] = {
        "hi_consistent": ("features.health_index=consistent",),
        "hi_pooled": ("features.health_index=pooled",),
    }
    if N_REGIMES[dataset] > 1:
        blocks["regime_onehot"] = ("features.regime_onehot=true",)
    extra = list(SEL["extra_sensors"][dataset])
    fittable, constant = fittable_extra_sensors(dataset, extra)
    informational: dict[str, tuple[str, ...]] = {}
    if constant:
        # Deviation from the pre-registration, recorded rather than worked around (rule 9):
        # the block as pre-registered cannot be fitted, so it is not adopted; the fittable
        # subset is run for information only and is not eligible for adoption.
        led.notes.append(
            f"A3 sensors_plus [{', '.join(extra)}] is infeasible as pre-registered: "
            + "; ".join(f"{s} is constant within regime(s) {r}" for s, r in constant.items())
            + " on the training data, so within-regime normalization is undefined (audit §E "
            "classed these sensors informative — contradiction flagged). Not adopted."
        )
        if fittable:
            informational["sensors_plus_fittable"] = (
                f"features.extra_sensors=[{', '.join(fittable)}]",
            )
            led.notes.append(
                f"DEVIATION (informational only, not eligible for adoption): "
                f"sensors_plus_fittable = [{', '.join(fittable)}] reported below."
            )
    else:
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
    for name, sets_i in informational.items():
        cfg_i = Config(f"{w2.tag}_{name}", f"{w2.label} + {name} (deviation)", (*w2.sets, *sets_i))
        _run(led, cfg_i)
        row = compare(led, "A3-informational", a3_inc, cfg_i)
        row["eligible"] = False
    winner: Config = a3_inc
    if len(adopted) >= 2:
        sets: list[str] = list(w2.sets)
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
        combined_name = "+".join(adopted_c)  # the blocks actually in the union
        comb = Config(f"{w2.tag}_combined", f"{w2.label} + {combined_name}", tuple(sets))
        _run(led, comb)
        r = compare(led, "A3-combined", a3_inc, comb)
        if r["better"]:
            winner = comb
        else:
            winner = min((a3[n] for n in adopted), key=lambda c: _headline(led, c))
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


# -- A1-check (addendum): cap 90 vs 125 where neither cap binds ------------------------------


def _mlflow_points(tag: str) -> pd.DataFrame:
    """Held-out predictions the FINISHED, valid MLflow parent run of selection run ``tag``
    logged (``heldout_predictions.csv``)."""
    import tempfile

    import mlflow

    from turbofan import tracking

    mlflow.set_tracking_uri(tracking._tracking_uri())
    exp = mlflow.get_experiment_by_name(tracking._experiment_name())
    if exp is None:
        raise RuntimeError("no MLflow experiment — the A1 runs must be on the tracking server")
    runs = mlflow.MlflowClient().search_runs(
        [exp.experiment_id],
        filter_string=(
            f"attributes.status = 'FINISHED' and tags.selection_tag = '{tag}' and "
            "tags.cv_role = 'parent'"
        ),
        order_by=["attributes.start_time DESC"],
    )
    valid = [r for r in runs if "invalid" not in r.data.tags]
    if not valid:
        raise RuntimeError(f"no FINISHED MLflow parent run for selection tag {tag}")
    with tempfile.TemporaryDirectory() as d:
        path = mlflow.MlflowClient().download_artifacts(
            valid[0].info.run_id, "heldout_predictions.csv", d
        )
        pts = pd.read_csv(path)
    pts.attrs["run_id"] = valid[0].info.run_id
    return pts


def cap_criteria(
    led: Ledger, substage: str, low: Candidate, high: Candidate
) -> tuple[bool, list[dict[str, Any]]]:
    """``low`` cap vs ``high`` cap on the metrics neither cap binds; returns (not worse, rows).
    Not worse = no urgent-RMSE / late-% Δ CI entirely above 0 and no decision-check Δ CI
    entirely below 0."""
    rows = [compare(led, substage, high, low, metric=m) for m in CAP_CHECK_METRICS]
    dc = decision_check(led, substage, high, low)
    worse = any(r["worse"] for r in rows) or any(r["diff_ci_hi"] < 0 for r in dc)
    return not worse, rows


def a1_check(datasets: list[str]) -> bool:
    """Writes ``reports/selection/a1_check.{md,json}``; True if cap 90 is confirmed on every
    dataset."""
    summary: dict[str, Any] = {"datasets": {}}
    lines = [
        "### A1-check — cap 90 vs cap 125 where neither cap binds (no new runs)",
        "",
        "XGBoost, base features, window 20, seed 42, 5 × 3 folds — the Stage A A1 runs, "
        "predictions read from their MLflow parent runs. Δ = cap 90 − cap 125; worse = Δ CI "
        "entirely on the bad side (above 0 for RMSE / late %, below 0 for caught %).",
        "",
        "| dataset | metric (uncapped truth) | cap 90 | cap 125 | Δ [95% CI] | n engines "
        "| verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    dc_lines = [
        "",
        "Decision check (caught % with lead ≥ 20 at matched mean wasted life):",
        "",
        "| dataset | budget (cycles) | cap 90 | cap 125 | Δ [95% CI] | n engines |",
        "|---|---|---|---|---|---|",
    ]
    info_lines = [
        "",
        "For information only (structurally favours the lower cap): overall late % vs uncapped "
        "truth, deployment view, per candidate.",
        "",
        "| dataset | cap 90 | cap 125 | n engines |",
        "|---|---|---|---|",
    ]
    confirmed = True
    for ds in datasets:
        led = Ledger(ds)
        cands: dict[int, Config] = {}
        run_ids: dict[int, str] = {}
        late: dict[int, str] = {}
        for cap in (90, 125):
            tag = f"A/{ds}/cap{cap:03d}"
            pts = _mlflow_points(tag)
            local = pd.read_parquet(SELECTION_DIR / tag / "points_xgboost.parquet")
            keys = ["unit", "cycle", "repeat", "fold", "seed"]
            a = pts.sort_values(keys).reset_index(drop=True)
            b = local.sort_values(keys).reset_index(drop=True)
            if not (
                a[keys].equals(b[keys]) and np.allclose(a["pred"], b["pred"], rtol=0, atol=1e-9)
            ):
                raise RuntimeError(f"{tag}: MLflow predictions differ from selection/{tag}")
            run_ids[cap] = str(pts.attrs["run_id"])
            led.configs[tag] = {"models": {"xgboost": {}}, "rul_cap": cap}
            cands[cap] = Config(tag, f"cap {cap}", ())
            t = metric_table(deployment_view(pts), cap, _rng("A1check-late", ds, str(cap)))
            lp = t[(t["truth"] == "uncapped") & (t["metric"] == "late_pct")].iloc[0]
            late[cap] = f"{lp['estimate']:.1f} [{lp['ci_lo']:.1f}, {lp['ci_hi']:.1f}]"
            n_eng = int(lp["n_engines"])
        ok, rows = cap_criteria(led, "A1-check", cands[90], cands[125])
        confirmed &= ok
        for r in rows:
            verdict = "worse" if r["worse"] else ("better" if r["better"] else "not different")
            lines.append(
                f"| {ds} | {r['metric']} | {r['estimate_challenger']:.2f} | "
                f"{r['estimate_incumbent']:.2f} | "
                f"{_ci(r['diff'], r['diff_ci_lo'], r['diff_ci_hi'])} | {r['n_engines']} | "
                f"{verdict} |"
            )
        for r in led.decisions:
            dc_lines.append(
                f"| {ds} | {r['budget']:g} | {r['caught_challenger']:.1f} | "
                f"{r['caught_incumbent']:.1f} | "
                f"{_ci(r['diff'], r['diff_ci_lo'], r['diff_ci_hi'], 1)} | {r['n_engines']} |"
            )
        info_lines.append(f"| {ds} | {late[90]} | {late[125]} | {n_eng} |")
        summary["datasets"][ds] = {
            "not_worse": ok,
            "comparisons": rows,
            "decisions": led.decisions,
            "mlflow_parent_runs": {str(k): v for k, v in run_ids.items()},
            "overall_late_pct_info": {str(k): v for k, v in late.items()},
        }
    summary["confirmed"] = confirmed
    verdict = (
        "**Cap 90 confirmed** — not worse on any pre-registered metric on any dataset."
        if confirmed
        else "**Cap 90 NOT confirmed** — worse on at least one metric; the addendum stops here."
    )
    lines += dc_lines + info_lines + ["", verdict]
    lines += [
        "",
        "MLflow parent runs: "
        + "; ".join(
            f"{ds} cap {c} `{rid}`"
            for ds, d in summary["datasets"].items()
            for c, rid in d["mlflow_parent_runs"].items()
        ),
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "a1_check.json").write_text(json.dumps(summary, indent=2, default=float) + "\n", "utf-8")
    (OUT / "a1_check.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return confirmed


# -- helpers shared by X / B / D ---------------------------------------------------------------


def block_screen(led: Ledger, substage: str, inc: Spec, names: list[str]) -> Spec:
    """Each block alone vs ``inc``; >= 2 adopted -> their union once (``<substage>-combined``),
    carried if it beats ``inc``, else the adopted block with the lowest estimate. The two health
    indices share their columns, so only the better of them can enter a union."""
    cands = {n: inc.plus(n) for n in names}
    for c in (inc, *cands.values()):
        _run(led, c)
    rows = [compare(led, substage, inc, c) for c in cands.values()]
    adopted = [n for n, r in zip(cands, rows, strict=True) if r["better"]]
    winner = inc
    if len(adopted) >= 2:
        hi = [n for n in adopted if n.startswith("hi_")]
        union = [n for n in adopted if not n.startswith("hi_")]
        if len(hi) == 2:
            best_hi = min(hi, key=lambda n: _headline(led, cands[n]))
            led.notes.append(
                f"{substage}-combined: both health indices adopted; {best_hi} (lower point "
                "estimate) enters the union, as the two share their columns"
            )
            union.append(best_hi)
        else:
            union += hi
        comb = inc.plus(*union)
        _run(led, comb)
        if len(union) >= 2:
            r = compare(led, f"{substage}-combined", inc, comb)
            winner = (
                comb
                if r["better"]
                else min((cands[n] for n in adopted), key=lambda c: _headline(led, c))
            )
        else:
            winner = comb
    elif adopted:
        winner = cands[adopted[0]]
    led.winners[substage] = winner.label
    if winner != inc:
        decision_check(led, substage, inc, winner)
    return winner


def _spec_from_stage_a(dataset: str) -> Spec:
    """The Stage A adopted configuration, read back from its run's overrides."""
    doc = json.loads((OUT / f"stage_A_{dataset}.json").read_text(encoding="utf-8"))
    tag = doc["winners"]["stage_A_tag"]
    over = yaml.safe_load((SELECTION_DIR / tag / "overrides.yaml").read_text(encoding="utf-8"))
    feats = over.get("features", {})
    blocks: list[str] = []
    if feats.get("health_index", "none") != "none":
        blocks.append(f"hi_{feats['health_index']}")
    if feats.get("regime_onehot"):
        blocks.append("regime_onehot")
    if feats.get("extra_sensors"):
        if list(feats["extra_sensors"]) != list(SEL["extra_sensors"][dataset]):
            raise ValueError(f"{tag}: extra sensors {feats['extra_sensors']} are not sensors_plus")
        blocks.append("sensors_plus")
    return Spec(
        dataset,
        "xgboost",
        int(over.get("rul_cap", DEFAULT_CAP)),
        int(over.get("window", WINDOW)),
        blocks=tuple(blocks),
    )


def _load(stage: str, dataset: str) -> dict[str, Any]:
    p = OUT / f"stage_{stage}_{dataset}.json"
    if not p.is_file():
        raise FileNotFoundError(f"{p} missing — run stage {stage} for {dataset} first")
    doc: dict[str, Any] = json.loads(p.read_text(encoding="utf-8"))
    return doc


def _spec(doc: dict[str, Any], model: str) -> Spec:
    return Spec.from_json(doc["specs"][model])


# -- Stage X (addendum) ------------------------------------------------------------------------


def _a4_gate(doc: dict[str, Any]) -> tuple[bool, str]:
    from turbofan.config import FEATURE_BLOCKS

    n = FEATURE_BLOCKS.subpop_prob_cycles
    real = {int(r["n_cycles"]): r for r in doc["results"]}
    ctrl = {int(r["n_cycles"]): r for r in doc["shuffled_control"]}
    if n not in real:
        raise ValueError(f"A4 has no result at N = {n} (features.subpop_prob_cycles)")
    r, c = real[n], ctrl[n]
    ok = (
        r["auc"] >= float(SEL["a5b_gate_auc"])
        and r["ci_lo"] >= float(SEL["identifiability"]["auc_strong_lower"])
        and c["ci_lo"] <= 0.5 <= c["ci_hi"]
    )
    why = (
        f"A4 at N = {n}: AUC {_ci(r['auc'], r['ci_lo'], r['ci_hi'], 3)}, shuffled-label control "
        f"{_ci(c['auc'], c['ci_lo'], c['ci_hi'], 3)} -> gate {'passed' if ok else 'failed'}"
    )
    return ok, why


def stage_x(dataset: str) -> Ledger:
    check = OUT / "a1_check.json"
    if not check.is_file() or not json.loads(check.read_text(encoding="utf-8"))["confirmed"]:
        raise RuntimeError("the A1-check has not confirmed cap 90 — the addendum does not run")
    led = Ledger(dataset)
    a_spec = _spec_from_stage_a(dataset)
    led.notes.append(
        f"Stage A adopted spec (re-run under current code where compared): {a_spec.label}"
    )

    # A2-ext — windows beyond the A2 winner, base features
    base = a_spec.with_(blocks=())
    ext = [base.with_(window=int(w)) for w in SEL["window_ext_grid"]]
    w2 = pick(led, "A2-ext", base, list(ext))
    assert isinstance(w2, Spec)
    if w2.window == max(int(w) for w in SEL["window_ext_grid"]):
        led.notes.append(
            f"A2-ext: window {w2.window} (the largest tested) won — carried forward; edge "
            "limitation, no further extension (pre-registered stopping rule)"
        )
    if w2.window == base.window:
        spec1 = a_spec
        _run(led, spec1)
    else:
        names = ["hi_consistent", "hi_pooled"]
        if N_REGIMES[dataset] > 1:
            names.append("regime_onehot")
        _fittable, constant = fittable_extra_sensors(dataset, list(SEL["extra_sensors"][dataset]))
        if not constant:
            names.append("sensors_plus")
        spec1 = block_screen(led, "A3-rerun", w2, names)

    # A5a — baseline deviation
    spec2 = pick(led, "A5a", spec1, [spec1.plus("baseline")])
    assert isinstance(spec2, Spec)

    # A4 re-run with the label-shuffle control
    a4 = Config(
        f"{RUN_PREFIX}/{dataset}/A4",
        "A4 identifiability + label-shuffle control",
        (f"pipeline.dataset={dataset}",),
        "identifiability",
    )
    led.configs[a4.tag] = ensure(a4)
    gate, why = _a4_gate(led.configs[a4.tag])
    led.extra["a4_gate"] = {"passed": gate, "detail": why}

    # A5b — subpopulation probability (scope and gate pre-registered)
    spec3 = spec2
    if dataset in SEL["a5b_datasets"]:
        led.notes.append(why)
        if gate:
            got = pick(led, "A5b", spec2, [spec2.plus("subpop_prob")])
            assert isinstance(got, Spec)
            spec3 = got
        else:
            led.notes.append("A5b not run: the pre-registered A4 gate failed")
    led.winners["stage_X"] = spec3.label
    led.specs["xgboost"] = spec3.to_json()
    return led


# -- Stage B (LSTM) ----------------------------------------------------------------------------


def stage_b(dataset: str) -> Ledger:
    led = Ledger(dataset)
    xgb = _spec(_load("X", dataset), "xgboost")
    base = Spec(dataset, "lstm", xgb.rul_cap, xgb.window, seq_len=SEQ_LEN)
    grid = [int(x) for x in PARAMS["seq_len_grid"]]
    if SEQ_LEN not in grid:
        raise ValueError(f"seq_len {SEQ_LEN} is not in seq_len_grid {grid}")
    b1 = pick(led, "B1", base, [base.with_(seq_len=x) for x in grid if x != SEQ_LEN])
    assert isinstance(b1, Spec)
    if b1.seq_len == max(grid):
        led.notes.append(
            f"B1: seq_len {b1.seq_len} is the grid's upper edge (flagged; no extension)"
        )
    names = ["regime_onehot"] if N_REGIMES[dataset] > 1 else []
    names += [b for b in xgb.blocks if b.startswith("hi_")]
    if "baseline" in xgb.blocks:
        names.append("baseline")
    b2 = block_screen(led, "B2", b1, names) if names else b1
    led.winners["stage_B"] = b2.label
    led.specs["xgboost"] = xgb.to_json()
    led.specs["lstm"] = b2.to_json()
    return led


# -- Stage C (tuning) --------------------------------------------------------------------------


def _distributions(space: dict[str, Any]) -> dict[str, Any]:
    import optuna.distributions as od

    out: dict[str, Any] = {}
    for k, v in space.items():
        if v["type"] == "int":
            out[k] = od.IntDistribution(int(v["low"]), int(v["high"]))
        elif v["type"] == "float":
            out[k] = od.FloatDistribution(float(v["low"]), float(v["high"]), log=bool(v.get("log")))
        elif v["type"] == "categorical":
            out[k] = od.CategoricalDistribution(list(v["choices"]))
        else:
            raise ValueError(f"tuning space {k}: unknown type {v['type']!r}")
    return out


def tune(led: Ledger, spec: Spec) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Optuna TPE on the repeat-1 folds; every trial is one pipeline run. A restart replays the
    study deterministically (same sampler seed and history) and reuses finished trial runs."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    cfg = PARAMS["tuning"]
    mcfg = cfg[spec.model]
    fixed = dict(mcfg["fixed"])
    dists = _distributions(mcfg["space"])
    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=int(cfg["sampler_seed"]))
    )
    trials = []
    for i in range(int(mcfg["n_trials"])):
        trial = study.ask(dists)
        params = {**fixed, **trial.params}
        tspec = spec.with_(tuned=tuple(params.items()), n_repeats=1, n_boot=int(cfg["n_boot"]))
        _run(led, tspec)
        value = _headline(led, tspec)
        study.tell(trial, value)
        trials.append({"trial": i, "params": params, HEADLINE: value, "tag": tspec.tag})
    best = {**fixed, **study.best_trial.params}
    return best, trials


def stage_c(dataset: str) -> Ledger:
    led = Ledger(dataset)
    prev = _load("B", dataset)
    repeats = tuple(range(1, int(PARAMS["cv"]["n_repeats"])))  # repeats 2..n (0-based 1..)
    for model in ("xgboost", "lstm"):
        spec = _spec(prev, model)
        best, trials = tune(led, spec)
        tuned = spec.with_(tuned=tuple(best.items()))
        for s in (spec, tuned):
            _run(led, s)
        r = compare(led, f"C-{model}", spec, tuned, repeats=repeats)
        final = tuned if r["better"] else spec
        if final is tuned:
            decision_check(led, f"C-{model}", spec, tuned, repeats=repeats)
        led.winners[f"C-{model}"] = final.label
        led.specs[model] = final.to_json()
        led.extra.setdefault("tuning", {})[model] = {"best": best, "trials": trials}
        led.sections.append(_render_trials(model, trials))
    return led


# -- Stage D (confirmation) --------------------------------------------------------------------


def adopted_changes(fin: Spec) -> list[tuple[str, Spec]]:
    """Every change of ``fin`` from the original defaults, with ``fin`` minus that change."""
    out: list[tuple[str, Spec]] = []
    if fin.rul_cap != DEFAULT_CAP:
        out.append(("cap", fin.with_(rul_cap=DEFAULT_CAP)))
    if fin.model == "xgboost" and fin.window != WINDOW:
        out.append(("window", fin.with_(window=WINDOW)))
    if fin.model == "lstm" and fin.seq_len != SEQ_LEN:
        out.append(("seq_len", fin.with_(seq_len=SEQ_LEN)))
    out += [(b, fin.minus(b)) for b in fin.blocks]
    if fin.tuned:
        out.append(("tuning", fin.with_(tuned=())))
    return out


def _decision_better(rows: list[dict[str, Any]], sign: int) -> bool:
    """``sign`` = +1: the challenger is better on the decision check (Δ CI > 0 at >= 1 budget,
    < 0 at none); -1: the incumbent is."""
    if sign > 0:
        return any(r["diff_ci_lo"] > 0 for r in rows) and not any(r["diff_ci_hi"] < 0 for r in rows)
    return any(r["diff_ci_hi"] < 0 for r in rows) and not any(r["diff_ci_lo"] > 0 for r in rows)


def confirm(led: Ledger, fin: Spec) -> Spec:
    """Leave-one-out parsimony check of ``fin`` (5 seeds); returns the confirmed spec."""
    changes = adopted_changes(fin)
    for s in (fin, *(m for _, m in changes)):
        _run(led, s)
    sub = f"D-LOO-{fin.model}"
    dropped = []
    for name, minus in changes:
        r = compare(led, sub, minus, fin)
        keep = bool(r["better"])
        if name == "cap":
            ok, _rows = cap_criteria(led, f"D-cap-{fin.model}", fin, minus)
            if keep and not ok:
                led.notes.append(
                    f"{fin.model}: cap 90 beats 125 on critical RMSE but is worse on "
                    "a cap-check metric at 5 seeds — dropped"
                )
            keep = keep and ok
        if not keep:
            dropped.append(name)
    led.extra.setdefault("loo", {})[fin.model] = {
        "changes": [n for n, _ in changes],
        "dropped": dropped,
    }
    if not dropped:
        return fin
    minus_of = dict(changes)
    if len(dropped) == 1:
        led.notes.append(f"{fin.model}: dropped {dropped[0]} (parsimony)")
        return minus_of[dropped[0]]
    reduced = fin
    for name in dropped:
        reduced = _revert(reduced, name)
    _run(led, reduced)
    r = compare(led, f"D-reduced-{fin.model}", reduced, fin)
    if r["better"]:
        led.notes.append(
            f"{fin.model}: dropping {dropped} together is worse than the full spec — an "
            "interaction; the full spec stays"
        )
        return fin
    led.notes.append(f"{fin.model}: dropped {dropped} (parsimony)")
    return reduced


def _revert(spec: Spec, name: str) -> Spec:
    if name == "cap":
        return spec.with_(rul_cap=DEFAULT_CAP)
    if name == "window":
        return spec.with_(window=WINDOW)
    if name == "seq_len":
        return spec.with_(seq_len=SEQ_LEN)
    if name == "tuning":
        return spec.with_(tuned=())
    return spec.minus(name)


def write_spec(spec: Spec, locked: bool, status: str, led: Ledger) -> Path:
    doc = led.configs[spec.tag]
    m = doc["models"][spec.model]
    feats = {
        "health_index": next((b[3:] for b in spec.blocks if b.startswith("hi_")), "none"),
        "regime_onehot": "regime_onehot" in spec.blocks,
        "extra_sensors": list(SEL["extra_sensors"][spec.dataset])
        if "sensors_plus" in spec.blocks
        else [],
        "baseline": "baseline" in spec.blocks,
        "subpop_prob": "subpop_prob" in spec.blocks,
    }
    out = {
        "dataset": spec.dataset,
        "model": spec.model,
        "seed": SEED,
        "rul_cap": spec.rul_cap,
        "locked": locked,
        "status": status,
        "window": spec.window,
        "seq_len": spec.seq_len,
        "features": feats,
        "model_params": dict(spec.tuned),
        "evidence": {
            "selection_tag": spec.tag,
            HEADLINE: m[HEADLINE],
            f"{HEADLINE}_ci": [m[f"{HEADLINE}_ci_lo"], m[f"{HEADLINE}_ci_hi"]],
            "n_engines": m["n_engines"],
            "n_splits": m["n_splits"],
            "seeds": list(spec.seeds),
            "report": f"reports/selection/stage_D_{spec.dataset}.md",
        },
    }
    SPECS_DIR.mkdir(exist_ok=True)
    path = SPECS_DIR / f"{spec.dataset}_{spec.model}.yaml"
    path.write_text(yaml.safe_dump(out, sort_keys=False), encoding="utf-8")
    return path


def stage_d(dataset: str) -> Ledger:
    led = Ledger(dataset)
    prev = _load("C", dataset)
    seeds = tuple(int(s) for s in SEEDS)
    finals = {m: confirm(led, _spec(prev, m).with_(seeds=seeds)) for m in ("xgboost", "lstm")}
    xgb, lstm = finals["xgboost"], finals["lstm"]
    for s in finals.values():
        _run(led, s)
    r = compare(led, "D-final", xgb, lstm)  # Δ = LSTM − XGBoost
    dc = decision_check(led, "D-final", xgb, lstm)  # Δ caught = LSTM − XGBoost
    if r["better"]:
        winner, loser, locked = lstm, xgb, _decision_better(dc, +1)
        why = "LSTM better on critical RMSE"
    elif r["worse"]:
        winner, loser, locked = xgb, lstm, _decision_better(dc, -1)
        why = "XGBoost better on critical RMSE"
    else:
        winner, loser, locked = xgb, lstm, not _decision_better(dc, +1)
        why = "tie on critical RMSE: XGBoost (simpler) wins"
    why += "; " + (
        "decision check agrees -> locked"
        if locked
        else "decision check does not support a single winner -> both locked-pending"
    )
    led.winners["stage_D"] = winner.label
    led.extra["outcome"] = {
        "winner": winner.model,
        "locked": locked,
        "reason": why,
        "finalists": {m: s.to_json() for m, s in finals.items()},
    }
    for m, s in finals.items():
        led.specs[m] = s.to_json()
    if locked:
        write_spec(winner, True, "locked", led)
        write_spec(loser, False, "runner-up", led)
    else:
        for s in finals.values():
            write_spec(s, False, "locked-pending", led)
    led.notes.append(why)
    return led


# -- rendering ---------------------------------------------------------------------------------


def _ci(est: float, lo: float, hi: float, p: int = 2) -> str:
    return f"{est:.{p}f} [{lo:.{p}f}, {hi:.{p}f}]"


def _render_trials(model: str, trials: list[dict[str, Any]]) -> str:
    keys = sorted({k for t in trials for k in t["params"]})
    lines = [
        f"Optuna trials — {model} (objective: critical RMSE over the 5 repeat-1 splits, seed 42):",
        "",
        "| trial | " + " | ".join(keys) + " | critical RMSE |",
        "|---" * (len(keys) + 2) + "|",
    ]
    for t in trials:
        vals = [t["params"][k] for k in keys]
        lines.append(
            f"| {t['trial']} | "
            + " | ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in vals)
            + f" | {t[HEADLINE]:.3f} |"
        )
    return "\n".join(lines)


def _render_a4(a4: dict[str, Any]) -> list[str]:
    lines = [
        "",
        f"A4 — subpopulation from the first N cycles (logistic regression, AUC [95% CI], "
        f"strong = CI lower bound ≥ {a4['strong_threshold_ci_lo']:.2f}; subpopulation sizes "
        f"{a4['subpopulation_sizes']}):",
        "",
    ]
    ctrl = {int(r["n_cycles"]): r for r in a4.get("shuffled_control", [])}
    head = "| N cycles | AUC [95% CI] | per repeat | n engines | strong |"
    if ctrl:
        head += " shuffled-label AUC [95% CI] | control holds 0.5 |"
    lines += [head, "|---" * (head.count("|") - 1) + "|"]
    for r in a4["results"]:
        per = ", ".join(f"{x:.3f}" for x in r["auc_per_repeat"])
        row = (
            f"| {r['n_cycles']} | {_ci(r['auc'], r['ci_lo'], r['ci_hi'], 3)} | {per} | "
            f"{r['n_engines']} | {'yes' if r['strong'] else 'no'} |"
        )
        c = ctrl.get(int(r["n_cycles"]))
        if c:
            row += (
                f" {_ci(c['auc'], c['ci_lo'], c['ci_hi'], 3)} | "
                f"{'yes' if c['holds_chance'] else 'NO'} |"
            )
        lines.append(row)
    return lines


def render(led: Ledger, stage: str = "A") -> str:
    ds = led.dataset
    lines = [f"### Stage {stage} — {ds}", ""]
    lines += [
        "| config | critical RMSE [95% CI] | n engines | n splits | subpop 0 | subpop 1 "
        "| wall (min) |",
        "|---|---|---|---|---|---|---|",
    ]
    for tag, doc in led.configs.items():
        if "models" not in doc or "subpopulations" not in doc:
            continue
        model = _model_of(doc)
        m = doc["models"][model]
        sp = doc["subpopulations"][model]
        lines.append(
            f"| `{tag}` | {_ci(m[HEADLINE], m[f'{HEADLINE}_ci_lo'], m[f'{HEADLINE}_ci_hi'])} | "
            f"{m['n_engines']} | {m.get('n_splits', '')} | "
            + " | ".join(
                f"{_ci(sp[g]['estimate'], sp[g]['ci_lo'], sp[g]['ci_hi'])} (n={sp[g]['n_engines']})"
                for g in ("subpop_0", "subpop_1")
            )
            + f" | {doc['wall_seconds'] / 60:.1f} |"
        )
    lines += [
        "",
        "Paired comparisons (Δ = challenger − incumbent; better = CI entirely < 0; metric = "
        "critical RMSE unless stated):",
        "",
        "| step | challenger | incumbent | Δ [95% CI] | engines challenger better "
        "| Wilcoxon p (n splits) | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in led.comparisons:
        verdict = "**better**" if r["better"] else ("worse" if r["worse"] else "not different")
        if r.get("eligible") is False:
            verdict += " (informational, not eligible)"
        step = r["substage"]
        if r.get("metric", HEADLINE) != HEADLINE:
            step += f" ({r['metric']})"
        if "repeats" in r:
            step += f" (repeats {', '.join(str(x + 1) for x in r['repeats'])})"
        lines.append(
            f"| {step} | {r['challenger']} | {r['incumbent']} | "
            f"{_ci(r['diff'], r['diff_ci_lo'], r['diff_ci_hi'])} | "
            f"{r['share_engines_challenger_better']:.0%} of {r['n_engines']} | "
            f"{r['wilcoxon_p']:.2g} ({r['wilcoxon_n_splits']}) | {verdict} |"
        )
    if led.decisions:
        lines += [
            "",
            "Decision checks (caught % with lead ≥ 20 at matched mean wasted life; Δ = "
            "challenger − incumbent, positive = challenger catches more):",
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
    for doc in led.configs.values():
        if "results" in doc and "strong_threshold_ci_lo" in doc:
            lines += _render_a4(doc)
    for sec in led.sections:
        lines += ["", sec]
    shown = {k: v for k, v in led.winners.items() if not k.endswith("_tag")}
    lines += ["", "Winners: " + "; ".join(f"{k}: {v}" for k, v in shown.items())]
    for n in led.notes:
        lines.append(f"- {n}")
    return "\n".join(lines) + "\n"


def write(led: Ledger, stage: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"stage_{stage}_{led.dataset}"
    doc = {
        "dataset": led.dataset,
        "winners": led.winners,
        "specs": led.specs,
        "comparisons": led.comparisons,
        "decisions": led.decisions,
        "notes": led.notes,
        "extra": led.extra,
        "configs": led.configs,
    }
    stem.with_suffix(".json").write_text(json.dumps(doc, indent=2, default=float) + "\n", "utf-8")
    stem.with_suffix(".md").write_text(render(led, stage), encoding="utf-8")
    return stem


STAGES = {"A": stage_a, "X": stage_x, "B": stage_b, "C": stage_c, "D": stage_d}


def main(stage: str, datasets: list[str]) -> None:
    if stage == "A1check":
        ok = a1_check(datasets)
        print(f"[select] wrote {OUT / 'a1_check.md'}", flush=True)
        if not ok:
            raise SystemExit("A1-check failed: cap 90 not confirmed — the addendum stops here")
        return
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    for ds in datasets:
        led = STAGES[stage](ds)
        print(f"[select] wrote {write(led, stage)}", flush=True)
