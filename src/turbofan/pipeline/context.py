"""What a pipeline stage runs with: parameters, inputs and where its outputs go.

``build_context(smoke)`` reads ``turbofan.config.PARAMS``. Under ``--smoke`` the entry point
(``__main__``) has already pointed ``TURBOFAN_PARAMS`` at a merged copy of ``params.yaml`` before
anything imported ``turbofan.config``, so every module-level default (epochs, trees, folds,
bootstrap replicates) is the smoke value; outputs go under ``smoke/`` instead of the repo root.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from turbofan.config import DATASETS, PARAMS

__all__ = ["REPO", "SMOKE_DIR", "Context", "build_context", "select_smoke_engines", "write_json"]

REPO = Path(__file__).resolve().parents[3]
SMOKE_DIR = REPO / "smoke"
KNOWN_MODELS = ("mean", "ridge", "rf", "xgboost", "lstm")


def _req(section: str, key: str) -> Any:
    try:
        return PARAMS[section][key]
    except KeyError:
        raise KeyError(f"params.yaml: missing key '{section}.{key}'") from None


@dataclass(frozen=True)
class Context:
    smoke: bool
    root: Path  # outputs are written under here
    raw: Path
    dataset: str
    models: list[str]
    seeds: list[int]
    resume: bool
    n_folds: int
    n_repeats: int
    max_folds: int | None
    n_engines: int | None
    spec: Path
    final_eval_enabled: bool
    tag: str | None = None  # a model-selection run (D51): outputs under selection/<tag>/

    # -- outputs, relative to ``root`` ----------------------------------------------------
    @property
    def validate_metrics(self) -> Path:
        return self.root / "reports" / "validate.json"

    @property
    def report(self) -> Path:
        return self.root / "reports" / "protocol_v2.md"

    @property
    def fig_dir(self) -> Path:
        return self.root / "reports" / "figures" / "protocol_v2"

    @property
    def cv_metrics(self) -> Path:
        return self.root / "reports" / "cv_select" / "metrics.json"

    @property
    def cv_plots(self) -> Path:
        return self.root / "reports" / "cv_select" / "plots"

    def points(self, model: str) -> Path:
        """A selection run's held-out predictions at every cycle (all folds x seeds)."""
        return self.root / f"points_{model}.parquet"

    @property
    def env_file(self) -> Path:
        return self.root / "env.json"

    @property
    def engines_file(self) -> Path:
        return self.root / "engines.json"

    @property
    def models_prod(self) -> Path:
        return self.root / "models" / "prod"

    @property
    def train_metrics(self) -> Path:
        return self.root / "reports" / "train_prod" / "metrics.json"

    @property
    def final_metrics(self) -> Path:
        return self.root / "reports" / "final_eval" / "metrics.json"

    @property
    def registered(self) -> Path:
        return self.root / "reports" / "register" / "registered.json"


def build_context(smoke: bool, tag: str | None = None) -> Context:
    unknown = [
        m for m in _req("pipeline", "models") if m not in ("mean", "ridge", "rf", "xgboost", "lstm")
    ]
    if unknown:
        raise ValueError(f"params.yaml: pipeline.models has unknown models {unknown}")
    dataset = _req("pipeline", "dataset")
    if dataset not in DATASETS:
        raise ValueError(f"params.yaml: pipeline.dataset {dataset!r} not in {DATASETS}")
    seeds = [int(s) for s in _req("pipeline", "seeds")]
    if not set(seeds) <= set(PARAMS["seeds"]):
        raise ValueError(f"params.yaml: pipeline.seeds {seeds} must be within `seeds`")
    enabled = _req("final_eval", "enabled")
    if not isinstance(enabled, bool):
        raise ValueError(f"params.yaml: final_eval.enabled must be true/false, got {enabled!r}")
    if smoke and tag:
        raise ValueError("a run is either a smoke run or a selection run, not both")
    root = SMOKE_DIR if smoke else (REPO / "selection" / tag if tag else REPO)
    return Context(
        smoke=smoke,
        root=root,
        raw=REPO / "data" / "raw",
        dataset=dataset,
        models=list(_req("pipeline", "models")),
        seeds=seeds,
        resume=bool(_req("pipeline", "resume")),
        n_folds=int(PARAMS["cv"]["n_folds"]),
        n_repeats=int(PARAMS["cv"]["n_repeats"]),
        max_folds=int(_req("smoke", "max_folds")) if smoke else None,
        n_engines=int(_req("smoke", "n_engines")) if smoke else None,
        spec=(root / "locked.yaml") if smoke else REPO / str(_req("release", "spec")),
        final_eval_enabled=enabled,
        tag=tag,
    )


def select_smoke_engines(labels: pd.Series[int], n: int) -> list[int]:
    """``n`` engines, balanced over the subpopulation labels, lowest ids first within each —
    deterministic, and every label keeps at least two engines so stratified folds can form."""
    groups = [sorted(g.index.tolist()) for _, g in labels.groupby(labels)]
    per = n // len(groups)
    picked: list[int] = []
    for g in groups:
        picked += g[:per]
    spare = sorted(set(labels.index) - set(picked))
    picked += spare[: n - len(picked)]  # remainder / small groups: fill by id
    chosen = sorted(int(u) for u in picked)
    counts = labels.loc[chosen].value_counts()
    if len(counts) < 2 or counts.min() < 2:
        raise ValueError(
            f"smoke engine selection {chosen} leaves a subpopulation with < 2 engines "
            f"({counts.to_dict()}); raise smoke.n_engines"
        )
    return chosen


def write_json(path: Path, doc: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
