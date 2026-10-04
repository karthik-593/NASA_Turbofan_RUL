"""Stage ``identifiability`` (D51, A4): can the subpopulation be told from the first N cycles?

    python -m turbofan.pipeline identifiability --tag A/FD001/A4 --set pipeline.dataset=FD001

Writes ``selection/<tag>/reports/cv_select/metrics.json`` (AUC per N with CI and n, and whether it
clears the pre-registered "strong" bar) and logs one MLflow run. Training files only.

Label-shuffle control (post-Stage-A addendum): the identical procedure on the same folds with the
labels permuted across engines (seed ``selection.seed``). Its AUC CI should hold 0.5; one that
lies above it would mean the pipeline leaks the label.
"""

from __future__ import annotations

from typing import Any

import mlflow
import numpy as np
import pandas as pd

from turbofan import tracking
from turbofan.analysis.identifiability import identifiability
from turbofan.analysis.subpopulation import subpopulation_labels
from turbofan.config import PARAMS
from turbofan.data.loader import load_train
from turbofan.evaluation.cv import make_folds
from turbofan.pipeline.context import Context, write_json


def run(ctx: Context) -> dict[str, Any]:
    if not ctx.tag:
        raise ValueError("identifiability is a model-selection analysis: run it with --tag")
    cfg = PARAMS["selection"]["identifiability"]
    seed = int(PARAMS["selection"]["seed"])
    train = load_train(ctx.dataset, ctx.raw)
    labels = subpopulation_labels(train, ctx.dataset)
    folds = make_folds(
        labels, n_folds=ctx.n_folds, n_repeats=ctx.n_repeats, seed=PARAMS["cv"]["seed"]
    )
    with tracking.run(
        dataset=ctx.dataset,
        model="logreg_subpopulation",
        seed=seed,
        run_type="cv",
        extra_params={"n_cycles": cfg["n_cycles"], "logreg_c": cfg["logreg_c"]},
        extra_tags={"analysis": "identifiability"},
    ) as active:

        def run_on(y: pd.Series[int]) -> pd.DataFrame:
            return identifiability(
                train,
                y,
                ctx.dataset,
                folds,
                [int(n) for n in cfg["n_cycles"]],
                float(cfg["logreg_c"]),
                np.random.default_rng(seed),
                int(PARAMS["bootstrap"]["n_boot"]),
                float(PARAMS["bootstrap"]["ci_level"]),
            )

        table = run_on(labels)
        perm = np.random.default_rng(seed).permutation(labels.to_numpy())
        control = run_on(pd.Series(perm, index=labels.index, name=labels.name))
        strong = float(cfg["auc_strong_lower"])
        table["strong"] = table["ci_lo"] >= strong
        control["holds_chance"] = (control["ci_lo"] <= 0.5) & (control["ci_hi"] >= 0.5)
        if active is not None:
            for kind, t in (("", table), ("_shuffled", control)):
                for r in t.to_dict("records"):
                    n = int(r["n_cycles"])
                    mlflow.log_metrics(
                        {
                            f"auc_n{n}{kind}": float(r["auc"]),
                            f"auc_n{n}{kind}_ci_lo": float(r["ci_lo"]),
                            f"auc_n{n}{kind}_ci_hi": float(r["ci_hi"]),
                        }
                    )
    doc = {
        "tag": ctx.tag,
        "dataset": ctx.dataset,
        "subpopulation_sizes": labels.value_counts().sort_index().astype(int).to_dict(),
        "strong_threshold_ci_lo": strong,
        "results": table.to_dict("records"),
        "shuffled_control": control.to_dict("records"),
    }
    write_json(ctx.cv_metrics, doc)
    return doc
