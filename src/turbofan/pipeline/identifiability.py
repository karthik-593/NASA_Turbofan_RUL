"""Stage ``identifiability`` (D51, A4): can the subpopulation be told from the first N cycles?

    python -m turbofan.pipeline identifiability --tag A/FD001/A4 --set pipeline.dataset=FD001

Writes ``selection/<tag>/reports/cv_select/metrics.json`` (AUC per N with CI and n, and whether it
clears the pre-registered "strong" bar) and logs one MLflow run. Training files only.
"""

from __future__ import annotations

from typing import Any

import mlflow
import numpy as np

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
        table = identifiability(
            train,
            labels,
            ctx.dataset,
            folds,
            [int(n) for n in cfg["n_cycles"]],
            float(cfg["logreg_c"]),
            np.random.default_rng(seed),
            int(PARAMS["bootstrap"]["n_boot"]),
            float(PARAMS["bootstrap"]["ci_level"]),
        )
        strong = float(cfg["auc_strong_lower"])
        table["strong"] = table["ci_lo"] >= strong
        if active is not None:
            for r in table.to_dict("records"):
                n = int(r["n_cycles"])
                mlflow.log_metrics(
                    {
                        f"auc_n{n}": float(r["auc"]),
                        f"auc_n{n}_ci_lo": float(r["ci_lo"]),
                        f"auc_n{n}_ci_hi": float(r["ci_hi"]),
                    }
                )
    doc = {
        "tag": ctx.tag,
        "dataset": ctx.dataset,
        "subpopulation_sizes": labels.value_counts().sort_index().astype(int).to_dict(),
        "strong_threshold_ci_lo": strong,
        "results": table.to_dict("records"),
    }
    write_json(ctx.cv_metrics, doc)
    return doc
