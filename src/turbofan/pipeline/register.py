"""Stage ``register``: log the production bundle as an MLflow pyfunc model and register it.

The version gets the ``challenger`` alias. Promotion to ``champion`` is deliberately not a
pipeline step: it follows a ``final_eval`` result a human has read.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from turbofan import tracking
from turbofan.config import RUL_CAP
from turbofan.pipeline.context import Context, write_json

CHALLENGER: Literal["challenger"] = "challenger"


def run(ctx: Context) -> dict[str, Any]:
    if not ctx.train_metrics.is_file():
        raise FileNotFoundError(f"{ctx.train_metrics} missing — run the train_prod stage first")
    trained = json.loads(ctx.train_metrics.read_text(encoding="utf-8"))
    bundle = ctx.root / trained["bundle"]
    if not bundle.is_dir():
        raise FileNotFoundError(f"bundle {bundle} missing — run the train_prod stage first")
    ds = trained["dataset"]
    with tracking.run(
        dataset=ds,
        model=trained["model"],
        seed=int(trained["seed"]),
        run_type="register",
        extra_params={"rul_cap": RUL_CAP, "alias": CHALLENGER},
        extra_tags={"spec_hash": trained["spec_hash"]},
    ) as active:
        info = tracking.log_production_model(bundle, ds)
        mv = tracking.register_production_model(info.model_uri, ds, alias=CHALLENGER)
        doc: dict[str, Any] = {
            "name": mv.name,
            "version": int(mv.version),
            "alias": CHALLENGER,
            "model_uri": info.model_uri,
            "mlflow_run_id": active.info.run_id if active is not None else None,
        }
    write_json(ctx.registered, doc)
    return doc
