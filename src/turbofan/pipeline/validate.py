"""Stage ``validate``: every training file passes the raw-data schema checks.

Opens ``train_FD00x.txt`` only (``load_train``): the sealed test files and RUL labels are not
read here (rule 3), so their integrity is not checked by this stage. Writes per-dataset counts
as DVC metrics.
"""

from __future__ import annotations

from typing import Any

from turbofan.config import DATASETS
from turbofan.data.loader import load_train
from turbofan.pipeline.context import Context, write_json


def run(ctx: Context) -> dict[str, Any]:
    doc: dict[str, Any] = {}
    for ds in DATASETS:
        train = load_train(ds, ctx.raw)  # raises on a schema violation
        life = train.groupby("unit")["cycle"].max()
        doc[ds] = {
            "n_engines": int(life.size),
            "n_rows": int(len(train)),
            "life_min": int(life.min()),
            "life_median": float(life.median()),
            "life_max": int(life.max()),
        }
    write_json(ctx.validate_metrics, doc)
    return doc
