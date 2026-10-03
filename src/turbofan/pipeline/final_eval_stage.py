"""Stage ``final_eval``: the sealed test read, only when ``final_eval.enabled`` is true.

Disabled (the default) the stage writes ``{"ran": false}`` and touches no test file. Enabled it
runs ``python -m turbofan.evaluation.final_eval`` in a subprocess — this package may not import
that module (tests/test_final_eval.py) — and that module still enforces everything it always
did: ``--confirm``, a ``locked: true`` spec, one run per spec hash, ``run_type=final_test``. This
stage passes ``--confirm`` on only if it was itself confirmed for this invocation: its own
``--confirm`` flag, or ``TURBOFAN_CONFIRM_FINAL_EVAL=1`` in the environment (what to use under
``dvc repro``, whose command line lives in a committed file that must not carry the confirmation).

A smoke run never reads the test set whatever the parameter says.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any

from turbofan.pipeline.context import Context, write_json

__all__ = ["CONFIRM_ENV", "run"]

CONFIRM_ENV = "TURBOFAN_CONFIRM_FINAL_EVAL"


def run(ctx: Context, confirm: bool) -> dict[str, Any]:
    if ctx.smoke or not ctx.final_eval_enabled:
        reason = (
            "smoke runs never read the test set"
            if ctx.smoke
            else "final_eval.enabled is false in params.yaml"
        )
        doc: dict[str, Any] = {"ran": False, "reason": reason}
        write_json(ctx.final_metrics, doc)
        return doc
    if not (confirm or os.environ.get(CONFIRM_ENV) == "1"):
        raise RuntimeError(
            f"final_eval.enabled is true but this invocation is not confirmed (pass --confirm or "
            f"set {CONFIRM_ENV}=1); the sealed test set is read only on explicit confirmation"
        )
    if not ctx.train_metrics.is_file():
        raise FileNotFoundError(f"{ctx.train_metrics} missing — run the train_prod stage first")
    bundle = ctx.root / json.loads(ctx.train_metrics.read_text(encoding="utf-8"))["bundle"]
    cmd = [
        sys.executable,
        "-m",
        "turbofan.evaluation.final_eval",
        "--spec",
        str(ctx.spec),
        "--bundle",
        str(bundle),
        "--raw",
        str(ctx.raw),
        "--confirm",
        "--metrics-out",
        str(ctx.final_metrics),
    ]
    subprocess.run(cmd, check=True)  # final_eval refuses (non-zero exit) on any guard
    out: dict[str, Any] = json.loads(ctx.final_metrics.read_text(encoding="utf-8"))
    return out
