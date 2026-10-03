"""The locked-candidate spec: one candidate named exactly, frozen after protocol-v2 selection.

Shared by ``final_eval`` (the sealed test read) and the production refit (``pipeline``), so
both read the same validated spec. Reads no data.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from turbofan.config import DATASETS, RUL_CAP

__all__ = ["REQUIRED", "FinalEvalRefused", "load_spec", "spec_hash"]

REQUIRED = ("dataset", "model", "seed", "rul_cap", "locked")


class FinalEvalRefused(RuntimeError):
    """final_eval declined to read the test set."""


def load_spec(path: str | Path) -> dict[str, Any]:
    spec = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(spec, dict):
        raise FinalEvalRefused(f"{path}: a spec is a YAML mapping")
    missing = [k for k in REQUIRED if k not in spec]
    if missing:
        raise FinalEvalRefused(f"{path}: spec lacks {missing}")
    if spec["locked"] is not True:
        raise FinalEvalRefused(
            f"{path}: spec is not locked (locked: true) — finish selection first"
        )
    if spec["dataset"] not in DATASETS:
        raise FinalEvalRefused(f"{path}: unknown dataset {spec['dataset']!r}")
    if float(spec["rul_cap"]) != RUL_CAP:
        raise FinalEvalRefused(
            f"{path}: rul_cap {spec['rul_cap']} != params.yaml rul_cap {RUL_CAP}; model "
            "clipping follows params.yaml, so set rul_cap there before a final evaluation"
        )
    return spec


def spec_hash(spec: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
