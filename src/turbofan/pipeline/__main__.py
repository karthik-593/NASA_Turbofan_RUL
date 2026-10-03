"""``python -m turbofan.pipeline <stage|all> [--smoke] [--confirm]``: what dvc.yaml calls.

``--smoke`` runs the same stage code on a tiny configuration in an isolated directory
(``smoke/``: outputs, a merged ``params.yaml`` from ``params.yaml``'s ``smoke.overrides``, and a
file-based MLflow store). It never reads the test set and never touches the repo's reports,
models or MLflow server. The smoke environment is set up before ``turbofan.config`` is first
imported — every module-level default then picks up the smoke values.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

STAGES = ("validate", "cv_select", "train_prod", "final_eval", "register")
REPO = Path(__file__).resolve().parents[3]
SMOKE_DIR = REPO / "smoke"
SMOKE_EXPERIMENT = "turbofan-rul-smoke"


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def prepare_smoke(keep: bool) -> Path:
    """Write the merged smoke params and point the process at them. Call before importing
    ``turbofan.config``."""
    if "turbofan.config" in sys.modules:
        raise RuntimeError("prepare_smoke() must run before turbofan.config is imported")
    src = Path(os.environ.get("TURBOFAN_PARAMS", REPO / "params.yaml"))
    doc = yaml.safe_load(src.read_text(encoding="utf-8"))
    if "smoke" not in doc or "overrides" not in doc["smoke"]:
        raise KeyError(f"{src}: missing key 'smoke.overrides'")
    if SMOKE_DIR.exists() and not keep:
        shutil.rmtree(SMOKE_DIR)
    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    merged = _merge({k: v for k, v in doc.items()}, doc["smoke"]["overrides"])
    path = SMOKE_DIR / "params.yaml"
    path.write_text(yaml.safe_dump(merged, sort_keys=False), encoding="utf-8")
    os.environ["TURBOFAN_PARAMS"] = str(path)
    os.environ["MLFLOW_TRACKING_URI"] = (SMOKE_DIR / "mlruns").as_uri()
    os.environ["MLFLOW_EXPERIMENT_NAME"] = SMOKE_EXPERIMENT
    return path


def write_smoke_spec(ctx: Any) -> None:
    """The smoke run's 'locked' spec: a stand-in so train_prod and register have a candidate to
    refit. It locks nothing real and lives only under smoke/."""
    from turbofan.config import PARAMS, RUL_CAP

    spec = {
        "dataset": ctx.dataset,
        "model": PARAMS["smoke"]["spec_model"],
        "seed": ctx.seeds[0],
        "rul_cap": RUL_CAP,
        "locked": True,
        "smoke": True,
    }
    ctx.spec.parent.mkdir(parents=True, exist_ok=True)
    ctx.spec.write_text(yaml.safe_dump(spec), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m turbofan.pipeline", description=__doc__)
    ap.add_argument("stage", choices=(*STAGES, "all"))
    ap.add_argument("--smoke", action="store_true", help="tiny isolated run (see module doc)")
    ap.add_argument("--keep", action="store_true", help="with --smoke: keep smoke/ from last run")
    ap.add_argument(
        "--confirm",
        action="store_true",
        help="final_eval stage: allow the sealed test read when final_eval.enabled is true",
    )
    args = ap.parse_args(argv)
    if args.keep and not args.smoke:
        ap.error("--keep only applies to --smoke")
    if args.smoke:
        prepare_smoke(args.keep)

    from turbofan.pipeline import cv_select, final_eval_stage, register, train_prod, validate
    from turbofan.pipeline.context import build_context

    ctx = build_context(args.smoke)
    if ctx.smoke and not ctx.spec.exists():
        write_smoke_spec(ctx)
    runners: dict[str, Callable[[], object]] = {
        "validate": lambda: validate.run(ctx),
        "cv_select": lambda: cv_select.run(ctx),
        "train_prod": lambda: train_prod.run(ctx),
        "final_eval": lambda: final_eval_stage.run(ctx, args.confirm),
        "register": lambda: register.run(ctx),
    }
    for stage in STAGES if args.stage == "all" else (args.stage,):
        print(f"== stage {stage}{' (smoke)' if ctx.smoke else ''}", flush=True)
        runners[stage]()


def _cuda_used() -> bool:
    torch = sys.modules.get("torch")
    return torch is not None and bool(torch.cuda.is_initialized())


def _end_mlflow_runs() -> None:
    """End every still-active MLflow run and flush asynchronous logging, so a forced exit can
    never leave a run RUNNING or drop a queued write."""
    mlflow = sys.modules.get("mlflow")
    if mlflow is None:
        return
    while mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.flush_artifact_async_logging()
    mlflow.flush_async_logging()


def _terminate_now() -> None:
    """Exit immediately with status 0, skipping native DLL teardown. ``os._exit`` still runs the
    DLL detach that crashes, so on Windows the process terminates itself directly."""
    if sys.platform == "win32":
        import ctypes

        kernel32 = getattr(ctypes, "windll").kernel32  # noqa: B009 (windll is Windows-only)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
        if not kernel32.TerminateProcess(kernel32.GetCurrentProcess(), 0):
            raise OSError(ctypes.get_last_error(), "TerminateProcess failed")
    os._exit(0)


def _skip_native_teardown_if_it_crashes() -> None:
    """Known issue (docs/decisions.md D50): once a stage has succeeded and every output is
    written, end the process now if torch initialized CUDA in it — on this Windows setup such a
    process dies during native teardown (exit 0xC0000409) after its work is done, and DVC would
    count the stage as failed. MLflow runs are ended and flushed first. Reached only after
    ``main()`` returned: an exception still propagates with its traceback."""
    if not _cuda_used():
        return
    sys.stdout.flush()
    sys.stderr.flush()
    _end_mlflow_runs()
    _terminate_now()


if __name__ == "__main__":
    main()
    _skip_native_teardown_if_it_crashes()
