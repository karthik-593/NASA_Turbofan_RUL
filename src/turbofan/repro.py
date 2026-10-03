"""Reproducibility context for analysis/audit computations written to ``reports/``.

Every number in a report must reproduce exactly on a rerun. Two things make that possible:

1. ``cpu_deterministic(n_threads)`` — a context manager the computation runs inside. It
   pins torch and every native thread pool (BLAS / OpenMP, via threadpoolctl) to a fixed
   thread count, since float reductions are order-dependent and the order depends on the
   number of threads; turns on ``torch.use_deterministic_algorithms``; and raises if the
   block initialized CUDA, so a GPU kernel can never slip into a CPU-only number.
   XGBoost's GPU path is invisible to that check — callers pass ``device="cpu"``
   explicitly rather than ``config.xgb_device()``.

2. ``environment_context(seeds=..., n_threads=...)`` — captures what a rerun must match:
   git SHA + dirty flag, DVC data hash, OS / CPU, Python and library versions, the native
   thread pools actually loaded, every ``turbofan.config`` constant (which includes the
   regime-KMeans seed and n_init), and the seeds of every stochastic step in the
   computation (KMeans, bootstrap resampling, piecewise-fit initialisation, ...).
   ``seeds`` has no default: a report states its seeds explicitly, or an empty mapping
   when nothing in it is stochastic. ``render_markdown`` turns the context into a
   "Reproducibility" section to append to the report; ``mismatches`` compares a recorded
   context with the current one so a rerun can say why it might not match.
"""

from __future__ import annotations

import json
import os
import platform
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _pkg_version
from pathlib import Path
from typing import Any

# Imported for their side effect too: threadpoolctl only reports the native thread pools
# of libraries already loaded in the process.
import numpy  # noqa: F401
import sklearn.cluster  # noqa: F401
import torch
from threadpoolctl import threadpool_info, threadpool_limits

from turbofan.devices import resolved_devices
from turbofan.tracking import _config_params, _dvc_data_hash
from turbofan.training.bundle import git_provenance

__all__ = [
    "GpuUsedError",
    "cpu_deterministic",
    "environment_context",
    "mismatches",
    "render_markdown",
    "write_context",
]

_RECORDED_LIBS = (
    "numpy",
    "pandas",
    "scikit-learn",
    "scipy",
    "torch",
    "xgboost",
    "threadpoolctl",
    "mlflow",
)

# Keys that must be equal for a rerun to be expected to reproduce bit-for-bit.
# Timestamp and hostname-like fields are deliberately excluded.
_MUST_MATCH = (
    "git_commit",
    "git_dirty",
    "dvc_data_hash",
    "os",
    "machine",
    "processor",
    "python",
    "libs",
    "n_threads",
    "thread_pools",
    "torch_deterministic",
    "torch_cuda_initialized",
    "config",
    "seeds",
)


def _cuda_initialized() -> bool:
    return bool(torch.cuda.is_initialized())  # type: ignore[no-untyped-call]


class GpuUsedError(RuntimeError):
    """A CPU-only computation initialized CUDA."""


@contextmanager
def cpu_deterministic(n_threads: int) -> Iterator[None]:
    """Run the block CPU-only, deterministic, on exactly ``n_threads`` threads.

    Restores torch's thread count and deterministic flag and the native thread-pool
    limits on exit, so it is safe to use inside a test process.
    """
    if n_threads < 1:
        raise ValueError(f"n_threads must be >= 1, got {n_threads}")
    if _cuda_initialized():
        raise GpuUsedError(
            "CUDA was already initialized before entering cpu_deterministic(); run the "
            "audit computation in a fresh process that never touches the GPU"
        )
    prev_threads = torch.get_num_threads()
    prev_det = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(n_threads)
    torch.use_deterministic_algorithms(True)
    try:
        with threadpool_limits(limits=n_threads):
            yield
        if _cuda_initialized():
            raise GpuUsedError(
                "CUDA was initialized inside cpu_deterministic() — some step ran on the "
                "GPU; pass device='cpu' explicitly to every model/fit in the computation"
            )
    finally:
        torch.set_num_threads(prev_threads)
        torch.use_deterministic_algorithms(prev_det)


def _lib_versions() -> dict[str, str]:
    out = {}
    for name in _RECORDED_LIBS:
        try:
            out[name] = _pkg_version(name)
        except PackageNotFoundError:
            out[name] = "not installed"
    # Package metadata drops the local build tag; torch.__version__ keeps it (e.g.
    # "2.11.0+cpu" vs "+cu124"), and which build ran is part of what must match.
    out["torch"] = str(torch.__version__)
    return out


def _thread_pools() -> list[dict[str, Any]]:
    return sorted(
        (
            {
                "user_api": p["user_api"],
                "internal_api": p["internal_api"],
                "version": p.get("version"),
                "library": Path(p["filepath"]).name,
                "num_threads": p["num_threads"],
            }
            for p in threadpool_info()
        ),
        key=lambda p: (p["user_api"], p["internal_api"], p["library"]),
    )


def environment_context(*, seeds: Mapping[str, int], n_threads: int) -> dict[str, Any]:
    """Everything a rerun must match. Call it *inside* ``cpu_deterministic(n_threads)``
    so the recorded thread pools and torch flags are the ones the computation ran with."""
    if torch.get_num_threads() != n_threads:
        raise RuntimeError(
            f"torch is running {torch.get_num_threads()} threads, not {n_threads} — call "
            "environment_context() inside cpu_deterministic(n_threads)"
        )
    prov = git_provenance()
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": prov["git_commit"],
        "git_dirty": prov["git_dirty"],
        "dvc_data_hash": _dvc_data_hash(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "libs": _lib_versions(),
        "n_threads": n_threads,
        "thread_pools": _thread_pools(),
        "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
        "torch_cuda_initialized": _cuda_initialized(),
        # informational, not in _MUST_MATCH: which devices model code would resolve to (D43)
        "devices": resolved_devices(),
        "config": _config_params(),
        "seeds": dict(seeds),
    }


def mismatches(recorded: Mapping[str, Any], current: Mapping[str, Any]) -> list[str]:
    """Names of the reproducibility-relevant fields that differ between two contexts."""
    return [k for k in _MUST_MATCH if recorded.get(k) != current.get(k)]


def write_context(ctx: Mapping[str, Any], report_path: str | Path) -> Path:
    """Write the full context as a JSON sidecar next to the report: ``<report>.env.json``."""
    out = Path(report_path).with_suffix(".env.json")
    out.write_text(json.dumps(ctx, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def render_markdown(ctx: Mapping[str, Any]) -> str:
    """A "Reproducibility" section to append to a ``reports/*.md`` file."""
    dirty = " (**dirty working tree — not reproducible from the commit alone**)"
    lines = [
        "## Reproducibility",
        "",
        f"- Generated: {ctx['timestamp_utc']}",
        f"- Git commit: `{ctx['git_commit']}`{dirty if ctx['git_dirty'] else ''}",
        f"- DVC data hash (`data/raw`): `{ctx['dvc_data_hash']}`",
        f"- Hardware: CPU only — {ctx['processor'] or ctx['machine']}, "
        f"{ctx['logical_cpus']} logical CPUs; OS {ctx['os']}",
        f"- Threads: {ctx['n_threads']} (torch and all native pools pinned); "
        f"torch deterministic algorithms: {ctx['torch_deterministic']}; "
        f"CUDA initialized: {ctx['torch_cuda_initialized']}",
        f"- Python {ctx['python']}",
        "",
        "| Library | Version |",
        "|---|---|",
        *(f"| {name} | {ver} |" for name, ver in ctx["libs"].items()),
        "",
        "| Native thread pool | Implementation | Version | Threads |",
        "|---|---|---|---|",
        *(
            f"| {p['library']} | {p['internal_api']} ({p['user_api']}) | {p['version']} | "
            f"{p['num_threads']} |"
            for p in ctx["thread_pools"]
        ),
        "",
    ]
    if ctx["seeds"]:
        lines += [
            "| Stochastic step | Seed |",
            "|---|---|",
            *(f"| {step} | {seed} |" for step, seed in ctx["seeds"].items()),
        ]
    else:
        lines.append("No stochastic steps (no seeds).")
    cfg_seeds = {k: v for k, v in ctx["config"].items() if "SEED" in k or "N_INIT" in k}
    lines += [
        "",
        "Pipeline constants in effect: "
        + ", ".join(f"`{k}={v}`" for k, v in sorted(cfg_seeds.items()))
        + " (full `turbofan.config` in the JSON sidecar).",
        "",
    ]
    return "\n".join(lines)
