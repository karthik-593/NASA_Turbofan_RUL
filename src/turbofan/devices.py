"""Which devices a run resolved to — recorded, never forced (D43).

Device selection stays automatic: the LSTM uses CUDA when torch can see a GPU, XGBoost when
``nvidia-smi`` is present (``config.xgb_device``). What each run actually resolved to, and
the GPU behind it, is logged to MLflow (``tracking.run``) and to ``repro.environment_context``
so a result can always be traced to the hardware that produced it.
"""

from __future__ import annotations

import subprocess

import torch

from turbofan.config import xgb_device

__all__ = ["resolved_devices"]


def _nvidia_smi(*args: str) -> str | None:
    try:
        res = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def resolved_devices() -> dict[str, str | None]:
    """``torch`` / ``xgboost``: 'cuda' or 'cpu' as auto-selected; ``gpu_name``,
    ``gpu_driver``, ``cuda_driver_version`` from nvidia-smi (None without a GPU);
    ``torch_cuda_build``: the CUDA version torch was built with (None for a CPU-only build,
    in which case the LSTM runs on CPU even when a GPU is present)."""
    query = _nvidia_smi("--query-gpu=name,driver_version", "--format=csv,noheader")
    name = driver = None
    if query:
        first = query.splitlines()[0].split(",")
        name, driver = first[0].strip(), first[1].strip() if len(first) > 1 else None
    header = _nvidia_smi() or ""
    cuda = next(
        (
            ln.split("CUDA Version:")[1].split()[0]
            for ln in header.splitlines()
            if "CUDA Version:" in ln
        ),
        None,
    )
    return {
        "torch": "cuda" if torch.cuda.is_available() else "cpu",
        "xgboost": xgb_device(),
        "gpu_name": name,
        "gpu_driver": driver,
        "cuda_driver_version": cuda,
        "torch_cuda_build": torch.version.cuda,
    }
