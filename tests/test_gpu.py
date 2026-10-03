"""The CUDA path, checked in a subprocess (the pytest process itself hides the GPU, see
conftest.py). Skipped on machines without an NVIDIA GPU or without a CUDA build of torch (D45)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

_CODE = """
import json, numpy as np, torch
from turbofan.devices import resolved_devices
from turbofan.models.lstm_model import LSTMRUL

rng = np.random.default_rng(0)
X = rng.normal(size=(64, 5, 3)).astype(np.float32)
y = rng.uniform(0, 125, 64).astype(np.float32)
m = LSTMRUL(n_features=3, hidden=4, layers=1, max_epochs=2, patience=1, batch_size=16, seed=0)
m.fit(X, y, X, y)
print(json.dumps({
    "net_device": next(m.net.parameters()).device.type,
    "pred_finite": bool(np.isfinite(m.predict(X)).all()),
    "devices": resolved_devices(),
}))
"""


def _cuda_torch_available() -> bool:
    if shutil.which("nvidia-smi") is None:
        return False
    out = subprocess.run(
        [sys.executable, "-c", "import torch; print(torch.cuda.is_available())"],
        capture_output=True,
        text=True,
        check=False,
        env={k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"},
    )
    return out.stdout.strip() == "True"


@pytest.mark.skipif(not _cuda_torch_available(), reason="no NVIDIA GPU or CPU-only torch build")
def test_lstm_fit_runs_on_cuda_and_device_is_recorded() -> None:
    import json

    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    res = subprocess.run(
        [sys.executable, "-c", _CODE], capture_output=True, text=True, check=True, env=env
    )
    out = json.loads(res.stdout.strip().splitlines()[-1])
    assert out["net_device"] == "cuda"
    assert out["pred_finite"]
    assert out["devices"]["torch"] == "cuda"
    assert out["devices"]["gpu_name"]
