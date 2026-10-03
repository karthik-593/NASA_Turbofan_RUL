"""Tests for turbofan.devices — the resolved device is recorded, never forced (D43)."""

from __future__ import annotations

import subprocess

import pytest

from turbofan import devices

SMI_HEADER = "| NVIDIA-SMI 581.86   Driver Version: 581.86   CUDA Version: 13.0 |"


def _fake_run(query_out: str | None, header_out: str | None):
    def run(cmd, **_kw):  # type: ignore[no-untyped-def]
        out = query_out if any(a.startswith("--query-gpu") for a in cmd) else header_out
        if out is None:
            raise FileNotFoundError("nvidia-smi")
        return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")

    return run


def test_gpu_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        devices.subprocess, "run", _fake_run("NVIDIA GeForce RTX 4060, 581.86\n", SMI_HEADER)
    )
    monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(devices, "xgb_device", lambda: "cuda")
    d = devices.resolved_devices()
    assert d["gpu_name"] == "NVIDIA GeForce RTX 4060"
    assert d["gpu_driver"] == "581.86"
    assert d["cuda_driver_version"] == "13.0"
    assert d["torch"] == "cpu" and d["xgboost"] == "cuda"


def test_no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(devices.subprocess, "run", _fake_run(None, None))
    monkeypatch.setattr(devices.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(devices, "xgb_device", lambda: "cpu")
    d = devices.resolved_devices()
    assert d["gpu_name"] is None and d["gpu_driver"] is None and d["cuda_driver_version"] is None
    assert d["torch"] == d["xgboost"] == "cpu"


def test_environment_context_records_devices() -> None:
    from turbofan import repro

    with repro.cpu_deterministic(n_threads=1):
        ctx = repro.environment_context(seeds={}, n_threads=1)
    assert set(ctx["devices"]) >= {"torch", "xgboost", "gpu_name", "torch_cuda_build"}
    assert "devices" not in repro._MUST_MATCH  # informational; not part of bit-repro checks
