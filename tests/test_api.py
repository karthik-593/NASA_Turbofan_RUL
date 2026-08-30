"""API tests — require a trained FD001/lstm bundle under models/.

Uses fastapi.testclient.TestClient (backed by httpx) for in-process HTTP testing;
no uvicorn process needed.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from turbofan.config import KEEP, SEQ_LEN
from turbofan.serving.app import app

# read by the lifespan function at TestClient startup, not at import time
os.environ.setdefault("TURBOFAN_MODELS_DIR", "models")
os.environ.setdefault("TURBOFAN_DATASET", "FD001")
os.environ.setdefault("TURBOFAN_MODEL", "lstm")


def _has_bundle() -> bool:
    root = Path("models") / "FD001" / "lstm"
    return root.exists() and next(root.iterdir(), None) is not None


pytestmark = pytest.mark.skipif(
    not _has_bundle(), reason="no models/FD001/lstm bundle — run train.py first"
)


def _cycles(n: int) -> list[dict]:  # type: ignore[type-arg]
    """Synthetic cycles with all KEEP sensors at zero."""
    return [
        {"op1": 0.0, "op2": 0.0, "op3": 0.0, "sensors": {s: 0.0 for s in KEEP}} for _ in range(n)
    ]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# -- /health -------------------------------------------------------------------


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["dataset"] == "FD001"
    assert "model" in data
    assert "version" in data


# -- /predict ------------------------------------------------------------------


def test_predict_valid_returns_200(client):
    r = client.post("/predict", json={"cycles": _cycles(SEQ_LEN + 5)})
    assert r.status_code == 200
    data = r.json()
    for key in (
        "predicted_rul",
        "maintenance_bucket",
        "confidence",
        "n_cycles_used",
        "dataset",
        "model_version",
    ):
        assert key in data, f"missing key: {key}"
    assert data["maintenance_bucket"] in ("critical", "urgent", "monitor", "healthy")
    assert 0.0 <= data["predicted_rul"] <= 125.0


def test_predict_confidence_has_band(client):
    r = client.post("/predict", json={"cycles": _cycles(SEQ_LEN + 5)})
    conf = r.json()["confidence"]
    assert "error_band_cycles" in conf
    assert "basis" in conf


def test_predict_short_history_returns_400(client):
    r = client.post("/predict", json={"cycles": _cycles(SEQ_LEN - 1)})
    assert r.status_code == 400


def test_predict_missing_sensor_returns_422(client):
    bad = [
        {"op1": 0.0, "op2": 0.0, "op3": 0.0, "sensors": {s: 0.0 for s in KEEP[:-1]}}
        for _ in range(SEQ_LEN + 5)
    ]
    r = client.post("/predict", json={"cycles": bad})
    assert r.status_code == 422


def test_predict_extra_sensor_returns_422(client):
    bad = [
        {"op1": 0.0, "op2": 0.0, "op3": 0.0, "sensors": {**{s: 0.0 for s in KEEP}, "s_extra": 0.0}}
        for _ in range(SEQ_LEN + 5)
    ]
    r = client.post("/predict", json={"cycles": bad})
    assert r.status_code == 422
