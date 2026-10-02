"""API tests — the request/response contract of GET /health and POST /predict.

This exercises the API contract (shapes, validation, status codes) against a tiny
LSTM bundle trained here on synthetic data, with no data/raw or pre-trained models/
required. Real-bundle prediction behaviour is covered by the container smoke test,
not here.

Uses fastapi.testclient.TestClient (backed by httpx) for in-process HTTP testing;
no uvicorn process needed.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from turbofan.config import KEEP, RUL_CAP, SENSOR_N_COLS, SEQ_LEN
from turbofan.evaluation.protocol import score
from turbofan.features.engineering import add_features
from turbofan.models.lstm_model import LSTMRUL, make_last_windows, make_sequences
from turbofan.serving.app import app
from turbofan.training.bundle import save_bundle


def _cycles(n: int) -> list[dict]:  # type: ignore[type-arg]
    """Synthetic cycles with all KEEP sensors at zero."""
    return [
        {"op1": 0.0, "op2": 0.0, "op3": 0.0, "sensors": {s: 0.0 for s in KEEP}} for _ in range(n)
    ]


def _synthetic_raw_df(final_ruls: list[int], cycles_per_unit: int, seed: int) -> pd.DataFrame:
    """Fabricated engine trajectories, one per `final_ruls` entry (its RUL at the last
    observed cycle), spanning all four maintenance buckets. Sensor values are random
    noise, not physically meaningful — this only has to exercise add_features'/the
    LSTM's plumbing, not produce an accurate model."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit, final_rul in enumerate(final_ruls, start=1):
        for cyc in range(1, cycles_per_unit + 1):
            row = {"unit": unit, "cycle": cyc, "op1": 0.0, "op2": 0.0, "op3": 0.0}
            row.update({s: float(rng.normal()) for s in KEEP})
            row["rul"] = min(final_rul + (cycles_per_unit - cyc), RUL_CAP)
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="session")
def trained_bundle_dir(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Train a tiny LSTM on synthetic data and write it as a models/ bundle.

    final_ruls span critical/urgent/monitor/healthy so every maintenance bucket in
    the fitted metrics is a real number, not the NaN per_bucket_metrics returns for
    an empty bucket (service.py would otherwise hand back a null confidence band).
    """
    df = _synthetic_raw_df(final_ruls=[10, 30, 70, 120], cycles_per_unit=40, seed=0)
    feat, stats = add_features(df, KEEP, "FD001")

    X, y = make_sequences(feat, SENSOR_N_COLS, SEQ_LEN)
    model = LSTMRUL(
        n_features=len(SENSOR_N_COLS),
        hidden=4,
        layers=1,
        max_epochs=2,
        patience=1,
        batch_size=8,
        seed=0,
    )
    model.fit(X, y, X, y)  # val = train: this is a plumbing smoke fit, not a quality one

    X_last, units = make_last_windows(feat, SENSOR_N_COLS, SEQ_LEN)
    y_last = feat.groupby("unit")["rul"].last().loc[units].to_numpy()
    metrics = {"test": score(y_last, model.predict(X_last))}

    models_dir = tmp_path_factory.mktemp("models")
    save_bundle(models_dir, "FD001", "lstm", "v1", model, stats, seed=0, metrics=metrics)
    return str(models_dir)


@pytest.fixture(scope="module")
def client(trained_bundle_dir: str):
    os.environ["TURBOFAN_MODELS_DIR"] = trained_bundle_dir
    os.environ["TURBOFAN_DATASET"] = "FD001"
    os.environ["TURBOFAN_MODEL"] = "lstm"
    os.environ["TURBOFAN_VERSION"] = "v1"
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
