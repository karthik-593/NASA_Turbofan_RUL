"""FastAPI inference app.

Loads one artifact bundle at startup (configurable by env) and serves:
    GET  /health   — liveness + which bundle is loaded
    POST /predict  — cycle history -> RUL + maintenance bucket + confidence

Env:
    TURBOFAN_MODELS_DIR (default 'models')
    TURBOFAN_DATASET    (default 'FD001')
    TURBOFAN_MODEL      (default 'lstm')
    TURBOFAN_VERSION    (default 'latest')
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from turbofan.serving.schemas import HealthResponse, PredictRequest, PredictResponse
from turbofan.serving.service import OutOfEnvelope, ShortHistory, predict_rul
from turbofan.training.bundle import Bundle, load_bundle

_STATE: dict[str, Bundle] = {}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _STATE["bundle"] = load_bundle(
        os.environ.get("TURBOFAN_MODELS_DIR", "models"),
        os.environ.get("TURBOFAN_DATASET", "FD001"),
        os.environ.get("TURBOFAN_MODEL", "lstm"),
        os.environ.get("TURBOFAN_VERSION", "latest"),
        device="cpu",
    )
    yield
    _STATE.clear()


app = FastAPI(title="Turbofan RUL API", version="1.0", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def _request_validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
    """422 with type/loc/msg and the rejected input as a string. FastAPI's default handler
    echoes the raw input, which crashes (HTTP 500) when that input is NaN or inf — exactly
    the values the schema rejects."""
    errors = [
        {"type": e["type"], "loc": list(e["loc"]), "msg": e["msg"], "input": repr(e.get("input"))}
        for e in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    m = _STATE["bundle"].manifest
    return HealthResponse(status="ok", dataset=m["dataset"], model=m["model"], version=m["version"])


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    try:
        return PredictResponse(**predict_rul(req.cycles, _STATE["bundle"]))
    except ShortHistory as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except OutOfEnvelope as e:
        raise HTTPException(
            status_code=422,
            detail={"error": "input outside the training operating envelope", "reasons": e.reasons},
        ) from e
