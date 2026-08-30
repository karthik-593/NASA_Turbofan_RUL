"""Pydantic schemas for the inference API.

The request is an engine's recent cycle history; the response is RUL plus a maintenance
bucket and a confidence band grounded in the shipped artifact's measured error.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from turbofan.config import KEEP


class CycleReading(BaseModel):
    """One operating cycle: the three op settings plus the KEEP sensor channels."""

    op1: float
    op2: float
    op3: float
    sensors: dict[str, float] = Field(..., description=f"must contain exactly these keys: {KEEP}")

    @field_validator("sensors")
    @classmethod
    def _exact_keys(cls, v: dict[str, float]) -> dict[str, float]:
        missing = set(KEEP) - set(v)
        extra = set(v) - set(KEEP)
        if missing or extra:
            raise ValueError(
                f"'sensors' must be exactly {KEEP}; missing={sorted(missing)} extra={sorted(extra)}"
            )
        return v


class PredictRequest(BaseModel):
    cycles: list[CycleReading] = Field(
        ..., min_length=1, description="cycles in chronological order (oldest first)"
    )


class Confidence(BaseModel):
    error_band_cycles: float | None = Field(
        None, description="+-cycles: the shipped model's test RMSE in this RUL bucket"
    )
    basis: str


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    predicted_rul: float
    maintenance_bucket: str
    confidence: Confidence
    n_cycles_used: int
    dataset: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    dataset: str
    model: str
    version: str
