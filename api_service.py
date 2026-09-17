#!/usr/bin/env python3
"""
api_service.py

FastAPI service for the frozen ESDP sequential controller.

The API is a thin transport layer around esdp_decide.py.

Operational decision rule
-------------------------
    p_continue >= 0.45  -> CONTINUE
    p_continue <  0.45  -> STOP

Legal decision rounds
---------------------
    R1-R4 only.

R5 is the maximum Racon round and proceeds directly to mandatory Medaka;
there is no ESDP decision after R5.

Operational predictors
----------------------
Exactly 10 frozen predictors are accepted:

    round
    coverage_est
    expected_genome_size
    raw_read_n50
    ai_cov_cv
    current_n50
    current_num_contigs
    current_assembly_frac
    delta_n50_last
    n50_from_R1

sample_id is metadata only.

BUSCO, QV, genus, reference-derived metrics, future-round information,
and post-hoc rule overrides are not part of operational inference.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from esdp_decide import (
    DECISION_THRESHOLD,
    FEATURE_SCHEMA_SHA256,
    FROZEN_FEATURES,
    FROZEN_MODEL_SHA256,
    LEGAL_DECISION_ROUNDS,
    MAX_RACON_ROUND,
    MODEL_VERSION,
    DEFAULT_MODEL_PATH,
    Decision,
    ESDPError,
    InputValidationError,
    ModelValidationError,
    PolishingState,
    decide,
    model_info,
)


# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    title="ESDP Sequential Decision API",
    description=(
        "REST API for the frozen ESDP sequential STOP/CONTINUE controller "
        "for adaptive bacterial long-read Racon polishing."
    ),
    version="2.0.0",
)


# =============================================================================
# REQUEST / RESPONSE MODELS
# =============================================================================

class PredictionRequest(BaseModel):
    """
    ESDP operational inference request.

    sample_id is metadata only and is not supplied to the model.
    """

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "sample_id": "sample_001",
                "round": 2,
                "coverage_est": 40.0,
                "expected_genome_size": 5000000.0,
                "raw_read_n50": 15000.0,
                "ai_cov_cv": 0.20,
                "current_n50": 4800000.0,
                "current_num_contigs": 2,
                "current_assembly_frac": 0.99,
                "delta_n50_last": 25000.0,
                "n50_from_R1": 40000.0,
            }
        },
    )

    sample_id: str = Field(
        ...,
        min_length=1,
        description="Sample identifier. Metadata only; not used as a predictor.",
    )

    round: int = Field(
        ...,
        ge=1,
        le=4,
        description="Current Racon decision round. Legal values: 1-4.",
    )

    coverage_est: Optional[float] = Field(
        None,
        ge=0,
        description="Estimated sequencing coverage.",
    )

    expected_genome_size: Optional[float] = Field(
        None,
        ge=0,
        description="Expected genome size in base pairs.",
    )

    raw_read_n50: Optional[float] = Field(
        None,
        ge=0,
        description="Raw-read N50.",
    )

    ai_cov_cv: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Coverage coefficient of variation from Flye assembly information: "
            "coverage standard deviation / mean coverage."
        ),
    )

    current_n50: Optional[float] = Field(
        None,
        ge=0,
        description="N50 of the current Racon-polished assembly.",
    )

    current_num_contigs: Optional[float] = Field(
        None,
        ge=0,
        description="Number of contigs in the current assembly.",
    )

    current_assembly_frac: Optional[float] = Field(
        None,
        ge=0,
        description=(
            "Current assembly length divided by expected genome size."
        ),
    )

    delta_n50_last: Optional[float] = Field(
        None,
        description=(
            "Current N50 minus N50 from the previous Racon round. "
            "May be missing at R1."
        ),
    )

    n50_from_R1: Optional[float] = Field(
        None,
        description=(
            "Current N50 minus N50 at R1. "
            "May be missing at R1."
        ),
    )


class PredictionResponse(BaseModel):
    sample_id: str
    round: int

    p_continue: float
    threshold: float

    decision: str
    next_action: str

    model_version: str
    model_sha256: str
    feature_schema_sha256: str

    missing_features: List[str]


class HealthResponse(BaseModel):
    status: str
    service: str
    api_version: str
    model_version: str
    timestamp_utc: str


class ModelInfoResponse(BaseModel):
    model_version: str
    model_type: str
    decision_threshold: float
    legal_decision_rounds: List[int]
    maximum_racon_round: int
    feature_count: int
    features: List[str]
    feature_schema_sha256: str
    model_sha256: str
    model_path: str


# =============================================================================
# ROOT
# =============================================================================

@app.get("/")
async def root():
    return {
        "service": "ESDP Sequential Decision API",
        "api_version": "2.0.0",
        "model_version": MODEL_VERSION,
        "decision_threshold": DECISION_THRESHOLD,
        "legal_decision_rounds": list(LEGAL_DECISION_ROUNDS),
        "maximum_racon_round": MAX_RACON_ROUND,
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "model_info": "/model/info",
            "docs": "/docs",
            "openapi": "/openapi.json",
        },
    }


# =============================================================================
# HEALTH
# =============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
)
async def health():
    """
    Service-level health endpoint.

    This endpoint confirms that the API process is running.
    Model integrity is reported separately by /model/info.
    """

    return HealthResponse(
        status="healthy",
        service="esdp-api",
        api_version="2.0.0",
        model_version=MODEL_VERSION,
        timestamp_utc=datetime.now(
            timezone.utc
        ).isoformat(),
    )


# =============================================================================
# MODEL INFO
# =============================================================================

@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
)
async def get_model_info():
    """
    Return metadata for the frozen ESDP model.

    Loading this endpoint verifies that the artifact is readable and
    satisfies the expected SHA256 contract.
    """

    try:
        info = model_info(
            model_path=DEFAULT_MODEL_PATH,
            verify_model_sha256=True,
        )

        return ModelInfoResponse(
            **info
        )

    except ModelValidationError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "model_validation_error",
                "message": str(exc),
            },
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "model_info_error",
                "message": str(exc),
            },
        ) from exc


# =============================================================================
# PREDICT
# =============================================================================

@app.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(
    request: PredictionRequest,
):
    """
    Return one sequential ESDP STOP/CONTINUE decision.

    Decision rule:
        p_continue >= 0.45 -> CONTINUE
        p_continue <  0.45 -> STOP

    Workflow:
        STOP -> MEDAKA

        CONTINUE at R1-R3 ->
            execute the next Racon round and re-evaluate.

        CONTINUE at R4 ->
            execute R5 and proceed directly to MEDAKA.
    """

    try:
        state = PolishingState(
            sample_id=request.sample_id,
            round=request.round,
            coverage_est=request.coverage_est,
            expected_genome_size=request.expected_genome_size,
            raw_read_n50=request.raw_read_n50,
            ai_cov_cv=request.ai_cov_cv,
            current_n50=request.current_n50,
            current_num_contigs=request.current_num_contigs,
            current_assembly_frac=request.current_assembly_frac,
            delta_n50_last=request.delta_n50_last,
            n50_from_R1=request.n50_from_R1,
        )

        result: Decision = decide(
            state,
            model_path=DEFAULT_MODEL_PATH,
            verify_model_sha256=True,
        )

        return PredictionResponse(
            **result.to_dict()
        )

    except InputValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "input_validation_error",
                "message": str(exc),
            },
        ) from exc

    except ModelValidationError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "model_validation_error",
                "message": str(exc),
            },
        ) from exc

    except ESDPError as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "esdp_error",
                "message": str(exc),
            },
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "unexpected_error",
                "message": str(exc),
            },
        ) from exc


# =============================================================================
# RUN LOCALLY
# =============================================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "api_service:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
