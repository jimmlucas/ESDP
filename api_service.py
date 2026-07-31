#!/usr/bin/env python3
"""
api_service.py - FastAPI service for ESDP polishing decisions

Production-grade REST API with:
- Structured logging (structlog)
- Metrics tracking
- Health checks
- Pydantic validation
- Error handling
- Transparent rule overrides
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Any
import structlog
import logging
import time
from datetime import datetime
import sys

# Import decision logic
from esdp_decide import decide, PolishingMetrics, Decision
from esdp_manifest import load_verified_model

# ============================================================
# Structured Logging Setup
# ============================================================
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# ============================================================
# FastAPI App
# ============================================================
app = FastAPI(
    title="ESDP Polishing Decision API",
    description="Production API for Oxford Nanopore polishing decisions",
    version="1.0.0"
)

# ============================================================
# In-memory metrics (simple implementation)
# ============================================================
METRICS = {
    "total_requests": 0,
    "successful_predictions": 0,
    "failed_predictions": 0,
    "avg_response_time_ms": 0.0,
    "decisions_by_rounds": {1: 0, 3: 0, 5: 0}
}

# ============================================================
# Pydantic Models (API Contract)
# ============================================================
class PredictionRequest(BaseModel):
    """Request schema - matches PolishingMetrics exactly"""
    # Metadata
    sample_id: str = Field(..., description="Unique sample identifier")
    genus: Optional[str] = Field(None, description="Bacterial genus")
    round: Optional[int] = Field(None, description="Current polishing round")

    # Core assembly metrics
    coverage: Optional[float] = None
    coverage_effective: Optional[float] = None
    n50: Optional[float] = None
    qv: Optional[float] = None
    error_rate: Optional[float] = None
    busco_complete: Optional[float] = None
    num_contigs: Optional[int] = None
    total_length: Optional[int] = None

    # Raw read metrics
    raw_total_bp: Optional[float] = None
    raw_read_n50: Optional[float] = None
    raw_mean_read_len: Optional[float] = None

    # Assembly info metrics
    ai_num_contigs: Optional[int] = None
    ai_total_bp: Optional[int] = None
    ai_mean_cov: Optional[float] = None
    ai_median_cov: Optional[float] = None
    ai_cov_cv: Optional[float] = None
    ai_circular_n: Optional[int] = None
    ai_circular_bp_frac: Optional[float] = None
    ai_repeat_bp_frac: Optional[float] = None
    ai_longest_len: Optional[int] = None
    ai_longest_cov: Optional[float] = None

    # Polishing metrics
    polish_mean_contig_cov: Optional[float] = None
    align_err_consensus: Optional[float] = None
    align_err_polishing: Optional[float] = None

    # Flye metrics
    ovlp_div_initial: Optional[float] = None
    ovlp_median_div_first: Optional[float] = None
    mean_edge_coverage: Optional[float] = None

    # Additional features
    delta_qv: Optional[float] = None
    delta_busco_complete: Optional[float] = None
    delta_error_rate: Optional[float] = None
    qv_improvement_rate: Optional[float] = None
    assembly_error: Optional[float] = None

    # Control parameters (NEW: paper-grade transparency)
    force_conservative: bool = Field(
        False, 
        description="Force conservative recommendation (always R5)"
    )
    confidence_threshold: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for automatic conservative bias (default: 0.5)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "sample_id": "sample_001",
                "genus": "Escherichia",
                "round": 1,
                "coverage": 50.0,
                "coverage_effective": 48.0,
                "n50": 4500000,
                "qv": 35.2,
                "error_rate": 0.0003,
                "busco_complete": 95.5,
                "num_contigs": 1,
                "total_length": 4800000,
                "force_conservative": False,
                "confidence_threshold": 0.5
            }
        }


class PredictionResponse(BaseModel):
    """Response schema with full transparency"""
    sample_id: str
    recommended_rounds: int
    confidence: float
    reasoning: str
    warnings: List[str]
    rule_overrides: Dict[str, Any]  # NEW: explicit rule tracking
    model_version: str
    feature_schema_version: str
    class_probabilities: Dict[str, float]
    processing_time_ms: float


# ============================================================
# Middleware for logging
# ============================================================
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests with structured logging"""
    start_time = time.time()

    # Log request
    logger.info(
        "request_received",
        method=request.method,
        path=request.url.path,
        client=request.client.host if request.client else "unknown"
    )

    # Process request
    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000

        # Log response
        logger.info(
            "request_completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration_ms, 2)
        )

        return response

    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        logger.error(
            "request_failed",
            method=request.method,
            path=request.url.path,
            error=str(e),
            duration_ms=round(duration_ms, 2)
        )
        raise


# ============================================================
# API Endpoints
# ============================================================
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "ESDP Polishing Decision API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "metrics": "/metrics",
            "model_info": "/model/info"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "esdp-api"
    }


@app.get("/metrics")
async def get_metrics():
    """Return service metrics"""
    return {
        "metrics": METRICS,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/model/info")
async def model_info():
    """Return model metadata"""
    try:
        verified = load_verified_model("models/model_manifest.v1.json")
        pipeline = verified.model

        return {
            "model_id": verified.manifest.model_id,
            "model_version": verified.manifest.model_version,
            "feature_schema_version": verified.manifest.feature_schema.version,
            "feature_schema_prospective": (
                verified.manifest.feature_schema.prospective
            ),
            "feature_count": len(getattr(pipeline, 'feature_names', [])),
            "model_type": type(pipeline.named_steps['model']).__name__ if hasattr(pipeline, 'named_steps') else "unknown"
        }
    except Exception as e:
        logger.error("model_info_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to load model info: {str(e)}")


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make polishing decision based on input metrics.

    Returns recommended number of polishing rounds with confidence score
    and full transparency on applied rules.

    Parameters:
    - force_conservative: Force recommendation to R5 (overrides all other logic)
    - confidence_threshold: Threshold for automatic conservative bias (default: 0.5)
    """
    start_time = time.time()
    METRICS["total_requests"] += 1

    try:
        # Convert request to PolishingMetrics
        metrics = PolishingMetrics(
            **request.dict(exclude={'force_conservative', 'confidence_threshold'})
        )

        # Make decision with explicit parameters
        decision = decide(
            metrics,
            model_path="models/best_model_pipeline.pkl",
            confidence_threshold=request.confidence_threshold,
            force_conservative=request.force_conservative
        )

        # Update metrics
        METRICS["successful_predictions"] += 1
        METRICS["decisions_by_rounds"][decision.recommended_rounds] = \
            METRICS["decisions_by_rounds"].get(decision.recommended_rounds, 0) + 1

        # Calculate processing time
        processing_time_ms = (time.time() - start_time) * 1000

        # Update avg response time
        total = METRICS["successful_predictions"]
        METRICS["avg_response_time_ms"] = (
            (METRICS["avg_response_time_ms"] * (total - 1) + processing_time_ms) / total
        )

        # Log decision
        logger.info(
            "prediction_success",
            sample_id=decision.sample_id,
            recommended_rounds=decision.recommended_rounds,
            confidence=round(decision.confidence, 3),
            rule_overrides=decision.rule_overrides,
            processing_time_ms=round(processing_time_ms, 2)
        )

        # Return response
        return PredictionResponse(
            sample_id=decision.sample_id,
            recommended_rounds=decision.recommended_rounds,
            confidence=decision.confidence,
            reasoning=decision.reasoning,
            warnings=decision.warnings,
            rule_overrides=decision.rule_overrides,  # NEW: explicit tracking
            model_version=decision.model_version,
            feature_schema_version=decision.feature_schema_version,
            class_probabilities=decision.class_probabilities,
            processing_time_ms=round(processing_time_ms, 2)
        )

    except Exception as e:
        METRICS["failed_predictions"] += 1
        logger.error(
            "prediction_failed",
            sample_id=request.sample_id,
            error=str(e),
            error_type=type(e).__name__
        )
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


# ============================================================
# Startup/Shutdown Events
# ============================================================
@app.on_event("startup")
async def startup_event():
    """Log startup"""
    logger.info("api_startup", message="ESDP API starting up")


@app.on_event("shutdown")
async def shutdown_event():
    """Log shutdown"""
    logger.info("api_shutdown", message="ESDP API shutting down", metrics=METRICS)


# ============================================================
# Run with: uvicorn api_service:app --reload --host 0.0.0.0 --port 8000
# ============================================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
