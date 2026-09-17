# =============================================================================
# ESDP Sequential Controller v2
# Reproducible FastAPI container
# =============================================================================


# -----------------------------------------------------------------------------
# Builder
# -----------------------------------------------------------------------------

FROM python:3.10-slim AS builder

WORKDIR /build

# Build dependencies kept only in the builder image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install Python dependencies into a relocatable user directory.
RUN python -m pip install --upgrade pip && \
    python -m pip install \
        --no-cache-dir \
        --user \
        -r requirements.txt


# -----------------------------------------------------------------------------
# Runtime
# -----------------------------------------------------------------------------

FROM python:3.10-slim AS runtime

LABEL maintainer="ESDP Team"
LABEL description="ESDP sequential STOP/CONTINUE controller for adaptive bacterial long-read polishing"
LABEL version="2.0.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

# -----------------------------------------------------------------------------
# Non-root runtime user
# -----------------------------------------------------------------------------

RUN useradd \
        --create-home \
        --uid 1000 \
        esdp \
    && mkdir -p \
        /app \
        /app/outputs/frozen_sequential_model_v2 \
    && chown -R esdp:esdp /app

WORKDIR /app


# -----------------------------------------------------------------------------
# Python environment
# -----------------------------------------------------------------------------

COPY --from=builder \
    /root/.local \
    /home/esdp/.local

ENV PATH=/home/esdp/.local/bin:$PATH


# -----------------------------------------------------------------------------
# Application
# -----------------------------------------------------------------------------

COPY --chown=esdp:esdp \
    esdp_decide.py \
    /app/esdp_decide.py

COPY --chown=esdp:esdp \
    api_service.py \
    /app/api_service.py


# -----------------------------------------------------------------------------
# Frozen ESDP v2 model
# -----------------------------------------------------------------------------

COPY --chown=esdp:esdp \
    outputs/frozen_sequential_model_v2/esdp_sequential_rf_v2.joblib \
    /app/outputs/frozen_sequential_model_v2/esdp_sequential_rf_v2.joblib

COPY --chown=esdp:esdp \
    outputs/frozen_sequential_model_v2/esdp_sequential_rf_v2.metadata.json \
    /app/outputs/frozen_sequential_model_v2/esdp_sequential_rf_v2.metadata.json

COPY --chown=esdp:esdp \
    outputs/frozen_sequential_model_v2/SHA256.txt \
    /app/outputs/frozen_sequential_model_v2/SHA256.txt


# -----------------------------------------------------------------------------
# Runtime
# -----------------------------------------------------------------------------

USER esdp

EXPOSE 8000


# -----------------------------------------------------------------------------
# Health check
# -----------------------------------------------------------------------------

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=10s \
    --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).read()" \
    || exit 1


# -----------------------------------------------------------------------------
# API
# -----------------------------------------------------------------------------

CMD ["uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]