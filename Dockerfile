# Multi-stage build for optimized production image (self-contained for Nextflow)
FROM python:3.10-slim AS builder

WORKDIR /build

# Build deps only in builder
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install dependencies into a relocatable user site dir
RUN pip install --no-cache-dir --user -r requirements.txt


FROM python:3.10-slim AS runtime

LABEL maintainer="ESDP Team"
LABEL description="Early Stop Decision Polishing - Production ML Service"
LABEL version="1.0.2"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_PATH=/app/models/best_model_pipeline.pkl \
    PORT=8000 \
    LOG_LEVEL=INFO \
    WORKERS=1

# Non-root user
RUN useradd -m -u 1000 esdp && \
    mkdir -p /app /app/models /app/logs && \
    chown -R esdp:esdp /app

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /root/.local /home/esdp/.local
ENV PATH=/home/esdp/.local/bin:$PATH

# Copy application code
COPY --chown=esdp:esdp esdp_decide.py .
COPY --chown=esdp:esdp esdp_features.py .
COPY --chown=esdp:esdp esdp_manifest.py .
COPY --chown=esdp:esdp esdp_trajectory.py .
COPY --chown=esdp:esdp esdp_cli.py .
COPY --chown=esdp:esdp api_service.py .
COPY --chown=esdp:esdp config.yaml .
COPY --chown=esdp:esdp docker-entrypoint.sh .

# Copy the verified model bundle INTO the image for portability (Nextflow)
COPY --chown=esdp:esdp models/best_model_pipeline.pkl ./models/
COPY --chown=esdp:esdp models/feature_names.txt ./models/
COPY --chown=esdp:esdp models/model_manifest.v1.json ./models/

RUN chmod +x /app/esdp_cli.py && \
    ln -s /app/esdp_cli.py /usr/local/bin/esdp

RUN chmod +x /app/docker-entrypoint.sh

USER esdp

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5).read()"

ENTRYPOINT ["/app/docker-entrypoint.sh"]

CMD ["uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "8000"]
