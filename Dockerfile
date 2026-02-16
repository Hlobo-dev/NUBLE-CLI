# ════════════════════════════════════════════════════════════
#  NUBLE ELITE — Production Dockerfile
# ════════════════════════════════════════════════════════════
#  Multi-stage build optimized for ECS Fargate (2048 CPU / 8192 MB)
#  - Stage 1: Build + install all Python dependencies
#  - Stage 2: Minimal production image with S3 data sync
#  - Models baked into image (~50MB); WRDS data synced from S3 at startup
# ════════════════════════════════════════════════════════════

# ── Stage 1: Builder ──────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency definitions first for layer caching
COPY pyproject.toml setup.py ./
COPY src/ ./src/

# Install all dependencies (production + ML inference)
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -e ".[all]" && \
    pip install --no-cache-dir \
        uvicorn[standard] \
        gunicorn \
        redis \
        boto3 \
        aiobotocore \
        httpx \
        aiohttp \
        hmmlearn \
        transformers \
        sentencepiece \
        polygon-api-client \
        fredapi \
        pyarrow && \
    pip install --no-cache-dir \
        torch --index-url https://download.pytorch.org/whl/cpu

# ── Stage 2: Production Image ────────────────────────────
FROM python:3.11-slim AS production

# Install runtime deps: curl (healthcheck), awscli (S3 data sync)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && pip install --no-cache-dir awscli \
    && rm -rf /var/lib/apt/lists/*

# Security: non-root user
RUN groupadd -r nuble && useradd -r -g nuble -m nuble

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=nuble:nuble src/ ./src/
COPY --chown=nuble:nuble config/ ./config/

# Copy models (baked into image — LightGBM + MLP models ~50MB total)
COPY --chown=nuble:nuble models/ ./models/

# Create data directories (WRDS panel data synced from S3 at startup)
RUN mkdir -p /app/data/wrds /app/data/cache /app/data/historical \
             /app/data/train /app/data/test /app/logs \
    && chown -R nuble:nuble /app

# Copy entrypoint script
COPY --chown=nuble:nuble infrastructure/aws/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN chmod +x /app/docker-entrypoint.sh

# Health check — generous start period for S3 sync + model loading
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -sf http://localhost:8000/api/health || exit 1

EXPOSE 8000

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH="/app/src:/app" \
    TOKENIZERS_PARALLELISM=false \
    PORT=8000 \
    WORKERS=4

# Entrypoint handles S3 data sync, then exec's into gunicorn
ENTRYPOINT ["/app/docker-entrypoint.sh"]

# Default command — gunicorn with uvicorn async workers
CMD ["gunicorn", "nuble.api.roket:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "300", \
     "--graceful-timeout", "120", \
     "--keep-alive", "5", \
     "--max-requests", "1000", \
     "--max-requests-jitter", "100", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--capture-output", \
     "--preload"]
