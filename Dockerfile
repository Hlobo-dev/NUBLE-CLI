# ============================================
# KYPERIAN ELITE - Production Dockerfile
# ============================================
# Multi-stage build for minimal image size
# Target: <100MB, optimized for ECS Fargate
# ============================================

# Stage 1: Builder
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY pyproject.toml setup.py ./
COPY src/ ./src/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[production]" && \
    pip install --no-cache-dir \
    uvicorn[standard] \
    gunicorn \
    redis \
    boto3 \
    aiobotocore \
    httpx

# Stage 2: Production Image
FROM python:3.11-slim as production

# Security: Run as non-root user
RUN groupadd -r kyperian && useradd -r -g kyperian kyperian

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=kyperian:kyperian src/ ./src/
COPY --chown=kyperian:kyperian config/ ./config/
COPY --chown=kyperian:kyperian models/ ./models/

# Create data directories
RUN mkdir -p /app/data /app/logs && \
    chown -R kyperian:kyperian /app

# Switch to non-root user
USER kyperian

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000 \
    WORKERS=4

# Run with Gunicorn + Uvicorn workers for production
CMD ["gunicorn", "src.kyperian.api.main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120", \
     "--keep-alive", "5", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "--capture-output"]
