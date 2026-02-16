#!/bin/bash
# ════════════════════════════════════════════════════════════
#  NUBLE ELITE — Docker Entrypoint
# ════════════════════════════════════════════════════════════
#  Runs BEFORE the main application starts:
#  1. Syncs WRDS panel data from S3 (15GB parquet)
#  2. Fetches secrets from AWS Secrets Manager
#  3. Execs into gunicorn (PID 1 handoff)
# ════════════════════════════════════════════════════════════
set -euo pipefail

echo "═══════════════════════════════════════════════"
echo " NUBLE ELITE — Container Starting"
echo " $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "═══════════════════════════════════════════════"

# ── 1. Resolve Secrets from AWS Secrets Manager ──────────
if [ -n "${NUBLE_SECRETS_ARN:-}" ]; then
    echo "[entrypoint] Fetching secrets from Secrets Manager..."
    REGION="${AWS_REGION:-us-east-1}"

    SECRETS_JSON=$(aws secretsmanager get-secret-value \
        --secret-id "$NUBLE_SECRETS_ARN" \
        --region "$REGION" \
        --query SecretString \
        --output text 2>/dev/null || true)

    if [ -n "$SECRETS_JSON" ]; then
        # Export each key-value pair as an environment variable
        export POLYGON_API_KEY=$(echo "$SECRETS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('POLYGON_API_KEY',''))" 2>/dev/null || true)
        export STOCKNEWS_API_KEY=$(echo "$SECRETS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('STOCKNEWS_API_KEY',''))" 2>/dev/null || true)
        export ANTHROPIC_API_KEY=$(echo "$SECRETS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('ANTHROPIC_API_KEY',''))" 2>/dev/null || true)
        export LAMBDA_API_URL=$(echo "$SECRETS_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('LAMBDA_API_URL',''))" 2>/dev/null || true)
        echo "[entrypoint] ✓ Secrets loaded from Secrets Manager"
    else
        echo "[entrypoint] ⚠ Could not fetch secrets — using existing env vars"
    fi
else
    echo "[entrypoint] No NUBLE_SECRETS_ARN set — using existing env vars"
fi

# ── 2. Sync WRDS Panel Data from S3 ─────────────────────
S3_DATA_BUCKET="${NUBLE_DATA_BUCKET:-}"

if [ -n "$S3_DATA_BUCKET" ]; then
    echo "[entrypoint] Syncing WRDS panel data from s3://${S3_DATA_BUCKET}/data/wrds/ ..."

    # Sync only WRDS parquet data (largest dataset)
    aws s3 sync "s3://${S3_DATA_BUCKET}/data/wrds/" /app/data/wrds/ \
        --region "${AWS_REGION:-us-east-1}" \
        --only-show-errors \
        --size-only \
        || echo "[entrypoint] ⚠ WRDS sync had errors (continuing anyway)"

    # Sync historical data
    aws s3 sync "s3://${S3_DATA_BUCKET}/data/historical/" /app/data/historical/ \
        --region "${AWS_REGION:-us-east-1}" \
        --only-show-errors \
        --size-only \
        || echo "[entrypoint] ⚠ Historical data sync had errors (continuing anyway)"

    # Sync cache data  
    aws s3 sync "s3://${S3_DATA_BUCKET}/data/cache/" /app/data/cache/ \
        --region "${AWS_REGION:-us-east-1}" \
        --only-show-errors \
        --size-only \
        || echo "[entrypoint] ⚠ Cache sync had errors (continuing anyway)"

    DATA_SIZE=$(du -sh /app/data/ 2>/dev/null | cut -f1 || echo "unknown")
    echo "[entrypoint] ✓ Data sync complete — total: ${DATA_SIZE}"
else
    echo "[entrypoint] No NUBLE_DATA_BUCKET set — skipping S3 data sync"
    echo "[entrypoint] (data must be mounted via volume or already in image)"
fi

# ── 3. Sync Models from S3 (if newer versions exist) ────
if [ -n "$S3_DATA_BUCKET" ]; then
    echo "[entrypoint] Checking for updated models in S3..."
    aws s3 sync "s3://${S3_DATA_BUCKET}/models/" /app/models/ \
        --region "${AWS_REGION:-us-east-1}" \
        --only-show-errors \
        --size-only \
        || echo "[entrypoint] ⚠ Model sync had errors (using baked-in models)"
    
    MODEL_COUNT=$(find /app/models -name "*.pt" -o -name "*.pkl" -o -name "*.joblib" 2>/dev/null | wc -l | tr -d ' ')
    echo "[entrypoint] ✓ ${MODEL_COUNT} model files available"
fi

# ── 4. Redis Connection Check ────────────────────────────
if [ -n "${REDIS_URL:-}" ]; then
    echo "[entrypoint] Redis URL configured: ${REDIS_URL}"
fi

# ── 5. Environment Summary ──────────────────────────────
echo ""
echo "═══════════════════════════════════════════════"
echo " Environment Summary"
echo "═══════════════════════════════════════════════"
echo " PYTHONPATH:  ${PYTHONPATH:-not set}"
echo " PORT:        ${PORT:-8000}"
echo " WORKERS:     ${WORKERS:-4}"
echo " DATA_BUCKET: ${NUBLE_DATA_BUCKET:-not set}"
echo " REDIS:       ${REDIS_URL:-not set}"
echo " POLYGON:     ${POLYGON_API_KEY:+configured}${POLYGON_API_KEY:-NOT SET}"
echo " ANTHROPIC:   ${ANTHROPIC_API_KEY:+configured}${ANTHROPIC_API_KEY:-NOT SET}"
echo " LAMBDA_URL:  ${LAMBDA_API_URL:+configured}${LAMBDA_API_URL:-NOT SET}"
echo "═══════════════════════════════════════════════"
echo ""
echo "[entrypoint] Starting application..."
echo ""

# ── 6. Exec into CMD (gunicorn becomes PID 1) ───────────
exec "$@"
