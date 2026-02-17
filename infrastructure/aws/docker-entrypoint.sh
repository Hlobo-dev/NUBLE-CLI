#!/bin/bash
# ════════════════════════════════════════════════════════════
#  NUBLE ELITE — Docker Entrypoint
# ════════════════════════════════════════════════════════════
#  Runs BEFORE the main application starts:
#  1. Syncs WRDS panel data from S3 (15GB parquet)
#  2. Fetches secrets from AWS Secrets Manager
#  3. Execs into gunicorn (PID 1 handoff)
# ════════════════════════════════════════════════════════════
set -uo pipefail
# NOTE: -e is intentionally omitted so S3 sync failures don't crash the container.
# The app should start in degraded mode and self-heal when data becomes available.

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

# ── 6. Verify critical dependencies ─────────────────────
echo "[entrypoint] Verifying critical dependencies..."
python3 -c "
import sys
errors = []
try:
    import lightgbm
    print(f'  ✓ LightGBM {lightgbm.__version__}')
except ImportError as e:
    errors.append(f'LightGBM: {e}')
    print(f'  ✗ LightGBM MISSING: {e}')
try:
    import numpy
    print(f'  ✓ NumPy {numpy.__version__}')
except ImportError as e:
    errors.append(f'NumPy: {e}')
try:
    import pandas
    print(f'  ✓ Pandas {pandas.__version__}')
except ImportError as e:
    errors.append(f'Pandas: {e}')
try:
    import torch
    print(f'  ✓ PyTorch {torch.__version__}')
except ImportError as e:
    errors.append(f'PyTorch: {e}')
try:
    import hmmlearn
    print(f'  ✓ hmmlearn {hmmlearn.__version__}')
except ImportError as e:
    errors.append(f'hmmlearn: {e}')

# Check data
import os
wrds_dir = '/app/data/wrds'
if os.path.isdir(wrds_dir):
    files = [f for f in os.listdir(wrds_dir) if f.endswith('.parquet')]
    print(f'  ✓ WRDS data: {len(files)} parquet files')
    if os.path.exists(os.path.join(wrds_dir, 'gkx_panel.parquet')):
        size_mb = os.path.getsize(os.path.join(wrds_dir, 'gkx_panel.parquet')) / 1e6
        print(f'    gkx_panel.parquet: {size_mb:.0f} MB')
else:
    print(f'  ⚠ WRDS data directory missing: {wrds_dir}')

# Check models
models_dir = '/app/models'
for subdir in ['lightgbm', 'production', 'regime', 'universal']:
    sd = os.path.join(models_dir, subdir)
    if os.path.isdir(sd):
        files = os.listdir(sd)
        print(f'  ✓ models/{subdir}: {len(files)} files')
    else:
        print(f'  ⚠ models/{subdir} missing')

if errors:
    print(f'  ⚠ {len(errors)} dependency issues (app may run in degraded mode)')
else:
    print('  ✓ All critical dependencies OK')
" 2>&1 || echo "[entrypoint] ⚠ Dependency check script failed (continuing)"

echo ""

# ── 7. Exec into CMD (gunicorn becomes PID 1) ───────────
exec "$@"
