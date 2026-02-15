#!/bin/bash
# =============================================================================
# NUBLE-CLI: Deploy ALL data + models to S3
# Bucket: s3://nuble-data-warehouse
# =============================================================================

set -e

BUCKET="s3://nuble-data-warehouse"
PROJECT_DIR="$HOME/Desktop/NUBLE-CLI"
REGION="us-east-1"

echo "============================================"
echo "  NUBLE-CLI → AWS S3 Data Deployment"
echo "  Bucket: $BUCKET"
echo "============================================"
echo ""

# Verify AWS credentials
echo "🔐 Verifying AWS credentials..."
ACCOUNT=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)
if [ -z "$ACCOUNT" ]; then
    echo "❌ AWS credentials not configured. Run 'aws configure' first."
    exit 1
fi
echo "✅ Authenticated as account: $ACCOUNT"
echo ""

# ─────────────────────────────────────────────
# 1. Sync WRDS data (raw + derived)
# ─────────────────────────────────────────────
echo "📦 [1/5] Syncing data/wrds/ → s3://nuble-data-warehouse/data/wrds/"
echo "    This is ~16 GB. Only new/changed files will be uploaded."
echo ""

aws s3 sync "$PROJECT_DIR/data/wrds/" "$BUCKET/data/wrds/" \
    --exclude "*.DS_Store" \
    --storage-class INTELLIGENT_TIERING \
    --region "$REGION"

echo "✅ WRDS data synced."
echo ""

# ─────────────────────────────────────────────
# 2. Upload models
# ─────────────────────────────────────────────
echo "📦 [2/5] Syncing models/ → s3://nuble-data-warehouse/models/"

aws s3 sync "$PROJECT_DIR/models/" "$BUCKET/models/" \
    --exclude "*.DS_Store" \
    --storage-class INTELLIGENT_TIERING \
    --region "$REGION"

echo "✅ Models synced."
echo ""

# ─────────────────────────────────────────────
# 3. Upload wrds_pipeline code (for reproducibility)
# ─────────────────────────────────────────────
echo "📦 [3/5] Syncing wrds_pipeline/ → s3://nuble-data-warehouse/pipeline/"

aws s3 sync "$PROJECT_DIR/wrds_pipeline/" "$BUCKET/pipeline/" \
    --exclude "*.DS_Store" \
    --exclude "__pycache__/*" \
    --exclude "*.pyc" \
    --storage-class INTELLIGENT_TIERING \
    --region "$REGION"

echo "✅ Pipeline code synced."
echo ""

# ─────────────────────────────────────────────
# 4. Upload training results & validation
# ─────────────────────────────────────────────
echo "📦 [4/5] Syncing training_results/ and validation/ → S3"

if [ -d "$PROJECT_DIR/training_results" ]; then
    aws s3 sync "$PROJECT_DIR/training_results/" "$BUCKET/training_results/" \
        --exclude "*.DS_Store" \
        --storage-class INTELLIGENT_TIERING \
        --region "$REGION"
fi

if [ -d "$PROJECT_DIR/validation_results" ]; then
    aws s3 sync "$PROJECT_DIR/validation_results/" "$BUCKET/validation_results/" \
        --exclude "*.DS_Store" \
        --storage-class INTELLIGENT_TIERING \
        --region "$REGION"
fi

echo "✅ Results synced."
echo ""

# ─────────────────────────────────────────────
# 5. Create a manifest of everything in the bucket
# ─────────────────────────────────────────────
echo "📦 [5/5] Creating bucket manifest..."

aws s3 ls "$BUCKET/" --recursive --summarize > /tmp/nuble_s3_manifest.txt 2>&1

TOTAL_FILES=$(grep "Total Objects:" /tmp/nuble_s3_manifest.txt | awk '{print $3}')
TOTAL_SIZE=$(grep "Total Size:" /tmp/nuble_s3_manifest.txt | awk '{print $3}')
TOTAL_SIZE_GB=$(echo "scale=2; $TOTAL_SIZE / 1073741824" | bc 2>/dev/null || echo "N/A")

echo ""
echo "============================================"
echo "  ✅ DEPLOYMENT COMPLETE"
echo "============================================"
echo ""
echo "  Bucket:      $BUCKET"
echo "  Total files:  $TOTAL_FILES"
echo "  Total size:   ${TOTAL_SIZE_GB} GB"
echo ""
echo "  Structure:"
echo "  ├── data/wrds/        — All parquet data (raw + derived)"
echo "  ├── models/           — ML models (.pt, .txt, .pkl)"
echo "  ├── pipeline/         — WRDS pipeline code"
echo "  ├── training_results/ — Training outputs"
echo "  └── validation_results/ — Validation outputs"
echo ""
echo "  💰 Cost estimate (S3 Intelligent Tiering):"
echo "  ~16 GB × \$0.023/GB/month = ~\$0.37/month for storage"
echo "  Data accessed frequently stays in hot tier automatically."
echo ""
echo "  🔗 To download on another machine:"
echo "  aws s3 sync s3://nuble-data-warehouse/data/wrds/ data/wrds/"
echo "  aws s3 sync s3://nuble-data-warehouse/models/ models/"
echo ""
