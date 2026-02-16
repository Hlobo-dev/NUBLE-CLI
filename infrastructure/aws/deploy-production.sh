#!/bin/bash
# ════════════════════════════════════════════════════════════════════
#  NUBLE ELITE — Unified Production Deployment Script
# ════════════════════════════════════════════════════════════════════
#  Deploys the complete NUBLE infrastructure using master-stack.yaml:
#
#  Usage:
#    ./deploy-production.sh                  # Full deploy (infrastructure + app)
#    ./deploy-production.sh --infra-only     # Deploy only CloudFormation
#    ./deploy-production.sh --app-only       # Build + push Docker + update ECS
#    ./deploy-production.sh --upload-data    # Upload data/models to S3
#    ./deploy-production.sh --destroy        # Tear down everything
#
#  Prerequisites:
#    - AWS CLI v2 configured with proper credentials
#    - Docker running locally
#    - jq installed (brew install jq)
# ════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Configuration ─────────────────────────────────────────
STACK_NAME="${NUBLE_STACK_NAME:-nuble-elite}"
REGION="${AWS_REGION:-us-east-1}"
ENVIRONMENT="${NUBLE_ENV:-production}"
PROJECT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
TEMPLATE="${PROJECT_ROOT}/infrastructure/aws/cloudformation/master-stack.yaml"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log()   { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[⚠]${NC} $*"; }
error() { echo -e "${RED}[✗]${NC} $*"; exit 1; }
info()  { echo -e "${BLUE}[→]${NC} $*"; }
header(){ echo -e "\n${CYAN}════════════════════════════════════════════════${NC}"; echo -e "${CYAN} $*${NC}"; echo -e "${CYAN}════════════════════════════════════════════════${NC}"; }

# ── Preflight Checks ─────────────────────────────────────
preflight() {
    header "Preflight Checks"

    # AWS CLI
    if ! command -v aws &>/dev/null; then
        error "AWS CLI not found. Install: brew install awscli"
    fi
    log "AWS CLI $(aws --version 2>&1 | head -1)"

    # Docker
    if ! command -v docker &>/dev/null; then
        error "Docker not found. Install Docker Desktop."
    fi
    if ! docker info &>/dev/null; then
        error "Docker daemon not running. Start Docker Desktop."
    fi
    log "Docker running"

    # jq
    if ! command -v jq &>/dev/null; then
        warn "jq not found — install with: brew install jq"
    fi

    # AWS credentials
    AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text 2>/dev/null) \
        || error "AWS credentials not configured. Run: aws configure"
    AWS_USER=$(aws sts get-caller-identity --query Arn --output text 2>/dev/null)
    log "AWS Account: ${AWS_ACCOUNT}"
    log "AWS Identity: ${AWS_USER}"

    # Template exists
    [ -f "$TEMPLATE" ] || error "Master stack template not found: $TEMPLATE"
    log "Template: $TEMPLATE"

    # Validate template
    info "Validating CloudFormation template..."
    aws cloudformation validate-template \
        --template-body "file://${TEMPLATE}" \
        --region "$REGION" &>/dev/null \
        || error "Template validation failed!"
    log "Template valid"
}

# ── Deploy Infrastructure ────────────────────────────────
deploy_infrastructure() {
    header "Deploying Infrastructure — ${STACK_NAME}"

    # Check if stack exists
    STACK_STATUS=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].StackStatus' \
        --output text 2>/dev/null || echo "DOES_NOT_EXIST")

    info "Current stack status: ${STACK_STATUS}"

    # Determine action
    if [ "$STACK_STATUS" = "DOES_NOT_EXIST" ]; then
        ACTION="create-stack"
        WAIT="stack-create-complete"
    elif [[ "$STACK_STATUS" == *ROLLBACK_COMPLETE* ]]; then
        warn "Stack in ROLLBACK_COMPLETE — deleting and recreating..."
        aws cloudformation delete-stack --stack-name "$STACK_NAME" --region "$REGION"
        aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME" --region "$REGION"
        ACTION="create-stack"
        WAIT="stack-create-complete"
    else
        ACTION="update-stack"
        WAIT="stack-update-complete"
    fi

    info "Action: ${ACTION}"
    info "Deploying with Environment=${ENVIRONMENT}..."

    # Get parameters interactively if needed
    PARAMS=(
        "ParameterKey=Environment,ParameterValue=${ENVIRONMENT}"
        "ParameterKey=ProjectName,ParameterValue=nuble"
    )

    # Optional: Certificate ARN for HTTPS
    if [ -n "${ACM_CERTIFICATE_ARN:-}" ]; then
        PARAMS+=("ParameterKey=CertificateArn,ParameterValue=${ACM_CERTIFICATE_ARN}")
    fi

    # Optional: Alert email
    if [ -n "${ALERT_EMAIL:-}" ]; then
        PARAMS+=("ParameterKey=AlertEmail,ParameterValue=${ALERT_EMAIL}")
    fi

    # Deploy
    aws cloudformation ${ACTION} \
        --stack-name "$STACK_NAME" \
        --template-body "file://${TEMPLATE}" \
        --parameters "${PARAMS[@]}" \
        --capabilities CAPABILITY_NAMED_IAM \
        --region "$REGION" \
        --tags \
            "Key=Project,Value=nuble-elite" \
            "Key=Environment,Value=${ENVIRONMENT}" \
            "Key=ManagedBy,Value=cloudformation" \
        2>/dev/null || {
            if [ "$ACTION" = "update-stack" ]; then
                warn "No updates to be performed (stack is current)"
                return 0
            else
                error "Stack deployment failed!"
            fi
        }

    info "Waiting for stack to complete (this can take 10-15 minutes)..."
    aws cloudformation wait "$WAIT" \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        || error "Stack deployment failed! Check CloudFormation console."

    log "Infrastructure deployed successfully!"

    # Print outputs
    header "Stack Outputs"
    aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
        --output table
}

# ── Build & Push Docker Image ─────────────────────────────
build_and_push() {
    header "Building & Pushing Docker Image"

    # Get ECR URI from stack outputs
    ECR_URI=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='ECRRepositoryUri'].OutputValue" \
        --output text 2>/dev/null)

    if [ -z "$ECR_URI" ] || [ "$ECR_URI" = "None" ]; then
        error "Could not get ECR URI from stack outputs. Deploy infrastructure first."
    fi

    ECR_REGISTRY="${ECR_URI%%/*}"
    IMAGE_TAG="$(git -C "$PROJECT_ROOT" rev-parse --short HEAD 2>/dev/null || date +%Y%m%d%H%M%S)"

    info "ECR URI: ${ECR_URI}"
    info "Image tag: ${IMAGE_TAG}"

    # Login to ECR
    info "Authenticating with ECR..."
    aws ecr get-login-password --region "$REGION" \
        | docker login --username AWS --password-stdin "$ECR_REGISTRY"
    log "ECR login successful"

    # Build image
    info "Building Docker image (this may take several minutes)..."
    cd "$PROJECT_ROOT"

    docker build \
        --platform linux/amd64 \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        --tag "${ECR_URI}:${IMAGE_TAG}" \
        --tag "${ECR_URI}:latest" \
        --file Dockerfile \
        .

    log "Docker image built"

    # Push image
    info "Pushing to ECR..."
    docker push "${ECR_URI}:${IMAGE_TAG}"
    docker push "${ECR_URI}:latest"
    log "Image pushed: ${ECR_URI}:${IMAGE_TAG}"
    log "Image pushed: ${ECR_URI}:latest"
}

# ── Update ECS Service ────────────────────────────────────
update_ecs() {
    header "Updating ECS Service"

    CLUSTER_NAME=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='ECSClusterName'].OutputValue" \
        --output text 2>/dev/null)

    SERVICE_NAME=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='ECSServiceName'].OutputValue" \
        --output text 2>/dev/null)

    if [ -z "$CLUSTER_NAME" ] || [ "$CLUSTER_NAME" = "None" ]; then
        error "Could not get ECS cluster name from stack outputs"
    fi

    info "Cluster: ${CLUSTER_NAME}"
    info "Service: ${SERVICE_NAME}"

    # Force new deployment (pulls latest image)
    aws ecs update-service \
        --cluster "$CLUSTER_NAME" \
        --service "$SERVICE_NAME" \
        --force-new-deployment \
        --region "$REGION" \
        --no-cli-pager

    log "ECS service update triggered"

    info "Waiting for deployment to stabilize..."
    aws ecs wait services-stable \
        --cluster "$CLUSTER_NAME" \
        --services "$SERVICE_NAME" \
        --region "$REGION" \
        || warn "Deployment may still be in progress — check ECS console"

    log "ECS deployment complete!"
}

# ── Upload Data to S3 ────────────────────────────────────
upload_data() {
    header "Uploading Data & Models to S3"

    DATA_BUCKET=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='DataBucketName'].OutputValue" \
        --output text 2>/dev/null)

    if [ -z "$DATA_BUCKET" ] || [ "$DATA_BUCKET" = "None" ]; then
        error "Could not get S3 data bucket from stack outputs"
    fi

    info "S3 Bucket: ${DATA_BUCKET}"

    # Upload WRDS data
    if [ -d "${PROJECT_ROOT}/data/wrds" ]; then
        info "Uploading WRDS panel data..."
        aws s3 sync "${PROJECT_ROOT}/data/wrds/" "s3://${DATA_BUCKET}/data/wrds/" \
            --region "$REGION" \
            --storage-class INTELLIGENT_TIERING \
            --only-show-errors
        log "WRDS data uploaded"
    else
        warn "No data/wrds/ directory found — skipping"
    fi

    # Upload historical data
    if [ -d "${PROJECT_ROOT}/data/historical" ]; then
        info "Uploading historical data..."
        aws s3 sync "${PROJECT_ROOT}/data/historical/" "s3://${DATA_BUCKET}/data/historical/" \
            --region "$REGION" \
            --only-show-errors
        log "Historical data uploaded"
    fi

    # Upload cache data
    if [ -d "${PROJECT_ROOT}/data_cache" ]; then
        info "Uploading cached data..."
        aws s3 sync "${PROJECT_ROOT}/data_cache/" "s3://${DATA_BUCKET}/data/cache/" \
            --region "$REGION" \
            --only-show-errors
        log "Cache data uploaded"
    fi

    # Upload models
    if [ -d "${PROJECT_ROOT}/models" ]; then
        info "Uploading models..."
        aws s3 sync "${PROJECT_ROOT}/models/" "s3://${DATA_BUCKET}/models/" \
            --region "$REGION" \
            --only-show-errors
        log "Models uploaded"
    fi

    # Show bucket size
    info "Bucket contents:"
    aws s3 ls "s3://${DATA_BUCKET}/" --summarize --recursive \
        | tail -2
}

# ── Destroy Stack ─────────────────────────────────────────
destroy() {
    header "⚠️  DESTROYING Stack: ${STACK_NAME}"

    echo -e "${RED}WARNING: This will delete ALL infrastructure including:${NC}"
    echo "  - VPC and all networking"
    echo "  - ECS cluster and services"
    echo "  - DynamoDB tables (DATA WILL BE LOST)"
    echo "  - ElastiCache Redis"
    echo "  - ECR repository and images"
    echo "  - S3 buckets (must be empty first)"
    echo ""
    read -rp "Type the stack name to confirm: " CONFIRM

    if [ "$CONFIRM" != "$STACK_NAME" ]; then
        error "Confirmation failed. Aborting."
    fi

    # Empty S3 buckets first
    DATA_BUCKET=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query "Stacks[0].Outputs[?OutputKey=='DataBucketName'].OutputValue" \
        --output text 2>/dev/null || true)

    if [ -n "$DATA_BUCKET" ] && [ "$DATA_BUCKET" != "None" ]; then
        warn "Emptying S3 bucket: ${DATA_BUCKET}"
        aws s3 rm "s3://${DATA_BUCKET}" --recursive --region "$REGION" || true
    fi

    # Delete stack
    info "Deleting CloudFormation stack..."
    aws cloudformation delete-stack \
        --stack-name "$STACK_NAME" \
        --region "$REGION"

    info "Waiting for deletion..."
    aws cloudformation wait stack-delete-complete \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        || error "Stack deletion failed — check CloudFormation console"

    log "Stack destroyed."
}

# ── Print Status ──────────────────────────────────────────
status() {
    header "Stack Status: ${STACK_NAME}"

    STACK_STATUS=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME" \
        --region "$REGION" \
        --query 'Stacks[0].StackStatus' \
        --output text 2>/dev/null || echo "DOES_NOT_EXIST")

    echo "Status: ${STACK_STATUS}"
    echo ""

    if [ "$STACK_STATUS" != "DOES_NOT_EXIST" ]; then
        echo "Outputs:"
        aws cloudformation describe-stacks \
            --stack-name "$STACK_NAME" \
            --region "$REGION" \
            --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
            --output table

        # ECS service status
        CLUSTER_NAME=$(aws cloudformation describe-stacks \
            --stack-name "$STACK_NAME" \
            --region "$REGION" \
            --query "Stacks[0].Outputs[?OutputKey=='ECSClusterName'].OutputValue" \
            --output text 2>/dev/null || true)

        SERVICE_NAME=$(aws cloudformation describe-stacks \
            --stack-name "$STACK_NAME" \
            --region "$REGION" \
            --query "Stacks[0].Outputs[?OutputKey=='ECSServiceName'].OutputValue" \
            --output text 2>/dev/null || true)

        if [ -n "$CLUSTER_NAME" ] && [ "$CLUSTER_NAME" != "None" ]; then
            echo ""
            echo "ECS Service:"
            aws ecs describe-services \
                --cluster "$CLUSTER_NAME" \
                --services "$SERVICE_NAME" \
                --region "$REGION" \
                --query 'services[0].{Status:status,Running:runningCount,Desired:desiredCount,Pending:pendingCount}' \
                --output table
        fi
    fi
}

# ── Main ──────────────────────────────────────────────────
main() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║          NUBLE ELITE — Production Deployment            ║"
    echo "║                                                        ║"
    echo "║  Stack:       ${STACK_NAME}                              "
    echo "║  Region:      ${REGION}                                  "
    echo "║  Environment: ${ENVIRONMENT}                             "
    echo "╚══════════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    case "${1:-}" in
        --infra-only)
            preflight
            deploy_infrastructure
            ;;
        --app-only)
            preflight
            build_and_push
            update_ecs
            ;;
        --upload-data)
            preflight
            upload_data
            ;;
        --status)
            status
            ;;
        --destroy)
            preflight
            destroy
            ;;
        --help|-h)
            echo "Usage: $0 [option]"
            echo ""
            echo "Options:"
            echo "  (no option)      Full deploy: infra + build + push + update ECS"
            echo "  --infra-only     Deploy CloudFormation infrastructure only"
            echo "  --app-only       Build Docker image, push to ECR, update ECS"
            echo "  --upload-data    Upload data/ and models/ to S3"
            echo "  --status         Show current stack status"
            echo "  --destroy        Tear down all infrastructure"
            echo "  --help           Show this help"
            echo ""
            echo "Environment Variables:"
            echo "  AWS_REGION           AWS region (default: us-east-1)"
            echo "  NUBLE_STACK_NAME     CloudFormation stack name (default: nuble-elite)"
            echo "  NUBLE_ENV            Environment: production|staging (default: production)"
            echo "  ACM_CERTIFICATE_ARN  ACM certificate ARN for HTTPS"
            echo "  ALERT_EMAIL          Email for CloudWatch alarm notifications"
            ;;
        "")
            # Full deploy
            preflight
            deploy_infrastructure
            upload_data
            build_and_push
            update_ecs

            header "🚀 Deployment Complete!"

            ALB_URL=$(aws cloudformation describe-stacks \
                --stack-name "$STACK_NAME" \
                --region "$REGION" \
                --query "Stacks[0].Outputs[?OutputKey=='ALBDNS'].OutputValue" \
                --output text 2>/dev/null || echo "unknown")

            echo -e "${GREEN}API Endpoint: http://${ALB_URL}/api/health${NC}"
            echo ""
            echo "Next steps:"
            echo "  1. Point your domain to the ALB: ${ALB_URL}"
            echo "  2. Create an ACM certificate for HTTPS"
            echo "  3. Subscribe to alerts: check your email for SNS confirmation"
            echo "  4. Test: curl http://${ALB_URL}/api/health"
            ;;
        *)
            error "Unknown option: $1 (use --help)"
            ;;
    esac
}

main "$@"
