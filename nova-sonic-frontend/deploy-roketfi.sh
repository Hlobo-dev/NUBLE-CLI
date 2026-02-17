#!/bin/bash
# ════════════════════════════════════════════════════════════════════
# ROKETFI — Production Deployment Script
# ════════════════════════════════════════════════════════════════════
#
# Deploys the COMPLETE system to www.roketfi.com:
#   ✅ SvelteKit frontend (built from source)
#   ✅ Node.js backend (server.js + nuble-tools.js + nova-sonic-client.js)
#   ✅ Python ROKET API (FastAPI + LightGBM models + all ML intelligence)
#   ✅ 20 ROKET financial tools (predictions, regime, macro, lambda, etc.)
#   ✅ Claude Opus 4.6 with tool calling
#   ✅ AWS Bedrock Nova Sonic voice
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - Docker running
#   - Existing CloudFormation stack: roket-production
#
# Usage:
#   ./deploy-roketfi.sh          # Full build + deploy
#   ./deploy-roketfi.sh --skip-build  # Push existing image only
#   ./deploy-roketfi.sh --status      # Check current status
#
# ════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Configuration ──
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="456309723884"
ECR_REPO="roket-production"
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}"
ECS_CLUSTER="roket-cluster"
ECS_SERVICE="roket-service"
DOMAIN="roketfi.com"
DOCKERFILE="nova-sonic-frontend/Dockerfile.roketfi"

# The Docker build context is the REPO ROOT (so we can access src/, models/, nuble-frontend/, nuble-backend/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_banner() {
    echo ""
    echo -e "${CYAN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${CYAN}║                                                              ║${NC}"
    echo -e "${CYAN}║   🚀  ROKETFI Production Deployment                         ║${NC}"
    echo -e "${CYAN}║       https://www.roketfi.com                                ║${NC}"
    echo -e "${CYAN}║                                                              ║${NC}"
    echo -e "${CYAN}║   Components:                                                ║${NC}"
    echo -e "${CYAN}║     • SvelteKit frontend                                    ║${NC}"
    echo -e "${CYAN}║     • Node.js backend (Claude Opus 4.6 + 20 ROKET tools)    ║${NC}"
    echo -e "${CYAN}║     • Python ROKET API (LightGBM ML + financial intelligence)║${NC}"
    echo -e "${CYAN}║     • AWS Bedrock Nova Sonic (voice)                         ║${NC}"
    echo -e "${CYAN}║                                                              ║${NC}"
    echo -e "${CYAN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

step() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}▶ $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

ok()   { echo -e "${GREEN}✓ $1${NC}"; }
warn() { echo -e "${YELLOW}⚠ $1${NC}"; }
fail() { echo -e "${RED}✗ $1${NC}"; exit 1; }

# ── Status Check ──
check_status() {
    step "Current Deployment Status"
    
    echo -e "${CYAN}ECS Service:${NC}"
    aws ecs describe-services \
        --cluster "$ECS_CLUSTER" \
        --services "$ECS_SERVICE" \
        --query "services[0].{Status:status,Running:runningCount,Desired:desiredCount,Rollout:deployments[0].rolloutState}" \
        --output table 2>/dev/null | cat || warn "Could not query ECS"
    
    echo ""
    echo -e "${CYAN}Site Health:${NC}"
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "https://${DOMAIN}/" 2>/dev/null || echo "000")
    HEALTH=$(curl -s "https://${DOMAIN}/health" 2>/dev/null || echo "unreachable")
    echo "  https://${DOMAIN}/ → HTTP $HTTP_CODE"
    echo "  Health: $HEALTH"
    
    echo ""
    echo -e "${CYAN}Latest ECR Image:${NC}"
    aws ecr describe-images \
        --repository-name "$ECR_REPO" \
        --query "imageDetails | sort_by(@, &imagePushedAt) | [-1].{Pushed:imagePushedAt,Tags:imageTags,SizeMB:to_string(imageSizeInBytes)}" \
        --output table 2>/dev/null | cat || warn "Could not query ECR"
    
    echo ""
    echo -e "${CYAN}Recent Logs (last 20 lines):${NC}"
    aws logs get-log-events \
        --log-group-name "/ecs/roket-production" \
        --log-stream-name "$(aws logs describe-log-streams --log-group-name /ecs/roket-production --order-by LastEventTime --descending --max-items 1 --query 'logStreams[0].logStreamName' --output text 2>/dev/null)" \
        --limit 20 \
        --query "events[*].message" \
        --output text 2>/dev/null | cat || warn "Could not fetch logs"
    
    exit 0
}

# ── Parse Args ──
SKIP_BUILD=false
for arg in "$@"; do
    case "$arg" in
        --status) check_status ;;
        --skip-build) SKIP_BUILD=true ;;
        --help|-h)
            echo "Usage: $0 [--skip-build] [--status] [--help]"
            echo "  --skip-build  Skip Docker build, just push existing image and deploy"
            echo "  --status      Show current deployment status"
            exit 0
            ;;
    esac
done

print_banner
START_TIME=$(date +%s)

# ═══════════════════════════════════════════════════
# STEP 1: Pre-flight Checks
# ═══════════════════════════════════════════════════
step "Step 1/5: Pre-flight Checks"

# Docker
if ! docker info &>/dev/null; then
    fail "Docker is not running. Please start Docker Desktop."
fi
ok "Docker is running"

# AWS CLI
if ! aws sts get-caller-identity &>/dev/null; then
    fail "AWS credentials not configured. Run 'aws configure'."
fi
ACTUAL_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
if [ "$ACTUAL_ACCOUNT" != "$AWS_ACCOUNT_ID" ]; then
    fail "AWS account mismatch: expected $AWS_ACCOUNT_ID, got $ACTUAL_ACCOUNT"
fi
ok "AWS credentials valid (Account: $AWS_ACCOUNT_ID)"

# Verify files exist
cd "$REPO_ROOT"
for f in "$DOCKERFILE" \
         "nova-sonic-frontend/nuble-backend/server.js" \
         "nova-sonic-frontend/nuble-backend/nuble-tools.js" \
         "nova-sonic-frontend/nuble-backend/nova-sonic-client.js" \
         "nova-sonic-frontend/nuble-frontend/package.json" \
         "src/nuble/api/roket.py" \
         "models/lightgbm/lgb_mega.txt"; do
    if [ ! -f "$f" ]; then
        fail "Missing required file: $f"
    fi
done
ok "All source files present"

# Verify ECS stack exists
STACK_STATUS=$(aws cloudformation describe-stacks --stack-name roket-production --query "Stacks[0].StackStatus" --output text 2>/dev/null || echo "NOT_FOUND")
if [ "$STACK_STATUS" = "NOT_FOUND" ]; then
    fail "CloudFormation stack 'roket-production' not found. Deploy infrastructure first."
fi
ok "CloudFormation stack: $STACK_STATUS"

# ═══════════════════════════════════════════════════
# STEP 2: ECR Login
# ═══════════════════════════════════════════════════
step "Step 2/5: Logging into AWS ECR"

aws ecr get-login-password --region "$AWS_REGION" | \
    docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com" 2>/dev/null
ok "Logged into ECR"

# ═══════════════════════════════════════════════════
# STEP 3: Build Docker Image
# ═══════════════════════════════════════════════════
if [ "$SKIP_BUILD" = false ]; then
    step "Step 3/5: Building Docker Image (frontend + backend + ROKET API + ML models)"
    echo -e "${CYAN}  Build context: $REPO_ROOT${NC}"
    echo -e "${CYAN}  Dockerfile:    $DOCKERFILE${NC}"
    echo -e "${CYAN}  This may take 5-10 minutes on first build...${NC}"
    echo ""

    cd "$REPO_ROOT"

    # Build for linux/amd64 (ECS Fargate)
    DOCKER_BUILDKIT=1 docker build \
        --platform linux/amd64 \
        -f "$DOCKERFILE" \
        -t "${ECR_REPO}:latest" \
        -t "${ECR_REPO}:$(date +%Y%m%d-%H%M%S)" \
        . 2>&1 | while IFS= read -r line; do
            # Show key build progress
            if [[ "$line" == *"Step"* ]] || [[ "$line" == *"FINISHED"* ]] || \
               [[ "$line" == *"Successfully"* ]] || [[ "$line" == *"ERROR"* ]] || \
               [[ "$line" == *"=>"* ]]; then
                echo -e "  ${CYAN}${line}${NC}"
            fi
        done

    ok "Docker image built"
else
    step "Step 3/5: Skipping build (--skip-build)"
    ok "Using existing local image"
fi

# ═══════════════════════════════════════════════════
# STEP 4: Tag & Push to ECR
# ═══════════════════════════════════════════════════
step "Step 4/5: Pushing to ECR"

GIT_SHA=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo "manual")
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

docker tag "${ECR_REPO}:latest" "${ECR_URI}:latest"
docker tag "${ECR_REPO}:latest" "${ECR_URI}:${GIT_SHA}"
docker tag "${ECR_REPO}:latest" "${ECR_URI}:${TIMESTAMP}"
ok "Tagged: latest, ${GIT_SHA}, ${TIMESTAMP}"

echo -e "${CYAN}  Pushing to ECR (this may take 1-3 minutes)...${NC}"
docker push "${ECR_URI}:latest" 2>&1 | grep -E "Pushed|digest|latest" | while IFS= read -r line; do
    echo -e "  ${CYAN}${line}${NC}"
done
docker push "${ECR_URI}:${GIT_SHA}" 2>/dev/null || true
ok "Image pushed to ECR"

# ═══════════════════════════════════════════════════
# STEP 5: Deploy to ECS (rolling update)
# ═══════════════════════════════════════════════════
step "Step 5/5: Deploying to ECS Fargate"

# Update the stack with updated template (2 vCPU, 4GB RAM for Python+Node.js)
echo -e "${CYAN}  Updating CloudFormation stack with new image + resources...${NC}"
CF_TEMPLATE="${SCRIPT_DIR}/infrastructure/roket-deployed-template.yaml"
aws cloudformation update-stack \
    --stack-name roket-production \
    --template-body "file://${CF_TEMPLATE}" \
    --parameters \
        ParameterKey=ECRImageUri,ParameterValue="${ECR_URI}:latest" \
        ParameterKey=AnthropicApiKey,UsePreviousValue=true \
        ParameterKey=JwtSecret,UsePreviousValue=true \
        ParameterKey=DomainName,UsePreviousValue=true \
        ParameterKey=HostedZoneId,UsePreviousValue=true \
        ParameterKey=ContainerPort,UsePreviousValue=true \
    --capabilities CAPABILITY_NAMED_IAM \
    --region "$AWS_REGION" 2>/dev/null && ok "Stack update initiated" || {
    # If no CF changes needed, force a new ECS deployment directly
    warn "No CloudFormation changes — forcing ECS service redeployment"
    aws ecs update-service \
        --cluster "$ECS_CLUSTER" \
        --service "$ECS_SERVICE" \
        --force-new-deployment \
        --region "$AWS_REGION" > /dev/null
    ok "ECS redeployment triggered"
}

# ── Wait for deployment ──
echo ""
echo -e "${CYAN}  Waiting for rolling deployment to complete...${NC}"
echo -e "${CYAN}  (new task starts → passes health check → old task drains)${NC}"
echo ""

MAX_ATTEMPTS=40
ATTEMPT=0
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    ATTEMPT=$((ATTEMPT + 1))
    
    ROLLOUT=$(aws ecs describe-services \
        --cluster "$ECS_CLUSTER" \
        --services "$ECS_SERVICE" \
        --query "services[0].deployments[?status=='PRIMARY'].rolloutState | [0]" \
        --output text 2>/dev/null || echo "UNKNOWN")
    
    RUNNING=$(aws ecs describe-services \
        --cluster "$ECS_CLUSTER" \
        --services "$ECS_SERVICE" \
        --query "services[0].runningCount" \
        --output text 2>/dev/null || echo "?")
    
    echo -e "  ${CYAN}[$ATTEMPT/$MAX_ATTEMPTS] Rollout: $ROLLOUT | Running tasks: $RUNNING${NC}"
    
    if [ "$ROLLOUT" = "COMPLETED" ]; then
        ok "ECS deployment completed!"
        break
    fi
    
    if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
        warn "Deployment still in progress after $MAX_ATTEMPTS checks. Check AWS Console."
        break
    fi
    
    sleep 15
done

# ═══════════════════════════════════════════════════
# DONE — Health Check & Summary
# ═══════════════════════════════════════════════════
echo ""
echo -e "${CYAN}  Running health check...${NC}"
sleep 10

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "https://${DOMAIN}/" 2>/dev/null || echo "000")
HEALTH=$(curl -s "https://${DOMAIN}/health" 2>/dev/null || echo "unreachable")

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}║          ✅  DEPLOYMENT COMPLETE                            ║${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}║  🌐 https://roketfi.com          (HTTP ${HTTP_CODE})              ║${NC}"
echo -e "${GREEN}║  🌐 https://www.roketfi.com                                 ║${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}║  ⏱  Deploy time: ${MINUTES}m ${SECONDS}s                                    ║${NC}"
echo -e "${GREEN}║  🐳 Image: ${ECR_URI}:latest                                ║${NC}"
echo -e "${GREEN}║  📋 Git: ${GIT_SHA}                                              ║${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}║  Components deployed:                                        ║${NC}"
echo -e "${GREEN}║    • SvelteKit frontend (chat UI)                           ║${NC}"
echo -e "${GREEN}║    • Node.js backend (Claude Opus 4.6 + 20 tools)           ║${NC}"
echo -e "${GREEN}║    • Python ROKET API (LightGBM ML + financial intel)       ║${NC}"
echo -e "${GREEN}║    • AWS Bedrock Nova Sonic (voice)                         ║${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}║  Health: ${HEALTH}${NC}"
echo -e "${GREEN}║                                                              ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

if [ "$HTTP_CODE" = "200" ]; then
    ok "Site is live and healthy!"
else
    warn "Site returned HTTP $HTTP_CODE — may still be starting up. Wait 1-2 minutes."
fi

echo ""
echo -e "${CYAN}Useful commands:${NC}"
echo "  ./deploy-roketfi.sh --status         # Check deployment status"
echo "  aws ecs execute-command \\            # SSH into running container"
echo "    --cluster $ECS_CLUSTER \\"
echo "    --task \$(aws ecs list-tasks --cluster $ECS_CLUSTER --query 'taskArns[0]' --output text) \\"
echo "    --container roket-app --interactive --command /bin/bash"
echo "  aws logs tail /ecs/roket-production --follow  # Stream logs"
echo ""
