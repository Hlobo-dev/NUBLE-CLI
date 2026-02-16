#!/bin/bash
# ============================================
# CloudLobo Production Deployment Script
# Fast, Expert-Level Deployment to AWS ECS
# ============================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
AWS_ACCOUNT_ID="191613668206"
AWS_REGION="us-east-1"
ECR_REPOSITORY="nova-sonic-production"
ECS_CLUSTER="nova-sonic-cluster-production"
ECS_SERVICE="nova-sonic-service"
IMAGE_NAME="cloudlobo-fullstack"
DOCKERFILE="Dockerfile.fullstack"

# ECR URL
ECR_URL="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPOSITORY}"

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║         CloudLobo Production Deployment Script                   ║${NC}"
echo -e "${CYAN}║                    cloudlobo.com                                 ║${NC}"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${CYAN}║  Target: AWS ECS (${AWS_REGION})                                     ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to print step
print_step() {
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}▶ $1${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

# Function to print success
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Function to print error and exit
print_error() {
    echo -e "${RED}✗ ERROR: $1${NC}"
    exit 1
}

# Record start time
START_TIME=$(date +%s)

# Step 1: Pre-flight checks
print_step "Step 1/6: Pre-flight Checks"

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed"
fi
print_success "Docker is available"

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    print_error "AWS CLI is not installed"
fi
print_success "AWS CLI is available"

# Check AWS credentials
if ! aws sts get-caller-identity &> /dev/null; then
    print_error "AWS credentials not configured"
fi
print_success "AWS credentials valid"

# Check Docker daemon
if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running"
fi
print_success "Docker daemon is running"

# Step 2: ECR Login
print_step "Step 2/6: Logging into AWS ECR"

aws ecr get-login-password --region ${AWS_REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
print_success "Logged into ECR"

# Step 3: Build Docker Image
print_step "Step 3/6: Building Docker Image"
echo -e "${CYAN}This may take 3-5 minutes...${NC}"

docker build \
    --platform linux/amd64 \
    -f ${DOCKERFILE} \
    -t ${IMAGE_NAME}:latest \
    . 2>&1 | while IFS= read -r line; do
        # Show key progress indicators
        if [[ "$line" == *"Step"* ]] || [[ "$line" == *"FINISHED"* ]] || [[ "$line" == *"Successfully"* ]]; then
            echo -e "${CYAN}  $line${NC}"
        fi
    done

# Check if build was successful
if [ $? -eq 0 ]; then
    print_success "Docker image built successfully"
else
    print_error "Docker build failed"
fi

# Step 4: Tag and Push
print_step "Step 4/6: Tagging and Pushing to ECR"

docker tag ${IMAGE_NAME}:latest ${ECR_URL}:latest
print_success "Tagged image"

echo -e "${CYAN}Pushing to ECR (this may take 1-3 minutes)...${NC}"
docker push ${ECR_URL}:latest 2>&1 | while IFS= read -r line; do
    if [[ "$line" == *"Pushed"* ]] || [[ "$line" == *"digest"* ]]; then
        echo -e "${CYAN}  $line${NC}"
    fi
done
print_success "Image pushed to ECR"

# Step 5: Deploy to ECS
print_step "Step 5/6: Deploying to ECS"

aws ecs update-service \
    --cluster ${ECS_CLUSTER} \
    --service ${ECS_SERVICE} \
    --force-new-deployment \
    --region ${AWS_REGION} \
    --output text > /dev/null

print_success "Deployment triggered"

# Step 6: Wait for deployment
print_step "Step 6/6: Waiting for Deployment to Complete"
echo -e "${CYAN}Monitoring deployment status...${NC}"

MAX_ATTEMPTS=30
ATTEMPT=0

while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do
    ATTEMPT=$((ATTEMPT + 1))
    
    # Get deployment status
    DEPLOYMENT_STATUS=$(aws ecs describe-services \
        --cluster ${ECS_CLUSTER} \
        --services ${ECS_SERVICE} \
        --region ${AWS_REGION} \
        --query 'services[0].deployments[?status==`PRIMARY`].rolloutState' \
        --output text 2>/dev/null)
    
    RUNNING_COUNT=$(aws ecs describe-services \
        --cluster ${ECS_CLUSTER} \
        --services ${ECS_SERVICE} \
        --region ${AWS_REGION} \
        --query 'services[0].runningCount' \
        --output text 2>/dev/null)
    
    echo -e "  ${CYAN}[$ATTEMPT/$MAX_ATTEMPTS] Status: $DEPLOYMENT_STATUS, Running: $RUNNING_COUNT${NC}"
    
    if [ "$DEPLOYMENT_STATUS" == "COMPLETED" ]; then
        print_success "Deployment completed!"
        break
    fi
    
    if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then
        echo -e "${YELLOW}⚠ Deployment still in progress. Check AWS Console for final status.${NC}"
        break
    fi
    
    sleep 10
done

# Calculate deployment time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

# Final status
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    DEPLOYMENT COMPLETE                           ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  🌐 Website: https://cloudlobo.com                               ║${NC}"
echo -e "${GREEN}║  ⏱  Time: ${MINUTES}m ${SECONDS}s                                              ║${NC}"
echo -e "${GREEN}║  🐳 Image: ${ECR_URL}:latest   ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Quick health check
echo -e "${CYAN}Running quick health check...${NC}"
sleep 5
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" https://cloudlobo.com/ 2>/dev/null || echo "000")

if [ "$HTTP_STATUS" == "200" ]; then
    echo -e "${GREEN}✓ Site is responding (HTTP $HTTP_STATUS)${NC}"
else
    echo -e "${YELLOW}⚠ Site returned HTTP $HTTP_STATUS - may still be starting up${NC}"
fi

echo ""
echo -e "${CYAN}Done! Your changes are now live at https://cloudlobo.com${NC}"
echo ""
