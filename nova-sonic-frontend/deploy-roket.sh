#!/bin/bash
# ============================================================
# ROKET Production Deployment
# Builds, pushes, and deploys to AWS ECS Fargate
# ============================================================
set -euo pipefail

# ---- Configuration ----
AWS_REGION="${AWS_REGION:-us-east-1}"
ENVIRONMENT="${ENVIRONMENT:-production}"
PROJECT_NAME="roket"
STACK_NAME="${PROJECT_NAME}-${ENVIRONMENT}"

# Auto-detect AWS Account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PROJECT_NAME}/${ENVIRONMENT}"

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║          ROKET — Production Deployment             ║"
echo "╠════════════════════════════════════════════════════╣"
echo "║  Account:  ${AWS_ACCOUNT_ID}                      "
echo "║  Region:   ${AWS_REGION}                           "
echo "║  Stack:    ${STACK_NAME}                           "
echo "║  ECR:      ${ECR_URI}                              "
echo "╚════════════════════════════════════════════════════╝"
echo ""

# ---- Step 1: Create Secrets (first-time only) ----
echo "Step 1: Checking Secrets Manager..."

ANTHROPIC_SECRET_ARN=""
JWT_SECRET_ARN=""

# Check if secrets exist
ANTHROPIC_SECRET_ARN=$(aws secretsmanager describe-secret \
  --secret-id "${PROJECT_NAME}/anthropic-api-key" \
  --query 'ARN' --output text 2>/dev/null || echo "")

if [ -z "$ANTHROPIC_SECRET_ARN" ] || [ "$ANTHROPIC_SECRET_ARN" = "None" ]; then
  echo "  Creating Anthropic API key secret..."
  read -sp "  Enter your ANTHROPIC_API_KEY: " ANTHROPIC_KEY
  echo ""
  ANTHROPIC_SECRET_ARN=$(aws secretsmanager create-secret \
    --name "${PROJECT_NAME}/anthropic-api-key" \
    --secret-string "$ANTHROPIC_KEY" \
    --query 'ARN' --output text)
  echo "  Created: $ANTHROPIC_SECRET_ARN"
else
  echo "  Anthropic secret exists: $ANTHROPIC_SECRET_ARN"
fi

JWT_SECRET_ARN=$(aws secretsmanager describe-secret \
  --secret-id "${PROJECT_NAME}/jwt-secret" \
  --query 'ARN' --output text 2>/dev/null || echo "")

if [ -z "$JWT_SECRET_ARN" ] || [ "$JWT_SECRET_ARN" = "None" ]; then
  echo "  Creating JWT secret (auto-generated)..."
  JWT_VALUE=$(openssl rand -base64 48)
  JWT_SECRET_ARN=$(aws secretsmanager create-secret \
    --name "${PROJECT_NAME}/jwt-secret" \
    --secret-string "$JWT_VALUE" \
    --query 'ARN' --output text)
  echo "  Created: $JWT_SECRET_ARN"
else
  echo "  JWT secret exists: $JWT_SECRET_ARN"
fi

# ---- Step 2: Deploy Infrastructure (if needed) ----
echo ""
echo "Step 2: Deploying CloudFormation infrastructure..."
aws cloudformation deploy \
  --template-file infrastructure/roket-production.yaml \
  --stack-name "$STACK_NAME" \
  --parameter-overrides \
    Environment="$ENVIRONMENT" \
    ProjectName="$PROJECT_NAME" \
    AnthropicApiKeyArn="$ANTHROPIC_SECRET_ARN" \
    JwtSecretArn="$JWT_SECRET_ARN" \
  --capabilities CAPABILITY_NAMED_IAM \
  --region "$AWS_REGION" \
  --no-fail-on-empty-changeset

# ---- Step 3: Build & Push Docker Image ----
echo ""
echo "Step 3: Building and pushing Docker image..."

# Login to ECR
aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Build with buildkit for faster builds
DOCKER_BUILDKIT=1 docker build \
  --platform linux/amd64 \
  -t "${ECR_URI}:latest" \
  -t "${ECR_URI}:$(git rev-parse --short HEAD 2>/dev/null || echo 'manual')" \
  .

# Push
docker push "${ECR_URI}:latest"
docker push "${ECR_URI}:$(git rev-parse --short HEAD 2>/dev/null || echo 'manual')" 2>/dev/null || true

# ---- Step 4: Force New Deployment ----
echo ""
echo "Step 4: Deploying new image to ECS..."

CLUSTER_NAME=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='ClusterName'].OutputValue" \
  --output text)

SERVICE_NAME=$(aws cloudformation describe-stacks \
  --stack-name "$STACK_NAME" \
  --query "Stacks[0].Outputs[?OutputKey=='ServiceName'].OutputValue" \
  --output text)

aws ecs update-service \
  --cluster "$CLUSTER_NAME" \
  --service "$SERVICE_NAME" \
  --force-new-deployment \
  --region "$AWS_REGION" > /dev/null

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║              DEPLOYMENT COMPLETE                   ║"
echo "╠════════════════════════════════════════════════════╣"
echo "║                                                    ║"
echo "║  Waiting for rolling deployment...                 ║"
echo "║                                                    ║"
echo "║  Monitor: aws ecs describe-services \\              ║"
echo "║    --cluster $CLUSTER_NAME \\                       "
echo "║    --services $SERVICE_NAME                        "
echo "║                                                    ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Optional: wait for deployment to stabilize
echo "Waiting for service to stabilize (this may take 2-5 minutes)..."
aws ecs wait services-stable \
  --cluster "$CLUSTER_NAME" \
  --services "$SERVICE_NAME" \
  --region "$AWS_REGION" 2>/dev/null || echo "  (wait timed out — check ECS console)"

echo ""
echo "Deployment complete. Site: https://cloudlobo.com"
