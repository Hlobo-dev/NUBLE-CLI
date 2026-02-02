#!/bin/bash
# ============================================================
# KYPERIAN ELITE - Decision Engine Deployment Script
# ============================================================
# Deploys the multi-timeframe fusion and notification system
# 
# Usage:
#   ./deploy-decision-engine.sh [--telegram-token TOKEN] [--telegram-chat-id ID] [--discord-webhook URL]
#
# Environment Variables (alternative to flags):
#   TELEGRAM_BOT_TOKEN - Telegram bot token
#   TELEGRAM_CHAT_ID   - Telegram chat ID
#   DISCORD_WEBHOOK_URL - Discord webhook URL
# ============================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENVIRONMENT="${ENVIRONMENT:-production}"
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_PROFILE="${AWS_PROFILE:-kyperian}"
STACK_NAME="kyperian-${ENVIRONMENT}-decision-engine"
LAMBDA_NAME="kyperian-${ENVIRONMENT}-decision-engine"

# Parse arguments
TELEGRAM_TOKEN="${TELEGRAM_BOT_TOKEN:-}"
TELEGRAM_CHAT="${TELEGRAM_CHAT_ID:-}"
DISCORD_WEBHOOK="${DISCORD_WEBHOOK_URL:-}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --telegram-token)
            TELEGRAM_TOKEN="$2"
            shift 2
            ;;
        --telegram-chat-id)
            TELEGRAM_CHAT="$2"
            shift 2
            ;;
        --discord-webhook)
            DISCORD_WEBHOOK="$2"
            shift 2
            ;;
        --environment)
            ENVIRONMENT="$2"
            STACK_NAME="kyperian-${ENVIRONMENT}-decision-engine"
            LAMBDA_NAME="kyperian-${ENVIRONMENT}-decision-engine"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${CYAN}"
echo "╔════════════════════════════════════════════════════════════╗"
echo "║     KYPERIAN ELITE - Decision Engine Deployment            ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  Multi-Timeframe Fusion + Notification System              ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BLUE}Configuration:${NC}"
echo "  Environment:    ${ENVIRONMENT}"
echo "  Region:         ${AWS_REGION}"
echo "  Profile:        ${AWS_PROFILE}"
echo "  Stack Name:     ${STACK_NAME}"
echo "  Telegram:       ${TELEGRAM_TOKEN:+configured}${TELEGRAM_TOKEN:-not configured}"
echo "  Discord:        ${DISCORD_WEBHOOK:+configured}${DISCORD_WEBHOOK:-not configured}"
echo ""

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI not installed${NC}"
    exit 1
fi

if ! aws sts get-caller-identity --profile "$AWS_PROFILE" &> /dev/null; then
    echo -e "${RED}Error: AWS credentials not configured for profile $AWS_PROFILE${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Prerequisites OK${NC}"

# Step 1: Package Lambda function
echo ""
echo -e "${YELLOW}Step 1: Packaging Lambda function...${NC}"

LAMBDA_DIR="${SCRIPT_DIR}/lambda/decision_engine"
PACKAGE_DIR="${SCRIPT_DIR}/.package/decision_engine"
PACKAGE_ZIP="${SCRIPT_DIR}/.package/decision_engine.zip"

rm -rf "$PACKAGE_DIR" 2>/dev/null || true
mkdir -p "$PACKAGE_DIR"

# Copy handler
cp "${LAMBDA_DIR}/handler.py" "${PACKAGE_DIR}/"

# Create zip
cd "$PACKAGE_DIR"
zip -r "$PACKAGE_ZIP" . -x "*.pyc" -x "__pycache__/*"
cd "$SCRIPT_DIR"

PACKAGE_SIZE=$(du -h "$PACKAGE_ZIP" | cut -f1)
echo -e "${GREEN}✓ Package created: $PACKAGE_ZIP ($PACKAGE_SIZE)${NC}"

# Step 2: Update API Gateway stack to export API ID
echo ""
echo -e "${YELLOW}Step 2: Updating API Gateway stack (adding API ID export)...${NC}"

aws cloudformation deploy \
    --template-file "${SCRIPT_DIR}/cloudformation/api-gateway-enterprise.yaml" \
    --stack-name "kyperian-${ENVIRONMENT}-api" \
    --parameter-overrides \
        Environment="${ENVIRONMENT}" \
        LambdaStackName="kyperian-${ENVIRONMENT}-lambda" \
    --capabilities CAPABILITY_NAMED_IAM \
    --region "$AWS_REGION" \
    --profile "$AWS_PROFILE" \
    --no-fail-on-empty-changeset \
    2>&1 || echo -e "${YELLOW}Note: API Gateway update may have no changes${NC}"

echo -e "${GREEN}✓ API Gateway stack updated${NC}"

# Step 3: Deploy Decision Engine stack
echo ""
echo -e "${YELLOW}Step 3: Deploying Decision Engine CloudFormation stack...${NC}"

PARAMS="Environment=${ENVIRONMENT}"
[ -n "$TELEGRAM_TOKEN" ] && PARAMS="${PARAMS} TelegramBotToken=${TELEGRAM_TOKEN}"
[ -n "$TELEGRAM_CHAT" ] && PARAMS="${PARAMS} TelegramChatId=${TELEGRAM_CHAT}"
[ -n "$DISCORD_WEBHOOK" ] && PARAMS="${PARAMS} DiscordWebhookUrl=${DISCORD_WEBHOOK}"

aws cloudformation deploy \
    --template-file "${SCRIPT_DIR}/cloudformation/decision-engine.yaml" \
    --stack-name "$STACK_NAME" \
    --parameter-overrides $PARAMS \
    --capabilities CAPABILITY_NAMED_IAM \
    --region "$AWS_REGION" \
    --profile "$AWS_PROFILE" \
    --no-fail-on-empty-changeset

echo -e "${GREEN}✓ CloudFormation stack deployed${NC}"

# Step 4: Upload Lambda code
echo ""
echo -e "${YELLOW}Step 4: Uploading Lambda code...${NC}"

aws lambda update-function-code \
    --function-name "$LAMBDA_NAME" \
    --zip-file "fileb://${PACKAGE_ZIP}" \
    --region "$AWS_REGION" \
    --profile "$AWS_PROFILE" \
    --output text \
    --query 'LastModified'

echo -e "${GREEN}✓ Lambda code uploaded${NC}"

# Wait for update to complete
echo -e "${YELLOW}Waiting for Lambda update to complete...${NC}"
aws lambda wait function-updated \
    --function-name "$LAMBDA_NAME" \
    --region "$AWS_REGION" \
    --profile "$AWS_PROFILE"

echo -e "${GREEN}✓ Lambda update complete${NC}"

# Step 5: Test the function
echo ""
echo -e "${YELLOW}Step 5: Testing Decision Engine...${NC}"

TEST_RESULT=$(aws lambda invoke \
    --function-name "$LAMBDA_NAME" \
    --payload '{}' \
    --region "$AWS_REGION" \
    --profile "$AWS_PROFILE" \
    /tmp/decision-engine-test.json \
    --output text \
    --query 'StatusCode')

if [ "$TEST_RESULT" = "200" ]; then
    echo -e "${GREEN}✓ Lambda test successful${NC}"
    echo ""
    echo -e "${CYAN}Test Result:${NC}"
    cat /tmp/decision-engine-test.json | python3 -m json.tool 2>/dev/null || cat /tmp/decision-engine-test.json
else
    echo -e "${RED}✗ Lambda test failed with status: $TEST_RESULT${NC}"
fi

# Step 6: Get API endpoints
echo ""
echo -e "${YELLOW}Step 6: Getting API endpoints...${NC}"

API_URL=$(aws cloudformation describe-stacks \
    --stack-name "kyperian-${ENVIRONMENT}-api" \
    --query "Stacks[0].Outputs[?OutputKey=='ApiEndpoint'].OutputValue" \
    --output text \
    --region "$AWS_REGION" \
    --profile "$AWS_PROFILE" 2>/dev/null || echo "")

if [ -z "$API_URL" ]; then
    API_ID=$(aws cloudformation describe-stacks \
        --stack-name "kyperian-${ENVIRONMENT}-api" \
        --query "Stacks[0].Outputs[?OutputKey=='ApiId'].OutputValue" \
        --output text \
        --region "$AWS_REGION" \
        --profile "$AWS_PROFILE" 2>/dev/null || echo "")
    
    if [ -n "$API_ID" ]; then
        API_URL="https://${API_ID}.execute-api.${AWS_REGION}.amazonaws.com/${ENVIRONMENT}"
    fi
fi

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║               DEPLOYMENT COMPLETE!                         ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Decision Engine Endpoints:${NC}"
echo "  Dashboard:     ${API_URL}/dashboard"
echo "  Status:        ${API_URL}/status"
echo "  Check Symbol:  ${API_URL}/check/{symbol}"
echo ""
echo -e "${CYAN}Test Commands:${NC}"
echo "  # Get all signal alignments"
echo "  curl ${API_URL}/dashboard"
echo ""
echo "  # Check specific symbol"
echo "  curl ${API_URL}/check/BTCUSD"
echo ""
echo -e "${CYAN}Scheduled Checks:${NC}"
echo "  The Decision Engine runs every 5 minutes automatically."
echo "  It also triggers immediately when new signals arrive."
echo ""

if [ -n "$TELEGRAM_TOKEN" ]; then
    echo -e "${GREEN}✓ Telegram notifications enabled${NC}"
else
    echo -e "${YELLOW}⚠ Telegram not configured. Add with:${NC}"
    echo "  ./deploy-decision-engine.sh --telegram-token YOUR_TOKEN --telegram-chat-id YOUR_CHAT_ID"
fi

if [ -n "$DISCORD_WEBHOOK" ]; then
    echo -e "${GREEN}✓ Discord notifications enabled${NC}"
else
    echo -e "${YELLOW}⚠ Discord not configured. Add with:${NC}"
    echo "  ./deploy-decision-engine.sh --discord-webhook YOUR_WEBHOOK_URL"
fi

echo ""
echo -e "${CYAN}CloudWatch Dashboard:${NC}"
echo "  https://${AWS_REGION}.console.aws.amazon.com/cloudwatch/home?region=${AWS_REGION}#dashboards:name=KYPERIAN-${ENVIRONMENT}-DecisionEngine"
echo ""
