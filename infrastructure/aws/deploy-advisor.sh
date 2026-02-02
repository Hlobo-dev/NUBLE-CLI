#!/bin/bash
#
# KYPERIAN ADVISOR - Deployment Script
# =====================================
#
# Deploys the complete KYPERIAN ADVISOR infrastructure:
# - CloudFormation stack (DynamoDB, Lambda, API Gateway, EventBridge)
# - Lambda function code
# - Telegram bot webhook configuration
#
# Prerequisites:
# - AWS CLI configured with 'kyperian' profile
# - Environment variables: POLYGON_API_KEY, ANTHROPIC_API_KEY
# - Optional: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
#
# Usage:
#   ./deploy-advisor.sh [environment]
#
# Examples:
#   ./deploy-advisor.sh              # Deploy to production
#   ./deploy-advisor.sh staging      # Deploy to staging
#

set -e

# Configuration
ENVIRONMENT=${1:-production}
AWS_PROFILE=${AWS_PROFILE:-kyperian}
AWS_REGION=${AWS_REGION:-us-east-1}
STACK_NAME="kyperian-${ENVIRONMENT}-advisor"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           KYPERIAN ADVISOR DEPLOYMENT                          ║${NC}"
echo -e "${BLUE}║           The Autonomous AI Wealth Manager                     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "Environment: ${GREEN}${ENVIRONMENT}${NC}"
echo -e "AWS Profile: ${GREEN}${AWS_PROFILE}${NC}"
echo -e "Region:      ${GREEN}${AWS_REGION}${NC}"
echo ""

# Check prerequisites
echo -e "${YELLOW}📋 Checking prerequisites...${NC}"

if ! command -v aws &> /dev/null; then
    echo -e "${RED}❌ AWS CLI not found. Please install it first.${NC}"
    exit 1
fi

if ! aws sts get-caller-identity --profile $AWS_PROFILE &> /dev/null; then
    echo -e "${RED}❌ AWS credentials not configured for profile '${AWS_PROFILE}'${NC}"
    exit 1
fi

# Get API keys from environment or prompt
if [ -z "$POLYGON_API_KEY" ]; then
    echo -e "${YELLOW}Enter Polygon.io API Key:${NC}"
    read -s POLYGON_API_KEY
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo -e "${YELLOW}Enter Anthropic API Key:${NC}"
    read -s ANTHROPIC_API_KEY
fi

# Optional: Telegram
TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN:-""}
TELEGRAM_CHAT_ID=${TELEGRAM_CHAT_ID:-""}

echo -e "${GREEN}✅ Prerequisites OK${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CLOUDFORMATION_DIR="${SCRIPT_DIR}/cloudformation"
LAMBDA_DIR="${SCRIPT_DIR}/lambda"

# Deploy CloudFormation stack
echo -e "${YELLOW}🚀 Deploying CloudFormation stack...${NC}"

aws cloudformation deploy \
    --template-file "${CLOUDFORMATION_DIR}/advisor.yaml" \
    --stack-name "${STACK_NAME}" \
    --parameter-overrides \
        Environment="${ENVIRONMENT}" \
        PolygonApiKey="${POLYGON_API_KEY}" \
        AnthropicApiKey="${ANTHROPIC_API_KEY}" \
        TelegramBotToken="${TELEGRAM_BOT_TOKEN}" \
        TelegramChatId="${TELEGRAM_CHAT_ID}" \
    --capabilities CAPABILITY_NAMED_IAM \
    --region "${AWS_REGION}" \
    --profile "${AWS_PROFILE}" \
    --no-fail-on-empty-changeset

echo -e "${GREEN}✅ CloudFormation stack deployed${NC}"
echo ""

# Get outputs
echo -e "${YELLOW}📡 Getting stack outputs...${NC}"

ADVISOR_API_URL=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --query "Stacks[0].Outputs[?OutputKey=='AdvisorApiUrl'].OutputValue" \
    --output text \
    --region "${AWS_REGION}" \
    --profile "${AWS_PROFILE}")

TELEGRAM_WEBHOOK_URL=$(aws cloudformation describe-stacks \
    --stack-name "${STACK_NAME}" \
    --query "Stacks[0].Outputs[?OutputKey=='TelegramWebhookUrl'].OutputValue" \
    --output text \
    --region "${AWS_REGION}" \
    --profile "${AWS_PROFILE}")

echo -e "Advisor API URL:    ${GREEN}${ADVISOR_API_URL}${NC}"
echo -e "Telegram Webhook:   ${GREEN}${TELEGRAM_WEBHOOK_URL}${NC}"
echo ""

# Deploy Lambda code - Advisor
echo -e "${YELLOW}📦 Deploying Advisor Lambda code...${NC}"

# Create temp directory for packaging
TEMP_DIR=$(mktemp -d)
ADVISOR_ZIP="${TEMP_DIR}/advisor.zip"

# Package advisor handler
cp "${LAMBDA_DIR}/advisor/handler.py" "${TEMP_DIR}/"
cd "${TEMP_DIR}"
zip -r "${ADVISOR_ZIP}" handler.py

# Update Lambda function
aws lambda update-function-code \
    --function-name "kyperian-${ENVIRONMENT}-advisor" \
    --zip-file "fileb://${ADVISOR_ZIP}" \
    --region "${AWS_REGION}" \
    --profile "${AWS_PROFILE}" \
    --output json | jq '{FunctionName, CodeSize, LastModified}'

echo -e "${GREEN}✅ Advisor Lambda deployed${NC}"
echo ""

# Deploy Lambda code - Telegram Bot
echo -e "${YELLOW}📦 Deploying Telegram Bot Lambda code...${NC}"

TELEGRAM_ZIP="${TEMP_DIR}/telegram.zip"

# Package telegram handler
cp "${LAMBDA_DIR}/telegram_bot/handler.py" "${TEMP_DIR}/handler.py"
cd "${TEMP_DIR}"
zip -r "${TELEGRAM_ZIP}" handler.py

# Update Lambda function
aws lambda update-function-code \
    --function-name "kyperian-${ENVIRONMENT}-telegram-bot" \
    --zip-file "fileb://${TELEGRAM_ZIP}" \
    --region "${AWS_REGION}" \
    --profile "${AWS_PROFILE}" \
    --output json | jq '{FunctionName, CodeSize, LastModified}'

# Update Telegram bot environment with actual API URL
aws lambda update-function-configuration \
    --function-name "kyperian-${ENVIRONMENT}-telegram-bot" \
    --environment "Variables={POLYGON_API_KEY=${POLYGON_API_KEY},ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY},TELEGRAM_BOT_TOKEN=${TELEGRAM_BOT_TOKEN},ADVISOR_API_URL=${ADVISOR_API_URL}}" \
    --region "${AWS_REGION}" \
    --profile "${AWS_PROFILE}" \
    --output json | jq '{FunctionName, LastModified}'

echo -e "${GREEN}✅ Telegram Bot Lambda deployed${NC}"
echo ""

# Cleanup
rm -rf "${TEMP_DIR}"

# Configure Telegram webhook (if token provided)
if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
    echo -e "${YELLOW}🔗 Configuring Telegram webhook...${NC}"
    
    TELEGRAM_SET_WEBHOOK_URL="https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/setWebhook?url=${TELEGRAM_WEBHOOK_URL}"
    
    WEBHOOK_RESULT=$(curl -s "${TELEGRAM_SET_WEBHOOK_URL}")
    
    if echo "$WEBHOOK_RESULT" | grep -q '"ok":true'; then
        echo -e "${GREEN}✅ Telegram webhook configured${NC}"
    else
        echo -e "${YELLOW}⚠️ Telegram webhook response: ${WEBHOOK_RESULT}${NC}"
    fi
    echo ""
fi

# Test the API
echo -e "${YELLOW}🧪 Testing Advisor API...${NC}"

TEST_RESULT=$(curl -s "${ADVISOR_API_URL}/" | jq -r '.service // .error // "Unknown response"')
echo -e "API Response: ${GREEN}${TEST_RESULT}${NC}"
echo ""

# Test single symbol
echo -e "${YELLOW}🧪 Testing symbol analysis...${NC}"

AAPL_TEST=$(curl -s "${ADVISOR_API_URL}/analyze/AAPL" | jq -r '.analysis.direction // .error // "Error"')
echo -e "AAPL Direction: ${GREEN}${AAPL_TEST}${NC}"
echo ""

# Summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║           DEPLOYMENT COMPLETE                                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}KYPERIAN ADVISOR is now live!${NC}"
echo ""
echo -e "📊 ${YELLOW}Advisor API:${NC}"
echo -e "   GET  ${ADVISOR_API_URL}/"
echo -e "   GET  ${ADVISOR_API_URL}/dashboard"
echo -e "   GET  ${ADVISOR_API_URL}/analyze/{symbol}"
echo -e "   POST ${ADVISOR_API_URL}/query"
echo -e "   POST ${ADVISOR_API_URL}/monitor"
echo ""

if [ -n "$TELEGRAM_BOT_TOKEN" ]; then
    BOT_USERNAME=$(curl -s "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/getMe" | jq -r '.result.username // "unknown"')
    echo -e "🤖 ${YELLOW}Telegram Bot:${NC} @${BOT_USERNAME}"
    echo -e "   Webhook: ${TELEGRAM_WEBHOOK_URL}"
    echo ""
fi

echo -e "⏰ ${YELLOW}Scheduled Tasks:${NC}"
echo -e "   • Autonomous monitoring: Every 5 minutes"
echo -e "   • Morning briefing: 8am ET (weekdays)"
echo -e "   • Daily digest: 4pm ET (weekdays)"
echo ""
echo -e "${GREEN}Ready to manage your wealth 24/7! 🚀${NC}"
