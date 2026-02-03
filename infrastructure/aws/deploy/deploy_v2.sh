#!/bin/bash
#
# NUBLE Decision Engine V2 - Deployment Script
# ================================================
# Deploys the institutional-grade decision engine to AWS
#
# Features:
# - 15+ data points across 4 layers
# - Signal Layer (40%): LuxAlgo, Momentum, Trend, S/R
# - Context Layer (30%): Regime, Sentiment, Volatility
# - Validation Layer (20%): Historical win rate, patterns
# - Risk Layer (10% + VETO): Position limits, drawdown
#
# Usage:
#   ./deploy_v2.sh [--telegram-token TOKEN] [--telegram-chat-id ID] [--discord-webhook URL]
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Configuration
AWS_PROFILE="${AWS_PROFILE:-nuble}"
AWS_REGION="${AWS_REGION:-us-east-1}"
ENVIRONMENT="production"
STACK_NAME="nuble-${ENVIRONMENT}-decision-engine-v2"
LAMBDA_NAME="nuble-${ENVIRONMENT}-decision-engine-v2"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CFN_TEMPLATE="$SCRIPT_DIR/../cloudformation/decision-engine-v2.yaml"
LAMBDA_CODE="$SCRIPT_DIR/../lambda/decision_engine/handler_v2.py"

# Parameters (can be overridden via CLI)
TELEGRAM_BOT_TOKEN=""
TELEGRAM_CHAT_ID=""
DISCORD_WEBHOOK_URL=""
ANALYSIS_INTERVAL=5

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --telegram-token)
            TELEGRAM_BOT_TOKEN="$2"
            shift 2
            ;;
        --telegram-chat-id)
            TELEGRAM_CHAT_ID="$2"
            shift 2
            ;;
        --discord-webhook)
            DISCORD_WEBHOOK_URL="$2"
            shift 2
            ;;
        --interval)
            ANALYSIS_INTERVAL="$2"
            shift 2
            ;;
        --profile)
            AWS_PROFILE="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo ""
echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   NUBLE ELITE - Decision Engine V2 Deployment                ║${NC}"
echo -e "${CYAN}║   Institutional-Grade Multi-Layer Analysis                   ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Verify AWS credentials
echo -e "${BLUE}▶ Verifying AWS credentials...${NC}"
if ! aws sts get-caller-identity --profile "$AWS_PROFILE" > /dev/null 2>&1; then
    echo -e "${RED}✗ Failed to authenticate with AWS profile: $AWS_PROFILE${NC}"
    exit 1
fi

ACCOUNT_ID=$(aws sts get-caller-identity --profile "$AWS_PROFILE" --query 'Account' --output text)
echo -e "${GREEN}✓ Authenticated: Account $ACCOUNT_ID${NC}"

# Check if template exists
echo -e "${BLUE}▶ Checking template...${NC}"
if [[ ! -f "$CFN_TEMPLATE" ]]; then
    echo -e "${RED}✗ Template not found: $CFN_TEMPLATE${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Template found${NC}"

# Validate template
echo -e "${BLUE}▶ Validating CloudFormation template...${NC}"
aws cloudformation validate-template \
    --template-body "file://$CFN_TEMPLATE" \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" > /dev/null
echo -e "${GREEN}✓ Template valid${NC}"

# Deploy CloudFormation stack
echo -e "${BLUE}▶ Deploying CloudFormation stack: $STACK_NAME${NC}"

PARAMS="ParameterKey=Environment,ParameterValue=$ENVIRONMENT"
PARAMS="$PARAMS ParameterKey=AnalysisIntervalMinutes,ParameterValue=$ANALYSIS_INTERVAL"

if [[ -n "$TELEGRAM_BOT_TOKEN" ]]; then
    PARAMS="$PARAMS ParameterKey=TelegramBotToken,ParameterValue=$TELEGRAM_BOT_TOKEN"
fi

if [[ -n "$TELEGRAM_CHAT_ID" ]]; then
    PARAMS="$PARAMS ParameterKey=TelegramChatId,ParameterValue=$TELEGRAM_CHAT_ID"
fi

if [[ -n "$DISCORD_WEBHOOK_URL" ]]; then
    PARAMS="$PARAMS ParameterKey=DiscordWebhookUrl,ParameterValue=$DISCORD_WEBHOOK_URL"
fi

aws cloudformation deploy \
    --template-file "$CFN_TEMPLATE" \
    --stack-name "$STACK_NAME" \
    --parameter-overrides $PARAMS \
    --capabilities CAPABILITY_NAMED_IAM \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" \
    --tags Application=NUBLE Environment=$ENVIRONMENT Component=DecisionEngineV2

echo -e "${GREEN}✓ Stack deployed${NC}"

# Get Lambda function details
echo -e "${BLUE}▶ Getting Lambda function details...${NC}"
LAMBDA_ARN=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='DecisionEngineLambdaArn'].OutputValue" \
    --output text)

API_ENDPOINT=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='APIEndpoint'].OutputValue" \
    --output text)

DASHBOARD_URL=$(aws cloudformation describe-stacks \
    --stack-name "$STACK_NAME" \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" \
    --query "Stacks[0].Outputs[?OutputKey=='DashboardURL'].OutputValue" \
    --output text)

echo -e "${GREEN}✓ Lambda ARN: $LAMBDA_ARN${NC}"
echo -e "${GREEN}✓ API Endpoint: $API_ENDPOINT${NC}"

# Package and upload Lambda code
echo -e "${BLUE}▶ Uploading Lambda code...${NC}"
if [[ ! -f "$LAMBDA_CODE" ]]; then
    echo -e "${RED}✗ Lambda code not found: $LAMBDA_CODE${NC}"
    exit 1
fi

# Create temporary directory for packaging
TEMP_DIR=$(mktemp -d)
cp "$LAMBDA_CODE" "$TEMP_DIR/handler_v2.py"

# Create zip file
cd "$TEMP_DIR"
zip -q lambda.zip handler_v2.py
cd -

# Update Lambda function code
aws lambda update-function-code \
    --function-name "$LAMBDA_NAME" \
    --zip-file "fileb://$TEMP_DIR/lambda.zip" \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" > /dev/null

# Wait for update
echo -e "${BLUE}  Waiting for Lambda update...${NC}"
aws lambda wait function-updated \
    --function-name "$LAMBDA_NAME" \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION"

# Cleanup
rm -rf "$TEMP_DIR"

echo -e "${GREEN}✓ Lambda code uploaded${NC}"

# Test the API
echo -e "${BLUE}▶ Testing API endpoint...${NC}"
RESPONSE=$(curl -s -w "\n%{http_code}" "$API_ENDPOINT/")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)
BODY=$(echo "$RESPONSE" | sed '$d')

if [[ "$HTTP_CODE" -eq 200 ]]; then
    echo -e "${GREEN}✓ API responding (HTTP $HTTP_CODE)${NC}"
else
    echo -e "${YELLOW}⚠ API returned HTTP $HTTP_CODE${NC}"
fi

# Test the dashboard
echo -e "${BLUE}▶ Testing dashboard...${NC}"
RESPONSE=$(curl -s -w "\n%{http_code}" "$DASHBOARD_URL")
HTTP_CODE=$(echo "$RESPONSE" | tail -n1)

if [[ "$HTTP_CODE" -eq 200 ]]; then
    echo -e "${GREEN}✓ Dashboard responding (HTTP $HTTP_CODE)${NC}"
else
    echo -e "${YELLOW}⚠ Dashboard returned HTTP $HTTP_CODE${NC}"
fi

# Trigger initial analysis
echo -e "${BLUE}▶ Triggering initial analysis...${NC}"
aws lambda invoke \
    --function-name "$LAMBDA_NAME" \
    --profile "$AWS_PROFILE" \
    --region "$AWS_REGION" \
    --cli-binary-format raw-in-base64-out \
    --payload '{"source": "manual.trigger", "detail-type": "Initial Analysis"}' \
    /tmp/lambda-response.json > /dev/null

echo -e "${GREEN}✓ Initial analysis triggered${NC}"

# Print summary
echo ""
echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║   🚀 DEPLOYMENT COMPLETE                                     ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}Decision Engine V2 is now running!${NC}"
echo ""
echo -e "${YELLOW}API Endpoints:${NC}"
echo -e "  • Root:      $API_ENDPOINT/"
echo -e "  • Dashboard: $DASHBOARD_URL"
echo -e "  • Check:     $API_ENDPOINT/check/{symbol}"
echo -e "  • Analyze:   $API_ENDPOINT/analyze/{symbol}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo -e "  • Analysis Interval: Every $ANALYSIS_INTERVAL minutes"
echo -e "  • Telegram: $([ -n "$TELEGRAM_BOT_TOKEN" ] && echo "Enabled" || echo "Disabled")"
echo -e "  • Discord:  $([ -n "$DISCORD_WEBHOOK_URL" ] && echo "Enabled" || echo "Disabled")"
echo ""
echo -e "${YELLOW}Monitored Symbols:${NC}"
echo -e "  BTCUSD, ETHUSD, SPY, QQQ, AAPL, TSLA, NVDA, AMD"
echo ""
echo -e "${YELLOW}Layer Weights:${NC}"
echo -e "  • Signal Layer:     40% (LuxAlgo, Momentum, Trend)"
echo -e "  • Context Layer:    30% (Regime, Volatility)"
echo -e "  • Validation Layer: 20% (Historical Win Rate)"
echo -e "  • Risk Layer:       10% + VETO Power"
echo ""
echo -e "${CYAN}Test it now:${NC}"
echo -e "  curl $API_ENDPOINT/check/BTCUSD | jq"
echo ""
