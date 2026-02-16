#!/bin/bash
# Deploy Nova Sonic AppSync Events Infrastructure
# This script deploys the AppSync Events API for real-time audio streaming

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
STACK_NAME="${STACK_NAME:-nova-sonic-appsync}"
ENVIRONMENT="${ENVIRONMENT:-production}"
PROJECT_NAME="${PROJECT_NAME:-nova-sonic}"
REGION="${AWS_REGION:-us-east-1}"
TEMPLATE_FILE="infrastructure/appsync-events.yaml"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       Nova Sonic - AppSync Events Deployment               ║${NC}"
echo -e "${BLUE}╠════════════════════════════════════════════════════════════╣${NC}"
echo -e "${BLUE}║  Stack:       ${YELLOW}${STACK_NAME}${NC}"
echo -e "${BLUE}║  Environment: ${YELLOW}${ENVIRONMENT}${NC}"
echo -e "${BLUE}║  Region:      ${YELLOW}${REGION}${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check AWS credentials
echo -e "${YELLOW}→ Checking AWS credentials...${NC}"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo -e "${RED}✗ AWS credentials not configured. Please run 'aws configure'.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ AWS Account: ${AWS_ACCOUNT_ID}${NC}"

# Validate template
echo -e "${YELLOW}→ Validating CloudFormation template...${NC}"
aws cloudformation validate-template \
    --template-body file://${TEMPLATE_FILE} \
    --region ${REGION} > /dev/null
echo -e "${GREEN}✓ Template is valid${NC}"

# Check if stack exists
STACK_EXISTS=$(aws cloudformation describe-stacks --stack-name ${STACK_NAME} --region ${REGION} 2>/dev/null || echo "")

if [ -z "$STACK_EXISTS" ]; then
    echo -e "${YELLOW}→ Creating new stack...${NC}"
    ACTION="create-stack"
    WAIT_ACTION="stack-create-complete"
else
    echo -e "${YELLOW}→ Updating existing stack...${NC}"
    ACTION="update-stack"
    WAIT_ACTION="stack-update-complete"
fi

# Deploy stack
echo -e "${YELLOW}→ Deploying CloudFormation stack...${NC}"

set +e
DEPLOY_OUTPUT=$(aws cloudformation ${ACTION} \
    --stack-name ${STACK_NAME} \
    --template-body file://${TEMPLATE_FILE} \
    --parameters \
        ParameterKey=Environment,ParameterValue=${ENVIRONMENT} \
        ParameterKey=ProjectName,ParameterValue=${PROJECT_NAME} \
    --capabilities CAPABILITY_NAMED_IAM \
    --region ${REGION} 2>&1)
DEPLOY_EXIT_CODE=$?
set -e

if [ $DEPLOY_EXIT_CODE -ne 0 ]; then
    if echo "$DEPLOY_OUTPUT" | grep -q "No updates are to be performed"; then
        echo -e "${GREEN}✓ Stack is already up to date${NC}"
    else
        echo -e "${RED}✗ Deployment failed: ${DEPLOY_OUTPUT}${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}→ Waiting for stack ${ACTION} to complete...${NC}"
    aws cloudformation wait ${WAIT_ACTION} \
        --stack-name ${STACK_NAME} \
        --region ${REGION}
    echo -e "${GREEN}✓ Stack deployment complete${NC}"
fi

# Get outputs
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Stack Outputs                            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"

OUTPUTS=$(aws cloudformation describe-stacks \
    --stack-name ${STACK_NAME} \
    --region ${REGION} \
    --query 'Stacks[0].Outputs' \
    --output json)

# Parse and display outputs
APPSYNC_API_ID=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="AppSyncApiId") | .OutputValue')
APPSYNC_HTTP=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="AppSyncHttpEndpoint") | .OutputValue')
APPSYNC_REALTIME=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="AppSyncRealtimeEndpoint") | .OutputValue')
APPSYNC_API_KEY=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="AppSyncApiKey") | .OutputValue')
CHAT_TABLE=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="ChatHistoryTableName") | .OutputValue')
SESSIONS_TABLE=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="SessionsTableName") | .OutputValue')
TASK_ROLE_ARN=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="TaskRoleArn") | .OutputValue')
IDENTITY_POOL_ID=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="CognitoIdentityPoolId") | .OutputValue')

echo -e "${GREEN}AppSync API ID:        ${YELLOW}${APPSYNC_API_ID}${NC}"
echo -e "${GREEN}AppSync HTTP:          ${YELLOW}${APPSYNC_HTTP}${NC}"
echo -e "${GREEN}AppSync Realtime:      ${YELLOW}${APPSYNC_REALTIME}${NC}"
echo -e "${GREEN}AppSync API Key:       ${YELLOW}${APPSYNC_API_KEY:0:20}...${NC}"
echo -e "${GREEN}Chat History Table:    ${YELLOW}${CHAT_TABLE}${NC}"
echo -e "${GREEN}Sessions Table:        ${YELLOW}${SESSIONS_TABLE}${NC}"
echo -e "${GREEN}Task Role ARN:         ${YELLOW}${TASK_ROLE_ARN}${NC}"
echo -e "${GREEN}Cognito Identity Pool: ${YELLOW}${IDENTITY_POOL_ID}${NC}"

# Create .env file for the project
echo ""
echo -e "${YELLOW}→ Creating .env.production file...${NC}"

cat > .env.production << EOF
# Nova Sonic Production Configuration
# Generated by deploy-appsync.sh on $(date)

# AWS Configuration
AWS_REGION=${REGION}
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID}

# AppSync Events Configuration
APPSYNC_API_ID=${APPSYNC_API_ID}
APPSYNC_HTTP_ENDPOINT=${APPSYNC_HTTP}
APPSYNC_REALTIME_ENDPOINT=${APPSYNC_REALTIME}
APPSYNC_API_KEY=${APPSYNC_API_KEY}

# DynamoDB Tables
DYNAMODB_CHAT_HISTORY_TABLE=${CHAT_TABLE}
DYNAMODB_SESSIONS_TABLE=${SESSIONS_TABLE}

# IAM
TASK_ROLE_ARN=${TASK_ROLE_ARN}

# Cognito
COGNITO_IDENTITY_POOL_ID=${IDENTITY_POOL_ID}

# Bedrock
BEDROCK_REGION=${REGION}
BEDROCK_MODEL_ID=amazon.nova-sonic-v1:0
EOF

echo -e "${GREEN}✓ .env.production file created${NC}"

# Create frontend config
echo -e "${YELLOW}→ Creating frontend configuration...${NC}"

cat > nuble-frontend/src/lib/config/appsync.ts << EOF
// Nova Sonic AppSync Configuration
// Generated by deploy-appsync.sh on $(date)

export const APPSYNC_CONFIG = {
  apiId: '${APPSYNC_API_ID}',
  httpEndpoint: '${APPSYNC_HTTP}',
  realtimeEndpoint: '${APPSYNC_REALTIME}',
  apiKey: '${APPSYNC_API_KEY}',
  region: '${REGION}',
};

export const COGNITO_CONFIG = {
  identityPoolId: '${IDENTITY_POOL_ID}',
  region: '${REGION}',
};

export default APPSYNC_CONFIG;
EOF

echo -e "${GREEN}✓ Frontend configuration created${NC}"

echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Deployment Complete! 🎉                        ║${NC}"
echo -e "${GREEN}╠════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║                                                             ║${NC}"
echo -e "${GREEN}║  Next steps:                                                ║${NC}"
echo -e "${GREEN}║  1. Update your frontend to use the AppSync config          ║${NC}"
echo -e "${GREEN}║  2. Deploy the ECS service with the new environment vars    ║${NC}"
echo -e "${GREEN}║  3. Test the real-time audio streaming                      ║${NC}"
echo -e "${GREEN}║                                                             ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
