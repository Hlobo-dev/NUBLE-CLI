#!/bin/bash
#
# Nova Sonic Production Deployment for cloudlobo.com
# This script deploys the complete infrastructure to AWS
#
set -e

# ============================================
# CONFIGURATION
# ============================================
STACK_NAME="cloudlobo-nova-sonic-production"
TEMPLATE_FILE="infrastructure/cloudlobo-production.yaml"
REGION="us-east-1"
DOMAIN_NAME="cloudlobo.com"
HOSTED_ZONE_ID="Z05234473O5MBYH58XZUM"
PROJECT_NAME="nova-sonic"
ENVIRONMENT="production"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║                                                                  ║${NC}"
echo -e "${CYAN}║    🚀 Nova Sonic Production Deployment                          ║${NC}"
echo -e "${CYAN}║       cloudlobo.com                                              ║${NC}"
echo -e "${CYAN}║                                                                  ║${NC}"
echo -e "${CYAN}╠══════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${CYAN}║  Stack:        ${YELLOW}${STACK_NAME}${NC}"
echo -e "${CYAN}║  Domain:       ${YELLOW}${DOMAIN_NAME}${NC}"
echo -e "${CYAN}║  Region:       ${YELLOW}${REGION}${NC}"
echo -e "${CYAN}║  Hosted Zone:  ${YELLOW}${HOSTED_ZONE_ID}${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# ============================================
# PRE-FLIGHT CHECKS
# ============================================

echo -e "${YELLOW}━━━ Pre-flight Checks ━━━${NC}"

# Check AWS credentials
echo -ne "  Checking AWS credentials... "
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null)
if [ -z "$AWS_ACCOUNT_ID" ]; then
    echo -e "${RED}FAILED${NC}"
    echo -e "${RED}  ✗ AWS credentials not configured. Run 'aws configure'.${NC}"
    exit 1
fi
echo -e "${GREEN}OK${NC} (Account: ${AWS_ACCOUNT_ID})"

# Check if jq is installed
echo -ne "  Checking jq... "
if ! command -v jq &> /dev/null; then
    echo -e "${YELLOW}Installing...${NC}"
    brew install jq 2>/dev/null || sudo apt-get install -y jq 2>/dev/null || echo "Please install jq manually"
else
    echo -e "${GREEN}OK${NC}"
fi

# Check if Docker is running (for building images later)
echo -ne "  Checking Docker... "
if docker info &> /dev/null; then
    echo -e "${GREEN}OK${NC}"
    DOCKER_AVAILABLE=true
else
    echo -e "${YELLOW}Not running (will skip image build)${NC}"
    DOCKER_AVAILABLE=false
fi

# Validate template
echo -ne "  Validating CloudFormation template... "
if aws cloudformation validate-template --template-body file://${TEMPLATE_FILE} --region ${REGION} &> /dev/null; then
    echo -e "${GREEN}OK${NC}"
else
    echo -e "${RED}FAILED${NC}"
    aws cloudformation validate-template --template-body file://${TEMPLATE_FILE} --region ${REGION}
    exit 1
fi

echo ""

# ============================================
# DEPLOY CLOUDFORMATION STACK
# ============================================

echo -e "${YELLOW}━━━ Deploying Infrastructure ━━━${NC}"

# Check if stack exists
STACK_STATUS=$(aws cloudformation describe-stacks --stack-name ${STACK_NAME} --region ${REGION} 2>/dev/null | jq -r '.Stacks[0].StackStatus' || echo "DOES_NOT_EXIST")

if [ "$STACK_STATUS" == "DOES_NOT_EXIST" ] || [ "$STACK_STATUS" == "null" ]; then
    echo -e "  ${BLUE}Creating new stack...${NC}"
    
    aws cloudformation create-stack \
        --stack-name ${STACK_NAME} \
        --template-body file://${TEMPLATE_FILE} \
        --parameters \
            ParameterKey=DomainName,ParameterValue=${DOMAIN_NAME} \
            ParameterKey=HostedZoneId,ParameterValue=${HOSTED_ZONE_ID} \
            ParameterKey=Environment,ParameterValue=${ENVIRONMENT} \
            ParameterKey=ProjectName,ParameterValue=${PROJECT_NAME} \
        --capabilities CAPABILITY_NAMED_IAM \
        --region ${REGION} \
        --tags \
            Key=Project,Value=${PROJECT_NAME} \
            Key=Environment,Value=${ENVIRONMENT} \
            Key=Domain,Value=${DOMAIN_NAME}
    
    echo -e "  ${BLUE}Waiting for stack creation (this may take 10-15 minutes for SSL certificate)...${NC}"
    echo -e "  ${YELLOW}Note: SSL certificate validation via DNS is automatic.${NC}"
    
    # Monitor stack creation
    while true; do
        STATUS=$(aws cloudformation describe-stacks --stack-name ${STACK_NAME} --region ${REGION} 2>/dev/null | jq -r '.Stacks[0].StackStatus')
        
        case $STATUS in
            CREATE_COMPLETE)
                echo -e "\n  ${GREEN}✓ Stack created successfully!${NC}"
                break
                ;;
            CREATE_IN_PROGRESS)
                echo -ne "."
                sleep 10
                ;;
            CREATE_FAILED|ROLLBACK_*)
                echo -e "\n  ${RED}✗ Stack creation failed!${NC}"
                echo -e "  ${RED}Checking events for error...${NC}"
                aws cloudformation describe-stack-events \
                    --stack-name ${STACK_NAME} \
                    --region ${REGION} \
                    --query 'StackEvents[?ResourceStatus==`CREATE_FAILED`].[LogicalResourceId,ResourceStatusReason]' \
                    --output table
                exit 1
                ;;
            *)
                echo -e "\n  ${YELLOW}Unknown status: ${STATUS}${NC}"
                sleep 10
                ;;
        esac
    done

elif [ "$STACK_STATUS" == "CREATE_COMPLETE" ] || [ "$STACK_STATUS" == "UPDATE_COMPLETE" ]; then
    echo -e "  ${BLUE}Updating existing stack...${NC}"
    
    set +e
    UPDATE_OUTPUT=$(aws cloudformation update-stack \
        --stack-name ${STACK_NAME} \
        --template-body file://${TEMPLATE_FILE} \
        --parameters \
            ParameterKey=DomainName,ParameterValue=${DOMAIN_NAME} \
            ParameterKey=HostedZoneId,ParameterValue=${HOSTED_ZONE_ID} \
            ParameterKey=Environment,ParameterValue=${ENVIRONMENT} \
            ParameterKey=ProjectName,ParameterValue=${PROJECT_NAME} \
        --capabilities CAPABILITY_NAMED_IAM \
        --region ${REGION} 2>&1)
    UPDATE_EXIT=$?
    set -e
    
    if [ $UPDATE_EXIT -ne 0 ]; then
        if echo "$UPDATE_OUTPUT" | grep -q "No updates are to be performed"; then
            echo -e "  ${GREEN}✓ Stack is already up to date${NC}"
        else
            echo -e "  ${RED}✗ Update failed: ${UPDATE_OUTPUT}${NC}"
            exit 1
        fi
    else
        echo -e "  ${BLUE}Waiting for stack update...${NC}"
        aws cloudformation wait stack-update-complete --stack-name ${STACK_NAME} --region ${REGION}
        echo -e "  ${GREEN}✓ Stack updated successfully!${NC}"
    fi
else
    echo -e "  ${YELLOW}Stack is in state: ${STACK_STATUS}. Please wait or delete it first.${NC}"
    exit 1
fi

echo ""

# ============================================
# GET STACK OUTPUTS
# ============================================

echo -e "${YELLOW}━━━ Stack Outputs ━━━${NC}"

OUTPUTS=$(aws cloudformation describe-stacks --stack-name ${STACK_NAME} --region ${REGION} --query 'Stacks[0].Outputs' --output json)

# Parse outputs
WEBSITE_URL=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="WebsiteURL") | .OutputValue')
APPSYNC_API_ID=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="AppSyncApiId") | .OutputValue')
APPSYNC_HTTP=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="AppSyncHttpEndpoint") | .OutputValue')
APPSYNC_REALTIME=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="AppSyncRealtimeEndpoint") | .OutputValue')
APPSYNC_API_KEY=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="AppSyncApiKey") | .OutputValue')
ECR_URI=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="ECRRepositoryUri") | .OutputValue')
ECS_CLUSTER=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="ECSClusterName") | .OutputValue')
ECS_SERVICE=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="ECSServiceName") | .OutputValue')
ALB_DNS=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="LoadBalancerDNS") | .OutputValue')
CHAT_TABLE=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="ChatHistoryTableName") | .OutputValue')
SESSIONS_TABLE=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="SessionsTableName") | .OutputValue')
IDENTITY_POOL=$(echo $OUTPUTS | jq -r '.[] | select(.OutputKey=="CognitoIdentityPoolId") | .OutputValue')

echo -e "  ${GREEN}Website URL:${NC}        ${YELLOW}${WEBSITE_URL}${NC}"
echo -e "  ${GREEN}ALB DNS:${NC}            ${YELLOW}${ALB_DNS}${NC}"
echo -e "  ${GREEN}ECR Repository:${NC}     ${YELLOW}${ECR_URI}${NC}"
echo -e "  ${GREEN}ECS Cluster:${NC}        ${YELLOW}${ECS_CLUSTER}${NC}"
echo -e "  ${GREEN}ECS Service:${NC}        ${YELLOW}${ECS_SERVICE}${NC}"
echo -e "  ${GREEN}AppSync API ID:${NC}     ${YELLOW}${APPSYNC_API_ID}${NC}"
echo -e "  ${GREEN}AppSync HTTP:${NC}       ${YELLOW}${APPSYNC_HTTP}${NC}"
echo -e "  ${GREEN}AppSync Realtime:${NC}   ${YELLOW}${APPSYNC_REALTIME}${NC}"
echo -e "  ${GREEN}Chat Table:${NC}         ${YELLOW}${CHAT_TABLE}${NC}"
echo -e "  ${GREEN}Sessions Table:${NC}     ${YELLOW}${SESSIONS_TABLE}${NC}"
echo -e "  ${GREEN}Cognito Pool:${NC}       ${YELLOW}${IDENTITY_POOL}${NC}"

echo ""

# ============================================
# CREATE ENVIRONMENT FILES
# ============================================

echo -e "${YELLOW}━━━ Creating Configuration Files ━━━${NC}"

# Create .env.production
cat > .env.production << EOF
# Nova Sonic Production Configuration
# Auto-generated on $(date)
# Domain: ${DOMAIN_NAME}

# AWS
AWS_REGION=${REGION}
AWS_ACCOUNT_ID=${AWS_ACCOUNT_ID}

# AppSync Events
APPSYNC_API_ID=${APPSYNC_API_ID}
APPSYNC_HTTP_ENDPOINT=${APPSYNC_HTTP}
APPSYNC_REALTIME_ENDPOINT=${APPSYNC_REALTIME}
APPSYNC_API_KEY=${APPSYNC_API_KEY}

# DynamoDB
DYNAMODB_CHAT_TABLE=${CHAT_TABLE}
DYNAMODB_SESSIONS_TABLE=${SESSIONS_TABLE}

# Cognito
COGNITO_IDENTITY_POOL_ID=${IDENTITY_POOL}

# ECS
ECR_REPOSITORY_URI=${ECR_URI}
ECS_CLUSTER=${ECS_CLUSTER}
ECS_SERVICE=${ECS_SERVICE}

# Bedrock
BEDROCK_REGION=${REGION}
BEDROCK_MODEL_ID=amazon.nova-sonic-v1:0

# URLs
WEBSITE_URL=${WEBSITE_URL}
ALB_DNS=${ALB_DNS}
EOF
echo -e "  ${GREEN}✓${NC} Created .env.production"

# Create frontend config
mkdir -p nuble-frontend/src/lib/config
cat > nuble-frontend/src/lib/config/appsync.ts << EOF
// Nova Sonic AppSync Configuration
// Auto-generated on $(date)
// Domain: ${DOMAIN_NAME}

export const APPSYNC_CONFIG = {
  apiId: '${APPSYNC_API_ID}',
  httpEndpoint: '${APPSYNC_HTTP}',
  realtimeEndpoint: '${APPSYNC_REALTIME}',
  apiKey: '${APPSYNC_API_KEY}',
  region: '${REGION}',
};

export const COGNITO_CONFIG = {
  identityPoolId: '${IDENTITY_POOL}',
  region: '${REGION}',
};

export const BEDROCK_CONFIG = {
  region: '${REGION}',
  modelId: 'amazon.nova-sonic-v1:0',
};

export const ENDPOINTS = {
  websiteUrl: '${WEBSITE_URL}',
  apiUrl: 'https://api.${DOMAIN_NAME}',
  sonicUrl: 'https://sonic.${DOMAIN_NAME}',
};

export default APPSYNC_CONFIG;
EOF
echo -e "  ${GREEN}✓${NC} Created nuble-frontend/src/lib/config/appsync.ts"

# Create .env for frontend
cat > nuble-frontend/.env.production << EOF
# Frontend Production Environment
VITE_APPSYNC_API_ID=${APPSYNC_API_ID}
VITE_APPSYNC_HTTP_ENDPOINT=${APPSYNC_HTTP}
VITE_APPSYNC_REALTIME_ENDPOINT=${APPSYNC_REALTIME}
VITE_APPSYNC_API_KEY=${APPSYNC_API_KEY}
VITE_AWS_REGION=${REGION}
VITE_COGNITO_IDENTITY_POOL_ID=${IDENTITY_POOL}
VITE_BEDROCK_REGION=${REGION}
VITE_BEDROCK_MODEL_ID=amazon.nova-sonic-v1:0
VITE_API_URL=https://api.${DOMAIN_NAME}
VITE_SONIC_URL=https://sonic.${DOMAIN_NAME}
EOF
echo -e "  ${GREEN}✓${NC} Created nuble-frontend/.env.production"

echo ""

# ============================================
# BUILD AND PUSH DOCKER IMAGE
# ============================================

if [ "$DOCKER_AVAILABLE" = true ]; then
    echo -e "${YELLOW}━━━ Building and Pushing Docker Image ━━━${NC}"
    
    # Login to ECR
    echo -e "  ${BLUE}Logging into ECR...${NC}"
    aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com
    
    # Build image
    echo -e "  ${BLUE}Building Docker image...${NC}"
    docker build -t ${PROJECT_NAME}:latest .
    
    # Tag and push
    echo -e "  ${BLUE}Pushing to ECR...${NC}"
    docker tag ${PROJECT_NAME}:latest ${ECR_URI}:latest
    docker push ${ECR_URI}:latest
    
    echo -e "  ${GREEN}✓${NC} Image pushed successfully"
    
    # Update ECS service to use new image
    echo -e "  ${BLUE}Updating ECS service...${NC}"
    aws ecs update-service \
        --cluster ${ECS_CLUSTER} \
        --service ${ECS_SERVICE} \
        --force-new-deployment \
        --region ${REGION} > /dev/null
    
    echo -e "  ${GREEN}✓${NC} ECS service updated (deployment in progress)"
else
    echo -e "${YELLOW}━━━ Skipping Docker Build (Docker not available) ━━━${NC}"
    echo -e "  To build and deploy the container later, run:"
    echo -e "  ${CYAN}./deploy-container.sh${NC}"
fi

echo ""

# ============================================
# FINAL SUMMARY
# ============================================

echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                                                                  ║${NC}"
echo -e "${GREEN}║    ✅ Deployment Complete!                                       ║${NC}"
echo -e "${GREEN}║                                                                  ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║                                                                  ║${NC}"
echo -e "${GREEN}║  Your site will be available at:                                 ║${NC}"
echo -e "${GREEN}║                                                                  ║${NC}"
echo -e "${GREEN}║    🌐 ${CYAN}https://${DOMAIN_NAME}${NC}"
echo -e "${GREEN}║    🌐 ${CYAN}https://www.${DOMAIN_NAME}${NC}"
echo -e "${GREEN}║    🌐 ${CYAN}https://api.${DOMAIN_NAME}${NC}"
echo -e "${GREEN}║    🌐 ${CYAN}https://sonic.${DOMAIN_NAME}${NC}"
echo -e "${GREEN}║                                                                  ║${NC}"
echo -e "${GREEN}║  Note: SSL certificate may take up to 30 minutes to validate.   ║${NC}"
echo -e "${GREEN}║                                                                  ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo -e "  1. Wait for SSL certificate to validate (check ACM in AWS Console)"
echo -e "  2. Build and deploy the frontend: ${CYAN}cd nuble-frontend && npm run build${NC}"
echo -e "  3. Monitor ECS service: ${CYAN}aws ecs describe-services --cluster ${ECS_CLUSTER} --services ${ECS_SERVICE}${NC}"
echo ""
