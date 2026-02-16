#!/bin/bash
#
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║                                                                              ║
# ║     NOVA SONIC PRODUCTION DEPLOYMENT                                        ║
# ║     Domain: beyondgrid.xyz                                                   ║
# ║                                                                              ║
# ║     Architecture:                                                            ║
# ║     • AppSync Events (WebSocket) for real-time communication               ║
# ║     • Lambda for Bedrock Nova Sonic streaming                               ║
# ║     • CloudFront + S3 for static hosting                                    ║
# ║     • DynamoDB for chat history persistence                                 ║
# ║     • WAF for security (production)                                         ║
# ║                                                                              ║
# ╚══════════════════════════════════════════════════════════════════════════════╝
#

set -e

# ─── Configuration ────────────────────────────────────────────────────────────

DOMAIN="beyondgrid.xyz"
AWS_REGION="${AWS_REGION:-us-east-1}"
STACK_NAME="nova-sonic-beyondgrid"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ─── Helper Functions ─────────────────────────────────────────────────────────

log_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

log_error() {
    echo -e "${RED}✗${NC} $1"
}

print_banner() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${NC}     Nova Sonic Production Deployment - ${DOMAIN}           ${BLUE}║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# ─── Pre-flight Checks ────────────────────────────────────────────────────────

preflight_checks() {
    log_info "Running pre-flight checks..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI not found. Please install it first."
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured. Run 'aws configure' first."
        exit 1
    fi
    
    # Get AWS Account ID
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    log_success "AWS Account: ${AWS_ACCOUNT_ID}"
    log_success "AWS Region: ${AWS_REGION}"
    
    # Check if domain has a hosted zone
    log_info "Checking Route 53 hosted zone for ${DOMAIN}..."
    HOSTED_ZONE_ID=$(aws route53 list-hosted-zones-by-name \
        --dns-name "${DOMAIN}." \
        --query "HostedZones[?Name=='${DOMAIN}.'].Id" \
        --output text | cut -d'/' -f3)
    
    if [ -z "$HOSTED_ZONE_ID" ] || [ "$HOSTED_ZONE_ID" == "None" ]; then
        log_warning "No hosted zone found for ${DOMAIN}"
        log_info "Creating hosted zone..."
        
        HOSTED_ZONE_ID=$(aws route53 create-hosted-zone \
            --name "${DOMAIN}" \
            --caller-reference "$(date +%s)" \
            --query 'HostedZone.Id' \
            --output text | cut -d'/' -f3)
        
        log_success "Created hosted zone: ${HOSTED_ZONE_ID}"
        
        # Get nameservers
        echo ""
        log_warning "IMPORTANT: Update your domain registrar with these nameservers:"
        aws route53 get-hosted-zone --id "${HOSTED_ZONE_ID}" \
            --query 'DelegationSet.NameServers' --output table
        echo ""
        read -p "Press Enter after updating nameservers (or Ctrl+C to exit)..."
    else
        log_success "Found hosted zone: ${HOSTED_ZONE_ID}"
    fi
}

# ─── Build Lambda Functions ───────────────────────────────────────────────────

build_lambdas() {
    log_info "Building Lambda functions..."
    
    LAMBDA_DIR="infrastructure/lambda"
    
    # Build nova-sonic-stream
    log_info "Building nova-sonic-stream Lambda..."
    cd "${LAMBDA_DIR}/nova-sonic-stream"
    npm install --omit=dev
    zip -r "../nova-sonic-stream.zip" . -x "*.git*" -x "node_modules/.cache/*"
    cd - > /dev/null
    log_success "Built nova-sonic-stream.zip"
    
    # Build session-manager
    log_info "Building session-manager Lambda..."
    cd "${LAMBDA_DIR}/session-manager"
    npm install --omit=dev
    zip -r "../session-manager.zip" . -x "*.git*" -x "node_modules/.cache/*"
    cd - > /dev/null
    log_success "Built session-manager.zip"
}

# ─── Upload Lambda Functions ──────────────────────────────────────────────────

upload_lambdas() {
    log_info "Uploading Lambda functions to S3..."
    
    LAMBDA_BUCKET="${STACK_NAME}-lambda-${AWS_ACCOUNT_ID}"
    
    # Create bucket if not exists
    if ! aws s3 ls "s3://${LAMBDA_BUCKET}" 2>&1 > /dev/null; then
        log_info "Creating Lambda deployment bucket..."
        aws s3 mb "s3://${LAMBDA_BUCKET}" --region "${AWS_REGION}"
    fi
    
    # Upload Lambda zips
    aws s3 cp "infrastructure/lambda/nova-sonic-stream.zip" \
        "s3://${LAMBDA_BUCKET}/lambda/nova-sonic-stream.zip"
    aws s3 cp "infrastructure/lambda/session-manager.zip" \
        "s3://${LAMBDA_BUCKET}/lambda/session-manager.zip"
    
    log_success "Lambda functions uploaded to S3"
}

# ─── Deploy CloudFormation Stack ──────────────────────────────────────────────

deploy_infrastructure() {
    log_info "Deploying CloudFormation stack: ${STACK_NAME}..."
    
    aws cloudformation deploy \
        --template-file infrastructure/beyondgrid-production.yaml \
        --stack-name "${STACK_NAME}" \
        --parameter-overrides \
            DomainName="${DOMAIN}" \
            HostedZoneId="${HOSTED_ZONE_ID}" \
            Environment="${ENVIRONMENT}" \
            BedrockRegion="${AWS_REGION}" \
        --capabilities CAPABILITY_NAMED_IAM \
        --region "${AWS_REGION}" \
        --no-fail-on-empty-changeset
    
    log_success "CloudFormation stack deployed"
}

# ─── Update Lambda Code ───────────────────────────────────────────────────────

update_lambda_code() {
    log_info "Updating Lambda function code..."
    
    # Get function names from stack
    NOVA_SONIC_FUNCTION=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" \
        --query "Stacks[0].Outputs[?OutputKey=='NovaSonicFunctionArn'].OutputValue" \
        --output text | xargs basename)
    
    SESSION_MANAGER_FUNCTION="${DOMAIN}-session-manager"
    
    # Update Nova Sonic function
    aws lambda update-function-code \
        --function-name "${NOVA_SONIC_FUNCTION}" \
        --s3-bucket "${LAMBDA_BUCKET}" \
        --s3-key "lambda/nova-sonic-stream.zip" \
        --region "${AWS_REGION}" || true
    
    # Update Session Manager function
    aws lambda update-function-code \
        --function-name "${SESSION_MANAGER_FUNCTION}" \
        --s3-bucket "${LAMBDA_BUCKET}" \
        --s3-key "lambda/session-manager.zip" \
        --region "${AWS_REGION}" || true
    
    log_success "Lambda functions updated"
}

# ─── Deploy Static Website ────────────────────────────────────────────────────

deploy_website() {
    log_info "Deploying static website to S3..."
    
    # Get bucket name from stack
    S3_BUCKET=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" \
        --query "Stacks[0].Outputs[?OutputKey=='S3BucketName'].OutputValue" \
        --output text)
    
    if [ -z "$S3_BUCKET" ] || [ "$S3_BUCKET" == "None" ]; then
        log_error "Could not find S3 bucket name from stack outputs"
        return 1
    fi
    
    # Get AppSync endpoints for config
    APPSYNC_HTTP=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" \
        --query "Stacks[0].Outputs[?OutputKey=='AppSyncEventApiEndpoint'].OutputValue" \
        --output text)
    
    APPSYNC_REALTIME=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" \
        --query "Stacks[0].Outputs[?OutputKey=='AppSyncRealtimeEndpoint'].OutputValue" \
        --output text)
    
    APPSYNC_API_KEY=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" \
        --query "Stacks[0].Outputs[?OutputKey=='AppSyncApiKey'].OutputValue" \
        --output text)
    
    REST_API=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" \
        --query "Stacks[0].Outputs[?OutputKey=='RestApiEndpoint'].OutputValue" \
        --output text)
    
    # Create config.js for frontend
    cat > public/config.js << EOF
// Auto-generated configuration
window.NOVA_CONFIG = {
  appSyncHttpEndpoint: '${APPSYNC_HTTP}',
  appSyncRealtimeEndpoint: '${APPSYNC_REALTIME}',
  appSyncApiKey: '${APPSYNC_API_KEY}',
  restApiEndpoint: '${REST_API}',
  region: '${AWS_REGION}',
  environment: '${ENVIRONMENT}'
};
EOF
    log_success "Created public/config.js with endpoints"
    
    # Sync website files
    aws s3 sync public/ "s3://${S3_BUCKET}/" \
        --delete \
        --exclude ".git/*" \
        --exclude "*.map"
    
    log_success "Website deployed to S3"
    
    # Invalidate CloudFront cache
    log_info "Invalidating CloudFront cache..."
    DISTRIBUTION_ID=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" \
        --query "Stacks[0].Outputs[?OutputKey=='CloudFrontDistributionId'].OutputValue" \
        --output text)
    
    if [ -n "$DISTRIBUTION_ID" ] && [ "$DISTRIBUTION_ID" != "None" ]; then
        aws cloudfront create-invalidation \
            --distribution-id "${DISTRIBUTION_ID}" \
            --paths "/*" \
            --query 'Invalidation.Id' \
            --output text
        log_success "CloudFront cache invalidation initiated"
    fi
}

# ─── Print Outputs ────────────────────────────────────────────────────────────

print_outputs() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    DEPLOYMENT COMPLETE! 🎉                        ║${NC}"
    echo -e "${GREEN}╠══════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${GREEN}║${NC}"
    
    # Get outputs
    WEBSITE_URL=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" \
        --query "Stacks[0].Outputs[?OutputKey=='WebsiteURL'].OutputValue" \
        --output text 2>/dev/null || echo "Pending...")
    
    APPSYNC_HTTP=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" \
        --query "Stacks[0].Outputs[?OutputKey=='AppSyncEventApiEndpoint'].OutputValue" \
        --output text 2>/dev/null || echo "Pending...")
    
    APPSYNC_WS=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" \
        --query "Stacks[0].Outputs[?OutputKey=='AppSyncRealtimeEndpoint'].OutputValue" \
        --output text 2>/dev/null || echo "Pending...")
    
    REST_API=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" \
        --query "Stacks[0].Outputs[?OutputKey=='RestApiEndpoint'].OutputValue" \
        --output text 2>/dev/null || echo "Pending...")
    
    echo -e "${GREEN}║${NC}  Website URL:        ${WEBSITE_URL}"
    echo -e "${GREEN}║${NC}  AppSync HTTP:       ${APPSYNC_HTTP}"
    echo -e "${GREEN}║${NC}  AppSync WebSocket:  ${APPSYNC_WS}"
    echo -e "${GREEN}║${NC}  REST API:           ${REST_API}"
    echo -e "${GREEN}║${NC}"
    echo -e "${GREEN}║${NC}  DNS may take up to 48 hours to propagate fully."
    echo -e "${GREEN}║${NC}  SSL certificate validation may take a few minutes."
    echo -e "${GREEN}║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
}

# ─── Main ─────────────────────────────────────────────────────────────────────

main() {
    print_banner
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --environment|-e)
                ENVIRONMENT="$2"
                shift 2
                ;;
            --region|-r)
                AWS_REGION="$2"
                shift 2
                ;;
            --skip-lambda)
                SKIP_LAMBDA=true
                shift
                ;;
            --website-only)
                WEBSITE_ONLY=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [options]"
                echo ""
                echo "Options:"
                echo "  -e, --environment   Environment (production, staging, development)"
                echo "  -r, --region        AWS Region (default: us-east-1)"
                echo "  --skip-lambda       Skip Lambda build/upload"
                echo "  --website-only      Only deploy website files"
                echo "  -h, --help          Show this help"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    if [ "$WEBSITE_ONLY" = true ]; then
        deploy_website
        print_outputs
        exit 0
    fi
    
    preflight_checks
    
    if [ "$SKIP_LAMBDA" != true ]; then
        build_lambdas
        upload_lambdas
    fi
    
    deploy_infrastructure
    update_lambda_code
    deploy_website
    print_outputs
}

main "$@"
