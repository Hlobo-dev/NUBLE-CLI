#!/bin/bash
# ============================================
# KYPERIAN ELITE - AWS Deployment Script
# ============================================
# Deploys entire infrastructure to AWS
# Usage: ./deploy.sh [environment] [action]
# Example: ./deploy.sh production deploy
# ============================================

set -e

# Configuration
ENVIRONMENT=${1:-production}
ACTION=${2:-deploy}
AWS_REGION=${AWS_REGION:-us-east-1}
PROJECT_NAME="kyperian"
STACK_PREFIX="${PROJECT_NAME}-${ENVIRONMENT}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is not installed. Please install it first."
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured. Run 'aws configure' first."
        exit 1
    fi
    
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    log_success "AWS Account: ${AWS_ACCOUNT_ID}"
    log_success "AWS Region: ${AWS_REGION}"
}

# Deploy VPC
deploy_vpc() {
    log_info "Deploying VPC infrastructure..."
    
    aws cloudformation deploy \
        --template-file cloudformation/vpc.yaml \
        --stack-name "${STACK_PREFIX}-vpc" \
        --parameter-overrides Environment=${ENVIRONMENT} \
        --capabilities CAPABILITY_NAMED_IAM \
        --region ${AWS_REGION} \
        --no-fail-on-empty-changeset
    
    log_success "VPC deployed successfully"
}

# Deploy Lambda and DynamoDB
deploy_lambda() {
    log_info "Deploying Lambda and DynamoDB..."
    
    # Package Lambda code
    log_info "Packaging Lambda function..."
    cd lambda/signal_validator
    pip install -r requirements.txt -t ./package --quiet
    cd package && zip -r ../function.zip . --quiet
    cd .. && zip -g function.zip handler.py --quiet
    cd ../..
    
    # Upload to S3 (create bucket if needed)
    LAMBDA_BUCKET="${PROJECT_NAME}-${ENVIRONMENT}-lambda-${AWS_ACCOUNT_ID}"
    aws s3 mb s3://${LAMBDA_BUCKET} --region ${AWS_REGION} 2>/dev/null || true
    aws s3 cp lambda/signal_validator/function.zip s3://${LAMBDA_BUCKET}/function.zip
    
    # Deploy Lambda stack
    aws cloudformation deploy \
        --template-file cloudformation/lambda.yaml \
        --stack-name "${STACK_PREFIX}-lambda" \
        --parameter-overrides \
            Environment=${ENVIRONMENT} \
            NetworkStackName="${STACK_PREFIX}-vpc" \
        --capabilities CAPABILITY_NAMED_IAM CAPABILITY_AUTO_EXPAND \
        --region ${AWS_REGION} \
        --no-fail-on-empty-changeset
    
    log_success "Lambda and DynamoDB deployed successfully"
}

# Deploy API Gateway
deploy_api_gateway() {
    log_info "Deploying API Gateway..."
    
    LAMBDA_ARN=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_PREFIX}-lambda" \
        --query "Stacks[0].Outputs[?OutputKey=='SignalValidatorFunctionArn'].OutputValue" \
        --output text \
        --region ${AWS_REGION})
    
    aws cloudformation deploy \
        --template-file cloudformation/api-gateway.yaml \
        --stack-name "${STACK_PREFIX}-api" \
        --parameter-overrides \
            Environment=${ENVIRONMENT} \
            NetworkStackName="${STACK_PREFIX}-vpc" \
            SignalValidatorLambdaArn=${LAMBDA_ARN} \
        --capabilities CAPABILITY_NAMED_IAM \
        --region ${AWS_REGION} \
        --no-fail-on-empty-changeset
    
    log_success "API Gateway deployed successfully"
}

# Deploy ElastiCache
deploy_cache() {
    log_info "Deploying ElastiCache Redis..."
    
    aws cloudformation deploy \
        --template-file cloudformation/elasticache.yaml \
        --stack-name "${STACK_PREFIX}-cache" \
        --parameter-overrides \
            Environment=${ENVIRONMENT} \
            NetworkStackName="${STACK_PREFIX}-vpc" \
        --capabilities CAPABILITY_NAMED_IAM \
        --region ${AWS_REGION} \
        --no-fail-on-empty-changeset
    
    log_success "ElastiCache deployed successfully"
}

# Build and push Docker image
build_and_push_image() {
    log_info "Building Docker image..."
    
    # Create ECR repository if it doesn't exist
    aws ecr describe-repositories --repository-names ${PROJECT_NAME}-${ENVIRONMENT} --region ${AWS_REGION} 2>/dev/null || \
        aws ecr create-repository --repository-name ${PROJECT_NAME}-${ENVIRONMENT} --region ${AWS_REGION}
    
    # Login to ECR
    aws ecr get-login-password --region ${AWS_REGION} | \
        docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com
    
    # Build image
    cd ../../
    docker build -t ${PROJECT_NAME}:${ENVIRONMENT} -f Dockerfile .
    
    # Tag and push
    ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PROJECT_NAME}-${ENVIRONMENT}:latest"
    docker tag ${PROJECT_NAME}:${ENVIRONMENT} ${ECR_URI}
    docker push ${ECR_URI}
    
    cd infrastructure/aws
    log_success "Docker image pushed to ECR: ${ECR_URI}"
    
    echo ${ECR_URI}
}

# Deploy ECS
deploy_ecs() {
    log_info "Deploying ECS Fargate..."
    
    # Build and push image first
    CONTAINER_IMAGE=$(build_and_push_image)
    
    aws cloudformation deploy \
        --template-file cloudformation/ecs.yaml \
        --stack-name "${STACK_PREFIX}-ecs" \
        --parameter-overrides \
            Environment=${ENVIRONMENT} \
            NetworkStackName="${STACK_PREFIX}-vpc" \
            LambdaStackName="${STACK_PREFIX}-lambda" \
            ContainerImage=${CONTAINER_IMAGE} \
        --capabilities CAPABILITY_NAMED_IAM \
        --region ${AWS_REGION} \
        --no-fail-on-empty-changeset
    
    log_success "ECS Fargate deployed successfully"
}

# Deploy Monitoring
deploy_monitoring() {
    log_info "Deploying CloudWatch monitoring..."
    
    aws cloudformation deploy \
        --template-file cloudformation/monitoring.yaml \
        --stack-name "${STACK_PREFIX}-monitoring" \
        --parameter-overrides \
            Environment=${ENVIRONMENT} \
            ECSStackName="${STACK_PREFIX}-ecs" \
            LambdaStackName="${STACK_PREFIX}-lambda" \
            CacheStackName="${STACK_PREFIX}-cache" \
        --capabilities CAPABILITY_NAMED_IAM \
        --region ${AWS_REGION} \
        --no-fail-on-empty-changeset
    
    log_success "Monitoring deployed successfully"
}

# Full deployment
deploy_all() {
    log_info "Starting full KYPERIAN ELITE deployment..."
    echo ""
    
    deploy_vpc
    deploy_cache
    deploy_lambda
    deploy_api_gateway
    deploy_ecs
    deploy_monitoring
    
    echo ""
    log_success "============================================"
    log_success "  KYPERIAN ELITE DEPLOYMENT COMPLETE! 🚀"
    log_success "============================================"
    echo ""
    
    # Print outputs
    API_ENDPOINT=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_PREFIX}-api" \
        --query "Stacks[0].Outputs[?OutputKey=='ApiEndpoint'].OutputValue" \
        --output text \
        --region ${AWS_REGION})
    
    ALB_DNS=$(aws cloudformation describe-stacks \
        --stack-name "${STACK_PREFIX}-ecs" \
        --query "Stacks[0].Outputs[?OutputKey=='LoadBalancerDNS'].OutputValue" \
        --output text \
        --region ${AWS_REGION})
    
    echo ""
    log_info "API Gateway Endpoint: ${API_ENDPOINT}"
    log_info "Load Balancer DNS: ${ALB_DNS}"
    log_info "Dashboard: https://${AWS_REGION}.console.aws.amazon.com/cloudwatch/home?region=${AWS_REGION}#dashboards:name=KYPERIAN-${ENVIRONMENT}-Main"
    echo ""
}

# Delete all stacks
delete_all() {
    log_warning "Deleting all KYPERIAN ELITE stacks..."
    
    STACKS=(
        "${STACK_PREFIX}-monitoring"
        "${STACK_PREFIX}-ecs"
        "${STACK_PREFIX}-api"
        "${STACK_PREFIX}-cache"
        "${STACK_PREFIX}-lambda"
        "${STACK_PREFIX}-vpc"
    )
    
    for stack in "${STACKS[@]}"; do
        log_info "Deleting ${stack}..."
        aws cloudformation delete-stack --stack-name ${stack} --region ${AWS_REGION} 2>/dev/null || true
    done
    
    log_info "Waiting for stacks to be deleted..."
    for stack in "${STACKS[@]}"; do
        aws cloudformation wait stack-delete-complete --stack-name ${stack} --region ${AWS_REGION} 2>/dev/null || true
    done
    
    log_success "All stacks deleted successfully"
}

# Print help
print_help() {
    echo "KYPERIAN ELITE Deployment Script"
    echo ""
    echo "Usage: ./deploy.sh [environment] [action]"
    echo ""
    echo "Environments:"
    echo "  development    Development environment"
    echo "  staging        Staging environment"
    echo "  production     Production environment (default)"
    echo ""
    echo "Actions:"
    echo "  deploy         Deploy all stacks (default)"
    echo "  vpc            Deploy only VPC"
    echo "  lambda         Deploy only Lambda/DynamoDB"
    echo "  api            Deploy only API Gateway"
    echo "  cache          Deploy only ElastiCache"
    echo "  ecs            Deploy only ECS"
    echo "  monitoring     Deploy only Monitoring"
    echo "  delete         Delete all stacks"
    echo "  help           Print this help"
    echo ""
}

# Main
check_prerequisites

case ${ACTION} in
    deploy)
        deploy_all
        ;;
    vpc)
        deploy_vpc
        ;;
    lambda)
        deploy_lambda
        ;;
    api)
        deploy_api_gateway
        ;;
    cache)
        deploy_cache
        ;;
    ecs)
        deploy_ecs
        ;;
    monitoring)
        deploy_monitoring
        ;;
    delete)
        delete_all
        ;;
    help|--help|-h)
        print_help
        ;;
    *)
        log_error "Unknown action: ${ACTION}"
        print_help
        exit 1
        ;;
esac
