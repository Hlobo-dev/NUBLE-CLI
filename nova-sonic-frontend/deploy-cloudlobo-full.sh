#!/bin/bash

# ============================================
# CloudLobo Production Deployment Script
# Three-phase deployment for ECS with Docker
# ============================================

set -e

# Configuration
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="191613668206"
PROJECT_NAME="nova-sonic"
ENVIRONMENT="production"
ECR_REPO_NAME="${PROJECT_NAME}-${ENVIRONMENT}"
STACK_NAME_PHASE1="cloudlobo-infrastructure"
STACK_NAME_PHASE2="cloudlobo-service"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}"
    echo ""
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

wait_for_stack() {
    local stack_name=$1
    local desired_status=$2
    echo "Waiting for stack ${stack_name} to reach ${desired_status}..."
    
    while true; do
        status=$(aws cloudformation describe-stacks --stack-name "$stack_name" --region "$AWS_REGION" --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo "NOT_FOUND")
        
        if [[ "$status" == *"COMPLETE"* ]]; then
            if [[ "$status" == *"ROLLBACK"* ]]; then
                print_error "Stack rolled back: $status"
                aws cloudformation describe-stack-events --stack-name "$stack_name" --region "$AWS_REGION" --query 'StackEvents[?ResourceStatus==`CREATE_FAILED`].[LogicalResourceId,ResourceStatusReason]' --output table
                return 1
            fi
            print_success "Stack status: $status"
            return 0
        elif [[ "$status" == *"FAILED"* ]]; then
            print_error "Stack failed: $status"
            aws cloudformation describe-stack-events --stack-name "$stack_name" --region "$AWS_REGION" --query 'StackEvents[?ResourceStatus==`CREATE_FAILED`].[LogicalResourceId,ResourceStatusReason]' --output table
            return 1
        fi
        
        echo "  Current status: $status"
        sleep 30
    done
}

# ============================================
# PHASE 1: Deploy Infrastructure
# ============================================
phase1_infrastructure() {
    print_header "PHASE 1: Deploying Infrastructure"
    
    # Check if stack exists
    stack_status=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME_PHASE1" --region "$AWS_REGION" --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo "NOT_FOUND")
    
    if [[ "$stack_status" == "NOT_FOUND" ]]; then
        echo "Creating new infrastructure stack..."
        aws cloudformation create-stack \
            --stack-name "$STACK_NAME_PHASE1" \
            --template-body file://infrastructure/cloudlobo-phase1-infra.yaml \
            --capabilities CAPABILITY_NAMED_IAM \
            --region "$AWS_REGION" \
            --tags Key=Project,Value="$PROJECT_NAME" Key=Environment,Value="$ENVIRONMENT"
        
        print_warning "Stack creation initiated. This will take 10-15 minutes..."
        print_warning "SSL certificate requires DNS validation via Route 53."
        echo ""
        
        if ! wait_for_stack "$STACK_NAME_PHASE1" "CREATE_COMPLETE"; then
            return 1
        fi
    elif [[ "$stack_status" == "CREATE_COMPLETE" ]] || [[ "$stack_status" == "UPDATE_COMPLETE" ]]; then
        print_success "Infrastructure stack already exists and is $stack_status"
    else
        print_error "Stack is in unexpected state: $stack_status"
        return 1
    fi
    
    print_success "Phase 1 complete!"
}

# ============================================
# PHASE 2: Build and Push Docker Image
# ============================================
phase2_docker() {
    print_header "PHASE 2: Building and Pushing Docker Image"
    
    # Get ECR repository URI from CloudFormation outputs
    ECR_URI=$(aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME_PHASE1" \
        --region "$AWS_REGION" \
        --query 'Stacks[0].Outputs[?OutputKey==`ECRRepositoryUri`].OutputValue' \
        --output text)
    
    if [[ -z "$ECR_URI" ]] || [[ "$ECR_URI" == "None" ]]; then
        print_error "Could not get ECR URI from stack outputs"
        return 1
    fi
    
    echo "ECR Repository: $ECR_URI"
    
    # Login to ECR
    echo "Logging in to ECR..."
    aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
    
    # Build Docker image
    echo "Building Docker image..."
    docker build -f Dockerfile.fullstack -t "${PROJECT_NAME}:latest" .
    
    # Tag for ECR
    echo "Tagging image for ECR..."
    docker tag "${PROJECT_NAME}:latest" "${ECR_URI}:latest"
    
    # Push to ECR
    echo "Pushing image to ECR..."
    docker push "${ECR_URI}:latest"
    
    print_success "Docker image pushed to ECR!"
    
    # Verify image exists
    echo "Verifying image in ECR..."
    aws ecr describe-images --repository-name "$ECR_REPO_NAME" --region "$AWS_REGION" --query 'imageDetails[0].imageTags' --output text
    
    print_success "Phase 2 complete!"
}

# ============================================
# PHASE 3: Deploy ECS Service
# ============================================
phase3_service() {
    print_header "PHASE 3: Deploying ECS Service"
    
    # Verify Docker image exists in ECR
    image_check=$(aws ecr describe-images --repository-name "$ECR_REPO_NAME" --region "$AWS_REGION" 2>/dev/null || echo "NOT_FOUND")
    
    if [[ "$image_check" == "NOT_FOUND" ]] || [[ -z "$image_check" ]]; then
        print_error "No Docker image found in ECR! Run phase 2 first."
        return 1
    fi
    
    # Check if service stack exists
    stack_status=$(aws cloudformation describe-stacks --stack-name "$STACK_NAME_PHASE2" --region "$AWS_REGION" --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo "NOT_FOUND")
    
    if [[ "$stack_status" == "NOT_FOUND" ]]; then
        echo "Creating ECS service stack..."
        aws cloudformation create-stack \
            --stack-name "$STACK_NAME_PHASE2" \
            --template-body file://infrastructure/cloudlobo-phase2-service.yaml \
            --capabilities CAPABILITY_NAMED_IAM \
            --region "$AWS_REGION" \
            --tags Key=Project,Value="$PROJECT_NAME" Key=Environment,Value="$ENVIRONMENT"
        
        print_warning "Service deployment initiated. This will take 5-10 minutes..."
        
        if ! wait_for_stack "$STACK_NAME_PHASE2" "CREATE_COMPLETE"; then
            return 1
        fi
    elif [[ "$stack_status" == "CREATE_COMPLETE" ]] || [[ "$stack_status" == "UPDATE_COMPLETE" ]]; then
        print_success "Service stack already exists, updating..."
        aws cloudformation update-stack \
            --stack-name "$STACK_NAME_PHASE2" \
            --template-body file://infrastructure/cloudlobo-phase2-service.yaml \
            --capabilities CAPABILITY_NAMED_IAM \
            --region "$AWS_REGION" 2>/dev/null || echo "No updates needed"
    else
        print_error "Stack is in unexpected state: $stack_status"
        return 1
    fi
    
    print_success "Phase 3 complete!"
}

# ============================================
# Show Stack Outputs
# ============================================
show_outputs() {
    print_header "DEPLOYMENT OUTPUTS"
    
    echo "Infrastructure Stack Outputs:"
    aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME_PHASE1" \
        --region "$AWS_REGION" \
        --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
        --output table 2>/dev/null || echo "Stack not found"
    
    echo ""
    echo "Service Stack Outputs:"
    aws cloudformation describe-stacks \
        --stack-name "$STACK_NAME_PHASE2" \
        --region "$AWS_REGION" \
        --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
        --output table 2>/dev/null || echo "Stack not found"
}

# ============================================
# Full Deployment
# ============================================
full_deploy() {
    print_header "FULL DEPLOYMENT - All Phases"
    
    phase1_infrastructure || { print_error "Phase 1 failed!"; exit 1; }
    phase2_docker || { print_error "Phase 2 failed!"; exit 1; }
    phase3_service || { print_error "Phase 3 failed!"; exit 1; }
    
    show_outputs
    
    print_header "DEPLOYMENT COMPLETE! 🎉"
    echo ""
    echo "Your application should be available at:"
    echo "  - https://cloudlobo.com"
    echo "  - https://www.cloudlobo.com"
    echo "  - https://api.cloudlobo.com"
    echo ""
    echo "Note: DNS propagation may take a few minutes."
}

# ============================================
# Clean Up
# ============================================
cleanup() {
    print_header "CLEANUP - Deleting All Stacks"
    
    print_warning "This will delete ALL infrastructure!"
    read -p "Are you sure? (yes/no): " confirm
    
    if [[ "$confirm" != "yes" ]]; then
        echo "Cancelled."
        return
    fi
    
    # Delete service first (depends on infrastructure)
    echo "Deleting service stack..."
    aws cloudformation delete-stack --stack-name "$STACK_NAME_PHASE2" --region "$AWS_REGION" 2>/dev/null || true
    
    echo "Waiting for service stack deletion..."
    aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME_PHASE2" --region "$AWS_REGION" 2>/dev/null || true
    
    # Delete infrastructure
    echo "Deleting infrastructure stack..."
    aws cloudformation delete-stack --stack-name "$STACK_NAME_PHASE1" --region "$AWS_REGION" 2>/dev/null || true
    
    echo "Waiting for infrastructure stack deletion..."
    aws cloudformation wait stack-delete-complete --stack-name "$STACK_NAME_PHASE1" --region "$AWS_REGION" 2>/dev/null || true
    
    print_success "Cleanup complete!"
}

# ============================================
# Main Menu
# ============================================
case "$1" in
    "phase1"|"1"|"infra")
        phase1_infrastructure
        ;;
    "phase2"|"2"|"docker")
        phase2_docker
        ;;
    "phase3"|"3"|"service")
        phase3_service
        ;;
    "full"|"all"|"deploy")
        full_deploy
        ;;
    "outputs"|"status")
        show_outputs
        ;;
    "cleanup"|"delete")
        cleanup
        ;;
    *)
        echo ""
        echo "CloudLobo Production Deployment"
        echo "================================"
        echo ""
        echo "Usage: $0 <command>"
        echo ""
        echo "Commands:"
        echo "  phase1, 1, infra   - Deploy infrastructure (VPC, ALB, ECR, DynamoDB, etc.)"
        echo "  phase2, 2, docker  - Build and push Docker image to ECR"
        echo "  phase3, 3, service - Deploy ECS service"
        echo "  full, all, deploy  - Run all phases"
        echo "  outputs, status    - Show stack outputs"
        echo "  cleanup, delete    - Delete all stacks"
        echo ""
        echo "Recommended: Run './deploy-cloudlobo-full.sh full' for complete deployment"
        echo ""
        ;;
esac
