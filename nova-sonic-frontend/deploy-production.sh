#!/bin/bash
# Deploy Nova Sonic Debug to AWS ECS
# Domain: cloudlobo.com

set -e

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║        Nova Sonic Debug - Production Deployment                   ║"
echo "║        Target: cloudlobo.com                                      ║"
echo "╚══════════════════════════════════════════════════════════════════╝"

# Configuration
AWS_REGION="us-east-1"
AWS_ACCOUNT_ID="456309723884"
ECR_REPO="nova-sonic-debug"
ECS_CLUSTER="nova-sonic-cluster"
ECS_SERVICE="nova-sonic-service"
HOSTED_ZONE_ID="Z01076362AU2XFIBK1J14"
DOMAIN="cloudlobo.com"

# Load credentials from .env if exists
if [ -f .env ]; then
    echo "📋 Loading credentials from .env..."
    export $(grep -v '^#' .env | xargs)
fi

echo ""
echo "Step 1: Login to ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

echo ""
echo "Step 2: Create ECR repository (if not exists)..."
aws ecr describe-repositories --repository-names $ECR_REPO --region $AWS_REGION 2>/dev/null || \
    aws ecr create-repository --repository-name $ECR_REPO --region $AWS_REGION

echo ""
echo "Step 3: Build Docker image..."
docker build -t $ECR_REPO:latest .

echo ""
echo "Step 4: Tag and push to ECR..."
docker tag $ECR_REPO:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest

echo ""
echo "Step 5: Request SSL Certificate..."
CERT_ARN=$(aws acm list-certificates --region $AWS_REGION --query "CertificateSummaryList[?DomainName=='$DOMAIN'].CertificateArn" --output text)

if [ -z "$CERT_ARN" ]; then
    echo "Creating new certificate for $DOMAIN..."
    CERT_ARN=$(aws acm request-certificate \
        --domain-name $DOMAIN \
        --subject-alternative-names "*.$DOMAIN" \
        --validation-method DNS \
        --region $AWS_REGION \
        --query 'CertificateArn' \
        --output text)
    echo "Certificate ARN: $CERT_ARN"
    echo ""
    echo "⚠️  You need to validate the certificate via DNS!"
    echo "   Run: aws acm describe-certificate --certificate-arn $CERT_ARN --region $AWS_REGION"
    echo "   Then add the CNAME record to Route 53."
else
    echo "Using existing certificate: $CERT_ARN"
fi

echo ""
echo "Step 6: Deploy CloudFormation stack..."
aws cloudformation deploy \
    --template-file cloudformation-stack.yaml \
    --stack-name nova-sonic-production \
    --parameter-overrides \
        DomainName=$DOMAIN \
        HostedZoneId=$HOSTED_ZONE_ID \
        CertificateArn=$CERT_ARN \
        ECRImageUri=$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPO:latest \
    --capabilities CAPABILITY_IAM \
    --region $AWS_REGION

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    DEPLOYMENT COMPLETE! 🎉                        ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║                                                                   ║"
echo "║  Your app will be available at:                                   ║"
echo "║                                                                   ║"
echo "║    https://cloudlobo.com                                          ║"
echo "║    https://www.cloudlobo.com                                      ║"
echo "║    https://sonic.cloudlobo.com                                    ║"
echo "║                                                                   ║"
echo "║  DNS propagation may take a few minutes to 48 hours.              ║"
echo "║                                                                   ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
