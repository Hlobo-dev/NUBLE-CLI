# Nova Sonic Production Deployment - beyondgrid.xyz

## 🏗️ Architecture Overview

This deployment uses AWS best practices for real-time voice AI applications:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         beyondgrid.xyz Production Architecture                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌─────────────┐     ┌─────────────────────────────────────────────────────┐   │
│   │   Browser   │     │                    AWS Cloud                        │   │
│   │   Client    │     │                                                     │   │
│   └──────┬──────┘     │  ┌────────────────────────────────────────────────┐ │   │
│          │            │  │              Route 53 DNS                       │ │   │
│          │ HTTPS      │  │         beyondgrid.xyz → CloudFront            │ │   │
│          │            │  └────────────────┬───────────────────────────────┘ │   │
│          │            │                   │                                 │   │
│          │            │  ┌────────────────▼───────────────────────────────┐ │   │
│          │            │  │           CloudFront CDN                       │ │   │
│          │            │  │    • SSL/TLS termination                       │ │   │
│          │            │  │    • Global edge caching                       │ │   │
│          │            │  │    • WAF protection                            │ │   │
│          │            │  └────────────────┬───────────────────────────────┘ │   │
│          │            │                   │                                 │   │
│          │            │  ┌────────────────▼───────────────────────────────┐ │   │
│          │            │  │              S3 Bucket                         │ │   │
│          │            │  │    • Static website files                      │ │   │
│          │            │  │    • index.html, main.js, etc.                 │ │   │
│          │            │  └────────────────────────────────────────────────┘ │   │
│          │            │                                                     │   │
│          │            │                                                     │   │
│   ┌──────▼──────┐     │  ┌────────────────────────────────────────────────┐ │   │
│   │  WebSocket  │────────▶        AppSync Events API                     │ │   │
│   │  Connection │     │  │    • Real-time pub/sub                        │ │   │
│   └─────────────┘     │  │    • Automatic scaling                         │ │   │
│                       │  │    • Connection management                     │ │   │
│                       │  │                                                │ │   │
│                       │  │  Channels:                                     │ │   │
│                       │  │    /audio/{sessionId}/input  (client → server)│ │   │
│                       │  │    /audio/{sessionId}/output (server → client)│ │   │
│                       │  │    /session/{sessionId}/control               │ │   │
│                       │  └────────────────┬───────────────────────────────┘ │   │
│                       │                   │                                 │   │
│                       │  ┌────────────────▼───────────────────────────────┐ │   │
│                       │  │           Lambda Functions                     │ │   │
│                       │  │                                                │ │   │
│                       │  │  ┌─────────────────┐  ┌────────────────────┐  │ │   │
│                       │  │  │ Session Manager │  │ Nova Sonic Stream  │  │ │   │
│                       │  │  │    Lambda       │  │      Lambda        │  │ │   │
│                       │  │  │                 │  │                    │  │ │   │
│                       │  │  │ • Create/End    │  │ • Bedrock stream   │  │ │   │
│                       │  │  │ • History       │  │ • Audio processing │  │ │   │
│                       │  │  │ • Validation    │  │ • Keep-alive       │  │ │   │
│                       │  │  └─────────┬───────┘  └─────────┬──────────┘  │ │   │
│                       │  │            │                    │             │ │   │
│                       │  └────────────┼────────────────────┼─────────────┘ │   │
│                       │               │                    │               │   │
│                       │  ┌────────────▼────────────────────▼─────────────┐ │   │
│                       │  │              DynamoDB                          │ │   │
│                       │  │    • Chat history table                       │ │   │
│                       │  │    • Session state table                      │ │   │
│                       │  │    • TTL for auto-cleanup                     │ │   │
│                       │  └────────────────────────────────────────────────┘ │   │
│                       │                                                     │   │
│                       │  ┌────────────────────────────────────────────────┐ │   │
│                       │  │         Amazon Bedrock                         │ │   │
│                       │  │    • Nova Sonic (amazon.nova-2-sonic-v1:0)    │ │   │
│                       │  │    • Bidirectional HTTP/2 streaming           │ │   │
│                       │  │    • Real-time speech processing              │ │   │
│                       │  └────────────────────────────────────────────────┘ │   │
│                       │                                                     │   │
│                       └─────────────────────────────────────────────────────┘   │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start Deployment

### Prerequisites

1. **AWS CLI** configured with appropriate credentials
2. **Node.js 20+** installed
3. **Domain** registered (beyondgrid.xyz)
4. **AWS Account** with Bedrock access enabled

### Step 1: Configure AWS Credentials

```bash
# Verify AWS CLI is configured
aws sts get-caller-identity

# Ensure Bedrock access is enabled in us-east-1
aws bedrock list-foundation-models --region us-east-1 --query "modelSummaries[?contains(modelId, 'nova-sonic')]"
```

### Step 2: Deploy Infrastructure

```bash
# Make deploy script executable
chmod +x deploy-beyondgrid.sh

# Run full deployment
./deploy-beyondgrid.sh

# Or deploy with specific options
./deploy-beyondgrid.sh --environment production --region us-east-1
```

### Step 3: Update Domain Nameservers

If this is a new hosted zone, update your domain registrar (GoDaddy, Namecheap, etc.) with the Route 53 nameservers shown in the deployment output.

### Step 4: Verify Deployment

```bash
# Check stack status
aws cloudformation describe-stacks --stack-name nova-sonic-beyondgrid

# Test the website
curl -I https://beyondgrid.xyz
```

## 📁 Project Structure

```
nova-sonic-debug/
├── infrastructure/
│   ├── beyondgrid-production.yaml    # Main CloudFormation template
│   └── lambda/
│       ├── nova-sonic-stream/        # Bedrock streaming Lambda
│       │   ├── index.js
│       │   └── package.json
│       └── session-manager/          # Session management Lambda
│           ├── index.js
│           └── package.json
├── public/
│   ├── index.html                    # Main HTML
│   ├── config.js                     # Auto-generated config (after deploy)
│   └── src/
│       ├── main.js                   # Main application
│       └── lib/
│           ├── AppSyncEventsClient.js # AppSync WebSocket client
│           └── play/                  # Audio playback
├── deploy-beyondgrid.sh              # Deployment script
└── DEPLOYMENT-BEYONDGRID.md          # This file
```

## 🔧 Configuration

### Environment Variables

The Lambda functions use these environment variables (set via CloudFormation):

| Variable | Description |
|----------|-------------|
| `BEDROCK_REGION` | AWS region for Bedrock (default: us-east-1) |
| `NOVA_SONIC_MODEL_ID` | Model ID (default: amazon.nova-2-sonic-v1:0) |
| `CHAT_HISTORY_TABLE` | DynamoDB table for chat history |
| `SESSION_TABLE` | DynamoDB table for sessions |
| `APPSYNC_ENDPOINT` | AppSync HTTP endpoint |
| `APPSYNC_API_KEY` | AppSync API key |

### Frontend Configuration

After deployment, `public/config.js` is auto-generated with:

```javascript
window.NOVA_CONFIG = {
  appSyncHttpEndpoint: 'https://xxx.appsync-api.us-east-1.amazonaws.com',
  appSyncRealtimeEndpoint: 'wss://xxx.appsync-realtime-api.us-east-1.amazonaws.com',
  appSyncApiKey: 'da2-xxxxxxxxxxxxxxxxxxxxxxxx',
  restApiEndpoint: 'https://xxx.execute-api.us-east-1.amazonaws.com/production',
  region: 'us-east-1',
  environment: 'production'
};
```

## 🔒 Security Features

### WAF (Web Application Firewall)
- AWS Managed Rules (Common Rule Set)
- Known Bad Inputs protection
- Rate limiting (2000 req/5min per IP)

### Authentication Options
1. **API Key** (default) - Simple, suitable for demos
2. **Cognito** - Full user authentication (can be added)
3. **IAM** - For backend services
4. **Lambda Authorizer** - Custom auth logic

### Data Protection
- S3 bucket encryption (AES-256)
- DynamoDB encryption at rest
- HTTPS-only traffic
- CloudFront Origin Access Control

## 📊 Monitoring

### CloudWatch Dashboard
After deployment, a dashboard is created at:
`https://console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=beyondgrid.xyz-nova-sonic`

### Key Metrics
- Lambda invocations, errors, duration
- AppSync events (publish/subscribe success)
- DynamoDB consumed capacity

### Alarms
- Lambda error rate > 5 in 5 minutes
- Lambda duration > 30 seconds average

## 🔄 Updates & Maintenance

### Update Website Only
```bash
./deploy-beyondgrid.sh --website-only
```

### Update Lambda Code
```bash
./deploy-beyondgrid.sh --skip-lambda  # Skip rebuild
# or full redeploy
./deploy-beyondgrid.sh
```

### Invalidate CloudFront Cache
```bash
aws cloudfront create-invalidation \
  --distribution-id $(aws cloudformation describe-stacks \
    --stack-name nova-sonic-beyondgrid \
    --query "Stacks[0].Outputs[?OutputKey=='CloudFrontDistributionId'].OutputValue" \
    --output text) \
  --paths "/*"
```

## 💰 Cost Estimation

| Service | Estimated Monthly Cost |
|---------|----------------------|
| CloudFront | $1-10 (depending on traffic) |
| S3 | < $1 |
| AppSync Events | $1-5 (depending on connections) |
| Lambda | $5-20 (depending on usage) |
| DynamoDB | $1-5 (pay per request) |
| Bedrock Nova Sonic | Variable (per minute of audio) |
| Route 53 | $0.50/zone + queries |
| WAF | $5 + $1 per million requests |

**Total estimated: $15-50/month** for moderate usage

## 🐛 Troubleshooting

### Certificate Not Validating
```bash
# Check certificate status
aws acm describe-certificate \
  --certificate-arn $(aws cloudformation describe-stacks \
    --stack-name nova-sonic-beyondgrid \
    --query "Stacks[0].Outputs[?OutputKey=='SSLCertificateArn'].OutputValue" \
    --output text)
```

### WebSocket Connection Fails
1. Check AppSync API key is valid
2. Verify CORS settings
3. Check browser console for errors

### Lambda Timeout
- Increase Lambda timeout in CloudFormation
- Check Bedrock endpoint connectivity
- Review CloudWatch logs

### Audio Not Working
1. Check microphone permissions
2. Verify sample rate (16kHz required)
3. Check browser console for AudioContext errors

## 📚 Additional Resources

- [AWS AppSync Events Documentation](https://docs.aws.amazon.com/appsync/latest/eventapi/)
- [Amazon Nova Sonic Documentation](https://docs.aws.amazon.com/nova/latest/userguide/speech.html)
- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)

## 📝 License

MIT License - See LICENSE file
