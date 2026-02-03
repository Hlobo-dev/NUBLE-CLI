# NUBLE AWS Migration Plan
## From Kyperian (456309723884) → NUBLE (191613668206)

**Date:** February 2, 2026  
**Prepared by:** NUBLE Migration System

---

## 📋 Executive Summary

This migration plan covers:
1. **Rebranding**: All "kyperian" references → "nuble"
2. **New AWS Account**: Deploy to account `191613668206`
3. **Infrastructure**: Complete AWS stack deployment
4. **Data Migration**: Export signals from old DynamoDB, import to new
5. **Webhook Update**: New API Gateway endpoint for TradingView

---

## 🏗️ Infrastructure Components to Deploy

### CloudFormation Stacks (in order):
| Stack | Description | Est. Cost/Month |
|-------|-------------|-----------------|
| `nuble-production-vpc` | VPC, Subnets, Security Groups | $0 |
| `nuble-production-lambda` | Signal Validator + DynamoDB | ~$5 |
| `nuble-production-api` | API Gateway + WAF | ~$10 |
| `nuble-production-cache` | ElastiCache Redis | ~$50 |
| `nuble-production-ecs` | ECS Fargate (2-20 tasks) | ~$200 |
| `nuble-production-monitoring` | CloudWatch Dashboards | ~$25 |

**Total Estimated:** ~$290/month

### DynamoDB Tables:
- `nuble-production-signals` - LuxAlgo webhook signals
- `nuble-production-decisions` - Trading decisions

### Lambda Functions:
- `nuble-production-signal-validator` - Webhook receiver
- `nuble-production-decision-engine` - Trading logic
- `nuble-production-advisor` - AI advisor
- `nuble-production-telegram-bot` - Notifications

### API Gateway Endpoints:
- `POST /webhook` - TradingView signals
- `POST /webhook/luxalgo` - LuxAlgo specific
- `POST /webhook/mtf` - Multi-timeframe
- `GET /health` - Health check
- `POST /decision` - Get trading decision
- `GET /signals/{symbol}` - Query signals

---

## 🔐 New AWS Credentials

**Old Account (Kyperian):**
- Account ID: `456309723884`
- Access Key: `AKIAWUPRN2LWASHFJEMO`
- Profile: `kyperian`

**New Account (NUBLE Production):**
- Account ID: `191613668206`
- Access Key: `AKIASZHIQRNXH5EZJOMK`
- Secret Key: `[TO BE PROVIDED]`
- Profile: `nuble-prod`

---

## 📝 Files to Update

### Configuration Files:
1. `.env` - Environment variables
2. `infrastructure/aws/deploy.sh` - Main deployment
3. `infrastructure/aws/deploy-decision-engine.sh`
4. `infrastructure/aws/deploy-advisor.sh`
5. `infrastructure/aws/.env.example`

### Source Code (kyperian → nuble):
- `src/nuble/data/aggregator.py` - DynamoDB table names
- `src/nuble/decision/ultimate_engine.py` - Table references
- `infrastructure/aws/lambda/*/handler.py` - Lambda handlers
- `infrastructure/aws/dashboard/index.html` - UI branding

---

## 🚀 Migration Steps

### Phase 1: Local Preparation (5 min)
```bash
# 1. Update .env with new credentials
# 2. Run rename script to change kyperian → nuble
# 3. Test locally
```

### Phase 2: AWS Setup (10 min)
```bash
# 1. Configure AWS CLI profile
aws configure --profile nuble-prod
# Access Key: AKIASZHIQRNXH5EZJOMK
# Secret Key: [your secret]
# Region: us-east-1
# Output: json

# 2. Verify access
aws sts get-caller-identity --profile nuble-prod
```

### Phase 3: Deploy Infrastructure (20-30 min)
```bash
cd infrastructure/aws
./deploy.sh production deploy
```

Deployment order:
1. VPC (2 min)
2. Lambda + DynamoDB (3 min)
3. API Gateway (2 min)
4. ElastiCache (10 min)
5. ECS (10 min)
6. Monitoring (2 min)

### Phase 4: Data Migration (5 min)
```bash
# Export signals from old account
python migrate_signals.py --export

# Import to new account
python migrate_signals.py --import
```

### Phase 5: Update TradingView Webhooks (5 min)
1. Get new webhook URL from deployment output
2. Update TradingView alerts with new URL
3. Test with a manual signal

### Phase 6: Validation (10 min)
```bash
# Run connection tests
python test_connections.py

# Send test webhook
curl -X POST https://NEW-API-ID.execute-api.us-east-1.amazonaws.com/production/webhook \
  -H "Content-Type: application/json" \
  -d '{"action":"BUY","symbol":"BTCUSD","price":95000,"timeframe":"4h"}'
```

---

## ⚠️ Rollback Plan

If migration fails:
1. Keep old account active for 30 days
2. Old webhook still functional
3. Can switch back by updating TradingView alerts

---

## 📊 Post-Migration Checklist

- [ ] VPC deployed and healthy
- [ ] DynamoDB tables created
- [ ] Lambda functions working
- [ ] API Gateway responding
- [ ] Webhook receiving signals
- [ ] CLI connecting to new backend
- [ ] TradingView alerts updated
- [ ] Test signal stored successfully
- [ ] Old account decommissioned

---

## 🔗 Important URLs (After Deployment)

| Resource | URL |
|----------|-----|
| Webhook | `https://[API-ID].execute-api.us-east-1.amazonaws.com/production/webhook` |
| Console | `https://us-east-1.console.aws.amazon.com/console/home?region=us-east-1` |
| CloudWatch | `https://us-east-1.console.aws.amazon.com/cloudwatch/home?region=us-east-1#dashboards:name=NUBLE-production-Main` |
| DynamoDB | `https://us-east-1.console.aws.amazon.com/dynamodbv2/home?region=us-east-1#tables` |

