# NUBLE Migration - Completed Tasks

**Date:** February 2, 2026  
**Migration:** Kyperian → NUBLE

---

## ✅ Completed: Code Rebranding

All "kyperian" references have been renamed to "nuble" across the codebase:

### Infrastructure Files Updated:
- ✅ `infrastructure/aws/deploy.sh` - Main deployment script
- ✅ `infrastructure/aws/deploy-advisor.sh` - Advisor deployment
- ✅ `infrastructure/aws/deploy-decision-engine.sh` - Decision engine deployment
- ✅ `infrastructure/aws/deploy-enterprise.sh` - Enterprise deployment
- ✅ `infrastructure/aws/deploy/deploy_v2.sh` - V2 deployment script
- ✅ `infrastructure/aws/dashboard/index.html` - Signal dashboard

### Application Files Updated:
- ✅ `Dockerfile` - Container configuration
- ✅ `Nuble_mltrading_system` - ML trading system docs

### Test Files Updated:
- ✅ `tests/real_baseline_*.txt` - All baseline test outputs
- ✅ `tests/trace_output.txt` - Trace outputs

---

## 🔧 AWS Account Details

| Account | Account ID | Access Key ID | Status |
|---------|------------|---------------|--------|
| OLD (Kyperian) | 456309723884 | AKIAWUPRN2LWASHFJEMO | Deprecated |
| NEW (NUBLE) | 191613668206 | AKIASZHIQRNXH5EZJOMK | **Pending Secret Key** |

---

## ⏳ Pending: AWS Deployment

### Required Before Deployment:

1. **AWS Secret Access Key** - You need to provide the Secret Access Key for:
   - Account ID: `191613668206`
   - Access Key: `AKIASZHIQRNXH5EZJOMK`

### Configure AWS Profile:

Once you have the secret key, run:

```bash
aws configure --profile nuble
# Enter:
#   Access Key ID: AKIASZHIQRNXH5EZJOMK
#   Secret Access Key: [YOUR SECRET KEY]
#   Region: us-east-1
#   Output: json
```

### Verify Access:

```bash
aws sts get-caller-identity --profile nuble
```

### Deploy Infrastructure:

```bash
cd /Users/humbertolobo/Desktop/NUBLE-CLI/infrastructure/aws
export AWS_PROFILE=nuble
./deploy.sh production deploy
```

---

## 📋 Resources to Deploy

| Stack | Description | Est. Time |
|-------|-------------|-----------|
| `nuble-production-vpc` | VPC, Subnets, Security Groups | 2 min |
| `nuble-production-lambda` | Signal Validator + DynamoDB | 3 min |
| `nuble-production-api` | API Gateway | 2 min |
| `nuble-production-cache` | ElastiCache Redis | 10 min |
| `nuble-production-ecs` | ECS Fargate | 10 min |
| `nuble-production-monitoring` | CloudWatch Dashboards | 2 min |

**Total Deployment Time:** ~30 minutes

---

## 📝 Post-Deployment Tasks

1. [ ] Get new webhook URL from deployment output
2. [ ] Update TradingView alerts with new URL
3. [ ] Run `python test_connections.py` to verify
4. [ ] Test webhook with sample signal
5. [ ] Decommission old Kyperian account

---

## 🚀 Quick Start (After Secret Key)

```bash
# 1. Configure AWS credentials
aws configure --profile nuble

# 2. Deploy
cd /Users/humbertolobo/Desktop/NUBLE-CLI/infrastructure/aws
export AWS_PROFILE=nuble
./deploy.sh production deploy

# 3. Test
python /Users/humbertolobo/Desktop/NUBLE-CLI/test_connections.py
```
