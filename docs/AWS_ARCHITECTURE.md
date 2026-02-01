# KYPERIAN ELITE - AWS Production Architecture

## 🏗️ Complete Infrastructure Documentation

This document provides a comprehensive overview of the KYPERIAN ELITE AWS infrastructure, designed to be the **world's most intelligent and robust trading system**.

---

## 📐 Architecture Philosophy

### Design Principles

1. **Ultra-Low Latency** - Every millisecond counts in trading
2. **High Availability** - Multi-AZ deployment with automatic failover
3. **Auto-Scaling** - Handle 10x traffic spikes automatically
4. **Cost Efficiency** - Pay only for what you use
5. **Security First** - Defense in depth at every layer

### Latency Budget

| Component | Target | Budget |
|-----------|--------|--------|
| CloudFront Edge | <10ms | 10ms |
| API Gateway | <5ms | 5ms |
| Lambda Validation | <10ms | 10ms |
| EventBridge | <5ms | 5ms |
| Redis Cache | <1ms | 1ms |
| DynamoDB | <10ms | 10ms |
| ECS Processing | <50ms | 50ms |
| **Total** | **<100ms** | **91ms** |

---

## 🌐 Network Architecture

### VPC Design

```
VPC: 10.0.0.0/16
├── Public Subnets
│   ├── 10.0.1.0/24 (AZ-a) - ALB, NAT Gateway
│   └── 10.0.2.0/24 (AZ-b) - ALB
└── Private Subnets
    ├── 10.0.10.0/24 (AZ-a) - ECS, Lambda, Redis
    └── 10.0.11.0/24 (AZ-b) - ECS, Lambda, Redis
```

### Security Groups

| Name | Inbound | Outbound |
|------|---------|----------|
| ALB-SG | 80, 443 from 0.0.0.0/0 | All |
| ECS-SG | 8000 from ALB-SG | All |
| Redis-SG | 6379 from ECS-SG | All |
| Lambda-SG | N/A (outbound only) | All |

---

## 🚀 Service Architecture

### 1. API Gateway (Edge Layer)

**Purpose**: Secure entry point with rate limiting and WAF protection

**Configuration**:
- HTTP API (lower latency than REST API)
- Rate limiting: 50 req/sec sustained, 100 burst
- WAF rules: Rate-based, Common Rule Set, Bad Inputs

**Routes**:
```
POST /webhook        → Lambda Signal Validator
POST /webhook/mtf    → Lambda MTF Handler
GET  /health         → Lambda Health Check
```

### 2. Lambda Signal Validator

**Purpose**: Ultra-fast signal validation (<10ms target)

**Configuration**:
- Runtime: Python 3.11
- Memory: 1024MB (more memory = faster CPU)
- Timeout: 10 seconds
- Reserved Concurrency: 100

**Processing Flow**:
```
Input → Validate → Enrich → Store (DynamoDB) → Publish (EventBridge)
```

**Validation Rules**:
- Required: symbol, action, source, timeframe
- Actions: BUY, SELL, STRONG_BUY, STRONG_SELL, NEUTRAL, WAIT
- Timeframes: 1h, 4h, 1D, 1W, 15m, 30m
- Sources: luxalgo, technicals, momentum, volatility, ai, custom

### 3. EventBridge (Event Bus)

**Purpose**: Decouple signal ingestion from processing

**Event Pattern**:
```json
{
  "source": ["kyperian.signal.validator"],
  "detail-type": ["SignalValidated"]
}
```

**Targets**:
- ECS Fargate tasks via SQS
- CloudWatch Logs for audit

### 4. ECS Fargate (Processing Layer)

**Purpose**: Run the MTF fusion engine and decision making

**Configuration**:
- CPU: 512 vCPU
- Memory: 1024 MB
- Min Tasks: 2
- Max Tasks: 20

**Auto-Scaling Policies**:
- CPU > 70% → Scale out
- Memory > 80% → Scale out
- Requests > 1000/target → Scale out
- Cooldown: 60s out, 300s in

### 5. ElastiCache Redis

**Purpose**: Ultra-low latency signal caching

**Configuration**:
- Engine: Redis 7.0
- Node Type: cache.t4g.micro (production: r6g.large)
- Nodes: 2 (Multi-AZ)
- Encryption: At rest and in transit

**Cache Strategy**:
| Data Type | TTL | Key Pattern |
|-----------|-----|-------------|
| Weekly Signal | 7 days | kyperian:signal:SYMBOL:1W:* |
| Daily Signal | 24 hours | kyperian:signal:SYMBOL:1D:* |
| 4H Signal | 8 hours | kyperian:signal:SYMBOL:4h:* |
| Decision | 5 minutes | kyperian:decision:SYMBOL |

### 6. DynamoDB

**Purpose**: Persistent signal and decision storage

**Tables**:

**Signals Table**:
```
PK: SIGNAL#SYMBOL
SK: TIMEFRAME#TIMESTAMP
GSI: symbol-timestamp-index
TTL: 7 days
```

**Decisions Table**:
```
PK: DECISION#SYMBOL
SK: TIMESTAMP
TTL: 30 days
```

---

## 📊 Monitoring & Observability

### CloudWatch Dashboards

**Main Dashboard** displays:
- Signals processed per minute
- Lambda latency (avg, p99, max)
- Error rates by type
- ECS CPU/Memory utilization
- Redis cache hit rate

**Trading Dashboard** displays:
- Signals by action (pie chart)
- Signals by timeframe
- Veto rates
- Position size distribution

### Alarms

| Alarm | Threshold | Action |
|-------|-----------|--------|
| High Latency | p99 > 100ms | SNS Alert |
| High Errors | > 10/5min | SNS Alert |
| ECS Unhealthy | Tasks < 1 | SNS Alert |
| Redis Memory | > 90% | SNS Alert |

### Logging

All logs flow to CloudWatch Logs:
- `/aws/lambda/kyperian-*` - Lambda logs
- `/ecs/kyperian-*` - ECS logs
- `/aws/apigateway/kyperian-*` - API Gateway logs

---

## 💰 Cost Optimization

### Monthly Cost Breakdown

| Service | Configuration | Monthly Cost |
|---------|--------------|--------------|
| API Gateway | 1M requests | ~$10 |
| Lambda | 100K invocations | ~$5 |
| ECS Fargate | 2 tasks (24/7) | ~$70 |
| ElastiCache | t4g.micro x2 | ~$25 |
| DynamoDB | On-demand | ~$25 |
| NAT Gateway | Single AZ | ~$35 |
| CloudWatch | Logs + Metrics | ~$20 |
| WAF | Basic rules | ~$10 |
| **Total** | | **~$200** |

### Cost Savings Tips

1. **Use FARGATE_SPOT** for non-critical processing (70% savings)
2. **Reserved Capacity** for Redis (30% savings)
3. **S3 Intelligent-Tiering** for historical data
4. **Lambda Provisioned Concurrency** only for production

---

## 🔒 Security Architecture

### Authentication & Authorization

```
TradingView → API Gateway (API Key) → Lambda → EventBridge → ECS
                   ↓
              WAF (Rate Limit)
```

### Secrets Management

All secrets stored in AWS SSM Parameter Store:
- `/kyperian/redis/host`
- `/kyperian/redis/port`
- `/kyperian/api/key`
- `/kyperian/broker/credentials`

### Encryption

- **In Transit**: TLS 1.2+ everywhere
- **At Rest**: 
  - DynamoDB: AWS-managed keys
  - Redis: AWS-managed keys
  - S3: SSE-S3

---

## 🚦 Deployment Pipeline

### CI/CD Flow

```
GitHub Push → GitHub Actions → Build Image → Push to ECR
                   ↓
            Deploy CloudFormation → Update ECS Service
                   ↓
            Run Integration Tests → Monitor Dashboard
```

### Blue/Green Deployment

ECS uses rolling updates with:
- Minimum healthy: 100%
- Maximum: 200%
- Circuit breaker: Enabled
- Rollback: Automatic on failure

---

## 🔄 Disaster Recovery

### Backup Strategy

| Data | Backup Frequency | Retention |
|------|-----------------|-----------|
| DynamoDB | Continuous (PITR) | 35 days |
| Redis | Daily snapshots | 7 days |
| CloudFormation | Git versioned | Forever |

### Recovery Procedures

**Region Failure**:
1. Update Route 53 to DR region
2. Restore DynamoDB from backup
3. Deploy CloudFormation stacks
4. Verify data integrity

**RTO**: 1 hour
**RPO**: 15 minutes

---

## 📋 Operations Runbook

### Common Tasks

**Scale ECS Manually**:
```bash
aws ecs update-service \
  --cluster kyperian-production \
  --service kyperian-production-service \
  --desired-count 5
```

**Invalidate Cache**:
```bash
aws elasticache create-replication-group-message \
  --replication-group-id kyperian-production-redis \
  --message-type FLUSHDB
```

**Check Lambda Logs**:
```bash
aws logs tail /aws/lambda/kyperian-production-signal-validator --follow
```

### Incident Response

1. **High Latency**:
   - Check Lambda cold starts
   - Verify Redis connectivity
   - Review DynamoDB throttling

2. **High Error Rate**:
   - Check CloudWatch Logs
   - Verify signal format
   - Review Lambda errors

3. **ECS Tasks Failing**:
   - Check task definition
   - Verify container health
   - Review memory limits

---

## 🎯 Performance Tuning

### Lambda Optimization

- **Memory**: 1024MB (optimal CPU allocation)
- **Packaging**: Use Lambda layers for dependencies
- **Connection Reuse**: Keep connections alive

### Redis Optimization

- **Connection Pooling**: Max 20 connections
- **Pipeline**: Batch read operations
- **Eviction**: volatile-lru policy

### ECS Optimization

- **Task Placement**: spread across AZs
- **Health Check**: 30s interval, 5s timeout
- **Graceful Shutdown**: 30s deregistration delay

---

## 🔗 Integration Points

### TradingView Webhook

**Endpoint**: `https://api.kyperian.com/production/webhook`

**Headers**:
```
Content-Type: application/json
X-API-Key: your-api-key
```

**Payload**:
```json
{
  "symbol": "{{ticker}}",
  "action": "{{strategy.order.action}}",
  "source": "luxalgo",
  "timeframe": "{{interval}}",
  "price": {{close}},
  "confidence": 85,
  "indicators": {
    "rsi": {{rsi}},
    "macd": {{macd}}
  }
}
```

### Broker Integration

The ECS service exposes internal APIs for broker connectivity:
- Alpaca: `/broker/alpaca`
- Interactive Brokers: `/broker/ibkr`
- TDAmeritrade: `/broker/tda`

---

## 📈 Scaling Roadmap

### Phase 1 (Current)
- 2 ECS tasks
- 100 signals/minute
- ~$200/month

### Phase 2 (Growth)
- 5 ECS tasks
- 500 signals/minute
- r6g.large Redis
- ~$500/month

### Phase 3 (Enterprise)
- 20 ECS tasks
- 2000+ signals/minute
- Redis cluster mode
- Multi-region
- ~$2000/month

---

**🚀 KYPERIAN ELITE - Built for Performance, Designed for Scale**
