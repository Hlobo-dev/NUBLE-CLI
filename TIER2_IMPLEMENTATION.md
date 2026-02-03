# Tier 2 Council-of-Experts Implementation Summary

## Overview

The Tier 2 Council-of-Experts (CEO) is now implemented as a **governed decision quality layer** that produces auditable deltas to Tier 1 decisions. It follows all the principles from the institutional blueprint.

## Files Created

### Core Tier 2 Package (`src/institutional/tier2/`)

| File | Purpose | Lines |
|------|---------|-------|
| `__init__.py` | Package exports and documentation | ~145 |
| `config.py` | Configuration, agent configs, escalation mapping | ~356 |
| `schemas.py` | Data schemas (Claim, AgentOutput, Tier2Decision, etc.) | ~640 |
| `agents.py` | 12 expert agent implementations | ~924 |
| `registry.py` | Agent lifecycle management | ~145 |
| `runtime.py` | Bedrock LLM execution with budgets | ~450 |
| `allocator.py` | Meta-orchestrator for agent selection | ~250 |
| `arbiter.py` | Synthesis layer with claim aggregation | ~433 |
| `circuit_breaker.py` | Safety mechanism | ~200 |
| `store.py` | DynamoDB storage for decisions/outcomes | ~320 |
| `escalation.py` | Escalation detection logic | ~200 |
| `orchestrator.py` | Main Tier2Orchestrator class | ~600 |
| `README.md` | Comprehensive documentation | ~350 |

### Infrastructure (`infrastructure/aws/`)

| File | Purpose |
|------|---------|
| `lambda/tier2_council/handler.py` | Lambda handler for Tier 2 API |
| `lambda/decision_engine/tier2_integration.py` | Integration with V6 APEX handler |
| `cloudformation/tier2-stack.yaml` | CloudFormation for DynamoDB + Lambda + API Gateway |

### Tests (`tests/`)

| File | Purpose |
|------|---------|
| `test_tier2_council.py` | Comprehensive unit tests |

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TIER 2 PIPELINE                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌───────────────┐   ┌────────────────┐   ┌────────────────────┐   │
│  │ TIER 1        │──►│ ESCALATION     │──►│ TIER 2 ORCHESTRATOR│   │
│  │ DECISION      │   │ DETECTOR       │   │ (CEO)              │   │
│  └───────────────┘   └────────────────┘   └─────────┬──────────┘   │
│                                                       │              │
│  ┌───────────────────────────────────────────────────┤              │
│  │                                                    │              │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  │ ALLOCATOR    │  │ LIGHT ROUND  │  │ ARBITER + SYNTHESIS  │   │
│  │  │ (Layer A)    │  │ (Layer B)    │  │ (Layer E)            │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│  │                                                                   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐   │
│  │  │ DEEP ROUND   │  │ CROSS-EXAM   │  │ CIRCUIT BREAKER      │   │
│  │  │ (Layer C)    │  │ (Layer D)    │  │ (Safety)             │   │
│  │  └──────────────┘  └──────────────┘  └──────────────────────┘   │
│  │                                                                   │
│  └───────────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────────┘
```

## The 12 Expert Agents (MVP Council)

| Agent | Domain | Key Responsibility |
|-------|--------|-------------------|
| MTF Dominance | Technical | Multi-timeframe trend alignment |
| Trend Integrity | Technical | Structure, support/resistance |
| Reversal vs Pullback | Technical | Distinguish corrections from reversals |
| Volatility State | Technical | ATR regime, VIX state |
| Regime Transition | Market | HMM-based regime detection |
| Event Window | Market | Earnings, Fed, macro timing |
| Red Team | Challenge | Devil's advocate |
| Risk Gatekeeper | Risk | VETO power |
| Data Integrity | Quality | Stale data, anomalies |
| Timing | Execution | Entry timing, session |
| Liquidity | Execution | Volume, spread, slippage |
| Concentration | Portfolio | Sector/position limits |

## Delta Types

| Type | Effect | When Used |
|------|--------|-----------|
| `NO_DELTA` | Pass through unchanged | No findings |
| `WAIT` | Delay execution | Timing concerns |
| `NO_TRADE` | Cancel trade | Severe risk |
| `CONFIDENCE_DOWN` | Reduce sizing | Signal conflicts |
| `CONFIDENCE_UP` | Increase sizing | Unanimous support |
| `POSITION_CAP` | Reduce max position | Concentration |

## Non-Negotiable Principles (Implemented)

✅ **Evidence-Bounded**: Every claim references evidence keys  
✅ **Budgeted Depth**: Token/time limits per layer  
✅ **Risk as Gate**: RiskGatekeeper has VETO power  
✅ **Deterministic Fallback**: On failure → NO_DELTA  
✅ **Self-Calibration**: Decision storage for future learning  

## Usage

### Direct Usage

```python
from src.institutional.tier2 import Tier2Orchestrator, Tier1DecisionPack

orchestrator = Tier2Orchestrator()

tier1_pack = Tier1DecisionPack(
    symbol="AAPL",
    action="BUY",
    confidence=75.0,
    # ... more fields
)

escalation = orchestrator.should_escalate(tier1_pack)
if escalation.should_escalate:
    decision = orchestrator.run(tier1_pack, escalation.reasons)
    print(f"Delta: {decision.delta_type}, Conf: {decision.delta_confidence:+.1f}%")
```

### V6 APEX Integration

```python
# In handler_v6_apex.py
from tier2_integration import apply_tier2_if_needed

decision = make_decision(symbol)
decision = apply_tier2_if_needed(decision)  # Applies Tier 2 if escalated
```

### Lambda API

```bash
# POST /evaluate
curl -X POST https://api.../evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "tier1_decision": {...},
    "escalation_reasons": ["signal_conflict"]
  }'
```

## Deployment

```bash
# Deploy CloudFormation stack
aws cloudformation deploy \
  --template-file infrastructure/aws/cloudformation/tier2-stack.yaml \
  --stack-name nuble-tier2-production \
  --parameter-overrides Environment=production \
  --capabilities CAPABILITY_NAMED_IAM
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TIER2_ENABLED` | false | Enable Tier 2 |
| `TIER2_USE_DYNAMODB` | false | Use DynamoDB storage |
| `BEDROCK_MODEL_ID` | claude-3-haiku | Model for agents |
| `LIGHT_ROUND_TIMEOUT_MS` | 3000 | Light round timeout |
| `DEEP_ROUND_TIMEOUT_MS` | 8000 | Deep round timeout |

## Tests

```bash
# Run tests
pytest tests/test_tier2_council.py -v

# With coverage
pytest tests/test_tier2_council.py --cov=src.institutional.tier2
```

## Next Steps

1. **Enable in Production**: Set `TIER2_ENABLED=true`
2. **Deploy CloudFormation**: Create DynamoDB tables and Lambda
3. **Monitor**: Watch circuit breaker state and latency
4. **Calibrate**: Collect outcomes and adjust agent weights
5. **Extend**: Add cross-exam layer and more agents

---

*Tier 2 Council-of-Experts - Institutional-grade decision governance.*
