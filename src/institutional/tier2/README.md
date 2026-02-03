# NUBLE Tier 2: Council-of-Experts Orchestrator (CEO)

## Governed Decision Quality Layer

Tier 2 is a secondary decision layer that **does not replace Tier 1**. It only runs on escalation and produces **auditable deltas** to Tier 1's decision.

## Mission

> Tier 2's job is to catch the 5-10% of decisions where Tier 1's confidence is misplaced or where additional scrutiny reveals hidden risks.

### Delta Types

| Delta | Description | When Applied |
|-------|-------------|--------------|
| `NO_DELTA` | Pass through Tier 1 unchanged | No significant findings |
| `WAIT` | Delay execution | Timing concerns, event proximity |
| `NO_TRADE` | Cancel the trade | Severe risk, data quality issues |
| `CONFIDENCE_DOWN` | Reduce position sizing | Signal conflicts, elevated volatility |
| `CONFIDENCE_UP` | Increase position sizing | Rare: unanimous support + risk OK |
| `POSITION_CAP` | Reduce maximum position | Concentration limits |

---

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
│                  ┌────────────────────────────────────┤              │
│                  │                                    │              │
│  ┌───────────────▼──────────────┐    ┌───────────────▼──────────┐  │
│  │ LAYER A: ALLOCATOR           │    │ LAYER B: LIGHT ROUND     │  │
│  │ - Token budgets              │    │ - 150-350 tokens each    │  │
│  │ - Agent selection            │    │ - JSON claims only       │  │
│  │ - Timeout management         │    │ - 10-12 MVP agents       │  │
│  └───────────────┬──────────────┘    └───────────────┬──────────┘  │
│                  │                                    │              │
│  ┌───────────────▼──────────────┐    ┌───────────────▼──────────┐  │
│  │ LAYER C: DEEP ROUND          │    │ LAYER D: CROSS-EXAM     │  │
│  │ - 1000-2500 tokens           │    │ (Optional, future)       │  │
│  │ - 3-8 activated agents       │    │ - Debate resolution      │  │
│  └───────────────┬──────────────┘    └───────────────┬──────────┘  │
│                  │                                    │              │
│                  └────────────────┬───────────────────┘              │
│                                   ▼                                  │
│                  ┌────────────────────────────────────┐             │
│                  │ LAYER E: ARBITER + SYNTHESIS       │             │
│                  │ ┌──────────────┐  ┌─────────────┐  │             │
│                  │ │ Claim Graph  │  │ Risk Gate   │  │             │
│                  │ │ (aggregate)  │  │ (VETO)      │  │             │
│                  │ └──────────────┘  └─────────────┘  │             │
│                  └───────────────┬────────────────────┘             │
│                                  ▼                                   │
│                  ┌────────────────────────────────────┐             │
│                  │ TIER 1 + DELTAS = FINAL DECISION   │             │
│                  └────────────────────────────────────┘             │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Non-Negotiable Principles

### 1. Evidence-Bounded
Every agent claim **must reference specific evidence keys** from the input data. Claims without evidence are discarded.

```json
{
  "claim": "Signal conflict detected",
  "evidence_keys": ["weekly_signal", "daily_signal"],
  "stance": "anti",
  "strength": 0.85
}
```

### 2. Budgeted Depth
- **Light Round**: 150-350 tokens per agent, 3s timeout
- **Deep Round**: 1000-2500 tokens per agent, 8s timeout
- Not every agent runs deep—only those activated by escalation reason

### 3. Risk as Gate, Not Vote
The Risk Gatekeeper agent can **veto** any trade, but cannot:
- Override hard risk constraints from Tier 1
- Increase position sizes beyond limits
- Force a trade that Tier 1 rejected

### 4. Deterministic Fallback
If Tier 2 fails (timeout, error, circuit breaker), it returns `NO_DELTA`. **Tier 2 can never block trading**.

### 5. Staged Calibration
Static weights → learned weights over time. Every decision is stored with outcomes for future calibration.

---

## MVP Council: 12 Expert Agents

| Agent | Domain | Purpose |
|-------|--------|---------|
| **MTF Dominance** | Technical | Multi-timeframe trend alignment |
| **Trend Integrity** | Technical | Structure, support/resistance, HL/LH patterns |
| **Reversal vs Pullback** | Technical | Distinguish corrections from reversals |
| **Volatility State** | Technical | ATR regime, VIX state, compression/expansion |
| **Regime Transition** | Market | HMM-based regime detection |
| **Event Window** | Market | Earnings, Fed, macro event timing |
| **Red Team** | Challenge | Devil's advocate, find counter-evidence |
| **Risk Gatekeeper** | Risk | Hard veto power, concentration checks |
| **Data Integrity** | Quality | Stale data, missing feeds, anomalies |
| **Timing** | Execution | Entry timing, session, gap risk |
| **Liquidity** | Execution | Volume, spread, slippage risk |
| **Concentration** | Portfolio | Sector/position concentration |

---

## Escalation Triggers

Tier 2 only runs when Tier 1 decisions meet certain criteria:

| Trigger | Threshold | Deep Agents |
|---------|-----------|-------------|
| Low Confidence | < 60% | MTF, Trend, Timing, Data |
| High Confidence | > 90% | Red Team, Risk, Regime |
| Signal Conflict | Weekly ≠ Daily | MTF, Trend, Regime, Red Team |
| High Volatility | VIX > 25 | Volatility, Regime, Timing, Risk |
| Regime Transition | Confidence < 60% | Regime, Volatility, Trend, Red Team |
| Earnings Imminent | Within 48h | Event, Timing, Volatility, Risk |
| Stale Data | Age > 5 min | Data, Timing |
| Portfolio Concentration | > 8% position | Concentration, Risk |

---

## Usage

### Direct Orchestrator

```python
from src.institutional.tier2 import (
    Tier2Orchestrator,
    Tier1DecisionPack,
    EscalationReason,
)

# Create orchestrator
orchestrator = Tier2Orchestrator()

# Build Tier 1 decision pack
tier1_pack = Tier1DecisionPack(
    symbol="AAPL",
    action="BUY",
    confidence=75.0,
    direction="BULLISH",
    price=185.50,
    rsi=58.2,
    macd_value=0.52,
    macd_signal=0.45,
    # ... more fields
)

# Check if escalation needed
escalation = orchestrator.should_escalate(tier1_pack)

if escalation.should_escalate:
    # Run Tier 2
    decision = orchestrator.run(
        tier1_pack=tier1_pack,
        escalation_reasons=escalation.reasons,
    )
    
    print(f"Delta: {decision.delta_type}")
    print(f"Confidence adjustment: {decision.delta_confidence:+.1f}%")
    print(f"Final confidence: {decision.final_confidence:.1f}%")
    print(f"Rationale: {decision.rationale}")
```

### Lambda Invocation

```python
import boto3
import json

lambda_client = boto3.client('lambda')

response = lambda_client.invoke(
    FunctionName='nuble-tier2-council-production',
    InvocationType='RequestResponse',
    Payload=json.dumps({
        'tier1_decision': {
            'symbol': 'AAPL',
            'action': 'BUY',
            'confidence': 75.0,
            # ... more fields
        },
        'escalation_reasons': ['signal_conflict'],
    }),
)

result = json.loads(response['Payload'].read())
```

---

## DynamoDB Schema

### Decisions Table
```
PK: DECISION#{symbol}
SK: {timestamp_iso}
GSI: decision_id-index

Attributes:
- decision_id
- tier1_action, tier1_confidence
- tier2_delta (JSON)
- final_action, final_confidence
- escalation_reasons (StringSet)
- agent_count, latency_ms
- ttl (90 days)
```

### Agent Runs Table
```
PK: RUN#{decision_id}
SK: AGENT#{name}#{round}

Attributes:
- claims (JSON)
- input_tokens, output_tokens
- latency_ms
- raw_output (optional, TTL 7 days)
```

### Outcomes Table (for calibration)
```
PK: OUTCOME#{symbol}
SK: {timestamp_iso}
GSI: decision_id-index

Attributes:
- entry_price, exit_price
- pnl_pct, holding_hours
- outcome_label (win/loss/scratch)
- tier1_confidence, tier2_confidence
```

---

## Circuit Breaker

Tier 2 has a circuit breaker that protects the system:

| State | Behavior | Trigger |
|-------|----------|---------|
| **Closed** | Normal operation | Default |
| **Open** | Returns NO_DELTA immediately | Error rate > 30% |
| **Half-Open** | Allows one probe request | After cooldown |

```python
# Check circuit breaker status
metrics = orchestrator.get_metrics()
print(f"Circuit Breaker: {metrics['circuit_breaker']['state']}")
```

---

## Deployment

### Prerequisites
- AWS account with Bedrock access
- Python 3.11+
- boto3

### CloudFormation
```bash
aws cloudformation deploy \
  --template-file infrastructure/aws/cloudformation/tier2-stack.yaml \
  --stack-name nuble-tier2-production \
  --parameter-overrides Environment=production \
  --capabilities CAPABILITY_NAMED_IAM
```

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| `TIER2_ENABLED` | true | Enable/disable Tier 2 |
| `USE_DYNAMODB` | true | Use DynamoDB (false = in-memory) |
| `BEDROCK_MODEL_ID` | claude-3-haiku | Model for agent inference |
| `LIGHT_ROUND_TIMEOUT_MS` | 3000 | Light round timeout |
| `DEEP_ROUND_TIMEOUT_MS` | 8000 | Deep round timeout |
| `LOG_LEVEL` | INFO | Logging verbosity |

---

## Testing

```bash
# Run Tier 2 tests
pytest tests/test_tier2_council.py -v

# Run with coverage
pytest tests/test_tier2_council.py --cov=src.institutional.tier2
```

---

## Roadmap

- [ ] **Cross-Exam Layer**: Debate resolution between conflicting agents
- [ ] **Calibration Pipeline**: Automated weight adjustment from outcomes
- [ ] **Agent Marketplace**: Plugin architecture for custom agents
- [ ] **Explainability Dashboard**: Visual claims graph
- [ ] **A/B Testing**: Compare agent configurations

---

## License

NUBLE Institutional - Proprietary

---

*Built for institutional-grade decision governance.*
