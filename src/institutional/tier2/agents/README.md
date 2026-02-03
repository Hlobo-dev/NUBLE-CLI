# Tier 2 Expert Agents

This package contains the modular expert agents for the Council-of-Experts trading system. Each agent is designed to be improved independently.

## Directory Structure

```
agents/
├── __init__.py           # Package exports
├── base.py               # BaseAgent, AgentContext, AgentOutput
├── registry.py           # Agent registry and factory
│
├── technical/            # Technical analysis agents
│   ├── mtf_dominance.py      # Multi-timeframe trend analysis
│   ├── trend_integrity.py    # Trend structure validation
│   ├── reversal_pullback.py  # Pullback vs reversal detection
│   └── volatility_state.py   # Volatility regime analysis
│
├── market/               # Market context agents
│   ├── regime_transition.py  # Market regime detection
│   └── event_window.py       # Event proximity analysis
│
├── risk/                 # Risk management agents
│   ├── risk_gatekeeper.py    # Hard risk constraints (VETO power)
│   ├── concentration.py      # Portfolio concentration
│   └── liquidity.py          # Execution feasibility
│
├── quality/              # Quality assurance agents
│   ├── data_integrity.py     # Data validation
│   └── timing.py             # Entry timing optimization
│
├── adversarial/          # Devil's advocate agents
│   └── red_team.py           # Thesis challenger
│
├── prompts/              # Prompt management system
│   └── __init__.py           # PromptManager, A/B testing
│
└── evals/                # Evaluation framework
    ├── __init__.py           # EvalRunner, EvalCase
    └── *_cases.json          # Per-agent test cases
```

## Agent Architecture

Each agent follows a consistent structure:

```python
# agents/technical/my_agent.py

PROMPT_VERSION = "1.0.0"  # Increment when changing prompts

SYSTEM_PROMPT = """..."""
LIGHT_ANALYSIS_PROMPT = """..."""
DEEP_ANALYSIS_PROMPT = """..."""

class MyAgent(BaseAgent):
    agent_id = "my_agent"
    agent_name = "My Agent"
    description = "What this agent does"
    expertise_tags = ["tag1", "tag2"]
    prompt_version = PROMPT_VERSION
    
    def get_system_prompt(self) -> str: ...
    def get_light_prompt(self, context) -> str: ...
    def get_deep_prompt(self, context) -> str: ...
    def calculate_relevance_score(self, context) -> float: ...
    def validate_output(self, output) -> bool: ...
    async def post_process(self, output) -> AgentOutput: ...

EVALUATION_CASES = [...]  # Test cases

def evaluate(agent, cases=None) -> dict: ...
```

## How to Improve an Agent

### 1. Understand the Agent's Purpose

Read the docstring and `description` field. Each agent has a specific responsibility:

| Agent | Purpose |
|-------|---------|
| `mtf_dominance` | Determines dominant trend across timeframes |
| `trend_integrity` | Validates trend structure quality |
| `reversal_pullback` | Distinguishes pullbacks from reversals |
| `volatility_state` | Assesses volatility regime |
| `regime_transition` | Detects market regime changes |
| `event_window` | Identifies event proximity risk |
| `risk_gatekeeper` | Enforces hard risk limits (VETO) |
| `concentration` | Monitors portfolio concentration |
| `liquidity` | Assesses execution feasibility |
| `data_integrity` | Validates data quality |
| `timing` | Optimizes entry timing |
| `red_team` | Challenges trading theses |

### 2. Review Current Performance

Run the evaluation suite to see baseline performance:

```python
from agents.technical.mtf_dominance import MTFDominanceAgent, EVALUATION_CASES, evaluate

agent = MTFDominanceAgent()
results = evaluate(agent, EVALUATION_CASES)
print(f"Accuracy: {results['correct_claims'] / results['total_cases']:.1%}")
```

### 3. Identify Improvement Areas

Look at:
- **Failed cases**: Which test cases fail and why?
- **Edge cases**: Are there scenarios the agent handles poorly?
- **Evidence quality**: Does the agent provide useful evidence?
- **Prompt clarity**: Is the prompt clear and specific?

### 4. Modify Prompts

The prompts are the main lever for improvement:

```python
# Before
SYSTEM_PROMPT = """You analyze trends."""

# After (more specific)
SYSTEM_PROMPT = """You are a Multi-Timeframe Trend expert. You:
1. IDENTIFY dominant trend across 1H, 4H, 1D timeframes
2. WEIGHT higher timeframes more heavily
3. DETECT alignment vs divergence
4. QUANTIFY trend strength and conviction

Your analysis prevents trading against the dominant trend."""
```

### 5. Add Domain Knowledge

Inject trading domain expertise into prompts:

```python
DEEP_ANALYSIS_PROMPT = """
Multi-Timeframe Analysis Framework:

WEIGHT HIERARCHY (importance):
- Weekly: 40% weight (macro trend)
- Daily: 35% weight (primary trend)  
- 4H: 15% weight (intermediate)
- 1H: 10% weight (short-term timing)

ALIGNMENT RULES:
- All aligned: High confidence signal
- Higher TF vs Lower: Trust higher
- Lower TF diverging: Early warning
...
"""
```

### 6. Increment Version

Always increment `PROMPT_VERSION` when making changes:

```python
# Old
PROMPT_VERSION = "1.0.0"

# New (after prompt changes)
PROMPT_VERSION = "1.1.0"
```

### 7. Add Test Cases

Add cases for any new scenarios you're handling:

```python
EVALUATION_CASES.append({
    "name": "new_edge_case",
    "context": {...},
    "expected_claim": "EXPECTED",
    "expected_evidence_keys": ["key1", "key2"],
    "tags": ["edge_case"]
})
```

### 8. Run Evaluation

Compare before/after performance:

```python
# Save old results
old_results = evaluate(agent_v1, cases)

# Test new version
new_results = evaluate(agent_v2, cases)

# Compare
print(f"Old: {old_results['accuracy']:.1%}")
print(f"New: {new_results['accuracy']:.1%}")
```

## A/B Testing

For production comparison:

```python
from agents.prompts import prompt_manager

# Create A/B test
ab_test = prompt_manager.create_ab_test(
    agent_id="mtf_dominance",
    control_version="v1.0",
    treatment_version="v1.1",
    traffic_split=0.5  # 50% see new version
)

# Get prompt for user (deterministic by user_id)
prompt = prompt_manager.get_prompt_ab("mtf_dominance", "system", user_id)
```

## Evaluation Framework

### Loading Test Cases

```python
from agents.evals import load_eval_cases, EvalRunner

cases = load_eval_cases("mtf_dominance")
runner = EvalRunner(timeout_ms=30000)
```

### Running Evaluation

```python
import asyncio

async def run_eval():
    results = await runner.run(agent, cases)
    report = runner.generate_report(agent, results, cases)
    print(report.to_markdown())

asyncio.run(run_eval())
```

### Report Format

```markdown
# Evaluation Report: mtf_dominance

**Prompt Version:** 1.1.0
**Timestamp:** 2024-01-15T10:30:00

## Summary

| Metric | Value |
|--------|-------|
| Total Cases | 8 |
| Passed | 7 |
| Failed | 1 |
| Accuracy | 87.5% |
| Avg Latency | 245.3ms |

## Accuracy by Tag

| Tag | Accuracy |
|-----|----------|
| happy_path | 100% |
| edge_case | 66.7% |
```

## Best Practices

### Prompt Engineering

1. **Be Specific**: Define exactly what the agent should analyze
2. **Provide Structure**: Use numbered steps, tables, JSON schemas
3. **Include Examples**: Show expected output format
4. **Define Edge Cases**: Tell agent how to handle ambiguity
5. **Set Priorities**: What's most important in the analysis?

### Evidence Keys

Define consistent evidence keys so downstream systems can rely on them:

```python
REQUIRED_EVIDENCE_KEYS = [
    "timeframe_analysis",  # Always include
    "alignment_score",     # 0-1 float
    "dominant_trend",      # up/down/neutral
    "confidence_factors",  # List of reasons
]
```

### Relevance Scoring

Tune `calculate_relevance_score()` to control when agent is activated:

```python
def calculate_relevance_score(self, context):
    score = 0.5  # Base
    
    # Higher for momentum trades
    if context.trade_context.get("trade_type") == "momentum":
        score += 0.3
    
    # Lower in low volatility
    if context.market_data.get("volatility") < 0.10:
        score -= 0.2
    
    return max(0.0, min(1.0, score))
```

### Validation

Add custom validation in `validate_output()`:

```python
def validate_output(self, output):
    if not super().validate_output(output):
        return False
    
    # Custom checks
    if output.claim == "BULLISH":
        if "alignment_score" not in output.evidence:
            return False
        if output.evidence["alignment_score"] < 0.6:
            return False  # Can't be bullish with low alignment
    
    return True
```

## Adding a New Agent

1. Create file in appropriate category directory
2. Follow the template structure
3. Register in category's `__init__.py`
4. Add to main `agents/__init__.py`
5. Create evaluation cases in `evals/`
6. Write unit tests

```python
# agents/technical/new_agent.py
from ..base import BaseAgent, AgentContext, AgentOutput

PROMPT_VERSION = "1.0.0"

class NewAgent(BaseAgent):
    agent_id = "new_agent"
    # ... implement required methods
```

## Agent Categories

### Technical Agents
Focus on chart/price analysis. Experts in:
- Trend analysis
- Pattern recognition
- Technical indicators
- Price action

### Market Agents
Focus on market context. Experts in:
- Regime detection
- Event impact
- Correlation analysis
- Sector rotation

### Risk Agents
Focus on risk management. Experts in:
- Position sizing
- Portfolio limits
- Liquidity constraints
- Concentration risk

### Quality Agents
Focus on data/execution quality. Experts in:
- Data validation
- Timing optimization
- Execution quality
- Error detection

### Adversarial Agents
Focus on challenging assumptions. Experts in:
- Counter-arguments
- Bias detection
- Stress testing
- Edge case analysis
