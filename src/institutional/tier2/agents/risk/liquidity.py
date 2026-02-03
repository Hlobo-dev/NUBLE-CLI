"""
Liquidity Agent
===============

Expert at assessing market liquidity and execution feasibility.
Evaluates whether positions can be entered/exited at reasonable cost.

Improvement Guidelines:
-----------------------
1. PROMPT_VERSION: Increment when modifying prompts
2. evaluate(): Run against EVALUATION_CASES to measure performance
3. Add domain knowledge about:
   - Bid-ask spread analysis
   - Volume profiles and VWAP
   - Market impact estimation
   - Optimal execution timing
   - Dark pool availability
"""

from typing import Any

from ..base import BaseAgent, AgentContext, AgentOutput

# Increment this when modifying prompts to track performance changes
PROMPT_VERSION = "1.0.0"

# =============================================================================
# PROMPTS - Edit these to improve agent performance
# =============================================================================

SYSTEM_PROMPT = """You are a Liquidity and Execution expert in a trading council. Your role is to:

1. ASSESS market liquidity for proposed trades
2. ESTIMATE market impact and execution costs
3. ANALYZE volume patterns and optimal execution windows
4. RECOMMEND execution strategies (TWAP, VWAP, arrival price)
5. FLAG liquidity concerns that could trap capital

Liquidity Metrics:
- Bid-Ask Spread: Transaction cost indicator
- Average Daily Volume (ADV): Size capacity
- Market Depth: Order book resilience
- Participation Rate: % of volume we'd represent
- Slippage Estimate: Expected vs. executed price difference

Output Format (JSON only):
{
    "claim": "LIQUID|ILLIQUID|CAUTION",
    "confidence": 0.0-1.0,
    "evidence": {
        "current_liquidity": {
            "avg_daily_volume": "shares/day",
            "avg_spread_bps": "basis points",
            "market_depth": "quality assessment",
            "typical_slippage_bps": "expected slippage"
        },
        "trade_analysis": {
            "shares_to_trade": "proposed quantity",
            "days_to_execute": "at target participation rate",
            "estimated_impact_bps": "market impact cost",
            "participation_rate": "% of ADV"
        }
    },
    "recommended_delta": {
        "execution_strategy": "TWAP|VWAP|LIMIT|MARKET",
        "optimal_timing": "market open/close/midday",
        "max_position_size": "liquidity-constrained max",
        "execution_horizon": "recommended days"
    }
}"""

LIGHT_ANALYSIS_PROMPT = """Quick liquidity check:

Trade Details:
{context}

Recent Volume Data:
{volume_data}

Quick Assessment:
1. What's the average daily volume?
2. How does our order size compare to ADV?
3. Are spreads normal or elevated?

Provide rapid liquidity assessment in JSON format."""

DEEP_ANALYSIS_PROMPT = """Comprehensive liquidity analysis:

Trade Proposal:
{proposal}

Detailed Volume Profile:
{volume_profile}

Order Book Data:
{order_book}

Historical Execution Data:
{execution_history}

Deep Liquidity Analysis Required:
1. VOLUME ANALYSIS
   - 20-day ADV and trend
   - Intraday volume distribution
   - Unusual volume patterns
   - Volume relative to float

2. SPREAD ANALYSIS
   - Current bid-ask spread
   - Historical spread distribution
   - Time-of-day spread patterns
   - Spread during volatility

3. MARKET IMPACT MODEL
   - Estimated permanent impact
   - Temporary impact decay
   - Total execution cost
   - Comparison to historical fills

4. EXECUTION STRATEGY
   - Optimal participation rate
   - TWAP vs VWAP recommendation
   - Timing recommendations
   - Dark pool availability

5. RISK ASSESSMENT
   - Exit liquidity (can we get out?)
   - Gap risk in illiquid markets
   - Event-driven liquidity changes
   - Cross-market arbitrage effects

6. RECOMMENDATIONS
   - Maximum advisable size
   - Execution horizon
   - Order type suggestions
   - Risk-adjusted cost analysis

Provide detailed liquidity assessment in JSON format."""


# =============================================================================
# AGENT IMPLEMENTATION
# =============================================================================

class LiquidityAgent(BaseAgent):
    """
    Assesses market liquidity and execution feasibility.
    
    This agent evaluates whether proposed trades can be executed
    at reasonable cost and without excessive market impact. It
    provides execution strategy recommendations.
    
    Metrics Tracked:
    - Predicted vs actual slippage
    - Execution cost accuracy
    - Liquidity warnings issued
    - Position size recommendations
    """
    
    agent_id = "liquidity"
    agent_name = "Liquidity"
    description = "Assesses market liquidity and execution feasibility"
    expertise_tags = ["liquidity", "execution", "volume", "spread", "market_impact"]
    prompt_version = PROMPT_VERSION
    
    # Complexity thresholds for this agent
    light_complexity_max = 0.5
    deep_complexity_min = 0.4
    
    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT
    
    def get_light_prompt(self, context: AgentContext) -> str:
        volume_data = context.market_data.get("volume_data", {
            "adv_20d": 0,
            "recent_volume": 0,
            "avg_spread_bps": 0
        })
        return LIGHT_ANALYSIS_PROMPT.format(
            context=context.trade_context,
            volume_data=volume_data
        )
    
    def get_deep_prompt(self, context: AgentContext) -> str:
        volume_profile = context.market_data.get("volume_profile", {})
        order_book = context.market_data.get("order_book", {"bids": [], "asks": []})
        execution_history = context.market_data.get("execution_history", [])
        
        return DEEP_ANALYSIS_PROMPT.format(
            proposal=context.trade_context,
            volume_profile=volume_profile,
            order_book=order_book,
            execution_history=execution_history
        )
    
    def calculate_relevance_score(self, context: AgentContext) -> float:
        """Higher relevance for larger trades or illiquid names."""
        base_score = 0.5
        
        # Higher relevance for larger position sizes
        position_value = context.trade_context.get("position_value", 0)
        if position_value > 100000:  # >$100k trades
            base_score = min(1.0, base_score + 0.2)
        if position_value > 500000:  # >$500k trades
            base_score = min(1.0, base_score + 0.2)
        
        # Higher relevance for lower volume stocks
        adv = context.market_data.get("volume_data", {}).get("adv_20d", 1000000)
        shares = context.trade_context.get("shares", 0)
        participation_rate = shares / adv if adv > 0 else 1.0
        
        if participation_rate > 0.01:  # >1% of ADV
            base_score = min(1.0, base_score + 0.2)
        if participation_rate > 0.05:  # >5% of ADV
            base_score = min(1.0, base_score + 0.2)
        
        # Higher relevance for wide spread stocks
        spread_bps = context.market_data.get("volume_data", {}).get("avg_spread_bps", 5)
        if spread_bps > 20:
            base_score = min(1.0, base_score + 0.15)
        
        return base_score
    
    def validate_output(self, output: AgentOutput) -> bool:
        """Ensure liquidity-specific output requirements are met."""
        if not super().validate_output(output):
            return False
        
        # Must include liquidity metrics
        evidence = output.evidence
        if "current_liquidity" not in evidence and "trade_analysis" not in evidence:
            return False
        
        return True
    
    async def post_process(self, output: AgentOutput) -> AgentOutput:
        """Add liquidity-specific calculations and flags."""
        output = await super().post_process(output)
        
        # Add execution cost estimate if not present
        trade_analysis = output.evidence.get("trade_analysis", {})
        if trade_analysis:
            impact_bps = trade_analysis.get("estimated_impact_bps", 0)
            spread_bps = output.evidence.get("current_liquidity", {}).get("avg_spread_bps", 0)
            output.evidence["total_cost_bps"] = impact_bps + spread_bps / 2
        
        return output


# =============================================================================
# EVALUATION - Test cases for measuring agent performance
# =============================================================================

EVALUATION_CASES = [
    {
        "name": "highly_liquid",
        "context": {
            "symbol": "SPY",
            "proposed_action": "buy",
            "shares": 1000,
            "position_value": 50000
        },
        "market_data": {
            "volume_data": {
                "adv_20d": 80000000,
                "avg_spread_bps": 1,
                "market_depth": "excellent"
            }
        },
        "portfolio_state": {},
        "expected_claim": "LIQUID",
        "expected_evidence_keys": ["current_liquidity", "trade_analysis"]
    },
    {
        "name": "illiquid_small_cap",
        "context": {
            "symbol": "SMCP",
            "proposed_action": "buy",
            "shares": 50000,
            "position_value": 250000
        },
        "market_data": {
            "volume_data": {
                "adv_20d": 100000,
                "avg_spread_bps": 75,
                "market_depth": "thin"
            }
        },
        "portfolio_state": {},
        "expected_claim": "ILLIQUID",
        "expected_evidence_keys": ["current_liquidity", "estimated_impact_bps"]
    },
    {
        "name": "caution_moderate_impact",
        "context": {
            "symbol": "AMD",
            "proposed_action": "buy",
            "shares": 25000,
            "position_value": 400000
        },
        "market_data": {
            "volume_data": {
                "adv_20d": 5000000,
                "avg_spread_bps": 5,
                "market_depth": "good"
            }
        },
        "portfolio_state": {},
        "expected_claim": "CAUTION",
        "expected_evidence_keys": ["current_liquidity", "execution_strategy"]
    }
]


def evaluate(agent: LiquidityAgent, cases: list = None) -> dict:
    """
    Evaluate agent performance against test cases.
    
    Returns:
        dict: Performance metrics including accuracy, cost estimation, etc.
    """
    cases = cases or EVALUATION_CASES
    results = {
        "total_cases": len(cases),
        "correct_claims": 0,
        "impact_estimation_accuracy": 0,
        "evidence_completeness": 0,
        "prompt_version": agent.prompt_version,
        "details": []
    }
    
    for case in cases:
        context = AgentContext(
            trade_context=case["context"],
            portfolio_state=case.get("portfolio_state", {}),
            market_data=case["market_data"],
            complexity_score=0.5,
            time_budget_ms=5000
        )
        
        case_result = {
            "name": case["name"],
            "expected": case["expected_claim"],
            "passed": False
        }
        
        results["details"].append(case_result)
    
    return results
