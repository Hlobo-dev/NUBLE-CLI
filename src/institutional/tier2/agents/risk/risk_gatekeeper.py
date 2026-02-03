"""
Risk Gatekeeper Agent
=====================

Expert at enforcing hard risk constraints and portfolio-level veto decisions.
This agent has the power to block trades that violate critical risk thresholds.

Improvement Guidelines:
-----------------------
1. PROMPT_VERSION: Increment when modifying prompts
2. evaluate(): Run against EVALUATION_CASES to measure performance
3. Add domain knowledge about:
   - Maximum drawdown limits
   - Portfolio VaR constraints
   - Correlation risk
   - Tail risk scenarios
   - Stop-loss enforcement
"""

from typing import Any

from ..base import BaseAgent, AgentContext, AgentOutput

# Increment this when modifying prompts to track performance changes
PROMPT_VERSION = "1.0.0"

# =============================================================================
# PROMPTS - Edit these to improve agent performance
# =============================================================================

SYSTEM_PROMPT = """You are a Risk Gatekeeper expert in a trading council. Your role is to:

1. ENFORCE hard risk constraints that cannot be overridden
2. CALCULATE portfolio-level risk metrics (VaR, expected shortfall)
3. DETECT correlation breakdowns and contagion risk
4. VETO trades that would breach risk limits
5. PROTECT capital through strict discipline

Risk Categories:
- Position Risk: Single position max loss
- Portfolio Risk: Total portfolio drawdown limits
- Correlation Risk: Over-exposure to correlated assets
- Tail Risk: Fat-tail event exposure
- Liquidity Risk: Ability to exit positions

Output Format (JSON only):
{
    "claim": "VETO|APPROVE|CONDITIONAL",
    "confidence": 0.0-1.0,
    "evidence": {
        "current_portfolio_risk": "risk metrics",
        "proposed_trade_impact": "impact analysis",
        "constraint_violations": ["list of violations"],
        "risk_adjusted_metrics": "sharpe, sortino, etc."
    },
    "recommended_delta": {
        "position_size": "max allowed size",
        "stop_loss": "required stop level",
        "conditions": ["required conditions for approval"]
    },
    "veto_reason": "null or reason for veto"
}

CRITICAL: This agent has VETO POWER. If any hard constraint is violated, 
return claim="VETO" regardless of other factors."""

LIGHT_ANALYSIS_PROMPT = """Perform a quick risk check:

Trade Context:
{context}

Portfolio State:
{portfolio_state}

Quick Risk Assessment:
1. Does this trade violate any hard position limits?
2. Would portfolio drawdown exceed maximum allowed?
3. Are there obvious concentration concerns?
4. Is there sufficient margin/capital?

Provide rapid risk assessment in JSON format."""

DEEP_ANALYSIS_PROMPT = """Perform comprehensive risk analysis:

Trade Proposal:
{proposal}

Full Portfolio State:
{portfolio_state}

Market Risk Environment:
{risk_environment}

Deep Risk Analysis Required:
1. POSITION RISK
   - Maximum loss on this trade
   - Gap risk (overnight, weekend)
   - Scenario analysis (2σ, 3σ moves)

2. PORTFOLIO IMPACT
   - Marginal VaR contribution
   - Current vs. proposed drawdown capacity
   - Capital utilization change

3. CORRELATION ANALYSIS
   - Beta to existing positions
   - Sector/factor concentration
   - Tail dependency

4. STRESS TESTING
   - Historical scenario replay (2008, 2020, etc.)
   - Hypothetical stress scenarios
   - Recovery capacity

5. CONSTRAINT CHECK
   - Hard limits: {hard_limits}
   - Soft limits: {soft_limits}
   - Breach severity

6. RISK-ADJUSTED RETURN
   - Expected Sharpe contribution
   - Risk budget consumption
   - Opportunity cost

Provide thorough risk assessment in JSON format. 
VETO if any hard constraint would be violated."""


# =============================================================================
# AGENT IMPLEMENTATION
# =============================================================================

class RiskGatekeeperAgent(BaseAgent):
    """
    Enforces hard risk constraints and has veto power over trades.
    
    This agent is the final check before any trade execution. It can
    block trades that would violate critical risk limits, regardless
    of the signal strength or other agent recommendations.
    
    Metrics Tracked:
    - Vetoes issued
    - Constraint violations caught
    - False positive rate (vetoes that would have been profitable)
    - Portfolio risk before/after decisions
    """
    
    agent_id = "risk_gatekeeper"
    agent_name = "Risk Gatekeeper"
    description = "Enforces hard risk constraints and portfolio-level veto decisions"
    expertise_tags = ["risk", "veto", "constraints", "var", "drawdown", "portfolio"]
    prompt_version = PROMPT_VERSION
    
    # Complexity thresholds for this agent
    light_complexity_max = 0.4
    deep_complexity_min = 0.3
    
    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT
    
    def get_light_prompt(self, context: AgentContext) -> str:
        return LIGHT_ANALYSIS_PROMPT.format(
            context=context.trade_context,
            portfolio_state=context.portfolio_state
        )
    
    def get_deep_prompt(self, context: AgentContext) -> str:
        risk_env = context.market_data.get("risk_environment", {})
        hard_limits = context.portfolio_state.get("hard_limits", {
            "max_position_pct": 0.05,
            "max_portfolio_drawdown": 0.10,
            "max_sector_concentration": 0.25,
            "max_single_name_var": 0.02
        })
        soft_limits = context.portfolio_state.get("soft_limits", {
            "target_position_pct": 0.02,
            "target_portfolio_vol": 0.12,
            "preferred_sector_limit": 0.15
        })
        
        return DEEP_ANALYSIS_PROMPT.format(
            proposal=context.trade_context,
            portfolio_state=context.portfolio_state,
            risk_environment=risk_env,
            hard_limits=hard_limits,
            soft_limits=soft_limits
        )
    
    def calculate_relevance_score(self, context: AgentContext) -> float:
        """Risk gatekeeper is always highly relevant for trade decisions."""
        base_score = 0.8  # Always important
        
        # Higher relevance for larger positions
        position_size = context.trade_context.get("position_pct", 0)
        if position_size > 0.03:
            base_score = min(1.0, base_score + 0.2)
        
        # Higher relevance in high volatility
        current_vol = context.market_data.get("current_volatility", 0.15)
        if current_vol > 0.25:
            base_score = min(1.0, base_score + 0.15)
        
        # Higher relevance when portfolio is already stressed
        current_drawdown = context.portfolio_state.get("current_drawdown", 0)
        if current_drawdown > 0.05:
            base_score = min(1.0, base_score + 0.1)
        
        return base_score
    
    def validate_output(self, output: AgentOutput) -> bool:
        """Ensure risk-specific output requirements are met."""
        if not super().validate_output(output):
            return False
        
        # Veto claims must have veto_reason
        if output.claim == "VETO":
            if not output.recommended_delta.get("veto_reason"):
                return False
        
        # Must include constraint analysis
        evidence = output.evidence
        if "constraint_violations" not in evidence:
            return False
        
        return True
    
    async def post_process(self, output: AgentOutput) -> AgentOutput:
        """Add risk-specific metrics and flags."""
        output = await super().post_process(output)
        
        # Mark if this is a hard veto (cannot be overridden)
        if output.claim == "VETO":
            violations = output.evidence.get("constraint_violations", [])
            output.evidence["is_hard_veto"] = any(
                "hard_limit" in v.lower() or "max_" in v.lower()
                for v in violations
            )
        
        return output


# =============================================================================
# EVALUATION - Test cases for measuring agent performance
# =============================================================================

EVALUATION_CASES = [
    {
        "name": "position_size_breach",
        "context": {
            "symbol": "TSLA",
            "proposed_action": "buy",
            "position_pct": 0.08,  # 8% of portfolio
            "current_price": 250.0
        },
        "portfolio_state": {
            "total_value": 1000000,
            "current_drawdown": 0.02,
            "hard_limits": {"max_position_pct": 0.05}
        },
        "market_data": {"current_volatility": 0.20},
        "expected_claim": "VETO",
        "expected_evidence_keys": ["constraint_violations", "proposed_trade_impact"]
    },
    {
        "name": "within_limits",
        "context": {
            "symbol": "AAPL",
            "proposed_action": "buy",
            "position_pct": 0.02,
            "current_price": 180.0
        },
        "portfolio_state": {
            "total_value": 1000000,
            "current_drawdown": 0.01,
            "hard_limits": {"max_position_pct": 0.05}
        },
        "market_data": {"current_volatility": 0.15},
        "expected_claim": "APPROVE",
        "expected_evidence_keys": ["current_portfolio_risk", "risk_adjusted_metrics"]
    },
    {
        "name": "drawdown_stress",
        "context": {
            "symbol": "AMD",
            "proposed_action": "buy",
            "position_pct": 0.03,
            "current_price": 150.0
        },
        "portfolio_state": {
            "total_value": 1000000,
            "current_drawdown": 0.08,  # Already at 8% drawdown
            "hard_limits": {"max_portfolio_drawdown": 0.10}
        },
        "market_data": {"current_volatility": 0.30},
        "expected_claim": "CONDITIONAL",
        "expected_evidence_keys": ["constraint_violations", "conditions"]
    }
]


def evaluate(agent: RiskGatekeeperAgent, cases: list = None) -> dict:
    """
    Evaluate agent performance against test cases.
    
    Returns:
        dict: Performance metrics including accuracy, veto accuracy, etc.
    """
    cases = cases or EVALUATION_CASES
    results = {
        "total_cases": len(cases),
        "correct_claims": 0,
        "veto_accuracy": 0,
        "approve_accuracy": 0,
        "evidence_completeness": 0,
        "prompt_version": agent.prompt_version,
        "details": []
    }
    
    veto_total = 0
    veto_correct = 0
    approve_total = 0
    approve_correct = 0
    
    for case in cases:
        context = AgentContext(
            trade_context=case["context"],
            portfolio_state=case["portfolio_state"],
            market_data=case["market_data"],
            complexity_score=0.5,
            time_budget_ms=5000
        )
        
        # This would need async handling in real use
        # For evaluation, we'd run this through the actual LLM
        case_result = {
            "name": case["name"],
            "expected": case["expected_claim"],
            "passed": False  # Would be set by actual evaluation
        }
        
        if case["expected_claim"] == "VETO":
            veto_total += 1
        elif case["expected_claim"] == "APPROVE":
            approve_total += 1
        
        results["details"].append(case_result)
    
    return results
