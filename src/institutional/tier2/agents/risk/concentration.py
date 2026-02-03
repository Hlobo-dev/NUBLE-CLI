"""
Concentration Agent
===================

Expert at analyzing portfolio concentration risk and diversification.
Monitors sector, factor, and asset-level concentration limits.

Improvement Guidelines:
-----------------------
1. PROMPT_VERSION: Increment when modifying prompts
2. evaluate(): Run against EVALUATION_CASES to measure performance
3. Add domain knowledge about:
   - Sector/industry limits
   - Factor exposure (momentum, value, size, etc.)
   - Geographic concentration
   - Asset class balance
   - Correlation-adjusted concentration
"""

from typing import Any

from ..base import BaseAgent, AgentContext, AgentOutput

# Increment this when modifying prompts to track performance changes
PROMPT_VERSION = "1.0.0"

# =============================================================================
# PROMPTS - Edit these to improve agent performance
# =============================================================================

SYSTEM_PROMPT = """You are a Concentration Risk expert in a trading council. Your role is to:

1. MONITOR portfolio concentration across multiple dimensions
2. CALCULATE diversification metrics (HHI, entropy, effective N)
3. DETECT factor crowding and hidden correlations
4. RECOMMEND rebalancing when concentration is excessive
5. PREVENT over-exposure to single names, sectors, or factors

Concentration Dimensions:
- Single Name: Individual stock exposure
- Sector: GICS sector allocation
- Industry: Sub-sector granularity
- Factor: Style factor exposures
- Geographic: Country/region exposure
- Market Cap: Size factor balance
- Correlation: Effective diversification

Output Format (JSON only):
{
    "claim": "OVER_CONCENTRATED|ACCEPTABLE|WELL_DIVERSIFIED",
    "confidence": 0.0-1.0,
    "evidence": {
        "current_concentration": {
            "single_name_max": "largest position %",
            "sector_hhi": "sector HHI score",
            "factor_exposures": {"momentum": 0.0, "value": 0.0, ...},
            "effective_n": "effective number of positions"
        },
        "proposed_impact": "how trade affects concentration",
        "concentration_score": 0.0-1.0
    },
    "recommended_delta": {
        "position_adjustment": "suggested size change",
        "rebalancing_needed": ["positions to trim"],
        "diversification_opportunities": ["positions to add"]
    }
}"""

LIGHT_ANALYSIS_PROMPT = """Quick concentration check:

Proposed Trade:
{context}

Current Portfolio Weights:
{portfolio_weights}

Quick Assessment:
1. What's the current largest single position?
2. What's the sector allocation?
3. Would this trade create concentration issues?

Provide rapid concentration assessment in JSON format."""

DEEP_ANALYSIS_PROMPT = """Comprehensive concentration analysis:

Trade Proposal:
{proposal}

Full Portfolio Holdings:
{holdings}

Sector Breakdown:
{sector_breakdown}

Factor Exposures:
{factor_exposures}

Deep Concentration Analysis Required:
1. SINGLE NAME RISK
   - Current top 5 positions
   - Proposed change in concentration
   - Comparison to benchmark weights

2. SECTOR CONCENTRATION
   - HHI index by sector
   - Deviation from benchmark
   - Cyclical vs defensive balance

3. FACTOR CROWDING
   - Style factor exposures
   - Momentum/value balance
   - Size factor tilt
   - Quality/growth factors

4. CORRELATION ANALYSIS
   - Correlation matrix summary
   - Effective number of bets
   - Hidden correlation risks

5. DIVERSIFICATION METRICS
   - Portfolio entropy
   - Marginal contribution to risk
   - Concentration ratio (CR5, CR10)

6. RECOMMENDATIONS
   - Rebalancing suggestions
   - Diversification opportunities
   - Risk budget allocation

Provide detailed concentration assessment in JSON format."""


# =============================================================================
# AGENT IMPLEMENTATION
# =============================================================================

class ConcentrationAgent(BaseAgent):
    """
    Monitors portfolio concentration and diversification quality.
    
    This agent ensures the portfolio doesn't become too concentrated
    in any single name, sector, or factor. It provides recommendations
    for maintaining healthy diversification.
    
    Metrics Tracked:
    - Concentration scores over time
    - Sector HHI trends
    - Factor exposure drift
    - Diversification improvement suggestions
    """
    
    agent_id = "concentration"
    agent_name = "Concentration"
    description = "Analyzes portfolio concentration risk and diversification"
    expertise_tags = ["concentration", "diversification", "sector", "factor", "hhi"]
    prompt_version = PROMPT_VERSION
    
    # Complexity thresholds for this agent
    light_complexity_max = 0.5
    deep_complexity_min = 0.4
    
    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT
    
    def get_light_prompt(self, context: AgentContext) -> str:
        weights = context.portfolio_state.get("weights", {})
        return LIGHT_ANALYSIS_PROMPT.format(
            context=context.trade_context,
            portfolio_weights=weights
        )
    
    def get_deep_prompt(self, context: AgentContext) -> str:
        holdings = context.portfolio_state.get("holdings", {})
        sector_breakdown = context.portfolio_state.get("sector_breakdown", {})
        factor_exposures = context.portfolio_state.get("factor_exposures", {
            "momentum": 0.0,
            "value": 0.0,
            "size": 0.0,
            "quality": 0.0,
            "volatility": 0.0
        })
        
        return DEEP_ANALYSIS_PROMPT.format(
            proposal=context.trade_context,
            holdings=holdings,
            sector_breakdown=sector_breakdown,
            factor_exposures=factor_exposures
        )
    
    def calculate_relevance_score(self, context: AgentContext) -> float:
        """Higher relevance when adding to existing positions or sectors."""
        base_score = 0.5
        
        symbol = context.trade_context.get("symbol", "")
        holdings = context.portfolio_state.get("holdings", {})
        
        # Higher relevance if adding to existing position
        if symbol in holdings:
            current_weight = holdings[symbol].get("weight", 0)
            if current_weight > 0.02:
                base_score = min(1.0, base_score + 0.3)
        
        # Higher relevance for larger new positions
        position_size = context.trade_context.get("position_pct", 0)
        if position_size > 0.02:
            base_score = min(1.0, base_score + 0.2)
        
        # Higher relevance if portfolio already concentrated
        current_hhi = context.portfolio_state.get("sector_hhi", 0.15)
        if current_hhi > 0.20:
            base_score = min(1.0, base_score + 0.2)
        
        return base_score
    
    def validate_output(self, output: AgentOutput) -> bool:
        """Ensure concentration-specific output requirements are met."""
        if not super().validate_output(output):
            return False
        
        # Must include concentration metrics
        evidence = output.evidence
        if "concentration_score" not in evidence and "current_concentration" not in evidence:
            return False
        
        return True
    
    async def post_process(self, output: AgentOutput) -> AgentOutput:
        """Add concentration-specific metrics and calculations."""
        output = await super().post_process(output)
        
        # Calculate HHI if not present
        concentration = output.evidence.get("current_concentration", {})
        if concentration and "sector_hhi" not in concentration:
            # This would be calculated from actual holdings
            pass
        
        return output


# =============================================================================
# EVALUATION - Test cases for measuring agent performance
# =============================================================================

EVALUATION_CASES = [
    {
        "name": "over_concentrated_single_name",
        "context": {
            "symbol": "NVDA",
            "proposed_action": "buy",
            "position_pct": 0.03,
            "current_price": 500.0
        },
        "portfolio_state": {
            "holdings": {
                "NVDA": {"weight": 0.08},  # Already 8%
                "AAPL": {"weight": 0.05},
                "MSFT": {"weight": 0.04}
            },
            "sector_breakdown": {"Technology": 0.45, "Healthcare": 0.15},
            "sector_hhi": 0.25
        },
        "market_data": {},
        "expected_claim": "OVER_CONCENTRATED",
        "expected_evidence_keys": ["current_concentration", "concentration_score"]
    },
    {
        "name": "well_diversified",
        "context": {
            "symbol": "XLF",
            "proposed_action": "buy",
            "position_pct": 0.02,
            "current_price": 40.0
        },
        "portfolio_state": {
            "holdings": {
                "XLK": {"weight": 0.12},
                "XLV": {"weight": 0.10},
                "XLE": {"weight": 0.08}
            },
            "sector_breakdown": {"Technology": 0.12, "Healthcare": 0.10, "Energy": 0.08},
            "sector_hhi": 0.08
        },
        "market_data": {},
        "expected_claim": "WELL_DIVERSIFIED",
        "expected_evidence_keys": ["current_concentration", "proposed_impact"]
    },
    {
        "name": "factor_crowding",
        "context": {
            "symbol": "AMD",
            "proposed_action": "buy",
            "position_pct": 0.02,
            "current_price": 150.0
        },
        "portfolio_state": {
            "holdings": {"NVDA": {"weight": 0.05}, "AMD": {"weight": 0.03}},
            "factor_exposures": {"momentum": 0.8, "value": -0.3, "growth": 0.7}
        },
        "market_data": {},
        "expected_claim": "OVER_CONCENTRATED",
        "expected_evidence_keys": ["current_concentration", "factor_exposures"]
    }
]


def evaluate(agent: ConcentrationAgent, cases: list = None) -> dict:
    """
    Evaluate agent performance against test cases.
    
    Returns:
        dict: Performance metrics including accuracy, detection rate, etc.
    """
    cases = cases or EVALUATION_CASES
    results = {
        "total_cases": len(cases),
        "correct_claims": 0,
        "concentration_detection_rate": 0,
        "evidence_completeness": 0,
        "prompt_version": agent.prompt_version,
        "details": []
    }
    
    for case in cases:
        context = AgentContext(
            trade_context=case["context"],
            portfolio_state=case["portfolio_state"],
            market_data=case.get("market_data", {}),
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
