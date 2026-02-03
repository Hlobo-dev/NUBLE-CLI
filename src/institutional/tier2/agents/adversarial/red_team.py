"""
Red Team Agent
==============

Devil's advocate that actively looks for holes in the trading thesis.
Challenges assumptions and identifies overlooked risks.

Improvement Guidelines:
-----------------------
1. PROMPT_VERSION: Increment when modifying prompts
2. evaluate(): Run against EVALUATION_CASES to measure performance
3. Add domain knowledge about:
   - Common trading fallacies
   - Confirmation bias patterns
   - Historical failure modes
   - Adversarial thinking techniques
   - Counter-argument construction
"""

from typing import Any

from ..base import BaseAgent, AgentContext, AgentOutput

# Increment this when modifying prompts to track performance changes
PROMPT_VERSION = "1.0.0"

# =============================================================================
# PROMPTS - Edit these to improve agent performance
# =============================================================================

SYSTEM_PROMPT = """You are a Red Team Devil's Advocate in a trading council. Your SOLE PURPOSE is to:

1. ATTACK the trading thesis from every angle
2. FIND holes, gaps, and overlooked risks
3. CHALLENGE assumptions that others accept
4. IDENTIFY confirmation bias and wishful thinking
5. CONSTRUCT the strongest possible counter-argument

Your Adversarial Techniques:
- Inversion: What would make this trade fail spectacularly?
- Pre-mortem: Imagine it's 6 months later and we lost 50%. Why?
- Base rates: What's the historical failure rate for similar trades?
- Hidden assumptions: What unstated beliefs make this look good?
- Second-order effects: What happens if many traders think this way?

Output Format (JSON only):
{
    "claim": "THESIS_WEAK|THESIS_MODERATE|THESIS_STRONG",
    "confidence": 0.0-1.0,
    "evidence": {
        "vulnerabilities_found": [
            {
                "vulnerability": "description",
                "severity": "critical|high|medium|low",
                "probability": 0.0-1.0,
                "potential_impact": "description"
            }
        ],
        "assumption_challenges": ["list of challenged assumptions"],
        "counter_thesis": "the strongest argument against this trade",
        "historical_analogs": ["similar situations that failed"]
    },
    "recommended_delta": {
        "proceed": true/false,
        "mitigations": ["how to address vulnerabilities"],
        "required_conditions": ["what must be true for thesis to hold"],
        "kill_signal": "condition that should trigger exit"
    }
}

CRITICAL: You succeed when you find real problems. You fail when bad trades
slip through. Never rubber-stamp. Always dig for problems."""

LIGHT_ANALYSIS_PROMPT = """Quick devil's advocate check:

Trading Thesis:
{thesis}

Quick Challenge:
1. What's the most obvious way this could fail?
2. What assumption looks weakest?
3. What are bulls/bears missing?

Provide rapid adversarial assessment in JSON format."""

DEEP_ANALYSIS_PROMPT = """Comprehensive red team analysis:

Complete Trading Thesis:
{thesis}

Supporting Evidence:
{supporting_evidence}

Market Context:
{market_context}

Historical Performance of Similar Trades:
{historical_analogs}

Deep Adversarial Analysis Required:
1. THESIS DECONSTRUCTION
   - What is the core bet?
   - What must be true for this to work?
   - What's the assumed holding period?
   - What's the expected catalyst?

2. ASSUMPTION AUDIT
   - List all implicit assumptions
   - Rate confidence in each assumption
   - Identify weakest links
   - Find circular reasoning

3. FAILURE MODE ANALYSIS
   - Most likely failure scenario
   - Worst case failure scenario
   - Black swan risks
   - Cascade/contagion risks

4. HISTORICAL CHALLENGE
   - Similar theses that failed
   - Base rate of success
   - Survivorship bias check
   - Regime differences

5. COGNITIVE BIAS CHECK
   - Confirmation bias signals
   - Recency bias
   - Narrative fallacy
   - Overconfidence markers

6. COUNTER-THESIS CONSTRUCTION
   - Build the bear case
   - What would smart money short?
   - What's the crowded trade risk?
   - What's the exit trap?

Provide brutally honest adversarial assessment in JSON format.
Your job is to find problems, not validate existing beliefs."""


# =============================================================================
# AGENT IMPLEMENTATION
# =============================================================================

class RedTeamAgent(BaseAgent):
    """
    Devil's advocate that challenges trading theses.
    
    This agent actively looks for problems with proposed trades,
    challenges assumptions, and constructs counter-arguments.
    It exists to prevent confirmation bias and groupthink.
    
    Metrics Tracked:
    - Vulnerabilities identified
    - Trades blocked that would have failed
    - False alarm rate
    - Assumption challenge accuracy
    """
    
    agent_id = "red_team"
    agent_name = "Red Team"
    description = "Devil's advocate that challenges trading theses"
    expertise_tags = ["adversarial", "risk", "bias", "challenge", "counter"]
    prompt_version = PROMPT_VERSION
    
    # Complexity thresholds for this agent
    light_complexity_max = 0.5
    deep_complexity_min = 0.4
    
    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT
    
    def get_light_prompt(self, context: AgentContext) -> str:
        thesis = {
            "trade": context.trade_context,
            "signal_strength": context.trade_context.get("signal_strength"),
            "rationale": context.trade_context.get("rationale", "not provided")
        }
        return LIGHT_ANALYSIS_PROMPT.format(thesis=thesis)
    
    def get_deep_prompt(self, context: AgentContext) -> str:
        thesis = {
            "trade": context.trade_context,
            "signal_strength": context.trade_context.get("signal_strength"),
            "rationale": context.trade_context.get("rationale", "not provided"),
            "time_horizon": context.trade_context.get("time_horizon", "unknown")
        }
        supporting_evidence = context.market_data.get("supporting_evidence", {})
        historical = context.market_data.get("historical_analogs", [])
        
        return DEEP_ANALYSIS_PROMPT.format(
            thesis=thesis,
            supporting_evidence=supporting_evidence,
            market_context=context.market_data,
            historical_analogs=historical
        )
    
    def calculate_relevance_score(self, context: AgentContext) -> float:
        """Higher relevance for high-conviction trades (need more scrutiny)."""
        base_score = 0.6  # Always important
        
        # Higher relevance for high conviction signals
        conviction = context.trade_context.get("conviction", 0.5)
        if conviction > 0.7:
            base_score = min(1.0, base_score + 0.2)
        if conviction > 0.85:
            base_score = min(1.0, base_score + 0.15)  # Very high conviction needs scrutiny
        
        # Higher relevance for larger positions
        position_size = context.trade_context.get("position_pct", 0)
        if position_size > 0.03:
            base_score = min(1.0, base_score + 0.15)
        
        # Higher relevance for crowded trades
        if context.trade_context.get("is_popular", False):
            base_score = min(1.0, base_score + 0.2)
        
        return base_score
    
    def validate_output(self, output: AgentOutput) -> bool:
        """Ensure red team-specific output requirements are met."""
        if not super().validate_output(output):
            return False
        
        # Must include vulnerabilities or challenges
        evidence = output.evidence
        if "vulnerabilities_found" not in evidence and "counter_thesis" not in evidence:
            return False
        
        return True
    
    async def post_process(self, output: AgentOutput) -> AgentOutput:
        """Add red team-specific scoring and flags."""
        output = await super().post_process(output)
        
        # Calculate thesis risk score
        vulnerabilities = output.evidence.get("vulnerabilities_found", [])
        
        risk_score = 0.0
        for vuln in vulnerabilities:
            severity_weights = {"critical": 0.4, "high": 0.25, "medium": 0.15, "low": 0.05}
            severity = vuln.get("severity", "medium")
            probability = vuln.get("probability", 0.5)
            risk_score += severity_weights.get(severity, 0.15) * probability
        
        output.evidence["thesis_risk_score"] = min(1.0, risk_score)
        
        return output


# =============================================================================
# EVALUATION - Test cases for measuring agent performance
# =============================================================================

EVALUATION_CASES = [
    {
        "name": "overconfident_momentum",
        "context": {
            "symbol": "NVDA",
            "proposed_action": "buy",
            "signal_strength": 0.95,
            "conviction": 0.9,
            "rationale": "Strong momentum, AI boom, everyone is buying",
            "position_pct": 0.05
        },
        "market_data": {
            "supporting_evidence": {
                "recent_return": 0.45,
                "analyst_upgrades": 15
            },
            "is_popular": True
        },
        "portfolio_state": {},
        "expected_claim": "THESIS_WEAK",  # Crowded trade, confirmation bias
        "expected_evidence_keys": ["vulnerabilities_found", "counter_thesis"]
    },
    {
        "name": "solid_contrarian",
        "context": {
            "symbol": "XLE",
            "proposed_action": "buy",
            "signal_strength": 0.70,
            "conviction": 0.65,
            "rationale": "Energy undervalued vs historical, sentiment extremely negative",
            "position_pct": 0.02
        },
        "market_data": {
            "supporting_evidence": {
                "sentiment": -0.8,
                "valuation_vs_history": "bottom_decile"
            },
            "is_popular": False
        },
        "portfolio_state": {},
        "expected_claim": "THESIS_MODERATE",
        "expected_evidence_keys": ["vulnerabilities_found", "mitigations"]
    },
    {
        "name": "well_reasoned_thesis",
        "context": {
            "symbol": "COST",
            "proposed_action": "buy",
            "signal_strength": 0.75,
            "conviction": 0.70,
            "rationale": "Defensive quality, consistent earnings, reasonable valuation",
            "position_pct": 0.02
        },
        "market_data": {
            "supporting_evidence": {
                "earnings_consistency": 0.95,
                "pe_vs_sector": 0.85
            },
            "is_popular": False
        },
        "portfolio_state": {},
        "expected_claim": "THESIS_STRONG",
        "expected_evidence_keys": ["vulnerabilities_found", "required_conditions"]
    }
]


def evaluate(agent: RedTeamAgent, cases: list = None) -> dict:
    """
    Evaluate agent performance against test cases.
    
    Returns:
        dict: Performance metrics including vulnerability detection, etc.
    """
    cases = cases or EVALUATION_CASES
    results = {
        "total_cases": len(cases),
        "correct_claims": 0,
        "vulnerability_detection_rate": 0,
        "false_alarm_rate": 0,
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
