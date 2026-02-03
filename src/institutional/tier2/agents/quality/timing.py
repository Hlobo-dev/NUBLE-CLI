"""
Timing Agent
============

Expert at optimizing entry/exit timing and market microstructure.
Analyzes intraday patterns, momentum, and optimal execution windows.

Improvement Guidelines:
-----------------------
1. PROMPT_VERSION: Increment when modifying prompts
2. evaluate(): Run against EVALUATION_CASES to measure performance
3. Add domain knowledge about:
   - Intraday volume patterns
   - Momentum and mean reversion timing
   - Earnings/event timing
   - Overnight vs intraday risk
   - Session-specific behavior
"""

from typing import Any

from ..base import BaseAgent, AgentContext, AgentOutput

# Increment this when modifying prompts to track performance changes
PROMPT_VERSION = "1.0.0"

# =============================================================================
# PROMPTS - Edit these to improve agent performance
# =============================================================================

SYSTEM_PROMPT = """You are a Timing and Market Microstructure expert in a trading council. Your role is to:

1. ANALYZE optimal entry/exit timing within trading sessions
2. ASSESS momentum vs mean reversion conditions
3. IDENTIFY favorable execution windows
4. CONSIDER overnight gaps and event risk
5. RECOMMEND timing adjustments for better execution

Timing Considerations:
- Session Dynamics: Open/close effects, lunch lull
- Momentum: Continuation vs exhaustion signals
- Volume Profile: High vs low liquidity periods
- Event Proximity: Earnings, economic data timing
- Technical Levels: Support/resistance timing

Output Format (JSON only):
{
    "claim": "GOOD_TIMING|WAIT|POOR_TIMING",
    "confidence": 0.0-1.0,
    "evidence": {
        "current_session_analysis": {
            "time_in_session": "open|midday|close",
            "volume_profile": "above|below average",
            "momentum_state": "strong|weak|exhausted"
        },
        "timing_factors": {
            "session_favorability": 0.0-1.0,
            "momentum_alignment": 0.0-1.0,
            "volume_support": 0.0-1.0,
            "event_risk": 0.0-1.0
        },
        "timing_score": 0.0-1.0
    },
    "recommended_delta": {
        "optimal_action": "execute_now|wait_for_pullback|wait_for_close|defer_to_tomorrow",
        "suggested_wait_time": "minutes or session",
        "target_entry_zone": "price range",
        "reasoning": "explanation"
    }
}"""

LIGHT_ANALYSIS_PROMPT = """Quick timing check:

Trade Context:
{context}

Current Session:
{session_info}

Quick Assessment:
1. Where are we in the trading session?
2. Is volume supporting the move?
3. Is this a good time to execute?

Provide rapid timing assessment in JSON format."""

DEEP_ANALYSIS_PROMPT = """Comprehensive timing analysis:

Trade Proposal:
{proposal}

Session Context:
{session_context}

Intraday Price Action:
{intraday_data}

Volume Profile:
{volume_profile}

Technical Levels:
{technical_levels}

Deep Timing Analysis Required:
1. SESSION ANALYSIS
   - Time in trading session
   - Historical session patterns
   - Typical volatility by period
   - End-of-day effects

2. MOMENTUM ASSESSMENT
   - Intraday momentum strength
   - RSI/momentum indicators
   - Momentum exhaustion signals
   - Mean reversion potential

3. VOLUME ANALYSIS
   - Current vs average volume
   - Volume trend intraday
   - Participation patterns
   - Block trade activity

4. TECHNICAL TIMING
   - Proximity to support/resistance
   - Breakout/breakdown timing
   - Moving average positioning
   - Fibonacci levels

5. EVENT TIMING
   - Upcoming events (earnings, FOMC)
   - Economic data releases
   - Overnight gap risk
   - Weekend effect

6. EXECUTION RECOMMENDATION
   - Optimal entry timing
   - Wait vs execute decision
   - Entry zone suggestion
   - Risk/reward timing

Provide detailed timing assessment in JSON format."""


# =============================================================================
# AGENT IMPLEMENTATION
# =============================================================================

class TimingAgent(BaseAgent):
    """
    Optimizes entry/exit timing based on market microstructure.
    
    This agent analyzes intraday patterns, momentum, and market
    conditions to recommend optimal execution timing. It can
    suggest waiting for better entry points.
    
    Metrics Tracked:
    - Timing recommendation accuracy
    - Entry price improvement
    - Wait recommendations success rate
    - Session pattern accuracy
    """
    
    agent_id = "timing"
    agent_name = "Timing"
    description = "Optimizes entry/exit timing and market microstructure"
    expertise_tags = ["timing", "execution", "momentum", "session", "microstructure"]
    prompt_version = PROMPT_VERSION
    
    # Complexity thresholds for this agent
    light_complexity_max = 0.5
    deep_complexity_min = 0.4
    
    def get_system_prompt(self) -> str:
        return SYSTEM_PROMPT
    
    def get_light_prompt(self, context: AgentContext) -> str:
        session_info = context.market_data.get("session_info", {
            "time_in_session": "unknown",
            "minutes_to_close": 0
        })
        return LIGHT_ANALYSIS_PROMPT.format(
            context=context.trade_context,
            session_info=session_info
        )
    
    def get_deep_prompt(self, context: AgentContext) -> str:
        return DEEP_ANALYSIS_PROMPT.format(
            proposal=context.trade_context,
            session_context=context.market_data.get("session_context", {}),
            intraday_data=context.market_data.get("intraday_data", []),
            volume_profile=context.market_data.get("volume_profile", {}),
            technical_levels=context.market_data.get("technical_levels", {})
        )
    
    def calculate_relevance_score(self, context: AgentContext) -> float:
        """Higher relevance for time-sensitive situations."""
        base_score = 0.5
        
        # Higher relevance near market open/close
        session_info = context.market_data.get("session_info", {})
        time_in_session = session_info.get("time_in_session", "midday")
        if time_in_session in ["open", "close"]:
            base_score = min(1.0, base_score + 0.25)
        
        # Higher relevance for momentum trades
        trade_type = context.trade_context.get("trade_type", "")
        if "momentum" in trade_type.lower():
            base_score = min(1.0, base_score + 0.2)
        
        # Higher relevance near events
        events_today = context.market_data.get("events_today", [])
        if events_today:
            base_score = min(1.0, base_score + 0.2)
        
        return base_score
    
    def validate_output(self, output: AgentOutput) -> bool:
        """Ensure timing-specific output requirements are met."""
        if not super().validate_output(output):
            return False
        
        # Must include timing score or factors
        evidence = output.evidence
        if "timing_score" not in evidence and "timing_factors" not in evidence:
            return False
        
        return True
    
    async def post_process(self, output: AgentOutput) -> AgentOutput:
        """Add timing-specific metrics and flags."""
        output = await super().post_process(output)
        
        # Calculate overall timing score if not present
        if "timing_score" not in output.evidence:
            factors = output.evidence.get("timing_factors", {})
            if factors:
                avg_score = sum(factors.values()) / len(factors)
                output.evidence["timing_score"] = avg_score
        
        return output


# =============================================================================
# EVALUATION - Test cases for measuring agent performance
# =============================================================================

EVALUATION_CASES = [
    {
        "name": "good_timing_midday",
        "context": {
            "symbol": "AAPL",
            "proposed_action": "buy",
            "trade_type": "trend_follow"
        },
        "market_data": {
            "session_info": {
                "time_in_session": "midday",
                "minutes_to_close": 180
            },
            "volume_profile": {"current_vs_avg": 1.2},
            "intraday_data": {
                "momentum": "positive",
                "trend": "up"
            }
        },
        "portfolio_state": {},
        "expected_claim": "GOOD_TIMING",
        "expected_evidence_keys": ["timing_score", "timing_factors"]
    },
    {
        "name": "poor_timing_exhaustion",
        "context": {
            "symbol": "TSLA",
            "proposed_action": "buy",
            "trade_type": "momentum"
        },
        "market_data": {
            "session_info": {
                "time_in_session": "close",
                "minutes_to_close": 15
            },
            "intraday_data": {
                "momentum": "exhausted",
                "rsi": 85,
                "up_percent_today": 5.5
            }
        },
        "portfolio_state": {},
        "expected_claim": "POOR_TIMING",
        "expected_evidence_keys": ["momentum_state", "timing_score"]
    },
    {
        "name": "wait_for_pullback",
        "context": {
            "symbol": "AMD",
            "proposed_action": "buy",
            "trade_type": "swing"
        },
        "market_data": {
            "session_info": {
                "time_in_session": "open",
                "minutes_from_open": 30
            },
            "intraday_data": {
                "gap_up": 2.5,
                "no_pullback_yet": True
            }
        },
        "portfolio_state": {},
        "expected_claim": "WAIT",
        "expected_evidence_keys": ["timing_factors", "suggested_wait_time"]
    }
]


def evaluate(agent: TimingAgent, cases: list = None) -> dict:
    """
    Evaluate agent performance against test cases.
    
    Returns:
        dict: Performance metrics including timing accuracy, etc.
    """
    cases = cases or EVALUATION_CASES
    results = {
        "total_cases": len(cases),
        "correct_claims": 0,
        "timing_accuracy": 0,
        "wait_success_rate": 0,
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
