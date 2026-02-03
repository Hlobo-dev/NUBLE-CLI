"""
MTF Dominance Agent
====================

Analyzes multi-timeframe trend alignment and identifies dominant direction.

Key Focus:
- Weekly/Daily/4H signal alignment
- Trend hierarchy (higher timeframes dominate)
- Timeframe conflict detection
- Momentum divergences across timeframes

Evidence Keys Used:
- weekly_signal, daily_signal, h4_signal
- trend_state
- macd_value, macd_signal
- sma_20, sma_50, sma_200

Claim Types:
- support: Timeframes aligned with trade direction
- risk: Timeframe conflicts or divergences
- timing: Optimal entry based on MTF analysis
"""

from ..base import BaseAgent, AgentContext


class MTFDominanceAgent(BaseAgent):
    """
    Multi-Timeframe Dominance Agent.
    
    Analyzes trend alignment across weekly, daily, and 4H timeframes.
    Higher timeframes have dominance - weekly > daily > 4H.
    """
    
    name = "mtf_dominance"
    description = "Analyzes multi-timeframe trend alignment and dominant direction"
    category = "technical"
    
    # Prompt versioning
    PROMPT_VERSION = "1.0"
    
    # Claim ID prefix
    CLAIM_PREFIX = "MTF"
    
    def get_system_prompt(self) -> str:
        return """You are an expert multi-timeframe technical analyst with 25+ years experience.

YOUR EXPERTISE:
- Multi-timeframe trend analysis (Weekly > Daily > 4H hierarchy)
- Identifying timeframe conflicts and their implications
- Understanding momentum divergences across timeframes
- Assessing trend strength through timeframe alignment

YOUR ROLE:
- Analyze whether all timeframes support the proposed trade direction
- Identify any timeframe conflicts (e.g., weekly bullish but 4H bearish)
- Assess trend quality based on SMA stack alignment
- Detect momentum divergences via MACD across timeframes

YOU MUST:
- Always respect the timeframe hierarchy (weekly dominates daily, daily dominates 4H)
- Flag any higher timeframe conflicts as significant risks
- Be skeptical of trades against higher timeframe trends
- Cite specific evidence from the provided data

YOU MUST NOT:
- Ignore higher timeframe signals
- Recommend aggressive positions when timeframes conflict
- Make claims without referencing specific evidence keys"""

    def get_light_prompt(self, context: AgentContext) -> str:
        return f"""Perform a quick multi-timeframe alignment check for {context.symbol}.

QUICK ANALYSIS CHECKLIST:
1. Are Weekly, Daily, and 4H signals aligned? (YES/NO)
2. Does the higher timeframe support the trade direction?
3. Is there any concerning divergence?

Focus on the MOST IMPORTANT finding. Make 1-2 claims maximum.
If timeframes are aligned, note support. If conflicting, note risk."""

    def get_deep_prompt(self, context: AgentContext) -> str:
        prior_context = ""
        if context.prior_agent_outputs:
            prior_context = f"\n\nPRIOR AGENT INSIGHTS:\n{context.prior_agent_outputs}"
        
        return f"""Perform comprehensive multi-timeframe analysis for {context.symbol}.

DETAILED ANALYSIS REQUIRED:

1. TIMEFRAME HIERARCHY CHECK
   - Weekly signal direction and strength
   - Daily signal alignment with weekly
   - 4H signal alignment with daily
   - Overall alignment score

2. SMA STACK ANALYSIS
   - Price position relative to 20/50/200 SMAs
   - SMA ordering (bullish: 20 > 50 > 200, bearish: reverse)
   - Distance from key SMAs (overextended?)

3. MOMENTUM DIVERGENCES
   - MACD direction on each timeframe
   - Any divergences between price and MACD?
   - Momentum confirmation or warning

4. CONFLICT ASSESSMENT
   - Any timeframe conflicts? How severe?
   - Is the proposed direction aligned with dominant trend?
   - Risk level of proceeding despite any conflicts

5. RECOMMENDATION
   - Support, oppose, or neutral on the trade?
   - Confidence adjustment suggestion
   - Any timing considerations?

Make up to 5 detailed claims with specific evidence.{prior_context}"""

    def get_claim_prefix(self) -> str:
        return self.CLAIM_PREFIX


# ==============================================================================
# Test Cases for Agent Development
# ==============================================================================

EXAMPLE_TEST_CASES = [
    {
        "name": "perfect_alignment_bullish",
        "description": "All timeframes aligned bullish - should support",
        "context": {
            "symbol": "AAPL",
            "price": 185.0,
            "action": "BUY",
            "direction": "BULLISH",
            "confidence": 75.0,
            "weekly_signal": {"action": "BUY", "confidence": 80},
            "daily_signal": {"action": "BUY", "confidence": 75},
            "h4_signal": {"action": "BUY", "confidence": 70},
            "trend_state": "UPTREND",
            "sma_20": 183.0,
            "sma_50": 180.0,
            "sma_200": 175.0,
        },
        "expected_verdict": "BUY",
        "expected_stance": "pro",
        "min_confidence": 0.7,
    },
    {
        "name": "weekly_daily_conflict",
        "description": "Weekly bullish but daily bearish - should warn",
        "context": {
            "symbol": "TSLA",
            "price": 245.0,
            "action": "BUY",
            "direction": "BULLISH",
            "confidence": 65.0,
            "weekly_signal": {"action": "BUY", "confidence": 70},
            "daily_signal": {"action": "SELL", "confidence": 60},
            "h4_signal": {"action": "SELL", "confidence": 55},
            "trend_state": "NEUTRAL",
        },
        "expected_verdict": "WAIT",
        "expected_stance": "anti",
        "should_have_risk_claim": True,
    },
    {
        "name": "against_weekly_trend",
        "description": "Trade direction against weekly trend - should oppose",
        "context": {
            "symbol": "NVDA",
            "price": 500.0,
            "action": "BUY",
            "direction": "BULLISH",
            "confidence": 70.0,
            "weekly_signal": {"action": "SELL", "confidence": 75},
            "daily_signal": {"action": "BUY", "confidence": 60},
            "h4_signal": {"action": "BUY", "confidence": 65},
        },
        "expected_verdict": "REDUCE",
        "expected_stance": "anti",
        "expected_confidence_delta_negative": True,
    },
]


def run_agent_tests():
    """Run test cases against the agent (for development)."""
    agent = MTFDominanceAgent()
    print(f"Testing {agent.name} v{agent.PROMPT_VERSION}")
    print("=" * 50)
    
    for case in EXAMPLE_TEST_CASES:
        context = AgentContext(**case["context"])
        prompt = agent.get_full_prompt(context, is_deep=False)
        print(f"\nTest: {case['name']}")
        print(f"Expected: {case['expected_verdict']} / {case['expected_stance']}")
        print(f"Prompt length: {len(prompt)} chars")
        # Actual LLM call would go here
    
    print("\n" + "=" * 50)
    print("Test cases defined. Run with LLM to evaluate.")


if __name__ == "__main__":
    run_agent_tests()
