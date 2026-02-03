"""
Reversal vs Pullback Agent
===========================

Distinguishes between healthy pullbacks and actual trend reversals.

Key Focus:
- Pullback depth and character
- Reversal warning signs
- False breakout detection
- Mean reversion vs trend continuation
- Volume/momentum character during pullbacks

Evidence Keys Used:
- rsi (oversold/overbought levels)
- atr_pct (pullback magnitude)
- trend_state
- sma_20, sma_50, sma_200
- macd_value, macd_signal

Claim Types:
- support: Healthy pullback, good entry
- risk: Reversal warning signs
- timing: Wait for confirmation
"""

from ..base import BaseAgent, AgentContext


class ReversalPullbackAgent(BaseAgent):
    """
    Reversal vs Pullback Agent.
    
    Distinguishes between corrections within a trend and actual reversals.
    """
    
    name = "reversal_pullback"
    description = "Distinguishes between healthy pullbacks and trend reversals"
    category = "technical"
    
    PROMPT_VERSION = "1.0"
    CLAIM_PREFIX = "RVP"
    
    def get_system_prompt(self) -> str:
        return """You are an expert at distinguishing pullbacks from reversals with decades of experience.

YOUR EXPERTISE:
- Identifying healthy corrections within trends
- Spotting reversal warning signs early
- Understanding pullback depth relative to volatility
- Detecting false breakouts and failed moves
- Analyzing momentum during corrections

YOUR ROLE:
- Determine if current price action is a pullback or reversal
- Assess if the trade timing is appropriate
- Identify warning signs that suggest trend exhaustion
- Recommend whether to act now or wait

KEY DISTINCTIONS:
- PULLBACK: Correction within intact trend, healthy consolidation, momentum intact
- REVERSAL: Trend break, failed structure, momentum divergence, exhaustion signs

YOU MUST:
- Consider ATR for "normal" vs "excessive" pullback depth
- Look at RSI for oversold/overbought context
- Check if MACD shows divergence (reversal sign)
- Reference specific levels and evidence

YOU MUST NOT:
- Automatically call every dip a "buying opportunity"
- Ignore reversal warning signs
- Make timing claims without technical support"""

    def get_light_prompt(self, context: AgentContext) -> str:
        rsi = context.rsi
        atr = context.atr_pct
        
        return f"""Quick pullback vs reversal assessment for {context.symbol}.

CURRENT STATE:
- RSI: {rsi:.1f} (Oversold < 30, Overbought > 70)
- ATR%: {atr:.2f}%
- Trend: {context.trend_state}

KEY QUESTION: Is this a healthy pullback or a potential reversal?

Make 1-2 claims about pullback/reversal character."""

    def get_deep_prompt(self, context: AgentContext) -> str:
        return f"""Detailed pullback vs reversal analysis for {context.symbol}.

COMPREHENSIVE ASSESSMENT:

1. PULLBACK DEPTH ANALYSIS
   - Current pullback magnitude relative to ATR ({context.atr_pct:.2f}%)
   - Price distance from key SMAs
   - Is depth "normal" for this trend?

2. MOMENTUM CHARACTER
   - RSI: {context.rsi:.1f} - Indicating what?
   - MACD: {context.macd_value:.4f} vs Signal {context.macd_signal:.4f}
   - Any divergences between price and momentum?

3. STRUCTURAL CLUES
   - Has price broken key support/resistance?
   - Are higher lows (uptrend) or lower highs (downtrend) intact?
   - Quality of recent price action

4. REVERSAL WARNING SIGNS (check all)
   - [ ] Momentum divergence
   - [ ] Failed breakout/breakdown
   - [ ] Excessive pullback depth (> 2x ATR)
   - [ ] RSI extreme with no recovery
   - [ ] Multiple failed attempts at key level

5. VERDICT
   - Pullback (good entry) vs Reversal (avoid/exit)
   - Confidence in assessment
   - Wait for confirmation signals?

Make up to 5 claims with specific evidence."""

    def get_claim_prefix(self) -> str:
        return self.CLAIM_PREFIX


EXAMPLE_TEST_CASES = [
    {
        "name": "healthy_pullback",
        "context": {
            "symbol": "MSFT",
            "price": 380.0,
            "action": "BUY",
            "direction": "BULLISH",
            "confidence": 70.0,
            "trend_state": "UPTREND",
            "rsi": 42.0,  # Pulled back but not oversold
            "atr_pct": 1.5,
            "sma_20": 385.0,
            "sma_50": 375.0,
            "sma_200": 350.0,
        },
        "expected_verdict": "BUY",
        "expected_stance": "pro",
    },
    {
        "name": "reversal_warning",
        "context": {
            "symbol": "COIN",
            "price": 120.0,
            "action": "BUY",
            "direction": "BULLISH",
            "confidence": 55.0,
            "trend_state": "UPTREND",
            "rsi": 28.0,  # Deeply oversold
            "atr_pct": 4.5,  # High volatility
            "sma_20": 135.0,  # Price way below
            "sma_50": 140.0,
            "sma_200": 130.0,
            "macd_value": -2.5,  # Negative
            "macd_signal": -1.5,
        },
        "expected_verdict": "WAIT",
        "expected_stance": "anti",
    },
]
