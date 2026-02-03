"""
Trend Integrity Agent
======================

Analyzes trend structure integrity and sustainability.

Key Focus:
- Higher highs / higher lows (uptrend)
- Lower highs / lower lows (downtrend)
- Support and resistance levels
- Trend breaks and false breakouts
- Price structure quality

Evidence Keys Used:
- trend_state
- price
- sma_20, sma_50, sma_200
- rsi
- atr_pct

Claim Types:
- support: Trend structure intact
- risk: Trend deterioration or breaks
- timing: Structure-based entry/exit points
"""

from ..base import BaseAgent, AgentContext


class TrendIntegrityAgent(BaseAgent):
    """
    Trend Integrity Agent.
    
    Evaluates the quality and sustainability of the current trend.
    """
    
    name = "trend_integrity"
    description = "Analyzes trend structure integrity and sustainability"
    category = "technical"
    
    PROMPT_VERSION = "1.0"
    CLAIM_PREFIX = "TRD"
    
    def get_system_prompt(self) -> str:
        return """You are an expert trend structure analyst with deep understanding of price action.

YOUR EXPERTISE:
- Identifying trend quality through HL/LH patterns
- Detecting trend breaks and potential reversals
- Assessing support/resistance relative to current price
- Understanding trend sustainability via SMA relationships

YOUR ROLE:
- Evaluate if the current trend structure is intact
- Identify any concerning trend breaks or weaknesses
- Assess price position relative to key moving averages
- Determine if the trend supports the proposed trade

YOU MUST:
- Focus on STRUCTURE, not just direction
- Consider SMA relationships as dynamic S/R
- Flag any trend deterioration signs
- Reference specific price levels and indicators

YOU MUST NOT:
- Confuse choppy action for trends
- Ignore trend breaks against the trade direction
- Make claims without structural evidence"""

    def get_light_prompt(self, context: AgentContext) -> str:
        return f"""Quick trend structure check for {context.symbol}.

RAPID ASSESSMENT:
1. Is price above or below key SMAs (20/50/200)?
2. Is the trend structure (HL/LH pattern) intact?
3. Any immediate structural concerns?

Make 1-2 focused claims about trend integrity."""

    def get_deep_prompt(self, context: AgentContext) -> str:
        return f"""Comprehensive trend structure analysis for {context.symbol}.

DETAILED ANALYSIS:

1. TREND DIRECTION
   - Current trend state: {context.trend_state}
   - Is this consistent with price action?
   - Trend duration and maturity

2. PRICE VS SMA STRUCTURE
   - Price vs SMA 20: ${context.price:.2f} vs ${context.sma_20:.2f}
   - Price vs SMA 50: ${context.price:.2f} vs ${context.sma_50:.2f}  
   - Price vs SMA 200: ${context.price:.2f} vs ${context.sma_200:.2f}
   - SMA ordering (bullish/bearish/mixed)

3. MOMENTUM HEALTH
   - RSI: {context.rsi:.1f} - Supports or diverges from trend?
   - Is momentum confirming or warning?

4. STRUCTURAL RISKS
   - Any recent trend breaks?
   - Distance from key levels (overextended?)
   - ATR {context.atr_pct:.2f}% - volatility context

5. SUSTAINABILITY
   - Can this trend continue?
   - What would invalidate it?

Make up to 5 claims with specific evidence."""

    def get_claim_prefix(self) -> str:
        return self.CLAIM_PREFIX


EXAMPLE_TEST_CASES = [
    {
        "name": "strong_uptrend",
        "context": {
            "symbol": "AAPL",
            "price": 190.0,
            "action": "BUY",
            "direction": "BULLISH",
            "confidence": 75.0,
            "trend_state": "UPTREND",
            "sma_20": 187.0,
            "sma_50": 182.0,
            "sma_200": 170.0,
            "rsi": 62.0,
        },
        "expected_verdict": "BUY",
        "expected_stance": "pro",
    },
    {
        "name": "trend_break_warning",
        "context": {
            "symbol": "META",
            "price": 320.0,
            "action": "BUY",
            "direction": "BULLISH",
            "confidence": 60.0,
            "trend_state": "UPTREND",
            "sma_20": 325.0,  # Price below SMA 20
            "sma_50": 315.0,
            "sma_200": 290.0,
            "rsi": 42.0,  # RSI weak
        },
        "expected_verdict": "WAIT",
        "expected_stance": "anti",
    },
]
