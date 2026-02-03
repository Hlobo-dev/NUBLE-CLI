"""
Volatility State Agent
=======================

Analyzes current volatility regime and its implications for trading.

Key Focus:
- ATR-based volatility assessment
- VIX context (market-wide fear)
- Volatility regime (low/normal/high/extreme)
- Volatility expansion vs compression
- Position sizing implications

Evidence Keys Used:
- atr_pct
- vix
- vix_state
- regime
- regime_confidence

Claim Types:
- risk: High volatility warning
- timing: Wait for volatility compression
- support: Favorable volatility environment
"""

from ..base import BaseAgent, AgentContext


class VolatilityStateAgent(BaseAgent):
    """
    Volatility State Agent.
    
    Assesses volatility conditions and their trading implications.
    """
    
    name = "volatility_state"
    description = "Analyzes volatility regime and trading implications"
    category = "technical"
    
    PROMPT_VERSION = "1.0"
    CLAIM_PREFIX = "VOL"
    
    def get_system_prompt(self) -> str:
        return """You are a volatility specialist with deep expertise in risk management.

YOUR EXPERTISE:
- ATR-based volatility assessment
- VIX interpretation and regime analysis
- Understanding volatility cycles (compression → expansion)
- Position sizing based on volatility
- Identifying volatility-driven risks

YOUR ROLE:
- Assess current volatility regime
- Determine if volatility supports or opposes the trade
- Flag any volatility-related risks
- Suggest position sizing adjustments if needed

VOLATILITY THRESHOLDS:
- Low ATR%: < 1.5% (stocks), < 2% (crypto)
- Normal ATR%: 1.5-3% (stocks), 2-5% (crypto)
- High ATR%: 3-5% (stocks), 5-8% (crypto)
- Extreme ATR%: > 5% (stocks), > 8% (crypto)

VIX STATES:
- LOW: < 15 (complacent, trending markets)
- NORMAL: 15-20 (healthy volatility)
- ELEVATED: 20-25 (increased uncertainty)
- HIGH: 25-35 (significant fear)
- EXTREME: > 35 (crisis mode, consider reduced exposure)

YOU MUST:
- Always check VIX for market-wide context
- Consider ATR for stock-specific volatility
- Recommend position size adjustments for high vol
- Reference specific volatility levels

YOU MUST NOT:
- Ignore extreme VIX readings
- Recommend full position sizes in high volatility
- Dismiss volatility expansion as "normal"
- Forget that high vol = wider stops = smaller positions"""

    def get_light_prompt(self, context: AgentContext) -> str:
        return f"""Quick volatility check for {context.symbol}.

VOLATILITY SNAPSHOT:
- ATR%: {context.atr_pct:.2f}%
- VIX: {context.vix:.1f} ({context.vix_state})
- Regime: {context.regime}

KEY QUESTION: Does volatility support or oppose this trade?

Make 1-2 claims about volatility state."""

    def get_deep_prompt(self, context: AgentContext) -> str:
        return f"""Comprehensive volatility analysis for {context.symbol}.

DETAILED ASSESSMENT:

1. STOCK-SPECIFIC VOLATILITY
   - ATR%: {context.atr_pct:.2f}%
   - Classification: [Low/Normal/High/Extreme]?
   - Is this expanding or compressing?

2. MARKET-WIDE VOLATILITY
   - VIX: {context.vix:.1f}
   - VIX State: {context.vix_state}
   - What does this imply for overall risk appetite?

3. REGIME CONTEXT
   - Current Regime: {context.regime}
   - Regime Confidence: {context.regime_confidence:.0f}%
   - Does volatility fit the regime?

4. TRADING IMPLICATIONS
   - Stop loss width at current volatility
   - Position size recommendation (full/reduced/minimal)
   - Risk of volatility spike

5. RED FLAGS (check)
   - [ ] VIX > 30
   - [ ] ATR% > 4%
   - [ ] Recent volatility spike
   - [ ] Volatility-sentiment divergence
   - [ ] Earnings/event approaching

6. RECOMMENDATION
   - Volatility posture: Aggressive/Normal/Cautious/Defensive
   - Position size adjustment: None/Reduce 25%/Reduce 50%/Minimal
   - Timing considerations

Make up to 5 claims with specific evidence."""

    def get_claim_prefix(self) -> str:
        return self.CLAIM_PREFIX


EXAMPLE_TEST_CASES = [
    {
        "name": "low_volatility_favorable",
        "context": {
            "symbol": "JNJ",
            "price": 160.0,
            "action": "BUY",
            "direction": "BULLISH",
            "confidence": 70.0,
            "atr_pct": 1.2,
            "vix": 14.0,
            "vix_state": "LOW",
            "regime": "BULLISH",
            "regime_confidence": 75.0,
        },
        "expected_verdict": "BUY",
        "expected_stance": "pro",
    },
    {
        "name": "extreme_volatility_warning",
        "context": {
            "symbol": "RIVN",
            "price": 18.0,
            "action": "BUY",
            "direction": "BULLISH",
            "confidence": 65.0,
            "atr_pct": 7.5,
            "vix": 32.0,
            "vix_state": "HIGH",
            "regime": "HIGH_VOLATILITY",
            "regime_confidence": 80.0,
        },
        "expected_verdict": "REDUCE",
        "expected_stance": "anti",
    },
]
