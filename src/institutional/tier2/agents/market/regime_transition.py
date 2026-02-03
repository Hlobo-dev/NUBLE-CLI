"""
Regime Transition Agent
========================

Detects and analyzes market regime transitions.

Key Focus:
- HMM-based regime detection
- Regime stability assessment
- Transition timing and implications
- Bull/Bear/Sideways/HighVol states

Evidence Keys Used:
- regime
- regime_confidence
- vix
- vix_state
- trend_state

Claim Types:
- risk: Unstable regime, potential transition
- support: Stable regime aligned with trade
- regime: Regime context information
"""

from ..base import BaseAgent, AgentContext


class RegimeTransitionAgent(BaseAgent):
    """
    Regime Transition Agent.
    
    Analyzes market regime and detects potential transitions.
    """
    
    name = "regime_transition"
    description = "Detects market regime transitions and stability"
    category = "market"
    
    PROMPT_VERSION = "1.0"
    CLAIM_PREFIX = "REG"
    
    def get_system_prompt(self) -> str:
        return """You are a regime detection specialist using statistical methods.

YOUR EXPERTISE:
- Hidden Markov Model (HMM) regime interpretation
- Understanding regime persistence and transitions
- Assessing regime confidence and stability
- Cross-validating regime with VIX and trend data

REGIME TYPES:
- BULL: Trending up, low volatility, risk-on
- BEAR: Trending down, elevated volatility, risk-off
- SIDEWAYS: Range-bound, low directional edge
- HIGH_VOLATILITY: Extreme moves, reduced position sizes

YOUR ROLE:
- Validate the detected regime
- Assess regime stability (high confidence = stable)
- Detect potential regime transitions
- Flag trades that conflict with regime

REGIME CONFIDENCE INTERPRETATION:
- > 80%: Very stable, trust the regime
- 60-80%: Stable, proceed with normal sizing
- 40-60%: Unstable, reduce sizing or wait
- < 40%: Very unstable, high transition risk

YOU MUST:
- Check if regime aligns with trade direction
- Flag low confidence regimes as risks
- Consider VIX state as regime validation
- Reference specific regime data"""

    def get_light_prompt(self, context: AgentContext) -> str:
        return f"""Quick regime check for {context.symbol}.

REGIME STATE:
- Current: {context.regime}
- Confidence: {context.regime_confidence:.0f}%
- VIX: {context.vix:.1f} ({context.vix_state})
- Trade Direction: {context.direction}

KEY QUESTION: Does the regime support this trade?

Make 1-2 claims about regime stability."""

    def get_deep_prompt(self, context: AgentContext) -> str:
        return f"""Comprehensive regime analysis for {context.symbol}.

DETAILED ASSESSMENT:

1. CURRENT REGIME
   - Detected Regime: {context.regime}
   - Confidence: {context.regime_confidence:.0f}%
   - Classification: [Stable/Unstable/Transitioning]

2. REGIME-TRADE ALIGNMENT
   - Trade Direction: {context.direction}
   - Does regime support this direction?
   - Historical success rate in this regime?

3. VIX VALIDATION
   - VIX: {context.vix:.1f} ({context.vix_state})
   - Is VIX consistent with detected regime?
   - Any VIX-regime divergence?

4. TRANSITION RISK
   - Signs of regime change?
   - Recent volatility spikes?
   - Trend state: {context.trend_state}

5. TRADING IMPLICATIONS
   - Position sizing for this regime
   - Time horizon considerations
   - Exit strategy adjustments

Make up to 5 claims with evidence."""

    def get_claim_prefix(self) -> str:
        return self.CLAIM_PREFIX
