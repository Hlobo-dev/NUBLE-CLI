"""
Tier 2 Expert Agents
=====================

The Council of Experts - MVP 12 agents covering most decision surface.

Each agent:
- Runs light (150-350 tokens) by default
- Runs deep (1000-2500 tokens) when activated
- Produces JSON-only claims
- References specific evidence keys
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json

from .schemas import (
    AgentOutput,
    Claim,
    RecommendedDeltas,
    Tier1DecisionPack,
    ClaimType,
    ClaimStance,
    Verdict,
)


@dataclass
class AgentContext:
    """Context provided to each agent."""
    tier1_pack: Tier1DecisionPack
    portfolio_snapshot: Optional[Dict] = None
    escalation_reasons: List[str] = None
    is_deep: bool = False
    
    def to_prompt_context(self) -> str:
        """Convert to prompt-friendly format."""
        pack = self.tier1_pack
        
        lines = [
            f"Symbol: {pack.symbol}",
            f"Current Price: ${pack.price:.2f}",
            f"Tier 1 Action: {pack.action}",
            f"Tier 1 Confidence: {pack.confidence:.0f}%",
            f"Tier 1 Direction: {pack.direction}",
            "",
            "=== TECHNICAL ===",
            f"RSI (14): {pack.rsi:.1f}",
            f"MACD: {pack.macd_value:.4f} (Signal: {pack.macd_signal:.4f})",
            f"Trend State: {pack.trend_state}",
            f"SMA Stack: 20={pack.sma_20:.2f}, 50={pack.sma_50:.2f}, 200={pack.sma_200:.2f}",
            f"ATR %: {pack.atr_pct:.2f}%",
            "",
            "=== REGIME ===",
            f"Market Regime: {pack.regime} (Confidence: {pack.regime_confidence:.0f}%)",
            f"VIX: {pack.vix:.1f} ({pack.vix_state})",
            "",
            "=== SENTIMENT ===",
            f"Sentiment Score: {pack.sentiment_score:.2f}",
            f"News Count (7d): {pack.news_count_7d}",
            "",
            "=== SIGNALS ===",
        ]
        
        if pack.weekly_signal:
            lines.append(f"Weekly: {pack.weekly_signal.get('action', 'N/A')}")
        if pack.daily_signal:
            lines.append(f"Daily: {pack.daily_signal.get('action', 'N/A')}")
        if pack.h4_signal:
            lines.append(f"4H: {pack.h4_signal.get('action', 'N/A')}")
        
        if self.escalation_reasons:
            lines.extend([
                "",
                "=== ESCALATION REASONS ===",
                ", ".join(self.escalation_reasons),
            ])
        
        if self.portfolio_snapshot:
            lines.extend([
                "",
                "=== PORTFOLIO ===",
                f"Current Position: {pack.current_position:.2f}%",
                f"Sector Exposure: {pack.sector_exposure_pct:.2f}%",
            ])
        
        return "\n".join(lines)


class BaseAgent(ABC):
    """
    Base class for all Tier 2 expert agents.
    
    Each agent must implement:
    - get_system_prompt(): The system prompt for the agent
    - get_light_prompt(): Prompt for light round (150-350 tokens)
    - get_deep_prompt(): Prompt for deep round (1000-2500 tokens)
    """
    
    name: str = "base_agent"
    description: str = "Base agent"
    
    # Token limits
    light_max_tokens: int = 350
    deep_max_tokens: int = 2500
    
    # Output schema reminder (included in all prompts)
    OUTPUT_SCHEMA = '''
Output ONLY valid JSON in this exact format:
{
  "agent": "<your_agent_name>",
  "mode": "light|deep",
  "verdict": "BUY|SELL|WAIT|NEUTRAL|REDUCE|VETO",
  "confidence": 0.0-1.0,
  "claims": [
    {
      "id": "<AGENT_PREFIX>_01",
      "type": "risk|support|timing|data_quality|regime|correlation",
      "stance": "pro|anti|neutral",
      "strength": 0.0-1.0,
      "statement": "One clear sentence about the claim",
      "evidence_keys": ["key1", "key2"],
      "conditions": ["optional condition that triggers this claim"]
    }
  ],
  "recommended_deltas": {
    "confidence_delta": -25 to +5,
    "position_pct_cap": null or 0.0-5.0,
    "wait_minutes": 0-240,
    "risk_posture": "aggressive|normal|cautious|defensive|no_trade"
  }
}

CRITICAL RULES:
1. Output ONLY JSON - no markdown, no explanation
2. Every claim MUST reference valid evidence_keys from the data provided
3. Claims without valid evidence_keys will be DISCARDED
4. Maximum 5 claims per output
5. confidence_delta cannot exceed +5 (rarely increase confidence)
'''
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass
    
    @abstractmethod
    def get_light_prompt(self, context: AgentContext) -> str:
        """Get the light round prompt (150-350 tokens output)."""
        pass
    
    @abstractmethod
    def get_deep_prompt(self, context: AgentContext) -> str:
        """Get the deep round prompt (1000-2500 tokens output)."""
        pass
    
    def get_prompt(self, context: AgentContext) -> str:
        """Get appropriate prompt based on context."""
        if context.is_deep:
            return self.get_deep_prompt(context)
        return self.get_light_prompt(context)


# =============================================================================
# CORE TECHNICAL AGENTS
# =============================================================================

class MTFDominanceAgent(BaseAgent):
    """
    Multi-Timeframe Dominance Agent
    
    Analyzes the hierarchy of timeframe signals.
    Weekly > Daily > 4H in terms of weight.
    """
    
    name = "mtf_dominance"
    description = "Analyzes multi-timeframe signal hierarchy and alignment"
    
    def get_system_prompt(self) -> str:
        return """You are the MTF (Multi-Timeframe) Dominance expert.

Your role is to analyze the HIERARCHY of timeframe signals:
- Weekly signals have HIGHEST weight (strategic direction)
- Daily signals have MEDIUM weight (tactical direction)  
- 4H signals have LOWEST weight (entry timing)

Key principles:
1. NEVER trade against Weekly direction
2. Daily confirms Weekly
3. 4H is for entry timing only
4. Conflicts between timeframes = uncertainty = WAIT

You produce JSON claims about timeframe alignment and dominance."""
    
    def get_light_prompt(self, context: AgentContext) -> str:
        return f"""Analyze multi-timeframe alignment for this trade.

{context.to_prompt_context()}

Produce 1-3 claims about:
1. Weekly dominance (does it support or oppose the trade?)
2. Alignment score (all aligned, partially, conflicting?)
3. Any veto-level conflicts

{self.OUTPUT_SCHEMA}"""
    
    def get_deep_prompt(self, context: AgentContext) -> str:
        return f"""Deep analysis of multi-timeframe dynamics.

{context.to_prompt_context()}

Analyze:
1. Weekly signal quality and recency
2. Daily confirmation or divergence
3. 4H timing appropriateness
4. Historical pattern of current alignment
5. Recommended action based on MTF analysis

Produce 3-5 detailed claims with strong evidence.

{self.OUTPUT_SCHEMA}"""


class TrendIntegrityAgent(BaseAgent):
    """
    Trend Integrity Agent
    
    Assesses the health and sustainability of the current trend.
    """
    
    name = "trend_integrity"
    description = "Assesses trend health, momentum, and sustainability"
    
    def get_system_prompt(self) -> str:
        return """You are the Trend Integrity expert.

Your role is to assess:
- Is the trend healthy or exhausted?
- Is momentum confirming or diverging?
- Are there signs of trend breakdown?

Key indicators you focus on:
1. SMA stack alignment (20 > 50 > 200 for uptrend)
2. Price position relative to SMAs
3. MACD momentum and histogram
4. RSI divergences
5. Volume confirmation (if available)

You produce JSON claims about trend quality."""
    
    def get_light_prompt(self, context: AgentContext) -> str:
        return f"""Quick trend integrity check.

{context.to_prompt_context()}

Assess:
1. Is the SMA stack supporting the trade direction?
2. Is momentum (MACD) confirming?
3. Any warning signs (divergences, exhaustion)?

Produce 1-3 claims.

{self.OUTPUT_SCHEMA}"""
    
    def get_deep_prompt(self, context: AgentContext) -> str:
        return f"""Deep trend integrity analysis.

{context.to_prompt_context()}

Comprehensive assessment:
1. SMA stack health and alignment
2. MACD histogram momentum (accelerating/decelerating)
3. RSI divergence check
4. Trend age and potential exhaustion
5. Key support/resistance levels implied by SMAs

Produce 3-5 claims with specific evidence.

{self.OUTPUT_SCHEMA}"""


class ReversalPullbackAgent(BaseAgent):
    """
    Reversal vs Pullback Agent
    
    Distinguishes between healthy pullbacks and potential reversals.
    """
    
    name = "reversal_pullback"
    description = "Distinguishes pullbacks from reversals"
    
    def get_system_prompt(self) -> str:
        return """You are the Reversal vs Pullback expert.

Your critical role is to determine:
- Is current price action a HEALTHY PULLBACK (buying opportunity)?
- Or is it the START OF A REVERSAL (danger)?

Key factors:
1. RSI levels (oversold in uptrend = pullback opportunity)
2. Price relative to key SMAs (pullback to 20/50 SMA = healthy)
3. MACD momentum (slowing vs crossing)
4. Volume patterns
5. ATR expansion (reversal often sees vol spike)

You produce JSON claims classifying the current move."""
    
    def get_light_prompt(self, context: AgentContext) -> str:
        return f"""Quick pullback vs reversal assessment.

{context.to_prompt_context()}

Classify:
1. Is current price action a pullback or reversal?
2. What's the key evidence?
3. Risk level of entering here?

Produce 1-3 claims.

{self.OUTPUT_SCHEMA}"""
    
    def get_deep_prompt(self, context: AgentContext) -> str:
        return f"""Deep pullback vs reversal analysis.

{context.to_prompt_context()}

Analyze:
1. Price behavior relative to key SMAs
2. Momentum exhaustion signals
3. Volume characteristics
4. Historical pullback patterns for this regime
5. Probability estimate: pullback vs reversal

Produce 3-5 claims with quantified evidence.

{self.OUTPUT_SCHEMA}"""


# =============================================================================
# VOLATILITY / REGIME AGENTS
# =============================================================================

class VolatilityStateAgent(BaseAgent):
    """
    Volatility State Agent
    
    Monitors ATR, VIX, and volatility regime.
    """
    
    name = "volatility_state"
    description = "Monitors volatility regime and risk levels"
    
    def get_system_prompt(self) -> str:
        return """You are the Volatility State expert.

Your role is to assess:
- Current volatility regime (low/normal/high/extreme)
- VIX level and its implications
- ATR-based position sizing guidance
- Gap risk and overnight risk

Key thresholds:
- VIX < 15: Low vol (complacency risk)
- VIX 15-20: Normal
- VIX 20-25: Elevated
- VIX 25-30: High (reduce size)
- VIX > 30: Extreme (defensive)
- VIX > 40: VETO territory

ATR % thresholds:
- < 1.5%: Low vol
- 1.5-3%: Normal
- 3-5%: High
- > 5%: Extreme

You produce JSON claims about volatility implications."""
    
    def get_light_prompt(self, context: AgentContext) -> str:
        return f"""Quick volatility assessment.

{context.to_prompt_context()}

Assess:
1. Current vol regime (VIX + ATR)
2. Position sizing implication
3. Any vol-based warnings?

Produce 1-3 claims.

{self.OUTPUT_SCHEMA}"""
    
    def get_deep_prompt(self, context: AgentContext) -> str:
        return f"""Deep volatility analysis.

{context.to_prompt_context()}

Analyze:
1. VIX absolute level and recent change
2. ATR regime and trend
3. Vol-adjusted position sizing recommendation
4. Gap/overnight risk assessment
5. Historical vol patterns for current regime

Produce 3-5 claims with specific thresholds.

{self.OUTPUT_SCHEMA}"""


class RegimeTransitionAgent(BaseAgent):
    """
    Regime Transition Agent
    
    Detects regime changes and transition risks.
    """
    
    name = "regime_transition"
    description = "Detects market regime transitions"
    
    def get_system_prompt(self) -> str:
        return """You are the Regime Transition expert.

Your role is to assess:
- Current regime stability (BULL/BEAR/SIDEWAYS/VOLATILE)
- Signs of regime transition
- Fragility indicators
- Recommended action during transitions

Key signals of regime change:
1. VIX spike (>20% in 1 day)
2. SMA crossovers (death cross, golden cross)
3. Trend state degradation
4. Sentiment divergence from price

You produce JSON claims about regime stability."""
    
    def get_light_prompt(self, context: AgentContext) -> str:
        return f"""Quick regime stability check.

{context.to_prompt_context()}

Assess:
1. How stable is the current regime?
2. Any transition signals?
3. Risk level of regime change?

Produce 1-3 claims.

{self.OUTPUT_SCHEMA}"""
    
    def get_deep_prompt(self, context: AgentContext) -> str:
        return f"""Deep regime transition analysis.

{context.to_prompt_context()}

Analyze:
1. Regime confidence and stability
2. Leading indicators of transition
3. Historical regime duration context
4. Fragility assessment
5. Action recommendation based on regime risk

Produce 3-5 claims with evidence.

{self.OUTPUT_SCHEMA}"""


# =============================================================================
# EVENTS / NARRATIVE AGENTS
# =============================================================================

class EventWindowAgent(BaseAgent):
    """
    Event Window Agent
    
    Monitors proximity to earnings, macro events, and other catalysts.
    """
    
    name = "event_window"
    description = "Monitors event proximity and risk"
    
    def get_system_prompt(self) -> str:
        return """You are the Event Window expert.

Your role is to identify:
- Proximity to earnings (within 48h = danger zone)
- Macro events (FOMC, CPI, NFP)
- Corporate events (guidance, M&A)
- Event volatility premiums

Key rules:
1. Within 48h of earnings = REDUCE SIZE or WAIT
2. FOMC day = elevated vol expected
3. Post-event = volatility crush opportunity
4. Unknown events = be defensive

You produce JSON claims about event risk."""
    
    def get_light_prompt(self, context: AgentContext) -> str:
        return f"""Quick event window check.

{context.to_prompt_context()}

Assess:
1. Any imminent events (earnings, macro)?
2. Current event risk level?
3. Recommended action?

Produce 1-3 claims.

{self.OUTPUT_SCHEMA}"""
    
    def get_deep_prompt(self, context: AgentContext) -> str:
        return f"""Deep event window analysis.

{context.to_prompt_context()}

Analyze:
1. Earnings proximity and expected move
2. Macro calendar impact
3. Implied vol premium (if visible in ATR)
4. Historical event patterns
5. Optimal entry timing around events

Produce 3-5 claims with specific timing.

{self.OUTPUT_SCHEMA}"""


class RedTeamAgent(BaseAgent):
    """
    Red Team Agent
    
    Devil's advocate - finds the best argument AGAINST the trade.
    """
    
    name = "red_team"
    description = "Devils advocate - argues against the trade"
    
    def get_system_prompt(self) -> str:
        return """You are the Red Team (Devil's Advocate) expert.

Your CRITICAL role is to find the BEST ARGUMENT AGAINST the proposed trade.

You are not trying to be balanced. You are trying to:
1. Find the fatal flaw
2. Identify hidden risks
3. Challenge overconfidence
4. Prevent catastrophic losses

Your job is to save the system from itself.

Key areas to attack:
1. Is confidence too high given the data?
2. What's the worst-case scenario?
3. What risk is being ignored?
4. Is there contradictory evidence?
5. What would make this trade fail?

You produce JSON claims that are primarily ANTI (opposing the trade)."""
    
    def get_light_prompt(self, context: AgentContext) -> str:
        return f"""Find the best argument AGAINST this trade.

{context.to_prompt_context()}

Your job: Challenge this trade. Find the flaw.

Produce 1-3 ANTI claims.

{self.OUTPUT_SCHEMA}"""
    
    def get_deep_prompt(self, context: AgentContext) -> str:
        return f"""Deep red team analysis - tear this trade apart.

{context.to_prompt_context()}

Attack from every angle:
1. What's wrong with the technical setup?
2. What risk is being underestimated?
3. Why might sentiment be misleading?
4. What if the regime is about to change?
5. What's the catastrophic scenario?

Produce 3-5 strong ANTI claims. Be aggressive.

{self.OUTPUT_SCHEMA}"""


# =============================================================================
# GOVERNANCE AGENTS
# =============================================================================

class RiskGatekeeperAgent(BaseAgent):
    """
    Risk Gatekeeper Agent
    
    ABSOLUTE VETO POWER. Always runs deep.
    """
    
    name = "risk_gatekeeper"
    description = "Final risk check with VETO power"
    deep_max_tokens = 3000
    
    def get_system_prompt(self) -> str:
        return """You are the RISK GATEKEEPER - the final line of defense.

YOU HAVE ABSOLUTE VETO POWER.

Your role is to ensure NO TRADE violates hard risk constraints:
1. VIX > 40 → VETO
2. ATR % > 5% → VETO or extreme size reduction
3. Position would exceed concentration limits → VETO
4. Multiple simultaneous risk factors → VETO
5. Data quality critical failures → VETO

You are the last check before execution.
Your verdict of VETO cannot be overridden.

You produce JSON with VETO or REDUCE verdicts when warranted."""
    
    def get_light_prompt(self, context: AgentContext) -> str:
        return f"""Risk gatekeeper check.

{context.to_prompt_context()}

Check all hard limits:
1. VIX level vs thresholds
2. Volatility regime
3. Position sizing sanity
4. Any veto-level risks?

Produce claims with VETO if warranted.

{self.OUTPUT_SCHEMA}"""
    
    def get_deep_prompt(self, context: AgentContext) -> str:
        return f"""COMPREHENSIVE risk gatekeeper analysis.

{context.to_prompt_context()}

You are the FINAL CHECK. Verify:
1. VIX: Is it above danger thresholds?
2. ATR: Is volatility extreme?
3. Concentration: Would this breach limits?
4. Correlation: Is portfolio risk elevated?
5. Data Quality: Is all critical data valid?
6. Regime: Is there fragility risk?
7. Events: Are we in a danger window?
8. Confidence: Is Tier 1 overconfident?

VETO if any hard limit is breached.
REDUCE if multiple soft limits are elevated.

Produce 3-5 claims. Be thorough - you're the last line.

{self.OUTPUT_SCHEMA}"""


class DataIntegrityAgent(BaseAgent):
    """
    Data Integrity Agent
    
    Validates data freshness, completeness, and quality.
    """
    
    name = "data_integrity"
    description = "Validates data quality and freshness"
    
    def get_system_prompt(self) -> str:
        return """You are the Data Integrity expert.

Your role is to ensure all data is:
1. FRESH - not stale (check data_age_seconds)
2. COMPLETE - no critical missing feeds
3. CONSISTENT - no internal contradictions
4. RELIABLE - from trusted sources

If data integrity is compromised, the decision should be WAIT or REDUCE.

Critical data (stale = problem):
- Price data > 60 seconds old
- Daily signals > 48 hours old
- Weekly signals > 7 days old

You produce JSON claims about data quality."""
    
    def get_light_prompt(self, context: AgentContext) -> str:
        return f"""Quick data integrity check.

{context.to_prompt_context()}

Check:
1. Data freshness (any stale feeds?)
2. Missing critical data?
3. Any inconsistencies?

Produce 1-3 claims.

{self.OUTPUT_SCHEMA}"""
    
    def get_deep_prompt(self, context: AgentContext) -> str:
        return f"""Deep data integrity analysis.

{context.to_prompt_context()}

Validate:
1. All data sources and their freshness
2. Signal recency across timeframes
3. Internal consistency of indicators
4. Missing feeds impact
5. Overall data quality score

Produce 3-5 claims with specific data quality metrics.

{self.OUTPUT_SCHEMA}"""


# =============================================================================
# OPTIONAL EARLY ADD AGENTS
# =============================================================================

class TimingAgent(BaseAgent):
    """
    Timing Agent
    
    Advises on optimal entry timing - WAIT vs enter now.
    """
    
    name = "timing"
    description = "Advises on entry timing"
    
    def get_system_prompt(self) -> str:
        return """You are the Timing expert.

Your role is to advise:
- Should we enter NOW or WAIT?
- If WAIT, for how long?
- What's the optimal entry condition?

Factors you consider:
1. Time of day (avoid first 30 min, last 30 min)
2. Day of week (Monday reversals, Friday exits)
3. Event proximity
4. Momentum state (catching falling knife?)
5. Volume patterns

You produce JSON claims with wait_minutes recommendations."""
    
    def get_light_prompt(self, context: AgentContext) -> str:
        return f"""Quick timing assessment.

{context.to_prompt_context()}

Advise:
1. Enter now or wait?
2. If wait, how long?
3. What to watch for?

Produce 1-3 claims.

{self.OUTPUT_SCHEMA}"""
    
    def get_deep_prompt(self, context: AgentContext) -> str:
        return f"""Deep timing analysis.

{context.to_prompt_context()}

Analyze:
1. Intraday timing considerations
2. Day-of-week patterns
3. Event calendar impact
4. Momentum entry optimization
5. Specific wait recommendation with rationale

Produce 3-5 claims with timing recommendations.

{self.OUTPUT_SCHEMA}"""


class ConcentrationAgent(BaseAgent):
    """
    Concentration Agent
    
    Monitors portfolio concentration and correlation.
    """
    
    name = "concentration"
    description = "Monitors portfolio concentration"
    
    def get_system_prompt(self) -> str:
        return """You are the Concentration expert.

Your role is to prevent:
1. Single position too large
2. Sector overexposure
3. Correlated position clusters
4. Portfolio heat too high

Thresholds:
- Single position: max 10%
- Sector: max 30%
- Correlated cluster: max 40%
- Total heat (beta-weighted): monitor

You produce JSON claims about concentration risk."""
    
    def get_light_prompt(self, context: AgentContext) -> str:
        return f"""Quick concentration check.

{context.to_prompt_context()}

Check:
1. Would this breach position limits?
2. Sector exposure concern?
3. Correlation with existing positions?

Produce 1-3 claims.

{self.OUTPUT_SCHEMA}"""
    
    def get_deep_prompt(self, context: AgentContext) -> str:
        return f"""Deep concentration analysis.

{context.to_prompt_context()}

Analyze:
1. Current position in context of portfolio
2. Sector exposure after this trade
3. Correlation with top holdings
4. Portfolio heat impact
5. Sizing recommendation based on concentration

Produce 3-5 claims with specific limits.

{self.OUTPUT_SCHEMA}"""


class LiquidityAgent(BaseAgent):
    """
    Liquidity Agent
    
    Monitors volume, spread, and slippage risk.
    """
    
    name = "liquidity"
    description = "Monitors liquidity and slippage risk"
    
    def get_system_prompt(self) -> str:
        return """You are the Liquidity expert.

Your role is to assess:
1. Is volume normal or abnormal?
2. Can we get in/out without excessive slippage?
3. Are there liquidity warning signs?

Key factors:
- Unusual volume (>2x average = watch)
- Time of day (lower liquidity at open/close)
- Asset-specific liquidity
- Position size vs daily volume

You produce JSON claims about liquidity risk."""
    
    def get_light_prompt(self, context: AgentContext) -> str:
        return f"""Quick liquidity check.

{context.to_prompt_context()}

Assess:
1. Is volume normal?
2. Any liquidity concerns?
3. Slippage risk level?

Produce 1-3 claims.

{self.OUTPUT_SCHEMA}"""
    
    def get_deep_prompt(self, context: AgentContext) -> str:
        return f"""Deep liquidity analysis.

{context.to_prompt_context()}

Analyze:
1. Volume relative to historical average
2. Unusual volume patterns
3. Time-of-day liquidity
4. Position size vs volume ratio
5. Slippage estimate and sizing impact

Produce 3-5 claims with specific liquidity metrics.

{self.OUTPUT_SCHEMA}"""


# =============================================================================
# AGENT FACTORY
# =============================================================================

def get_agent_class(name: str) -> type:
    """Get agent class by name."""
    agents = {
        "mtf_dominance": MTFDominanceAgent,
        "trend_integrity": TrendIntegrityAgent,
        "reversal_pullback": ReversalPullbackAgent,
        "volatility_state": VolatilityStateAgent,
        "regime_transition": RegimeTransitionAgent,
        "event_window": EventWindowAgent,
        "red_team": RedTeamAgent,
        "risk_gatekeeper": RiskGatekeeperAgent,
        "data_integrity": DataIntegrityAgent,
        "timing": TimingAgent,
        "concentration": ConcentrationAgent,
        "liquidity": LiquidityAgent,
    }
    return agents.get(name, BaseAgent)


def create_agent(name: str) -> BaseAgent:
    """Create an agent instance by name."""
    agent_class = get_agent_class(name)
    return agent_class()
