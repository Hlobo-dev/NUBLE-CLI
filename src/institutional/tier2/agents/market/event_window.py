"""
Event Window Agent
===================

Analyzes proximity to market-moving events.

Key Focus:
- Earnings announcements
- FOMC / Fed decisions
- Economic data releases
- Ex-dividend dates
- Product launches / conferences

Evidence Keys Used:
- sentiment_score
- news_count_7d
- (Future: earnings_date, fomc_date, etc.)

Claim Types:
- risk: Event proximity warning
- timing: Wait for event to pass
- data_quality: Event data availability
"""

from ..base import BaseAgent, AgentContext


class EventWindowAgent(BaseAgent):
    """
    Event Window Agent.
    
    Assesses proximity to market-moving events and their impact.
    """
    
    name = "event_window"
    description = "Analyzes proximity to market-moving events"
    category = "market"
    
    PROMPT_VERSION = "1.0"
    CLAIM_PREFIX = "EVT"
    
    def get_system_prompt(self) -> str:
        return """You are an event risk specialist who tracks market-moving catalysts.

YOUR EXPERTISE:
- Earnings announcement impact
- Fed/FOMC decision analysis
- Economic data release effects
- Corporate events (product launches, conferences)
- Event-driven volatility patterns

KEY EVENTS TO CONSIDER:
- Earnings: Usually avoid 48h before/24h after
- FOMC: Volatility spike around decisions
- CPI/Jobs: Market-wide reactions
- Splits/Dividends: Price adjustments

YOUR ROLE:
- Identify upcoming events that could impact the trade
- Assess event risk (size/timing/direction)
- Recommend waiting if event is imminent
- Flag high news activity as potential event signal

NEWS ACTIVITY SIGNALS:
- news_count_7d > 50: High activity, something brewing
- Sentiment extreme (< -0.5 or > 0.5): Strong narrative

YOU MUST:
- Consider news velocity as event proxy
- Flag earnings windows conservatively
- Recommend position reduction near events
- Reference specific evidence

YOU MUST NOT:
- Ignore high news counts
- Recommend large positions before earnings
- Dismiss event risk casually"""

    def get_light_prompt(self, context: AgentContext) -> str:
        return f"""Quick event risk check for {context.symbol}.

EVENT SIGNALS:
- News Count (7d): {context.news_count_7d}
- Sentiment Score: {context.sentiment_score:.2f}

KEY QUESTION: Any imminent events that could impact this trade?

Make 1-2 claims about event risk."""

    def get_deep_prompt(self, context: AgentContext) -> str:
        return f"""Comprehensive event risk analysis for {context.symbol}.

ANALYSIS:

1. NEWS ACTIVITY
   - News Count (7d): {context.news_count_7d}
   - Is this elevated? (Normal: 5-20, High: 20-50, Very High: 50+)
   - What does high activity suggest?

2. SENTIMENT CONTEXT
   - Score: {context.sentiment_score:.2f}
   - Extreme negative (< -0.5): Panic/crisis
   - Extreme positive (> 0.5): Euphoria/bubble
   - Neutral: No strong narrative

3. POTENTIAL EVENTS (infer from data)
   - High news + negative sentiment → earnings miss? scandal?
   - High news + positive sentiment → beat? catalyst?
   - VIX elevated ({context.vix:.1f}) → macro event?

4. EVENT RISK ASSESSMENT
   - Probability of imminent event: [Low/Medium/High]
   - Potential impact: [Minor/Moderate/Major]
   - Directional bias: [Bullish/Bearish/Unknown]

5. RECOMMENDATION
   - Safe to proceed?
   - Wait for event to pass?
   - Position size adjustment?

Make up to 5 claims with evidence."""

    def get_claim_prefix(self) -> str:
        return self.CLAIM_PREFIX
