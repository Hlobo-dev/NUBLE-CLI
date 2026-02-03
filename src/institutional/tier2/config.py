"""
Tier 2 Configuration
=====================

All configuration for the Council-of-Experts Orchestrator.
Implements static weights with calibration hooks for future learning.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum


class EscalationReason(Enum):
    """Reasons for escalating to Tier 2."""
    # Signal quality issues
    SIGNAL_CONFLICT = "signal_conflict"  # Timeframes disagree
    LOW_CONFIDENCE = "low_confidence"    # Tier 1 confidence < threshold
    STALE_DATA = "stale_data"           # Data freshness issues
    
    # Market conditions
    HIGH_VOLATILITY = "high_volatility"  # VIX spike or ATR extreme
    REGIME_TRANSITION = "regime_transition"  # HMM detected shift
    EARNINGS_IMMINENT = "earnings_imminent"  # Within 48h of earnings
    
    # Portfolio concerns
    PORTFOLIO_CONCENTRATION = "portfolio_concentration"  # Sector/position limits
    CORRELATION_CLUSTER = "correlation_cluster"  # Highly correlated positions
    DRAWDOWN_ELEVATED = "drawdown_elevated"  # Portfolio DD > threshold
    
    # Risk flags
    NEWS_SENTIMENT_CONFLICT = "news_sentiment_conflict"  # TA vs sentiment
    UNUSUAL_VOLUME = "unusual_volume"  # Volume anomaly
    MACRO_EVENT = "macro_event"  # Fed, FOMC, etc.
    
    # Edge cases
    HIGH_CONFIDENCE_CHECK = "high_confidence_check"  # Verify overconfidence
    FIRST_TRADE_OF_DAY = "first_trade_of_day"  # Extra scrutiny


# Mapping of escalation reasons to agents that should run deep
ESCALATION_REASONS = {
    EscalationReason.SIGNAL_CONFLICT: [
        "mtf_dominance",
        "trend_integrity",
        "regime_transition",
        "red_team",
    ],
    EscalationReason.LOW_CONFIDENCE: [
        "mtf_dominance",
        "trend_integrity",
        "timing",
        "data_integrity",
    ],
    EscalationReason.STALE_DATA: [
        "data_integrity",
        "timing",
    ],
    EscalationReason.HIGH_VOLATILITY: [
        "volatility_state",
        "regime_transition",
        "timing",
        "risk_gatekeeper",
    ],
    EscalationReason.REGIME_TRANSITION: [
        "regime_transition",
        "volatility_state",
        "trend_integrity",
        "red_team",
    ],
    EscalationReason.EARNINGS_IMMINENT: [
        "event_window",
        "timing",
        "volatility_state",
        "risk_gatekeeper",
    ],
    EscalationReason.PORTFOLIO_CONCENTRATION: [
        "concentration",
        "risk_gatekeeper",
    ],
    EscalationReason.CORRELATION_CLUSTER: [
        "concentration",
        "risk_gatekeeper",
    ],
    EscalationReason.DRAWDOWN_ELEVATED: [
        "risk_gatekeeper",
        "volatility_state",
        "timing",
    ],
    EscalationReason.NEWS_SENTIMENT_CONFLICT: [
        "red_team",
        "data_integrity",
        "timing",
    ],
    EscalationReason.UNUSUAL_VOLUME: [
        "liquidity",
        "volatility_state",
        "red_team",
    ],
    EscalationReason.MACRO_EVENT: [
        "event_window",
        "volatility_state",
        "risk_gatekeeper",
    ],
    EscalationReason.HIGH_CONFIDENCE_CHECK: [
        "red_team",
        "reversal_pullback",
        "risk_gatekeeper",
    ],
    EscalationReason.FIRST_TRADE_OF_DAY: [
        "data_integrity",
        "timing",
        "volatility_state",
    ],
}


@dataclass
class AgentConfig:
    """Configuration for a single expert agent."""
    name: str
    enabled: bool = True
    
    # Token budgets
    light_max_tokens: int = 350
    deep_max_tokens: int = 2500
    
    # Timeouts
    light_timeout_ms: int = 2000
    deep_timeout_ms: int = 8000
    
    # Weight (static, will be learned later)
    weight: float = 1.0
    
    # Domain activation
    activation_reasons: List[str] = field(default_factory=list)
    
    # Flags
    always_run_deep: bool = False  # Risk gatekeeper always deep
    can_veto: bool = False  # Only risk gatekeeper can veto
    
    # Model selection
    bedrock_model_light: str = "anthropic.claude-3-haiku-20240307-v1:0"
    bedrock_model_deep: str = "anthropic.claude-3-5-sonnet-20241022-v2:0"


@dataclass
class Tier2Config:
    """
    Master configuration for Tier 2 CEO.
    
    This config controls:
    - Agent selection and budgets
    - Thresholds for escalation
    - Safety limits
    - Fallback behavior
    """
    
    # =========== AGENT COUNCIL ===========
    # Start with MVP 10-12 agents, expand later
    agents: Dict[str, AgentConfig] = field(default_factory=lambda: {
        # Core Technical
        "mtf_dominance": AgentConfig(
            name="mtf_dominance",
            weight=1.2,
            activation_reasons=["signal_conflict", "low_confidence"],
        ),
        "trend_integrity": AgentConfig(
            name="trend_integrity",
            weight=1.1,
            activation_reasons=["signal_conflict", "regime_transition"],
        ),
        "reversal_pullback": AgentConfig(
            name="reversal_pullback",
            weight=1.0,
            activation_reasons=["high_confidence_check"],
        ),
        
        # Volatility / Regime
        "volatility_state": AgentConfig(
            name="volatility_state",
            weight=1.15,
            activation_reasons=["high_volatility", "regime_transition", "earnings_imminent"],
        ),
        "regime_transition": AgentConfig(
            name="regime_transition",
            weight=1.1,
            activation_reasons=["regime_transition", "signal_conflict"],
        ),
        
        # Events / Narrative
        "event_window": AgentConfig(
            name="event_window",
            weight=1.2,
            activation_reasons=["earnings_imminent", "macro_event"],
        ),
        "red_team": AgentConfig(
            name="red_team",
            weight=1.25,
            activation_reasons=["high_confidence_check", "signal_conflict"],
        ),
        
        # Governance
        "risk_gatekeeper": AgentConfig(
            name="risk_gatekeeper",
            weight=2.0,  # Highest weight
            always_run_deep=True,
            can_veto=True,
            activation_reasons=[],  # Always activated
            deep_max_tokens=3000,
        ),
        "data_integrity": AgentConfig(
            name="data_integrity",
            weight=1.3,
            activation_reasons=["stale_data", "low_confidence", "first_trade_of_day"],
        ),
        
        # Optional early adds
        "timing": AgentConfig(
            name="timing",
            weight=1.0,
            activation_reasons=["earnings_imminent", "high_volatility", "first_trade_of_day"],
        ),
        "concentration": AgentConfig(
            name="concentration",
            weight=1.1,
            activation_reasons=["portfolio_concentration", "correlation_cluster"],
            enabled=True,  # Enable if portfolio tracking active
        ),
        "liquidity": AgentConfig(
            name="liquidity",
            weight=0.9,
            activation_reasons=["unusual_volume"],
            enabled=True,
        ),
    })
    
    # =========== ESCALATION THRESHOLDS ===========
    min_confidence_for_escalation: float = 0.75  # Below this, always escalate
    max_confidence_for_escalation: float = 0.92  # Above this, check overconfidence
    
    escalation_volatility_threshold: float = 30.0  # VIX level
    escalation_atr_pct_threshold: float = 3.5  # ATR as % of price
    
    earnings_window_hours: int = 48  # Hours before/after earnings
    
    # =========== BUDGET LIMITS ===========
    max_total_tokens_light: int = 4000  # All agents combined (light)
    max_total_tokens_deep: int = 20000  # All agents combined (deep)
    
    max_agents_deep: int = 8  # Max agents for deep dive
    min_agents_deep: int = 3  # Min agents for deep dive
    
    max_tier2_latency_ms: int = 15000  # 15 second hard limit
    
    # =========== VALIDATION THRESHOLDS ===========
    min_valid_json_rate: float = 0.70  # If < 70% valid JSON, abort Tier 2
    min_claims_per_agent: int = 1
    max_claims_per_agent: int = 5
    
    # =========== DELTA LIMITS ===========
    # Tier 2 can only apply limited adjustments
    max_confidence_increase: int = 5   # Rarely increase
    max_confidence_decrease: int = 25  # Can significantly decrease
    max_position_cap_pct: float = 3.0  # Max position % after delta
    min_position_cap_pct: float = 0.0  # Can reduce to zero
    max_wait_minutes: int = 240  # Max delay
    
    # =========== FALLBACK BEHAVIOR ===========
    fallback_on_timeout: bool = True  # Use Tier 1 if Tier 2 times out
    fallback_on_low_json_rate: bool = True  # Use Tier 1 if JSON invalid
    fallback_on_error: bool = True  # Use Tier 1 on any error
    
    # =========== CIRCUIT BREAKER ===========
    circuit_breaker_enabled: bool = True
    circuit_breaker_error_threshold: int = 5  # Errors before trip
    circuit_breaker_cooldown_seconds: int = 300  # 5 min cooldown
    
    # =========== BEDROCK CONFIG ===========
    bedrock_region: str = "us-east-1"
    bedrock_max_retries: int = 2
    
    # =========== CALIBRATION ===========
    # Phase 1: Static weights (current)
    # Phase 2: Regime-conditioned (future)
    # Phase 3: Learned weights (future)
    calibration_phase: int = 1
    calibration_min_trades: int = 100  # Min trades before learning
    
    # =========== LOGGING ===========
    log_all_claims: bool = True
    log_raw_llm_output: bool = False  # Privacy/cost
    store_decisions_dynamodb: bool = True
    
    def get_agent(self, name: str) -> Optional[AgentConfig]:
        """Get agent config by name."""
        return self.agents.get(name)
    
    def get_enabled_agents(self) -> List[AgentConfig]:
        """Get all enabled agents."""
        return [a for a in self.agents.values() if a.enabled]
    
    def get_agents_for_deep(
        self,
        escalation_reasons: List[str],
        conflict_agents: List[str] = None,
    ) -> List[str]:
        """
        Determine which agents should run deep based on escalation.
        
        Logic:
        1. Map escalation reasons to activated agents
        2. Add agents with material conflicts
        3. Always include risk_gatekeeper
        4. Respect max_agents_deep limit
        """
        deep_agents = set()
        
        # 1. Add agents activated by escalation reasons
        for reason in escalation_reasons:
            try:
                reason_enum = EscalationReason(reason)
                agents = ESCALATION_REASONS.get(reason_enum, [])
                deep_agents.update(agents)
            except ValueError:
                pass  # Unknown reason
        
        # 2. Add conflicting agents
        if conflict_agents:
            deep_agents.update(conflict_agents)
        
        # 3. Always include risk_gatekeeper
        deep_agents.add("risk_gatekeeper")
        
        # 4. Add always_run_deep agents
        for name, config in self.agents.items():
            if config.always_run_deep and config.enabled:
                deep_agents.add(name)
        
        # 5. Filter to enabled agents and respect limit
        enabled = {a.name for a in self.get_enabled_agents()}
        deep_agents = deep_agents.intersection(enabled)
        
        # Sort by weight (highest first) and limit
        weighted = sorted(
            deep_agents,
            key=lambda a: self.agents.get(a, AgentConfig(name=a)).weight,
            reverse=True
        )
        
        return weighted[:self.max_agents_deep]


# Default configuration instance
DEFAULT_CONFIG = Tier2Config()
