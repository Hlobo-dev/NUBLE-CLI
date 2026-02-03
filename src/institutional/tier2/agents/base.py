"""
Base Agent Classes
===================

Core abstractions for all Tier 2 agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
import json
import hashlib


@dataclass
class AgentMetrics:
    """Performance metrics for an agent."""
    agent_name: str
    prompt_version: str = "1.0"
    
    # Execution metrics
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    
    # Timing
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    
    # Token usage
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    avg_output_tokens: float = 0.0
    
    # Quality metrics
    json_valid_rate: float = 1.0
    avg_claims_per_output: float = 0.0
    evidence_hit_rate: float = 1.0  # % of claims with valid evidence
    
    # Outcome correlation (from calibration)
    accuracy_score: float = 0.5  # 0-1 correlation with trade outcomes
    confidence_calibration: float = 0.5  # How well confidence predicts outcomes
    
    def record_run(
        self,
        success: bool,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        claims_count: int = 0,
        valid_evidence: int = 0,
        total_evidence: int = 0,
    ):
        """Record metrics from a single run."""
        self.total_runs += 1
        
        if success:
            self.successful_runs += 1
        else:
            self.failed_runs += 1
        
        # Update averages
        n = self.total_runs
        self.avg_latency_ms = ((n - 1) * self.avg_latency_ms + latency_ms) / n
        self.avg_output_tokens = ((n - 1) * self.avg_output_tokens + output_tokens) / n
        self.avg_claims_per_output = ((n - 1) * self.avg_claims_per_output + claims_count) / n
        
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        if total_evidence > 0:
            run_evidence_rate = valid_evidence / total_evidence
            self.evidence_hit_rate = ((n - 1) * self.evidence_hit_rate + run_evidence_rate) / n
    
    def to_dict(self) -> Dict:
        return {
            "agent_name": self.agent_name,
            "prompt_version": self.prompt_version,
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "success_rate": self.successful_runs / max(1, self.total_runs),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "avg_output_tokens": round(self.avg_output_tokens, 1),
            "avg_claims_per_output": round(self.avg_claims_per_output, 2),
            "evidence_hit_rate": round(self.evidence_hit_rate, 3),
            "accuracy_score": round(self.accuracy_score, 3),
        }


@dataclass
class AgentContext:
    """
    Context provided to each agent for analysis.
    
    Contains all data the agent needs to make claims.
    Agents should ONLY reference data from this context.
    """
    # Core decision pack
    symbol: str
    price: float
    action: str
    direction: str
    confidence: float
    
    # Technical indicators
    rsi: float = 50.0
    macd_value: float = 0.0
    macd_signal: float = 0.0
    trend_state: str = "NEUTRAL"
    sma_20: float = 0.0
    sma_50: float = 0.0
    sma_200: float = 0.0
    atr_pct: float = 2.0
    
    # Regime context
    regime: str = "NEUTRAL"
    regime_confidence: float = 50.0
    vix: float = 20.0
    vix_state: str = "NORMAL"
    
    # Sentiment
    sentiment_score: float = 0.0
    news_count_7d: int = 0
    
    # Multi-timeframe signals
    weekly_signal: Optional[Dict] = None
    daily_signal: Optional[Dict] = None
    h4_signal: Optional[Dict] = None
    
    # Portfolio context
    current_position: float = 0.0
    sector_exposure_pct: float = 0.0
    portfolio_heat: float = 0.0
    
    # Data quality
    data_age_seconds: float = 0.0
    missing_feeds: List[str] = field(default_factory=list)
    
    # Escalation context
    escalation_reasons: List[str] = field(default_factory=list)
    is_deep_round: bool = False
    prior_agent_outputs: List[Dict] = field(default_factory=list)
    
    def to_prompt_string(self) -> str:
        """Convert context to a formatted string for prompts."""
        lines = [
            f"Symbol: {self.symbol}",
            f"Current Price: ${self.price:.2f}",
            f"Tier 1 Action: {self.action}",
            f"Tier 1 Confidence: {self.confidence:.0f}%",
            f"Tier 1 Direction: {self.direction}",
            "",
            "=== TECHNICAL ===",
            f"RSI (14): {self.rsi:.1f}",
            f"MACD: {self.macd_value:.4f} (Signal: {self.macd_signal:.4f})",
            f"Trend State: {self.trend_state}",
            f"SMA Stack: 20={self.sma_20:.2f}, 50={self.sma_50:.2f}, 200={self.sma_200:.2f}",
            f"ATR %: {self.atr_pct:.2f}%",
            "",
            "=== REGIME ===",
            f"Market Regime: {self.regime} (Confidence: {self.regime_confidence:.0f}%)",
            f"VIX: {self.vix:.1f} ({self.vix_state})",
            "",
            "=== SENTIMENT ===",
            f"Sentiment Score: {self.sentiment_score:.2f}",
            f"News Count (7d): {self.news_count_7d}",
            "",
            "=== SIGNALS ===",
        ]
        
        if self.weekly_signal:
            lines.append(f"Weekly: {self._format_signal(self.weekly_signal)}")
        if self.daily_signal:
            lines.append(f"Daily: {self._format_signal(self.daily_signal)}")
        if self.h4_signal:
            lines.append(f"4H: {self._format_signal(self.h4_signal)}")
        
        if self.escalation_reasons:
            lines.extend([
                "",
                "=== ESCALATION REASONS ===",
                ", ".join(self.escalation_reasons),
            ])
        
        if self.current_position > 0 or self.sector_exposure_pct > 0:
            lines.extend([
                "",
                "=== PORTFOLIO ===",
                f"Current Position: {self.current_position:.2f}%",
                f"Sector Exposure: {self.sector_exposure_pct:.2f}%",
            ])
        
        if self.data_age_seconds > 60 or self.missing_feeds:
            lines.extend([
                "",
                "=== DATA QUALITY ===",
                f"Data Age: {self.data_age_seconds:.0f}s",
                f"Missing Feeds: {', '.join(self.missing_feeds) or 'None'}",
            ])
        
        return "\n".join(lines)
    
    def _format_signal(self, signal: Dict) -> str:
        if not signal:
            return "N/A"
        action = signal.get("action", "N/A")
        conf = signal.get("confidence", 0)
        return f"{action} ({conf:.0f}%)"
    
    def get_evidence_keys(self) -> set:
        """Get all valid evidence keys for claim validation."""
        keys = {
            "symbol", "price", "action", "direction", "confidence",
            "rsi", "macd_value", "macd_signal", "trend_state",
            "sma_20", "sma_50", "sma_200", "atr_pct",
            "regime", "regime_confidence", "vix", "vix_state",
            "sentiment_score", "news_count_7d",
            "current_position", "sector_exposure_pct",
            "data_age_seconds", "missing_feeds",
        }
        
        if self.weekly_signal:
            keys.add("weekly_signal")
        if self.daily_signal:
            keys.add("daily_signal")
        if self.h4_signal:
            keys.add("h4_signal")
        
        return keys
    
    @classmethod
    def from_tier1_pack(cls, pack) -> "AgentContext":
        """Create context from Tier1DecisionPack."""
        return cls(
            symbol=pack.symbol,
            price=pack.price,
            action=pack.action,
            direction=pack.direction,
            confidence=pack.confidence,
            rsi=pack.rsi,
            macd_value=pack.macd_value,
            macd_signal=pack.macd_signal,
            trend_state=pack.trend_state,
            sma_20=pack.sma_20,
            sma_50=pack.sma_50,
            sma_200=pack.sma_200,
            atr_pct=pack.atr_pct,
            regime=pack.regime,
            regime_confidence=pack.regime_confidence,
            vix=pack.vix,
            vix_state=pack.vix_state,
            sentiment_score=pack.sentiment_score,
            news_count_7d=pack.news_count_7d,
            weekly_signal=pack.weekly_signal,
            daily_signal=pack.daily_signal,
            h4_signal=pack.h4_signal,
            current_position=pack.current_position,
            sector_exposure_pct=pack.sector_exposure_pct,
            data_age_seconds=pack.data_age_seconds,
            missing_feeds=pack.missing_feeds,
        )


class BaseAgent(ABC):
    """
    Base class for all Tier 2 expert agents.
    
    Override Points:
    ----------------
    - name: Unique agent identifier
    - description: What this agent does
    - PROMPT_VERSION: Current prompt version
    - get_system_prompt(): Agent's role and expertise
    - get_light_prompt(): Quick analysis prompt
    - get_deep_prompt(): Detailed analysis prompt
    - validate_output(): Custom output validation
    
    Key Principles:
    ---------------
    1. Evidence-bounded: All claims must reference valid evidence keys
    2. Budgeted: Respect token limits
    3. Deterministic: Same input → consistent output style
    4. Auditable: Every claim has clear reasoning
    """
    
    # Override these in subclasses
    name: str = "base_agent"
    description: str = "Base agent - override in subclass"
    category: str = "base"  # technical, market, risk, quality, adversarial
    
    # Prompt versioning for A/B testing
    PROMPT_VERSION: str = "1.0"
    
    # Token limits
    LIGHT_MAX_TOKENS: int = 350
    DEEP_MAX_TOKENS: int = 2500
    
    # Output schema (shared across all agents)
    OUTPUT_SCHEMA = '''
{
  "agent": "<agent_name>",
  "mode": "light|deep",
  "verdict": "BUY|SELL|WAIT|NEUTRAL|REDUCE|VETO",
  "confidence": 0.0-1.0,
  "claims": [
    {
      "id": "<PREFIX>_01",
      "type": "risk|support|timing|data_quality|regime|correlation",
      "stance": "pro|anti|neutral",
      "strength": 0.0-1.0,
      "statement": "One clear sentence",
      "evidence_keys": ["key1", "key2"]
    }
  ],
  "recommended_deltas": {
    "confidence_delta": -25 to +5,
    "position_pct_cap": null or 0.0-5.0,
    "wait_minutes": 0-240,
    "risk_posture": "aggressive|normal|cautious|defensive|no_trade"
  }
}
'''
    
    CRITICAL_RULES = '''
CRITICAL RULES:
1. Output ONLY valid JSON - no markdown, no explanation
2. Every claim MUST reference valid evidence_keys from the data provided
3. Claims without valid evidence_keys will be DISCARDED
4. Maximum 5 claims per output
5. confidence_delta cannot exceed +5 (rarely increase confidence)
'''
    
    def __init__(self):
        self.metrics = AgentMetrics(
            agent_name=self.name,
            prompt_version=self.PROMPT_VERSION,
        )
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt that defines this agent's expertise.
        
        Should include:
        - Agent's role and expertise
        - What it should focus on
        - What it should NOT do
        """
        pass
    
    @abstractmethod
    def get_light_prompt(self, context: AgentContext) -> str:
        """
        Get prompt for light round analysis (150-350 tokens output).
        
        Light round should:
        - Quickly identify key concerns
        - Make 1-3 focused claims
        - Flag if deep analysis needed
        """
        pass
    
    @abstractmethod
    def get_deep_prompt(self, context: AgentContext) -> str:
        """
        Get prompt for deep round analysis (1000-2500 tokens output).
        
        Deep round should:
        - Thorough analysis of all relevant factors
        - Up to 5 detailed claims
        - Consider edge cases
        - Review prior agent outputs if available
        """
        pass
    
    def get_full_prompt(self, context: AgentContext, is_deep: bool = False) -> str:
        """Get complete prompt with context and schema."""
        context_str = context.to_prompt_string()
        
        if is_deep:
            analysis_prompt = self.get_deep_prompt(context)
            mode = "deep"
            max_tokens = self.DEEP_MAX_TOKENS
        else:
            analysis_prompt = self.get_light_prompt(context)
            mode = "light"
            max_tokens = self.LIGHT_MAX_TOKENS
        
        return f"""
{analysis_prompt}

=== MARKET DATA ===
{context_str}

=== REQUIRED OUTPUT FORMAT ===
{self.OUTPUT_SCHEMA}

{self.CRITICAL_RULES}

Valid evidence_keys you can reference: {sorted(context.get_evidence_keys())}

Mode: {mode} (max {max_tokens} tokens)
"""
    
    def validate_output(self, output: Dict) -> tuple[bool, List[str]]:
        """
        Validate agent output.
        
        Returns:
            (is_valid, list_of_errors)
        """
        errors = []
        
        # Check required fields
        required = ["agent", "verdict", "confidence", "claims"]
        for field in required:
            if field not in output:
                errors.append(f"Missing required field: {field}")
        
        # Validate confidence range
        if "confidence" in output:
            conf = output["confidence"]
            if not (0.0 <= conf <= 1.0):
                errors.append(f"Confidence {conf} out of range [0, 1]")
        
        # Validate claims
        claims = output.get("claims", [])
        if len(claims) > 5:
            errors.append(f"Too many claims: {len(claims)} (max 5)")
        
        for i, claim in enumerate(claims):
            if not claim.get("evidence_keys"):
                errors.append(f"Claim {i+1} has no evidence_keys")
            if not claim.get("statement"):
                errors.append(f"Claim {i+1} has no statement")
        
        return len(errors) == 0, errors
    
    def get_claim_prefix(self) -> str:
        """Get the prefix for claim IDs (e.g., 'MTF' for MTFDominance)."""
        # Default: first 3 chars of name uppercased
        return self.name[:3].upper()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', version='{self.PROMPT_VERSION}')"
