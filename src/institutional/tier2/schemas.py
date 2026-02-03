"""
Tier 2 Data Schemas
====================

Strict JSON schemas for the claims graph and decision outputs.
Every agent output MUST conform to these schemas.

Hard Rule: If JSON fails OR evidence keys don't exist → discard output.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import json


class ClaimType(Enum):
    """Types of claims an agent can make."""
    RISK = "risk"           # Risk-related claim
    SUPPORT = "support"     # Supports the trade
    TIMING = "timing"       # Timing-related
    DATA_QUALITY = "data_quality"  # Data integrity
    REGIME = "regime"       # Market regime context
    CORRELATION = "correlation"  # Cross-asset correlation


class ClaimStance(Enum):
    """Stance of a claim relative to the trade."""
    PRO = "pro"       # Supports the trade direction
    ANTI = "anti"     # Opposes the trade direction
    NEUTRAL = "neutral"  # Informational, no directional bias


class Verdict(Enum):
    """Agent verdict on the trade."""
    BUY = "BUY"
    SELL = "SELL"
    WAIT = "WAIT"
    NEUTRAL = "NEUTRAL"
    REDUCE = "REDUCE"
    VETO = "VETO"


class RiskPosture(Enum):
    """Overall risk posture recommendation."""
    AGGRESSIVE = "aggressive"  # Full position OK
    NORMAL = "normal"          # Standard sizing
    CAUTIOUS = "cautious"      # Reduced size
    DEFENSIVE = "defensive"    # Minimal exposure
    NO_TRADE = "no_trade"      # Exit/avoid


@dataclass
class Claim:
    """
    A single claim from an agent.
    
    This is the atomic unit of the claims graph.
    Every claim must have:
    - Unique ID within the agent's output
    - Type classification
    - Stance (pro/anti/neutral)
    - Strength score
    - Evidence keys that can be validated
    """
    id: str
    type: str  # ClaimType value
    stance: str  # ClaimStance value
    strength: float  # 0.0 to 1.0
    statement: str
    evidence_keys: List[str]
    conditions: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        # Validate strength
        self.strength = max(0.0, min(1.0, self.strength))
        
        # Ensure evidence_keys is a list
        if not isinstance(self.evidence_keys, list):
            self.evidence_keys = [str(self.evidence_keys)]
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.type,
            "stance": self.stance,
            "strength": round(self.strength, 3),
            "statement": self.statement,
            "evidence_keys": self.evidence_keys,
            "conditions": self.conditions,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Claim":
        return cls(
            id=data.get("id", "UNKNOWN"),
            type=data.get("type", "risk"),
            stance=data.get("stance", "neutral"),
            strength=float(data.get("strength", 0.5)),
            statement=data.get("statement", ""),
            evidence_keys=data.get("evidence_keys", []),
            conditions=data.get("conditions", []),
        )
    
    def validate(self, available_keys: set) -> bool:
        """Validate that evidence keys exist in available data."""
        if not self.evidence_keys:
            return False
        return all(key in available_keys for key in self.evidence_keys)


@dataclass
class RecommendedDeltas:
    """Delta recommendations from an agent."""
    confidence_delta: int = 0  # -25 to +5
    position_pct_cap: Optional[float] = None  # Max position %
    wait_minutes: int = 0  # Minutes to delay
    risk_posture: Optional[str] = None  # RiskPosture value
    
    def to_dict(self) -> Dict:
        result = {"confidence_delta": self.confidence_delta}
        if self.position_pct_cap is not None:
            result["position_pct_cap"] = self.position_pct_cap
        if self.wait_minutes > 0:
            result["wait_minutes"] = self.wait_minutes
        if self.risk_posture:
            result["risk_posture"] = self.risk_posture
        return result
    
    @classmethod
    def from_dict(cls, data: Dict) -> "RecommendedDeltas":
        return cls(
            confidence_delta=int(data.get("confidence_delta", 0)),
            position_pct_cap=data.get("position_pct_cap"),
            wait_minutes=int(data.get("wait_minutes", 0)),
            risk_posture=data.get("risk_posture"),
        )


@dataclass
class AgentOutput:
    """
    Complete output from a single agent.
    
    This is the MANDATORY format for all agent responses.
    If an agent fails to produce valid JSON conforming to this schema,
    its output is discarded.
    """
    agent: str
    mode: str  # "light" or "deep"
    verdict: str  # Verdict value
    confidence: float  # 0.0 to 1.0
    claims: List[Claim]
    recommended_deltas: RecommendedDeltas
    
    # Metadata
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    latency_ms: float = 0.0
    tokens_used: int = 0
    model: str = ""
    
    # Validation
    valid: bool = True
    validation_errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "agent": self.agent,
            "mode": self.mode,
            "verdict": self.verdict,
            "confidence": round(self.confidence, 3),
            "claims": [c.to_dict() for c in self.claims],
            "recommended_deltas": self.recommended_deltas.to_dict(),
            "timestamp": self.timestamp.isoformat(),
            "latency_ms": round(self.latency_ms, 1),
            "tokens_used": self.tokens_used,
            "model": self.model,
            "valid": self.valid,
            "validation_errors": self.validation_errors,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "AgentOutput":
        """Parse agent output from JSON dict."""
        claims = [Claim.from_dict(c) for c in data.get("claims", [])]
        deltas = RecommendedDeltas.from_dict(data.get("recommended_deltas", {}))
        
        return cls(
            agent=data.get("agent", "unknown"),
            mode=data.get("mode", "light"),
            verdict=data.get("verdict", "NEUTRAL"),
            confidence=float(data.get("confidence", 0.5)),
            claims=claims,
            recommended_deltas=deltas,
            latency_ms=data.get("latency_ms", 0),
            tokens_used=data.get("tokens_used", 0),
            model=data.get("model", ""),
        )
    
    @classmethod
    def from_json(cls, json_str: str, agent_name: str) -> "AgentOutput":
        """Parse agent output from JSON string with validation."""
        try:
            data = json.loads(json_str)
            output = cls.from_dict(data)
            output.agent = agent_name
            return output
        except json.JSONDecodeError as e:
            # Return invalid output
            return cls(
                agent=agent_name,
                mode="light",
                verdict="NEUTRAL",
                confidence=0.0,
                claims=[],
                recommended_deltas=RecommendedDeltas(),
                valid=False,
                validation_errors=[f"JSON parse error: {str(e)}"],
            )
    
    def validate_claims(self, available_keys: set) -> bool:
        """Validate all claims have existing evidence keys."""
        for claim in self.claims:
            if not claim.validate(available_keys):
                self.validation_errors.append(
                    f"Claim {claim.id}: evidence keys not found in data"
                )
                self.valid = False
        return self.valid


class AgentRound(Enum):
    """Which round an agent ran in."""
    LIGHT = "light"
    DEEP = "deep"


@dataclass
class ClaimsGraph:
    """
    The complete claims graph from all agents.
    
    This is the primary input to the Arbiter.
    Contains all valid claims, conflicts, and metadata.
    """
    decision_id: str
    symbol: str
    
    # All valid agent outputs by round
    light_outputs: Dict[str, AgentOutput] = field(default_factory=dict)
    deep_outputs: Dict[str, AgentOutput] = field(default_factory=dict)
    
    # Aggregated claims
    all_claims: List[Claim] = field(default_factory=list)
    pro_claims: List[Claim] = field(default_factory=list)
    anti_claims: List[Claim] = field(default_factory=list)
    neutral_claims: List[Claim] = field(default_factory=list)
    
    # Conflicts
    conflicts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Validation
    total_agents: int = 0
    valid_agents: int = 0
    json_valid_rate: float = 1.0
    
    def add_output(self, output: AgentOutput, round: AgentRound):
        """Add an agent output to the graph."""
        self.total_agents += 1
        
        if not output.valid:
            return
        
        self.valid_agents += 1
        
        # Store by round
        if round == AgentRound.LIGHT:
            self.light_outputs[output.agent] = output
        else:
            self.deep_outputs[output.agent] = output
        
        # Categorize claims
        for claim in output.claims:
            self.all_claims.append(claim)
            if claim.stance == ClaimStance.PRO.value:
                self.pro_claims.append(claim)
            elif claim.stance == ClaimStance.ANTI.value:
                self.anti_claims.append(claim)
            else:
                self.neutral_claims.append(claim)
        
        # Update JSON valid rate
        self.json_valid_rate = self.valid_agents / self.total_agents if self.total_agents > 0 else 0.0
    
    def detect_conflicts(self):
        """Detect conflicting claims between agents."""
        self.conflicts = []
        
        # Group claims by type
        claims_by_type: Dict[str, List[Claim]] = {}
        for claim in self.all_claims:
            if claim.type not in claims_by_type:
                claims_by_type[claim.type] = []
            claims_by_type[claim.type].append(claim)
        
        # Find conflicts (pro vs anti on same type)
        for claim_type, claims in claims_by_type.items():
            pros = [c for c in claims if c.stance == ClaimStance.PRO.value]
            antis = [c for c in claims if c.stance == ClaimStance.ANTI.value]
            
            for pro in pros:
                for anti in antis:
                    self.conflicts.append({
                        "type": claim_type,
                        "pro_claim": pro.to_dict(),
                        "anti_claim": anti.to_dict(),
                        "severity": abs(pro.strength - anti.strength),
                    })
    
    def get_weighted_sentiment(self, agent_weights: Dict[str, float]) -> float:
        """
        Calculate weighted sentiment from all claims.
        
        Returns: -1.0 (strong anti) to +1.0 (strong pro)
        """
        total_weight = 0.0
        weighted_sum = 0.0
        
        all_outputs = {**self.light_outputs, **self.deep_outputs}
        
        for agent_name, output in all_outputs.items():
            agent_weight = agent_weights.get(agent_name, 1.0)
            
            for claim in output.claims:
                claim_weight = agent_weight * claim.strength
                total_weight += claim_weight
                
                if claim.stance == ClaimStance.PRO.value:
                    weighted_sum += claim_weight
                elif claim.stance == ClaimStance.ANTI.value:
                    weighted_sum -= claim_weight
        
        if total_weight == 0:
            return 0.0
        
        return weighted_sum / total_weight
    
    def to_dict(self) -> Dict:
        return {
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "light_outputs": {k: v.to_dict() for k, v in self.light_outputs.items()},
            "deep_outputs": {k: v.to_dict() for k, v in self.deep_outputs.items()},
            "pro_claims_count": len(self.pro_claims),
            "anti_claims_count": len(self.anti_claims),
            "neutral_claims_count": len(self.neutral_claims),
            "conflicts_count": len(self.conflicts),
            "json_valid_rate": round(self.json_valid_rate, 3),
        }


@dataclass
class Tier2Delta:
    """
    The delta adjustments from Tier 2 to apply to Tier 1 decision.
    
    This is the ONLY output type from Tier 2.
    """
    # Direction change
    final_direction: str  # "BUY", "SELL", "WAIT", "NO_TRADE"
    
    # Confidence adjustment
    confidence_delta: int  # -25 to +5
    
    # Position sizing
    position_pct_override: Optional[float] = None  # Override max position %
    
    # Risk posture
    risk_posture: str = "normal"  # RiskPosture value
    
    # Timing
    wait_minutes: int = 0
    
    # Key factors
    key_risks: List[str] = field(default_factory=list)
    key_support: List[str] = field(default_factory=list)
    
    # Veto
    veto_active: bool = False
    veto_reason: Optional[str] = None
    
    # Rationale
    decision_rationale: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "final_direction": self.final_direction,
            "confidence_delta": self.confidence_delta,
            "position_pct_override": self.position_pct_override,
            "risk_posture": self.risk_posture,
            "wait_minutes": self.wait_minutes,
            "key_risks": self.key_risks,
            "key_support": self.key_support,
            "veto_active": self.veto_active,
            "veto_reason": self.veto_reason,
            "decision_rationale": self.decision_rationale,
        }
    
    @classmethod
    def no_delta(cls) -> "Tier2Delta":
        """Return a neutral delta (no changes)."""
        return cls(
            final_direction="NEUTRAL",
            confidence_delta=0,
            decision_rationale="Tier 2 made no adjustments",
        )


@dataclass
class ArbiterOutput:
    """
    Output from the Arbiter synthesis layer.
    
    Wraps the delta with additional metadata for auditing.
    """
    delta_type: str  # "NO_DELTA", "WAIT", "NO_TRADE", "CONFIDENCE_DOWN", "CONFIDENCE_UP"
    confidence_delta: float  # -25 to +5
    position_cap_delta: Optional[float] = None  # Reduce position cap
    timing_recommendation: Optional[str] = None  # "delay_30m", etc.
    rationale: str = ""
    claims_graph: List[Dict] = field(default_factory=list)
    veto_active: bool = False
    veto_reason: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "delta_type": self.delta_type,
            "confidence_delta": self.confidence_delta,
            "position_cap_delta": self.position_cap_delta,
            "timing_recommendation": self.timing_recommendation,
            "rationale": self.rationale,
            "veto_active": self.veto_active,
            "veto_reason": self.veto_reason,
        }
    
    @classmethod
    def no_delta(cls) -> "ArbiterOutput":
        """Return a no-delta output."""
        return cls(
            delta_type="NO_DELTA",
            confidence_delta=0.0,
            rationale="No significant adjustments recommended",
        )


@dataclass
class Tier1DecisionPack:
    """
    The Tier 1 decision pack that gets escalated to Tier 2.
    
    Contains all context needed for expert analysis.
    """
    # Identification
    decision_id: str
    symbol: str
    timestamp: datetime
    
    # Tier 1 decision
    action: str  # "BUY", "SELL", "HOLD", "WAIT"
    direction: str  # "BULLISH", "BEARISH", "NEUTRAL"
    confidence: float  # 0-100
    score: float  # 0-100
    
    # Position sizing
    position_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    
    # Market data
    price: float
    volume: float
    
    # Technical snapshot
    rsi: float
    macd_value: float
    macd_signal: float
    trend_state: str
    sma_20: float
    sma_50: float
    sma_200: float
    atr_pct: float
    
    # Regime
    regime: str
    regime_confidence: float
    vix: float
    vix_state: str
    
    # Sentiment
    sentiment_score: float
    news_count_7d: int
    
    # LuxAlgo signals
    weekly_signal: Optional[Dict] = None
    daily_signal: Optional[Dict] = None
    h4_signal: Optional[Dict] = None
    
    # Portfolio context (optional)
    current_position: float = 0.0
    sector_exposure_pct: float = 0.0
    portfolio_heat: float = 0.0
    
    # Data quality
    data_age_seconds: float = 0.0
    missing_feeds: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def get_evidence_keys(self) -> set:
        """Get all available evidence keys for claim validation."""
        keys = {
            "symbol", "price", "volume",
            "rsi", "macd_value", "macd_signal", "trend_state",
            "sma_20", "sma_50", "sma_200", "atr_pct",
            "regime", "regime_confidence", "vix", "vix_state",
            "sentiment_score", "news_count_7d",
            "confidence", "score", "direction",
        }
        
        # Add prefixed keys
        for prefix in ["polygon.", "stocknews.", "cryptonews."]:
            for key in ["rsi", "macd", "vix", "sentiment", "atr_pct", "volume"]:
                keys.add(f"{prefix}{key}")
        
        # Add signal keys
        if self.weekly_signal:
            keys.add("weekly_signal")
            keys.add("weekly.action")
        if self.daily_signal:
            keys.add("daily_signal")
            keys.add("daily.action")
        if self.h4_signal:
            keys.add("h4_signal")
            keys.add("h4.action")
        
        return keys


@dataclass
class EscalationRequest:
    """Request to escalate a decision to Tier 2."""
    tier1_pack: Tier1DecisionPack
    escalation_reasons: List[str]
    priority: str = "normal"  # "normal", "high", "critical"
    
    # Portfolio snapshot (optional)
    portfolio_snapshot: Optional[Dict] = None
    
    # Additional context
    extra_context: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "tier1_pack": self.tier1_pack.to_dict(),
            "escalation_reasons": self.escalation_reasons,
            "priority": self.priority,
            "portfolio_snapshot": self.portfolio_snapshot,
            "extra_context": self.extra_context,
        }


@dataclass
class Tier2Decision:
    """
    Complete Tier 2 decision output.
    
    Contains the delta, claims graph summary, and audit trail.
    """
    decision_id: str
    symbol: str
    timestamp: datetime
    
    # The delta
    delta: Tier2Delta
    
    # Claims graph summary
    total_agents: int
    valid_agents: int
    json_valid_rate: float
    pro_claims_count: int
    anti_claims_count: int
    conflicts_count: int
    
    # Execution metadata
    tier2_latency_ms: float
    light_round_latency_ms: float
    deep_round_latency_ms: float
    
    # Status
    status: str  # "completed", "fallback", "error", "disabled"
    fallback_reason: Optional[str] = None
    
    # Full claims graph (for audit)
    claims_graph: Optional[ClaimsGraph] = None
    
    def to_dict(self) -> Dict:
        result = {
            "decision_id": self.decision_id,
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "delta": self.delta.to_dict(),
            "total_agents": self.total_agents,
            "valid_agents": self.valid_agents,
            "json_valid_rate": round(self.json_valid_rate, 3),
            "pro_claims_count": self.pro_claims_count,
            "anti_claims_count": self.anti_claims_count,
            "conflicts_count": self.conflicts_count,
            "tier2_latency_ms": round(self.tier2_latency_ms, 1),
            "light_round_latency_ms": round(self.light_round_latency_ms, 1),
            "deep_round_latency_ms": round(self.deep_round_latency_ms, 1),
            "status": self.status,
            "fallback_reason": self.fallback_reason,
        }
        
        if self.claims_graph:
            result["claims_graph_summary"] = self.claims_graph.to_dict()
        
        return result
