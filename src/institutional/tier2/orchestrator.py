"""
Tier 2 Council-of-Experts Orchestrator
========================================

The main orchestrator that runs the 5-layer decision pipeline.

Pipeline:
    Layer A: Allocator (meta-orchestrator)
    Layer B: Light Round (quick agents)
    Layer C: Deep Round (detailed analysis)
    Layer D: Cross-Exam (optional, if conflicts)
    Layer E: Arbiter (synthesis)
    
Returns an auditable delta to Tier 1's decision:
- WAIT/NO_TRADE
- Confidence adjustment (usually down, rarely up)
- Position cap reduction
- Timing delay
"""

import time
import uuid
import logging
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field

from .config import Tier2Config, DEFAULT_CONFIG, EscalationReason
from .schemas import (
    Tier1DecisionPack,
    Tier2Decision,
    AgentOutput,
    ArbiterOutput,
)
from .registry import AgentRegistry, create_default_registry
from .runtime import AgentRuntime
from .allocator import Allocator, AllocationResult
from .arbiter import Arbiter
from .circuit_breaker import CircuitBreaker, CircuitBreakerState
from .escalation import EscalationDetector, EscalationResult
from .store import (
    DecisionStore,
    InMemoryStore,
    DecisionRecord,
    AgentRunRecord,
    generate_decision_id,
)

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorMetrics:
    """Metrics from orchestrator run."""
    decision_id: str
    total_latency_ms: float
    light_round_latency_ms: float
    deep_round_latency_ms: float
    arbiter_latency_ms: float
    agents_run: int
    tokens_used: int
    escalation_reasons: List[str]
    circuit_breaker_state: str
    cross_exam_triggered: bool = False


class Tier2Orchestrator:
    """
    Main Tier 2 Council-of-Experts Orchestrator.
    
    Implements the governed decision quality layer that produces
    auditable deltas to Tier 1 decisions.
    
    Hard Rules (from blueprint):
    1. Evidence-bounded: Agents must cite from provided data
    2. Budgeted depth: Token and time limits per layer
    3. Risk as gate, not vote: RiskGatekeeper can veto
    4. Deterministic fallback: On failure, return "no delta"
    5. Staged calibration: Track outcomes for weight adjustment
    """
    
    def __init__(
        self,
        config: Tier2Config = None,
        registry: AgentRegistry = None,
        runtime: AgentRuntime = None,
        store: DecisionStore = None,
        allocator: Allocator = None,
        arbiter: Arbiter = None,
        circuit_breaker: CircuitBreaker = None,
        escalation_detector: EscalationDetector = None,
    ):
        """
        Initialize orchestrator with components.
        
        All components are optional and will be created with defaults.
        """
        self.config = config or DEFAULT_CONFIG
        self.registry = registry or create_default_registry(self.config)
        self.runtime = runtime or AgentRuntime(self.config)
        self.store = store or InMemoryStore()
        self.allocator = allocator or Allocator(self.config, self.registry)
        self.arbiter = arbiter or Arbiter(self.config)
        self.circuit_breaker = circuit_breaker or CircuitBreaker(self.config)
        self.escalation_detector = escalation_detector or EscalationDetector(self.config)
    
    def should_escalate(
        self,
        tier1_pack: Tier1DecisionPack,
        portfolio_snapshot: Optional[Dict] = None,
    ) -> EscalationResult:
        """
        Determine if Tier 1 decision should be escalated to Tier 2.
        
        Args:
            tier1_pack: The Tier 1 decision
            portfolio_snapshot: Optional portfolio context
            
        Returns:
            EscalationResult with decision and reasons
        """
        return self.escalation_detector.detect(tier1_pack, portfolio_snapshot)
    
    def run(
        self,
        tier1_pack: Tier1DecisionPack,
        escalation_reasons: List[str] = None,
        portfolio_snapshot: Optional[Dict] = None,
        force_deep: bool = False,
    ) -> Tier2Decision:
        """
        Run the Tier 2 pipeline on a Tier 1 decision.
        
        Args:
            tier1_pack: The Tier 1 decision pack
            escalation_reasons: Why we're escalating (from detector)
            portfolio_snapshot: Optional portfolio context
            force_deep: Force deep round on all agents
            
        Returns:
            Tier2Decision with delta or no-delta result
        """
        start_time = time.time()
        decision_id = generate_decision_id(
            tier1_pack.symbol,
            datetime.now(timezone.utc).isoformat(),
        )
        
        # Check circuit breaker first
        if self.circuit_breaker.is_open():
            logger.warning(f"Circuit breaker OPEN - returning no delta for {tier1_pack.symbol}")
            return self._no_delta_decision(
                tier1_pack,
                decision_id,
                reason="circuit_breaker_open",
                latency_ms=(time.time() - start_time) * 1000,
            )
        
        try:
            # Run the pipeline
            decision = self._run_pipeline(
                tier1_pack,
                decision_id,
                escalation_reasons or [],
                portfolio_snapshot,
                force_deep,
            )
            
            # Record success
            latency_ms = (time.time() - start_time) * 1000
            self.circuit_breaker.record_success(latency_ms)
            
            # Store decision
            self._store_decision(decision, tier1_pack, escalation_reasons or [])
            
            return decision
            
        except Exception as e:
            logger.error(f"Pipeline failed for {tier1_pack.symbol}: {e}", exc_info=True)
            
            # Record failure
            latency_ms = (time.time() - start_time) * 1000
            self.circuit_breaker.record_failure(latency_ms, str(e))
            
            # Deterministic fallback: return no delta
            return self._no_delta_decision(
                tier1_pack,
                decision_id,
                reason=f"pipeline_error: {str(e)[:100]}",
                latency_ms=latency_ms,
            )
    
    def _run_pipeline(
        self,
        tier1_pack: Tier1DecisionPack,
        decision_id: str,
        escalation_reasons: List[str],
        portfolio_snapshot: Optional[Dict],
        force_deep: bool,
    ) -> Tier2Decision:
        """
        Run the 5-layer pipeline.
        
        Layer A: Allocator decides which agents to run
        Layer B: Light round (quick assessment)
        Layer C: Deep round (detailed analysis, if needed)
        Layer D: Cross-exam (if conflicts, optional)
        Layer E: Arbiter synthesizes and produces delta
        """
        metrics_start = time.time()
        
        # ========== LAYER A: ALLOCATION ==========
        allocation = self.allocator.allocate(
            tier1_pack,
            escalation_reasons,
            portfolio_snapshot,
        )
        
        logger.info(
            f"[{tier1_pack.symbol}] Allocated: "
            f"{len(allocation.light_agents)} light, "
            f"{len(allocation.deep_agents)} deep agents"
        )
        
        # ========== LAYER B: LIGHT ROUND ==========
        light_start = time.time()
        light_outputs = self._run_light_round(
            tier1_pack,
            allocation.light_agents,
            decision_id,
        )
        light_latency_ms = (time.time() - light_start) * 1000
        
        # Check if we need deep round
        deep_outputs: List[AgentOutput] = []
        deep_latency_ms = 0.0
        
        need_deep = (
            force_deep
            or len(allocation.deep_agents) > 0
            or self._should_go_deep(light_outputs, escalation_reasons)
        )
        
        # ========== LAYER C: DEEP ROUND ==========
        if need_deep:
            deep_start = time.time()
            deep_outputs = self._run_deep_round(
                tier1_pack,
                allocation.deep_agents or allocation.light_agents,
                light_outputs,
                decision_id,
            )
            deep_latency_ms = (time.time() - deep_start) * 1000
        
        # ========== LAYER D: CROSS-EXAM (Optional) ==========
        cross_exam_triggered = False
        all_outputs = light_outputs + deep_outputs
        
        if self._has_conflicts(all_outputs):
            cross_exam_triggered = True
            # For now, just note the conflict - full cross-exam is future work
            logger.info(f"[{tier1_pack.symbol}] Conflict detected, would trigger cross-exam")
        
        # ========== LAYER E: ARBITER SYNTHESIS ==========
        arbiter_start = time.time()
        arbiter_output = self.arbiter.synthesize(
            tier1_pack,
            all_outputs,
            allocation.severity,
        )
        arbiter_latency_ms = (time.time() - arbiter_start) * 1000
        
        # Build final decision
        total_latency_ms = (time.time() - metrics_start) * 1000
        
        decision = Tier2Decision(
            decision_id=decision_id,
            symbol=tier1_pack.symbol,
            timestamp=datetime.now(timezone.utc).isoformat(),
            tier1_action=tier1_pack.action,
            tier1_confidence=tier1_pack.confidence,
            delta_type=arbiter_output.delta_type,
            delta_confidence=arbiter_output.confidence_delta,
            delta_position_cap=arbiter_output.position_cap_delta,
            delta_timing=arbiter_output.timing_recommendation,
            final_action=self._compute_final_action(tier1_pack, arbiter_output),
            final_confidence=self._compute_final_confidence(tier1_pack, arbiter_output),
            rationale=arbiter_output.rationale,
            claims_graph=arbiter_output.claims_graph,
            agent_count=len(all_outputs),
            latency_ms=total_latency_ms,
            escalation_reasons=escalation_reasons,
            cross_exam_triggered=cross_exam_triggered,
        )
        
        # Store agent runs
        for output in all_outputs:
            self._store_agent_run(output, decision_id)
        
        logger.info(
            f"[{tier1_pack.symbol}] Tier2 complete: "
            f"delta={arbiter_output.delta_type}, "
            f"conf_delta={arbiter_output.confidence_delta:+.1f}%, "
            f"latency={total_latency_ms:.0f}ms"
        )
        
        return decision
    
    def _run_light_round(
        self,
        tier1_pack: Tier1DecisionPack,
        agent_names: List[str],
        decision_id: str,
    ) -> List[AgentOutput]:
        """Run light round with quick agents."""
        if not agent_names:
            return []
        
        agents = [
            self.registry.get_agent(name)
            for name in agent_names
            if self.registry.get_agent(name) is not None
        ]
        
        if not agents:
            return []
        
        outputs = self.runtime.run_agents_parallel(
            agents,
            tier1_pack,
            round_type="light",
        )
        
        return outputs
    
    def _run_deep_round(
        self,
        tier1_pack: Tier1DecisionPack,
        agent_names: List[str],
        light_outputs: List[AgentOutput],
        decision_id: str,
    ) -> List[AgentOutput]:
        """Run deep round with detailed analysis."""
        if not agent_names:
            return []
        
        agents = [
            self.registry.get_agent(name)
            for name in agent_names
            if self.registry.get_agent(name) is not None
        ]
        
        if not agents:
            return []
        
        # Deep round gets light round context
        outputs = self.runtime.run_agents_parallel(
            agents,
            tier1_pack,
            round_type="deep",
            prior_outputs=light_outputs,
        )
        
        return outputs
    
    def _should_go_deep(
        self,
        light_outputs: List[AgentOutput],
        escalation_reasons: List[str],
    ) -> bool:
        """Determine if we need deep analysis."""
        # High severity reasons always go deep
        high_severity = {
            EscalationReason.SIGNAL_CONFLICT.value,
            EscalationReason.REGIME_TRANSITION.value,
            EscalationReason.HIGH_VOLATILITY.value,
        }
        
        if any(r in high_severity for r in escalation_reasons):
            return True
        
        # Check for low confidence or disagreement in light round
        if light_outputs:
            avg_confidence = sum(o.confidence for o in light_outputs) / len(light_outputs)
            if avg_confidence < 50:
                return True
            
            # Check for claim conflicts
            claim_directions = []
            for output in light_outputs:
                for claim in output.claims:
                    if claim.direction in ["BULLISH", "BEARISH"]:
                        claim_directions.append(claim.direction)
            
            if "BULLISH" in claim_directions and "BEARISH" in claim_directions:
                return True
        
        return False
    
    def _has_conflicts(self, outputs: List[AgentOutput]) -> bool:
        """Check if agent outputs have significant conflicts."""
        directions = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}
        
        for output in outputs:
            for claim in output.claims:
                if claim.direction in directions:
                    directions[claim.direction] += claim.weight
        
        # Conflict if both bullish and bearish have significant weight
        if directions["BULLISH"] > 0.3 and directions["BEARISH"] > 0.3:
            return True
        
        return False
    
    def _compute_final_action(
        self,
        tier1_pack: Tier1DecisionPack,
        arbiter: ArbiterOutput,
    ) -> str:
        """Compute final action after delta."""
        if arbiter.delta_type == "WAIT":
            return "WAIT"
        if arbiter.delta_type == "NO_TRADE":
            return "NO_TRADE"
        
        # Otherwise keep Tier 1 action
        return tier1_pack.action
    
    def _compute_final_confidence(
        self,
        tier1_pack: Tier1DecisionPack,
        arbiter: ArbiterOutput,
    ) -> float:
        """Compute final confidence after delta."""
        final = tier1_pack.confidence + arbiter.confidence_delta
        # Clamp to [0, 100]
        return max(0.0, min(100.0, final))
    
    def _no_delta_decision(
        self,
        tier1_pack: Tier1DecisionPack,
        decision_id: str,
        reason: str,
        latency_ms: float,
    ) -> Tier2Decision:
        """Create a no-delta decision (deterministic fallback)."""
        return Tier2Decision(
            decision_id=decision_id,
            symbol=tier1_pack.symbol,
            timestamp=datetime.now(timezone.utc).isoformat(),
            tier1_action=tier1_pack.action,
            tier1_confidence=tier1_pack.confidence,
            delta_type="NO_DELTA",
            delta_confidence=0.0,
            delta_position_cap=None,
            delta_timing=None,
            final_action=tier1_pack.action,
            final_confidence=tier1_pack.confidence,
            rationale=f"No delta applied: {reason}",
            claims_graph=[],
            agent_count=0,
            latency_ms=latency_ms,
            escalation_reasons=[],
            cross_exam_triggered=False,
        )
    
    def _store_decision(
        self,
        decision: Tier2Decision,
        tier1_pack: Tier1DecisionPack,
        escalation_reasons: List[str],
    ) -> None:
        """Store decision to persistent storage."""
        try:
            from datetime import timedelta
            
            ttl = int((datetime.now(timezone.utc) + timedelta(days=90)).timestamp())
            
            record = DecisionRecord(
                decision_id=decision.decision_id,
                symbol=decision.symbol,
                timestamp=decision.timestamp,
                tier1_action=decision.tier1_action,
                tier1_confidence=decision.tier1_confidence,
                tier2_delta={
                    "type": decision.delta_type,
                    "confidence": decision.delta_confidence,
                    "position_cap": decision.delta_position_cap,
                    "timing": decision.delta_timing,
                },
                final_action=decision.final_action,
                final_confidence=decision.final_confidence,
                escalation_reasons=escalation_reasons,
                agent_count=decision.agent_count,
                latency_ms=decision.latency_ms,
                ttl=ttl,
            )
            
            self.store.save_decision(record)
            
        except Exception as e:
            logger.warning(f"Failed to store decision: {e}")
    
    def _store_agent_run(
        self,
        output: AgentOutput,
        decision_id: str,
    ) -> None:
        """Store agent run to persistent storage."""
        try:
            from datetime import timedelta
            
            # Shorter TTL for raw outputs
            ttl = int((datetime.now(timezone.utc) + timedelta(days=7)).timestamp())
            
            record = AgentRunRecord(
                decision_id=decision_id,
                agent_name=output.agent_name,
                round=output.round_type,
                input_tokens=output.input_tokens,
                output_tokens=output.output_tokens,
                latency_ms=output.latency_ms,
                claims=[
                    {
                        "type": c.claim_type,
                        "direction": c.direction,
                        "magnitude": c.magnitude,
                        "weight": c.weight,
                        "evidence": c.evidence,
                    }
                    for c in output.claims
                ],
                raw_output=output.raw_output if self.config.store_raw_outputs else None,
                timestamp=datetime.now(timezone.utc).isoformat(),
                ttl=ttl,
            )
            
            self.store.save_agent_run(record)
            
        except Exception as e:
            logger.warning(f"Failed to store agent run: {e}")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current orchestrator metrics."""
        cb_metrics = self.circuit_breaker.get_metrics()
        
        return {
            "circuit_breaker": cb_metrics,
            "registry": {
                "agent_count": len(self.registry.list_agents()),
                "agents": self.registry.list_agents(),
            },
            "config": {
                "light_round_timeout": self.config.light_round_timeout_ms,
                "deep_round_timeout": self.config.deep_round_timeout_ms,
                "max_tokens_light": self.config.max_tokens_light,
                "max_tokens_deep": self.config.max_tokens_deep,
            },
        }


def create_orchestrator(
    use_dynamodb: bool = False,
    dynamodb_tables: Optional[Dict[str, str]] = None,
    config: Tier2Config = None,
) -> Tier2Orchestrator:
    """
    Factory function to create orchestrator with appropriate store.
    
    Args:
        use_dynamodb: Whether to use DynamoDB for storage
        dynamodb_tables: Table name overrides
        config: Optional config override
        
    Returns:
        Configured Tier2Orchestrator
    """
    cfg = config or DEFAULT_CONFIG
    
    if use_dynamodb:
        from .store import DynamoDBStore
        
        tables = dynamodb_tables or {}
        store = DynamoDBStore(
            decisions_table=tables.get("decisions", "nuble-decisions"),
            runs_table=tables.get("runs", "nuble-agent-runs"),
            outcomes_table=tables.get("outcomes", "nuble-outcomes"),
        )
    else:
        store = InMemoryStore()
    
    return Tier2Orchestrator(
        config=cfg,
        store=store,
    )
