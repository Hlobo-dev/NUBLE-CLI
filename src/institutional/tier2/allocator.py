"""
Tier 2 Allocator
=================

Layer A of the Tier 2 pipeline.

The Allocator (Meta-Orchestrator) decides:
- Which agents run deep
- Token budgets
- Whether to trigger conflict deepening
- Model selection
"""

from typing import List, Dict, Set, Optional
from dataclasses import dataclass

from .config import Tier2Config, AgentConfig, EscalationReason, ESCALATION_REASONS, DEFAULT_CONFIG
from .schemas import Tier1DecisionPack, ClaimsGraph, AgentOutput


@dataclass
class AllocationDecision:
    """Result of allocator's decision."""
    light_agents: List[str]  # Agents to run in light mode
    deep_agents: List[str]   # Agents to run in deep mode
    
    total_token_budget: int
    estimated_latency_ms: int
    
    # Conflict deepening
    conflict_deepening: bool = False
    conflict_agents: List[str] = None
    
    # Model upgrades (rare)
    arbiter_model_upgrade: bool = False
    
    # Rationale
    rationale: str = ""


class Allocator:
    """
    Meta-Orchestrator for Tier 2.
    
    Responsibilities:
    1. Determine which agents should run in light vs deep mode
    2. Manage token budgets across all agents
    3. Trigger conflict deepening when needed
    4. Select models for special cases
    """
    
    def __init__(self, config: Tier2Config = None):
        self.config = config or DEFAULT_CONFIG
    
    def allocate(
        self,
        tier1_pack: Tier1DecisionPack,
        escalation_reasons: List[str],
        portfolio_snapshot: Optional[Dict] = None,
    ) -> AllocationDecision:
        """
        Make allocation decision.
        
        Args:
            tier1_pack: Tier 1 decision context
            escalation_reasons: List of reason strings
            portfolio_snapshot: Optional portfolio context
            
        Returns:
            AllocationDecision with agent assignments
        """
        # Get all enabled agents
        enabled_agents = [a.name for a in self.config.get_enabled_agents()]
        
        # Determine deep agents based on escalation
        deep_agents = self._select_deep_agents(escalation_reasons)
        
        # Light agents = all enabled - deep
        light_agents = [a for a in enabled_agents if a not in deep_agents]
        
        # Calculate token budgets
        light_budget = sum(
            self.config.agents[a].light_max_tokens 
            for a in light_agents 
            if a in self.config.agents
        )
        deep_budget = sum(
            self.config.agents[a].deep_max_tokens 
            for a in deep_agents 
            if a in self.config.agents
        )
        total_budget = light_budget + deep_budget
        
        # Enforce budget limits
        if total_budget > self.config.max_total_tokens_deep:
            # Reduce deep agents
            while (
                total_budget > self.config.max_total_tokens_deep 
                and len(deep_agents) > self.config.min_agents_deep
            ):
                # Remove lowest weight deep agent (except risk_gatekeeper)
                removable = [
                    a for a in deep_agents 
                    if a != "risk_gatekeeper" 
                    and not self.config.agents.get(a, AgentConfig(name=a)).always_run_deep
                ]
                if not removable:
                    break
                    
                # Sort by weight (ascending) and remove lowest
                removable.sort(key=lambda a: self.config.agents.get(a, AgentConfig(name=a)).weight)
                agent_to_remove = removable[0]
                deep_agents.remove(agent_to_remove)
                light_agents.append(agent_to_remove)
                
                # Recalculate
                light_budget = sum(
                    self.config.agents[a].light_max_tokens 
                    for a in light_agents 
                    if a in self.config.agents
                )
                deep_budget = sum(
                    self.config.agents[a].deep_max_tokens 
                    for a in deep_agents 
                    if a in self.config.agents
                )
                total_budget = light_budget + deep_budget
        
        # Estimate latency
        # Parallel execution: max of all agents + overhead
        max_light_time = max(
            self.config.agents[a].light_timeout_ms 
            for a in light_agents 
            if a in self.config.agents
        ) if light_agents else 0
        max_deep_time = max(
            self.config.agents[a].deep_timeout_ms 
            for a in deep_agents 
            if a in self.config.agents
        ) if deep_agents else 0
        estimated_latency = max_light_time + max_deep_time + 500  # 500ms overhead
        
        # Check for arbiter model upgrade (rare)
        arbiter_upgrade = self._should_upgrade_arbiter(
            tier1_pack, escalation_reasons, len(deep_agents)
        )
        
        # Build rationale
        rationale = self._build_rationale(
            escalation_reasons, light_agents, deep_agents
        )
        
        return AllocationDecision(
            light_agents=light_agents,
            deep_agents=deep_agents,
            total_token_budget=total_budget,
            estimated_latency_ms=estimated_latency,
            conflict_deepening=False,  # Will be set after light round
            conflict_agents=None,
            arbiter_model_upgrade=arbiter_upgrade,
            rationale=rationale,
        )
    
    def _select_deep_agents(self, escalation_reasons: List[str]) -> List[str]:
        """Select agents for deep dive based on escalation reasons."""
        return self.config.get_agents_for_deep(escalation_reasons)
    
    def _should_upgrade_arbiter(
        self,
        tier1_pack: Tier1DecisionPack,
        escalation_reasons: List[str],
        num_deep_agents: int,
    ) -> bool:
        """
        Determine if arbiter should use upgraded model.
        
        Upgrade when:
        - Many deep agents (complex synthesis needed)
        - Critical escalation reasons
        - High stakes decision (large position)
        """
        # Many deep agents = complex synthesis
        if num_deep_agents >= 6:
            return True
        
        # Critical reasons
        critical_reasons = {
            "earnings_imminent",
            "drawdown_elevated",
            "regime_transition",
        }
        if any(r in critical_reasons for r in escalation_reasons):
            return True
        
        # High confidence needing verification
        if tier1_pack.confidence > 90:
            return True
        
        return False
    
    def _build_rationale(
        self,
        escalation_reasons: List[str],
        light_agents: List[str],
        deep_agents: List[str],
    ) -> str:
        """Build human-readable rationale for allocation."""
        parts = []
        
        if escalation_reasons:
            parts.append(f"Escalated for: {', '.join(escalation_reasons)}")
        
        parts.append(f"Light round: {len(light_agents)} agents")
        parts.append(f"Deep round: {len(deep_agents)} agents ({', '.join(deep_agents)})")
        
        return " | ".join(parts)
    
    def reallocate_after_light(
        self,
        claims_graph: ClaimsGraph,
        original_allocation: AllocationDecision,
    ) -> AllocationDecision:
        """
        Optionally reallocate after light round based on conflicts.
        
        If there are significant conflicts in the light round,
        we may want to deep-dive additional agents.
        """
        # Detect conflicts
        claims_graph.detect_conflicts()
        
        if not claims_graph.conflicts:
            return original_allocation
        
        # Find agents involved in conflicts
        conflict_agents = set()
        for conflict in claims_graph.conflicts:
            # Find which agents made these claims
            pro_claim = conflict.get("pro_claim", {})
            anti_claim = conflict.get("anti_claim", {})
            
            for agent_name, output in claims_graph.light_outputs.items():
                for claim in output.claims:
                    if claim.id == pro_claim.get("id") or claim.id == anti_claim.get("id"):
                        conflict_agents.add(agent_name)
        
        if not conflict_agents:
            return original_allocation
        
        # Add conflicting agents to deep round if not already there
        new_deep = list(original_allocation.deep_agents)
        new_light = list(original_allocation.light_agents)
        
        for agent in conflict_agents:
            if agent in new_light and agent not in new_deep:
                new_light.remove(agent)
                new_deep.append(agent)
        
        # Respect max deep limit
        new_deep = new_deep[:self.config.max_agents_deep]
        
        return AllocationDecision(
            light_agents=new_light,
            deep_agents=new_deep,
            total_token_budget=original_allocation.total_token_budget,
            estimated_latency_ms=original_allocation.estimated_latency_ms,
            conflict_deepening=True,
            conflict_agents=list(conflict_agents),
            arbiter_model_upgrade=original_allocation.arbiter_model_upgrade,
            rationale=f"{original_allocation.rationale} | Conflict deepening: {conflict_agents}",
        )
