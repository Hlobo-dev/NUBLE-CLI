"""
Tier 2 Arbiter
===============

Layer E of the Tier 2 pipeline.

The Arbiter synthesizes the claims graph into deltas.
It aggregates, does not invent.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
import logging

from .config import Tier2Config, DEFAULT_CONFIG
from .schemas import (
    ClaimsGraph,
    Tier2Delta,
    AgentOutput,
    Claim,
    Verdict,
    RiskPosture,
    ClaimStance,
)

logger = logging.getLogger(__name__)


@dataclass 
class ArbiterInput:
    """Structured input for the arbiter."""
    claims_graph: ClaimsGraph
    escalation_reasons: List[str]
    tier1_confidence: float
    tier1_direction: str
    agent_weights: Dict[str, float]


class Arbiter:
    """
    Synthesizes claims into final deltas.
    
    Key principles:
    1. Aggregate, don't invent - only synthesize from claims
    2. Respect veto triggers
    3. Default to conservative adjustments
    4. Produce auditable rationale
    
    Tie-break rules:
    - If conflicts are high → default to WAIT
    - If risk posture != OK → cannot increase size
    - Cannot override veto triggers
    """
    
    def __init__(self, config: Tier2Config = None):
        self.config = config or DEFAULT_CONFIG
    
    def synthesize(self, input: ArbiterInput) -> Tier2Delta:
        """
        Synthesize claims graph into delta recommendation.
        
        Args:
            input: ArbiterInput with claims graph and context
            
        Returns:
            Tier2Delta with adjustments
        """
        graph = input.claims_graph
        weights = input.agent_weights
        
        # Check for veto first
        veto_active, veto_reason = self._check_veto(graph)
        if veto_active:
            return Tier2Delta(
                final_direction="NO_TRADE",
                confidence_delta=-100,
                veto_active=True,
                veto_reason=veto_reason,
                risk_posture=RiskPosture.NO_TRADE.value,
                decision_rationale=f"VETO: {veto_reason}",
            )
        
        # Check JSON validity threshold
        if graph.json_valid_rate < self.config.min_valid_json_rate:
            logger.warning(
                f"JSON valid rate {graph.json_valid_rate:.1%} below threshold "
                f"{self.config.min_valid_json_rate:.1%} - returning no delta"
            )
            return Tier2Delta.no_delta()
        
        # Calculate weighted sentiment from claims
        sentiment = graph.get_weighted_sentiment(weights)
        
        # Aggregate verdicts from agents
        verdict_scores = self._aggregate_verdicts(graph, weights)
        
        # Aggregate recommended deltas
        agg_deltas = self._aggregate_deltas(graph, weights)
        
        # Determine final direction
        final_direction = self._determine_direction(
            sentiment, verdict_scores, input.tier1_direction
        )
        
        # Calculate confidence delta
        confidence_delta = self._calculate_confidence_delta(
            sentiment, graph, agg_deltas, input.tier1_confidence
        )
        
        # Determine position cap
        position_cap = self._calculate_position_cap(agg_deltas, graph)
        
        # Determine wait time
        wait_minutes = self._calculate_wait_minutes(agg_deltas, graph)
        
        # Determine risk posture
        risk_posture = self._determine_risk_posture(
            sentiment, graph, confidence_delta
        )
        
        # Extract key risks and support
        key_risks, key_support = self._extract_key_factors(graph)
        
        # Build rationale
        rationale = self._build_rationale(
            sentiment, verdict_scores, graph, confidence_delta, final_direction
        )
        
        return Tier2Delta(
            final_direction=final_direction,
            confidence_delta=confidence_delta,
            position_pct_override=position_cap,
            risk_posture=risk_posture,
            wait_minutes=wait_minutes,
            key_risks=key_risks,
            key_support=key_support,
            veto_active=False,
            veto_reason=None,
            decision_rationale=rationale,
        )
    
    def _check_veto(self, graph: ClaimsGraph) -> tuple:
        """Check if any agent triggered a veto."""
        # Check deep outputs first (more weight)
        for agent_name, output in graph.deep_outputs.items():
            if output.verdict == Verdict.VETO.value:
                # Find the veto claim
                for claim in output.claims:
                    if claim.stance == ClaimStance.ANTI.value and claim.strength > 0.8:
                        return True, f"{agent_name}: {claim.statement}"
                return True, f"{agent_name} issued VETO"
        
        # Check light outputs
        for agent_name, output in graph.light_outputs.items():
            if output.verdict == Verdict.VETO.value:
                for claim in output.claims:
                    if claim.stance == ClaimStance.ANTI.value and claim.strength > 0.8:
                        return True, f"{agent_name}: {claim.statement}"
                return True, f"{agent_name} issued VETO"
        
        return False, None
    
    def _aggregate_verdicts(
        self, 
        graph: ClaimsGraph, 
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Aggregate verdicts from all agents."""
        verdict_scores = {
            "BUY": 0.0,
            "SELL": 0.0,
            "WAIT": 0.0,
            "NEUTRAL": 0.0,
            "REDUCE": 0.0,
        }
        total_weight = 0.0
        
        all_outputs = {**graph.light_outputs, **graph.deep_outputs}
        
        for agent_name, output in all_outputs.items():
            weight = weights.get(agent_name, 1.0) * output.confidence
            total_weight += weight
            
            verdict = output.verdict
            if verdict in verdict_scores:
                verdict_scores[verdict] += weight
        
        # Normalize
        if total_weight > 0:
            for v in verdict_scores:
                verdict_scores[v] /= total_weight
        
        return verdict_scores
    
    def _aggregate_deltas(
        self,
        graph: ClaimsGraph,
        weights: Dict[str, float],
    ) -> Dict[str, Any]:
        """Aggregate recommended deltas from all agents."""
        confidence_deltas = []
        position_caps = []
        wait_minutes_list = []
        risk_postures = []
        
        all_outputs = {**graph.light_outputs, **graph.deep_outputs}
        
        for agent_name, output in all_outputs.items():
            weight = weights.get(agent_name, 1.0)
            deltas = output.recommended_deltas
            
            if deltas.confidence_delta != 0:
                confidence_deltas.append((deltas.confidence_delta, weight))
            
            if deltas.position_pct_cap is not None:
                position_caps.append((deltas.position_pct_cap, weight))
            
            if deltas.wait_minutes > 0:
                wait_minutes_list.append((deltas.wait_minutes, weight))
            
            if deltas.risk_posture:
                risk_postures.append((deltas.risk_posture, weight))
        
        return {
            "confidence_deltas": confidence_deltas,
            "position_caps": position_caps,
            "wait_minutes": wait_minutes_list,
            "risk_postures": risk_postures,
        }
    
    def _determine_direction(
        self,
        sentiment: float,
        verdict_scores: Dict[str, float],
        tier1_direction: str,
    ) -> str:
        """Determine final direction recommendation."""
        # If WAIT is dominant, recommend WAIT
        if verdict_scores["WAIT"] > 0.4:
            return "WAIT"
        
        # If REDUCE is dominant, recommend reducing
        if verdict_scores["REDUCE"] > 0.3:
            return "REDUCE"
        
        # If sentiment is strongly against tier1 direction
        if tier1_direction == "BULLISH" and sentiment < -0.3:
            return "WAIT"
        elif tier1_direction == "BEARISH" and sentiment > 0.3:
            return "WAIT"
        
        # Check buy/sell dominance
        if verdict_scores["BUY"] > verdict_scores["SELL"] + 0.2:
            return "BUY"
        elif verdict_scores["SELL"] > verdict_scores["BUY"] + 0.2:
            return "SELL"
        
        # If unclear, preserve tier1 direction but cautiously
        if verdict_scores["NEUTRAL"] > 0.4:
            return "NEUTRAL"
        
        return tier1_direction if tier1_direction in ["BUY", "SELL"] else "NEUTRAL"
    
    def _calculate_confidence_delta(
        self,
        sentiment: float,
        graph: ClaimsGraph,
        agg_deltas: Dict,
        tier1_confidence: float,
    ) -> int:
        """Calculate confidence adjustment."""
        # Start with weighted average of agent recommendations
        confidence_deltas = agg_deltas["confidence_deltas"]
        
        if not confidence_deltas:
            # No explicit recommendations - use sentiment
            if sentiment < -0.3:
                base_delta = -10
            elif sentiment < -0.1:
                base_delta = -5
            elif sentiment > 0.3:
                base_delta = 3  # Conservative increase
            else:
                base_delta = 0
        else:
            # Weighted average of recommendations
            total_weight = sum(w for _, w in confidence_deltas)
            if total_weight > 0:
                base_delta = sum(d * w for d, w in confidence_deltas) / total_weight
            else:
                base_delta = 0
        
        # Adjust based on conflict level
        conflict_penalty = len(graph.conflicts) * 2
        base_delta -= conflict_penalty
        
        # Apply limits
        base_delta = max(
            -self.config.max_confidence_decrease,
            min(self.config.max_confidence_increase, base_delta)
        )
        
        return int(round(base_delta))
    
    def _calculate_position_cap(
        self,
        agg_deltas: Dict,
        graph: ClaimsGraph,
    ) -> Optional[float]:
        """Calculate position size cap."""
        position_caps = agg_deltas["position_caps"]
        
        if not position_caps:
            return None
        
        # Take the minimum (most conservative)
        min_cap = min(cap for cap, _ in position_caps)
        
        # Apply limits
        min_cap = max(
            self.config.min_position_cap_pct,
            min(self.config.max_position_cap_pct, min_cap)
        )
        
        return min_cap
    
    def _calculate_wait_minutes(
        self,
        agg_deltas: Dict,
        graph: ClaimsGraph,
    ) -> int:
        """Calculate recommended wait time."""
        wait_list = agg_deltas["wait_minutes"]
        
        if not wait_list:
            return 0
        
        # Take weighted average
        total_weight = sum(w for _, w in wait_list)
        if total_weight > 0:
            avg_wait = sum(m * w for m, w in wait_list) / total_weight
        else:
            avg_wait = 0
        
        # Apply limit
        return min(int(round(avg_wait)), self.config.max_wait_minutes)
    
    def _determine_risk_posture(
        self,
        sentiment: float,
        graph: ClaimsGraph,
        confidence_delta: int,
    ) -> str:
        """Determine overall risk posture."""
        # Check for dominant anti claims
        anti_ratio = len(graph.anti_claims) / max(len(graph.all_claims), 1)
        
        if anti_ratio > 0.6 or confidence_delta < -15:
            return RiskPosture.DEFENSIVE.value
        elif anti_ratio > 0.4 or confidence_delta < -8:
            return RiskPosture.CAUTIOUS.value
        elif sentiment > 0.3 and confidence_delta >= 0:
            return RiskPosture.NORMAL.value
        elif sentiment < -0.2:
            return RiskPosture.CAUTIOUS.value
        
        return RiskPosture.NORMAL.value
    
    def _extract_key_factors(
        self,
        graph: ClaimsGraph,
    ) -> tuple:
        """Extract key risk and support factors from claims."""
        key_risks = []
        key_support = []
        
        # Sort claims by strength
        sorted_anti = sorted(
            graph.anti_claims,
            key=lambda c: c.strength,
            reverse=True
        )
        sorted_pro = sorted(
            graph.pro_claims,
            key=lambda c: c.strength,
            reverse=True
        )
        
        # Take top 3 of each
        for claim in sorted_anti[:3]:
            key_risks.append(claim.statement)
        
        for claim in sorted_pro[:3]:
            key_support.append(claim.statement)
        
        return key_risks, key_support
    
    def _build_rationale(
        self,
        sentiment: float,
        verdict_scores: Dict[str, float],
        graph: ClaimsGraph,
        confidence_delta: int,
        final_direction: str,
    ) -> str:
        """Build human-readable rationale."""
        parts = []
        
        # Sentiment summary
        if sentiment > 0.2:
            parts.append(f"Sentiment: positive ({sentiment:.2f})")
        elif sentiment < -0.2:
            parts.append(f"Sentiment: negative ({sentiment:.2f})")
        else:
            parts.append(f"Sentiment: neutral ({sentiment:.2f})")
        
        # Verdict summary
        top_verdict = max(verdict_scores, key=verdict_scores.get)
        parts.append(f"Dominant verdict: {top_verdict} ({verdict_scores[top_verdict]:.0%})")
        
        # Claims summary
        parts.append(
            f"Claims: {len(graph.pro_claims)} pro, "
            f"{len(graph.anti_claims)} anti, "
            f"{len(graph.conflicts)} conflicts"
        )
        
        # Decision
        parts.append(f"Direction: {final_direction}, confidence delta: {confidence_delta:+d}")
        
        return " | ".join(parts)
