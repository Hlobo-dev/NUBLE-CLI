"""
Tier 2 Integration for V6 APEX Handler
========================================

This module provides integration between the V6 APEX decision engine
and the Tier 2 Council-of-Experts orchestrator.

Usage:
    from tier2_integration import apply_tier2_if_needed
    
    decision = make_decision(symbol)
    decision = apply_tier2_if_needed(decision)
"""

import os
import logging
from typing import Dict, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# Environment variable to enable/disable Tier 2
TIER2_ENABLED = os.environ.get("TIER2_ENABLED", "false").lower() == "true"

# Lazy-load orchestrator
_orchestrator = None


def get_orchestrator():
    """Lazy load the Tier 2 orchestrator."""
    global _orchestrator
    
    if _orchestrator is None:
        try:
            from src.institutional.tier2 import (
                Tier2Orchestrator,
                InMemoryStore,
            )
            
            # Use in-memory store for Lambda (or DynamoDB if configured)
            use_dynamodb = os.environ.get("TIER2_USE_DYNAMODB", "false").lower() == "true"
            
            if use_dynamodb:
                from src.institutional.tier2 import DynamoDBStore
                store = DynamoDBStore(
                    decisions_table=os.environ.get("TIER2_DECISIONS_TABLE", "nuble-tier2-decisions"),
                    runs_table=os.environ.get("TIER2_AGENT_RUNS_TABLE", "nuble-tier2-agent-runs"),
                    outcomes_table=os.environ.get("TIER2_OUTCOMES_TABLE", "nuble-tier2-outcomes"),
                )
            else:
                store = InMemoryStore()
            
            _orchestrator = Tier2Orchestrator(store=store)
            logger.info("Tier 2 orchestrator initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize Tier 2 orchestrator: {e}")
            return None
    
    return _orchestrator


def build_tier1_pack_from_decision(decision: Dict[str, Any]):
    """
    Build a Tier1DecisionPack from the V6 APEX decision dictionary.
    
    Maps the V6 decision structure to the Tier 2 schema.
    """
    from src.institutional.tier2.schemas import Tier1DecisionPack
    from datetime import datetime, timezone
    
    # Extract polygon data
    polygon = decision.get("polygon_summary", {})
    
    # Extract LuxAlgo signals
    luxalgo = decision.get("luxalgo_signals", {})
    
    # Map direction to expected format
    direction = decision.get("direction", "NEUTRAL")
    if direction == "BUY":
        direction = "BULLISH"
    elif direction == "SELL":
        direction = "BEARISH"
    
    # Build the pack
    return Tier1DecisionPack(
        decision_id=f"{decision.get('symbol')}_{decision.get('timestamp', '')}",
        symbol=decision.get("symbol", "UNKNOWN"),
        timestamp=datetime.fromisoformat(decision.get("timestamp", datetime.now(timezone.utc).isoformat())),
        
        # Decision
        action=decision.get("direction", "WAIT"),
        direction=direction,
        confidence=float(decision.get("confidence", 50.0)),
        score=float(decision.get("layers", {}).get("technical", {}).get("score", 0) * 100),
        
        # Position sizing
        position_pct=float(decision.get("trade_setup", {}).get("position_pct", 0) if decision.get("trade_setup") else 0),
        stop_loss_pct=float(decision.get("trade_setup", {}).get("stop_pct", 2.0) if decision.get("trade_setup") else 2.0),
        take_profit_pct=float(decision.get("trade_setup", {}).get("target_pcts", [6.0])[0] if decision.get("trade_setup") else 6.0),
        
        # Market data
        price=float(polygon.get("price", 0) or 0),
        volume=0.0,  # Not available in summary
        
        # Technical
        rsi=float(polygon.get("rsi", 50) or 50),
        macd_value=0.0,  # Summary doesn't have raw MACD values
        macd_signal=0.0,
        trend_state=polygon.get("trend", "NEUTRAL") or "NEUTRAL",
        sma_20=0.0,  # Not in summary
        sma_50=0.0,
        sma_200=0.0,
        atr_pct=float(polygon.get("atr_pct", 2.0) or 2.0),
        
        # Regime
        regime=decision.get("regime", "NEUTRAL"),
        regime_confidence=70.0,  # Not directly available
        vix=float(polygon.get("vix", 20) or 20),
        vix_state=polygon.get("vix_state", "NORMAL") or "NORMAL",
        
        # Sentiment
        sentiment_score=float(polygon.get("news_sentiment", 0) or 0),
        news_count_7d=int(decision.get("stocknews_summary", {}).get("news_count_7d", 0) or 
                         decision.get("cryptonews_summary", {}).get("news_count_7d", 0) or 0),
        
        # Signals
        weekly_signal=luxalgo.get("weekly"),
        daily_signal=luxalgo.get("daily"),
        h4_signal=luxalgo.get("h4"),
        
        # Data quality
        data_age_seconds=0.0,  # Fresh data
        missing_feeds=[],
    )


def apply_tier2_if_needed(decision: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply Tier 2 evaluation if conditions are met.
    
    This function:
    1. Checks if Tier 2 is enabled
    2. Checks if the decision should be escalated
    3. Runs Tier 2 if needed
    4. Applies deltas to the original decision
    
    Args:
        decision: The V6 APEX decision dictionary
        
    Returns:
        Modified decision with Tier 2 deltas applied (or original if not escalated)
    """
    # Skip if disabled
    if not TIER2_ENABLED:
        return decision
    
    # Skip if already vetoed
    if decision.get("veto"):
        return decision
    
    # Skip if no trade
    if decision.get("strength") == "NO_TRADE":
        return decision
    
    try:
        orchestrator = get_orchestrator()
        if orchestrator is None:
            logger.warning("Tier 2 orchestrator not available, skipping")
            return decision
        
        # Build Tier 1 pack
        tier1_pack = build_tier1_pack_from_decision(decision)
        
        # Check escalation
        escalation = orchestrator.should_escalate(tier1_pack)
        
        if not escalation.should_escalate:
            # No escalation needed
            decision["tier2"] = {
                "escalated": False,
                "reason": "No escalation triggers met",
            }
            return decision
        
        logger.info(
            f"[{decision['symbol']}] Escalating to Tier 2: {', '.join(escalation.reasons)}"
        )
        
        # Run Tier 2
        tier2_decision = orchestrator.run(
            tier1_pack=tier1_pack,
            escalation_reasons=escalation.reasons,
        )
        
        # Apply deltas
        original_confidence = decision.get("confidence", 50)
        
        # Build Tier 2 metadata
        tier2_meta = {
            "escalated": True,
            "reasons": escalation.reasons,
            "decision_id": tier2_decision.decision_id,
            "delta_type": tier2_decision.delta_type,
            "confidence_delta": tier2_decision.delta_confidence,
            "position_cap_delta": tier2_decision.delta_position_cap,
            "timing_recommendation": tier2_decision.delta_timing,
            "agent_count": tier2_decision.agent_count,
            "latency_ms": tier2_decision.latency_ms,
            "rationale": tier2_decision.rationale,
        }
        
        # Apply confidence delta
        if tier2_decision.delta_type != "NO_DELTA":
            new_confidence = original_confidence + tier2_decision.delta_confidence
            new_confidence = max(0, min(100, new_confidence))
            
            decision["confidence"] = round(new_confidence, 2)
            tier2_meta["original_confidence"] = original_confidence
            tier2_meta["adjusted_confidence"] = new_confidence
            
            # Recalculate strength based on new confidence
            if new_confidence >= 75:
                new_strength = "STRONG"
            elif new_confidence >= 55:
                new_strength = "MODERATE"
            elif new_confidence >= 35:
                new_strength = "WEAK"
            else:
                new_strength = "NO_TRADE"
            
            if new_strength != decision.get("strength"):
                tier2_meta["original_strength"] = decision.get("strength")
                tier2_meta["adjusted_strength"] = new_strength
                decision["strength"] = new_strength
            
            # Handle special delta types
            if tier2_decision.delta_type == "WAIT":
                decision["should_trade"] = False
                decision["reasoning"].append(
                    f"⏸️ TIER 2 WAIT: {tier2_decision.rationale[:100]}"
                )
            
            elif tier2_decision.delta_type == "NO_TRADE":
                decision["should_trade"] = False
                decision["strength"] = "NO_TRADE"
                decision["reasoning"].append(
                    f"🚫 TIER 2 NO_TRADE: {tier2_decision.rationale[:100]}"
                )
            
            elif tier2_decision.delta_type == "CONFIDENCE_DOWN":
                decision["reasoning"].append(
                    f"⚠️ TIER 2: Confidence adjusted "
                    f"{tier2_decision.delta_confidence:+.1f}% "
                    f"({tier2_decision.rationale[:80]})"
                )
            
            elif tier2_decision.delta_type == "CONFIDENCE_UP":
                decision["reasoning"].append(
                    f"✅ TIER 2: Confidence boosted "
                    f"{tier2_decision.delta_confidence:+.1f}% "
                    f"({tier2_decision.rationale[:80]})"
                )
        
        # Apply position cap reduction if specified
        if tier2_decision.delta_position_cap and decision.get("trade_setup"):
            original_position = decision["trade_setup"].get("position_pct", 0)
            new_position = original_position * tier2_decision.delta_position_cap
            decision["trade_setup"]["position_pct"] = round(new_position, 2)
            tier2_meta["original_position_pct"] = original_position
            tier2_meta["adjusted_position_pct"] = new_position
            decision["reasoning"].append(
                f"📉 TIER 2: Position reduced to {new_position:.1f}%"
            )
        
        # Update should_trade if confidence dropped below threshold
        if decision.get("strength") == "NO_TRADE":
            decision["should_trade"] = False
        
        decision["tier2"] = tier2_meta
        
        return decision
        
    except Exception as e:
        logger.error(f"Tier 2 evaluation failed: {e}", exc_info=True)
        decision["tier2"] = {
            "escalated": False,
            "error": str(e)[:100],
        }
        return decision


def get_tier2_health() -> Dict[str, Any]:
    """Get Tier 2 system health status."""
    if not TIER2_ENABLED:
        return {
            "enabled": False,
            "status": "disabled",
        }
    
    try:
        orchestrator = get_orchestrator()
        if orchestrator is None:
            return {
                "enabled": True,
                "status": "error",
                "error": "Orchestrator not initialized",
            }
        
        metrics = orchestrator.get_metrics()
        
        return {
            "enabled": True,
            "status": "healthy",
            "circuit_breaker": metrics.get("circuit_breaker", {}).get("state", "unknown"),
            "agent_count": metrics.get("registry", {}).get("agent_count", 0),
        }
        
    except Exception as e:
        return {
            "enabled": True,
            "status": "error",
            "error": str(e)[:100],
        }
