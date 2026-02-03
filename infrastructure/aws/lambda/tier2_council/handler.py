"""
NUBLE Tier 2 Lambda Handler
================================

Lambda function that receives Tier 1 decisions and produces Tier 2 deltas.

Invocation modes:
1. Synchronous: Called directly by Tier 1 for high-priority escalations
2. Async: Called via SQS for batch processing

Environment Variables:
- TIER2_DECISIONS_TABLE: DynamoDB table for decisions
- TIER2_AGENT_RUNS_TABLE: DynamoDB table for agent runs  
- TIER2_OUTCOMES_TABLE: DynamoDB table for outcomes
- BEDROCK_MODEL_ID: Bedrock model to use (default: anthropic.claude-3-haiku-20240307-v1:0)
- TIER2_ENABLED: Enable/disable Tier 2 (default: true)
- LOG_LEVEL: Logging level (default: INFO)
"""

import os
import json
import logging
import traceback
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Configure logging
log_level = os.environ.get("LOG_LEVEL", "INFO")
logging.basicConfig(level=log_level)
logger = logging.getLogger(__name__)

# Lazy load heavy imports
_orchestrator = None


def get_orchestrator():
    """Lazy load the Tier 2 orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        from src.institutional.tier2 import (
            Tier2Orchestrator,
            Tier2Config,
            DynamoDBStore,
            InMemoryStore,
        )
        
        use_dynamodb = os.environ.get("USE_DYNAMODB", "true").lower() == "true"
        
        if use_dynamodb:
            store = DynamoDBStore(
                decisions_table=os.environ.get("TIER2_DECISIONS_TABLE", "nuble-tier2-decisions"),
                runs_table=os.environ.get("TIER2_AGENT_RUNS_TABLE", "nuble-tier2-agent-runs"),
                outcomes_table=os.environ.get("TIER2_OUTCOMES_TABLE", "nuble-tier2-outcomes"),
            )
        else:
            store = InMemoryStore()
        
        # Load config from environment or use defaults
        config = Tier2Config(
            bedrock_model_id=os.environ.get("BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"),
            light_round_timeout_ms=int(os.environ.get("LIGHT_ROUND_TIMEOUT_MS", "3000")),
            deep_round_timeout_ms=int(os.environ.get("DEEP_ROUND_TIMEOUT_MS", "8000")),
            max_tokens_light=int(os.environ.get("MAX_TOKENS_LIGHT", "350")),
            max_tokens_deep=int(os.environ.get("MAX_TOKENS_DEEP", "2500")),
        )
        
        _orchestrator = Tier2Orchestrator(
            config=config,
            store=store,
        )
    
    return _orchestrator


def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Lambda handler for Tier 2 Council-of-Experts.
    
    Expected event format:
    {
        "tier1_decision": {
            "symbol": "AAPL",
            "action": "BUY",
            "confidence": 78.5,
            "direction": "BULLISH",
            "price": 185.50,
            "rsi": 58.2,
            "macd_value": 0.52,
            "macd_signal": 0.45,
            "trend_state": "UPTREND",
            "sma_20": 183.2,
            "sma_50": 180.5,
            "sma_200": 175.0,
            "atr_pct": 1.85,
            "regime": "BULLISH",
            "regime_confidence": 72.0,
            "vix": 15.5,
            "vix_state": "LOW",
            "sentiment_score": 0.35,
            "news_count_7d": 12,
            "weekly_signal": {"action": "BUY", ...},
            "daily_signal": {"action": "BUY", ...},
            "h4_signal": {"action": "NEUTRAL", ...},
            "data_age_seconds": 5,
            "missing_feeds": [],
            ...
        },
        "escalation_reasons": ["signal_conflict", "high_volatility"],
        "portfolio_snapshot": {...},  // optional
        "force_deep": false
    }
    
    Response format:
    {
        "statusCode": 200,
        "body": {
            "decision_id": "abc123",
            "symbol": "AAPL",
            "delta_type": "CONFIDENCE_DOWN",
            "delta_confidence": -12.0,
            "delta_position_cap": 0.75,
            "delta_timing": "delay_30m",
            "final_action": "BUY",
            "final_confidence": 66.5,
            "rationale": "Signal conflict between weekly and 4H requires caution...",
            "agent_count": 5,
            "latency_ms": 2340.5,
            "circuit_breaker_state": "closed"
        }
    }
    """
    start_time = datetime.now(timezone.utc)
    
    try:
        # Check if Tier 2 is enabled
        if os.environ.get("TIER2_ENABLED", "true").lower() != "true":
            return _no_delta_response("Tier 2 is disabled")
        
        # Handle SQS batch events
        if "Records" in event:
            return _handle_sqs_batch(event, context)
        
        # Handle direct invocation
        return _handle_direct(event, context)
        
    except Exception as e:
        logger.error(f"Handler error: {e}")
        logger.error(traceback.format_exc())
        
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "delta_type": "NO_DELTA",
                "rationale": f"Tier 2 error: {str(e)[:100]}",
            }),
        }


def _handle_direct(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Handle direct Lambda invocation."""
    from src.institutional.tier2.schemas import Tier1DecisionPack
    
    # Parse tier1 decision
    tier1_data = event.get("tier1_decision", event)
    escalation_reasons = event.get("escalation_reasons", [])
    portfolio_snapshot = event.get("portfolio_snapshot")
    force_deep = event.get("force_deep", False)
    
    # Build Tier1DecisionPack
    tier1_pack = _build_tier1_pack(tier1_data)
    
    # Get orchestrator and run
    orchestrator = get_orchestrator()
    
    # Check if we should escalate (if reasons not provided)
    if not escalation_reasons:
        escalation_result = orchestrator.should_escalate(tier1_pack, portfolio_snapshot)
        if not escalation_result.should_escalate:
            return _no_delta_response("No escalation needed")
        escalation_reasons = escalation_result.reasons
    
    # Run Tier 2 pipeline
    decision = orchestrator.run(
        tier1_pack=tier1_pack,
        escalation_reasons=escalation_reasons,
        portfolio_snapshot=portfolio_snapshot,
        force_deep=force_deep,
    )
    
    # Get circuit breaker state
    cb_metrics = orchestrator.circuit_breaker.get_metrics()
    
    return {
        "statusCode": 200,
        "body": json.dumps({
            "decision_id": decision.decision_id,
            "symbol": decision.symbol,
            "timestamp": decision.timestamp,
            "delta_type": decision.delta_type,
            "delta_confidence": decision.delta_confidence,
            "delta_position_cap": decision.delta_position_cap,
            "delta_timing": decision.delta_timing,
            "tier1_action": decision.tier1_action,
            "tier1_confidence": decision.tier1_confidence,
            "final_action": decision.final_action,
            "final_confidence": decision.final_confidence,
            "rationale": decision.rationale,
            "agent_count": decision.agent_count,
            "latency_ms": decision.latency_ms,
            "escalation_reasons": decision.escalation_reasons,
            "cross_exam_triggered": decision.cross_exam_triggered,
            "circuit_breaker_state": cb_metrics["state"],
        }),
    }


def _handle_sqs_batch(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Handle SQS batch of Tier 1 decisions."""
    results = []
    
    for record in event.get("Records", []):
        try:
            body = json.loads(record.get("body", "{}"))
            result = _handle_direct(body, context)
            results.append({
                "messageId": record.get("messageId"),
                "result": json.loads(result.get("body", "{}")),
            })
        except Exception as e:
            logger.error(f"Error processing SQS record: {e}")
            results.append({
                "messageId": record.get("messageId"),
                "error": str(e),
            })
    
    return {
        "statusCode": 200,
        "body": json.dumps({
            "processed": len(results),
            "results": results,
        }),
    }


def _build_tier1_pack(data: Dict[str, Any]):
    """Build a Tier1DecisionPack from raw data."""
    from src.institutional.tier2.schemas import Tier1DecisionPack
    
    return Tier1DecisionPack(
        symbol=data.get("symbol", "UNKNOWN"),
        action=data.get("action", "WAIT"),
        confidence=float(data.get("confidence", 50.0)),
        direction=data.get("direction", "NEUTRAL"),
        price=float(data.get("price", 0.0)),
        
        # Technical
        rsi=float(data.get("rsi", 50.0)),
        macd_value=float(data.get("macd_value", 0.0)),
        macd_signal=float(data.get("macd_signal", 0.0)),
        trend_state=data.get("trend_state", "NEUTRAL"),
        sma_20=float(data.get("sma_20", 0.0)),
        sma_50=float(data.get("sma_50", 0.0)),
        sma_200=float(data.get("sma_200", 0.0)),
        atr_pct=float(data.get("atr_pct", 2.0)),
        
        # Regime
        regime=data.get("regime", "NEUTRAL"),
        regime_confidence=float(data.get("regime_confidence", 50.0)),
        vix=float(data.get("vix", 20.0)),
        vix_state=data.get("vix_state", "NORMAL"),
        
        # Sentiment
        sentiment_score=float(data.get("sentiment_score", 0.0)),
        news_count_7d=int(data.get("news_count_7d", 0)),
        
        # Signals (multi-timeframe)
        weekly_signal=data.get("weekly_signal"),
        daily_signal=data.get("daily_signal"),
        h4_signal=data.get("h4_signal"),
        
        # Data quality
        data_age_seconds=float(data.get("data_age_seconds", 0.0)),
        missing_feeds=data.get("missing_feeds", []),
        
        # Portfolio context
        current_position=float(data.get("current_position", 0.0)),
        sector_exposure_pct=float(data.get("sector_exposure_pct", 0.0)),
    )


def _no_delta_response(reason: str) -> Dict[str, Any]:
    """Return a no-delta response."""
    return {
        "statusCode": 200,
        "body": json.dumps({
            "delta_type": "NO_DELTA",
            "delta_confidence": 0.0,
            "delta_position_cap": None,
            "delta_timing": None,
            "rationale": reason,
            "agent_count": 0,
            "latency_ms": 0,
        }),
    }


# Health check endpoint
def health_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """Health check handler."""
    try:
        orchestrator = get_orchestrator()
        metrics = orchestrator.get_metrics()
        
        return {
            "statusCode": 200,
            "body": json.dumps({
                "status": "healthy",
                "tier2_version": "1.0.0",
                "circuit_breaker": metrics["circuit_breaker"]["state"],
                "agent_count": metrics["registry"]["agent_count"],
            }),
        }
    except Exception as e:
        return {
            "statusCode": 500,
            "body": json.dumps({
                "status": "unhealthy",
                "error": str(e),
            }),
        }
