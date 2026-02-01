"""
LuxAlgo Webhook API

FastAPI routes for receiving LuxAlgo signals from TradingView.

Endpoints:
- POST /webhooks/luxalgo - Receive LuxAlgo signal from TradingView alert
- GET /signals/{symbol} - Get recent signals for a symbol
- GET /signals/{symbol}/latest - Get the latest signal
- GET /signals/consensus/{symbol} - Get signal consensus
- GET /signals/status - Get status of all tracked symbols

TradingView Alert Configuration:
-------------------------------
1. Set webhook URL to: https://your-server.com/webhooks/luxalgo
2. Use this JSON template in the alert message:

{
    "action": "BUY",
    "symbol": "{{ticker}}",
    "exchange": "{{exchange}}",
    "price": {{close}},
    "timeframe": "{{interval}}",
    "signal_type": "Bullish Confirmation",
    "confirmations": 12,
    "trend_strength": 54.04,
    "trend_tracer": "bullish",
    "smart_trail": "bullish",
    "neo_cloud": "bullish",
    "time": "{{time}}"
}

For SELL signals, change "action" to "SELL" and "signal_type" to "Bearish Confirmation"
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import json

logger = logging.getLogger(__name__)

# Try FastAPI import
try:
    from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from ..signals.luxalgo_webhook import (
    parse_luxalgo_webhook,
    get_signal_store,
    LuxAlgoSignal
)
from ..signals.fusion_engine import SignalFusionEngine


# Pydantic models for API
if HAS_FASTAPI:
    
    class LuxAlgoWebhookPayload(BaseModel):
        """Webhook payload from TradingView LuxAlgo alert."""
        action: str = Field(..., description="BUY or SELL")
        symbol: str = Field(..., description="Asset symbol")
        exchange: Optional[str] = Field("UNKNOWN", description="Exchange name")
        price: float = Field(..., description="Price at signal time")
        timeframe: Optional[str] = Field("4h", description="Chart timeframe")
        signal_type: Optional[str] = Field(None, description="LuxAlgo signal type")
        confirmations: Optional[int] = Field(1, description="Number of confirmations (1-12)")
        trend_strength: Optional[float] = Field(50, description="Trend strength (0-100)")
        trend_tracer: Optional[str] = Field(None, description="Trend Tracer state")
        smart_trail: Optional[str] = Field(None, description="Smart Trail state")
        neo_cloud: Optional[str] = Field(None, description="Neo Cloud state")
        trend_catcher: Optional[str] = Field(None, description="Trend Catcher state")
        message: Optional[str] = Field(None, description="Alert message")
        time: Optional[str] = Field(None, description="Signal timestamp")
        
        class Config:
            extra = "allow"  # Allow additional fields from TradingView
    
    
    class SignalResponse(BaseModel):
        """Response for signal queries."""
        symbol: str
        signals: List[Dict]
        consensus: Dict
        latest: Optional[Dict]
        count: int
    
    
    class WebhookResponse(BaseModel):
        """Response for webhook receipt."""
        status: str
        signal_id: str
        symbol: str
        action: str
        confidence: float
        is_strong: bool
        message: str


def create_luxalgo_router() -> 'APIRouter':
    """Create the LuxAlgo API router."""
    if not HAS_FASTAPI:
        raise ImportError("FastAPI not installed. Run: pip install fastapi")
    
    router = APIRouter(tags=["LuxAlgo Signals"])
    
    # Signal processing callbacks
    signal_callbacks = []
    
    def register_callback(callback):
        """Register a callback for new signals."""
        signal_callbacks.append(callback)
    
    async def process_signal_background(signal: LuxAlgoSignal):
        """Process signal in background (notifications, trading, etc.)."""
        logger.info(
            f"Processing LuxAlgo signal: {signal.symbol} {signal.action} "
            f"(conf={signal.confidence:.0%}, strong={signal.is_strong})"
        )
        
        # Call registered callbacks
        for callback in signal_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(signal)
                else:
                    callback(signal)
            except Exception as e:
                logger.error(f"Signal callback failed: {e}")
        
        # Log strong signals prominently
        if signal.is_strong:
            logger.info(
                f"🚀 STRONG SIGNAL: {signal.symbol} {signal.action} "
                f"({signal.confirmations} confirmations on {signal.timeframe})"
            )
    
    @router.post(
        "/webhooks/luxalgo",
        response_model=WebhookResponse,
        summary="Receive LuxAlgo Signal",
        description="Webhook endpoint for TradingView LuxAlgo alerts"
    )
    async def receive_luxalgo_webhook(
        request: Request,
        background_tasks: BackgroundTasks
    ):
        """
        Receive a webhook from TradingView LuxAlgo alert.
        
        Configure your TradingView alert with:
        - Webhook URL: https://your-domain.com/webhooks/luxalgo
        - Alert message: JSON with action, symbol, price, confirmations, etc.
        """
        try:
            # Get raw body
            body = await request.json()
            
            logger.info(f"Received LuxAlgo webhook: {json.dumps(body)[:500]}")
            
            # Parse signal
            signal = parse_luxalgo_webhook(body)
            
            # Store signal
            store = get_signal_store()
            store.add_signal(signal)
            
            # Process in background
            background_tasks.add_task(process_signal_background, signal)
            
            return WebhookResponse(
                status="received",
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                action=signal.action,
                confidence=round(signal.confidence, 4),
                is_strong=signal.is_strong,
                message=f"{signal.action} signal for {signal.symbol} with {signal.confirmations} confirmations"
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in webhook: {e}")
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
        except Exception as e:
            logger.error(f"Error processing webhook: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get(
        "/signals/{symbol}",
        response_model=SignalResponse,
        summary="Get Recent Signals",
        description="Get recent LuxAlgo signals for a symbol"
    )
    async def get_signals(
        symbol: str,
        hours: int = 24,
        min_confirmations: int = 1
    ):
        """
        Get recent signals for a symbol.
        
        Args:
            symbol: Asset symbol (e.g., ETHUSD, BTCUSD, AAPL)
            hours: Lookback period in hours (default 24)
            min_confirmations: Minimum confirmations to include (default 1)
        """
        store = get_signal_store()
        symbol = symbol.upper()
        
        signals = store.get_recent_signals(symbol, hours, min_confirmations)
        consensus = store.get_signal_consensus(symbol, hours)
        latest = store.get_latest_signal(symbol)
        
        return SignalResponse(
            symbol=symbol,
            signals=[s.to_dict() for s in signals],
            consensus=consensus,
            latest=latest.to_dict() if latest else None,
            count=len(signals)
        )
    
    @router.get(
        "/signals/{symbol}/latest",
        summary="Get Latest Signal",
        description="Get the most recent LuxAlgo signal for a symbol"
    )
    async def get_latest_signal(symbol: str):
        """Get the latest signal for a symbol."""
        store = get_signal_store()
        signal = store.get_latest_signal(symbol.upper())
        
        if signal:
            return {
                "symbol": symbol.upper(),
                "signal": signal.to_dict(),
                "received_at": signal.timestamp.isoformat()
            }
        else:
            return {
                "symbol": symbol.upper(),
                "signal": None,
                "message": "No signals received for this symbol"
            }
    
    @router.get(
        "/signals/{symbol}/consensus",
        summary="Get Signal Consensus",
        description="Get consensus from recent signals for a symbol"
    )
    async def get_signal_consensus(symbol: str, hours: int = 24):
        """Get signal consensus from recent signals."""
        store = get_signal_store()
        consensus = store.get_signal_consensus(symbol.upper(), hours)
        
        return {
            "symbol": symbol.upper(),
            "period_hours": hours,
            "consensus": consensus
        }
    
    @router.get(
        "/signals/{symbol}/strong",
        summary="Get Strong Signals",
        description="Get only strong signals (4+ confirmations on 4h+)"
    )
    async def get_strong_signals(symbol: str, hours: int = 24):
        """Get only strong signals."""
        store = get_signal_store()
        signals = store.get_strong_signals(symbol.upper(), hours)
        
        return {
            "symbol": symbol.upper(),
            "period_hours": hours,
            "strong_signals": [s.to_dict() for s in signals],
            "count": len(signals)
        }
    
    @router.get(
        "/signals/status",
        summary="Get Status",
        description="Get status of all tracked symbols"
    )
    async def get_status():
        """Get current status of all tracked symbols."""
        store = get_signal_store()
        
        status = {}
        for symbol in store.get_all_symbols():
            signal = store.latest_signal[symbol]
            consensus = store.get_signal_consensus(symbol, hours=24)
            
            status[symbol] = {
                "direction": store.active_direction.get(symbol),
                "last_signal": signal.timestamp.isoformat(),
                "last_action": signal.action,
                "confidence": round(signal.confidence, 4),
                "timeframe": signal.timeframe,
                "is_strong": signal.is_strong,
                "consensus": consensus.get('direction'),
                "consensus_confidence": consensus.get('confidence', 0)
            }
        
        return {
            "tracked_symbols": len(status),
            "symbols": status,
            "store_stats": store.get_stats()
        }
    
    @router.delete(
        "/signals/clear",
        summary="Clear Signals",
        description="Clear all stored signals (for testing)"
    )
    async def clear_signals():
        """Clear all signals (for testing)."""
        store = get_signal_store()
        store.clear()
        return {"status": "cleared", "message": "All signals cleared"}
    
    # Store router attributes
    router.register_callback = register_callback
    
    return router


# Convenience function to get fused signal via API
def create_fusion_router() -> 'APIRouter':
    """Create router for signal fusion endpoints."""
    if not HAS_FASTAPI:
        raise ImportError("FastAPI not installed")
    
    router = APIRouter(prefix="/fusion", tags=["Signal Fusion"])
    engine = SignalFusionEngine()
    
    @router.get(
        "/{symbol}",
        summary="Get Fused Signal",
        description="Get combined signal from all sources for a symbol"
    )
    async def get_fused_signal(
        symbol: str,
        sentiment: Optional[float] = None,
        regime: Optional[str] = None
    ):
        """
        Get fused signal combining LuxAlgo, ML, and other sources.
        
        Args:
            symbol: Asset symbol
            sentiment: Pre-computed sentiment score (-1 to +1)
            regime: Market regime (BULL, BEAR, SIDEWAYS, VOLATILE)
        """
        signal = engine.generate_fused_signal(
            symbol=symbol,
            sentiment=sentiment,
            regime=regime
        )
        
        return {
            "symbol": symbol.upper(),
            "signal": signal.to_dict(),
            "is_actionable": signal.is_actionable,
            "summary": str(signal)
        }
    
    @router.get(
        "/stats",
        summary="Get Source Stats",
        description="Get statistics about each signal source"
    )
    async def get_source_stats():
        """Get statistics about signal sources."""
        return {
            "source_stats": engine.get_source_stats(),
            "prediction_count": len(engine.prediction_history)
        }
    
    return router


# Import for async
import asyncio


def add_luxalgo_routes(app: 'FastAPI'):
    """
    Add LuxAlgo signal routes to a FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    if not HAS_FASTAPI:
        logger.warning("FastAPI not installed - LuxAlgo routes not added")
        return
    
    try:
        # Add LuxAlgo webhook routes
        luxalgo_router = create_luxalgo_router()
        app.include_router(luxalgo_router)
        
        # Add fusion routes
        fusion_router = create_fusion_router()
        app.include_router(fusion_router)
        
        logger.info("✅ LuxAlgo and Fusion routes added to API")
    except Exception as e:
        logger.error(f"Failed to add LuxAlgo routes: {e}")
        raise
