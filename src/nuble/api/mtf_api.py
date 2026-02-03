"""
NUBLE ELITE: Multi-Timeframe API

FastAPI routes for the institutional multi-timeframe signal system.

Endpoints:
- POST /mtf/webhook - Receive signals from any timeframe
- GET /mtf/decision/{symbol} - Get trading decision
- GET /mtf/signals/{symbol} - Get all signals for a symbol
- GET /mtf/alignment/{symbol} - Get timeframe alignment
- GET /mtf/veto/{symbol} - Check if trading is vetoed
- GET /mtf/status - Get system status

TradingView Alert Configuration:
-------------------------------
Set up SEPARATE alerts for each timeframe:
1. Weekly chart → Webhook URL: /mtf/webhook
2. Daily chart → Same webhook
3. 4H chart → Same webhook

Each alert will be processed and stored by timeframe.
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Try FastAPI import
try:
    from fastapi import APIRouter, HTTPException, Request, BackgroundTasks, Query
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from ..signals.mtf_fusion import (
    MTFFusionEngine, 
    TradingDecision, 
    get_mtf_engine
)
from ..signals.timeframe_manager import (
    TimeframeSignal,
    Timeframe,
    parse_mtf_webhook
)
from ..signals.veto_engine import VetoResult


if HAS_FASTAPI:
    
    class MTFWebhookPayload(BaseModel):
        """Webhook payload for multi-timeframe signals."""
        # Required
        action: str = Field(..., description="BUY, SELL, or NEUTRAL")
        symbol: str = Field(..., description="Asset symbol")
        timeframe: str = Field(..., description="1W, 1D, 4h, or 1h")
        price: float = Field(..., description="Current price")
        
        # Optional LuxAlgo fields
        exchange: Optional[str] = Field("UNKNOWN", description="Exchange name")
        signal_type: Optional[str] = Field(None, description="Signal type")
        strength: Optional[str] = Field("normal", description="normal or strong")
        confirmations: Optional[int] = Field(1, description="1-12")
        trend_strength: Optional[float] = Field(50, description="0-100")
        smart_trail_sentiment: Optional[str] = Field(None, description="bullish/bearish")
        smart_trail_level: Optional[float] = Field(None, description="Smart Trail price")
        neo_cloud_sentiment: Optional[str] = Field(None, description="bullish/bearish")
        reversal_zone_upper: Optional[float] = Field(None)
        reversal_zone_lower: Optional[float] = Field(None)
        ml_classification: Optional[int] = Field(None, description="1-4")
        
        # Timestamp
        time: Optional[str] = Field(None, description="Signal timestamp")
        timestamp: Optional[str] = Field(None, description="Alternative timestamp field")
        
        class Config:
            extra = "allow"
    
    
    class DecisionResponse(BaseModel):
        """Response for trading decision."""
        symbol: str
        can_trade: bool
        action: str
        direction: int
        strength: str
        confidence: float
        position_size_pct: Optional[float]
        position_dollars: Optional[float]
        entry_price: Optional[float]
        stop_loss: Optional[float]
        take_profit_1: Optional[float]
        take_profit_2: Optional[float]
        reasoning: list
        timeframes: dict
    
    
    class WebhookResponse(BaseModel):
        """Response for webhook receipt."""
        status: str
        signal_id: str
        symbol: str
        timeframe: str
        action: str
        freshness: float
        message: str


def create_mtf_router() -> 'APIRouter':
    """Create the MTF API router."""
    if not HAS_FASTAPI:
        raise ImportError("FastAPI not installed")
    
    router = APIRouter(prefix="/mtf", tags=["Multi-Timeframe"])
    
    # Get engine instance
    engine = get_mtf_engine()
    
    @router.post(
        "/webhook",
        response_model=WebhookResponse,
        summary="Receive MTF Signal",
        description="Webhook endpoint for TradingView alerts from any timeframe"
    )
    async def receive_mtf_webhook(
        request: Request,
        background_tasks: BackgroundTasks
    ):
        """
        Receive a signal from TradingView for any timeframe.
        
        The timeframe is determined from the payload.
        Signals are stored and used for multi-timeframe fusion.
        """
        try:
            # Parse JSON payload
            try:
                payload = await request.json()
            except Exception:
                body = await request.body()
                payload = {"raw": body.decode(), "action": "UNKNOWN"}
            
            logger.info(f"MTF webhook received: {payload}")
            
            # Parse into TimeframeSignal
            signal = parse_mtf_webhook(payload)
            
            # Add to engine
            engine.add_signal(signal)
            
            # Background processing
            background_tasks.add_task(process_mtf_signal, signal)
            
            return WebhookResponse(
                status="received",
                signal_id=signal.signal_id,
                symbol=signal.symbol,
                timeframe=signal.timeframe.value,
                action=signal.action,
                freshness=signal.freshness,
                message=f"{signal.timeframe.value} {signal.action} signal for {signal.symbol}"
            )
            
        except Exception as e:
            logger.error(f"MTF webhook error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
    
    @router.get(
        "/decision/{symbol}",
        summary="Get Trading Decision",
        description="Get complete multi-timeframe trading decision"
    )
    async def get_decision(
        symbol: str,
        current_price: Optional[float] = Query(None, description="Current price"),
        regime: Optional[str] = Query("NORMAL", description="Market regime"),
        portfolio_value: Optional[float] = Query(None, description="Portfolio value")
    ):
        """
        Generate a trading decision using all available timeframe signals.
        
        Returns complete analysis with position sizing, stops, and reasoning.
        """
        decision = engine.generate_decision(
            symbol=symbol.upper(),
            current_price=current_price,
            regime=regime or "NORMAL",
            portfolio_value=portfolio_value
        )
        
        return decision.to_dict()
    
    @router.get(
        "/signals/{symbol}",
        summary="Get All Signals",
        description="Get all current signals for a symbol across timeframes"
    )
    async def get_signals(symbol: str):
        """Get all current signals for a symbol."""
        signals = engine.get_signals(symbol.upper())
        alignment = engine.get_alignment(symbol.upper())
        
        return {
            "symbol": symbol.upper(),
            "signals": signals,
            "alignment": alignment,
            "timestamp": datetime.now().isoformat()
        }
    
    @router.get(
        "/alignment/{symbol}",
        summary="Get Timeframe Alignment",
        description="Check if timeframes are aligned for trading"
    )
    async def get_alignment(symbol: str):
        """
        Get alignment analysis across all timeframes.
        
        Returns alignment score, direction, and breakdown.
        """
        alignment = engine.get_alignment(symbol.upper())
        return {
            "symbol": symbol.upper(),
            **alignment,
            "timestamp": datetime.now().isoformat()
        }
    
    @router.get(
        "/veto/{symbol}",
        summary="Check Veto Status",
        description="Check if trading is vetoed for a symbol"
    )
    async def check_veto(symbol: str):
        """
        Check if the veto engine would block a trade.
        
        Returns veto decision with full reasoning.
        """
        veto = engine.check_veto(symbol.upper())
        return {
            "symbol": symbol.upper(),
            **veto.to_dict(),
            "timestamp": datetime.now().isoformat()
        }
    
    @router.get(
        "/status",
        summary="Get System Status",
        description="Get status of the MTF system"
    )
    async def get_status():
        """Get status of all tracked symbols and system health."""
        status = engine.get_status()
        status["timestamp"] = datetime.now().isoformat()
        return status
    
    @router.post(
        "/cleanup",
        summary="Cleanup Expired Signals",
        description="Remove all expired signals from the system"
    )
    async def cleanup_expired():
        """Cleanup expired signals."""
        removed = engine.cleanup()
        return {
            "status": "success",
            "signals_removed": removed,
            "timestamp": datetime.now().isoformat()
        }
    
    @router.get(
        "/config",
        summary="Get Configuration",
        description="Get current MTF system configuration"
    )
    async def get_config():
        """Get current configuration."""
        return {
            "portfolio_value": engine.portfolio_value,
            "max_risk": engine.position_calc.max_risk,
            "max_position": engine.position_calc.max_position,
            "kelly_fraction": engine.position_calc.kelly_fraction,
            "timeframe_weights": {
                tf.value: tf.weight for tf in Timeframe
            },
            "signal_expiry_hours": {
                tf.value: tf.max_age_hours for tf in Timeframe
            },
            "optimal_sensitivity": {
                tf.value: tf.optimal_sensitivity for tf in Timeframe
            }
        }
    
    return router


async def process_mtf_signal(signal: TimeframeSignal):
    """Process MTF signal in background."""
    logger.info(
        f"Processing MTF signal: {signal.timeframe.value} {signal.symbol} "
        f"{signal.action} (fresh={signal.freshness:.0%})"
    )
    
    # Log strong signals
    if signal.is_strong:
        logger.info(
            f"🚀 STRONG {signal.timeframe.value} SIGNAL: {signal.symbol} {signal.action} "
            f"({signal.confirmations} confirmations)"
        )


def add_mtf_routes(app: 'FastAPI'):
    """
    Add MTF routes to a FastAPI application.
    
    Args:
        app: FastAPI application instance
    """
    if not HAS_FASTAPI:
        logger.warning("FastAPI not installed - MTF routes not added")
        return
    
    try:
        mtf_router = create_mtf_router()
        app.include_router(mtf_router)
        logger.info("✅ MTF routes added to API")
    except Exception as e:
        logger.error(f"Failed to add MTF routes: {e}")
        raise
