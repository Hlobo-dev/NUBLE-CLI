"""
LuxAlgo Webhook Receiver

Receives signals from TradingView LuxAlgo alerts via webhook.
Parses LuxAlgo indicators including:
- Buy/Sell Confirmations (1-12)
- Trend Tracer
- Smart Trail
- Neo Cloud
- Trend Catcher
- Trend Strength

This is a PROVEN signal source from TradingView's premium LuxAlgo indicator.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import json
import logging
from collections import deque
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LuxAlgoSignalType(Enum):
    """Types of LuxAlgo signals."""
    BUY_CONFIRMATION = "buy_confirmation"
    SELL_CONFIRMATION = "sell_confirmation"
    BULLISH_CONFIRMATION = "bullish_confirmation"
    BEARISH_CONFIRMATION = "bearish_confirmation"
    TREND_TRACER_BULLISH = "trend_tracer_bullish"
    TREND_TRACER_BEARISH = "trend_tracer_bearish"
    SMART_TRAIL_BULLISH = "smart_trail_bullish"
    SMART_TRAIL_BEARISH = "smart_trail_bearish"
    TREND_CATCHER_BULLISH = "trend_catcher_bullish"
    TREND_CATCHER_BEARISH = "trend_catcher_bearish"
    NEO_CLOUD_BULLISH = "neo_cloud_bullish"
    NEO_CLOUD_BEARISH = "neo_cloud_bearish"
    UPPER_ZONE = "upper_zone"
    LOWER_ZONE = "lower_zone"


# Timeframe strength multipliers
# Higher timeframes are more reliable
TIMEFRAME_MULTIPLIERS = {
    "1m": 0.40,
    "3m": 0.45,
    "5m": 0.50,
    "15m": 0.60,
    "30m": 0.70,
    "1h": 0.80,
    "2h": 0.85,
    "4h": 0.95,
    "1D": 1.00,
    "1W": 1.00,
    "1M": 1.00,
    # Alternative formats
    "1": 0.40,
    "3": 0.45,
    "5": 0.50,
    "15": 0.60,
    "30": 0.70,
    "60": 0.80,
    "120": 0.85,
    "240": 0.95,
    "D": 1.00,
    "W": 1.00,
}


@dataclass
class LuxAlgoSignal:
    """
    A signal received from LuxAlgo via TradingView webhook.
    
    Attributes:
        signal_id: Unique identifier for this signal
        timestamp: When the signal was received
        symbol: Asset symbol (e.g., 'ETHUSD', 'BTCUSD', 'AAPL')
        exchange: Exchange name (e.g., 'COINBASE', 'BINANCE')
        timeframe: Chart timeframe (e.g., '4h', '1D')
        signal_type: Type of LuxAlgo signal
        action: 'BUY' or 'SELL'
        price: Price at signal time
        confirmations: Number of LuxAlgo confirmations (1-12)
        trend_strength: Trend strength indicator (0-100)
        raw_message: Original message from webhook
        metadata: Additional data from webhook
    """
    signal_id: str
    timestamp: datetime
    symbol: str
    exchange: str
    timeframe: str
    signal_type: LuxAlgoSignalType
    action: str                 # "BUY" or "SELL"
    price: float
    confirmations: int          # Number of confirmations (1-12)
    trend_strength: float       # 0-100
    
    # Additional context from alert
    raw_message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Optional indicator values
    trend_tracer_bullish: bool = False
    smart_trail_bullish: bool = False
    neo_cloud_bullish: bool = False
    trend_catcher_bullish: bool = False
    in_upper_zone: bool = False
    in_lower_zone: bool = False
    
    @property
    def is_bullish(self) -> bool:
        return self.action == "BUY" or "bullish" in self.signal_type.value.lower()
    
    @property
    def is_bearish(self) -> bool:
        return self.action == "SELL" or "bearish" in self.signal_type.value.lower()
    
    @property
    def is_strong(self) -> bool:
        """
        Signal is strong if 4+ confirmations on 4h+ timeframe.
        
        Based on LuxAlgo best practices:
        - Higher timeframe = more reliable
        - More confirmations = stronger signal
        """
        strong_timeframes = ["4h", "1D", "1W", "1M", "240", "D", "W"]
        return (
            self.confirmations >= 4 and 
            self.timeframe in strong_timeframes
        )
    
    @property
    def confirmation_score(self) -> float:
        """Normalized confirmation score (0 to 1)."""
        return min(1.0, self.confirmations / 12)
    
    @property
    def timeframe_multiplier(self) -> float:
        """Get reliability multiplier based on timeframe."""
        return TIMEFRAME_MULTIPLIERS.get(self.timeframe, 0.7)
    
    @property
    def indicator_agreement_count(self) -> int:
        """Count how many indicators agree with the signal direction."""
        if self.is_bullish:
            return sum([
                self.trend_tracer_bullish,
                self.smart_trail_bullish,
                self.neo_cloud_bullish,
                self.trend_catcher_bullish
            ])
        else:
            return sum([
                not self.trend_tracer_bullish,
                not self.smart_trail_bullish,
                not self.neo_cloud_bullish,
                not self.trend_catcher_bullish
            ])
    
    @property
    def confidence(self) -> float:
        """
        Calculate confidence score from multiple factors.
        
        Formula:
        - Base: confirmation score (0-1) from confirmations/12
        - Multiplied by: timeframe reliability (0.4-1.0)
        - Boosted by: trend strength (adds 0-0.3)
        - Boosted by: indicator agreement (adds 0-0.2)
        
        Returns:
            Float between 0 and 1
        """
        # Base confidence from confirmations (1-12 scale)
        conf_score = self.confirmation_score
        
        # Timeframe multiplier (higher timeframes = more reliable)
        tf_mult = self.timeframe_multiplier
        
        # Trend strength factor (0-100 -> 0-0.3 bonus)
        trend_bonus = (self.trend_strength / 100) * 0.3 if self.trend_strength else 0
        
        # Indicator agreement bonus (0-4 indicators -> 0-0.2 bonus)
        agreement_bonus = (self.indicator_agreement_count / 4) * 0.2
        
        # Combined confidence
        confidence = (conf_score * tf_mult) + trend_bonus + agreement_bonus
        
        return min(1.0, confidence)
    
    @property
    def normalized_direction(self) -> float:
        """
        Get direction as normalized float (-1 to +1).
        
        Factors in:
        - Base direction (BUY = +1, SELL = -1)
        - Confirmation strength
        - Trend strength
        """
        base = 1.0 if self.is_bullish else -1.0
        
        # Scale by confirmation strength (50% base + 50% from confirmations)
        strength = 0.5 + (0.5 * self.confirmation_score)
        
        return base * strength
    
    def to_dict(self) -> Dict:
        return {
            'signal_id': self.signal_id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'exchange': self.exchange,
            'timeframe': self.timeframe,
            'signal_type': self.signal_type.value,
            'action': self.action,
            'price': self.price,
            'confirmations': self.confirmations,
            'trend_strength': self.trend_strength,
            'confidence': round(self.confidence, 4),
            'normalized_direction': round(self.normalized_direction, 4),
            'is_strong': self.is_strong,
            'is_bullish': self.is_bullish,
            'indicator_agreement': self.indicator_agreement_count,
            'trend_tracer_bullish': self.trend_tracer_bullish,
            'smart_trail_bullish': self.smart_trail_bullish,
            'neo_cloud_bullish': self.neo_cloud_bullish
        }
    
    def __repr__(self) -> str:
        return (
            f"LuxAlgoSignal({self.symbol} {self.action} "
            f"{self.confirmations}conf {self.timeframe} "
            f"conf={self.confidence:.0%})"
        )


class LuxAlgoSignalStore:
    """
    Stores and manages LuxAlgo signals.
    
    Features:
    - Recent signals by symbol
    - Signal history for analysis
    - Active direction tracking
    - Consensus calculation from multiple signals
    
    Example:
        store = LuxAlgoSignalStore()
        store.add_signal(signal)
        
        # Get latest signal
        latest = store.get_latest_signal('ETHUSD')
        
        # Get consensus from last 24 hours
        consensus = store.get_signal_consensus('ETHUSD', hours=24)
    """
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        
        # Signal storage
        self.signals: deque = deque(maxlen=max_history)
        self.signals_by_symbol: Dict[str, List[LuxAlgoSignal]] = {}
        
        # Current state
        self.latest_signal: Dict[str, LuxAlgoSignal] = {}  # Latest signal per symbol
        self.active_direction: Dict[str, str] = {}  # Current direction per symbol
        
        # Tracking
        self.total_signals_received: int = 0
        self.signals_by_type: Dict[str, int] = {}
    
    def add_signal(self, signal: LuxAlgoSignal):
        """Add a new signal to the store."""
        # Store in history
        self.signals.append(signal)
        self.total_signals_received += 1
        
        # Track by type
        type_name = signal.signal_type.value
        self.signals_by_type[type_name] = self.signals_by_type.get(type_name, 0) + 1
        
        # Store by symbol
        if signal.symbol not in self.signals_by_symbol:
            self.signals_by_symbol[signal.symbol] = []
        self.signals_by_symbol[signal.symbol].append(signal)
        
        # Keep only recent signals per symbol (last 100)
        if len(self.signals_by_symbol[signal.symbol]) > 100:
            self.signals_by_symbol[signal.symbol] = self.signals_by_symbol[signal.symbol][-100:]
        
        # Update latest
        self.latest_signal[signal.symbol] = signal
        self.active_direction[signal.symbol] = signal.action
        
        logger.info(
            f"LuxAlgo signal: {signal.symbol} {signal.action} "
            f"({signal.confirmations} confirmations, {signal.timeframe}, "
            f"confidence={signal.confidence:.0%})"
        )
    
    def get_latest_signal(self, symbol: str) -> Optional[LuxAlgoSignal]:
        """Get the most recent signal for a symbol."""
        return self.latest_signal.get(symbol.upper())
    
    def get_current_direction(self, symbol: str) -> Optional[str]:
        """Get current direction (BUY/SELL) for a symbol."""
        return self.active_direction.get(symbol.upper())
    
    def get_recent_signals(
        self, 
        symbol: str, 
        hours: int = 24,
        min_confirmations: int = 1
    ) -> List[LuxAlgoSignal]:
        """
        Get signals from the last N hours.
        
        Args:
            symbol: Asset symbol
            hours: Lookback period in hours
            min_confirmations: Minimum confirmations to include
        """
        cutoff = datetime.now().timestamp() - (hours * 3600)
        
        signals = self.signals_by_symbol.get(symbol.upper(), [])
        return [
            s for s in signals 
            if s.timestamp.timestamp() > cutoff 
            and s.confirmations >= min_confirmations
        ]
    
    def get_strong_signals(
        self,
        symbol: str,
        hours: int = 24
    ) -> List[LuxAlgoSignal]:
        """Get only strong signals (4+ confirmations on 4h+)."""
        recent = self.get_recent_signals(symbol, hours)
        return [s for s in recent if s.is_strong]
    
    def get_signal_consensus(
        self, 
        symbol: str, 
        hours: int = 24,
        weight_by_recency: bool = True
    ) -> Dict:
        """
        Get consensus from recent signals.
        
        Returns:
            Dict with:
            - direction: 'BUY', 'SELL', or None
            - confidence: 0-1
            - buy_signals: count
            - sell_signals: count
            - signal_count: total
            - latest: most recent signal
        """
        recent = self.get_recent_signals(symbol, hours)
        
        if not recent:
            return {
                'direction': None, 
                'confidence': 0, 
                'signal_count': 0,
                'buy_signals': 0,
                'sell_signals': 0,
                'latest': None
            }
        
        # Separate buy and sell signals
        buys = [s for s in recent if s.action == "BUY"]
        sells = [s for s in recent if s.action == "SELL"]
        
        # Calculate weighted scores
        now = datetime.now().timestamp()
        
        def calculate_weight(signal: LuxAlgoSignal) -> float:
            """Weight by confidence and recency."""
            base_weight = signal.confidence
            
            if weight_by_recency:
                # Decay: signal from 24h ago has 50% weight
                age_hours = (now - signal.timestamp.timestamp()) / 3600
                recency_weight = max(0.5, 1.0 - (age_hours / 48))  # Decay over 48h
                return base_weight * recency_weight
            
            return base_weight
        
        buy_weight = sum(calculate_weight(s) for s in buys)
        sell_weight = sum(calculate_weight(s) for s in sells)
        
        total_weight = buy_weight + sell_weight
        
        if total_weight == 0:
            return {
                'direction': None, 
                'confidence': 0, 
                'signal_count': len(recent),
                'buy_signals': len(buys),
                'sell_signals': len(sells),
                'latest': recent[-1].to_dict() if recent else None
            }
        
        if buy_weight > sell_weight:
            direction = "BUY"
            confidence = buy_weight / total_weight
        else:
            direction = "SELL"
            confidence = sell_weight / total_weight
        
        return {
            'direction': direction,
            'confidence': round(confidence, 4),
            'buy_signals': len(buys),
            'sell_signals': len(sells),
            'signal_count': len(recent),
            'buy_weight': round(buy_weight, 4),
            'sell_weight': round(sell_weight, 4),
            'latest': recent[-1].to_dict() if recent else None,
            'strongest': max(recent, key=lambda s: s.confidence).to_dict()
        }
    
    def get_all_symbols(self) -> List[str]:
        """Get all symbols with signals."""
        return list(self.latest_signal.keys())
    
    def get_stats(self) -> Dict:
        """Get store statistics."""
        return {
            'total_signals': self.total_signals_received,
            'symbols_tracked': len(self.latest_signal),
            'signals_by_type': self.signals_by_type,
            'current_directions': self.active_direction
        }
    
    def clear(self):
        """Clear all signals."""
        self.signals.clear()
        self.signals_by_symbol.clear()
        self.latest_signal.clear()
        self.active_direction.clear()


def parse_luxalgo_webhook(payload: Dict) -> LuxAlgoSignal:
    """
    Parse incoming webhook payload from TradingView LuxAlgo alert.
    
    Expected payload format (configure in TradingView alert message):
    
    For strategy alerts:
    {
        "action": "{{strategy.order.action}}",
        "symbol": "{{ticker}}",
        "exchange": "{{exchange}}",
        "price": {{close}},
        "time": "{{time}}",
        "timeframe": "{{interval}}",
        "message": "{{strategy.order.comment}}",
        "confirmations": 4,
        "trend_strength": 75
    }
    
    For indicator alerts (recommended):
    {
        "action": "BUY",
        "symbol": "ETHUSD",
        "exchange": "COINBASE",
        "price": 2340.61,
        "timeframe": "4h",
        "signal_type": "Bullish Confirmation",
        "confirmations": 12,
        "trend_strength": 54.04,
        "trend_tracer": "bullish",
        "smart_trail": "bullish",
        "neo_cloud": "bullish",
        "trend_catcher": "bullish"
    }
    
    Args:
        payload: Dict from webhook POST body
        
    Returns:
        LuxAlgoSignal parsed from payload
    """
    
    # Generate unique signal ID
    signal_id = hashlib.md5(
        f"{payload.get('symbol')}_{payload.get('time', datetime.now().isoformat())}_{payload.get('action')}_{datetime.now().timestamp()}".encode()
    ).hexdigest()[:12]
    
    # Parse action (BUY or SELL)
    action_raw = str(payload.get('action', '')).upper()
    if any(x in action_raw for x in ['BUY', 'LONG', 'BULLISH']):
        action = 'BUY'
    elif any(x in action_raw for x in ['SELL', 'SHORT', 'BEARISH']):
        action = 'SELL'
    else:
        # Try to infer from other fields
        if 'bull' in str(payload).lower():
            action = 'BUY'
        elif 'bear' in str(payload).lower():
            action = 'SELL'
        else:
            action = 'BUY'  # Default to BUY
    
    # Parse signal type
    signal_type_raw = str(payload.get('signal_type', payload.get('message', ''))).lower()
    
    if 'trend tracer' in signal_type_raw:
        signal_type = LuxAlgoSignalType.TREND_TRACER_BULLISH if action == 'BUY' else LuxAlgoSignalType.TREND_TRACER_BEARISH
    elif 'smart trail' in signal_type_raw:
        signal_type = LuxAlgoSignalType.SMART_TRAIL_BULLISH if action == 'BUY' else LuxAlgoSignalType.SMART_TRAIL_BEARISH
    elif 'trend catcher' in signal_type_raw:
        signal_type = LuxAlgoSignalType.TREND_CATCHER_BULLISH if action == 'BUY' else LuxAlgoSignalType.TREND_CATCHER_BEARISH
    elif 'neo cloud' in signal_type_raw:
        signal_type = LuxAlgoSignalType.NEO_CLOUD_BULLISH if action == 'BUY' else LuxAlgoSignalType.NEO_CLOUD_BEARISH
    else:
        signal_type = LuxAlgoSignalType.BULLISH_CONFIRMATION if action == 'BUY' else LuxAlgoSignalType.BEARISH_CONFIRMATION
    
    # Parse timestamp
    time_str = payload.get('time', '')
    try:
        if 'T' in str(time_str):
            timestamp = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
        else:
            timestamp = datetime.now()
    except:
        timestamp = datetime.now()
    
    # Parse confirmations (1-12)
    confirmations = payload.get('confirmations', 1)
    if isinstance(confirmations, str):
        try:
            confirmations = int(confirmations)
        except:
            confirmations = 1
    confirmations = max(1, min(12, confirmations))  # Clamp to 1-12
    
    # Parse trend strength (0-100)
    trend_strength = payload.get('trend_strength', 50)
    if isinstance(trend_strength, str):
        try:
            trend_strength = float(trend_strength.replace('%', ''))
        except:
            trend_strength = 50
    trend_strength = max(0, min(100, trend_strength))
    
    # Parse indicator states
    def parse_bullish(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ['bullish', 'bull', 'true', 'buy', '1']
        return bool(value)
    
    return LuxAlgoSignal(
        signal_id=signal_id,
        timestamp=timestamp,
        symbol=str(payload.get('symbol', 'UNKNOWN')).upper(),
        exchange=str(payload.get('exchange', 'UNKNOWN')).upper(),
        timeframe=str(payload.get('timeframe', payload.get('interval', '4h'))),
        signal_type=signal_type,
        action=action,
        price=float(payload.get('price', 0)),
        confirmations=confirmations,
        trend_strength=trend_strength,
        raw_message=str(payload.get('message', str(payload))),
        metadata=payload,
        trend_tracer_bullish=parse_bullish(payload.get('trend_tracer', '')),
        smart_trail_bullish=parse_bullish(payload.get('smart_trail', '')),
        neo_cloud_bullish=parse_bullish(payload.get('neo_cloud', '')),
        trend_catcher_bullish=parse_bullish(payload.get('trend_catcher', ''))
    )


# Global signal store singleton
_signal_store: Optional[LuxAlgoSignalStore] = None


def get_signal_store() -> LuxAlgoSignalStore:
    """Get or create the global signal store."""
    global _signal_store
    if _signal_store is None:
        _signal_store = LuxAlgoSignalStore()
    return _signal_store


def reset_signal_store():
    """Reset the global signal store (for testing)."""
    global _signal_store
    _signal_store = None
