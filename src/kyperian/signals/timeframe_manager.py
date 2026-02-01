"""
KYPERIAN ELITE: Multi-Timeframe Signal Manager

Stores and manages LuxAlgo signals across multiple timeframes with:
- Signal freshness decay
- Timeframe hierarchy
- Automatic expiration
- Historical tracking

The hierarchy:
- Weekly (1W): VETO POWER - Determines IF we trade
- Daily (1D): DIRECTION - Determines WHICH WAY we trade  
- 4-Hour (4H): ENTRY TRIGGER - Determines WHEN we enter
- 1-Hour (1H): FINE TUNING - Optional, for reducing slippage
"""

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import json
import threading

logger = logging.getLogger(__name__)


class Timeframe(Enum):
    """Supported timeframes in order of importance."""
    WEEKLY = "1W"
    DAILY = "1D"
    FOUR_HOUR = "4h"
    ONE_HOUR = "1h"
    
    @property
    def weight(self) -> float:
        """Get the weight of this timeframe in decision making."""
        weights = {
            "1W": 0.40,   # 40% - Most important
            "1D": 0.35,   # 35% - Direction
            "4h": 0.25,   # 25% - Entry timing
            "1h": 0.00,   # 0% - Only for fine-tuning, not decision
        }
        return weights.get(self.value, 0.25)
    
    @property
    def max_age_hours(self) -> float:
        """Maximum age in hours before signal expires."""
        ages = {
            "1W": 168,    # 7 days
            "1D": 24,     # 1 day
            "4h": 8,      # 8 hours
            "1h": 2,      # 2 hours
        }
        return ages.get(self.value, 24)
    
    @property
    def decay_age_hours(self) -> float:
        """Age at which signal starts decaying (50% weight)."""
        return self.max_age_hours * 0.75
    
    @property
    def optimal_sensitivity(self) -> int:
        """Recommended LuxAlgo sensitivity for this timeframe."""
        sensitivities = {
            "1W": 22,     # Long-term, less noise
            "1D": 15,     # Balanced
            "4h": 11,     # More responsive
            "1h": 9,      # Most responsive
        }
        return sensitivities.get(self.value, 12)
    
    @classmethod
    def from_string(cls, value: str) -> 'Timeframe':
        """Parse timeframe from string."""
        value = value.upper().strip()
        
        # Handle various formats
        mappings = {
            "1W": cls.WEEKLY,
            "W": cls.WEEKLY,
            "WEEKLY": cls.WEEKLY,
            "1D": cls.DAILY,
            "D": cls.DAILY,
            "DAILY": cls.DAILY,
            "4H": cls.FOUR_HOUR,
            "4HR": cls.FOUR_HOUR,
            "240": cls.FOUR_HOUR,  # TradingView format
            "1H": cls.ONE_HOUR,
            "H": cls.ONE_HOUR,
            "60": cls.ONE_HOUR,    # TradingView format
            "HOURLY": cls.ONE_HOUR,
        }
        
        if value in mappings:
            return mappings[value]
        
        # Default to 4H for unknown
        logger.warning(f"Unknown timeframe '{value}', defaulting to 4H")
        return cls.FOUR_HOUR
    
    def __lt__(self, other):
        """Compare timeframes by importance (higher = more important)."""
        order = [Timeframe.WEEKLY, Timeframe.DAILY, Timeframe.FOUR_HOUR, Timeframe.ONE_HOUR]
        return order.index(self) > order.index(other)


@dataclass
class TimeframeSignal:
    """
    A signal from a specific timeframe.
    
    Contains all relevant LuxAlgo data plus metadata for fusion.
    """
    # Core identification
    symbol: str
    timeframe: Timeframe
    timestamp: datetime
    
    # Signal data
    direction: int                  # -1 (sell), 0 (neutral), 1 (buy)
    action: str                     # "BUY", "SELL", "NEUTRAL"
    strength: str = "normal"        # "normal", "strong"
    confirmations: int = 1          # 1-12
    
    # LuxAlgo indicators
    trend_strength: float = 50.0    # 0-100
    smart_trail_sentiment: str = "neutral"  # "bullish", "bearish", "neutral"
    smart_trail_level: Optional[float] = None
    neo_cloud_sentiment: str = "neutral"
    reversal_zone_upper: Optional[float] = None
    reversal_zone_lower: Optional[float] = None
    ml_classification: int = 0      # 1-2 reversal, 3-4 continuation
    
    # Price data at signal time
    price: float = 0.0
    
    # Metadata
    signal_id: str = ""
    exchange: str = "UNKNOWN"
    raw_payload: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Generate signal ID if not provided."""
        if not self.signal_id:
            import hashlib
            data = f"{self.symbol}{self.timeframe.value}{self.timestamp.isoformat()}"
            self.signal_id = hashlib.md5(data.encode()).hexdigest()[:12]
    
    @property
    def age_hours(self) -> float:
        """How old is this signal in hours."""
        delta = datetime.now() - self.timestamp
        return delta.total_seconds() / 3600
    
    @property
    def freshness(self) -> float:
        """
        Calculate signal freshness (0.0 = expired, 1.0 = fresh).
        
        Implements decay curve:
        - First 75% of max age: Linear decay from 1.0 to 0.5
        - Last 25% of max age: Accelerated decay from 0.5 to 0.0
        - After max age: 0.0 (expired)
        """
        age = self.age_hours
        max_age = self.timeframe.max_age_hours
        decay_age = self.timeframe.decay_age_hours
        
        if age >= max_age:
            return 0.0  # Expired
        
        if age <= decay_age:
            # Linear decay from 1.0 to 0.5
            return 1.0 - (age / decay_age) * 0.5
        else:
            # Accelerated decay from 0.5 to 0.0
            remaining_ratio = (age - decay_age) / (max_age - decay_age)
            return 0.5 * (1.0 - remaining_ratio)
    
    @property
    def is_expired(self) -> bool:
        """Check if signal has expired."""
        return self.freshness <= 0.0
    
    @property
    def is_fresh(self) -> bool:
        """Check if signal is still reasonably fresh (>30% freshness)."""
        return self.freshness >= 0.3
    
    @property
    def is_strong(self) -> bool:
        """Check if this is a strong signal."""
        return self.strength.lower() == "strong" or self.confirmations >= 8
    
    @property
    def is_bullish(self) -> bool:
        """Check if signal is bullish."""
        return self.direction > 0
    
    @property
    def is_bearish(self) -> bool:
        """Check if signal is bearish."""
        return self.direction < 0
    
    @property
    def weighted_direction(self) -> float:
        """
        Get direction weighted by freshness and strength.
        
        Returns value between -1.0 and +1.0
        """
        base = float(self.direction)
        
        # Apply freshness
        base *= self.freshness
        
        # Apply strength multiplier
        if self.is_strong:
            base *= 1.2
        
        # Apply confirmations bonus (8+ confirmations = up to 20% boost)
        if self.confirmations >= 8:
            conf_bonus = 1.0 + (self.confirmations - 8) * 0.05  # 5% per confirmation above 8
            base *= min(conf_bonus, 1.2)  # Cap at 20% boost
        
        # Apply trend strength bonus
        if self.trend_strength > 70:
            base *= 1.1
        elif self.trend_strength < 30:
            base *= 0.9
        
        return max(-1.0, min(1.0, base))
    
    @property
    def confidence(self) -> float:
        """
        Calculate confidence in this signal (0-100%).
        
        Based on:
        - Freshness
        - Strength
        - Confirmations
        - Indicator agreement
        """
        # Base confidence from freshness
        conf = self.freshness * 0.5  # Up to 50% from freshness
        
        # Strength bonus
        if self.is_strong:
            conf += 0.15
        
        # Confirmations bonus (up to 25%)
        conf += (self.confirmations / 12) * 0.25
        
        # Indicator agreement bonus (up to 10%)
        agreement = 0
        if self.direction > 0:
            if self.smart_trail_sentiment == "bullish":
                agreement += 1
            if self.neo_cloud_sentiment == "bullish":
                agreement += 1
        elif self.direction < 0:
            if self.smart_trail_sentiment == "bearish":
                agreement += 1
            if self.neo_cloud_sentiment == "bearish":
                agreement += 1
        conf += agreement * 0.05
        
        return min(1.0, conf)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "timeframe": self.timeframe.value,
            "timestamp": self.timestamp.isoformat(),
            "direction": self.direction,
            "action": self.action,
            "strength": self.strength,
            "confirmations": self.confirmations,
            "trend_strength": self.trend_strength,
            "smart_trail_sentiment": self.smart_trail_sentiment,
            "smart_trail_level": self.smart_trail_level,
            "neo_cloud_sentiment": self.neo_cloud_sentiment,
            "reversal_zone_upper": self.reversal_zone_upper,
            "reversal_zone_lower": self.reversal_zone_lower,
            "ml_classification": self.ml_classification,
            "price": self.price,
            "exchange": self.exchange,
            "age_hours": round(self.age_hours, 2),
            "freshness": round(self.freshness, 3),
            "is_fresh": self.is_fresh,
            "is_strong": self.is_strong,
            "weighted_direction": round(self.weighted_direction, 3),
            "confidence": round(self.confidence, 3),
        }
    
    @classmethod
    def from_webhook(cls, payload: Dict[str, Any]) -> 'TimeframeSignal':
        """
        Create TimeframeSignal from webhook payload.
        
        Supports both simple and complex payload formats.
        """
        # Parse timeframe
        tf_str = payload.get("timeframe", payload.get("interval", "4h"))
        timeframe = Timeframe.from_string(tf_str)
        
        # Parse direction/action
        action = payload.get("action", "").upper()
        if not action and "signal" in payload:
            action = payload["signal"].get("action", "").upper()
        
        direction = 1 if action == "BUY" else -1 if action == "SELL" else 0
        
        # Parse timestamp
        ts_str = payload.get("timestamp", payload.get("time", ""))
        try:
            if ts_str:
                if isinstance(ts_str, (int, float)):
                    timestamp = datetime.fromtimestamp(ts_str)
                else:
                    # Try various formats
                    for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"]:
                        try:
                            timestamp = datetime.strptime(ts_str, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        timestamp = datetime.now()
            else:
                timestamp = datetime.now()
        except Exception:
            timestamp = datetime.now()
        
        # Parse signal details (handle nested structure)
        signal_data = payload.get("signal", {})
        indicators = payload.get("indicators", {})
        
        # Get strength
        strength = signal_data.get("strength", payload.get("strength", "normal"))
        if payload.get("signal_type", "").lower().startswith("strong"):
            strength = "strong"
        
        # Get confirmations
        confirmations = signal_data.get("confirmations", payload.get("confirmations", 1))
        try:
            confirmations = int(confirmations)
        except (ValueError, TypeError):
            confirmations = 1
        
        # Get indicator values
        trend_strength = indicators.get("trend_strength", payload.get("trend_strength", 50))
        try:
            trend_strength = float(trend_strength)
        except (ValueError, TypeError):
            trend_strength = 50.0
        
        smart_trail = indicators.get("smart_trail_sentiment", payload.get("smart_trail", "neutral"))
        if isinstance(smart_trail, str):
            smart_trail = smart_trail.lower()
        else:
            smart_trail = "neutral"
        
        neo_cloud = indicators.get("neo_cloud_sentiment", payload.get("neo_cloud", "neutral"))
        if isinstance(neo_cloud, str):
            neo_cloud = neo_cloud.lower()
        else:
            neo_cloud = "neutral"
        
        # Parse reversal zones
        rz_upper = indicators.get("reversal_zone_upper", payload.get("reversal_zone_upper"))
        rz_lower = indicators.get("reversal_zone_lower", payload.get("reversal_zone_lower"))
        
        # Parse ML classification
        ml_class = indicators.get("ml_classification", payload.get("ml_classification", 0))
        try:
            ml_class = int(ml_class)
        except (ValueError, TypeError):
            ml_class = 0
        
        return cls(
            symbol=payload.get("symbol", payload.get("ticker", "UNKNOWN")).upper(),
            timeframe=timeframe,
            timestamp=timestamp,
            direction=direction,
            action=action,
            strength=strength,
            confirmations=confirmations,
            trend_strength=trend_strength,
            smart_trail_sentiment=smart_trail,
            smart_trail_level=indicators.get("smart_trail_level"),
            neo_cloud_sentiment=neo_cloud,
            reversal_zone_upper=rz_upper,
            reversal_zone_lower=rz_lower,
            ml_classification=ml_class,
            price=float(payload.get("price", payload.get("close", 0))),
            exchange=payload.get("exchange", "UNKNOWN"),
            raw_payload=payload,
        )
    
    def __str__(self) -> str:
        """Human-readable string representation."""
        freshness_str = f"{self.freshness:.0%}" if self.freshness > 0 else "EXPIRED"
        strength_str = "💪" if self.is_strong else ""
        return (
            f"{self.timeframe.value} {self.symbol}: {self.action} {strength_str} "
            f"(conf={self.confirmations}, fresh={freshness_str})"
        )


class TimeframeManager:
    """
    Manages signals across multiple timeframes for each symbol.
    
    Features:
    - Stores latest signal for each timeframe
    - Tracks signal history
    - Calculates multi-timeframe alignment
    - Handles signal expiration
    """
    
    def __init__(self, history_size: int = 100):
        """
        Initialize TimeframeManager.
        
        Args:
            history_size: Number of historical signals to keep per symbol/timeframe
        """
        self._lock = threading.Lock()
        self.history_size = history_size
        
        # Current signals: {symbol: {timeframe: TimeframeSignal}}
        self._current: Dict[str, Dict[Timeframe, TimeframeSignal]] = {}
        
        # Historical signals: {symbol: {timeframe: [TimeframeSignal]}}
        self._history: Dict[str, Dict[Timeframe, List[TimeframeSignal]]] = {}
        
        logger.info(f"TimeframeManager initialized (history_size={history_size})")
    
    def add_signal(self, signal: TimeframeSignal) -> None:
        """
        Add a new signal to the manager.
        
        Args:
            signal: The TimeframeSignal to add
        """
        with self._lock:
            symbol = signal.symbol.upper()
            tf = signal.timeframe
            
            # Initialize structures if needed
            if symbol not in self._current:
                self._current[symbol] = {}
                self._history[symbol] = {}
            
            if tf not in self._history[symbol]:
                self._history[symbol][tf] = []
            
            # Move old current to history
            if tf in self._current[symbol]:
                old_signal = self._current[symbol][tf]
                self._history[symbol][tf].append(old_signal)
                
                # Trim history
                if len(self._history[symbol][tf]) > self.history_size:
                    self._history[symbol][tf] = self._history[symbol][tf][-self.history_size:]
            
            # Store new signal
            self._current[symbol][tf] = signal
            
            logger.info(f"Added {tf.value} signal for {symbol}: {signal.action} (fresh={signal.freshness:.0%})")
    
    def get_signal(self, symbol: str, timeframe: Timeframe) -> Optional[TimeframeSignal]:
        """
        Get the current signal for a symbol/timeframe.
        
        Args:
            symbol: Asset symbol
            timeframe: Timeframe to get
            
        Returns:
            TimeframeSignal or None if not found/expired
        """
        symbol = symbol.upper()
        
        with self._lock:
            if symbol not in self._current:
                return None
            if timeframe not in self._current[symbol]:
                return None
            
            signal = self._current[symbol][timeframe]
            
            # Check if expired
            if signal.is_expired:
                logger.debug(f"{timeframe.value} signal for {symbol} has expired")
                return None
            
            return signal
    
    def get_all_signals(self, symbol: str) -> Dict[Timeframe, TimeframeSignal]:
        """
        Get all current (non-expired) signals for a symbol.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Dictionary of timeframe -> signal
        """
        symbol = symbol.upper()
        result = {}
        
        with self._lock:
            if symbol not in self._current:
                return result
            
            for tf, signal in self._current[symbol].items():
                if not signal.is_expired:
                    result[tf] = signal
        
        return result
    
    def get_cascade(self, symbol: str) -> Tuple[
        Optional[TimeframeSignal],  # Weekly
        Optional[TimeframeSignal],  # Daily
        Optional[TimeframeSignal],  # 4H
        Optional[TimeframeSignal],  # 1H
    ]:
        """
        Get the full timeframe cascade for a symbol.
        
        Args:
            symbol: Asset symbol
            
        Returns:
            Tuple of (weekly, daily, 4h, 1h) signals
        """
        return (
            self.get_signal(symbol, Timeframe.WEEKLY),
            self.get_signal(symbol, Timeframe.DAILY),
            self.get_signal(symbol, Timeframe.FOUR_HOUR),
            self.get_signal(symbol, Timeframe.ONE_HOUR),
        )
    
    def get_alignment(self, symbol: str) -> Dict[str, Any]:
        """
        Calculate alignment across timeframes for a symbol.
        
        Returns:
            Dict with alignment score, direction, and details
        """
        weekly, daily, four_hour, hourly = self.get_cascade(symbol)
        
        signals = [s for s in [weekly, daily, four_hour] if s is not None]
        
        if not signals:
            return {
                "aligned": False,
                "alignment_score": 0.0,
                "direction": 0,
                "reason": "No signals available",
                "signals": {}
            }
        
        # Count directions
        bullish = sum(1 for s in signals if s.is_bullish and s.is_fresh)
        bearish = sum(1 for s in signals if s.is_bearish and s.is_fresh)
        neutral = sum(1 for s in signals if s.direction == 0 or not s.is_fresh)
        
        # Calculate weighted direction
        weighted_sum = 0.0
        weight_total = 0.0
        
        for signal in signals:
            if signal.is_fresh:
                weight = signal.timeframe.weight
                weighted_sum += signal.weighted_direction * weight
                weight_total += weight
        
        if weight_total > 0:
            weighted_direction = weighted_sum / weight_total
        else:
            weighted_direction = 0.0
        
        # Determine overall direction
        if weighted_direction > 0.2:
            direction = 1
        elif weighted_direction < -0.2:
            direction = -1
        else:
            direction = 0
        
        # Calculate alignment score (0-100%)
        # Full alignment = all signals same direction and fresh
        if bullish == len(signals) or bearish == len(signals):
            alignment_score = 1.0
            aligned = True
            reason = "✅ Perfect alignment across all timeframes"
        elif bullish > 0 and bearish > 0:
            # Conflicting signals
            alignment_score = 0.3
            aligned = False
            reason = f"⚠️ Conflicting signals: {bullish} bullish, {bearish} bearish"
        elif neutral == len(signals):
            alignment_score = 0.2
            aligned = False
            reason = "⏸️ All signals are neutral"
        else:
            # Partial alignment
            max_direction = max(bullish, bearish)
            alignment_score = max_direction / len(signals)
            aligned = alignment_score >= 0.66
            reason = f"Partial alignment: {bullish} bullish, {bearish} bearish, {neutral} neutral"
        
        # Check weekly veto
        if weekly and weekly.is_fresh:
            if direction != 0 and weekly.direction != 0 and direction != weekly.direction:
                alignment_score *= 0.25  # Heavy penalty for going against weekly
                reason = "🚫 Direction conflicts with weekly trend (VETO)"
                aligned = False
        
        return {
            "aligned": aligned,
            "alignment_score": alignment_score,
            "direction": direction,
            "weighted_direction": weighted_direction,
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": neutral,
            "reason": reason,
            "signals": {
                "weekly": weekly.to_dict() if weekly else None,
                "daily": daily.to_dict() if daily else None,
                "4h": four_hour.to_dict() if four_hour else None,
                "1h": hourly.to_dict() if hourly else None,
            }
        }
    
    def get_history(
        self, 
        symbol: str, 
        timeframe: Timeframe,
        limit: int = 10
    ) -> List[TimeframeSignal]:
        """
        Get historical signals for a symbol/timeframe.
        
        Args:
            symbol: Asset symbol
            timeframe: Timeframe to get history for
            limit: Maximum number of signals to return
            
        Returns:
            List of historical signals (most recent first)
        """
        symbol = symbol.upper()
        
        with self._lock:
            if symbol not in self._history:
                return []
            if timeframe not in self._history[symbol]:
                return []
            
            history = self._history[symbol][timeframe][-limit:]
            return list(reversed(history))  # Most recent first
    
    def get_tracked_symbols(self) -> List[str]:
        """Get list of all symbols being tracked."""
        with self._lock:
            return list(self._current.keys())
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get status of all tracked symbols.
        
        Returns:
            Dict with symbol -> status information
        """
        status = {}
        
        with self._lock:
            for symbol in self._current.keys():
                signals = self.get_all_signals(symbol)
                alignment = self.get_alignment(symbol)
                
                status[symbol] = {
                    "timeframes_available": [tf.value for tf in signals.keys()],
                    "alignment_score": alignment["alignment_score"],
                    "direction": alignment["direction"],
                    "aligned": alignment["aligned"],
                    "reason": alignment["reason"],
                }
        
        return status
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired signals.
        
        Returns:
            Number of signals removed
        """
        removed = 0
        
        with self._lock:
            for symbol in list(self._current.keys()):
                for tf in list(self._current[symbol].keys()):
                    if self._current[symbol][tf].is_expired:
                        # Move to history before removing
                        self._history[symbol][tf].append(self._current[symbol][tf])
                        del self._current[symbol][tf]
                        removed += 1
                        logger.debug(f"Removed expired {tf.value} signal for {symbol}")
                
                # Remove empty symbol entries
                if not self._current[symbol]:
                    del self._current[symbol]
        
        if removed > 0:
            logger.info(f"Cleaned up {removed} expired signals")
        
        return removed
    
    def to_json(self) -> str:
        """Serialize all current signals to JSON."""
        data = {}
        
        with self._lock:
            for symbol, signals in self._current.items():
                data[symbol] = {
                    tf.value: signal.to_dict() 
                    for tf, signal in signals.items()
                }
        
        return json.dumps(data, indent=2, default=str)


# Global instance
_timeframe_manager: Optional[TimeframeManager] = None


def get_timeframe_manager() -> TimeframeManager:
    """Get the global TimeframeManager instance."""
    global _timeframe_manager
    if _timeframe_manager is None:
        _timeframe_manager = TimeframeManager()
    return _timeframe_manager


def parse_mtf_webhook(payload: Dict[str, Any]) -> TimeframeSignal:
    """
    Parse a multi-timeframe webhook payload.
    
    Args:
        payload: Webhook JSON payload
        
    Returns:
        TimeframeSignal instance
    """
    return TimeframeSignal.from_webhook(payload)
