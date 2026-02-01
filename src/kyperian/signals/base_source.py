"""
Base Signal Source

Abstract base class for all signal sources in the fusion system.
Each source generates normalized signals that can be combined intelligently.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SignalDirection(Enum):
    """Signal direction."""
    STRONG_BUY = 2
    BUY = 1
    NEUTRAL = 0
    SELL = -1
    STRONG_SELL = -2


@dataclass
class NormalizedSignal:
    """
    A normalized signal from any source.
    
    All signals are normalized to [-1, +1] scale for fusion.
    """
    source_name: str
    symbol: str
    timestamp: datetime
    
    # Core signal data
    direction: float          # -1 to +1 (continuous)
    confidence: float         # 0 to 1
    
    # Metadata
    raw_data: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""
    
    @property
    def is_bullish(self) -> bool:
        return self.direction > 0.1
    
    @property
    def is_bearish(self) -> bool:
        return self.direction < -0.1
    
    @property
    def is_neutral(self) -> bool:
        return -0.1 <= self.direction <= 0.1
    
    @property
    def strength(self) -> float:
        """Signal strength (0 to 1)."""
        return abs(self.direction)
    
    @property
    def weighted_signal(self) -> float:
        """Direction weighted by confidence."""
        return self.direction * self.confidence
    
    def to_dict(self) -> Dict:
        return {
            'source': self.source_name,
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'direction': round(self.direction, 4),
            'confidence': round(self.confidence, 4),
            'weighted_signal': round(self.weighted_signal, 4),
            'is_bullish': self.is_bullish,
            'is_bearish': self.is_bearish,
            'reasoning': self.reasoning
        }


class SignalSource(ABC):
    """
    Abstract base class for signal sources.
    
    Each signal source must implement:
    - generate_signal(): Generate normalized signal for a symbol
    - get_confidence(): Return current confidence in this source
    
    Optionally override:
    - update_accuracy(): Track prediction accuracy
    - get_weight(): Get current weight for fusion
    """
    
    # Class attributes to override
    name: str = "base"
    base_weight: float = 0.15
    
    def __init__(self, weight: float = None):
        self._weight = weight if weight is not None else self.base_weight
        self._accuracy_history: List[float] = []
        self._last_signal: Optional[NormalizedSignal] = None
        self._enabled = True
    
    @abstractmethod
    def generate_signal(
        self,
        symbol: str,
        data: Any = None,
        context: Dict = None
    ) -> Optional[NormalizedSignal]:
        """
        Generate a normalized signal for a symbol.
        
        Args:
            symbol: Asset symbol (e.g., 'AAPL', 'BTCUSD')
            data: Price data or other relevant data
            context: Additional context (regime, etc.)
            
        Returns:
            NormalizedSignal or None if no signal
        """
        pass
    
    @abstractmethod
    def get_confidence(self) -> float:
        """
        Get current confidence in this source (0 to 1).
        
        Based on:
        - Recent accuracy
        - Data freshness
        - Source reliability
        """
        pass
    
    def get_weight(self) -> float:
        """
        Get current weight for fusion.
        
        Can be adjusted based on:
        - Recent accuracy
        - Regime appropriateness
        - User preferences
        """
        return self._weight
    
    def set_weight(self, weight: float):
        """Set the weight for this source."""
        self._weight = max(0, min(1, weight))
    
    def update_accuracy(self, was_correct: bool):
        """
        Track prediction accuracy.
        
        Args:
            was_correct: Whether the last prediction was correct
        """
        self._accuracy_history.append(1.0 if was_correct else 0.0)
        
        # Keep only last 100 predictions
        if len(self._accuracy_history) > 100:
            self._accuracy_history = self._accuracy_history[-100:]
    
    def get_recent_accuracy(self, lookback: int = 20) -> float:
        """Get accuracy over last N predictions."""
        if not self._accuracy_history:
            return 0.5  # No data = neutral
        
        recent = self._accuracy_history[-lookback:]
        return sum(recent) / len(recent)
    
    def enable(self):
        """Enable this signal source."""
        self._enabled = True
    
    def disable(self):
        """Disable this signal source."""
        self._enabled = False
    
    @property
    def is_enabled(self) -> bool:
        return self._enabled
    
    @property
    def last_signal(self) -> Optional[NormalizedSignal]:
        return self._last_signal
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', weight={self._weight:.2f})"


class CompositeSignalSource(SignalSource):
    """
    A signal source composed of multiple sub-sources.
    
    Useful for combining related signals (e.g., multiple technical indicators).
    """
    
    name = "composite"
    
    def __init__(self, sources: List[SignalSource], weight: float = None):
        super().__init__(weight)
        self.sources = sources
    
    def generate_signal(
        self,
        symbol: str,
        data: Any = None,
        context: Dict = None
    ) -> Optional[NormalizedSignal]:
        """Generate combined signal from all sub-sources."""
        signals = []
        
        for source in self.sources:
            if not source.is_enabled:
                continue
            try:
                signal = source.generate_signal(symbol, data, context)
                if signal:
                    signals.append((source.get_weight(), signal))
            except Exception as e:
                logger.warning(f"Sub-source {source.name} failed: {e}")
        
        if not signals:
            return None
        
        # Weighted average
        total_weight = sum(w for w, _ in signals)
        if total_weight == 0:
            return None
        
        weighted_direction = sum(w * s.direction for w, s in signals) / total_weight
        weighted_confidence = sum(w * s.confidence for w, s in signals) / total_weight
        
        return NormalizedSignal(
            source_name=self.name,
            symbol=symbol,
            timestamp=datetime.now(),
            direction=weighted_direction,
            confidence=weighted_confidence,
            raw_data={'sub_signals': [s.to_dict() for _, s in signals]},
            reasoning=f"Combined from {len(signals)} sub-sources"
        )
    
    def get_confidence(self) -> float:
        """Average confidence of enabled sub-sources."""
        enabled = [s for s in self.sources if s.is_enabled]
        if not enabled:
            return 0
        return sum(s.get_confidence() for s in enabled) / len(enabled)
