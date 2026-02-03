"""
Technical Signal Source - LuxAlgo

Uses LuxAlgo signals from TradingView as the primary technical signal source.
This is a PROVEN signal source used by professional traders.
"""

from datetime import datetime
from typing import Dict, Optional, Any
import logging

from ..base_source import SignalSource, NormalizedSignal
from ..luxalgo_webhook import get_signal_store

logger = logging.getLogger(__name__)


class TechnicalLuxAlgoSource(SignalSource):
    """
    LuxAlgo signal source from TradingView webhooks.
    
    LuxAlgo is a premium TradingView indicator suite that provides:
    - Buy/Sell confirmations (1-12)
    - Trend Tracer
    - Smart Trail
    - Neo Cloud
    - Trend Catcher
    
    This source receives signals via webhook and provides them
    to the fusion engine.
    
    Signal Strength:
    - Strong: 4+ confirmations on 4h+ timeframe
    - Medium: 2-3 confirmations or lower timeframe
    - Weak: 1 confirmation
    
    Example:
        source = TechnicalLuxAlgoSource()
        signal = source.generate_signal('ETHUSD')
        if signal:
            print(f"LuxAlgo: {signal.direction:.2f} ({signal.confidence:.0%})")
    """
    
    name = "luxalgo"
    base_weight = 0.50  # Primary source - highest weight
    
    def __init__(
        self,
        weight: float = None,
        lookback_hours: int = 24,
        min_confirmations: int = 1,
        require_strong: bool = False
    ):
        """
        Initialize LuxAlgo source.
        
        Args:
            weight: Custom weight (default uses base_weight)
            lookback_hours: Hours to look back for signals
            min_confirmations: Minimum confirmations to consider
            require_strong: Only use strong signals (4+ conf on 4h+)
        """
        super().__init__(weight)
        self.lookback_hours = lookback_hours
        self.min_confirmations = min_confirmations
        self.require_strong = require_strong
        self.signal_store = get_signal_store()
    
    def generate_signal(
        self,
        symbol: str,
        data: Any = None,
        context: Dict = None
    ) -> Optional[NormalizedSignal]:
        """
        Generate normalized signal from LuxAlgo data.
        
        Args:
            symbol: Asset symbol
            data: Not used (signals come from webhook store)
            context: Additional context
            
        Returns:
            NormalizedSignal or None if no recent signals
        """
        symbol = symbol.upper()
        
        # Get consensus from recent signals
        consensus = self.signal_store.get_signal_consensus(
            symbol, 
            hours=self.lookback_hours
        )
        
        if consensus['signal_count'] == 0:
            return None
        
        # If requiring strong signals, check
        if self.require_strong:
            strong_signals = self.signal_store.get_strong_signals(
                symbol, 
                hours=self.lookback_hours
            )
            if not strong_signals:
                return None
        
        # Calculate direction (-1 to +1)
        if consensus['direction'] == 'BUY':
            direction = consensus['confidence']
        elif consensus['direction'] == 'SELL':
            direction = -consensus['confidence']
        else:
            direction = 0
        
        # Get confidence
        confidence = consensus['confidence']
        
        # Build reasoning
        reasoning = (
            f"LuxAlgo: {consensus['buy_signals']} buy, "
            f"{consensus['sell_signals']} sell signals in {self.lookback_hours}h"
        )
        
        # Add latest signal info
        latest = self.signal_store.get_latest_signal(symbol)
        if latest:
            reasoning += f" (latest: {latest.confirmations} confirmations on {latest.timeframe})"
        
        signal = NormalizedSignal(
            source_name=self.name,
            symbol=symbol,
            timestamp=datetime.now(),
            direction=direction,
            confidence=confidence,
            raw_data=consensus,
            reasoning=reasoning
        )
        
        self._last_signal = signal
        return signal
    
    def get_confidence(self) -> float:
        """
        Get confidence in this source.
        
        Based on:
        - Number of recent signals
        - Signal consistency
        - Recent accuracy
        """
        # Base confidence
        base_confidence = 0.7  # LuxAlgo is proven
        
        # Boost from recent accuracy
        recent_accuracy = self.get_recent_accuracy(20)
        accuracy_boost = (recent_accuracy - 0.5) * 0.3  # ±15% boost
        
        return min(1.0, max(0.3, base_confidence + accuracy_boost))
    
    def has_recent_signal(self, symbol: str) -> bool:
        """Check if there's a recent signal for the symbol."""
        signal = self.signal_store.get_latest_signal(symbol)
        return signal is not None
    
    def get_signal_details(self, symbol: str) -> Optional[Dict]:
        """Get detailed signal information."""
        latest = self.signal_store.get_latest_signal(symbol)
        if not latest:
            return None
        
        return {
            'signal': latest.to_dict(),
            'consensus': self.signal_store.get_signal_consensus(symbol),
            'is_strong': latest.is_strong
        }
