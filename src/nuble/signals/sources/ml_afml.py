"""
ML Signal Source - AFML Pipeline

Uses NUBLE's AFML (Advances in Financial Machine Learning) pipeline
as a signal source.

Components:
- EnhancedSignalGenerator: Multi-timeframe, regime-adaptive signals
- Triple Barrier Labeling: For training labels
- HMM Regime Detection: For regime context
- Meta-Labeling: For position sizing
"""

from datetime import datetime
from typing import Dict, Optional, Any
import logging
import pandas as pd

from ..base_source import SignalSource, NormalizedSignal

logger = logging.getLogger(__name__)


class MLAFMLSource(SignalSource):
    """
    ML signal source using NUBLE AFML pipeline.
    
    This source uses your trained ML models and the EnhancedSignalGenerator
    to produce trading signals.
    
    Features:
    - Multi-timeframe momentum analysis
    - Regime-adaptive parameters
    - Mean reversion signals
    - Sentiment integration
    
    Example:
        source = MLAFMLSource()
        signal = source.generate_signal('AAPL', prices_df, {'regime': 'BULL'})
    """
    
    name = "ml"
    base_weight = 0.25  # Secondary source
    
    def __init__(
        self,
        weight: float = None,
        short_period: int = 5,
        medium_period: int = 20,
        long_period: int = 60
    ):
        """
        Initialize ML AFML source.
        
        Args:
            weight: Custom weight
            short_period: Short-term lookback
            medium_period: Medium-term lookback
            long_period: Long-term lookback
        """
        super().__init__(weight)
        self.short_period = short_period
        self.medium_period = medium_period
        self.long_period = long_period
        self._generator = None
    
    def _get_generator(self):
        """Lazy load the signal generator."""
        if self._generator is None:
            try:
                from institutional.signals.enhanced_signals import EnhancedSignalGenerator
                self._generator = EnhancedSignalGenerator(
                    short_period=self.short_period,
                    medium_period=self.medium_period,
                    long_period=self.long_period
                )
            except ImportError as e:
                logger.warning(f"Could not import EnhancedSignalGenerator: {e}")
                self._generator = None
        return self._generator
    
    def generate_signal(
        self,
        symbol: str,
        data: Any = None,
        context: Dict = None
    ) -> Optional[NormalizedSignal]:
        """
        Generate normalized signal from ML pipeline.
        
        Args:
            symbol: Asset symbol
            data: OHLCV DataFrame (required)
            context: Additional context (regime, sentiment, etc.)
            
        Returns:
            NormalizedSignal or None if cannot generate
        """
        if data is None:
            logger.debug("ML source requires price data")
            return None
        
        if not isinstance(data, pd.DataFrame):
            logger.warning("ML source requires DataFrame")
            return None
        
        if len(data) < self.long_period + 10:
            logger.debug(f"Insufficient data for ML: {len(data)} rows")
            return None
        
        generator = self._get_generator()
        if generator is None:
            return None
        
        try:
            # Get context
            context = context or {}
            regime = context.get('regime', 'SIDEWAYS')
            sentiment = context.get('sentiment', 0.0)
            
            # Generate signal
            ml_signal = generator.generate_signal(
                symbol=symbol,
                prices=data,
                sentiment=sentiment,
                regime=regime
            )
            
            # Normalize to our format
            # Direction is already -1 to +1 from the generator
            direction = ml_signal.direction * ml_signal.confidence
            
            signal = NormalizedSignal(
                source_name=self.name,
                symbol=symbol,
                timestamp=datetime.now(),
                direction=direction,
                confidence=ml_signal.confidence,
                raw_data=ml_signal.to_dict(),
                reasoning=(
                    f"ML: {ml_signal.strength.label} "
                    f"(mom={ml_signal.short_term_signal:.2f}, "
                    f"regime={regime})"
                )
            )
            
            self._last_signal = signal
            return signal
            
        except Exception as e:
            logger.warning(f"ML signal generation failed: {e}")
            return None
    
    def get_confidence(self) -> float:
        """
        Get confidence in this source.
        
        ML confidence is based on:
        - Whether generator is available
        - Recent accuracy
        """
        if self._get_generator() is None:
            return 0.3  # Low confidence if not available
        
        # Base confidence
        base_confidence = 0.5
        
        # Boost from recent accuracy
        recent_accuracy = self.get_recent_accuracy(20)
        accuracy_boost = (recent_accuracy - 0.5) * 0.4
        
        return min(1.0, max(0.2, base_confidence + accuracy_boost))
    
    def generate_features(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Generate ML features from price data.
        
        Useful for external analysis.
        """
        generator = self._get_generator()
        if generator is None:
            return None
        
        try:
            # Use the primary signal model if available
            from institutional.models.primary.ml_primary_signal import MLPrimarySignal
            
            primary = MLPrimarySignal()
            features = primary.generate_features(data)
            return features
        except ImportError:
            return None
        except Exception as e:
            logger.warning(f"Feature generation failed: {e}")
            return None
