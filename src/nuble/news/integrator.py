"""
News Signal Integrator

Combines ML trading signals with real-time news sentiment
for enhanced prediction accuracy.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

from .pipeline import NewsPipeline, NewsSignal

logger = logging.getLogger(__name__)


@dataclass
class CombinedSignal:
    """Combined ML + News signal."""
    timestamp: datetime
    symbol: str
    
    # ML Component
    ml_signal: float  # -1 to +1
    ml_confidence: float
    ml_direction: str
    
    # News Component
    news_signal: float  # -1 to +1
    news_confidence: float
    news_direction: str
    news_article_count: int
    
    # Combined
    combined_signal: float  # -1 to +1
    combined_confidence: float
    final_direction: str  # LONG, SHORT, NEUTRAL
    
    # Meta
    signal_agreement: bool  # ML and News agree
    actionable: bool
    reason: str
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'ml_signal': self.ml_signal,
            'ml_confidence': self.ml_confidence,
            'ml_direction': self.ml_direction,
            'news_signal': self.news_signal,
            'news_confidence': self.news_confidence,
            'news_direction': self.news_direction,
            'news_article_count': self.news_article_count,
            'combined_signal': self.combined_signal,
            'combined_confidence': self.combined_confidence,
            'final_direction': self.final_direction,
            'signal_agreement': self.signal_agreement,
            'actionable': self.actionable,
            'reason': self.reason
        }


class NewsSignalIntegrator:
    """
    Integrates ML predictions with news sentiment.
    
    Strategy:
    - ML signal is primary (has been validated with OOS testing)
    - News signal can BOOST or REDUCE confidence
    - News can VETO ML signal in extreme cases
    
    Modes:
    1. CONFIRMATION: News confirms ML signal → Higher confidence
    2. CONTRADICTION: News contradicts ML → Lower confidence
    3. NEWS_OVERRIDE: Extreme news overrides ML (e.g., earnings miss)
    
    Usage:
        integrator = NewsSignalIntegrator(ml_model, news_pipeline)
        
        # Get combined signal
        signal = await integrator.get_signal('AAPL', features)
        
        if signal.actionable:
            execute_trade(signal.final_direction, signal.combined_confidence)
    """
    
    def __init__(
        self,
        ml_model=None,
        news_pipeline: Optional[NewsPipeline] = None,
        ml_weight: float = 0.7,
        news_weight: float = 0.3,
        news_override_threshold: float = 0.8,
        confirmation_boost: float = 0.15,
        contradiction_penalty: float = 0.25
    ):
        """
        Initialize integrator.
        
        Args:
            ml_model: Trained ML model with predict_proba method
            news_pipeline: NewsPipeline instance
            ml_weight: Weight for ML signal (default 0.7)
            news_weight: Weight for news signal (default 0.3)
            news_override_threshold: News confidence to override ML
            confirmation_boost: Confidence boost when signals agree
            contradiction_penalty: Confidence penalty when signals disagree
        """
        self.ml_model = ml_model
        self.news_pipeline = news_pipeline
        
        # Weights
        self.ml_weight = ml_weight
        self.news_weight = news_weight
        
        # Thresholds
        self.news_override_threshold = news_override_threshold
        self.confirmation_boost = confirmation_boost
        self.contradiction_penalty = contradiction_penalty
        
        # Signal thresholds
        self.long_threshold = 0.2
        self.short_threshold = -0.2
        self.min_confidence = 0.55
        
        logger.info("NewsSignalIntegrator initialized")
    
    async def get_signal(
        self,
        symbol: str,
        features: Optional[pd.DataFrame] = None,
        ml_prediction: Optional[Tuple[float, float]] = None
    ) -> CombinedSignal:
        """
        Get combined ML + News signal.
        
        Args:
            symbol: Ticker symbol
            features: Feature DataFrame for ML prediction
            ml_prediction: Pre-computed (signal, confidence) if available
            
        Returns:
            CombinedSignal with combined analysis
        """
        timestamp = datetime.utcnow()
        
        # Get ML signal
        if ml_prediction is not None:
            ml_signal, ml_confidence = ml_prediction
        elif self.ml_model is not None and features is not None:
            ml_signal, ml_confidence = self._get_ml_signal(features)
        else:
            ml_signal, ml_confidence = 0.0, 0.0
        
        ml_direction = self._signal_to_direction(ml_signal)
        
        # Get news signal
        if self.news_pipeline is not None:
            news_result = await self.news_pipeline.fetch_and_analyze(symbol, lookback_hours=24)
            news_signal = news_result.sentiment_score
            news_confidence = news_result.confidence
            news_article_count = news_result.article_count
        else:
            news_signal, news_confidence, news_article_count = 0.0, 0.0, 0
        
        news_direction = self._signal_to_direction(news_signal)
        
        # Combine signals
        combined_signal, combined_confidence, reason = self._combine_signals(
            ml_signal, ml_confidence, ml_direction,
            news_signal, news_confidence, news_direction,
            news_article_count
        )
        
        final_direction = self._signal_to_direction(combined_signal)
        signal_agreement = (ml_direction == news_direction) or news_direction == 'NEUTRAL'
        
        # Determine if actionable
        actionable = (
            abs(combined_signal) > self.long_threshold and
            combined_confidence > self.min_confidence
        )
        
        return CombinedSignal(
            timestamp=timestamp,
            symbol=symbol,
            ml_signal=ml_signal,
            ml_confidence=ml_confidence,
            ml_direction=ml_direction,
            news_signal=news_signal,
            news_confidence=news_confidence,
            news_direction=news_direction,
            news_article_count=news_article_count,
            combined_signal=combined_signal,
            combined_confidence=combined_confidence,
            final_direction=final_direction,
            signal_agreement=signal_agreement,
            actionable=actionable,
            reason=reason
        )
    
    def _get_ml_signal(self, features: pd.DataFrame) -> Tuple[float, float]:
        """Get ML signal and confidence."""
        if self.ml_model is None:
            return 0.0, 0.0
        
        try:
            proba = self.ml_model.predict_proba(features.values)[-1]
            
            # Assuming binary classification: [bearish, bullish]
            if len(proba) == 2:
                signal = proba[1] - proba[0]  # bullish - bearish
                confidence = max(proba)
            else:
                # Multi-class
                signal = proba[2] - proba[0]  # up - down
                confidence = max(proba)
            
            return signal, confidence
            
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return 0.0, 0.0
    
    def _signal_to_direction(self, signal: float) -> str:
        """Convert signal to direction string."""
        if signal > self.long_threshold:
            return 'LONG'
        elif signal < self.short_threshold:
            return 'SHORT'
        else:
            return 'NEUTRAL'
    
    def _combine_signals(
        self,
        ml_signal: float,
        ml_confidence: float,
        ml_direction: str,
        news_signal: float,
        news_confidence: float,
        news_direction: str,
        news_article_count: int
    ) -> Tuple[float, float, str]:
        """
        Combine ML and news signals with sophisticated logic.
        
        Returns:
            (combined_signal, combined_confidence, reason)
        """
        # Case 1: No news data
        if news_article_count == 0 or news_confidence < 0.3:
            return (
                ml_signal,
                ml_confidence * 0.9,  # Slight penalty for no news confirmation
                "No significant news. Using ML signal only."
            )
        
        # Case 2: News override (extreme news event)
        if news_confidence > self.news_override_threshold and abs(news_signal) > 0.6:
            if news_direction != ml_direction and ml_direction != 'NEUTRAL':
                # News strongly contradicts ML - reduce position or skip
                return (
                    news_signal * 0.5,  # Muted signal
                    news_confidence * 0.7,
                    f"⚠️ NEWS OVERRIDE: Strong {news_direction} news contradicts ML. Caution advised."
                )
        
        # Case 3: Confirmation (signals agree)
        if ml_direction == news_direction and ml_direction != 'NEUTRAL':
            boosted_confidence = min(1.0, (ml_confidence + news_confidence) / 2 + self.confirmation_boost)
            combined = self.ml_weight * ml_signal + self.news_weight * news_signal
            return (
                combined,
                boosted_confidence,
                f"✅ CONFIRMATION: ML and news both {ml_direction}. High confidence."
            )
        
        # Case 4: Contradiction (signals disagree)
        if ml_direction != news_direction and ml_direction != 'NEUTRAL' and news_direction != 'NEUTRAL':
            penalized_confidence = max(0.3, ml_confidence - self.contradiction_penalty)
            combined = self.ml_weight * ml_signal + self.news_weight * news_signal
            return (
                combined,
                penalized_confidence,
                f"⚠️ CONTRADICTION: ML says {ml_direction}, news says {news_direction}. Reduced confidence."
            )
        
        # Case 5: One signal neutral
        if ml_direction == 'NEUTRAL':
            # Let news provide direction
            return (
                news_signal * 0.5,
                news_confidence * 0.7,
                f"ML neutral, using news signal: {news_direction}"
            )
        
        if news_direction == 'NEUTRAL':
            # ML provides direction, news doesn't object
            return (
                ml_signal,
                ml_confidence,
                f"News neutral, using ML signal: {ml_direction}"
            )
        
        # Default: Weighted average
        combined = self.ml_weight * ml_signal + self.news_weight * news_signal
        combined_conf = self.ml_weight * ml_confidence + self.news_weight * news_confidence
        
        return (
            combined,
            combined_conf,
            "Weighted combination of ML and news signals."
        )
    
    async def get_multi_symbol_signals(
        self,
        symbols: List[str],
        features_dict: Optional[Dict[str, pd.DataFrame]] = None,
        ml_predictions: Optional[Dict[str, Tuple[float, float]]] = None
    ) -> Dict[str, CombinedSignal]:
        """
        Get signals for multiple symbols efficiently.
        
        Args:
            symbols: List of ticker symbols
            features_dict: Dict of {symbol: features_df}
            ml_predictions: Dict of {symbol: (signal, confidence)}
            
        Returns:
            Dict of {symbol: CombinedSignal}
        """
        import asyncio
        
        async def get_one(symbol):
            features = features_dict.get(symbol) if features_dict else None
            ml_pred = ml_predictions.get(symbol) if ml_predictions else None
            return symbol, await self.get_signal(symbol, features, ml_pred)
        
        results = await asyncio.gather(*[get_one(s) for s in symbols])
        return {symbol: signal for symbol, signal in results}
    
    def get_portfolio_allocation(
        self,
        signals: Dict[str, CombinedSignal],
        max_position_pct: float = 0.25,
        min_confidence: float = 0.6
    ) -> Dict[str, float]:
        """
        Get portfolio allocation based on signals.
        
        Args:
            signals: Dict of {symbol: CombinedSignal}
            max_position_pct: Maximum position size per symbol
            min_confidence: Minimum confidence to take position
            
        Returns:
            Dict of {symbol: allocation_pct} (positive = long, negative = short)
        """
        allocations = {}
        
        actionable_signals = {
            s: sig for s, sig in signals.items()
            if sig.actionable and sig.combined_confidence >= min_confidence
        }
        
        if not actionable_signals:
            return {}
        
        # Calculate raw weights based on signal strength
        total_weight = sum(abs(s.combined_signal) for s in actionable_signals.values())
        
        for symbol, signal in actionable_signals.items():
            # Base allocation proportional to signal strength
            weight = abs(signal.combined_signal) / total_weight
            
            # Scale by confidence
            scaled = weight * signal.combined_confidence
            
            # Cap at max position
            capped = min(scaled, max_position_pct)
            
            # Sign by direction
            if signal.final_direction == 'LONG':
                allocations[symbol] = capped
            elif signal.final_direction == 'SHORT':
                allocations[symbol] = -capped
        
        return allocations


class BacktestNewsIntegrator:
    """
    Backtesting version of NewsSignalIntegrator.
    Uses historical sentiment data instead of real-time.
    """
    
    def __init__(self, sentiment_history: pd.DataFrame):
        """
        Initialize with historical sentiment data.
        
        Args:
            sentiment_history: DataFrame with columns [date, symbol, sentiment_score, article_count]
        """
        self.sentiment_history = sentiment_history.set_index(['date', 'symbol'])
    
    def get_historical_sentiment(
        self,
        symbol: str,
        date: str
    ) -> Tuple[float, float, int]:
        """
        Get sentiment for a specific date.
        
        Returns:
            (sentiment_score, confidence, article_count)
        """
        try:
            row = self.sentiment_history.loc[(date, symbol)]
            return (
                row['sentiment_score'],
                row.get('confidence', 0.5),
                row.get('article_count', 1)
            )
        except KeyError:
            return 0.0, 0.0, 0
    
    def enhance_backtest_signals(
        self,
        backtest_results: pd.DataFrame,
        symbol: str
    ) -> pd.DataFrame:
        """
        Add news sentiment to backtest results.
        
        Args:
            backtest_results: DataFrame with 'date' and 'ml_signal' columns
            symbol: Ticker symbol
            
        Returns:
            Enhanced DataFrame with news integration
        """
        enhanced = backtest_results.copy()
        
        sentiments = []
        confidences = []
        article_counts = []
        
        for date in enhanced.index:
            date_str = date.strftime('%Y-%m-%d') if hasattr(date, 'strftime') else str(date)
            sent, conf, count = self.get_historical_sentiment(symbol, date_str)
            sentiments.append(sent)
            confidences.append(conf)
            article_counts.append(count)
        
        enhanced['news_sentiment'] = sentiments
        enhanced['news_confidence'] = confidences
        enhanced['news_article_count'] = article_counts
        
        # Compute combined signal
        ml_weight = 0.7
        news_weight = 0.3
        
        enhanced['combined_signal'] = (
            ml_weight * enhanced['ml_signal'] +
            news_weight * enhanced['news_sentiment']
        )
        
        return enhanced


# Test function
async def test_integrator():
    """Test the news signal integrator."""
    print("Testing News Signal Integrator...")
    print("=" * 60)
    
    # Create pipeline
    pipeline = NewsPipeline()
    
    # Create integrator
    integrator = NewsSignalIntegrator(
        ml_model=None,  # No ML model for this test
        news_pipeline=pipeline
    )
    
    try:
        # Test with just news signal
        print("\n1. Getting combined signal for AAPL (news only)...")
        signal = await integrator.get_signal(
            'AAPL',
            ml_prediction=(0.3, 0.65)  # Simulate ML prediction
        )
        
        print(f"   ML Signal: {signal.ml_signal:+.2f} ({signal.ml_direction})")
        print(f"   News Signal: {signal.news_signal:+.2f} ({signal.news_direction})")
        print(f"   Combined: {signal.combined_signal:+.2f}")
        print(f"   Confidence: {signal.combined_confidence:.1%}")
        print(f"   Direction: {signal.final_direction}")
        print(f"   Agreement: {signal.signal_agreement}")
        print(f"   Actionable: {signal.actionable}")
        print(f"   Reason: {signal.reason}")
        
        # Test multi-symbol
        print("\n2. Getting signals for multiple symbols...")
        multi_signals = await integrator.get_multi_symbol_signals(
            ['AAPL', 'MSFT', 'NVDA'],
            ml_predictions={
                'AAPL': (0.3, 0.65),
                'MSFT': (-0.2, 0.55),
                'NVDA': (0.5, 0.75)
            }
        )
        
        print("\n   Symbol | ML | News | Combined | Direction")
        print("   " + "-" * 50)
        for symbol, sig in multi_signals.items():
            print(f"   {symbol:6} | {sig.ml_signal:+.2f} | {sig.news_signal:+.2f} | {sig.combined_signal:+.2f} | {sig.final_direction}")
        
        # Test allocation
        print("\n3. Getting portfolio allocation...")
        allocation = integrator.get_portfolio_allocation(multi_signals)
        
        print("\n   Allocation:")
        for symbol, pct in allocation.items():
            direction = "LONG" if pct > 0 else "SHORT"
            print(f"   {symbol}: {abs(pct):.1%} {direction}")
        
        print("\n" + "=" * 60)
        print("✅ News Signal Integrator working!")
        
    finally:
        await pipeline.stop()


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_integrator())
